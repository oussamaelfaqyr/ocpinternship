"""
OCP Youssoufia 60 kV HTB — Professional SCADA Digital Twin Web Server
======================================================================
Flask Backend Server providing REST API endpoints for telemetry, AI diagnosis,
fault injection, alarm management, and serving the HTML5/CSS3/JS Web UI.
Automatically launches default browser upon start.

AI Model : GRU Optuna E3-B (gru_optuna_e3b_best.pt)  [PRIMARY — Val F1: 0.7715]
           Fallback: GRU Champion E3-B (gru_e3b.pt)   [Val F1: 0.7298]
           input  → (1, window=20, features=35)
           output → 8 fault classes
"""

import os
import sys
import time
import datetime
import threading
import webbrowser
from flask import Flask, jsonify, request, send_from_directory

# Ensure submodules are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for d in ["network_model", "fault_simulator", "ai"]:
    p = os.path.join(BASE_DIR, d)
    if p not in sys.path:
        sys.path.insert(0, p)

import build_network as bn
import fault_simulator as fs
from fault_classifier import FaultClassifier

app = Flask(__name__, static_folder="static", static_url_path="")

# ── Global State ─────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.net = bn.build_network()
        self.classifier = FaultClassifier()
        self.sim_step = 0
        self.running = True
        self.fault = "normal"
        self.fault_tgt = "Mine Mzinda DIS TR"
        self.fault_exp = 0
        self.fault_duration = 30
        self.history = []
        self.alarms = []
        self.events = []
        self.telemetry = {}
        self.ai_result = {}
        self.shap_result = {}
        self.sel_sub = "Mine Mzinda DIS TR"
        self.lock = threading.RLock()

state = State()

FAULT_LABELS = {
    "normal":           "Normal Operation",
    "lg_fault":         "Line-to-Ground Fault",
    "ll_fault":         "Line-to-Line Fault",
    "over_frequency":   "Over-Frequency (81O)",
    "overload":         "Thermal Overload",
    "transformer_trip": "Transformer Trip",
    "under_frequency":  "Under-Frequency (81U)",
    "voltage_sag":      "Grid Voltage Sag",
    # kept for simulator labels (not classified by LSTM but may appear)
    "breaker_trip":     "Breaker Trip",
    "source_outage":    "Source Outage",
}

AI_SEVERITY = {
    "lg_fault":         "CRITICAL",
    "ll_fault":         "CRITICAL",
    "transformer_trip": "CRITICAL",
    "source_outage":    "CRITICAL",
    "breaker_trip":     "CRITICAL",
    "voltage_sag":      "WARNING",
    "overload":         "WARNING",
    "under_frequency":  "WARNING",
    "over_frequency":   "WARNING",
    "normal":           "NONE",
}

# ── Core Logic ───────────────────────────────────────────────────────────────
def push_event(ts, text):
    state.events.insert(0, {"time": ts, "event": text})
    if len(state.events) > 250:
        state.events = state.events[:150]

def check_alarms(sub, row, ts, ai_res=None):
    lp = row.get("loading_pct", 0)
    lbl = row.get("label", "normal")
    v = row.get("V_hv_kV", 60.0)

    def _add(pri, msg):
        if any(a["sub"] == sub and a["msg"] == msg for a in state.alarms[:10]):
            return
        state.alarms.insert(0, {"time": ts, "priority": pri, "sub": sub, "msg": msg})
        push_event(ts, f"[{sub}] {msg}")
        if len(state.alarms) > 300:
            state.alarms = state.alarms[:200]

    # Physical threshold alarms
    if lbl != "normal":
        _add("CRITICAL", FAULT_LABELS.get(lbl, lbl))
    elif lp > 100:
        _add("CRITICAL", f"Overload {lp:.0f}%")
    elif lp > 80:
        _add("WARNING", f"High load {lp:.0f}%")
    if 0 < v < 54:
        _add("CRITICAL", f"Undervoltage {v:.1f} kV")

    # AI Model Alert Trigger
    if ai_res:
        ai_label = ai_res.get("pred_label", "normal")
        conf = ai_res.get("confidence", 0.0)
        if ai_label != "normal" and conf >= 40.0:
            sev = AI_SEVERITY.get(ai_label, "WARNING")
            friendly_name = FAULT_LABELS.get(ai_label, ai_label)
            _add(sev, f"AI ALERT: {friendly_name} Detected ({conf:.1f}% conf)")

def step_simulation():
    with state.lock:
        t = state.sim_step
        flt = state.fault
        tgt = state.fault_tgt

        if flt != "normal" and t >= state.fault_exp:
            flt = state.fault = "normal"
            # Reset the AI rolling window so it doesn't keep predicting the old fault
            state.classifier.reset_buffers()

        # Build per-substation fault map with explicit params for frequency faults
        if flt in ["under_frequency", "over_frequency", "voltage_sag", "source_outage"] \
                and fs._TRAFO_SUBS:
            # Grid-wide faults — apply to all substations with explicit duration
            params = {
                "duration_steps": state.fault_duration,
                "immediate_start": True,  # jump to anomalous freq immediately
            }
            sub_fault_map = {sub: (flt, params) for sub in fs._TRAFO_SUBS}
        elif flt != "normal" and fs._TRAFO_SUBS:
            # Sub-specific fault
            sub_fault_map = {
                sub: (flt, {}) if sub == tgt else ("normal", {})
                for sub in fs._TRAFO_SUBS
            }
        else:
            sub_fault_map = None  # use simple API (normal or first step)

        if sub_fault_map is not None:
            tel = fs.simulate_step_per_sub(state.net, current_step=t, sub_fault_map=sub_fault_map)
        else:
            tel = fs.simulate_step(state.net, current_step=t, active_fault=flt, target_sub=tgt)

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        state.telemetry = tel

        # Execute LSTM inference
        ai_out = state.classifier.predict_telemetry(tel)
        state.ai_result = ai_out

        # Compute real-time gradient-based feature attribution (SHAP) for selected substation
        try:
            state.shap_result = state.classifier.compute_shap(state.sel_sub)
        except Exception:
            state.shap_result = {}

        for sub, row in tel.items():
            state.history.append(dict(t=t, sub=sub, **row))
            ai_res = ai_out.get(sub)
            check_alarms(sub, row, ts, ai_res)

        if len(state.history) > 12000:
            state.history = state.history[-8000:]

        state.sim_step += 1

def sim_loop():
    while True:
        if state.running:
            try:
                step_simulation()
            except Exception as e:
                print(f"[SimLoop Error] {e}")
        time.sleep(1.0)

# Start background thread
sim_thread = threading.Thread(target=sim_loop, daemon=True)
sim_thread.start()

# ── REST API Endpoints ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def get_status():
    with state.lock:
        sub_names = [n.replace(" trafo", "") for n in state.net.trafo["name"]]
        return jsonify({
            "running": state.running,
            "sim_step": state.sim_step,
            "active_fault": state.fault,
            "fault_target": state.fault_tgt,
            "fault_expiry": state.fault_exp,
            "substations": sub_names,
            "selected_substation": state.sel_sub,
            "model_info": state.classifier.model_info()
        })

@app.route("/api/ai/model", methods=["GET", "POST"])
def manage_ai_model():
    with state.lock:
        if request.method == "POST":
            data = request.json or {}
            key = data.get("model", "lstm")
            success = state.classifier.switch_model(key)
            if success:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                m_info = state.classifier.model_info()
                push_event(ts, f"[AI ENGINE] Switched active model to {m_info['display_name']} ({m_info['arch']})")
            return jsonify({"success": success, "model_info": state.classifier.model_info()})
        else:
            return jsonify(state.classifier.model_info())

@app.route("/api/telemetry")
def get_telemetry():
    with state.lock:
        return jsonify(state.telemetry)

@app.route("/api/ai_diagnosis")
def get_ai_diagnosis():
    with state.lock:
        return jsonify(state.ai_result)

@app.route("/api/alarms")
def get_alarms():
    with state.lock:
        return jsonify(state.alarms[:50])

@app.route("/api/events")
def get_events():
    with state.lock:
        return jsonify(state.events[:100])

@app.route("/api/history")
def get_history():
    sub = request.args.get("substation", state.sel_sub)
    with state.lock:
        sub_hist = [h for h in state.history if h.get("sub") == sub][-60:]
        return jsonify(sub_hist)

@app.route("/api/sim/control", methods=["POST"])
def sim_control():
    data = request.json or {}
    action = data.get("action")
    with state.lock:
        if action == "play":
            state.running = True
        elif action == "pause":
            state.running = False
        elif action == "toggle":
            state.running = not state.running
        elif action == "step":
            step_simulation()
        elif action == "set_substation":
            state.sel_sub = data.get("substation", state.sel_sub)
    return jsonify({"success": True, "running": state.running})

@app.route("/api/fault/inject", methods=["POST"])
def inject_fault():
    data = request.json or {}
    fault_type = data.get("type", "lg_fault")
    target_sub = data.get("target", state.sel_sub)
    duration = int(data.get("duration", 15))

    with state.lock:
        state.fault = fault_type
        state.fault_tgt = target_sub
        state.fault_exp = state.sim_step + duration
        state.running = True
        # Store duration for use by step_simulation params
        state.fault_duration = duration

    return jsonify({"success": True, "active_fault": state.fault, "target": state.fault_tgt})

@app.route("/api/fault/clear", methods=["POST"])
def clear_fault():
    with state.lock:
        state.fault = "normal"
        # Reset AI rolling windows so prediction returns to normal immediately
        state.classifier.reset_buffers()
    return jsonify({"success": True, "active_fault": "normal"})

@app.route("/api/shap")
def get_shap():
    sub = request.args.get("substation", state.sel_sub)
    with state.lock:
        try:
            if sub == state.sel_sub and state.shap_result:
                return jsonify(state.shap_result)
            return jsonify(state.classifier.compute_shap(sub))
        except Exception:
            return jsonify(state.shap_result or {"substation": sub, "values": {}})


if __name__ == "__main__":
    PORT = 8080
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n=======================================================")
    print(f" OCP Youssoufia SCADA Digital Twin Web Server")
    print(f" Running at: {url}")
    print(f"=======================================================\n")

    # Automatically launch default browser after 1.2 seconds
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=PORT, debug=False)
