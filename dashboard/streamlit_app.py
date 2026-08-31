"""
OCP Youssoufia 60 kV HTB — Industrial SCADA Digital Twin Dashboard
====================================================================
Clean, un-cluttered industrial layout with tabbed navigation for high clarity.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ["network_model", "fault_simulator", "ai"]:
    p = os.path.join(BASE_DIR, d)
    if p not in sys.path:
        sys.path.insert(0, p)

import fault_simulator as _fs
import build_network   as _bn
from dataset_generator  import DatasetGenerator
from fault_classifier   import FaultClassifier
from components.common  import apply_global_css, WIDTH_STRETCH, FAULT_LABELS, AI_SEVERITY
from components.sld_viewer import create_sld_figure
from components.charts  import (create_voltage_chart, create_current_loading_chart,
                                 create_power_chart, create_frequency_chart)
from components.ai_panel import render_ai_diagnosis
from components.alarms   import render_alarms
from components.network_table import render_network_table
from components.soe_log import render_soe_log, render_fault_lifecycle_badge
from components.ai_confidence import render_ai_confidence_chart, render_feature_inspector
from components.dataset_stats import render_dataset_stats

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="OCP Youssoufia SCADA", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
apply_global_css()

# ── Session state ────────────────────────────────────────────────────────────
def _init():
    if "net" not in st.session_state:
        net = _bn.build_network()
        st.session_state.net    = net
        st.session_state.base_p = net.load["p_mw"].copy()
        st.session_state.base_q = net.load["q_mvar"].copy()
    for k, v in {"sim_step":0,"running":False,"fault":"normal",
                 "fault_tgt":"Mine Mzinda DIS TR","fault_exp":0,
                 "history":[],"alarms":[],"events":[],
                 "telemetry":{},"sel_sub":"Mine Mzinda DIS TR","ai_result":{},
                 "cpc_generating":False}.items():
        if k not in st.session_state: st.session_state[k] = v
    if "classifier" not in st.session_state:
        st.session_state.classifier = FaultClassifier()
    if "dgen" not in st.session_state:
        st.session_state.dgen = DatasetGenerator(
            output_path=os.path.join(BASE_DIR, "youssoufia_pandapower_dataset.csv"))
_init()

# ── Core helpers ─────────────────────────────────────────────────────────────
def _push_event(ts, text):
    st.session_state.events.insert(0, {"time": ts, "event": text})
    if len(st.session_state.events) > 200:
        st.session_state.events = st.session_state.events[:150]

def _check_alarms(sub, row, ts, ai_res=None):
    lp, lbl, v = row.get("loading_pct",0), row.get("label","normal"), row.get("V_hv_kV",60.)
    def _add(pri, msg):
        if any(a["sub"]==sub and a["msg"]==msg for a in st.session_state.alarms[:10]): return
        st.session_state.alarms.insert(0, {"time":ts,"priority":pri,"sub":sub,"msg":msg})
        _push_event(ts, f"[{sub}] {msg}")
        if len(st.session_state.alarms) > 300:
            st.session_state.alarms = st.session_state.alarms[:200]
    if lbl != "normal":  _add("CRITICAL", FAULT_LABELS.get(lbl, lbl))
    elif lp > 100:       _add("CRITICAL", f"Overload {lp:.0f}%")
    elif lp > 80:        _add("WARNING",  f"High load {lp:.0f}%")
    if 0 < v < 54:       _add("CRITICAL", f"Undervoltage {v:.1f} kV")

    # AI Model Alert Trigger
    if ai_res:
        ai_label = ai_res.get("pred_label", "normal")
        conf = ai_res.get("confidence", 0.0)
        if ai_label != "normal" and conf >= 40.0:
            sev = AI_SEVERITY.get(ai_label, "WARNING")
            friendly_name = FAULT_LABELS.get(ai_label, ai_label)
            _add(sev, f"AI ALERT: {friendly_name} Detected ({conf:.1f}% conf)")

def _run_ai(tel):
    clf = st.session_state.classifier
    return clf.predict_telemetry(tel)

def _step_sim():
    t, flt, tgt = st.session_state.sim_step, st.session_state.fault, st.session_state.fault_tgt
    if flt != "normal" and t >= st.session_state.fault_exp:
        flt = st.session_state.fault = "normal"
    tel = _fs.simulate_step(st.session_state.net, current_step=t, active_fault=flt, target_sub=tgt)
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.telemetry = tel
    st.session_state.ai_result = _run_ai(tel)

    for sub, row in tel.items():
        st.session_state.history.append(dict(t=t, sub=sub, **row))
        ai_res = st.session_state.ai_result.get(sub)
        _check_alarms(sub, row, ts, ai_res)

    if len(st.session_state.history) > 8000:
        st.session_state.history = st.session_state.history[-6000:]
    st.session_state.sim_step += 1

def _kpi(label, value, unit="", color="#38BDF8"):
    st.markdown(f"""<div style='background:#111827;border:1px solid #1E293B;border-radius:6px;
        padding:10px 14px;border-left:3px solid {color};'>
        <div style='font-size:10px;color:#94A3B8;text-transform:uppercase;font-weight:700;
            letter-spacing:.5px'>{label}</div>
        <div style='font-size:18px;font-weight:800;color:{color};
            font-family:"JetBrains Mono",monospace;margin-top:2px'>
            {value}<span style='font-size:11px;color:#64748B;font-weight:400'> {unit}</span>
        </div></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  LIVE FRAGMENT  (module-level, ticks every 1 s when running)
# ════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every="1s")
def _live():
    if st.session_state.running:
        _step_sim()

    sel = st.session_state.get("sel_sub", "Mine Mzinda DIS TR")
    tel = st.session_state.telemetry.get(sel, {})

    # ── Fault Lifecycle Badge (Only shows if fault active) ───────────────────
    render_fault_lifecycle_badge()
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── 7-column KPI bar ─────────────────────────────────────────────────────
    load  = tel.get("loading_pct", 0)
    c_ld  = "#EF4444" if load > 100 else ("#F59E0B" if load > 80 else "#10B981")
    kc = st.columns(7)
    with kc[0]: _kpi("HV Bus",   f"{tel.get('V_hv_kV',0):.2f}",  "kV",  "#38BDF8")
    with kc[1]: _kpi("LV Bus",   f"{tel.get('V_lv_kV',0):.3f}",  "kV",  "#818CF8")
    with kc[2]: _kpi("Current",  f"{tel.get('I_hv_A',0):.1f}",   "A",   "#F43F5E")
    with kc[3]: _kpi("P Active", f"{tel.get('P_MW',0):.3f}",     "MW",  "#34D399")
    with kc[4]: _kpi("Q React",  f"{tel.get('Q_Mvar',0):.3f}",   "Mvar","#A855F7")
    with kc[5]: _kpi("Freq",     f"{tel.get('freq_Hz',50):.3f}", "Hz",  "#84CC16")
    with kc[6]: _kpi("Loading",  f"{load:.1f}",                  "%",   c_ld)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── TABBED SCADA LAYOUT — Keeps panels wide, spacious and un-collapsed ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Network SLD & Trends",
        "🌐 All Substations Status",
        "🤖 AI Diagnosis & Physics Features",
        "📋 Alarms & Event Log (SOE)"
    ])

    with tab1:
        st.plotly_chart(
            create_sld_figure(st.session_state.telemetry, st.session_state.fault, sel),
            width="stretch", config={"displayModeBar": False}, key="k_sld")
        
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("""<div style='color:#38BDF8;font-size:14px;font-weight:700;
            border-bottom:1px solid #1E293B;padding-bottom:4px;margin-bottom:12px;
            text-transform:uppercase'>Signal Curves — Live Trends ({})</div>""".format(sel), unsafe_allow_html=True)

        st.plotly_chart(create_voltage_chart(sel),
                        width="stretch", config={"displayModeBar": False}, key="k_v")
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(create_current_loading_chart(sel),
                        width="stretch", config={"displayModeBar": False}, key="k_i")
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(create_power_chart(sel),
                        width="stretch", config={"displayModeBar": False}, key="k_p")
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(create_frequency_chart(sel),
                        width="stretch", config={"displayModeBar": False}, key="k_f")

    with tab2:
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        render_network_table(st.session_state.telemetry)

    with tab3:
        col_ai1, col_ai2 = st.columns([1, 1])
        with col_ai1:
            render_ai_diagnosis(sel, st.session_state.ai_result, st.session_state.telemetry)
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            render_ai_confidence_chart(sel, st.session_state.ai_result)
        with col_ai2:
            render_feature_inspector(sel, st.session_state.telemetry)

    with tab4:
        col_ev1, col_ev2 = st.columns([1, 1])
        with col_ev1:
            render_alarms()
        with col_ev2:
            render_soe_log()


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style='padding:4px 0 12px'>
        <div style='color:#38BDF8;font-size:16px;font-weight:800;
            letter-spacing:.5px;text-transform:uppercase'>OCP SCADA</div>
        <div style='color:#475569;font-size:11px;margin-top:2px'>
            Youssoufia 60 kV Digital Twin</div></div>""", unsafe_allow_html=True)
    st.divider()
    nav = st.radio("Nav", ["Real-Time Digital Twin", "Dataset Generator"],
                   label_visibility="collapsed")
    st.divider()
    st.markdown("**SIMULATION CONTROLS**")
    if st.button("▶ Run" if not st.session_state.running else "⏸ Pause",
                 **WIDTH_STRETCH):
        st.session_state.running = not st.session_state.running
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1: REAL-TIME DIGITAL TWIN
# ════════════════════════════════════════════════════════════════════════════
if nav == "Real-Time Digital Twin":

    sub_names = [n.replace(" trafo", "") for n in st.session_state.net.trafo["name"]]

    def _on_sub_change():
        st.session_state.sel_sub = st.session_state._sub_picker

    col_pick, _ = st.columns([1, 3])
    with col_pick:
        st.selectbox(
            "Selected Substation", sub_names,
            index=sub_names.index(st.session_state.sel_sub)
                  if st.session_state.sel_sub in sub_names else 0,
            key="_sub_picker",
            on_change=_on_sub_change,
        )

    # Fault injection — static
    FAULT_MAP = {
        "Line-to-Ground (LG)": "lg_fault",   "Line-to-Line (LL)": "ll_fault",
        "Thermal Overload":    "overload",    "Grid Voltage Sag":  "voltage_sag",
        "Under-Frequency":     "under_frequency",
        "Source Outage":       "source_outage",
        "Breaker Trip":        "breaker_trip",
    }
    with st.expander("⚡ Fault Injection Control", expanded=False):
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            sf = st.selectbox("Type", list(FAULT_MAP.keys()),
                              label_visibility="collapsed", key="fi_type")
        with fc2:
            fi_dur = st.number_input("s", min_value=1, max_value=300, value=15,
                                     label_visibility="collapsed")
        with fc3:
            if st.button("⚡ Inject", **WIDTH_STRETCH):
                st.session_state.fault     = FAULT_MAP[sf]
                st.session_state.fault_tgt = st.session_state.sel_sub
                st.session_state.fault_exp = st.session_state.sim_step + fi_dur
                st.session_state.running   = True
                st.rerun()
        if st.session_state.fault != "normal":
            st.warning(f"Active: **{FAULT_LABELS.get(st.session_state.fault)}** "
                       f"on {st.session_state.fault_tgt}")
            if st.button("✖ Clear Fault", **WIDTH_STRETCH):
                st.session_state.fault = "normal"
                st.rerun()

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Live panel (fragment) ────────────────────────────────────────────────
    _live()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2: DATASET GENERATOR
# ════════════════════════════════════════════════════════════════════════════
elif nav == "Dataset Generator":
    st.markdown("### 💾 Dataset Generator")
    st.markdown("Generate synthetic SCADA data for ML training from the Digital Twin engine.")

    gen_tab1, gen_tab2 = st.tabs(["⏱ Duration-Based", "🎯 Samples per Class"])

    # ── TAB 1: existing duration-based approach ───────────────────────────────
    with gen_tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### Generation Parameters")
                dur = st.number_input("Duration (seconds)",
                                      min_value=100, max_value=360000, value=3600, step=100,
                                      key="dur_gen")
                
                # New parameter to deal with class imbalance
                density = st.slider("Fault Density (%)",
                                    min_value=0.1, max_value=30.0, value=2.0, step=0.1,
                                    help="Increase this to force more faults to happen during the simulation period (helps balance classes).")

                if st.button("▶ Generate", type="primary", **WIDTH_STRETCH, key="btn_dur_gen"):
                    pbar = st.progress(0.0)
                    txt  = st.empty()
                    st.session_state.dgen.generate_dataset_sync(
                        total_seconds=dur,
                        fault_density=density / 100.0,
                        progress_callback=lambda pct,s,n: (pbar.progress(pct),
                                                            txt.text(f"Step {s}/{n}...")))
                    st.success(f"Done — {dur}s dataset generated!")
        with col2:
            with st.container(border=True):
                st.markdown("#### Export")
                csv_path = os.path.join(BASE_DIR, "youssoufia_pandapower_dataset.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    st.metric("Total Rows", f"{len(df):,}")
                    st.dataframe(df.head(8), width="stretch")
                    with open(csv_path) as f:
                        st.download_button("⬇ Download CSV", f,
                                           "ocp_youssoufia_dataset.csv",
                                           "text/csv", **WIDTH_STRETCH, key="dl_dur")
                else:
                    st.info("No dataset yet — generate one first.")

    # ── TAB 2: samples-per-class mode ────────────────────────────────────────
    with gen_tab2:
        st.markdown(
            """
            <div style='color:#94A3B8;font-size:13px;margin-bottom:16px'>
            Set the exact number of samples you want for <b>each fault class</b>.
            The generator will simulate each fault type independently until the
            quota is met, then shuffle and save a balanced CSV.
            </div>
            """,
            unsafe_allow_html=True
        )

        _ALL_CLASSES = [
            ("normal",           "Normal Operation",       "#10B981"),
            ("lg_fault",         "Line-to-Ground Fault",   "#EF4444"),
            ("ll_fault",         "Line-to-Line Fault",     "#A855F7"),
            ("overload",         "Thermal Overload",       "#F59E0B"),
            ("voltage_sag",      "Grid Voltage Sag",       "#38BDF8"),
            ("under_frequency",  "Under-Frequency (81U)",  "#EAB308"),
            ("over_frequency",   "Over-Frequency (81O)",   "#E879F9"),
            ("source_outage",    "Source Outage",          "#6B7280"),
            ("breaker_trip",     "Breaker Trip",           "#F97316"),
            ("transformer_trip", "Transformer Trip",       "#EC4899"),
        ]

        # Initialize session-state defaults ONCE (avoids widget/session-state conflict)
        _CPC_DEFAULTS = {k: (200 if k == "normal" else 50) for k, _, _ in _ALL_CLASSES}
        for _k, _default in _CPC_DEFAULTS.items():
            if f"cpc_{_k}" not in st.session_state:
                st.session_state[f"cpc_{_k}"] = _default

        cpc_col1, cpc_col2 = st.columns([2, 1])

        with cpc_col1:
            with st.container(border=True):
                st.markdown("#### Samples per Class")

                # Quick-fill row
                qf_c1, qf_c2, qf_c3 = st.columns(3)
                with qf_c1:
                    quick_n = st.number_input("Fill all classes with:", min_value=0,
                                              max_value=50000, value=200, step=50,
                                              key="cpc_quick_n")
                with qf_c2:
                    st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                    if st.button("⚡ Apply to All", key="cpc_apply_all"):
                        for key, _, _ in _ALL_CLASSES:
                            st.session_state[f"cpc_{key}"] = quick_n
                        st.rerun()
                with qf_c3:
                    st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                    if st.button("↺ Reset All", key="cpc_reset_all"):
                        for key, _, _ in _ALL_CLASSES:
                            st.session_state[f"cpc_{key}"] = 0
                        st.rerun()

                st.divider()

                # Per-class number inputs (two-column grid)
                left_classes  = _ALL_CLASSES[:5]
                right_classes = _ALL_CLASSES[5:]
                row_cols = st.columns(2)

                class_samples_input = {}
                for col_idx, class_group in enumerate([left_classes, right_classes]):
                    with row_cols[col_idx]:
                        for fault_key, fault_label, fault_color in class_group:
                            n_val = st.number_input(
                                f"🔹 {fault_label}",
                                min_value=0, max_value=50000,
                                step=10,
                                key=f"cpc_{fault_key}",
                                help=f"fault type: `{fault_key}`"
                            )
                            class_samples_input[fault_key] = n_val

                st.divider()

                # Keep total_requested in outer scope for cpc_col2 percentage table
                total_requested = sum(class_samples_input.values())

                # ── Live generation controls ──────────
                def _cpc_controls():
                    import threading
                    dgen = st.session_state.dgen

                    # Always read current widget values from session state
                    _samples = {k: st.session_state.get(f"cpc_{k}", 0)
                                for k, _, _ in _ALL_CLASSES}
                    _active  = {k: v for k, v in _samples.items() if v > 0}
                    _total   = sum(_samples.values())
                    _path    = os.path.join(BASE_DIR, "youssoufia_pandapower_dataset.csv")

                    st.markdown(
                        f"<div style='color:#94A3B8;font-size:12px'>Total samples requested: "
                        f"<b style='color:#38BDF8'>{_total:,}</b></div>",
                        unsafe_allow_html=True)
                    st.markdown("<div style='margin-top:10px'></div>",
                                unsafe_allow_html=True)

                    if not dgen.is_generating:
                        # ── Generate button ────────────────────────────────────
                        if st.button("▶ Generate Custom Dataset", type="primary",
                                     **WIDTH_STRETCH, key="btn_cpc_gen",
                                     disabled=(_total == 0)):
                            _ac = dict(_active)
                            _op = _path
                            dgen._stop_event.clear()
                            dgen.is_generating = True
                            dgen.progress      = 0.0
                            dgen.status_message = "Starting…"

                            def _run_gen(_ac=_ac, _op=_op):
                                try:
                                    dgen.generate_dataset_by_class_samples(
                                        class_samples=_ac, output_file=_op)
                                except Exception as exc:
                                    import traceback
                                    dgen.is_generating   = False
                                    dgen.status_message  = f"❌ Error: {exc}"
                                    traceback.print_exc()

                            threading.Thread(target=_run_gen, daemon=True).start()
                            st.session_state.cpc_generating = True
                            st.rerun()

                        # Show success / stop message after previous run
                        if "cpc_last_msg" in st.session_state and st.session_state.cpc_last_msg:
                            if "Stopped" in st.session_state.cpc_last_msg:
                                st.warning(st.session_state.cpc_last_msg)
                            else:
                                st.success(st.session_state.cpc_last_msg)
                    else:
                        # ── Progress bar + Stop & Save button ─────────────────
                        pct = min(1.0, dgen.progress)
                        done = int(pct * max(1, _total))
                        st.progress(pct, text=f"{done:,} / {_total:,} samples collected…")
                        st.caption(dgen.status_message)
                        st.markdown("<div style='margin-top:8px'></div>",
                                    unsafe_allow_html=True)
                        if st.button("⏹ Stop & Save Now", key="btn_cpc_stop",
                                     type="secondary", **WIDTH_STRETCH):
                            dgen._stop_event.set()
                            st.session_state.cpc_generating = False
                            st.rerun()
                        st.info("⏳ Generation running — data will be auto-saved "
                                "when stopped or completed.")

                    # Sync generating flag & capture last message
                    if st.session_state.cpc_generating and not dgen.is_generating:
                        st.session_state.cpc_generating = False
                        st.session_state.cpc_last_msg = dgen.status_message
                        st.rerun()

                _cpc_controls()

        with cpc_col2:
            with st.container(border=True):
                st.markdown("#### Class Distribution Preview")
                active = {k: v for k, v in class_samples_input.items() if v > 0}
                if active:
                    import plotly.graph_objects as go
                    colors_map = {k: c for k, _, c in _ALL_CLASSES}
                    labels_map = {k: l for k, l, _ in _ALL_CLASSES}
                    fig_pie = go.Figure(go.Pie(
                        labels=[labels_map[k] for k in active],
                        values=list(active.values()),
                        marker_colors=[colors_map[k] for k in active],
                        hole=0.45,
                        textinfo="percent",
                        hovertemplate="<b>%{label}</b><br>%{value:,} samples<extra></extra>",
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#CBD5E1",
                        margin=dict(t=10, b=10, l=10, r=10),
                        showlegend=False,
                        height=260,
                    )
                    st.plotly_chart(fig_pie, width="stretch",
                                    config={"displayModeBar": False}, key="k_pie_cpc")

                    # Class table
                    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                    for k, v in active.items():
                        label_text = labels_map[k]
                        color      = colors_map[k]
                        pct        = 100 * v / max(1, total_requested)
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:3px 6px;border-radius:3px;margin-bottom:3px;"
                            f"border-left:3px solid {color}'>"
                            f"<span style='font-size:12px;color:#CBD5E1'>{label_text}</span>"
                            f"<span style='font-size:12px;font-weight:700;color:{color}'>"
                            f"{v:,} ({pct:.1f}%)</span></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Set at least one class to a non-zero value.")

            with st.container(border=True):
                st.markdown("#### Export")
                csv_path_cpc = os.path.join(BASE_DIR, "youssoufia_pandapower_dataset.csv")
                if os.path.exists(csv_path_cpc):
                    df_cpc = pd.read_csv(csv_path_cpc)
                    st.metric("Total Rows", f"{len(df_cpc):,}")
                    with open(csv_path_cpc) as f:
                        st.download_button("⬇ Download CSV", f,
                                           "ocp_youssoufia_custom_dataset.csv",
                                           "text/csv", **WIDTH_STRETCH, key="dl_cpc")
                else:
                    st.info("No dataset yet.")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    csv_path = os.path.join(BASE_DIR, "youssoufia_pandapower_dataset.csv")
    render_dataset_stats(csv_path)

# ── Auto-refresh loop for background generation ──────────────────────────────
import time
if st.session_state.get("cpc_generating", False):
    time.sleep(1)
    st.rerun()
