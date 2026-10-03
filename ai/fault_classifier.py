"""
AI Fault Classifier -- OCP Youssoufia 60 kV Digital Twin
=========================================================
Multi-architecture sequence classifier for 8 fault categories:
  0  normal            5  transformer_trip
  1  lg_fault          6  under_frequency
  2  ll_fault          7  voltage_sag
  3  over_frequency
  4  overload

Supported architectures:
  - GRU Optuna E3-B   gru_optuna_e3b_best.pt (Active/Champion)
  - GRU E3-B          gru_e3b.pt             (Champion Macro-F1 0.7494)
  - GRU Baseline      gru_baseline.pt        (Baseline E2)
  - LSTM Legacy       lstm_optuna_best.pt

Input shape : (1, window_size=20, n_features=35)
Features    : 24 numeric (scaled) + 11 OHE substation columns
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Model architectures ----------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        last = self.dropout(output[:, -1, :])
        return self.fc(last)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes, conv1_out, conv2_out, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv1_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv2_out * 5, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc2(self.dropout(self.relu3(self.fc1(x))))
        return x


# ---------- Model registry ---------------------------------------------

AVAILABLE_MODELS = {
    "gru_optuna": {"file": "gru_optuna_e3b_best.pt", "display_name": "GRU Optuna (E3-B)",
                   "arch": "GRU (2 Layers, 256 Hidden)", "val_f1": 0.7715},
    "gru_e3b":    {"file": "gru_e3b.pt",              "display_name": "GRU Champion (E3-B)",
                   "arch": "GRU (2 Layers, 128 Hidden)", "val_f1": 0.7298},
    "gru":        {"file": "gru_e3b.pt",              "display_name": "GRU E3-B Active",
                   "arch": "GRU (2 Layers, 128 Hidden)", "val_f1": 0.7298},
    "gru_base":   {"file": "gru_baseline.pt",         "display_name": "GRU Baseline (E2)",
                   "arch": "GRU (2 Layers, 64 Hidden)",  "val_f1": 0.688},
    "lstm":       {"file": "lstm_optuna_best.pt",     "display_name": "LSTM Legacy",
                   "arch": "LSTM (2 Layers, 128 Hidden)", "val_f1": 0.699},
}


# ---------- FaultClassifier --------------------------------------------

class FaultClassifier:
    """
    Wraps a trained sequence model (GRU / LSTM / CNN) for real-time SCADA inference.
    Handles dynamic feature computation:
      - Relative voltages (V_HV_pu, V_LV_pu, V_ratio, delta_V_pu)
      - Substation baselines (P_ratio, I_ratio, S_ratio, I2t_pu)
      - Power & Current dynamics (delta_I, abs_delta_I, delta_P, delta_S)
    """

    WINDOW_SIZE   = 20
    INPUT_SIZE    = 35
    NUM_CLASSES   = 8
    DEFAULT_MODEL = "gru_optuna"

    SUBSTATIONS = [
        "Laverie/Sechage SN1", "Laverie/Sechage SN2", "Mine Mzinda DIS TR",
        "PSF Mine Bouchane SN6", "Recette 3 SN3", "Recette 9 SN9",
        "SSP Local ST1", "SSP Local ST2", "U Calcination SN1",
        "U Calcination SN2", "U Calcination SN3",
    ]

    SUBSTATION_BASELINES = {
        "Laverie/Sechage SN1":   {"P": 5.739, "I": 66.10, "S": 6.403},
        "Laverie/Sechage SN2":   {"P": 5.575, "I": 63.60, "S": 6.215},
        "Mine Mzinda DIS TR":    {"P": 4.390, "I": 48.60, "S": 4.992},
        "PSF Mine Bouchane SN6": {"P": 3.738, "I": 41.50, "S": 4.249},
        "Recette 3 SN3":         {"P": 2.766, "I": 30.60, "S": 3.143},
        "Recette 9 SN9":         {"P": 0.887, "I":  9.70, "S": 1.007},
        "SSP Local ST1":         {"P": 1.782, "I": 19.60, "S": 2.019},
        "SSP Local ST2":         {"P": 1.792, "I": 19.70, "S": 2.029},
        "U Calcination SN1":     {"P": 2.684, "I": 29.50, "S": 3.037},
        "U Calcination SN2":     {"P": 2.686, "I": 29.70, "S": 3.047},
        "U Calcination SN3":     {"P": 2.673, "I": 29.40, "S": 3.029},
    }

    def __init__(self, model_key=None, scaler_path=None, vocab_path=None,
                 sub_encoder_path=None):
        self._base       = os.path.dirname(os.path.abspath(__file__))
        self._models_dir = os.path.join(self._base, "models", "saved_models")
        processed        = os.path.join(self._base, "data processing", "data", "processed")
        scaler_path      = scaler_path      or os.path.join(processed, "scaler.joblib")
        vocab_path       = vocab_path       or os.path.join(processed, "label_vocab.json")
        sub_encoder_path = sub_encoder_path or os.path.join(processed, "substation_encoder.joblib")

        self.device       = torch.device("cpu")
        self.scaler       = None
        self.sub_encoder  = None
        self.idx_to_label = {}
        self.classes_     = []
        self.model        = None
        self._active_key  = None
        self._active_info = {}
        self._buffers     = {}
        self._prev        = {}

        self._load_shared_artifacts(scaler_path, vocab_path, sub_encoder_path)
        self.switch_model(model_key or self.DEFAULT_MODEL)

    # ---- model switching --------------------------------------------

    def switch_model(self, key):
        key = key.lower()
        if key not in AVAILABLE_MODELS:
            print("[FaultClassifier] Unknown model key %r. Choose: %s" % (key, list(AVAILABLE_MODELS)))
            return False
        info  = AVAILABLE_MODELS[key]
        path  = os.path.join(self._models_dir, info["file"])
        
        # Fallback to gru_e3b.pt if Optuna model is not done yet
        if not os.path.exists(path) and key == "gru_optuna":
            fallback_path = os.path.join(self._models_dir, "gru_e3b.pt")
            if os.path.exists(fallback_path):
                print(f"[FaultClassifier] {info['file']} not ready yet, falling back to gru_e3b.pt")
                path = fallback_path

        model = self._load_model(key, path)
        if model is None:
            return False
        self.model        = model
        self._active_key  = key
        self._active_info = info
        self.reset_buffers()
        print("[FaultClassifier] Active model -> %s  (val F1: %.3f)" % (info["display_name"], info["val_f1"]))
        return True

    def model_info(self):
        return {
            "key":          self._active_key,
            "display_name": self._active_info.get("display_name", "Unknown"),
            "arch":         self._active_info.get("arch", "-"),
            "val_f1":       self._active_info.get("val_f1", 0.0),
            "available":    {k: v["display_name"] for k, v in AVAILABLE_MODELS.items()},
        }

    # ---- public API --------------------------------------------------

    def predict_telemetry(self, grid_telemetry):
        if not grid_telemetry:
            return {}
        return {sub: self._predict_single(sub, row) for sub, row in grid_telemetry.items()}

    def reset_buffers(self, sub_names=None):
        if sub_names is None:
            self._buffers.clear()
            self._prev.clear()
        else:
            for s in sub_names:
                self._buffers.pop(s, None)
                self._prev.pop(s, None)

    # ---- artifact loading -------------------------------------------

    def _load_shared_artifacts(self, scaler_path, vocab_path, sub_encoder_path):
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.idx_to_label = {v: k for k, v in vocab.items()}
            self.classes_     = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
            print("[FaultClassifier] Label vocab loaded: %s" % vocab)
        else:
            print("[FaultClassifier] WARNING: vocab not found at %s" % vocab_path)
            self.idx_to_label = {i: ("class_%d" % i) for i in range(self.NUM_CLASSES)}
            self.classes_     = list(self.idx_to_label.values())

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("[FaultClassifier] StandardScaler loaded from %s (n_features=%d)" % 
                  (scaler_path, getattr(self.scaler, "n_features_in_", 0)))
        else:
            print("[FaultClassifier] WARNING: scaler not found at %s" % scaler_path)

        if os.path.exists(sub_encoder_path):
            self.sub_encoder = joblib.load(sub_encoder_path)
            print("[FaultClassifier] Substation OneHotEncoder loaded from %s" % sub_encoder_path)

    def _load_model(self, key, path):
        if not os.path.exists(path):
            print("[FaultClassifier] WARNING: checkpoint not found at %s" % path)
            return None
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            input_size = ck.get("input_size", self.INPUT_SIZE)
            self._model_input_size = input_size
            hidden_size = ck.get("hidden_size", 128)
            num_layers = ck.get("num_layers", 2)
            num_classes = ck.get("num_classes", self.NUM_CLASSES)
            dropout = ck.get("dropout", 0.2)

            if "lstm" in key:
                model = LSTMClassifier(
                    input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, num_classes=num_classes, dropout=dropout)
            elif "cnn" in key:
                params = ck.get("params", {})
                model = CNN1DClassifier(
                    input_size=input_size, num_classes=num_classes,
                    conv1_out=params.get("conv1_out", 128),
                    conv2_out=params.get("conv2_out", 64),
                    kernel_size=params.get("kernel_size", 3),
                    dropout=params.get("dropout", 0.33))
            else:  # GRU (default)
                model = GRUClassifier(
                    input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, num_classes=num_classes, dropout=dropout)

            model.load_state_dict(ck["model_state_dict"])
            model.to(self.device).eval()
            val_f1 = ck.get("val_macro_f1", "N/A")
            f1_str = ("%.4f" % val_f1) if isinstance(val_f1, float) else str(val_f1)
            print("[FaultClassifier] %s loaded from %s -- val macro-F1: %s" % 
                  (key.upper(), os.path.basename(path), f1_str))
            return model
        except Exception as e:
            print("[FaultClassifier] ERROR loading %s: %s" % (key.upper(), e))
            return None

    # ---- internal helpers -------------------------------------------

    def _normalize_sub(self, name):
        clean = name.replace(" trafo", "").strip()
        if clean in self.SUBSTATIONS:
            return clean
        for s in self.SUBSTATIONS:
            if s in clean or clean in s:
                return s
        return self.SUBSTATIONS[0]

    def _ohe_substation(self, name):
        norm = self._normalize_sub(name)
        if self.sub_encoder is not None:
            try:
                return self.sub_encoder.transform([[norm]])[0].astype(np.float32)
            except Exception:
                pass
        vec = np.zeros(len(self.SUBSTATIONS), dtype=np.float32)
        if norm in self.SUBSTATIONS:
            vec[self.SUBSTATIONS.index(norm)] = 1.0
        return vec

    def _build_step_feature(self, sub, row):
        norm_sub = self._normalize_sub(sub)
        v_hv   = float(row.get("V_hv_kV",    60.0))
        v_lv   = float(row.get("V_lv_kV",     5.5))
        i_hv   = float(row.get("I_hv_A",     50.0))
        p_mw   = float(row.get("P_MW",        3.0))
        q_mvar = float(row.get("Q_Mvar",       1.5))
        s_mva  = float(row.get("S_MVA",        3.5))
        pf     = float(row.get("PF",          0.95))
        load   = float(row.get("loading_pct", 50.0))
        freq   = float(row.get("freq_Hz",      50.0))

        prev = self._prev.get(sub, {
            "freq_Hz": freq, "V_hv_kV": v_hv, "I_hv_A": i_hv, "P_MW": p_mw, "S_MVA": s_mva
        })
        delta_freq = freq - prev["freq_Hz"]
        delta_v_hv = v_hv - prev["V_hv_kV"]
        delta_i_hv = i_hv - prev["I_hv_A"]
        abs_delta_i_hv = abs(delta_i_hv)
        delta_p_mw = p_mw - prev["P_MW"]
        delta_s_mva = s_mva - prev["S_MVA"]

        self._prev[sub] = {
            "freq_Hz": freq, "V_hv_kV": v_hv, "I_hv_A": i_hv, "P_MW": p_mw, "S_MVA": s_mva
        }

        # Substation baselines
        base = self.SUBSTATION_BASELINES.get(norm_sub, {"P": 3.0, "I": 40.0, "S": 4.0})
        p_base = base["P"]
        i_base = base["I"]
        s_base = base["S"]
        eps = 1e-4

        # E1 Voltage features
        v_hv_pu = v_hv / 60.0
        v_lv_pu = v_lv / 5.5
        v_ratio = v_lv_pu / (v_hv_pu + eps)
        delta_v_pu = v_lv_pu - v_hv_pu

        # E2 Power and Dynamic Ratios
        p_ratio = p_mw / (p_base + eps)
        p_delta_ratio = (p_mw - p_base) / (p_base + eps)
        i_ratio = i_hv / (i_base + eps)
        s_ratio = s_mva / (s_base + eps)
        i2t_pu = i_ratio ** 2

        # 24 Full Numeric Features (matching E3-B pipeline)
        num_24 = [
            v_hv, v_lv, i_hv, p_mw, q_mvar, s_mva, pf, load, freq,
            delta_freq, delta_v_hv,
            v_hv_pu, v_lv_pu, v_ratio, delta_v_pu,
            p_ratio, p_delta_ratio, i_ratio, s_ratio,
            delta_i_hv, abs_delta_i_hv, delta_p_mw, delta_s_mva, i2t_pu
        ]

        expected_num = getattr(self, "_model_input_size", 35) - len(self.SUBSTATIONS)
        
        if expected_num == 24:
            raw_numeric = np.array(num_24, dtype=np.float32)
        elif expected_num == 25:
            raw_numeric = np.array(num_24 + [abs_delta_i_hv], dtype=np.float32)
        else:
            raw_numeric = np.array([v_hv, v_lv, i_hv, p_mw, q_mvar, s_mva, pf, load, freq, delta_freq, delta_v_hv], dtype=np.float32)

        if self.scaler is not None:
            scaler_dim = getattr(self.scaler, "n_features_in_", 0)
            if scaler_dim == len(raw_numeric):
                try:
                    scaled_numeric = self.scaler.transform(raw_numeric.reshape(1, -1))[0].astype(np.float32)
                except Exception:
                    scaled_numeric = raw_numeric
            elif scaler_dim > len(raw_numeric):
                mean = self.scaler.mean_[:len(raw_numeric)]
                scale = self.scaler.scale_[:len(raw_numeric)]
                scaled_numeric = ((raw_numeric - mean) / (scale + 1e-8)).astype(np.float32)
            elif scaler_dim < len(raw_numeric) and scaler_dim == 24 and len(raw_numeric) == 25:
                scaled_24 = self.scaler.transform(raw_numeric[:24].reshape(1, -1))[0]
                scaled_numeric = np.concatenate([scaled_24, [raw_numeric[24]]]).astype(np.float32)
            else:
                scaled_numeric = raw_numeric
        else:
            scaled_numeric = raw_numeric

        return np.concatenate([scaled_numeric, self._ohe_substation(sub)])

    def _predict_single(self, sub, row):
        step = self._build_step_feature(sub, row)
        buf  = self._buffers.setdefault(sub, [])
        buf.append(step)
        if len(buf) > self.WINDOW_SIZE:
            buf.pop(0)
        padded = buf if len(buf) == self.WINDOW_SIZE else [buf[0]] * (self.WINDOW_SIZE - len(buf)) + buf
        
        if self.model is not None:
            x = torch.tensor(np.array(padded), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1).squeeze(0).cpu().numpy()
            pred_idx   = int(np.argmax(probs))
            pred_label = self.idx_to_label.get(pred_idx, "normal")
            confidence = float(probs[pred_idx]) * 100.0
            top_probs  = {self.idx_to_label.get(i, ("class_%d" % i)): float(p) * 100.0
                          for i, p in enumerate(probs)}
            return {"pred_label": pred_label, "confidence": confidence, "top_probs": top_probs}

        lbl = self._rule_based(
            float(row.get("V_hv_kV", 60.0)), float(row.get("V_lv_kV", 5.5)),
            float(row.get("I_hv_A", 100.0)), float(row.get("loading_pct", 50.0)),
            float(row.get("freq_Hz", 50.0)),  float(row.get("PF", 0.95)))
        return {"pred_label": lbl, "confidence": 92.0, "top_probs": {lbl: 92.0, "normal": 8.0}}

    @staticmethod
    def _rule_based(v_hv, v_lv, i_hv, load, freq, pf):
        if v_hv < 5.0:   return "source_outage"
        if v_lv < 0.5:   return "transformer_trip"
        if v_hv < 35.0:  return "lg_fault" if pf < 0.3 else "ll_fault"
        if v_hv < 51.0:  return "voltage_sag"
        if load > 105.0: return "overload"
        if freq < 49.6:  return "under_frequency"
        if freq > 50.4:  return "over_frequency"
        if i_hv < 1.0:   return "breaker_trip"
        return "normal"

    FEATURE_NAMES_24 = [
        "V_hv_kV", "V_lv_kV", "I_hv_A", "P_MW", "Q_Mvar", "S_MVA", "PF", "loading_pct",
        "freq_Hz", "delta_freq_Hz", "delta_V_hv", "V_hv_pu", "V_lv_pu", "V_ratio", "delta_V_pu",
        "P_ratio", "P_delta_ratio", "I_ratio", "S_ratio", "delta_I_hv", "abs_delta_I_hv", "delta_P_MW", "delta_S_MVA", "I2t_pu"
    ]

    def compute_shap(self, sub_name, row=None, n_faulted=0):
        """
        Computes real-time Gradient x Input feature attributions for the active model.
        Returns a dict with feature names and signed importance values.
        """
        if self.model is None:
            return {"substation": sub_name, "values": {}}

        buf = self._buffers.get(sub_name, [])
        if not buf:
            if row is not None:
                step = self._build_step_feature(sub_name, row)
                buf = [step]
            else:
                return {"substation": sub_name, "values": {}}

        padded = buf if len(buf) == self.WINDOW_SIZE else [buf[0]] * (self.WINDOW_SIZE - len(buf)) + buf

        try:
            self.model.eval()
            x = torch.tensor(np.array(padded), dtype=torch.float32).unsqueeze(0).to(self.device)
            x.requires_grad = True

            out = self.model(x)
            pred_idx = int(torch.argmax(out, dim=1).item())
            pred_label = self.idx_to_label.get(pred_idx, "normal")

            # Backward on target class logit
            target_score = out[0, pred_idx]
            self.model.zero_grad()
            target_score.backward()

            if x.grad is not None:
                # Saliency attribution: (Input * Grad) summed across timesteps
                grad = x.grad[0] # (20, n_features)
                attr = (x[0] * grad).sum(dim=0).detach().cpu().numpy() # (n_features,)
                
                # Scale for visual clarity on horizontal bar chart (-1.0 to +1.0 relative)
                max_val = np.max(np.abs(attr[:len(self.FEATURE_NAMES_24)])) + 1e-8
                scaled_attr = attr[:len(self.FEATURE_NAMES_24)] / max_val

                values = {
                    name: float(round(float(scaled_attr[i]), 4))
                    for i, name in enumerate(self.FEATURE_NAMES_24)
                }
                return {
                    "substation": sub_name,
                    "pred_label": pred_label,
                    "values": values
                }
        except Exception as e:
            pass

        return {"substation": sub_name, "values": {}}
