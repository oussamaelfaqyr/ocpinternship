"""
Realistic SCADA Dataset Generator for OCP Youssoufia 60 kV Digital Twin.
========================================================================
Generates physically consistent synthetic SCADA datasets with multi-stage fault lifecycles,
localized fault indication, stochastic impedance, equipment aging, and sensor imperfections.
Includes dual-dataset generation (Balanced Training vs Imbalanced Realistic) and
N-1 routine maintenance topologies.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
NETWORK_MODEL_DIR = os.path.join(PARENT_DIR, "network_model")

if NETWORK_MODEL_DIR not in sys.path:
    sys.path.insert(0, NETWORK_MODEL_DIR)

from build_network import build_network
from fault_simulator import simulate_step_per_sub


# ---------------------------------------------------------------------------
# Fault types that are LOCALIZED (only affect the target substation)
# ---------------------------------------------------------------------------
LOCALIZED_FAULTS = ["overload", "lg_fault", "ll_fault", "breaker_trip", "transformer_trip", "maintenance"]

# Fault types that are NETWORK-WIDE (affect all substations physically)
NETWORK_FAULTS = ["voltage_sag", "source_outage", "under_frequency", "over_frequency"]

ALL_FAULT_TYPES = LOCALIZED_FAULTS + NETWORK_FAULTS


def _build_per_substation_schedule(substation_names, total_seconds, rng, fault_density=0.18):
    """
    Build an INDEPENDENT random fault schedule for each substation.
    """
    MIN_START_S   = 30
    MIN_GAP_S     = 60
    MIN_DUR_S     = 25
    MAX_DUR_S     = 120

    n_subs = len(substation_names)
    per_sub_fault_budget_s = int(total_seconds * fault_density)

    schedules = {sub: [] for sub in substation_names}

    # Network-wide events
    n_global = max(1, int(total_seconds / (800 / max(0.01, fault_density))))
    global_starts = sorted(
        rng.integers(MIN_START_S, max(MIN_START_S + 1, total_seconds - MAX_DUR_S),
                     size=n_global)
    )
    network_fault_windows = []
    for gs in global_starts:
        ftype = rng.choice(NETWORK_FAULTS)
        dur   = int(rng.integers(MIN_DUR_S, min(40, MAX_DUR_S)))
        end_s = min(gs + dur, total_seconds)
        if network_fault_windows and gs < network_fault_windows[-1][1] + MIN_GAP_S:
            continue
        network_fault_windows.append((int(gs), int(end_s), ftype))

    schedules["__global__"] = network_fault_windows

    # Per-substation localized faults
    for sub in substation_names:
        allocated_s = 0
        occupied = []

        for gstart, gend, _ in network_fault_windows:
            occupied.append((gstart - MIN_GAP_S, gend + MIN_GAP_S))

        attempts = 0
        while allocated_s < per_sub_fault_budget_s and attempts < 500:
            attempts += 1
            # Give maintenance a distinct probability (e.g., 10% of localized events)
            if rng.random() < 0.1:
                ftype = "maintenance"
                # Maintenance takes longer
                dur = int(rng.integers(1800, 7200)) # 30 min to 2 hours
            else:
                ftype = rng.choice([f for f in LOCALIZED_FAULTS if f != "maintenance"])
                dur = int(rng.integers(MIN_DUR_S, MAX_DUR_S))
                
            dur = min(dur, per_sub_fault_budget_s - allocated_s)
            if dur < MIN_DUR_S:
                break

            start = int(rng.integers(MIN_START_S, max(MIN_START_S + 1, total_seconds - dur)))
            end   = start + dur

            overlap = False
            for (os_, oe_) in occupied:
                if not (end + MIN_GAP_S <= os_ or start >= oe_ + MIN_GAP_S):
                    overlap = True
                    break
            if overlap:
                continue

            params = _sample_fault_params(ftype, rng)
            schedules[sub].append((ftype, start, end, params))
            occupied.append((start, end))
            allocated_s += dur

        schedules[sub].sort(key=lambda x: x[1])

    return schedules


def _sample_fault_params(fault_type, rng):
    """Return randomised physical parameters for a given fault type."""
    params = {}
    if fault_type in ("lg_fault", "ll_fault"):
        # Log-normal distribution for R_fault: many high-resistance (hidden) faults, occasional solid faults
        # Mean ~ 3.5 ohms, but with a long tail up to 30 ohms
        r = rng.lognormal(mean=1.0, sigma=0.8)
        params["r_fault_ohm"] = min(50.0, max(0.01, float(r)))
    elif fault_type == "overload":
        params["overload_factor"] = float(rng.uniform(2.20, 3.50))
    elif fault_type == "voltage_sag":
        params["sag_depth_pu"] = float(rng.uniform(0.60, 0.85))
    elif fault_type in ("under_frequency", "over_frequency"):
        params["freq_deviation_hz"] = float(rng.uniform(0.3, 2.0))
    elif fault_type == "maintenance":
        # Randomly choose which line to drop for N-1 simulation
        params["dropped_line"] = rng.choice(["SSP-U Calcination (UC2, parallel)", "Portique US-Portique Recette 2"])
    return params


def _active_fault_for_sub(sub, t, schedules):
    for (ftype, start_s, end_s, params) in schedules.get(sub, []):
        if start_s <= t < end_s:
            return ftype, params
    for (gstart, gend, ftype) in schedules.get("__global__", []):
        if gstart <= t < gend:
            return ftype, {}
    return "normal", {}


# ---------------------------------------------------------------------------
# Main Dataset Generator Class
# ---------------------------------------------------------------------------

class DatasetGenerator:
    def __init__(self, base_output_name="youssoufia_pandapower", output_path=None):
        if output_path:
            self.base_output_name = output_path.replace(".csv", "")
        else:
            self.base_output_name = base_output_name
        self._thread = None
        self._stop_event = threading.Event()
        self.is_generating = False
        self.progress = 0.0
        self.status_message = "Idle"

    def generate_datasets(self, total_seconds=7200, dt_s=1, seed=None):
        """Generates BOTH a balanced and an imbalanced realistic dataset."""
        print("Generating Balanced Dataset (Training)...")
        self.generate_dataset_sync(
            total_seconds=total_seconds,
            dt_s=dt_s,
            fault_density=0.20,  # 20% faults for balanced training
            output_file=f"{self.base_output_name}_balanced.csv",
            seed=seed
        )
        print("Generating Imbalanced Realistic Dataset (Validation/Testing)...")
        self.generate_dataset_sync(
            total_seconds=total_seconds,
            dt_s=dt_s,
            fault_density=0.002, # 0.2% faults for real-world representation
            output_file=f"{self.base_output_name}_realistic.csv",
            seed=seed + 1 if seed else None
        )

    def generate_dataset_sync(self, total_seconds=1800, dt_s=1, fault_density=0.18,
                               output_file=None, seed=None, progress_callback=None):
        self.is_generating = True
        self._stop_event.clear()
        
        output_path = output_file or f"{self.base_output_name}_dataset.csv"
        self.status_message = "Initializing Pandapower Network..."

        rng = np.random.default_rng(seed)
        net = build_network()

        substation_names = [n.replace(" trafo", "") for n in net.trafo["name"]]

        self.status_message = f"Building per-substation fault schedules (density={fault_density})..."
        schedules = _build_per_substation_schedule(substation_names, total_seconds, rng, fault_density=fault_density)

        n_steps = total_seconds // dt_s
        all_rows = []

        self.status_message = "Generating simulation timesteps..."

        for step in range(n_steps):
            if self._stop_event.is_set():
                self.status_message = f"Stopped early at step {step}/{n_steps}"
                break

            t = step * dt_s
            self.progress = (step + 1) / n_steps

            if progress_callback:
                progress_callback(self.progress, step, n_steps)

            sub_fault_map = {}
            for sub in substation_names:
                ftype, params = _active_fault_for_sub(sub, t, schedules)
                sub_fault_map[sub] = (ftype, params)

            tel_dict = simulate_step_per_sub(net, current_step=step, sub_fault_map=sub_fault_map)

            # Empty dict means runpp failed — skip this timestep entirely
            if not tel_dict:
                continue

            for sub, r in tel_dict.items():
                # For maintenance, the network is N-1 but the label remains 'normal' to avoid false positives!
                active_fault = sub_fault_map.get(sub, ("normal", {}))[0]
                is_maint = (active_fault == "maintenance")
                label_val = "normal" if is_maint else r["label"]

                all_rows.append({
                    "time_s":          t,
                    "substation":      sub,
                    "V_hv_kV":         r["V_hv_kV"],
                    "V_lv_kV":         r["V_lv_kV"],
                    "I_hv_A":          r["I_hv_A"],
                    "P_MW":            r["P_MW"],
                    "Q_Mvar":          r["Q_Mvar"],
                    "S_MVA":           r["S_MVA"],
                    "PF":              r["PF"],
                    "loading_pct":     r["loading_pct"],
                    "freq_Hz":         r.get("freq_Hz", 50.0),
                    "equip_status":    r["equip_status"],
                    "breaker_status":  r["breaker_status"],
                    "relay_status":    r["relay_status"],
                    "label":           label_val,
                    "event_label":     "normal" if is_maint else r.get("event_label", label_val),
                    "local_label":     "normal" if is_maint else r.get("local_label", label_val),
                    "lifecycle_stage": "normal" if is_maint else r.get("lifecycle_stage", "normal"),
                    "fault_location":  "NONE" if (label_val == "normal") else sub,
                })

        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        self.is_generating = False
        if not self._stop_event.is_set():
            self.status_message = f"Dataset generation complete! ({len(df)} samples saved to {output_path})"
        return df

    def generate_dataset_by_class_samples(self, class_samples: dict, dt_s: int = 1,
                                             output_file: str = None, seed: int = None,
                                             progress_callback=None):
        """
        Generate a dataset with a user-specified number of samples per fault class.

        Parameters
        ----------
        class_samples : dict
            Mapping of fault label -> number of samples wanted, e.g.
            {'normal': 500, 'lg_fault': 100, 'overload': 100, ...}
        dt_s : int
            Simulation timestep in seconds (default 1).
        output_file : str
            Output CSV path.  Defaults to ``<base_output_name>_custom.csv``.
        seed : int | None
            RNG seed for reproducibility.
        progress_callback : callable | None
            Called as ``callback(fraction, done_samples, total_samples)``.
        """
        self.is_generating = True
        self._stop_event.clear()

        output_path = output_file or f"{self.base_output_name}_custom.csv"
        rng = np.random.default_rng(seed)
        net = build_network()
        substation_names = [n.replace(" trafo", "") for n in net.trafo["name"]]
        n_subs = len(substation_names)

        # How many rows does one simulated timestep contribute?
        rows_per_step = n_subs  # one row per substation

        total_wanted = sum(class_samples.values())
        all_rows = []
        done_samples = 0
        global_step = 0

        self.status_message = "Generating samples per class..."

        for fault_type, n_wanted in class_samples.items():
            if self._stop_event.is_set():
                break
            if n_wanted <= 0:
                continue

            # For LOCALIZED faults we inject on one substation at a time;
            # for NETWORK faults they affect all subs so one step = n_subs rows.
            is_network = fault_type in NETWORK_FAULTS
            is_normal  = (fault_type == "normal")

            class_rows = []
            step = 0
            current_params = {}

            while True:
                import time
                time.sleep(0.01)  # Yield GIL so Streamlit UI can refresh
                
                if self._stop_event.is_set():
                    break

                # Sample parameters once per 15-step lifecycle
                if step % 15 == 0:
                    current_params = {} if is_normal else _sample_fault_params(fault_type, rng)

                # Step 14 is a "normal" step used to clear all trackers before the next lifecycle
                is_clear_step = (step % 15 == 14)

                if is_normal or is_clear_step:
                    # Run a plain normal step — clears trackers on all substations
                    sub_fault_map = {sub: ("normal", {}) for sub in substation_names}
                elif is_network:
                    # One network-wide fault step gives n_subs labelled rows
                    sub_fault_map = {sub: (fault_type, current_params) for sub in substation_names}
                else:
                    # Localized: rotate target substation per lifecycle, not per step
                    target_sub = substation_names[(step // 15) % n_subs]
                    sub_fault_map = {sub: ("normal", {}) for sub in substation_names}
                    sub_fault_map[target_sub] = (fault_type, current_params)

                try:
                    tel_dict = simulate_step_per_sub(net, current_step=global_step, sub_fault_map=sub_fault_map)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.status_message = f"❌ Sim Crash: {exc}"
                    self.is_generating = False
                    break

                # Empty dict means runpp failed — skip without polluting dataset
                if not tel_dict:
                    step += 1
                    global_step += 1
                    continue

                for sub, r in tel_dict.items():
                    active_fault = sub_fault_map[sub][0]
                    is_maint = (active_fault == "maintenance")
                    actual_label = r.get("label", "normal")

                    # For the "normal" class we only collect rows labelled normal
                    if is_normal and actual_label != "normal":
                        continue
                    # For fault classes we collect rows matching our target label
                    if not is_normal and not is_maint and actual_label != fault_type:
                        # Localized faults only tag the target sub; non-target subs stay normal —
                        # skip those to avoid polluting the fault class bucket.
                        continue

                    label_val = "normal" if is_maint else actual_label
                    class_rows.append({
                        "time_s":          global_step * dt_s,
                        "substation":      sub,
                        "V_hv_kV":         r["V_hv_kV"],
                        "V_lv_kV":         r["V_lv_kV"],
                        "I_hv_A":          r["I_hv_A"],
                        "P_MW":            r["P_MW"],
                        "Q_Mvar":          r["Q_Mvar"],
                        "S_MVA":           r["S_MVA"],
                        "PF":              r["PF"],
                        "loading_pct":     r["loading_pct"],
                        "freq_Hz":         r.get("freq_Hz", 50.0),
                        "equip_status":    r["equip_status"],
                        "breaker_status":  r["breaker_status"],
                        "relay_status":    r["relay_status"],
                        "label":           label_val,
                        "event_label":     "normal" if is_maint else r.get("event_label", label_val),
                        "local_label":     "normal" if is_maint else r.get("local_label", label_val),
                        "lifecycle_stage": "normal" if is_maint else r.get("lifecycle_stage", "normal"),
                        "fault_location":  "NONE" if (label_val == "normal") else sub,
                    })

                    # We no longer break early inside the sub loop to ensure full lifecycles

                step += 1
                global_step += 1
                done_samples = len(all_rows) + len(class_rows)
                pct = min(1.0, done_samples / max(1, total_wanted))

                # Always update self.progress / status so the UI fragment can read them
                # without needing a callback (supports background-thread mode)
                self.progress = pct
                self.status_message = (
                    f"[{fault_type}] {done_samples:,} / {total_wanted:,} samples…"
                )

                if progress_callback:
                    progress_callback(pct, done_samples, total_wanted)

                # Stop as soon as the quota is met (no need to wait for a lifecycle boundary)
                if len(class_rows) >= n_wanted:
                    break

            # Trim to exactly n_wanted so totals are always precise
            all_rows.extend(class_rows[:n_wanted])

        # Do NOT shuffle the rows! Time-series data must remain chronological for windowing.
        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        self.is_generating = False
        if self._stop_event.is_set():
            self.status_message = (
                f"Stopped & saved {len(df):,} samples → {os.path.basename(output_path)}"
            )
        else:
            self.status_message = (
                f"Custom dataset complete! ({len(df):,} samples saved)"
            )
        return df


if __name__ == "__main__":
    import os
    AI_DIR = os.path.abspath(os.path.join(PARENT_DIR, "ai", "dataset"))
    os.makedirs(AI_DIR, exist_ok=True)

    generator = DatasetGenerator()

    # Balanced training dataset — all 10 classes, ~2000 rows each
    print("Generating Balanced Dataset (Training)...")
    generator.generate_dataset_by_class_samples(
        class_samples={
            "normal":           10000,
            "lg_fault":          2000,
            "ll_fault":          2000,
            "overload":          2000,
            "voltage_sag":       2000,
            "under_frequency":   2000,
            "over_frequency":    2000,
            "source_outage":     2000,
            "breaker_trip":      2000,
            "transformer_trip":  2000,
        },
        output_file=os.path.join(AI_DIR, "train.csv"),
        seed=42,
    )

    # Realistic validation/test dataset — slightly imbalanced (normal is more prevalent)
    print("Generating Imbalanced Realistic Dataset (Validation/Testing)...")
    generator.generate_dataset_by_class_samples(
        class_samples={
            "normal":           3000,
            "lg_fault":          200,
            "ll_fault":          200,
            "overload":          200,
            "voltage_sag":       200,
            "under_frequency":   200,
            "over_frequency":    200,
            "source_outage":     200,
            "breaker_trip":      200,
            "transformer_trip":  200,
        },
        output_file=os.path.join(AI_DIR, "test_val.csv"),
        seed=99,
    )
