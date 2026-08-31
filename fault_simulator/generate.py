"""
OCP Youssoufia 60 kV Network Dataset Generator
================================================
Generates a large, realistic synthetic dataset of SCADA telemetry for training
the GRU fault detection model.
"""

import numpy as np
import pandas as pd
import pandapower as pp
import time
import os
import sys
import warnings
from pathlib import Path

# Suppress pandas/pandapower future warnings to keep the console clean
warnings.filterwarnings('ignore')

# --- PATH FIX ---
CURRENT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = CURRENT_DIR.parent / "network_model"
sys.path.append(str(NETWORK_DIR))

from build_network import build_network

from load_profile import LoadProfileEngine
from protection_engine import ProtectionSequenceTracker
from lg_fault import LGFaultSimulator
from ll_fault import LLFaultSimulator
from overload import OverloadSimulator
from frequency_fault import FrequencySimulator, generate_freq_trajectory
from voltage_sag import VoltageSagSimulator
from outage_fault import OutageSimulator
from sensor_effects import SensorImperfectionEngine

# ============================================================
# CONFIGURATION
# ============================================================
NUM_EPISODES = 200         # Change to 2000 for the final full dataset
STEPS_PER_EPISODE = 150    # Length of each episode in seconds (timesteps)

OUTPUT_DIR = CURRENT_DIR.parent / "dataset"
OUTPUT_FILE = OUTPUT_DIR / "train.csv"

FAULT_PROBABILITIES = {
    "normal": 0.40,
    "lg_fault": 0.15,
    "ll_fault": 0.15,
    "overload": 0.10,
    "under_frequency": 0.05,
    "over_frequency": 0.05,
    "voltage_sag": 0.05,
    "breaker_trip": 0.025,
    "transformer_trip": 0.025,
    "source_outage": 0.025,
}

# ============================================================
# MAIN GENERATOR CLASS
# ============================================================
class DatasetGenerator:
    def __init__(self):
        self.substations = [
            "Laverie/Sechage SN1", "Laverie/Sechage SN2", "Mine Mzinda DIS TR",
            "PSF Mine Bouchane SN6", "Recette 3 SN3", "Recette 9 SN9",
            "U Calcination SN1", "U Calcination SN2", "U Calcination SN3",
            "SSP Local ST1", "SSP Local ST2"
        ]
        
        self.load_engine = LoadProfileEngine()
        self.protection_tracker = ProtectionSequenceTracker()
        self.sensor_engine = SensorImperfectionEngine(self.substations)
        
        self.lg_sim = LGFaultSimulator()
        self.ll_sim = LLFaultSimulator()
        self.overload_sim = OverloadSimulator()
        self.freq_sim = FrequencySimulator()
        self.sag_sim = VoltageSagSimulator()
        self.outage_sim = OutageSimulator()

    def _select_fault_type(self):
        r = np.random.rand()
        cum_prob = 0.0
        for fault, prob in FAULT_PROBABILITIES.items():
            cum_prob += prob
            if r < cum_prob:
                return fault
        return "normal"

    def _select_target_substation(self):
        target_subs = [s for s in self.substations if "SSP" not in s]
        return np.random.choice(target_subs)

    def generate_episode(self, episode_idx):
        """Generates a single episode of data and returns a DataFrame."""
        net = build_network()  # Get a fresh network
        
        # --- CRITICAL FIX: RESET ENGINES TO PREVENT STATE LEAKAGE ---
        self.sensor_engine.reset()
        self.overload_sim.reset()
        self.outage_sim.reset()
        # Voltage sag simulator state is reset implicitly by apply_fault
        
        fault_type = self._select_fault_type()
        target_sub = self._select_target_substation()
        
        r_fault = np.random.uniform(0.1, 12.0)
        fault_start_step = np.random.randint(20, 50)
        duration = np.random.randint(40, 80)
        overload_factor = np.random.uniform(1.20, 1.70)
        sag_depth = np.random.uniform(0.70, 0.85)
        
        self.protection_tracker.trigger_fault(
            ftype=fault_type,
            target_sub=target_sub,
            start_step=fault_start_step,
            duration_steps=duration,
            r_ohm=r_fault,
            overload_factor=overload_factor
        )
        
        if fault_type == "under_frequency":
            self.protection_tracker.set_freq_trajectory(
                generate_freq_trajectory("under", STEPS_PER_EPISODE)
            )
        elif fault_type == "over_frequency":
            self.protection_tracker.set_freq_trajectory(
                generate_freq_trajectory("over", STEPS_PER_EPISODE)
            )

        episode_rows = []
        
        for step in range(STEPS_PER_EPISODE):
            t_sec = step
            
            self.load_engine.update_network_loads(net, t_sec)
            stage_info = self.protection_tracker.get_lifecycle_stage(step)
            telemetry = {}
            
            if fault_type == "lg_fault":
                telemetry = self.lg_sim.apply_fault(net, target_sub, stage_info, r_fault)
            elif fault_type == "ll_fault":
                telemetry = self.ll_sim.apply_fault(net, target_sub, stage_info, r_fault)
            elif fault_type == "overload":
                self.overload_sim.apply_fault(net, f"{target_sub} load", step, overload_factor)
                telemetry = self.overload_sim.get_telemetry_override(net, target_sub, stage_info, step)
            elif fault_type == "under_frequency":
                freq_hz = self.protection_tracker.get_freq_value()
                telemetry = self.freq_sim.get_telemetry_override(net, mode="under", freq_hz=freq_hz, stage_info=stage_info)
            elif fault_type == "over_frequency":
                freq_hz = self.protection_tracker.get_freq_value()
                telemetry = self.freq_sim.get_telemetry_override(net, mode="over", freq_hz=freq_hz, stage_info=stage_info)
            elif fault_type == "voltage_sag":
                self.sag_sim.apply_fault(net, sag_depth_pu=sag_depth)
                telemetry = self.sag_sim.get_telemetry_override(net)
            elif fault_type == "breaker_trip":
                if stage_info["stage"] == "isolated" and step == fault_start_step + 1:
                    self.outage_sim.apply_breaker_trip(net, target_sub)
                try:
                    pp.runpp(net, numba=False)
                    conv = True
                except:
                    conv = False
                telemetry = self.outage_sim.get_breaker_trip_telemetry(net, target_sub, stage_info, conv)
            elif fault_type == "transformer_trip":
                if stage_info["stage"] == "isolated" and step == fault_start_step + 1:
                    self.outage_sim.apply_transformer_trip(net, target_sub)
                try:
                    pp.runpp(net, numba=False)
                    conv = True
                except:
                    conv = False
                telemetry = self.outage_sim.get_transformer_trip_telemetry(net, target_sub, stage_info, conv)
            elif fault_type == "source_outage":
                if stage_info["stage"] == "isolated" and step == fault_start_step + 1:
                    self.outage_sim.apply_source_outage(net)
                telemetry = self.outage_sim.get_source_outage_telemetry(net, stage_info)
            else:
                try:
                    pp.runpp(net, numba=False)
                    conv = True
                except:
                    conv = False
                telemetry = {}
                for i, tname in net.trafo["name"].items():
                    sub = tname.replace(" trafo", "")
                    hv_b = net.trafo.at[i, "hv_bus"]
                    lv_b = net.trafo.at[i, "lv_bus"]
                    rt = net.res_trafo.loc[i] if conv else None
                    
                    v_hv = net.res_bus.loc[hv_b, "vm_pu"] * 60.0 if conv else 59.8
                    v_lv = net.res_bus.loc[lv_b, "vm_pu"] * 5.5 if conv else 5.48
                    p = float(rt["p_hv_mw"]) if rt is not None else 5.0
                    q = float(rt["q_hv_mvar"]) if rt is not None else 2.2
                    s = np.sqrt(p**2 + q**2) + 1e-6
                    pf = abs(p) / s
                    i_hv = float(rt["i_hv_ka"] * 1000.0) if rt is not None else 115.0
                    load_pct = float(rt["loading_percent"]) if rt is not None else 55.0
                    
                    freq = 50.0 + 0.012 * np.random.randn()
                    
                    telemetry[sub] = {
                        "V_hv_kV": round(v_hv, 3), "V_lv_kV": round(v_lv, 3),
                        "I_hv_A": round(i_hv, 2), "P_MW": round(p, 3), "Q_Mvar": round(q, 3),
                        "S_MVA": round(s, 3), "PF": round(min(1.0, pf), 3), "loading_pct": round(load_pct, 2),
                        "freq_Hz": round(freq, 3), "equip_status": "HEALTHY",
                        "breaker_status": "CLOSED", "relay_status": "NORMAL",
                        "label": "normal", "lifecycle_stage": "normal"
                    }

            for sub, data in telemetry.items():
                data["fault_location"] = target_sub if fault_type != "normal" else "NONE"
                data["event_label"] = fault_type
                data["local_label"] = data.get("label", "normal")
                
                data = self.sensor_engine.apply_sensor_imperfections(sub, data)
                
                row = {"time_s": t_sec, "substation": sub, "episode_id": episode_idx}
                row.update(data)
                episode_rows.append(row)

        return pd.DataFrame(episode_rows)

    def run(self):
        print("=" * 60)
        print("OCP DATASET GENERATION STARTED")
        print("=" * 60)
        print(f"Generating {NUM_EPISODES} episodes...")
        
        all_dfs = []
        start_time = time.time()
        
        for ep in range(NUM_EPISODES):
            df = self.generate_episode(ep)
            all_dfs.append(df)
            
            elapsed = time.time() - start_time
            print(f"\r  Progress: {ep + 1}/{NUM_EPISODES} episodes generated | Elapsed: {elapsed:.1f}s", end="", flush=True)
        
        print("\n\nConcatenating dataset...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to {OUTPUT_FILE}...")
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("=" * 60)
        print("GENERATION COMPLETE!")
        print(f"Total rows: {len(final_df)}")
        print(f"Label distribution:\n{final_df['label'].value_counts()}")
        print("=" * 60)

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.run()