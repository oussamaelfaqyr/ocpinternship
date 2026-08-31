"""
Industrial SCADA Sensor Imperfections & Calibration Bias Engine.
================================================================
Models real-world field transducer errors:
- Constant PT/CT transducer calibration offset & gain bias per substation
- Frozen sensor measurements (packet hold)
- Missing telemetry drops (intermittent nulls/NaNs)
- Equipment aging degradation
- Asynchronous sampling noise
"""

import numpy as np
import copy

class SensorImperfectionEngine:
    def __init__(self, substations: list):
        # Generate persistent PT/CT calibration bias per substation (±1.5%)
        np.random.seed(42)
        self.voltage_bias = {sub: float(np.random.uniform(0.985, 1.015)) for sub in substations}
        self.current_bias = {sub: float(np.random.uniform(0.982, 1.018)) for sub in substations}
        self.efficiency_factor = {sub: float(np.random.uniform(0.965, 0.988)) for sub in substations}
        self.reset()

    def reset(self):
        """Clears state to prevent leakage between episodes."""
        self._last_values = {}
        self._freeze_counter = {}

    def apply_sensor_imperfections(self, sub: str, telemetry: dict, noise_std=0.003) -> dict:
        """Applies calibration bias, Gaussian noise, occasional packet drops/freezes, and NaNs."""
        out = dict(telemetry)
        
        v_bias = self.voltage_bias.get(sub, 1.0)
        i_bias = self.current_bias.get(sub, 1.0)
        eff = self.efficiency_factor.get(sub, 0.98)

        # 1. Apply PT/CT Calibration, Gain Bias, and Async sampling noise
        async_noise = noise_std * 1.5 if np.random.rand() < 0.1 else noise_std

        if out["V_hv_kV"] > 0:
            out["V_hv_kV"] = round(float(out["V_hv_kV"] * v_bias * (1.0 + async_noise * np.random.randn())), 3)
        if out["V_lv_kV"] > 0:
            out["V_lv_kV"] = round(float(out["V_lv_kV"] * v_bias * (1.0 + async_noise * np.random.randn())), 3)
        if out["I_hv_A"] > 0:
            out["I_hv_A"] = round(float(out["I_hv_A"] * i_bias * (1.0 + async_noise * np.random.randn())), 2)
        
        # Apply transformer efficiency impact on active power
        out["P_MW"] = round(float(out["P_MW"] * eff), 3)

        # Recompute physical derivatives to maintain consistency (S = sqrt(P^2+Q^2))
        p = out.get("P_MW", 0.0)
        q = out.get("Q_Mvar", 0.0)
        s = np.sqrt(p**2 + q**2) + 1e-6
        out["S_MVA"] = round(float(s), 3)
        out["PF"] = round(float(abs(p) / s), 3)

        # 2. Complete Data Drop (NaN Injection) - 0.1% chance
        if np.random.rand() < 0.001:
            for k in ["V_hv_kV", "V_lv_kV", "I_hv_A", "P_MW", "Q_Mvar", "S_MVA", "PF", "loading_pct"]:
                if k in out:
                    out[k] = np.nan
            return out

        # 3. Random Packet Freeze (1.2% probability of sticking for 2-5 timesteps)
        is_frozen = (sub in self._freeze_counter and self._freeze_counter[sub] > 0)
        
        if is_frozen:
            self._freeze_counter[sub] -= 1
            # Return previous snapshot (stuck value)
            if sub in self._last_values:
                last = copy.deepcopy(self._last_values[sub])
                # Maintain the current label/stage, but freeze numerical data
                last["label"] = out.get("label", "normal")
                last["lifecycle_stage"] = out.get("lifecycle_stage", "normal")
                last["event_label"] = out.get("event_label", "normal")
                last["local_label"] = out.get("local_label", "normal")
                last["fault_location"] = out.get("fault_location", "NONE")
                return last
        elif np.random.rand() < 0.012:
            self._freeze_counter[sub] = np.random.randint(2, 6)
            if sub in self._last_values:
                last = copy.deepcopy(self._last_values[sub])
                last["label"] = out.get("label", "normal")
                last["lifecycle_stage"] = out.get("lifecycle_stage", "normal")
                last["event_label"] = out.get("event_label", "normal")
                last["local_label"] = out.get("local_label", "normal")
                last["fault_location"] = out.get("fault_location", "NONE")
                return last

        # ONLY update last_values if we are not frozen
        self._last_values[sub] = dict(out)
        return out