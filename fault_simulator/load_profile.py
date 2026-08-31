"""
Industrial Load Profile & Temporal Dynamics Engine for OCP Youssoufia 60 kV Network.
===================================================================================
Keys MUST match actual pandapower load names in build_network.py exactly.
"""

import numpy as np


_LOAD_PROFILES = {
    "Laverie/Sechage SN1 load": {"type": "batch_washing",  "period_s": 1200, "base_mw": 6.2,  "var_mw": 2.5},
    "Laverie/Sechage SN2 load": {"type": "batch_washing",  "period_s": 1380, "base_mw": 6.2,  "var_mw": 2.2},
    "Mine Mzinda DIS TR load":  {"type": "dragline_spikes", "period_s": 240,  "base_mw": 3.7,  "var_mw": 1.5},  # Lowered var_mw to prevent >100% normal loading
    "PSF Mine Bouchane SN6 load":{"type": "hoist_conveyor", "period_s": 600,  "base_mw": 1.24, "var_mw": 0.5},
    "Recette 3 SN3 load":       {"type": "hoist_conveyor",  "period_s": 900,  "base_mw": 0.99, "var_mw": 0.4},
    "Recette 9 SN9 load":       {"type": "hoist_conveyor",  "period_s": 450,  "base_mw": 1.24, "var_mw": 0.5},
    "U Calcination SN1 load":   {"type": "thermal_ramp",    "period_s": 2700, "base_mw": 6.2,  "var_mw": 1.8},
    "U Calcination SN2 load":   {"type": "thermal_ramp",    "period_s": 2900, "base_mw": 6.2,  "var_mw": 1.6},
    "U Calcination SN3 load":   {"type": "thermal_ramp",    "period_s": 3100, "base_mw": 4.95, "var_mw": 1.2},
    "SSP Local ST1 load":        {"type": "steady",          "period_s": 3600, "base_mw": 2.48, "var_mw": 0.3},
    "SSP Local ST2 load":        {"type": "steady",          "period_s": 3600, "base_mw": 2.48, "var_mw": 0.3},
}


class LoadProfileEngine:
    def __init__(self):
        self._ou_state = {name: 0.0 for name in _LOAD_PROFILES}
        self._qp_ou_state = {name: 0.0 for name in _LOAD_PROFILES}  # Separate OU state for Q/P ratio drift

    def get_temperature(self, t_sec: float) -> float:
        hour = (t_sec % 86400) / 3600.0
        base_temp = 25.0
        temp_swing = 12.0
        temp = base_temp - temp_swing * np.cos(2 * np.pi * (hour - 4.0) / 24.0)
        return temp

    def update_network_loads(self, net, t_sec: float):
        hour = (t_sec % 86400) / 3600.0
        
        if 6 <= hour < 14:
            shift_multiplier = 1.05 
        elif 14 <= hour < 22:
            shift_multiplier = 0.95 
        else:
            shift_multiplier = 0.70 
            
        temp_c = self.get_temperature(t_sec)
        cooling_factor = 1.0 + max(0, temp_c - 25.0) * 0.005

        alpha_alu = 0.0039
        temp_factor = 1.0 + alpha_alu * (temp_c - 20.0)
        if not hasattr(self, "_base_line_r"):
            self._base_line_r = net.line["r_ohm_per_km"].copy()
        
        assert len(self._base_line_r) == len(net.line), "Network line count changed across generation calls!"
            
        net.line["r_ohm_per_km"] = self._base_line_r * temp_factor

        theta = 0.05   
        sigma = 0.06   
        qp_theta = 0.02  # Slower mean reversion for PF drift
        qp_sigma = 0.015 # Smaller volatility for Q/P ratio

        for i, row in net.load.iterrows():
            load_name = row["name"]
            prof = _LOAD_PROFILES.get(load_name)
            if prof is None:
                continue

            ptype   = prof["type"]
            period  = prof["period_s"]
            base_mw = prof["base_mw"]
            var_mw  = prof["var_mw"]
            phase   = (t_sec % period) / float(period)

            if ptype == "batch_washing":
                cycle = 0.9 if phase < 0.75 else -0.5
            elif ptype == "dragline_spikes":
                cycle = 2.0 if (0.25 < phase < 0.55) else -0.15
            elif ptype == "thermal_ramp":
                cycle = 2.0 * (phase - 0.5)
            elif ptype == "hoist_conveyor":
                cycle = 0.7 if phase < 0.60 else -0.45
            else:  
                cycle = 0.1 * np.sin(2 * np.pi * phase)

            ou = self._ou_state[load_name]
            ou += -theta * ou + sigma * np.random.randn()
            self._ou_state[load_name] = ou

            p_mw = (base_mw + var_mw * cycle + ou) * shift_multiplier * cooling_factor
            p_mw = max(0.2, float(p_mw))    
            
            # Ornstein-Uhlenbeck drift for Q/P ratio (PF drifts naturally between 0.90 and 0.94)
            qp_ou = self._qp_ou_state[load_name]
            qp_ou += -qp_theta * qp_ou + qp_sigma * np.random.randn()
            self._qp_ou_state[load_name] = qp_ou
            
            qp_ratio = 0.42 + qp_ou  # Base 0.42, drifting slowly
            q_mvar = round(p_mw * qp_ratio, 4)

            net.load.at[i, "p_mw"]   = p_mw
            net.load.at[i, "q_mvar"] = q_mvar