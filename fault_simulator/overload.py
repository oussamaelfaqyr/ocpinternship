"""
Thermal Overload Simulator for OCP Youssoufia 60 kV Network.
============================================================
Simulates heavy industrial load surges (mine mill motor start, crusher overload, kiln surge).

Engineering Physics:
- Increases active and reactive load power gradually (100% to 160%+ of rating).
- Current increases proportionally (I = S / (sqrt(3)*V)).
- Transformer loading percent exceeds 100% (100% - 165%).
- Increased I^2*R losses cause additional local secondary voltage drop (V_LV drops 3-8%).
- Inverse-time thermal protection: trip time is determined by the ProtectionSequenceTracker,
  which applies the IEC inverse-time curve. This module ONLY handles load physics.
"""

import numpy as np
import pandapower as pp

HV_KV = 60.0
MV_KV = 5.5


class OverloadSimulator:
    def __init__(self, overload_factor=1.55):
        """
        :param overload_factor: Default max load multiplier (1.20 to 1.70).
        """
        self.overload_factor = overload_factor
        self._original_loads = {}
        self.reset()

    def reset(self):
        """Clears state for a new episode to prevent leakage."""
        self._original_loads.clear()

    def apply_fault(self, net, target_load_name, current_step_s=0, overload_factor=None):
        """
        Applies gradual thermal overload condition to target load.
        Modifies net.load in place — does NOT run power flow itself.
        """
        factor = overload_factor if overload_factor is not None else self.overload_factor
        matching_loads = net.load.index[net.load["name"] == target_load_name]
        load_idx = matching_loads[0] if len(matching_loads) > 0 else net.load.index[0]

        if load_idx not in self._original_loads:
            self._original_loads[load_idx] = (
                net.load.at[load_idx, "p_mw"],
                net.load.at[load_idx, "q_mvar"]
            )

        orig_p, orig_q = self._original_loads[load_idx]

        # Smooth gradual load ramp over first 10 steps (1.0 -> factor)
        ramp_progress = min(1.0, (current_step_s + 1) / 10.0)
        current_factor = 1.0 + (factor - 1.0) * ramp_progress

        net.load.at[load_idx, "p_mw"]   = orig_p * current_factor
        net.load.at[load_idx, "q_mvar"] = orig_q * current_factor

    def isolate_in_pandapower(self, net, target_sub_name):
        """
        Physically isolates the overloaded substation transformer in Pandapower.
        Called when the breaker opens (stage == 'isolated').
        Returns True if an element was taken offline.
        """
        taken_offline = False
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            if target_sub_name in sub or sub in target_sub_name:
                if net.trafo.at[i, "in_service"]:
                    net.trafo.at[i, "in_service"] = False
                    taken_offline = True
        return taken_offline

    def clear_fault(self, net):
        """Restores baseline loads."""
        for load_idx, (orig_p, orig_q) in self._original_loads.items():
            if load_idx in net.load.index:
                net.load.at[load_idx, "p_mw"]   = orig_p
                net.load.at[load_idx, "q_mvar"] = orig_q
        self._original_loads.clear()

    def get_telemetry_override(self, net, target_sub_name, stage_info: dict,
                                duration_elapsed_s: int = 0):
        """
        Returns telemetry dictionary for overload state.
        Protection state is read from stage_info (supplied by ProtectionSequenceTracker).
        This method runs its own runpp because it has already modified the network loads.
        """
        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False

        stage = stage_info.get("stage", "fault_detected")
        equip_status  = stage_info.get("equip_status",  "OVERLOADED")
        relay_status  = stage_info.get("relay_status",  "ALARM")
        breaker_status = stage_info.get("breaker_status", "CLOSED")

        results = {}
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            is_target = (sub in target_sub_name or target_sub_name in sub)
            is_isolated_stage = (stage in ["isolated", "breaker_open"])

            if is_target and is_isolated_stage:
                # Breaker has opened — substation is de-energised on LV side
                v_hv = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else 59.5
                results[sub] = {
                    "V_hv_kV":     round(v_hv, 3),
                    "V_lv_kV":     0.0,
                    "I_hv_A":      0.0,
                    "P_MW":        0.0,
                    "Q_Mvar":      0.0,
                    "S_MVA":       0.0,
                    "PF":          0.0,
                    "loading_pct": 0.0,
                    "freq_Hz":     round(50.0 + 0.012 * np.random.randn(), 3),
                    "equip_status":   equip_status,
                    "breaker_status": breaker_status,
                    "relay_status":   relay_status,
                    "label":          "overload",
                    "lifecycle_stage": stage,
                }
                continue

            if converged and i in net.res_trafo.index:
                rt = net.res_trafo.loc[i]
                v_hv      = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV
                v_lv_base = net.res_bus.loc[lv_b, "vm_pu"] * MV_KV
                p         = float(rt["p_hv_mw"])
                q         = float(rt["q_hv_mvar"])
                i_hv_a    = float(rt["i_hv_ka"] * 1000.0)
                loading_pct = float(rt["loading_percent"])
            else:
                v_hv      = 59.0
                v_lv_base = 5.3
                p         = 8.0
                q         = 3.5
                i_hv_a    = 180.0
                loading_pct = 115.0

            s_mva = np.sqrt(p**2 + q**2) + 1e-6
            pf    = abs(p) / s_mva

            if is_target:
                # Additional I²R voltage drop on LV side (3-8% of nominal)
                # ΔV ≈ R_trafo * I / V_nominal, simplified as loading-proportional
                lv_drop_factor = max(0.92, 1.0 - 0.0005 * max(0, loading_pct - 100.0))
                v_lv = v_lv_base * lv_drop_factor
                sub_label = "overload"
                sub_equip = equip_status
                sub_relay = relay_status
                sub_breaker = breaker_status
                sub_stage = stage
            else:
                v_lv = v_lv_base
                sub_label = "normal"
                sub_equip = "HEALTHY"
                sub_relay = "NORMAL"
                sub_breaker = "CLOSED"
                sub_stage = "normal"

            results[sub] = {
                "V_hv_kV":     round(v_hv, 3),
                "V_lv_kV":     round(max(0.0, v_lv), 3),
                "I_hv_A":      round(i_hv_a, 2),
                "P_MW":        round(p, 3),
                "Q_Mvar":      round(q, 3),
                "S_MVA":       round(s_mva, 3),
                "PF":          round(min(1.0, pf), 3),
                "loading_pct": round(loading_pct, 2),
                "freq_Hz":     round(50.0 + 0.012 * np.random.randn(), 3),
                "equip_status":   sub_equip,
                "breaker_status": sub_breaker,
                "relay_status":   sub_relay,
                "label":          sub_label,
                "lifecycle_stage": sub_stage,
            }

        return results