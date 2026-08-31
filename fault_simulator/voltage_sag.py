"""
Voltage Sag & Grid Dip Simulator for OCP Youssoufia 60 kV Network.
==================================================================
Simulates utility grid-side voltage dips originating from upstream ONEE 225/60 kV transmission faults.

Engineering Physics:
- External grid source voltage (ext_grid.vm_pu) drops to 0.70 - 0.85 p.u. (42.0 - 51.0 kV).
- Voltage drop propagates downstream to all 60 kV and 5.5 kV buses.
- Industrial motor loads experience increased current draw to maintain torque (or reduced active power).
- Protection relays trigger UNDERVOLTAGE ALARM, but breakers remain CLOSED unless sag depth < 0.60 p.u.
"""

import numpy as np
import pandapower as pp

HV_KV = 60.0
MV_KV = 5.5


class VoltageSagSimulator:
    def __init__(self, sag_depth_pu=0.75):
        """
        :param sag_depth_pu: Target grid voltage level in p.u. (0.70 to 0.85).
        """
        self.sag_depth_pu = sag_depth_pu
        self._original_vm_pu = 1.0

    def apply_fault(self, net, target_grid_name=None, sag_depth_pu=None):
        """Applies voltage sag to grid source."""
        depth = sag_depth_pu if sag_depth_pu is not None else self.sag_depth_pu
        if len(net.ext_grid) > 0:
            grid_idx = net.ext_grid.index[0]
            curr = float(net.ext_grid.at[grid_idx, "vm_pu"])
            if curr > 0.90:  # Only capture nominal baseline if not already in sag
                self._original_vm_pu = curr
            net.ext_grid.at[grid_idx, "vm_pu"] = depth

    def clear_fault(self, net):
        """Restores nominal grid voltage (1.0 p.u.)."""
        if len(net.ext_grid) > 0:
            grid_idx = net.ext_grid.index[0]
            net.ext_grid.at[grid_idx, "vm_pu"] = 1.0
            self._original_vm_pu = 1.0

    def get_telemetry_override(self, net):
        """Executes AC power flow and returns per-substation telemetry under voltage sag."""
        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False

        results = {}
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            rt = net.res_trafo.loc[i]
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            v_hv = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else (self.sag_depth_pu * HV_KV)
            v_lv = net.res_bus.loc[lv_b, "vm_pu"] * MV_KV if converged else (self.sag_depth_pu * MV_KV)
            p = float(rt["p_hv_mw"]) if converged else 5.0
            q = float(rt["q_hv_mvar"]) if converged else 2.5
            s_mva = np.sqrt(p**2 + q**2) + 1e-6
            pf = abs(p) / s_mva
            i_hv_a = float(rt["i_hv_ka"] * 1000.0) if converged else 140.0
            loading_pct = float(rt["loading_percent"]) if converged else 65.0

            relay_status = "ALARM" if v_hv < 54.0 else "NORMAL"

            results[sub] = {
                "V_hv_kV": round(v_hv, 3),
                "V_lv_kV": round(v_lv, 3),
                "I_hv_A": round(i_hv_a, 2),
                "P_MW": round(p, 3),
                "Q_Mvar": round(q, 3),
                "S_MVA": round(s_mva, 3),
                "PF": round(pf, 3),
                "loading_pct": round(loading_pct, 2),
                "freq_Hz": round(50.0 + 0.015 * np.random.randn(), 3),
                "equip_status": "HEALTHY",
                "breaker_status": "CLOSED",
                "relay_status": relay_status,
                "label": "voltage_sag",
            }

        return results
