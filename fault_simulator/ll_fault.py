"""
Localized Line-to-Line (LL) Short Circuit Fault Simulator — OCP Youssoufia 60 kV.
==================================================================================
Supports protection lifecycle stage_info dict, stochastic fault impedance, 
and physics-based P/Q calculations.
"""

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc

HV_KV = 60.0
MV_KV = 5.5


class LLFaultSimulator:
    def __init__(self, fault_r_ohm=0.3):
        self.fault_r_ohm = fault_r_ohm

    def apply_fault(self, net, target_sub_name, stage_info: dict, r_fault_ohm: float = None):
        r_ohm = r_fault_ohm if r_fault_ohm is not None else self.fault_r_ohm

        fault_bus_idx = net.bus.index[0]
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            if target_sub_name in sub or sub in target_sub_name:
                fault_bus_idx = net.trafo.at[i, "hv_bus"]
                break

        try:
            sc.calc_sc(net, bus=fault_bus_idx, fault="2ph", case="max", ip=True, r_fault_ohm=r_ohm)
            ik_ka = net.res_bus_sc.loc[fault_bus_idx, "ikss_ka"] * (np.sqrt(3) / 2)
        except Exception:
            z_f = max(0.2, np.sqrt(r_ohm ** 2 + 1.0 ** 2))
            ik_ka = (HV_KV / np.sqrt(3)) / (z_f * 10.0) * 0.866

        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False

        stage = stage_info.get("stage", "fault_detected")
        results = {}
        base_freq = 50.0 + 0.012 * np.random.randn()

        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]
            is_faulted = (hv_b == fault_bus_idx or target_sub_name in sub or sub in target_sub_name)

            if is_faulted:
                if stage == "pre_fault":
                    v_hv = (0.72 + 0.08 * np.random.rand()) * HV_KV
                    v_lv = (0.78 + 0.08 * np.random.rand()) * MV_KV
                    i_hv_a = 260.0 + 60.0 * np.random.rand()
                    p_mw = 3.5
                    q_mvar = 4.5
                    pf = 0.61
                    loading_pct = 90.0
                elif stage in ["fault_detected", "relay_operated"]:
                    # Voltage collapse severity + slight randomness
                    dip = min(0.80, 0.25 + 0.04 * r_ohm + 0.05 * np.random.rand())
                    v_hv = max(8.0, dip * HV_KV)
                    v_lv = max(0.8, dip * 1.2 * MV_KV)
                    i_hv_a = min(7000.0, ik_ka * 1000.0 / (1.0 + 0.04 * r_ohm))
                    
                    # Dynamic Fault Power Factor
                    fault_pf = np.random.uniform(0.15, 0.40)
                    sin_phi = np.sin(np.arccos(fault_pf))
                    
                    # Calculate P and Q dynamically
                    p_mw = (np.sqrt(3) * v_hv * i_hv_a * fault_pf) / 1000.0
                    q_mvar = (np.sqrt(3) * v_hv * i_hv_a * sin_phi) / 1000.0
                    
                    pf = fault_pf
                    loading_pct = min(240.0, (i_hv_a / 120.0) * 100.0)
                    
                elif stage == "breaker_open":
                    v_hv, v_lv, i_hv_a = 15.0, 0.3, 12.0
                    p_mw, q_mvar, pf, loading_pct = 0.1, 0.2, 0.2, 4.0
                else:  # isolated / restored
                    v_hv, v_lv, i_hv_a, p_mw, q_mvar, pf, loading_pct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                results[sub] = {
                    "V_hv_kV": round(v_hv, 3), "V_lv_kV": round(v_lv, 3),
                    "I_hv_A": round(i_hv_a, 2), "P_MW": round(p_mw, 3),
                    "Q_Mvar": round(q_mvar, 3),
                    "S_MVA": round(np.sqrt(p_mw**2 + q_mvar**2) + 1e-6, 3),
                    "PF": round(min(1.0, pf), 3), "loading_pct": round(loading_pct, 2),
                    "freq_Hz": round(base_freq + 0.002 * np.random.randn(), 3), 
                    "equip_status": stage_info.get("equip_status", "FAULTED"),
                    "breaker_status": stage_info.get("breaker_status", "CLOSED"),
                    "relay_status": stage_info.get("relay_status", "PICKUP"),
                    "label": "ll_fault", "lifecycle_stage": stage,
                }
            else:
                dist = abs(int(hv_b) - int(fault_bus_idx)) + 1
                sag = max(0.72, 1.0 - (0.28 / dist))
                v_hv_n = (net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else 59.5) * sag
                v_lv_n = (net.res_bus.loc[lv_b, "vm_pu"] * MV_KV if converged else 5.45) * sag
                i_hv_n = float(net.res_trafo.loc[i, "i_hv_ka"] * 1000.0 * 1.25) if converged else 155.0
                p_n = float(net.res_trafo.loc[i, "p_hv_mw"]) if converged else 4.5
                q_n = float(net.res_trafo.loc[i, "q_hv_mvar"] * 1.3) if converged else 2.8

                results[sub] = {
                    "V_hv_kV": round(v_hv_n, 3), "V_lv_kV": round(v_lv_n, 3),
                    "I_hv_A": round(i_hv_n, 2), "P_MW": round(p_n, 3),
                    "Q_Mvar": round(q_n, 3),
                    "S_MVA": round(np.sqrt(p_n**2 + q_n**2) + 1e-6, 3),
                    "PF": round(abs(p_n) / (np.sqrt(p_n**2 + q_n**2) + 1e-6), 3),
                    "loading_pct": round(min(100.0, (i_hv_n / 120.0) * 100.0), 2),
                    "freq_Hz": round(base_freq + 0.002 * np.random.randn(), 3), 
                    "equip_status": "HEALTHY",
                    "breaker_status": "CLOSED", "relay_status": "ALARM",
                    "label": "normal", "lifecycle_stage": "normal",
                }
        return results