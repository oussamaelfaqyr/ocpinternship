"""
Localized Line-to-Ground (LG) Short Circuit Fault Simulator.
===========================================================
Uses cached IEC 60909 short-circuit currents (sc_engine) and graph-based
radial voltage sag propagation. No per-step sc.calc_sc() or extra runpp —
values come from the cache computed once per fault lifecycle.
"""

import numpy as np

from sc_engine import calc_sc_cached, compute_sag_factors

HV_KV = 60.0
MV_KV = 5.5


class LGFaultSimulator:
    def __init__(self, fault_r_ohm=0.5, fault_x_ohm=1.2):
        self.fault_r_ohm = fault_r_ohm
        self.fault_x_ohm = fault_x_ohm

    def apply_fault(self, net, target_sub_name, stage_info: dict,
                    r_fault_ohm: float = None):
        """
        LG fault telemetry from cached SC physics + radial sag propagation.
        """
        r_ohm = r_fault_ohm if r_fault_ohm is not None else self.fault_r_ohm

        fault_bus_idx = net.bus.index[0]
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            if target_sub_name in sub or sub in target_sub_name:
                fault_bus_idx = net.trafo.at[i, "hv_bus"]
                break

        # --- Cached IEC 60909 short-circuit current + fault-bus voltage ---
        ik_ka, vm_pu_fault = calc_sc_cached(net, fault_bus_idx, "1ph", r_ohm)

        # Radial sag propagation to all buses (cached per fault bus + severity)
        sag = compute_sag_factors(net, int(fault_bus_idx), vm_pu_fault)

        stage = stage_info.get("stage", "fault_detected")
        results = {}
        base_freq = 50.0 + 0.012 * np.random.randn()

        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            # Per-transformer nominal current (physical rating)
            sn_mva   = float(net.trafo.at[i, "sn_mva"])
            i_nom_a  = sn_mva / (np.sqrt(3) * HV_KV) * 1000.0

            is_faulted_sub = (hv_b == fault_bus_idx or
                              target_sub_name in sub or sub in target_sub_name)

            if is_faulted_sub:
                if stage == "pre_fault":
                    v_hv = (0.70 + 0.10 * np.random.rand()) * HV_KV
                    v_lv = (0.75 + 0.10 * np.random.rand()) * MV_KV
                    i_hv_a = 280.0 + 80.0 * np.random.rand()
                    p_mw = 4.0
                    q_mvar = 5.0
                    pf = 0.62
                    loading_pct = 95.0
                elif stage in ["fault_detected", "relay_operated"]:
                    # Voltage collapse from the SC voltage-divider result
                    v_hv = max(2.0, vm_pu_fault * HV_KV * (0.95 + 0.05 * np.random.rand()))
                    v_lv = max(0.3, vm_pu_fault * 1.3 * MV_KV)
                    # Fault current: ikss already includes r_fault via IEC 60909
                    i_hv_a = float(min(12000.0, ik_ka * 1000.0))

                    # Arc fault power factor (0.10 arc, 0.35 near-bolted)
                    fault_pf = np.random.uniform(0.10, 0.35)
                    sin_phi = np.sin(np.arccos(fault_pf))

                    p_mw = (np.sqrt(3) * v_hv * i_hv_a * fault_pf) / 1000.0
                    q_mvar = (np.sqrt(3) * v_hv * i_hv_a * sin_phi) / 1000.0

                    pf = fault_pf
                    # Loading relative to the actual trafo rating
                    loading_pct = min(2500.0, (i_hv_a / i_nom_a) * 100.0)

                elif stage == "breaker_open":
                    v_hv = 12.0
                    v_lv = 0.2
                    i_hv_a = 15.0
                    p_mw = 0.1
                    q_mvar = 0.2
                    pf = 0.2
                    loading_pct = 5.0
                else:  # isolated
                    v_hv, v_lv, i_hv_a = 0.0, 0.0, 0.0
                    p_mw, q_mvar, pf, loading_pct = 0.0, 0.0, 0.0, 0.0

                results[sub] = {
                    "V_hv_kV": round(v_hv, 3),
                    "V_lv_kV": round(v_lv, 3),
                    "I_hv_A": round(i_hv_a, 2),
                    "P_MW": round(p_mw, 3),
                    "Q_Mvar": round(q_mvar, 3),
                    "S_MVA": round(np.sqrt(p_mw**2 + q_mvar**2), 3),
                    "PF": round(min(1.0, pf), 3),
                    "loading_pct": round(loading_pct, 2),
                    "freq_Hz": round(base_freq + 0.002 * np.random.randn(), 3),
                    "equip_status": stage_info.get("equip_status", "FAULTED"),
                    "breaker_status": stage_info.get("breaker_status", "CLOSED"),
                    "relay_status": stage_info.get("relay_status", "PICKUP"),
                    "label": "lg_fault",
                    "lifecycle_stage": stage,
                }
            else:
                # --- Neighboring substations: radial sag propagation ---
                sag_factor = sag.get(int(hv_b), 1.0)

                # Read from the main power flow result (already converged)
                try:
                    v_hv_n = float(net.res_bus.at[hv_b, "vm_pu"]) * sag_factor * HV_KV
                    v_lv_n = float(net.res_bus.at[lv_b, "vm_pu"]) * sag_factor * MV_KV
                    i_hv_n = float(net.res_trafo.at[i, "i_hv_ka"]) * 1000.0
                    p_n    = float(net.res_trafo.at[i, "p_hv_mw"])
                    q_n    = float(net.res_trafo.at[i, "q_hv_mvar"])
                except Exception:
                    v_hv_n, v_lv_n = 59.5 * sag_factor, 5.45 * sag_factor
                    i_hv_n, p_n, q_n = 160.0, 4.5, 3.0

                results[sub] = {
                    "V_hv_kV": round(v_hv_n, 3),
                    "V_lv_kV": round(v_lv_n, 3),
                    "I_hv_A": round(i_hv_n, 2),
                    "P_MW": round(p_n, 3),
                    "Q_Mvar": round(q_n, 3),
                    "S_MVA": round(np.sqrt(p_n**2 + q_n**2) + 1e-6, 3),
                    "PF": round(abs(p_n) / (np.sqrt(p_n**2 + q_n**2) + 1e-6), 3),
                    "loading_pct": round(min(500.0, (i_hv_n / i_nom_a) * 100.0), 2),
                    "freq_Hz": round(base_freq + 0.002 * np.random.randn(), 3),
                    "equip_status": "HEALTHY",
                    "breaker_status": "CLOSED",
                    "relay_status": "MONITORING" if stage == "isolated" else "ALARM",
                    "label": "normal",
                    "lifecycle_stage": "normal",
                }

        return results
