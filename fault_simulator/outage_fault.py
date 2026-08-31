"""
Outage & Protection Trip Simulator for OCP Youssoufia 60 kV Network.
====================================================================
Simulates topology changes and isolation events:

1. Source Outage (source_outage):
   - Disconnects ONEE 60 kV external grid source (ext_grid.in_service = False).
   - All substations lose supply: V → 0, I → 0, P → 0, Q → 0.
   - label = 'source_outage' for ALL substations (global event).

2. Breaker Trip (breaker_trip):
   - Opens incoming feeder line to target substation.
   - ONLY the isolated substation gets label = 'breaker_trip'.
   - All other substations remain 'normal' (may see minor power-flow redistribution).

3. Transformer Trip (transformer_trip):
   - Takes target transformer out of service.
   - ONLY the tripped substation gets label = 'transformer_trip'.
   - Neighboring substations remain 'normal'.
"""

import numpy as np
import pandapower as pp

HV_KV = 60.0
MV_KV = 5.5


class OutageSimulator:
    def __init__(self):
        self._disabled_elements = []
        self.reset()

    def reset(self):
        """Clears disabled elements list for a new episode."""
        self._disabled_elements.clear()

    # ------------------------------------------------------------------
    # Source Outage
    # ------------------------------------------------------------------
    def apply_source_outage(self, net):
        """
        Takes the external grid source offline. Modifies net in place.
        Does NOT run power flow.
        """
        if len(net.ext_grid) > 0:
            grid_idx = net.ext_grid.index[0]
            if net.ext_grid.at[grid_idx, "in_service"]:
                net.ext_grid.at[grid_idx, "in_service"] = False
                self._disabled_elements.append(("ext_grid", grid_idx))

    def get_source_outage_telemetry(self, net, stage_info: dict) -> dict:
        """
        Returns telemetry for source outage (all substations lose power).
        """
        stage = stage_info.get("stage", "isolated")
        relay = stage_info.get("relay_status", "TRIPPED")
        breaker = stage_info.get("breaker_status", "OPEN")
        equip = stage_info.get("equip_status", "OFFLINE")
        freq = round(50.0 + 0.02 * np.random.randn(), 3)

        results = {}
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            results[sub] = {
                "V_hv_kV":     0.0,
                "V_lv_kV":     0.0,
                "I_hv_A":      0.0,
                "P_MW":        0.0,
                "Q_Mvar":      0.0,
                "S_MVA":       0.0,
                "PF":          0.0,
                "loading_pct": 0.0,
                "freq_Hz":     freq,
                "equip_status":   equip,
                "breaker_status": breaker,
                "relay_status":   relay,
                "label":          "source_outage",
                "lifecycle_stage": stage,
            }
        return results

    # ------------------------------------------------------------------
    # Breaker Trip (localized)
    # ------------------------------------------------------------------
    def apply_breaker_trip(self, net, target_sub_name: str) -> list:
        """
        Opens the incoming feeder line(s) to target_sub_name.
        """
        buses = net.bus.index[net.bus["name"].str.contains(target_sub_name, case=False, regex=False)]
        if len(buses) == 0:
            return []
        target_bus_idx = buses[0]

        lines_to_trip = net.line.index[
            (net.line["from_bus"] == target_bus_idx) |
            (net.line["to_bus"] == target_bus_idx)
        ]
        for l_idx in lines_to_trip:
            if net.line.at[l_idx, "in_service"]:
                net.line.at[l_idx, "in_service"] = False
                self._disabled_elements.append(("line", l_idx))

    def get_breaker_trip_telemetry(self, net, target_sub_name: str,
                                    stage_info: dict, converged: bool) -> dict:
        """
        Extracts telemetry after a breaker trip power flow has already been run.
        """
        stage   = stage_info.get("stage",          "isolated")
        relay   = stage_info.get("relay_status",   "TRIPPED")
        breaker = stage_info.get("breaker_status",  "OPEN")
        equip   = stage_info.get("equip_status",    "OFFLINE")
        freq    = round(50.0 + 0.015 * np.random.randn(), 3)

        results = {}
        for i, tname in net.trafo["name"].items():
            sub  = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            is_isolated = (sub in target_sub_name or target_sub_name in sub)

            if is_isolated:
                results[sub] = {
                    "V_hv_kV":     0.0,
                    "V_lv_kV":     0.0,
                    "I_hv_A":      0.0,
                    "P_MW":        0.0,
                    "Q_Mvar":      0.0,
                    "S_MVA":       0.0,
                    "PF":          0.0,
                    "loading_pct": 0.0,
                    "freq_Hz":     freq,
                    "equip_status":   equip,
                    "breaker_status": breaker,
                    "relay_status":   relay,
                    "label":          "breaker_trip",
                    "lifecycle_stage": stage,
                }
            else:
                rt = net.res_trafo.loc[i] if (converged and i in net.res_trafo.index) else None
                v_hv = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else 59.8
                v_lv = net.res_bus.loc[lv_b, "vm_pu"] * MV_KV if converged else 5.48
                p    = float(rt["p_hv_mw"])          if rt is not None else 4.5
                q    = float(rt["q_hv_mvar"])         if rt is not None else 2.0
                s    = np.sqrt(p**2 + q**2) + 1e-6
                pf   = abs(p) / s
                i_hv_a   = float(rt["i_hv_ka"] * 1000.0) if rt is not None else 110.0
                load_pct = float(rt["loading_percent"])   if rt is not None else 52.0

                results[sub] = {
                    "V_hv_kV":     round(v_hv, 3),
                    "V_lv_kV":     round(v_lv, 3),
                    "I_hv_A":      round(i_hv_a, 2),
                    "P_MW":        round(p, 3),
                    "Q_Mvar":      round(q, 3),
                    "S_MVA":       round(s, 3),
                    "PF":          round(min(1.0, pf), 3),
                    "loading_pct": round(load_pct, 2),
                    "freq_Hz":     freq,
                    "equip_status":   "HEALTHY",
                    "breaker_status": "CLOSED",
                    "relay_status":   "NORMAL",
                    "label":          "normal",
                    "lifecycle_stage": "normal",
                }
        return results

    # ------------------------------------------------------------------
    # Transformer Trip (localized)
    # ------------------------------------------------------------------
    def apply_transformer_trip(self, net, target_sub_name: str) -> list:
        """
        Takes target substation's transformer out of service.
        """
        trafos_to_trip = net.trafo.index[
            net.trafo["name"].str.contains(target_sub_name, case=False, regex=False)
        ]
        for t_idx in trafos_to_trip:
            if net.trafo.at[t_idx, "in_service"]:
                net.trafo.at[t_idx, "in_service"] = False
                self._disabled_elements.append(("trafo", t_idx))

    def get_transformer_trip_telemetry(self, net, target_sub_name: str,
                                        stage_info: dict, converged: bool) -> dict:
        """
        Extracts telemetry after a transformer trip power flow has already been run.
        """
        stage   = stage_info.get("stage",          "isolated")
        relay   = stage_info.get("relay_status",   "TRIPPED")
        breaker = stage_info.get("breaker_status",  "OPEN")
        equip   = stage_info.get("equip_status",    "OFFLINE")
        freq    = round(50.0 + 0.015 * np.random.randn(), 3)

        results = {}
        for i, tname in net.trafo["name"].items():
            sub  = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            is_tripped = not net.trafo.at[i, "in_service"] or \
                         (sub in target_sub_name or target_sub_name in sub)

            if is_tripped:
                v_hv = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV \
                       if (converged and hv_b in net.res_bus.index) else 60.0
                results[sub] = {
                    "V_hv_kV":     round(v_hv, 3),
                    "V_lv_kV":     0.0,
                    "I_hv_A":      0.0,
                    "P_MW":        0.0,
                    "Q_Mvar":      0.0,
                    "S_MVA":       0.0,
                    "PF":          0.0,
                    "loading_pct": 0.0,
                    "freq_Hz":     freq,
                    "equip_status":   equip,
                    "breaker_status": breaker,
                    "relay_status":   relay,
                    "label":          "transformer_trip",
                    "lifecycle_stage": stage,
                }
            else:
                rt = net.res_trafo.loc[i] if (converged and i in net.res_trafo.index) else None
                v_hv = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else 59.8
                v_lv = net.res_bus.loc[lv_b, "vm_pu"] * MV_KV if converged else 5.48
                p    = float(rt["p_hv_mw"])          if rt is not None else 5.0
                q    = float(rt["q_hv_mvar"])         if rt is not None else 2.2
                s    = np.sqrt(p**2 + q**2) + 1e-6
                pf   = abs(p) / s
                i_hv_a   = float(rt["i_hv_ka"] * 1000.0) if rt is not None else 115.0
                load_pct = float(rt["loading_percent"])   if rt is not None else 55.0

                results[sub] = {
                    "V_hv_kV":     round(v_hv, 3),
                    "V_lv_kV":     round(v_lv, 3),
                    "I_hv_A":      round(i_hv_a, 2),
                    "P_MW":        round(p, 3),
                    "Q_Mvar":      round(q, 3),
                    "S_MVA":       round(s, 3),
                    "PF":          round(min(1.0, pf), 3),
                    "loading_pct": round(load_pct, 2),
                    "freq_Hz":     freq,
                    "equip_status":   "HEALTHY",
                    "breaker_status": "CLOSED",
                    "relay_status":   "NORMAL",
                    "label":          "normal",
                    "lifecycle_stage": "normal",
                }
        return results

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------
    def simulate_source_outage(self, net):
        self.apply_source_outage(net)
        stage_info = {"stage": "isolated", "relay_status": "TRIPPED",
                      "breaker_status": "OPEN", "equip_status": "OFFLINE"}
        return self.get_source_outage_telemetry(net, stage_info)

    def simulate_breaker_trip(self, net, target_sub_name):
        self.apply_breaker_trip(net, target_sub_name)
        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False
        stage_info = {"stage": "isolated", "relay_status": "TRIPPED",
                      "breaker_status": "OPEN", "equip_status": "OFFLINE"}
        result = self.get_breaker_trip_telemetry(net, target_sub_name, stage_info, converged)
        self.restore_all(net)
        return result

    def simulate_transformer_trip(self, net, target_sub_name):
        self.apply_transformer_trip(net, target_sub_name)
        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False
        stage_info = {"stage": "isolated", "relay_status": "TRIPPED",
                      "breaker_status": "OPEN", "equip_status": "OFFLINE"}
        result = self.get_transformer_trip_telemetry(net, target_sub_name, stage_info, converged)
        self.restore_all(net)
        return result

    def restore_all(self, net):
        """Restores all elements disabled by this simulator instance."""
        for elem_type, idx in self._disabled_elements:
            if elem_type == "ext_grid" and idx in net.ext_grid.index:
                net.ext_grid.at[idx, "in_service"] = True
            elif elem_type == "line" and idx in net.line.index:
                net.line.at[idx, "in_service"] = True
            elif elem_type == "trafo" and idx in net.trafo.index:
                net.trafo.at[idx, "in_service"] = True
        self._disabled_elements.clear()