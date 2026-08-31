"""
Master Unified Physics & Protection Simulation Engine for OCP Youssoufia 60 kV.
================================================================================
Architecture:
  1. LoadProfile   – update loads, line resistances (temperature)
  2. FaultInjection– modify network parameters (topology / load / ext_grid)
  3. ONE runpp()   – single AC power flow for the combined network state
  4. ProtectionEngine – single authority for relay/trip timing
  5. Telemetry     – extract from ONE power flow result
  6. SensorEngine  – apply calibration bias and packet freeze
  7. Labels        – event_label (network-wide) + local_label (per substation)

Key properties:
  - Equipment states persist across timesteps via EquipmentStateRegistry
  - No fake fallback values when runpp() fails — timestep is invalidated
  - ProtectionSequenceTracker is the single relay/trip authority
  - All faults applied before the single power flow
  - Dual labeling: event_label and local_label in every output row
"""

import numpy as np
import pandapower as pp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from load_profile import LoadProfileEngine
from protection_engine import ProtectionSequenceTracker
from sensor_effects import SensorImperfectionEngine
from frequency_fault import FrequencySimulator, generate_freq_trajectory

from lg_fault import LGFaultSimulator
from ll_fault import LLFaultSimulator
from overload import OverloadSimulator
from voltage_sag import VoltageSagSimulator
from outage_fault import OutageSimulator

HV_KV = 60.0
MV_KV = 5.5

# Module-level singletons
_load_engine    = LoadProfileEngine()
_lg_sim         = LGFaultSimulator()
_ll_sim         = LLFaultSimulator()
_overload_sim   = OverloadSimulator()
_sag_sim        = VoltageSagSimulator()
_freq_sim       = FrequencySimulator()
_outage_sim     = OutageSimulator()

_trackers      = {}   # sub → ProtectionSequenceTracker
_TRAFO_SUBS    = None
_sensor_engine = None

# RNG for trajectory generation
_rng = np.random.default_rng()


# ===========================================================================
# Equipment State Registry — persists topology across timesteps
# ===========================================================================

class EquipmentStateRegistry:
    """
    Tracks which network elements are intentionally offline across timesteps.
    This replaces the old restore_all()-at-start-of-every-step pattern.
    """
    def __init__(self):
        self._lines_offline:   set = set()   # line indices
        self._trafos_offline:  set = set()   # trafo indices
        self._grids_offline:   set = set()   # ext_grid indices

    def take_line_offline(self, idx):
        self._lines_offline.add(idx)

    def restore_line(self, idx):
        self._lines_offline.discard(idx)

    def take_trafo_offline(self, idx):
        self._trafos_offline.add(idx)

    def restore_trafo(self, idx):
        self._trafos_offline.discard(idx)

    def take_grid_offline(self, idx):
        self._grids_offline.add(idx)

    def restore_grid(self, idx):
        self._grids_offline.discard(idx)

    def apply_to_network(self, net):
        """Enforce registry state onto the pandapower network."""
        net.line["in_service"]     = True
        net.trafo["in_service"]    = True
        net.ext_grid["in_service"] = True

        for idx in self._lines_offline:
            if idx in net.line.index:
                net.line.at[idx, "in_service"] = False

        for idx in self._trafos_offline:
            if idx in net.trafo.index:
                net.trafo.at[idx, "in_service"] = False

        for idx in self._grids_offline:
            if idx in net.ext_grid.index:
                net.ext_grid.at[idx, "in_service"] = False

    def clear_sub(self, net, sub_name: str):
        """Restore all elements associated with a given substation."""
        # Lines
        for idx in list(self._lines_offline):
            if idx in net.line.index:
                from_bus_name = net.bus.loc[net.line.at[idx, "from_bus"], "name"] \
                    if net.line.at[idx, "from_bus"] in net.bus.index else ""
                to_bus_name = net.bus.loc[net.line.at[idx, "to_bus"], "name"] \
                    if net.line.at[idx, "to_bus"] in net.bus.index else ""
                if sub_name in from_bus_name or sub_name in to_bus_name:
                    self._lines_offline.discard(idx)
        # Trafos
        for idx in list(self._trafos_offline):
            if idx in net.trafo.index:
                tname = net.trafo.at[idx, "name"]
                if sub_name in tname:
                    self._trafos_offline.discard(idx)
        # External Grids
        # Source outage is a network-wide fault, so any clear_sub clears the grid
        self._grids_offline.clear()

    def clear_all_topology(self):
        """Full network restore."""
        self._lines_offline.clear()
        self._trafos_offline.clear()
        self._grids_offline.clear()

    @property
    def is_grid_offline(self):
        return len(self._grids_offline) > 0


_registry = EquipmentStateRegistry()


# ===========================================================================
# Helper: sensor engine & tracker access
# ===========================================================================

def _get_sensor_engine(net):
    global _TRAFO_SUBS, _sensor_engine
    if _TRAFO_SUBS is None:
        _TRAFO_SUBS = [n.replace(" trafo", "") for n in net.trafo["name"]]
        _sensor_engine = SensorImperfectionEngine(_TRAFO_SUBS)
    return _sensor_engine


def _get_tracker(sub) -> ProtectionSequenceTracker:
    if sub not in _trackers:
        _trackers[sub] = ProtectionSequenceTracker()
    return _trackers[sub]


def _telemetry_from_runpp(net, converged: bool) -> dict:
    """Extract baseline telemetry from the current pandapower result."""
    freq = round(50.0 + 0.012 * np.random.randn(), 3)
    results = {}
    for i, tname in net.trafo["name"].items():
        sub  = tname.replace(" trafo", "")
        hv_b = net.trafo.at[i, "hv_bus"]
        lv_b = net.trafo.at[i, "lv_bus"]

        # If trafo is offline, res_trafo has NaN but the HV bus is still valid
        trafo_online = bool(net.trafo.at[i, "in_service"])
        rt   = net.res_trafo.loc[i] if (converged and i in net.res_trafo.index) else None

        # Read HV bus voltage (valid even for offline trafos — HV bus stays energised)
        v_hv_raw = net.res_bus.loc[hv_b, "vm_pu"] if converged else None
        v_hv     = float(v_hv_raw) * HV_KV if (v_hv_raw is not None and not np.isnan(float(v_hv_raw))) else None

        v_lv_raw = net.res_bus.loc[lv_b, "vm_pu"] if converged else None
        v_lv     = float(v_lv_raw) * MV_KV if (v_lv_raw is not None and not np.isnan(float(v_lv_raw))) else None

        # For offline trafos, secondary measurements are zero (not NaN)
        if not trafo_online:
            p        = 0.0
            q        = 0.0
            i_hv_a   = 0.0
            load_pct = 0.0
            v_lv     = 0.0  # LV bus is dead
        else:
            p        = float(rt["p_hv_mw"])         if rt is not None else None
            q        = float(rt["q_hv_mvar"])        if rt is not None else None
            i_hv_a   = float(rt["i_hv_ka"] * 1000.0) if rt is not None else None
            load_pct = float(rt["loading_percent"])  if rt is not None else None
            # Treat NaN results as invalid (pandapower can return NaN for isolated elements)
            if p is not None and np.isnan(p): p = None
            if q is not None and np.isnan(q): q = None
            if i_hv_a is not None and np.isnan(i_hv_a): i_hv_a = None
            if load_pct is not None and np.isnan(load_pct): load_pct = None
            if v_lv is None: v_lv = 0.0

        if None in (v_hv, p, q, i_hv_a, load_pct):
            results[sub] = None   # mark invalid; caller will skip
            continue

        s_mva = np.sqrt(p**2 + q**2) + 1e-6
        pf    = abs(p) / s_mva

        results[sub] = {
            "V_hv_kV":     round(max(0.0, v_hv),  3),
            "V_lv_kV":     round(max(0.0, v_lv),  3),
            "I_hv_A":      round(max(0.0, i_hv_a), 2),
            "P_MW":        round(p, 3),
            "Q_Mvar":      round(q, 3),
            "S_MVA":       round(s_mva, 3),
            "PF":          round(min(1.0, pf), 3),
            "loading_pct": round(load_pct, 2),
            "freq_Hz":     freq,
            "equip_status":    "HEALTHY",
            "breaker_status":  "CLOSED",
            "relay_status":    "NORMAL",
            "label":           "normal",
            "local_label":     "normal",
            "event_label":     "normal",
            "lifecycle_stage": "normal",
        }
    return results


# ===========================================================================
# Main simulation step
# ===========================================================================

def simulate_step_per_sub(net, current_step: int, sub_fault_map: dict) -> dict:
    """
    Simulates one timestep where each substation may have a different fault.

    Parameters
    ----------
    net           : pandapower network (reused across calls)
    current_step  : integer timestep index
    sub_fault_map : dict mapping substation_name → (fault_type, params_dict)

    Returns
    -------
    dict mapping substation_name → telemetry_row, or empty dict if runpp failed.
    Telemetry rows include both 'local_label' and 'event_label'.
    """
    t_sec = float(current_step)
    sensor_engine = _get_sensor_engine(net)

    # ---------------------------------------------------------------
    # 1. Update loads and line resistances (temperature-dependent)
    # ---------------------------------------------------------------
    _load_engine.update_network_loads(net, t_sec)

    # ---------------------------------------------------------------
    # 2. Apply persisted topology state from registry
    # ---------------------------------------------------------------
    _registry.apply_to_network(net)

    # ---------------------------------------------------------------
    # 3. Process protection trackers and inject fault modifications
    # ---------------------------------------------------------------
    subs = list(_TRAFO_SUBS) if _TRAFO_SUBS else []

    # Determine the network-wide event label (for rows that have no local fault)
    active_network_fault = "normal"    # global event if any
    active_network_stage_info = None

    # Collect info about what each sub is doing
    sub_actions = {}   # sub → (ftype, params, stage_info)

    for sub in subs:
        ftype, params = sub_fault_map.get(sub, ("normal", {}))
        tracker = _get_tracker(sub)

        if ftype == "normal":
            if tracker.active_fault is not None and tracker.active_fault != "normal":
                # Fault was cleared externally or expired — restore topology and clear tracker
                _registry.clear_sub(net, sub)
                tracker.clear_fault()
            stage_info = tracker.get_lifecycle_stage(current_step)

        elif tracker.active_fault != ftype:
            # New fault — trigger lifecycle
            r_ohm = params.get("r_fault_ohm", float(np.random.uniform(0.2, 12.0)))
            overload_factor = params.get("overload_factor", 1.55)
            tracker.trigger_fault(ftype, sub, current_step,
                                  duration_steps=params.get("duration_steps", 60),
                                  r_ohm=r_ohm,
                                  overload_factor=overload_factor)

            # For frequency events, pre-compute the full trajectory
            if ftype in ["under_frequency", "over_frequency"]:
                mode = "under" if ftype == "under_frequency" else "over"
                duration = params.get("duration_steps", 60)
                traj = generate_freq_trajectory(mode, duration, rng=_rng)
                tracker.set_freq_trajectory(traj)

            stage_info = tracker.get_lifecycle_stage(current_step)
        else:
            stage_info = tracker.get_lifecycle_stage(current_step)

        # Track network-wide events
        if ftype in ["voltage_sag", "under_frequency", "over_frequency", "source_outage"]:
            active_network_fault = ftype
            active_network_stage_info = stage_info

        sub_actions[sub] = (ftype, params, stage_info, tracker)

    # ---------------------------------------------------------------
    # 4. Apply ALL fault modifications to net BEFORE running runpp
    # ---------------------------------------------------------------
    applied_overloads = {}   # sub → (target_load, elapsed_s, stage_info)
    applied_source_outage = False

    for sub, (ftype, params, stage_info, tracker) in sub_actions.items():
        stage = stage_info["stage"]

        if ftype == "normal" or stage in ["restored", "normal"]:
            continue

        if ftype == "overload":
            elapsed_s = max(0, current_step - tracker.fault_start_step)
            target_load = f"{sub} load"
            if stage in ["isolated", "breaker_open"]:
                # Physically isolate the transformer in pandapower
                _overload_sim.isolate_in_pandapower(net, sub)
                # Register it in the persistent registry
                for i, tname in net.trafo["name"].items():
                    if sub in tname.replace(" trafo", "") or tname.replace(" trafo", "") in sub:
                        _registry.take_trafo_offline(i)
            else:
                _overload_sim.apply_fault(net, target_load,
                                          current_step_s=elapsed_s,
                                          overload_factor=params.get("overload_factor", 1.55))
            applied_overloads[sub] = (target_load, elapsed_s, stage_info)

        elif ftype == "source_outage":
            if not applied_source_outage and not _registry.is_grid_offline:
                if stage in ["relay_operated", "breaker_open", "isolated"]:
                    _registry.take_grid_offline(net.ext_grid.index[0])
                    _registry.apply_to_network(net)   # apply immediately
                    applied_source_outage = True

        elif ftype == "breaker_trip":
            if stage in ["relay_operated", "breaker_open", "isolated"]:
                # Apply and register if not already done
                for i, tname in net.trafo["name"].items():
                    s = tname.replace(" trafo", "")
                    if (sub in s or s in sub) and i not in _registry._trafos_offline:
                        # No trafo offline for breaker trip — lines are tripped
                        pass
                buses = net.bus.index[net.bus["name"].str.contains(sub, case=False, regex=False)]
                if len(buses) > 0:
                    target_bus = buses[0]
                    lines = net.line.index[
                        (net.line["from_bus"] == target_bus) |
                        (net.line["to_bus"] == target_bus)
                    ]
                    for l_idx in lines:
                        _registry.take_line_offline(l_idx)
                _registry.apply_to_network(net)

        elif ftype == "transformer_trip":
            if stage in ["relay_operated", "breaker_open", "isolated"]:
                for i, tname in net.trafo["name"].items():
                    s = tname.replace(" trafo", "")
                    if (sub in s or s in sub):
                        _registry.take_trafo_offline(i)
                _registry.apply_to_network(net)

        elif ftype == "voltage_sag":
            sag_depth = params.get("sag_depth_pu", 0.75)
            _sag_sim.apply_fault(net, sag_depth_pu=sag_depth)
            # voltage_sag clears after runpp

    # ---------------------------------------------------------------
    # 5. Run ONE AC power flow for the combined network state
    # ---------------------------------------------------------------
    try:
        pp.runpp(net, numba=False)
        converged = True
    except Exception:
        converged = False

    # Clear voltage sag modification after power flow
    if any(v[0] == "voltage_sag" for v in sub_actions.values()):
        _sag_sim.clear_fault(net)

    if not converged and active_network_fault != "source_outage":
        # Don't fabricate measurements — return empty dict to signal invalid step
        return {}

    # ---------------------------------------------------------------
    # 6. Extract base telemetry from the single power flow
    # ---------------------------------------------------------------
    base_telem = _telemetry_from_runpp(net, converged)

    # ---------------------------------------------------------------
    # 7. Apply per-substation protection state and fault-specific
    #    telemetry overlays (relays, labels, voltage corrections)
    # ---------------------------------------------------------------
    final_raw = {}

    for sub in subs:
        ftype, params, stage_info, tracker = sub_actions.get(sub, ("normal", {}, {"stage": "normal"}, _get_tracker(sub)))
        stage = stage_info["stage"]

        if stage in ["restored"] or ftype == "normal":
            ftype = "normal"
            stage_info = {"stage": "normal", "relay_status": "NORMAL",
                          "breaker_status": "CLOSED", "equip_status": "HEALTHY"}

        base = base_telem.get(sub)

        # Source outage — all subs get zero measurements
        if active_network_fault == "source_outage" and active_network_stage_info and \
           active_network_stage_info["stage"] not in ["pre_fault", "normal"]:
            freq = round(50.0 + 0.02 * np.random.randn(), 3)
            row = {
                "V_hv_kV": 0.0, "V_lv_kV": 0.0, "I_hv_A": 0.0,
                "P_MW": 0.0, "Q_Mvar": 0.0, "S_MVA": 0.0,
                "PF": 0.0, "loading_pct": 0.0, "freq_Hz": freq,
                "equip_status": "OFFLINE", "breaker_status": "OPEN",
                "relay_status": "TRIPPED",
                "local_label": "source_outage",
                "event_label": "source_outage",
                "label": "source_outage",
                "lifecycle_stage": active_network_stage_info["stage"],
            }
            final_raw[sub] = row
            continue

        if base is None:
            continue   # invalid step for this substation — skip

        # Start from base flow result
        row = dict(base)

        if ftype == "normal":
            row["local_label"] = "normal"
            row["event_label"] = active_network_fault if active_network_fault != "normal" else "normal"
            row["label"]       = "normal"

        elif ftype == "overload":
            if sub in applied_overloads:
                target_load, elapsed_s, si = applied_overloads[sub]
                # Re-apply protection state from tracker
                row.update({
                    "equip_status":   si.get("equip_status", "OVERLOADED"),
                    "relay_status":   si.get("relay_status", "ALARM"),
                    "breaker_status": si.get("breaker_status", "CLOSED"),
                    "lifecycle_stage": stage,
                })
                if stage in ["isolated", "breaker_open"]:
                    row.update({"V_lv_kV": 0.0, "I_hv_A": 0.0,
                                "P_MW": 0.0, "Q_Mvar": 0.0,
                                "S_MVA": 0.0, "PF": 0.0, "loading_pct": 0.0})
                else:
                    # Apply LV voltage drop
                    lv_drop = max(0.92, 1.0 - 0.0005 * max(0, row["loading_pct"] - 100.0))
                    row["V_lv_kV"] = round(row["V_lv_kV"] * lv_drop, 3)
            row["local_label"] = "overload"
            row["event_label"] = "overload"
            row["label"]       = "overload"

        elif ftype == "lg_fault":
            r_ohm = tracker.fault_r_ohm
            lg_raw = _lg_sim.apply_fault(net, sub, stage_info, r_fault_ohm=r_ohm)
            if sub in lg_raw:
                row = dict(lg_raw[sub])
            row["local_label"] = "lg_fault"
            row["event_label"] = "lg_fault"
            row["label"]       = "lg_fault"

        elif ftype == "ll_fault":
            r_ohm = tracker.fault_r_ohm
            ll_raw = _ll_sim.apply_fault(net, sub, stage_info, r_fault_ohm=r_ohm)
            if sub in ll_raw:
                row = dict(ll_raw[sub])
            row["local_label"] = "ll_fault"
            row["event_label"] = "ll_fault"
            row["label"]       = "ll_fault"

        elif ftype == "voltage_sag":
            # All subs get voltage_sag label (global event)
            row.update({
                "equip_status":   stage_info.get("equip_status", "HEALTHY"),
                "relay_status":   stage_info.get("relay_status", "MONITORING"),
                "breaker_status": stage_info.get("breaker_status", "CLOSED"),
                "lifecycle_stage": stage,
            })
            row["local_label"] = "voltage_sag"
            row["event_label"] = "voltage_sag"
            row["label"]       = "voltage_sag"

        elif ftype in ["under_frequency", "over_frequency"]:
            freq_val = tracker.get_freq_value()
            row["freq_Hz"] = round(freq_val, 3)
            row.update({
                "equip_status":   stage_info.get("equip_status", "HEALTHY"),
                "relay_status":   stage_info.get("relay_status", "ALARM_81U"),
                "breaker_status": stage_info.get("breaker_status", "CLOSED"),
                "lifecycle_stage": stage,
            })
            row["local_label"] = ftype
            row["event_label"] = ftype
            row["label"]       = ftype

        elif ftype == "breaker_trip":
            is_isolated = (sub == sub)   # always True here
            row.update({
                "V_lv_kV": 0.0, "I_hv_A": 0.0,
                "P_MW": 0.0, "Q_Mvar": 0.0,
                "S_MVA": 0.0, "PF": 0.0, "loading_pct": 0.0,
                "equip_status":   stage_info.get("equip_status", "OFFLINE"),
                "relay_status":   stage_info.get("relay_status", "TRIPPED"),
                "breaker_status": stage_info.get("breaker_status", "OPEN"),
                "lifecycle_stage": stage,
            })
            row["local_label"] = "breaker_trip"
            row["event_label"] = "breaker_trip"
            row["label"]       = "breaker_trip"

        elif ftype == "transformer_trip":
            row.update({
                "V_lv_kV": 0.0, "I_hv_A": 0.0,
                "P_MW": 0.0, "Q_Mvar": 0.0,
                "S_MVA": 0.0, "PF": 0.0, "loading_pct": 0.0,
                "equip_status":   stage_info.get("equip_status", "OFFLINE"),
                "relay_status":   stage_info.get("relay_status", "TRIPPED"),
                "breaker_status": stage_info.get("breaker_status", "OPEN"),
                "lifecycle_stage": stage,
            })
            row["local_label"] = "transformer_trip"
            row["event_label"] = "transformer_trip"
            row["label"]       = "transformer_trip"

        # Ensure S and PF are always derived from P and Q (physical consistency)
        p = row.get("P_MW", 0.0)
        q = row.get("Q_Mvar", 0.0)
        s = np.sqrt(p**2 + q**2) + 1e-9
        row["S_MVA"] = round(s, 3)
        row["PF"]    = round(min(1.0, abs(p) / s), 3)

        final_raw[sub] = row

    # ---------------------------------------------------------------
    # 8. Apply SCADA sensor imperfections
    # ---------------------------------------------------------------
    final = {}
    for sub, row in final_raw.items():
        final[sub] = sensor_engine.apply_sensor_imperfections(sub, row)

    # Clear overload fault modification from network (restore loads to baseline)
    if applied_overloads:
        for sub, (target_load, elapsed_s, si) in applied_overloads.items():
            stage = si["stage"]
            if stage not in ["isolated", "breaker_open"]:
                _overload_sim.clear_fault(net)

    return final


# ===========================================================================
# Backward compatibility wrappers
# ===========================================================================

def simulate_step(net, current_step: int, active_fault: str = "normal",
                  target_sub: str = "Mine Mzinda DIS TR",
                  r_fault_ohm: float = None) -> dict:
    if _TRAFO_SUBS is None:
        _get_sensor_engine(net)
    sub_fault_map = {}
    is_grid_wide = active_fault in ["voltage_sag", "under_frequency", "over_frequency", "source_outage"]
    for sub in _TRAFO_SUBS:
        if is_grid_wide or sub == target_sub:
            params = {}
            if r_fault_ohm:
                params["r_fault_ohm"] = r_fault_ohm
            sub_fault_map[sub] = (active_fault, params)
        else:
            sub_fault_map[sub] = ("normal", {})
    return simulate_step_per_sub(net, current_step, sub_fault_map)


def simulate_normal(net):
    return simulate_step(net, 0, "normal")

def simulate_LG_fault(net, target_sub_name="Mine Mzinda DIS TR"):
    return simulate_step(net, 3, "lg_fault", target_sub_name)

def simulate_LL_fault(net, target_sub_name="Laverie/Sechage SN1"):
    return simulate_step(net, 3, "ll_fault", target_sub_name)

def simulate_voltage_sag(net, sag_depth_pu=0.75):
    if _TRAFO_SUBS is None:
        _get_sensor_engine(net)
    sub_fault_map = {sub: ("voltage_sag", {"sag_depth_pu": sag_depth_pu})
                     for sub in _TRAFO_SUBS}
    return simulate_step_per_sub(net, 0, sub_fault_map)

def simulate_overload(net, target_sub_name="Mine Mzinda DIS TR",
                      overload_factor=1.55, current_step=3):
    return simulate_step(net, current_step, "overload", target_sub_name)

def simulate_under_frequency(net, freq_deviation_hz=0.5):
    if _TRAFO_SUBS is None:
        _get_sensor_engine(net)
    sub_fault_map = {sub: ("under_frequency", {"freq_deviation_hz": freq_deviation_hz})
                     for sub in _TRAFO_SUBS}
    return simulate_step_per_sub(net, 0, sub_fault_map)

def simulate_over_frequency(net, freq_deviation_hz=0.5):
    if _TRAFO_SUBS is None:
        _get_sensor_engine(net)
    sub_fault_map = {sub: ("over_frequency", {"freq_deviation_hz": freq_deviation_hz})
                     for sub in _TRAFO_SUBS}
    return simulate_step_per_sub(net, 0, sub_fault_map)

def simulate_source_outage(net):
    if _TRAFO_SUBS is None:
        _get_sensor_engine(net)
    sub_fault_map = {sub: ("source_outage", {}) for sub in _TRAFO_SUBS}
    return simulate_step_per_sub(net, 5, sub_fault_map)

def simulate_breaker_trip(net, target_sub_name="Mine Mzinda DIS TR"):
    return simulate_step(net, 5, "breaker_trip", target_sub_name)

def simulate_transformer_trip(net, target_sub_name="Mine Mzinda DIS TR"):
    return simulate_step(net, 5, "transformer_trip", target_sub_name)
