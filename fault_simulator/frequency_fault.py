"""
Frequency Event Simulator for OCP Youssoufia 60 kV Network.
============================================================
Simulates grid frequency excursions resulting from national generation-load imbalance.
Generates STATEFUL, TEMPORALLY COHERENT frequency trajectories — not independent random
values per timestep.

Under-Frequency (49.0–49.5 Hz):
  - Caused by sudden loss of major power generation or tie-line trip in ONEE grid.
  - Relay logic: f < 49.5 → ALARM (81U), f < 49.0 sustained → PICKUP → TRIP.

Over-Frequency (50.5–51.0 Hz):
  - Caused by sudden load rejection or excess renewable generation.
  - Relay logic: f > 50.5 → ALARM (81O), f > 51.0 sustained → PICKUP → TRIP.
"""

import numpy as np
import pandapower as pp

HV_KV = 60.0
MV_KV = 5.5


def generate_freq_trajectory(mode: str, n_steps: int, rng=None,
                             immediate_start: bool = False) -> list:
    """
    Generates a physically realistic frequency trajectory for an entire fault episode.

    The trajectory follows:
      normal (50 Hz) → smooth ramp to excursion → sustained excursion → smooth recovery

    Parameters
    ----------
    mode : 'under' or 'over'
    n_steps : total number of timesteps in the episode
    rng : numpy random Generator (optional)
    immediate_start : if True, skip the pre-fault normal phase and start
                      immediately at nadir (useful for server-injected faults
                      so the model window fills with anomalous data quickly).

    Returns
    -------
    List of freq_Hz values of length n_steps
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Phase durations (each step = 1 second) ---
    ramp_down_steps  = max(3, n_steps // 6)     # ~5-10s smooth ramp toward nadir
    sustained_steps  = max(5, n_steps // 3)     # sustained excursion
    ramp_up_steps    = max(3, n_steps // 6)     # recovery
    # remaining steps: pre-fault normal and post-recovery normal

    # --- Nadir frequency (the worst point of the excursion) ---
    if mode == "under":
        nadir = rng.uniform(48.5, 49.3)        # 48.5–49.3 Hz nadir
    else:
        nadir = rng.uniform(50.7, 51.2)        # 50.7–51.2 Hz nadir

    trajectory = []

    if immediate_start:
        # Skip pre-fault normal — start directly at nadir for fast model detection
        pre_normal_steps = 0
        post_normal_steps = max(0, n_steps - ramp_down_steps - sustained_steps - ramp_up_steps)
    else:
        pre_normal_steps = max(2, n_steps - ramp_down_steps - sustained_steps - ramp_up_steps - 2)
        post_normal_steps = n_steps - pre_normal_steps - ramp_down_steps - sustained_steps - ramp_up_steps

    # 1. Pre-event normal
    for _ in range(pre_normal_steps):
        trajectory.append(50.0 + 0.012 * rng.standard_normal())

    # 2. Smooth ramp toward nadir (cosine transition) — shortened when immediate
    actual_ramp = 1 if immediate_start else ramp_down_steps
    for k in range(actual_ramp):
        frac = (k + 1) / actual_ramp
        smooth = 0.5 * (1 - np.cos(np.pi * frac))
        f = 50.0 + (nadir - 50.0) * smooth
        trajectory.append(f + 0.015 * rng.standard_normal())

    # 3. Sustained excursion at nadir (with small noise) — extend to fill most of n_steps
    actual_sustained = n_steps - pre_normal_steps - actual_ramp - ramp_up_steps
    actual_sustained = max(5, actual_sustained)
    for _ in range(actual_sustained):
        trajectory.append(nadir + 0.020 * rng.standard_normal())

    # 4. Smooth recovery back to 50 Hz
    for k in range(ramp_up_steps):
        frac = (k + 1) / ramp_up_steps
        smooth = 0.5 * (1 - np.cos(np.pi * frac))
        f = nadir + (50.0 - nadir) * smooth
        trajectory.append(f + 0.015 * rng.standard_normal())

    # 5. Post-recovery normal
    for _ in range(max(0, post_normal_steps)):
        trajectory.append(50.0 + 0.012 * rng.standard_normal())

    return trajectory[:n_steps]  # trim to exact length


class FrequencySimulator:
    def __init__(self, mode="under", target_freq_hz=49.0):
        """
        :param mode: 'under' or 'over'.
        :param target_freq_hz: Default frequency level.
        """
        self.mode = mode
        self.target_freq_hz = target_freq_hz

    def get_telemetry_override(self, net, mode=None, freq_hz=None,
                               stage_info: dict = None, relay_status: str = None):
        """
        Executes AC power flow and applies frequency shift across all substation SCADA sensors.

        Parameters
        ----------
        net        : pandapower network
        mode       : 'under' or 'over'
        freq_hz    : specific frequency value for this timestep (from trajectory)
        stage_info : lifecycle stage dict from ProtectionSequenceTracker
        relay_status: override relay status string
        """
        evt_mode = mode if mode is not None else self.mode

        # If no trajectory value supplied, draw a consistent per-step value
        if freq_hz is None:
            if evt_mode == "under":
                freq_hz = round(49.0 + 0.3 * np.random.randn(), 3)
            else:
                freq_hz = round(51.0 + 0.3 * np.random.randn(), 3)

        freq_hz = round(float(freq_hz), 3)

        # --- Threshold-based relay state ---
        if relay_status is None:
            if evt_mode == "under":
                if freq_hz <= 49.0:
                    relay_status = "PICKUP_81U"
                elif freq_hz <= 49.5:
                    relay_status = "ALARM_81U"
                else:
                    relay_status = "MONITORING"
            else:
                if freq_hz >= 51.0:
                    relay_status = "PICKUP_81O"
                elif freq_hz >= 50.5:
                    relay_status = "ALARM_81O"
                else:
                    relay_status = "MONITORING"

        stage = stage_info.get("stage", "fault_detected") if stage_info else "fault_detected"
        if stage in ["isolated", "breaker_open"]:
            relay_status = "TRIPPED"

        try:
            pp.runpp(net, numba=False)
            converged = True
        except Exception:
            converged = False

        label_name = "under_frequency" if evt_mode == "under" else "over_frequency"

        results = {}
        for i, tname in net.trafo["name"].items():
            sub = tname.replace(" trafo", "")
            hv_b = net.trafo.at[i, "hv_bus"]
            lv_b = net.trafo.at[i, "lv_bus"]

            rt = net.res_trafo.loc[i] if (converged and i in net.res_trafo.index) else None

            v_hv      = net.res_bus.loc[hv_b, "vm_pu"] * HV_KV if converged else 59.8
            v_lv      = net.res_bus.loc[lv_b, "vm_pu"] * MV_KV if converged else 5.48
            p         = float(rt["p_hv_mw"])         if rt is not None else 5.0
            q         = float(rt["q_hv_mvar"])        if rt is not None else 2.2
            s_mva     = np.sqrt(p**2 + q**2) + 1e-6
            pf        = abs(p) / s_mva
            i_hv_a    = float(rt["i_hv_ka"] * 1000.0) if rt is not None else 115.0
            loading_pct = float(rt["loading_percent"]) if rt is not None else 55.0

            results[sub] = {
                "V_hv_kV":     round(v_hv, 3),
                "V_lv_kV":     round(v_lv, 3),
                "I_hv_A":      round(i_hv_a, 2),
                "P_MW":        round(p, 3),
                "Q_Mvar":      round(q, 3),
                "S_MVA":       round(s_mva, 3),
                "PF":          round(min(1.0, pf), 3),
                "loading_pct": round(loading_pct, 2),
                "freq_Hz":     freq_hz,
                "equip_status":   stage_info.get("equip_status", "HEALTHY") if stage_info else "HEALTHY",
                "breaker_status": stage_info.get("breaker_status", "CLOSED") if stage_info else "CLOSED",
                "relay_status":   relay_status,
                "label":          label_name,
                "lifecycle_stage": stage,
            }

        return results
