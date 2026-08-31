"""
Protection Sequence Lifecycle & Coordination Engine for OCP 60 kV Network.
==========================================================================
Manages multi-stage fault lifecycles with per-fault-type timing parameters:

  normal -> pre_fault -> fault_detected -> relay_operated -> breaker_open -> isolated -> restored

Key design principle: this is the SINGLE authority for relay/trip decisions.
Individual fault simulators describe PHYSICS only. The tracker decides PROTECTION.
"""

import numpy as np


class ProtectionSequenceTracker:
    def __init__(self):
        self.active_fault = None
        self.target_sub = None
        self.fault_start_step = 0
        self.fault_duration_steps = 15
        self.fault_r_ohm = 0.5
        self.is_backup_trip = False

        # Per-fault timing (steps, at dt=1s each step ~ 1 second)
        # Overload uses severity-dependent timing; stored as callable or fixed value
        self._pre_fault_steps = 2        # steps in pre_fault stage
        self._detected_steps  = 2        # steps in fault_detected
        self._operated_steps  = 2        # steps in relay_operated
        self._opening_steps   = 2        # steps in breaker_open (contacts separating)
        # Total to breaker open = pre_fault + detected + operated + opening = 8 steps
        # remaining steps up to duration = isolated

        # For severity-dependent overload timing:
        self._overload_factor = 1.0       # set by trigger_fault for inverse-time curve
        self._thermal_delay_s = 30        # default; overwritten for overload

        # For frequency events: continuous state
        self._freq_trajectory = None      # list of freq_hz values
        self._freq_step_idx   = 0

    # ------------------------------------------------------------------
    # Inverse-time relay curve (IEC standard inverse)
    # trip_time(s) ≈ TDS * k / ((I/Ipickup)^α - 1)
    # Simplified for thermal overload: trip_time = base / (factor - 1.0)^0.8
    # ------------------------------------------------------------------
    @staticmethod
    def _inverse_time_delay(overload_factor: float) -> int:
        """
        Returns trip delay in steps for a given overload factor.
        105% → very long (120 steps), 120% → ~60 steps, 150% → ~30 steps,
        200%+ → ~10 steps.
        """
        if overload_factor <= 1.05:
            return 999   # effectively no trip below 5% overload
        excess = overload_factor - 1.0
        delay_s = max(8, int(28.0 / (excess ** 0.9)))
        return delay_s  # each step = 1 second

    def trigger_fault(self, ftype: str, target_sub: str, start_step: int,
                      duration_steps: int = 60, r_ohm: float = None,
                      overload_factor: float = 1.0):
        """Starts a new multi-stage fault lifecycle."""
        self.active_fault = ftype
        self.target_sub = target_sub
        self.fault_start_step = start_step
        self.fault_duration_steps = duration_steps
        self.fault_r_ohm = r_ohm if r_ohm is not None else float(np.random.uniform(0.2, 12.0))
        # 5% chance of primary breaker failure triggering backup SSP trip
        self.is_backup_trip = (np.random.rand() < 0.05) if ftype in ["lg_fault", "ll_fault"] else False

        # Set per-fault-type timing
        if ftype == "overload":
            self._overload_factor = overload_factor
            self._thermal_delay_s = self._inverse_time_delay(overload_factor)
            # For overload: pre_fault ramp = 10 steps, then ALARM until thermal limit
            self._pre_fault_steps = 10
        elif ftype in ["breaker_trip", "transformer_trip", "source_outage"]:
            # Topology events have no pre-fault ramp — immediate trip
            self._pre_fault_steps = 1
        else:
            # LG/LL fault: short pre-fault arc inception
            self._pre_fault_steps = 2

        # Reset frequency trajectory for frequency events
        self._freq_trajectory = None
        self._freq_step_idx   = 0

    def set_freq_trajectory(self, trajectory: list):
        """Sets a pre-computed frequency trajectory for under/over frequency events."""
        self._freq_trajectory = trajectory
        self._freq_step_idx   = 0

    def get_freq_value(self) -> float:
        """Returns the current frequency value and advances the trajectory index."""
        if self._freq_trajectory is None:
            return 50.0 + 0.012 * np.random.randn()
        if self._freq_step_idx < len(self._freq_trajectory):
            val = self._freq_trajectory[self._freq_step_idx]
            self._freq_step_idx += 1
            return float(val)
        return self._freq_trajectory[-1]

    def clear_fault(self):
        """Restores normal operation."""
        self.active_fault = None
        self.target_sub = None
        self._freq_trajectory = None
        self._freq_step_idx   = 0

    def get_lifecycle_stage(self, current_step: int) -> dict:
        """
        Determines current protection stage based on elapsed steps.

        For short-circuit faults (lg_fault, ll_fault):
          0..1  steps: pre_fault  (arc inception)
          2..3  steps: fault_detected (relay pickup 50/51)
          4..5  steps: relay_operated (trip signal dispatched)
          6..7  steps: breaker_open  (contacts separating)
          8+    steps: isolated

        For overload:
          0..pre_fault_steps-1: pre_fault (load ramp)
          pre_fault_steps .. thermal_delay: overloaded / ALARM
          thermal_delay .. thermal_delay+5: relay_operated → breaker_open
          thermal_delay+5+: isolated

        For breaker/transformer/source trips:
          0: breaker_open immediately
          1+: isolated
        """
        if not self.active_fault or self.active_fault == "normal":
            return {
                "stage": "normal",
                "relay_status": "NORMAL",
                "breaker_status": "CLOSED",
                "equip_status": "HEALTHY",
            }

        elapsed = current_step - self.fault_start_step

        if elapsed >= self.fault_duration_steps:
            return {
                "stage": "restored",
                "relay_status": "RESET",
                "breaker_status": "CLOSED",
                "equip_status": "HEALTHY",
            }

        ftype = self.active_fault

        # --- Topology trips (instantaneous) ---
        if ftype in ["breaker_trip", "transformer_trip", "source_outage"]:
            if elapsed < 1:
                return {"stage": "relay_operated", "relay_status": "PICKUP",   "breaker_status": "TRIP_COMMAND", "equip_status": "FAULTED"}
            elif elapsed < 2:
                return {"stage": "breaker_open",   "relay_status": "TRIPPED",  "breaker_status": "OPENING",      "equip_status": "FAULTED"}
            else:
                return {"stage": "isolated",        "relay_status": "TRIPPED",  "breaker_status": "OPEN",         "equip_status": "OFFLINE"}

        # --- Overload (inverse-time) ---
        if ftype == "overload":
            pf = self._pre_fault_steps
            td = self._thermal_delay_s
            if elapsed < pf:
                return {"stage": "pre_fault",      "relay_status": "MONITORING", "breaker_status": "CLOSED", "equip_status": "DEGRADED"}
            elif elapsed < pf + td:
                return {"stage": "fault_detected", "relay_status": "ALARM",      "breaker_status": "CLOSED", "equip_status": "OVERLOADED"}
            elif elapsed < pf + td + 3:
                return {"stage": "relay_operated", "relay_status": "OPERATED",   "breaker_status": "TRIP_COMMAND", "equip_status": "OVERLOADED"}
            elif elapsed < pf + td + 5:
                return {"stage": "breaker_open",   "relay_status": "TRIPPED",    "breaker_status": "OPENING", "equip_status": "FAULTED"}
            else:
                return {"stage": "isolated",        "relay_status": "TRIPPED",    "breaker_status": "OPEN",    "equip_status": "OFFLINE"}

        # --- Frequency events ---
        if ftype in ["under_frequency", "over_frequency"]:
            # Use a slow staged relay response
            if elapsed < 3:
                return {"stage": "pre_fault",      "relay_status": "MONITORING", "breaker_status": "CLOSED", "equip_status": "HEALTHY"}
            elif elapsed < 8:
                return {"stage": "fault_detected", "relay_status": "ALARM_81U" if ftype == "under_frequency" else "ALARM_81O",
                        "breaker_status": "CLOSED", "equip_status": "HEALTHY"}
            elif elapsed < 12:
                return {"stage": "relay_operated", "relay_status": "PICKUP",   "breaker_status": "CLOSED", "equip_status": "HEALTHY"}
            else:
                return {"stage": "isolated",        "relay_status": "TRIPPED",  "breaker_status": "OPEN",   "equip_status": "OFFLINE"}

        # --- Short-circuit faults (lg, ll) and voltage_sag ---
        if elapsed <= 1:
            return {"stage": "pre_fault",      "relay_status": "MONITORING",  "breaker_status": "CLOSED",       "equip_status": "DEGRADED"}
        elif elapsed <= 3:
            return {"stage": "fault_detected", "relay_status": "PICKUP",      "breaker_status": "CLOSED",       "equip_status": "FAULTED"}
        elif elapsed <= 5:
            return {"stage": "relay_operated", "relay_status": "TRIPPED",     "breaker_status": "TRIP_COMMAND", "equip_status": "FAULTED"}
        elif elapsed <= 7:
            return {"stage": "breaker_open",   "relay_status": "TRIPPED",     "breaker_status": "OPENING",      "equip_status": "FAULTED"}
        else:
            return {"stage": "isolated",        "relay_status": "TRIPPED",     "breaker_status": "OPEN",         "equip_status": "OFFLINE"}
