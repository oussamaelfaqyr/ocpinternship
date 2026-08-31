"""
Cached Short-Circuit Calculator & Radial Voltage Sag Propagation.
=================================================================
Eliminates redundant sc.calc_sc() calls and replaces fake bus-index-distance
voltage sag with graph-based radial propagation using real line impedances.

Physics:
  - IEC 60909 short-circuit currents (1ph / 2ph) via pandapower
  - Voltage sag propagation along the radial tree: buses electrically closer
    to the fault experience a deeper dip; buses closer to the source are
    less affected. Uses BFS from the source bus to compute cumulative
    line impedances, then linearly interpolates sag depth.
"""

import numpy as np
import pandapower.shortcircuit as sc

_sc_cache = {}
_SC_R_PREC = 2


def clear_sc_cache():
    _sc_cache.clear()


def calc_sc_cached(net, bus_idx: int, fault_type: str = "1ph",
                     r_fault_ohm: float = 0.5) -> tuple:
    """
    IEC 60909 short-circuit calculation with caching.

    Returns (ikss_ka, vm_pu_fault_bus).
    """
    r_key = round(r_fault_ohm, _SC_R_PREC)
    key = (bus_idx, fault_type, r_key)

    if key in _sc_cache:
        return _sc_cache[key]

    try:
        sc.calc_sc(net, bus=bus_idx, fault=fault_type, case="max",
                   ip=True, r_fault_ohm=r_fault_ohm)
        ikss_ka = float(net.res_bus_sc.loc[bus_idx, "ikss_ka"])

        if "vkss_kv" in net.res_bus_sc.columns:
            vkss = float(net.res_bus_sc.loc[bus_idx, "vkss_kv"])
            vn = float(net.bus.at[bus_idx, "vn_kv"])
            vm_pu = vkss / vn if vn > 0 else 0.0
        else:
            zf = max(r_fault_ohm, 0.01)
            vn = float(net.bus.at[bus_idx, "vn_kv"])
            vm_pu = min(1.0, np.sqrt(3) * ikss_ka * zf / vn)
    except Exception:
        zf = max(r_fault_ohm, 0.01)
        ikss_ka = 60.0 / (np.sqrt(3) * (zf + 5.0))
        vm_pu = zf / (zf + 5.0)

    result = (ikss_ka, min(1.0, max(0.0, vm_pu)))
    _sc_cache[key] = result

    if len(_sc_cache) > 500:
        for k in list(_sc_cache.keys())[:250]:
            del _sc_cache[k]

    return result


def compute_sag_factors(net, fault_bus_idx: int, vm_pu_fault: float) -> dict:
    """
    Compute voltage sag propagation factors for every bus via BFS from
    the external-grid source bus through the radial network.

    Returns dict: bus_idx -> sag_factor (1.0 = no sag, vm_pu_fault = deep sag).
    """
    # Source bus = where ext_grid is connected
    src = int(net.ext_grid.at[0, "bus"])
    n = len(net.bus)

    # Build adjacency with line impedances
    adj = [[] for _ in range(n)]
    for _, ln in net.line.iterrows():
        fb, tb = int(ln["from_bus"]), int(ln["to_bus"])
        z = np.sqrt(ln["r_ohm_per_km"]**2 + ln["x_ohm_per_km"]**2) * ln["length_km"]
        adj[fb].append((tb, z))
        adj[tb].append((fb, z))

    # BFS cumulative impedance from source
    cum_z = [np.inf] * n
    cum_z[src] = 0.0
    queue = [src]
    head = 0
    while head < len(queue):
        b = queue[head]; head += 1
        for nb, z in adj[b]:
            nz = cum_z[b] + z
            if nz < cum_z[nb]:
                cum_z[nb] = nz
                queue.append(nb)

    z_src_to_fault = cum_z[fault_bus_idx]
    if z_src_to_fault <= 0 or np.isinf(z_src_to_fault):
        z_src_to_fault = 1.0

    sag = {}
    for b in range(n):
        if np.isinf(cum_z[b]):
            sag[b] = 1.0
        elif b == fault_bus_idx:
            sag[b] = vm_pu_fault
        else:
            # Linear interpolation of voltage dip along the radial path
            # Closer to source -> less dip; closer to fault -> more dip
            fraction = min(1.0, cum_z[b] / z_src_to_fault)
            dip_at_fault = 1.0 - vm_pu_fault
            sag[b] = max(vm_pu_fault, 1.0 - dip_at_fault * fraction)

    return sag
