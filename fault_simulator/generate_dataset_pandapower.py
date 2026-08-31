"""
Youssoufia digital twin -- pandapower time-series dataset generator.

Runs a full AC power flow at every timestep (quasi-steady-state SCADA
snapshot, 1 sample/second) across the whole 60 kV backbone built in
build_network.py, with scheduled fault/anomaly events injected.

KEY FIX: Faults are now scheduled INDEPENDENTLY per substation.
"""

import numpy as np
import pandas as pd
import pandapower as pp
from build_network import build_network, MV_KV, HV_KV

np.random.seed(42)

def get_trafo_names(net):
    return [n.replace(" trafo", "") for n in net.trafo["name"]]


def snapshot(net, t, sub_labels, freq):
    """Pull one row of measurements per transformer from a converged net."""
    rows = []
    for i, name in net.trafo["name"].items():
        base = name.replace(" trafo", "")
        r = net.res_trafo.loc[i]
        v_hv = net.res_bus.loc[net.trafo.at[i, "hv_bus"], "vm_pu"] * HV_KV
        v_lv = net.res_bus.loc[net.trafo.at[i, "lv_bus"], "vm_pu"] * MV_KV
        p = r["p_hv_mw"]
        q = r["q_hv_mvar"]
        s = np.sqrt(p**2 + q**2) + 1e-9
        pf = abs(p) / s
        i_hv_a = r["i_hv_ka"] * 1000
        label = sub_labels.get(base, "normal")
        rows.append({
            "time_s": t, "substation": base,
            "V_hv_kV": v_hv, "V_lv_kV": v_lv,
            "I_hv_A": i_hv_a, "P_MW": p, "Q_Mvar": q, "PF": pf,
            "loading_pct": r["loading_percent"],
            "freq_Hz": freq, "label": label,
        })
    return rows


def generate_schedules(subs, total_time):
    """Generate independent fault schedules for each substation."""
    schedules = {sub: [] for sub in subs}
    fault_types = ["overload", "line_fault"]
    global_faults = ["voltage_sag", "source_outage"]
    
    # Global faults affecting all subs at the same time
    schedules["__global__"] = []
    t = 100
    while t < total_time - 100:
        if np.random.rand() < 0.05: # 5% chance every ~500s
            dur = int(np.random.uniform(10, 30))
            ftype = np.random.choice(global_faults)
            schedules["__global__"].append((ftype, t, dur))
            t += dur + 500
        else:
            t += 500

    # Localized faults
    for sub in subs:
        t = 50
        while t < total_time - 100:
            if np.random.rand() < 0.15: # 15% chance every ~200s
                dur = int(np.random.uniform(15, 60))
                ftype = np.random.choice(fault_types)
                schedules[sub].append((ftype, t, dur))
                t += dur + int(np.random.uniform(200, 500))
            else:
                t += 200
                
    return schedules

def get_active_faults(schedules, t):
    active = {}
    
    # Check global faults
    g_fault = None
    for (ftype, st, dur) in schedules.get("__global__", []):
        if st <= t < st + dur:
            g_fault = ftype
            break
            
    # Check local faults
    for sub, events in schedules.items():
        if sub == "__global__": continue
        active[sub] = "normal"
        if g_fault:
            active[sub] = g_fault
            continue
            
        for (ftype, st, dur) in events:
            if st <= t < st + dur:
                active[sub] = ftype
                break
                
    return active, g_fault

def run():
    net = build_network()
    pp.runpp(net, numba=False)  # sanity check baseline

    base_p = net.load["p_mw"].copy()
    base_q = net.load["q_mvar"].copy()

    dt = 1
    T_total = 90000  
    n = T_total // dt

    subs = get_trafo_names(net)
    schedules = generate_schedules(subs, T_total)

    all_rows = []
    fault_extra_load_idxs = {}

    for k in range(n):
        t = k * dt
        
        # Reset baseline + noise
        net.load["p_mw"] = base_p * (1 + 0.03 * (np.random.rand(len(base_p)) - 0.5))
        net.load["q_mvar"] = base_q * (1 + 0.03 * (np.random.rand(len(base_q)) - 0.5))
        net.ext_grid["vm_pu"] = 1.0
        net.ext_grid["in_service"] = True
        
        for idx in fault_extra_load_idxs.values():
            if idx in net.load.index:
                net.load.drop(index=idx, inplace=True)
        fault_extra_load_idxs.clear()

        freq = 50.0 + 0.02 * (np.random.rand() - 0.5)

        active_faults, g_fault = get_active_faults(schedules, t)
        
        # Apply Global Faults
        if g_fault == "voltage_sag":
            net.ext_grid["vm_pu"] = 0.7
        elif g_fault == "source_outage":
            net.ext_grid["in_service"] = False
            
        # Apply Localized Faults
        if g_fault is None:
            for sub, ftype in active_faults.items():
                if ftype == "overload":
                    target = f"{sub} load"
                    if target in net.load["name"].values:
                        idx = net.load.index[net.load["name"] == target][0]
                        net.load.at[idx, "p_mw"] *= 1.6
                elif ftype == "line_fault":
                    target = f"{sub} MV"
                    if target in net.bus["name"].values:
                        mv_bus = net.bus.index[net.bus["name"] == target][0]
                        fault_extra_load_idxs[sub] = pp.create_load(
                            net, bus=mv_bus, p_mw=14.0, q_mvar=14.0, name=f"__fault_{sub}__"
                        )

        try:
            pp.runpp(net, init="results", numba=False)
            all_rows.extend(snapshot(net, t, active_faults, freq))
        except Exception:
            # non-convergence
            for name in subs:
                label = active_faults.get(name, "normal")
                if g_fault == "source_outage": label = "source_outage"
                all_rows.append({
                    "time_s": t, "substation": name,
                    "V_hv_kV": 0.0, "V_lv_kV": 0.0, "I_hv_A": 0.0,
                    "P_MW": 0.0, "Q_Mvar": 0.0, "PF": 0.0, "loading_pct": 0.0,
                    "freq_Hz": freq, "label": label,
                })

    df = pd.DataFrame(all_rows)
    df.to_csv("youssoufia_pandapower_dataset.csv", index=False)
    print("Rows:", len(df), "| timesteps:", n, "| substations:", df["substation"].nunique())
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    run()
