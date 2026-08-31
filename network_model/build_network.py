"""
OCP Youssoufia HTB 60 kV Network Model using Pandapower.

Topology, cable types, lengths, and transformer ratings are modeled based on the 
OCP Youssoufia HTB network diagram (2022 edition).

Conductor types & technical parameters:
- Alu-acier 118 mm² : R = 0.28 Ohm/km, X = 0.38 Ohm/km, I_max = 0.36 kA
- Alu-acier 116 mm² / 116.2 mm² : R = 0.29 Ohm/km, X = 0.38 Ohm/km, I_max = 0.35 kA
- Alu-acier 181 mm² : R = 0.18 Ohm/km, X = 0.36 Ohm/km, I_max = 0.46 kA
- Alu-acier 54.4 mm² : R = 0.61 Ohm/km, X = 0.40 Ohm/km, I_max = 0.21 kA
- Almelec 59 mm² : R = 0.56 Ohm/km, X = 0.39 Ohm/km, I_max = 0.23 kA
"""

import numpy as np
import pandapower as pp

# Nominals
HV_KV = 60.0
MV_KV = 5.5  # Secondary MV voltage (assumed standard OCP 5.5 kV HTA site voltage)

# Conductor Technical Parameters Dictionary
CONDUCTOR_PARAMS = {
    "118_alu_acier": {"r": 0.28, "x": 0.38, "c": 9.2, "i_max": 0.36},
    "116_alu_acier": {"r": 0.29, "x": 0.38, "c": 9.0, "i_max": 0.35},
    "181_alu_acier": {"r": 0.18, "x": 0.36, "c": 9.8, "i_max": 0.46},
    "54.4_alu_acier": {"r": 0.61, "x": 0.40, "c": 8.5, "i_max": 0.21},
    "59_almelec": {"r": 0.56, "x": 0.39, "c": 8.7, "i_max": 0.23},
}

# Transformer Parameters (Standard HTB/HTA industrial values)
TRAFO_VK_PCT = 8.0
TRAFO_VKR_PCT = 0.5
TRAFO_PFE_KW_PER_MVA = 1.2
TRAFO_I0_PCT = 0.3


def add_line(net, bus_from, bus_to, length_km, name, cond_type="116_alu_acier"):
    """Helper to add a 60 kV overhead line with specific conductor properties."""
    params = CONDUCTOR_PARAMS.get(cond_type, CONDUCTOR_PARAMS["116_alu_acier"])
    return pp.create_line_from_parameters(
        net,
        from_bus=bus_from,
        to_bus=bus_to,
        length_km=length_km,
        r_ohm_per_km=params["r"],
        x_ohm_per_km=params["x"],
        c_nf_per_km=params["c"],
        max_i_ka=params["i_max"],
        name=name,
    )


def add_trafo(net, hv_bus, mv_bus, sn_mva, name):
    """Helper to add a step-down transformer (60 kV / 5.5 kV)."""
    return pp.create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=mv_bus,
        sn_mva=sn_mva,
        vn_hv_kv=HV_KV,
        vn_lv_kv=MV_KV,
        vk_percent=TRAFO_VK_PCT,
        vkr_percent=TRAFO_VKR_PCT,
        pfe_kw=TRAFO_PFE_KW_PER_MVA * sn_mva,
        i0_percent=TRAFO_I0_PCT,
        name=name,
    )


def build_network():
    """Builds and returns the complete OCP Youssoufia 60 kV HTB network."""
    net = pp.create_empty_network(name="OCP Youssoufia HTB (60 kV)", f_hz=50.0)

    # ==================== 1. 60 kV Backbone Buses ====================
    b_ssp = pp.create_bus(net, vn_kv=HV_KV, name="Sous Station Principale")
    b_lav = pp.create_bus(net, vn_kv=HV_KV, name="Poste Laverie/Sechage")
    b_us = pp.create_bus(net, vn_kv=HV_KV, name="Portique US")
    b_mzinda = pp.create_bus(net, vn_kv=HV_KV, name="Poste Mine Mzinda")
    b_bouchane = pp.create_bus(net, vn_kv=HV_KV, name="PSF Mine Bouchane")
    b_rec2 = pp.create_bus(net, vn_kv=HV_KV, name="Portique Recette 2")
    b_rec3 = pp.create_bus(net, vn_kv=HV_KV, name="Poste Recette 3")
    b_rec9 = pp.create_bus(net, vn_kv=HV_KV, name="Poste Recette 9")
    b_uc = pp.create_bus(net, vn_kv=HV_KV, name="Poste U Calcination")

    # ==================== 2. External Grid Connection ====================
    # ONEE 60 kV Grid source connected at Sous Station Principale (SSP)
    pp.create_ext_grid(
        net,
        bus=b_ssp,
        vm_pu=1.0,
        name="Grid (ONEE 60 kV)",
        s_sc_max_mva=500.0,
        rx_max=0.1,
    )

    # ==================== 3. 60 kV Lines & Lengths ====================
    # SSP to Laverie/Sechage (D.LAV branch)
    add_line(net, b_ssp, b_lav, 0.667, "Ligne SSP - Laverie/Sechage (667m)", "118_alu_acier")

    # SSP to Portique US (D.REC branch)
    add_line(net, b_ssp, b_us, 0.550, "Ligne SSP - Portique US (550m)", "116_alu_acier")

    # Portique US to Mine Mzinda
    add_line(net, b_us, b_mzinda, 0.150, "Ligne Portique US - Mzinda Sec 1 (150m)", "118_alu_acier")
    add_line(net, b_us, b_mzinda, 5.075, "Ligne Portique US - Mzinda Sec 2 (5075m)", "54.4_alu_acier")

    # Mine Mzinda to PSF Mine Bouchane
    add_line(net, b_mzinda, b_bouchane, 1.058, "Ligne Mzinda - Bouchane Sec 1 (1058m)", "54.4_alu_acier")
    add_line(net, b_mzinda, b_bouchane, 19.788, "Ligne Mzinda - Bouchane Sec 2 (19.788km)", "59_almelec")

    # Portique US to Portique Recette 2
    add_line(net, b_us, b_rec2, 4.169, "Ligne Portique US - Recette 2 (4169m)", "116_alu_acier")

    # Portique Recette 2 to Recette 3
    add_line(net, b_rec2, b_rec3, 1.200, "Ligne Recette 2 - Recette 3 (1200m)", "116_alu_acier")

    # Portique Recette 2 to Recette 9
    add_line(net, b_rec2, b_rec9, 3.061, "Ligne Recette 2 - Recette 9 (3061m)", "116_alu_acier")

    # SSP to U Calcination (Parallel circuits UC1 and UC2)
    add_line(net, b_ssp, b_uc, 4.980, "Ligne SSP - U Calcination UC1 (4980m)", "118_alu_acier")
    add_line(net, b_ssp, b_uc, 4.924, "Ligne SSP - U Calcination UC2 (4924m)", "181_alu_acier")

    # ==================== 4. MV Secondary Buses, Transformers & Loads ====================
    def add_substation_load(hv_bus, sn_mva, name, loading_pct=0.55, pf=0.90):
        mv_bus = pp.create_bus(net, vn_kv=MV_KV, name=f"{name} MV")
        add_trafo(net, hv_bus, mv_bus, sn_mva, f"{name} trafo")
        p_mw = sn_mva * loading_pct * pf
        q_mvar = p_mw * ((1.0 - pf**2) ** 0.5) / pf
        pp.create_load(net, bus=mv_bus, p_mw=p_mw, q_mvar=q_mvar, name=f"{name} load")
        return mv_bus

    # Laverie / Sechage
    add_substation_load(b_lav, 12.5, "Laverie/Sechage SN1")
    add_substation_load(b_lav, 12.5, "Laverie/Sechage SN2")

    # Mine Mzinda
    add_substation_load(b_mzinda, 7.5, "Mine Mzinda DIS TR")

    # PSF Mine Bouchane
    add_substation_load(b_bouchane, 2.5, "PSF Mine Bouchane SN6")

    # Recette 3
    add_substation_load(b_rec3, 2.0, "Recette 3 SN3")

    # Recette 9
    add_substation_load(b_rec9, 2.5, "Recette 9 SN9")

    # U Calcination
    add_substation_load(b_uc, 12.5, "U Calcination SN1")
    add_substation_load(b_uc, 12.5, "U Calcination SN2")
    add_substation_load(b_uc, 10.0, "U Calcination SN3")

    # Sous Station Principale (SSP Local Auxiliary Transformers)
    add_substation_load(b_ssp, 5.0, "SSP Local ST1")
    add_substation_load(b_ssp, 5.0, "SSP Local ST2")

    # ==================== 5. Zero-Sequence Impedances (IEC 60909) ====================
    # Required for pandapower single-phase (1ph) and two-phase (2ph) short-circuit
    # calculations. Without these, calc_sc() crashes on modern pandapower versions.
    #
    # Overhead line zero-sequence ratios (effectively grounded 60 kV network):
    #   r0/r1 ≈ 1.0 – 1.5,  x0/x1 ≈ 2.0 – 2.5  (IEC 60909 Table 4, typical OHL)
    #   c0/c1 ≈ 0.5 – 0.7
    net.line["r0_ohm_per_km"] = net.line["r_ohm_per_km"] * 1.2
    net.line["x0_ohm_per_km"] = net.line["x_ohm_per_km"] * 2.2
    net.line["c0_nf_per_km"]  = net.line["c_nf_per_km"]  * 0.6

    # Transformer zero-sequence parameters (Dyn / YNyn vector group):
    net.trafo["vk0_percent"]    = net.trafo["vk_percent"]  * 0.85
    net.trafo["vkr0_percent"]   = net.trafo["vkr_percent"]
    net.trafo["vector_group"]   = "YNyn"
    net.trafo["mag0_percent"]   = 100.0
    net.trafo["mag0_rx"]        = 0.0
    net.trafo["si0_hv_partial"] = 0.9

    # External grid zero-sequence source impedance:
    net.ext_grid["x0x_max"]  = 1.0
    net.ext_grid["r0x0_max"] = 0.1

    return net


if __name__ == "__main__":
    net = build_network()
    pp.runpp(net)
    print("=== OCP Youssoufia 60 kV Network Power Flow Summary ===")
    print(f"Network Name: {net.name}")
    print(f"Buses: {len(net.bus)} | Lines: {len(net.line)} | Transformers: {len(net.trafo)} | Loads: {len(net.load)}")
    print(f"Power Flow Converged: {net['converged']}")
    print("\n--- Bus Voltages (kV / p.u.) ---")
    print(net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]])
    print("\n--- Transformer Loadings (%) ---")
    print(net.res_trafo[["loading_percent", "i_hv_ka", "p_hv_mw", "q_hv_mvar"]])
