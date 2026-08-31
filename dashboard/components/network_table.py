"""
Network-Wide Health Overview Table
===================================
Shows all substations simultaneously with live color-coded status.
"""
import streamlit as st
import pandas as pd
from .common import FAULT_LABELS, FAULT_COLORS

def render_network_table(telemetry: dict):
    st.markdown("""
        <div style='color:#38BDF8;font-size:14px;font-weight:700;margin-bottom:8px;
                    border-bottom:1px solid #1E293B;padding-bottom:4px;
                    text-transform:uppercase'>
            🌐 Network-Wide Status — All Substations
        </div>
    """, unsafe_allow_html=True)

    if not telemetry:
        st.markdown("<div style='color:#64748B;font-size:13px;'>Awaiting simulation data...</div>",
                    unsafe_allow_html=True)
        return

    rows = []
    for sub, r in telemetry.items():
        lbl   = r.get("label", "normal")
        load  = r.get("loading_pct", 0)
        v_hv  = r.get("V_hv_kV", 0)
        freq  = r.get("freq_Hz", 50.0)

        if lbl != "normal":
            status_icon = "🔴"
            row_bg = "rgba(239,68,68,0.08)"
        elif load > 100:
            status_icon = "🔴"
            row_bg = "rgba(239,68,68,0.08)"
        elif load > 80:
            status_icon = "🟡"
            row_bg = "rgba(245,158,11,0.06)"
        elif v_hv < 54 and v_hv > 0:
            status_icon = "🔴"
            row_bg = "rgba(239,68,68,0.08)"
        else:
            status_icon = "🟢"
            row_bg = "rgba(16,185,129,0.05)"

        rows.append({
            "row_bg":    row_bg,
            "status":    status_icon,
            "sub":       sub,
            "label":     FAULT_LABELS.get(lbl, lbl),
            "v_hv":      f"{v_hv:.2f}",
            "v_lv":      f"{r.get('V_lv_kV',0):.3f}",
            "i_hv":      f"{r.get('I_hv_A',0):.1f}",
            "p":         f"{r.get('P_MW',0):.2f}",
            "load":      load,
            "freq":      f"{freq:.3f}",
            "relay":     r.get("relay_status", "—"),
            "breaker":   r.get("breaker_status", "—"),
        })

    # Build HTML table
    header = """
    <table style='width:100%;border-collapse:collapse;font-family:"JetBrains Mono",monospace;
                  font-size:12px;'>
      <thead>
        <tr style='border-bottom:2px solid #1E293B;'>
          <th style='padding:6px 8px;color:#38BDF8;text-align:left;font-weight:700;'>St</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:left;font-weight:700;'>Substation</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:left;font-weight:700;'>State</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>HV (kV)</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>LV (kV)</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>I (A)</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>P (MW)</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>Load %</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:right;font-weight:700;'>Freq (Hz)</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:center;font-weight:700;'>Relay</th>
          <th style='padding:6px 8px;color:#38BDF8;text-align:center;font-weight:700;'>CB</th>
        </tr>
      </thead><tbody>"""

    body = ""
    for r in rows:
        load_val = r["load"]
        if load_val > 100:
            load_color = "#EF4444"
        elif load_val > 80:
            load_color = "#F59E0B"
        else:
            load_color = "#34D399"

        # Loading bar
        bar_w = min(100, load_val)
        load_bar = f"""
        <div style='display:flex;align-items:center;gap:6px;justify-content:flex-end;'>
            <div style='width:50px;background:#1E293B;border-radius:2px;height:6px;'>
                <div style='width:{bar_w}%;background:{load_color};height:6px;border-radius:2px;'></div>
            </div>
            <span style='color:{load_color};min-width:36px;text-align:right;'>{load_val:.1f}%</span>
        </div>"""

        relay_color = "#EF4444" if r["relay"] not in ("NORMAL","—") else "#64748B"
        cb_color    = "#EF4444" if r["breaker"] == "OPEN" else "#34D399"

        body += f"""
        <tr style='background:{r["row_bg"]};border-bottom:1px solid #0F172A;'>
          <td style='padding:7px 8px;text-align:center;'>{r["status"]}</td>
          <td style='padding:7px 8px;color:#E2E8F0;font-weight:600;'>{r["sub"]}</td>
          <td style='padding:7px 8px;color:#94A3B8;'>{r["label"]}</td>
          <td style='padding:7px 8px;color:#38BDF8;text-align:right;'>{r["v_hv"]}</td>
          <td style='padding:7px 8px;color:#818CF8;text-align:right;'>{r["v_lv"]}</td>
          <td style='padding:7px 8px;color:#F43F5E;text-align:right;'>{r["i_hv"]}</td>
          <td style='padding:7px 8px;color:#34D399;text-align:right;'>{r["p"]}</td>
          <td style='padding:7px 8px;'>{load_bar}</td>
          <td style='padding:7px 8px;color:#84CC16;text-align:right;'>{r["freq"]}</td>
          <td style='padding:7px 8px;color:{relay_color};text-align:center;font-size:11px;'>{r["relay"]}</td>
          <td style='padding:7px 8px;color:{cb_color};text-align:center;font-size:11px;'>{r["breaker"]}</td>
        </tr>"""

    st.markdown(header + body + "</tbody></table>", unsafe_allow_html=True)
