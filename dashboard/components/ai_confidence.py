"""
AI Confidence Bar Chart + Feature Vector Inspector
===================================================
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from .common import FAULT_LABELS, FAULT_COLORS

def render_ai_confidence_chart(selected_sub: str, ai_result: dict):
    res = ai_result.get(selected_sub, {})
    if not res or not res.get("top_probs"):
        return

    probs  = res["top_probs"]
    labels = [FAULT_LABELS.get(k, k) for k in probs]
    values = list(probs.values())
    colors = [FAULT_COLORS.get(k, "#64748B") for k in probs]

    # Sort descending
    sorted_data = sorted(zip(values, labels, colors), reverse=True)
    values, labels, colors = zip(*sorted_data) if sorted_data else ([], [], [])

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(labels),
        orientation="h",
        marker=dict(
            color=list(colors),
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=11),
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#060D1A",
        uirevision="ai_conf",
        height=220,
        margin=dict(l=0, r=60, t=30, b=10),
        xaxis=dict(range=[0, 110], gridcolor="#0F1F36", ticksuffix="%",
                   tickfont=dict(size=10, color="#64748B"), zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#E2E8F0")),
        title=dict(text="<b>Fault Class Probabilities</b>",
                   font=dict(size=13, color="#E2E8F0"), x=0, xanchor="left"),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                    key="k_ai_conf")


def render_feature_inspector(selected_sub: str, telemetry: dict):
    tel = (telemetry or {}).get(selected_sub, {})
    if not tel:
        return

    features = {
        "V_hv_kV":     ("HV Voltage",    "kV",   tel.get("V_hv_kV", 0)),
        "V_lv_kV":     ("LV Voltage",    "kV",   tel.get("V_lv_kV", 0)),
        "I_hv_A":      ("HV Current",    "A",    tel.get("I_hv_A",  0)),
        "P_MW":        ("Active Power",  "MW",   tel.get("P_MW",    0)),
        "Q_Mvar":      ("Reactive Power","Mvar",  tel.get("Q_Mvar",  0)),
        "S_MVA":       ("Apparent S",   "MVA",  tel.get("S_MVA",   0)),
        "PF":          ("Power Factor",  "—",    tel.get("PF",      0)),
        "loading_pct": ("Trafo Loading", "%",    tel.get("loading_pct", 0)),
        "freq_Hz":     ("Frequency",     "Hz",   tel.get("freq_Hz", 50)),
    }

    rows = ""
    for key, (name, unit, val) in features.items():
        nan_val = val != val  # NaN check
        val_str = "NaN ⚠" if nan_val else f"{val:.4f}"
        val_color = "#F87171" if nan_val else "#E2E8F0"
        rows += f"""
        <tr style='border-bottom:1px solid #0F172A;'>
            <td style='padding:5px 10px;color:#64748B;font-size:11px;'>{name}</td>
            <td style='padding:5px 10px;color:#475569;font-size:10px;
                       font-family:"JetBrains Mono",monospace;'>{key}</td>
            <td style='padding:5px 10px;color:{val_color};font-size:12px;
                       font-family:"JetBrains Mono",monospace;text-align:right;
                       font-weight:700;'>{val_str}</td>
            <td style='padding:5px 10px;color:#475569;font-size:10px;'>{unit}</td>
        </tr>"""

    with st.expander("🔬 Live Feature Vector (AI Input)", expanded=False):
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;'>
          <thead>
            <tr style='border-bottom:2px solid #1E293B;'>
              <th style='padding:5px 10px;color:#38BDF8;font-size:11px;text-align:left;'>Feature</th>
              <th style='padding:5px 10px;color:#38BDF8;font-size:11px;text-align:left;'>Key</th>
              <th style='padding:5px 10px;color:#38BDF8;font-size:11px;text-align:right;'>Value</th>
              <th style='padding:5px 10px;color:#38BDF8;font-size:11px;text-align:left;'>Unit</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
