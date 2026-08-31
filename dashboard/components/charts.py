"""
SCADA-Grade Chart Components for OCP Youssoufia 60 kV Digital Twin.
===================================================================
All charts are designed to look and behave like industrial SCADA trend displays.
Key anti-blink settings:
  - uirevision="constant"  →  Plotly keeps zoom/pan state and smoothly animates new data
  - key=<unique string>    →  Streamlit patches the existing DOM element (caller must pass key)
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Shared Layout Base ────────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#060D1A",
    uirevision="constant",          # <<< CRITICAL: prevents full re-render on data update
    font=dict(family="'JetBrains Mono', 'Consolas', monospace", color="#94A3B8", size=12),
    margin=dict(l=70, r=30, t=60, b=55),
    legend=dict(
        orientation="h",
        y=1.05, x=0,
        font=dict(size=12, color="#CBD5E1"),
        bgcolor="rgba(6,13,26,0.7)",
        bordercolor="#1E293B",
        borderwidth=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0F172A",
        bordercolor="#38BDF8",
        font=dict(size=12, color="#F8FAFC"),
    ),
    xaxis=dict(
        gridcolor="#0F1F36",
        linecolor="#1E293B",
        zerolinecolor="#1E293B",
        tickfont=dict(size=11, color="#64748B"),
        title=dict(text="Simulation Time (s)", font=dict(size=12, color="#64748B")),
        showspikes=True,
        spikesnap="cursor",
        spikecolor="#38BDF8",
        spikethickness=1,
        spikemode="across",
    ),
    yaxis=dict(
        gridcolor="#0F1F36",
        linecolor="#1E293B",
        zerolinecolor="#1E293B",
        tickfont=dict(size=12, color="#94A3B8"),
        zeroline=False,
        showspikes=True,
        spikecolor="#38BDF8",
        spikethickness=1,
    ),
)

_HEIGHT = 420


def _get_history(sub: str, n: int = 600) -> pd.DataFrame:
    if not st.session_state.get("history"):
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.history)
    if df.empty:
        return pd.DataFrame()
    return df[df["sub"] == sub].sort_values("t").tail(n)


def _add_alarm_zone(fig, y_lo, y_hi, color="rgba(239,68,68,0.08)", label=""):
    fig.add_hrect(y0=y_lo, y1=y_hi, fillcolor=color, layer="below", line_width=0,
                  annotation_text=label, annotation_position="top right",
                  annotation_font=dict(size=10, color=color.replace("0.08", "0.6")))


def _add_nominal_line(fig, y_val, label, color="#1E3A5F", dash="dash"):
    fig.add_hline(
        y=y_val,
        line_color=color,
        line_dash=dash,
        line_width=1.5,
        annotation_text=f"  {label}",
        annotation_position="top left",
        annotation_font=dict(size=10, color=color),
    )


def _base_layout(extra_yaxis=None, **overrides):
    layout = dict(_LAYOUT_BASE)
    layout["yaxis"] = dict(_LAYOUT_BASE["yaxis"])
    if extra_yaxis:
        layout["yaxis2"] = extra_yaxis
    layout.update(overrides)
    return layout


# ─── 1. Voltage Chart ─────────────────────────────────────────────────────────
def create_voltage_chart(sub: str) -> go.Figure:
    ds = _get_history(sub)
    fig = go.Figure()

    if not ds.empty:
        t = ds["t"]
        _add_alarm_zone(fig, 0,  54, color="rgba(239,68,68,0.07)",  label="⚠ UV Alarm")
        _add_alarm_zone(fig, 66, 72, color="rgba(239,68,68,0.07)",  label="⚠ OV Alarm")

        fig.add_trace(go.Scatter(
            x=t, y=ds["V_hv_kV"], name="HV Bus (kV)", mode="lines",
            line=dict(color="#38BDF8", width=2.5),
            hovertemplate="<b>HV</b>: %{y:.3f} kV<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=t, y=ds["V_lv_kV"], name="LV Bus (kV)", mode="lines",
            line=dict(color="#A78BFA", width=2, dash="dot"),
            hovertemplate="<b>LV</b>: %{y:.3f} kV<extra></extra>",
        ))
        _add_nominal_line(fig, 60.0, "Vn=60 kV",  color="#1E4976")
        _add_nominal_line(fig, 5.5,  "Vn=5.5 kV", color="#2D1B5E")

    fig.update_layout(
        title=dict(text="<b>Voltage Trends — HV / LV Bus</b>",
                   font=dict(size=14, color="#E2E8F0"), x=0.0, xanchor="left"),
        height=_HEIGHT,
        yaxis=dict(_LAYOUT_BASE["yaxis"],
                   title=dict(text="Voltage (kV)", font=dict(size=13, color="#64748B"))),
        **{k: v for k, v in _LAYOUT_BASE.items() if k != "yaxis"},
    )
    return fig


# ─── 2. Current & Loading Chart ───────────────────────────────────────────────
def create_current_loading_chart(sub: str) -> go.Figure:
    ds = _get_history(sub)
    fig = go.Figure()

    if not ds.empty:
        t = ds["t"]
        _add_alarm_zone(fig, 100, 150, color="rgba(239,68,68,0.10)",  label="⚠ OVERLOAD")
        _add_alarm_zone(fig, 80,  100, color="rgba(245,158,11,0.07)", label="⚠ High Load")

        fig.add_trace(go.Scatter(
            x=t, y=ds["I_hv_A"], name="HV Current (A)", mode="lines",
            line=dict(color="#F43F5E", width=2.5),
            hovertemplate="<b>I_HV</b>: %{y:.1f} A<extra></extra>",
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=t, y=ds["loading_pct"], name="Trafo Loading (%)", mode="lines",
            line=dict(color="#F59E0B", width=2),
            hovertemplate="<b>Loading</b>: %{y:.1f}%<extra></extra>",
            yaxis="y2",
        ))
        _add_nominal_line(fig, 100, "100% Load", color="#991B1B", dash="dot")

    fig.update_layout(
        title=dict(text="<b>Current & Transformer Loading</b>",
                   font=dict(size=14, color="#E2E8F0"), x=0.0, xanchor="left"),
        height=_HEIGHT,
        yaxis=dict(_LAYOUT_BASE["yaxis"],
                   title=dict(text="Current (A)", font=dict(size=13, color="#F43F5E"))),
        yaxis2=dict(
            title=dict(text="Loading (%)", font=dict(size=13, color="#F59E0B")),
            overlaying="y", side="right",
            gridcolor="#0F1F36",
            tickfont=dict(size=12, color="#F59E0B"),
            zeroline=False,
        ),
        **{k: v for k, v in _LAYOUT_BASE.items() if k != "yaxis"},
    )
    return fig


# ─── 3. Power Flow Chart ──────────────────────────────────────────────────────
def create_power_chart(sub: str) -> go.Figure:
    ds = _get_history(sub)
    fig = go.Figure()

    if not ds.empty:
        t = ds["t"]
        fig.add_trace(go.Scatter(
            x=t, y=ds["P_MW"], name="Active Power P (MW)", mode="lines",
            line=dict(color="#34D399", width=2.5),
            hovertemplate="<b>P</b>: %{y:.3f} MW<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=t, y=ds["Q_Mvar"], name="Reactive Power Q (Mvar)", mode="lines",
            line=dict(color="#C084FC", width=2, dash="dot"),
            hovertemplate="<b>Q</b>: %{y:.3f} Mvar<extra></extra>",
        ))
        if "S_MVA" in ds.columns:
            fig.add_trace(go.Scatter(
                x=t, y=ds["S_MVA"], name="Apparent Power S (MVA)", mode="lines",
                line=dict(color="#FCD34D", width=1.5, dash="dashdot"),
                hovertemplate="<b>S</b>: %{y:.3f} MVA<extra></extra>",
                opacity=0.7,
            ))

    fig.update_layout(
        title=dict(text="<b>Power Flow — P / Q / S</b>",
                   font=dict(size=14, color="#E2E8F0"), x=0.0, xanchor="left"),
        height=_HEIGHT,
        yaxis=dict(_LAYOUT_BASE["yaxis"],
                   title=dict(text="Power (MW / Mvar / MVA)", font=dict(size=13, color="#64748B"))),
        **{k: v for k, v in _LAYOUT_BASE.items() if k != "yaxis"},
    )
    return fig


# ─── 4. Frequency Chart ───────────────────────────────────────────────────────
def create_frequency_chart(sub: str) -> go.Figure:
    ds = _get_history(sub)
    fig = go.Figure()

    if not ds.empty and "freq_Hz" in ds.columns:
        t = ds["t"]
        _add_alarm_zone(fig, 47.5, 49.0, color="rgba(239,68,68,0.09)", label="⚠ Under-freq")
        _add_alarm_zone(fig, 51.0, 52.5, color="rgba(239,68,68,0.09)", label="⚠ Over-freq")

        fig.add_trace(go.Scatter(
            x=t, y=ds["freq_Hz"], name="Grid Frequency (Hz)", mode="lines",
            line=dict(color="#84CC16", width=2.5),
            hovertemplate="<b>f</b>: %{y:.3f} Hz<extra></extra>",
        ))
        _add_nominal_line(fig, 50.0, "fn=50 Hz", color="#1A3A0F")

    fig.update_layout(
        title=dict(text="<b>Grid Frequency</b>",
                   font=dict(size=14, color="#E2E8F0"), x=0.0, xanchor="left"),
        height=_HEIGHT,
        yaxis=dict(_LAYOUT_BASE["yaxis"],
                   title=dict(text="Frequency (Hz)", font=dict(size=13, color="#84CC16")),
                   range=[48.5, 51.5]),
        **{k: v for k, v in _LAYOUT_BASE.items() if k != "yaxis"},
    )
    return fig
