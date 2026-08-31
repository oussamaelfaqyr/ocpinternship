"""
Dataset Statistics Panel
========================
Shows label distribution, loading histogram, and per-fault breakdown.
"""
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from .common import FAULT_LABELS, FAULT_COLORS


def render_dataset_stats(csv_path: str):
    if not os.path.exists(csv_path):
        st.info("No dataset generated yet. Generate one to see statistics.")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "label" not in df.columns:
        st.warning("Dataset exists but has no label column.")
        return

    st.markdown("""
        <div style='color:#38BDF8;font-size:14px;font-weight:700;margin-bottom:12px;
                    border-bottom:1px solid #1E293B;padding-bottom:4px;
                    text-transform:uppercase'>
            📊 Dataset Quality Statistics
        </div>
    """, unsafe_allow_html=True)

    total = len(df)
    n_faults = (df["label"] != "normal").sum()
    n_subs   = df["substation"].nunique() if "substation" in df.columns else "—"

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples", f"{total:,}")
    m2.metric("Fault Samples", f"{n_faults:,}")
    m3.metric("Normal Samples", f"{total - n_faults:,}")
    m4.metric("Substations", str(n_subs))

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    col_pie, col_hist = st.columns(2)

    # ── Label pie chart ──────────────────────────────────────────────────────
    with col_pie:
        counts = df["label"].value_counts()
        pie_labels = [FAULT_LABELS.get(l, l) for l in counts.index]
        pie_colors = [FAULT_COLORS.get(l, "#64748B") for l in counts.index]

        fig_pie = go.Figure(go.Pie(
            labels=pie_labels,
            values=counts.values,
            hole=0.55,
            marker=dict(colors=pie_colors,
                        line=dict(color="#060D1A", width=2)),
            textfont=dict(size=11, color="#CBD5E1"),
            hovertemplate="<b>%{label}</b><br>%{value:,} samples (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(size=10, color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            title=dict(text="<b>Label Distribution</b>",
                       font=dict(size=13, color="#E2E8F0"), x=0, xanchor="left"),
            annotations=[dict(text=f"<b>{total:,}</b><br>total",
                              x=0.5, y=0.5, font_size=14, showarrow=False,
                              font_color="#E2E8F0")],
        )
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    # ── Loading % histogram ──────────────────────────────────────────────────
    with col_hist:
        if "loading_pct" in df.columns:
            fig_hist = go.Figure(go.Histogram(
                x=df["loading_pct"].clip(0, 150),
                nbinsx=40,
                marker=dict(color="#38BDF8", opacity=0.8,
                            line=dict(color="#060D1A", width=0.5)),
                hovertemplate="Loading: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=80,  line_color="#F59E0B", line_dash="dash",
                               annotation_text="80%", annotation_font_color="#F59E0B")
            fig_hist.add_vline(x=100, line_color="#EF4444", line_dash="dash",
                               annotation_text="100%", annotation_font_color="#EF4444")
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#060D1A",
                height=300,
                margin=dict(l=10, r=10, t=40, b=30),
                xaxis=dict(title="Loading (%)", gridcolor="#0F1F36",
                           tickfont=dict(size=10, color="#64748B")),
                yaxis=dict(title="Count", gridcolor="#0F1F36",
                           tickfont=dict(size=10, color="#64748B")),
                title=dict(text="<b>Transformer Loading Distribution</b>",
                           font=dict(size=13, color="#E2E8F0"), x=0, xanchor="left"),
            )
            st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

    # ── Per-fault bar chart ──────────────────────────────────────────────────
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    fault_counts = df[df["label"] != "normal"]["label"].value_counts()
    if not fault_counts.empty:
        fig_bar = go.Figure(go.Bar(
            x=[FAULT_LABELS.get(l, l) for l in fault_counts.index],
            y=fault_counts.values,
            marker=dict(
                color=[FAULT_COLORS.get(l, "#64748B") for l in fault_counts.index],
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=fault_counts.values,
            textposition="outside",
            textfont=dict(color="#CBD5E1", size=11),
            hovertemplate="<b>%{x}</b>: %{y:,} samples<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#060D1A",
            height=280,
            margin=dict(l=10, r=10, t=40, b=60),
            xaxis=dict(gridcolor="#0F1F36", tickfont=dict(size=10, color="#94A3B8"),
                       tickangle=-20),
            yaxis=dict(gridcolor="#0F1F36", tickfont=dict(size=10, color="#64748B"),
                       title="Samples"),
            title=dict(text="<b>Fault Class Sample Counts</b>",
                       font=dict(size=13, color="#E2E8F0"), x=0, xanchor="left"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
