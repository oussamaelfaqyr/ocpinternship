import plotly.graph_objects as go
from .common import FAULT_COLORS

def create_sld_figure(telemetry: dict, active_fault: str, selected_sub: str = None) -> go.Figure:
    """
    Plotly single-line diagram mimic of OCP Youssoufia 60 kV network.
    Occupies central space with clear industrial colors.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False, range=[0, 65]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
        showlegend=False,
        hovermode="closest",
    )

    nodes = {
        "SSP":   (10, 32, "SSP",  "Sous-Station\nPrincipale"),
        "LAV":   (28, 52, "LAV",  "Laverie/\nSechage"),
        "US":    (32, 32, "US",   "Portique\nUS"),
        "MZI":   (52, 44, "MZI",  "Mine\nMzinda"),
        "BOU":   (70, 54, "BOU",  "PSF Bouchane"),
        "REC2":  (52, 20, "REC2", "Portique\nRecette 2"),
        "REC3":  (68, 28, "REC3", "Recette 3"),
        "REC9":  (68, 10, "REC9", "Recette 9"),
        "UC":    (18, 10, "UC",   "U Calcination"),
    }

    edges = [
        ("SSP", "LAV"), ("SSP", "US"), ("SSP", "UC"),
        ("US",  "MZI"), ("MZI","BOU"), ("US",  "REC2"),
        ("REC2","REC3"),("REC2","REC9"),
    ]

    def node_col(nid):
        if active_fault in ("source_outage", "grid_failure"): return "#6B7280" # Grey
        for sub, row in telemetry.items():
            if nid in sub.upper().replace(" ","").replace("/",""):
                lbl = row.get("label", "normal")
                lp  = row.get("loading_pct", 0)
                if lbl in FAULT_COLORS: return FAULT_COLORS[lbl]
                if lp > 100: return "#EF4444" # Red
                if lp > 80:  return "#F59E0B" # Yellow/Amber
                return "#10B981" # Green
        return "#10B981"

    # Draw Edges (Feeders)
    for (a, b) in edges:
        x0, y0 = nodes[a][:2]; x1, y1 = nodes[b][:2]
        # Feeder color depends on node states. If both are normal, feeder is cyan.
        # If either is faulty, feeder becomes red. If outage, grey.
        c_a = node_col(a)
        c_b = node_col(b)
        
        if c_a == "#6B7280" or c_b == "#6B7280":
            ec = "#475569" # Dark Grey
        elif c_a == "#10B981" and c_b == "#10B981":
            ec = "#0EA5E9" # Normal Cyan feeder
        else:
            ec = "#EF4444" # Faulted feeder
            
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=ec, width=2.5),
                                 hoverinfo="skip", showlegend=False))

    # Grid Source (ONEE)
    grid_col = "#6B7280" if active_fault in ("source_outage", "grid_failure") else "#38BDF8"
    fig.add_trace(go.Scatter(x=[4], y=[32], mode="markers+text",
                             marker=dict(symbol="star", size=20, color=grid_col, line=dict(color="#0EA5E9", width=1.5)),
                             text=["ONEE Source"], textposition="bottom center",
                             textfont=dict(color=grid_col, size=10, family="Inter", weight="bold"),
                             hovertemplate="<b>ONEE 60 kV Feeder</b><extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=[4, 10], y=[32, 32], mode="lines",
                             line=dict(color=grid_col, width=3), hoverinfo="skip", showlegend=False))

    # Draw Nodes
    for nid, (x, y, short, label) in nodes.items():
        col = node_col(nid)
        w, h = (6, 3.5)
        
        # Determine if this node is currently selected in the UI
        is_selected = False
        if selected_sub:
            if nid in selected_sub.upper().replace(" ","").replace("/",""):
                is_selected = True

        # Outer highlight if selected
        if is_selected:
            fig.add_shape(type="rect", x0=x-w/2-0.8, y0=y-h/2-0.8, x1=x+w/2+0.8, y1=y+h/2+0.8,
                          line=dict(color="#F8FAFC", width=2, dash="dash"), fillcolor="rgba(255,255,255,0.05)")

        # Node background
        fig.add_shape(type="rect", x0=x-w/2, y0=y-h/2, x1=x+w/2, y1=y+h/2,
                      line=dict(color=col, width=2.5), fillcolor="#0F172A")
        # Node top bar (colored)
        fig.add_shape(type="rect", x0=x-w/2, y0=y+h/2-0.6, x1=x+w/2, y1=y+h/2,
                      line=dict(width=0), fillcolor=col, opacity=0.9)

        tel_txt = ""
        for sub, row in telemetry.items():
            if nid in sub.upper().replace(" ","").replace("/",""):
                tel_txt = f"{row.get('loading_pct', 0):.0f}% | {row.get('V_hv_kV', 60.0):.1f}kV"
                break

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(color="rgba(0,0,0,0)", size=40),
            text=[short], textfont=dict(color="#E2E8F0", size=11, family="Inter"),
            textposition="middle center",
            hovertemplate=f"<b>{label.replace(chr(10),' ')}</b><br>{tel_txt}<extra></extra>",
            showlegend=False,
        ))
        
        # Telemetry text below node
        if tel_txt:
            fig.add_annotation(x=x, y=y-h/2-1.2, text=tel_txt,
                               font=dict(size=9, color=col, family="JetBrains Mono", weight="bold"),
                               showarrow=False, bgcolor="rgba(15,23,42,0.8)", bordercolor="#1E293B", borderwidth=1, borderpad=2)

    return fig
