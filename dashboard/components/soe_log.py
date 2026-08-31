"""
Sequence of Events (SOE) Log + Fault Lifecycle Badge
=====================================================
Industrial DCS-style event historian and active fault status tracker.
"""
import streamlit as st
import datetime

def render_soe_log():
    st.markdown("""
        <div style='color:#38BDF8;font-size:14px;font-weight:700;margin-bottom:8px;
                    border-bottom:1px solid #1E293B;padding-bottom:4px;
                    text-transform:uppercase'>
            📋 Sequence of Events (SOE)
        </div>
    """, unsafe_allow_html=True)

    events = st.session_state.get("events", [])
    if not events:
        st.markdown("<div style='color:#64748B;font-size:12px;'>No events recorded yet.</div>",
                    unsafe_allow_html=True)
        return

    rows = ""
    for i, ev in enumerate(events[:30]):
        bg = "#0D1929" if i % 2 == 0 else "#080E1C"
        # Colour-code CMD vs alarm events
        if ev["event"].startswith("CMD:"):
            color = "#FBBF24"
            icon  = "⚡"
        elif "CRITICAL" in ev["event"] or "Fault" in ev["event"] or "fault" in ev["event"]:
            color = "#F87171"
            icon  = "🔴"
        elif "WARNING" in ev["event"] or "High load" in ev["event"]:
            color = "#FB923C"
            icon  = "🟡"
        else:
            color = "#94A3B8"
            icon  = "ℹ"

        rows += f"""
        <tr style='background:{bg};'>
            <td style='padding:5px 8px;color:#475569;font-size:10px;
                       font-family:"JetBrains Mono",monospace;white-space:nowrap;'>{ev['time']}</td>
            <td style='padding:5px 4px;font-size:12px;'>{icon}</td>
            <td style='padding:5px 8px;color:{color};font-size:11px;'>{ev['event']}</td>
        </tr>"""

    st.markdown(f"""
    <div style='max-height:280px;overflow-y:auto;border:1px solid #1E293B;border-radius:6px;'>
      <table style='width:100%;border-collapse:collapse;'>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


def render_fault_lifecycle_badge():
    """Shows the active fault, its lifecycle stage, and a countdown timer."""
    fault   = st.session_state.get("fault", "normal")
    tgt     = st.session_state.get("fault_tgt", "")
    exp     = st.session_state.get("fault_exp", 0)
    step    = st.session_state.get("sim_step", 0)
    remaining = max(0, exp - step)

    from .common import FAULT_LABELS, FAULT_COLORS
    color = FAULT_COLORS.get(fault, "#64748B")
    label = FAULT_LABELS.get(fault, fault)

    if fault == "normal":
        st.markdown("""
            <div style='background:#052E16;border:1px solid #065F46;border-radius:6px;
                        padding:10px 14px;display:flex;align-items:center;gap:10px;'>
                <div style='font-size:20px;'>🟢</div>
                <div>
                    <div style='font-size:11px;color:#6EE7B7;font-weight:700;
                                text-transform:uppercase;'>System Status</div>
                    <div style='font-size:14px;color:#FFFFFF;font-weight:600;'>
                        Normal Operation</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # Active fault — show lifecycle stage + countdown
    if remaining > exp * 0.6:
        stage, s_color = "PRE-FAULT / DETECTED", "#FBBF24"
    elif remaining > 5:
        stage, s_color = "FAULT ACTIVE / TRIPPED", "#EF4444"
    else:
        stage, s_color = "RECOVERING", "#F97316"

    pct = int((1 - remaining / max(1, exp - step + remaining)) * 100)

    st.markdown(f"""
        <div style='background:#1A0A0A;border:2px solid {color};border-radius:6px;
                    padding:12px 14px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:8px;'>
                <div style='font-size:12px;color:{color};font-weight:800;
                            text-transform:uppercase;letter-spacing:1px;'>⚡ {label}</div>
                <div style='background:{color}22;border:1px solid {color};color:{color};
                            font-size:18px;font-weight:900;padding:2px 10px;border-radius:4px;
                            font-family:"JetBrains Mono",monospace;'>{remaining}s</div>
            </div>
            <div style='font-size:11px;color:#94A3B8;margin-bottom:4px;'>
                Target: <span style='color:#E2E8F0;font-weight:600;'>{tgt}</span>
            </div>
            <div style='font-size:11px;color:{s_color};font-weight:700;
                        margin-bottom:8px;'>{stage}</div>
            <div style='background:#1E293B;border-radius:3px;height:6px;'>
                <div style='width:{pct}%;background:{color};height:6px;
                            border-radius:3px;transition:width 0.5s;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
