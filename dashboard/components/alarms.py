import streamlit as st

def render_alarms():
    st.markdown("""
        <div style='color:#38BDF8;font-size:14px;font-weight:700;margin-top:10px;
                    margin-bottom:8px;border-bottom:1px solid #1E293B;
                    padding-bottom:4px;text-transform:uppercase'>
            Active Alarms
        </div>
    """, unsafe_allow_html=True)

    alarms = st.session_state.alarms[:6]
    if not alarms:
        st.markdown(
            "<div style='font-size:13px;color:#64748B;padding:4px 0;'>"
            "No active alarms. System operating normally.</div>",
            unsafe_allow_html=True)
        return

    cards = ""
    for a in alarms:
        pri = a["priority"]
        if pri == "CRITICAL":
            col, bg = "#FCA5A5", "#7F1D1D"
        elif pri == "WARNING":
            col, bg = "#FDBA74", "#7C2D12"
        else:
            col, bg = "#6EE7B7", "#064E3B"

        cards += f"""
        <div style='background:#111827;border-left:3px solid {col};
                    padding:8px 10px;border-radius:4px;margin-bottom:6px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-family:"JetBrains Mono",monospace;
                             font-size:10px;color:#94A3B8;'>{a['time']}</span>
                <span style='background:{bg};color:{col};font-size:9px;
                             font-weight:800;padding:2px 4px;
                             border-radius:3px;'>{pri}</span>
            </div>
            <div style='font-size:11px;font-weight:700;
                        color:#E2E8F0;margin-bottom:2px;'>{a['sub']}</div>
            <div style='font-size:12px;color:#94A3B8;'>{a['msg']}</div>
        </div>"""

    st.markdown(
        f"<div style='display:flex;flex-direction:column;'>{cards}</div>",
        unsafe_allow_html=True)
