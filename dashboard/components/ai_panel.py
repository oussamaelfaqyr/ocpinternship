import streamlit as st
from .common import AI_RECOMMENDATIONS, AI_SEVERITY, FAULT_LABELS

def render_ai_diagnosis(selected_sub: str, ai_result: dict, current_telemetry: dict = None):
    st.markdown("""
        <div style='color:#38BDF8;font-size:14px;font-weight:700;margin-bottom:8px;
                    border-bottom:1px solid #1E293B;padding-bottom:4px;text-transform:uppercase'>
            AI Diagnosis &amp; Protection Lifecycle
        </div>
    """, unsafe_allow_html=True)

    res = ai_result.get(selected_sub, {})
    tel = (current_telemetry or {}).get(selected_sub, {})

    if not res:
        st.info("AI Analysis pending...")
        return

    pred       = res.get("pred_label", "normal")
    conf       = res.get("confidence", 100.0)
    stage      = tel.get("lifecycle_stage", "normal").replace("_", " ").upper()
    relay_st   = tel.get("relay_status",   "NORMAL")
    breaker_st = tel.get("breaker_status", "CLOSED")
    sev        = AI_SEVERITY.get(pred, "NONE")

    if sev == "CRITICAL":
        bg, text, bord = "#450A0A", "#FCA5A5", "#991B1B"
    elif sev == "MODERATE":
        bg, text, bord = "#422006", "#FDBA74", "#9A3412"
    else:
        bg, text, bord = "#052E16", "#6EE7B7", "#065F46"

    label = FAULT_LABELS.get(pred, pred)
    rec   = AI_RECOMMENDATIONS.get(pred, "No action required.")

    st.markdown(f"""
    <div style='background:{bg};border:1px solid {bord};border-radius:6px;
                padding:12px;margin-bottom:10px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div style='font-size:12px;color:{text};font-weight:600;
                        text-transform:uppercase;'>Predicted State</div>
            <div style='background:#1E293B;color:#38BDF8;font-size:10px;font-weight:800;
                        padding:2px 6px;border-radius:4px;
                        font-family:"JetBrains Mono",monospace;'>STAGE: {stage}</div>
        </div>
        <div style='font-size:18px;font-weight:700;color:#FFFFFF;margin:4px 0 8px;'>{label}</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
            <div>
                <div style='font-size:10px;color:{text};opacity:0.8;'>Confidence</div>
                <div style='font-size:14px;color:#FFFFFF;
                            font-family:"JetBrains Mono",monospace;'>{conf:.1f}%</div>
            </div>
            <div>
                <div style='font-size:10px;color:{text};opacity:0.8;'>Relay / Breaker</div>
                <div style='font-size:12px;color:#FFFFFF;
                            font-family:"JetBrains Mono",monospace;'>{relay_st} / {breaker_st}</div>
            </div>
            <div>
                <div style='font-size:10px;color:{text};opacity:0.8;'>Location</div>
                <div style='font-size:12px;color:#FFFFFF;
                            font-family:"JetBrains Mono",monospace;'>{selected_sub}</div>
            </div>
        </div>
        <div style='border-top:1px dashed {bord};margin:8px 0;padding-top:8px;'>
            <div style='font-size:11px;color:{text};font-weight:600;'>Recommended Action</div>
            <div style='font-size:12px;color:#FFFFFF;margin-top:2px;'>{rec}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
