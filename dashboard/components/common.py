import streamlit as st

# ── Streamlit Width Constants ────────────────────────────────────────────────────
WIDTH_STRETCH = {"width": "stretch"}
WIDTH_CONTENT = {"width": "content"}

# ── Colors & Styling ─────────────────────────────────────────────────────────────
FAULT_COLORS = {
    "normal":            "#10B981", # Green
    "voltage_sag":       "#38BDF8", # Blue
    "overload":          "#F59E0B", # Amber
    "lg_fault":          "#EF4444", # Red
    "ll_fault":          "#A855F7", # Purple
    "under_frequency":   "#EAB308", # Yellow
    "over_frequency":    "#E879F9", # Pink
    "source_outage":     "#6B7280", # Grey
    "breaker_trip":      "#F97316", # Orange
    "transformer_trip":  "#EC4899", # Rose
}

FAULT_LABELS = {
    "normal":            "Normal Operation",
    "voltage_sag":       "Grid Voltage Sag",
    "overload":          "Thermal Overload",
    "lg_fault":          "Line-to-Ground Fault",
    "ll_fault":          "Line-to-Line Fault",
    "under_frequency":   "Under-Frequency (81U)",
    "over_frequency":    "Over-Frequency (81O)",
    "source_outage":     "Source Outage",
    "breaker_trip":      "Breaker Trip",
    "transformer_trip":  "Transformer Trip",
}

AI_RECOMMENDATIONS = {
    "lg_fault":          "Isolate ground fault feeder. Check zero-sequence relay.",
    "ll_fault":          "Open phase CB. Inspect inter-phase insulation.",
    "overload":          "Shed non-critical loads. Check trafo cooling.",
    "voltage_sag":       "Monitor ONEE PCC. Verify tap changer position.",
    "under_frequency":   "Prepare load shedding (81U). Check spinning reserve.",
    "over_frequency":    "Reduce local generation or alert grid operator.",
    "source_outage":     "Emergency switchover to diesel/backup generators.",
    "breaker_trip":      "Inspect isolated feeder and perform line test.",
    "transformer_trip":  "Inspect Buchholz relay & oil temp before reclosing.",
    "normal":            "Grid operating nominally. No action required.",
}

AI_SEVERITY = {
    "lg_fault": "CRITICAL", "ll_fault": "CRITICAL",
    "source_outage": "CRITICAL", "breaker_trip": "CRITICAL",
    "transformer_trip": "CRITICAL",
    "voltage_sag": "MODERATE", "overload": "MODERATE",
    "under_frequency": "MODERATE", "over_frequency": "MODERATE",
    "normal": "NONE",
}

def apply_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, .stApp {
        background: #090D16 !important;
        color: #CBD5E1 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .block-container {
        padding: 1rem 1.5rem 1rem 1.5rem !important;
        max-width: 100% !important;
    }

    [data-testid="stSidebar"] {
        background: #0B1120 !important;
        border-right: 1px solid #1E293B !important;
        min-width: 260px !important;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebarNav"] { display: none; }

    footer { visibility: hidden; }
    [data-testid="stDeployButton"] { display: none; }

    h1, h2, h3 { color: #38BDF8 !important; font-weight: 700 !important; }

    .stButton > button {
        background: #0F172A !important;
        color: #38BDF8 !important;
        border: 1px solid #2B3442 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #1E293B !important;
        border-color: #38BDF8 !important;
    }

    .stSelectbox > div > div {
        background: #0F172A !important;
        border: 1px solid #2B3442 !important;
        color: #CBD5E1 !important;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #2B3442; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)
