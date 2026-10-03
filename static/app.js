/* OCP Youssoufia SCADA Digital Twin — Web Dashboard JavaScript Application */

const FAULT_LABELS = {
    "normal": "Normal Operation",
    "voltage_sag": "Grid Voltage Sag",
    "overload": "Thermal Overload",
    "lg_fault": "Line-to-Ground Fault",
    "ll_fault": "Line-to-Line Fault",
    "under_frequency": "Under-Frequency (81U)",
    "over_frequency": "Over-Frequency (81O)",
    "source_outage": "Source Outage",
    "breaker_trip": "Breaker Trip",
    "transformer_trip": "Transformer Trip",
};

const AI_RECOMMENDATIONS = {
    "lg_fault": "Isolate ground fault feeder. Check zero-sequence relay.",
    "ll_fault": "Open phase CB. Inspect inter-phase insulation.",
    "overload": "Shed non-critical loads. Check transformer cooling.",
    "voltage_sag": "Monitor ONEE PCC. Verify tap changer position.",
    "under_frequency": "Prepare load shedding (81U). Check spinning reserve.",
    "over_frequency": "Reduce local generation or alert grid operator.",
    "source_outage": "Emergency switchover to diesel/backup generators.",
    "breaker_trip": "Inspect isolated feeder and perform line test.",
    "transformer_trip": "Inspect Buchholz relay & oil temp before reclosing.",
    "normal": "Grid operating nominally. No protective action required.",
};

const AI_SEVERITY = {
    "lg_fault": "CRITICAL", "ll_fault": "CRITICAL",
    "source_outage": "CRITICAL", "breaker_trip": "CRITICAL",
    "transformer_trip": "CRITICAL",
    "voltage_sag": "WARNING", "overload": "WARNING",
    "under_frequency": "WARNING", "over_frequency": "WARNING",
    "normal": "NONE",
};

// Global App State
const state = {
    selectedSubstation: "Mine Mzinda DIS TR",
    substations: [],
    telemetry: {},
    aiDiagnosis: {},
    alarms: [],
    events: [],
    history: [],
    status: {},
    charts: {}
};

// DOM Elements
const el = {
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    btnPlayPause: document.getElementById("btnPlayPause"),
    btnOpenDetailsModal: document.getElementById("btnOpenDetailsModal"),
    btnOpenFaultModal: document.getElementById("btnOpenFaultModal"),
    btnCloseFaultModal: document.getElementById("btnCloseFaultModal"),
    faultModal: document.getElementById("faultModal"),
    btnSubmitFault: document.getElementById("btnSubmitFault"),
    btnClearFaultBtn: document.getElementById("btnClearFaultBtn"),
    // Details Modal
    detailsModal: document.getElementById("detailsModal"),
    btnCloseDetailsModal: document.getElementById("btnCloseDetailsModal"),
    btnCloseDetailsModalBottom: document.getElementById("btnCloseDetailsModalBottom"),
    subPills: document.getElementById("subPills"),
    faultTargetSelect: document.getElementById("faultTargetSelect"),
    faultTypeSelect: document.getElementById("faultTypeSelect"),
    faultDurationInput: document.getElementById("faultDurationInput"),
    // KPIs
    kpiHv: document.getElementById("kpiHv"),
    kpiLv: document.getElementById("kpiLv"),
    kpiCurrent: document.getElementById("kpiCurrent"),
    kpiP: document.getElementById("kpiP"),
    kpiQ: document.getElementById("kpiQ"),
    kpiFreq: document.getElementById("kpiFreq"),
    kpiLoad: document.getElementById("kpiLoad"),
    kpiLoadCard: document.getElementById("kpiLoadCard"),
    // Table & SVG
    telemetryTableBody: document.getElementById("telemetryTableBody"),
    stateActiveSubBadge: document.getElementById("stateActiveSubBadge"),
    sysStateText: document.getElementById("sysStateText"),
    sysStateSubText: document.getElementById("sysStateSubText"),
    sysStateIcon: document.getElementById("sysStateIcon"),
    
    // AI Panel
    aiCard: document.getElementById("aiCard"),
    aiSubBadge: document.getElementById("aiSubBadge"),
    aiPredState: document.getElementById("aiPredState"),
    aiConfVal: document.getElementById("aiConfVal"),
    aiProgressFill: document.getElementById("aiProgressFill"),
    aiSubName: document.getElementById("aiSubName"),
    aiRelayStatus: document.getElementById("aiRelayStatus"),
    aiActionDesc: document.getElementById("aiActionDesc"),
    aiModelArch: document.getElementById("aiModelArch"),
    aiModelCardTitle: document.getElementById("aiModelCardTitle"),
    chartProbTitle: document.getElementById("chartProbTitle"),
    kpiModelSelect: document.getElementById("kpiModelSelect"),
    aiModelSelector: document.getElementById("aiModelSelector"),
    featureInspectorGrid: document.getElementById("featureInspectorGrid"),
    // Alarms & SOE
    alarmsList: document.getElementById("alarmsList"),
    activeAlarmCountBadge: document.getElementById("activeAlarmCountBadge"),
    soeTableBody: document.getElementById("soeTableBody"),
    soeSearch: document.getElementById("soeSearch"),
    shapSubBadge: document.getElementById("shapSubBadge")
};

// ── INITIALIZATION ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupTabNavigation();
    setupEventListeners();
    initCharts();
    fetchInitialStatus();
    startPolling();
});

function setupTabNavigation() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");
        });
    });
}

function setupEventListeners() {
    el.btnPlayPause.addEventListener("click", () => controlSim("toggle"));
    
    if (el.btnOpenDetailsModal) {
        el.btnOpenDetailsModal.addEventListener("click", () => el.detailsModal.classList.add("active"));
    }
    if (el.btnCloseDetailsModal) {
        el.btnCloseDetailsModal.addEventListener("click", () => el.detailsModal.classList.remove("active"));
    }
    if (el.btnCloseDetailsModalBottom) {
        el.btnCloseDetailsModalBottom.addEventListener("click", () => el.detailsModal.classList.remove("active"));
    }

    el.btnOpenFaultModal.addEventListener("click", () => el.faultModal.classList.add("active"));
    el.btnCloseFaultModal.addEventListener("click", () => el.faultModal.classList.remove("active"));
    el.btnSubmitFault.addEventListener("click", submitFault);
    el.btnClearFaultBtn.addEventListener("click", clearFault);
    el.soeSearch.addEventListener("input", renderSOE);

    if (el.kpiModelSelect) {
        el.kpiModelSelect.addEventListener("change", (e) => switchAIModel(e.target.value));
    }
    if (el.aiModelSelector) {
        el.aiModelSelector.addEventListener("change", (e) => switchAIModel(e.target.value));
    }
}

async function switchAIModel(modelKey) {
    try {
        const res = await fetch("/api/ai/model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelKey })
        });
        const data = await res.json();
        if (data.success && data.model_info) {
            updateModelUI(data.model_info);
        }
        updateDashboard();
    } catch (e) {
        console.error("Error switching AI model:", e);
    }
}

function updateModelUI(mInfo) {
    if (!mInfo) return;
    const key = mInfo.key || "gru_optuna";
    // Sync both selects without triggering change events
    if (el.kpiModelSelect && el.kpiModelSelect.value !== key) {
        el.kpiModelSelect.value = key;
    }
    if (el.aiModelSelector && el.aiModelSelector.value !== key) {
        el.aiModelSelector.value = key;
    }
    if (el.aiModelArch) {
        const f1 = mInfo.val_f1 ? ` — Val F1: ${mInfo.val_f1.toFixed(3)}` : "";
        el.aiModelArch.textContent = `${mInfo.arch}${f1}`;
    }
    if (el.aiModelCardTitle) {
        el.aiModelCardTitle.textContent = `${mInfo.display_name} Fault Diagnosis`;
    }
    if (el.chartProbTitle) {
        el.chartProbTitle.textContent = `Multi-Class Probability Distribution (${mInfo.display_name})`;
    }
}

// ── API POLLING & FETCHING ──────────────────────────────────────────────────
async function fetchInitialStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        state.status = data;
        state.substations = data.substations || [];
        state.selectedSubstation = data.selected_substation || state.substations[0];

        renderSubstationPills();
        populateFaultTargets();
        updateModelUI(state.status.model_info);
    } catch (e) {
        console.error("Error fetching initial status:", e);
    }
}

function startPolling() {
    setInterval(updateDashboard, 1000);
}

async function updateDashboard() {
    try {
        const [statusRes, telRes, aiRes, alarmRes, eventRes, histRes] = await Promise.all([
            fetch("/api/status"),
            fetch("/api/telemetry"),
            fetch("/api/ai_diagnosis"),
            fetch("/api/alarms"),
            fetch("/api/events"),
            fetch(`/api/history?substation=${encodeURIComponent(state.selectedSubstation)}`)
        ]);

        state.status = await statusRes.json();
        state.telemetry = await telRes.json();
        state.aiDiagnosis = await aiRes.json();
        state.alarms = await alarmRes.json();
        state.events = await eventRes.json();
        state.history = await histRes.json();

        // Also fetch SHAP values for selected substation
        try {
            const shapRes = await fetch(`/api/shap?substation=${encodeURIComponent(state.selectedSubstation)}`);
            state.shapData = await shapRes.json();
        } catch(e) { state.shapData = {}; }
        
        // Always keep substations in sync with the latest status response
        const freshSubs = state.status.substations || [];
        if (freshSubs.length > 0) {
            state.substations = freshSubs;
        }
        if (el.faultTargetSelect.options.length === 0 && state.substations.length > 0) {
            populateFaultTargets();
        }

        updateModelUI(state.status.model_info);
        renderStatusHeader();
        renderKPIs();
        renderSubstationPills();
        renderSystemState();
        renderTelemetryTable();
        renderCharts();
        renderAIDiagnosis();
        renderShapChart();
        renderAlarms();
        renderSOE();
    } catch (e) {
        console.error("Error updating dashboard:", e);
    }
}

// ── SIMULATION CONTROL ──────────────────────────────────────────────────────
async function controlSim(action) {
    await fetch("/api/sim/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, substation: state.selectedSubstation })
    });
    updateDashboard();
}

async function submitFault() {
    const target = el.faultTargetSelect.value;
    const type = el.faultTypeSelect.value;
    const duration = parseInt(el.faultDurationInput.value, 10);

    await fetch("/api/fault/inject", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, type, duration })
    });

    el.faultModal.classList.remove("active");
    updateDashboard();
}

async function clearFault() {
    await fetch("/api/fault/clear", { method: "POST" });
    el.faultModal.classList.remove("active");
    updateDashboard();
}

// ── RENDER HELPERS ──────────────────────────────────────────────────────────
function renderStatusHeader() {
    const st = state.status;
    if (st.active_fault && st.active_fault !== "normal") {
        el.statusDot.className = "status-dot red";
        el.statusText.textContent = `FAULT ACTIVE: ${FAULT_LABELS[st.active_fault] || st.active_fault} (${st.fault_target})`;
    } else if (st.running) {
        el.statusDot.className = "status-dot green";
        el.statusText.textContent = "SYSTEM RUNNING";
        el.btnPlayPause.textContent = "⏸ Pause";
    } else {
        el.statusDot.className = "status-dot amber";
        el.statusText.textContent = "SYSTEM PAUSED";
        el.btnPlayPause.textContent = "▶ Run";
    }
}

function renderKPIs() {
    const tel = state.telemetry[state.selectedSubstation] || {};
    el.kpiHv.innerHTML = `${(tel.V_hv_kV || 0).toFixed(2)} <span class="kpi-unit">kV</span>`;
    el.kpiLv.innerHTML = `${(tel.V_lv_kV || 0).toFixed(3)} <span class="kpi-unit">kV</span>`;
    el.kpiCurrent.innerHTML = `${(tel.I_hv_A || 0).toFixed(1)} <span class="kpi-unit">A</span>`;
    el.kpiP.innerHTML = `${(tel.P_MW || 0).toFixed(3)} <span class="kpi-unit">MW</span>`;
    el.kpiQ.innerHTML = `${(tel.Q_Mvar || 0).toFixed(3)} <span class="kpi-unit">Mvar</span>`;
    el.kpiFreq.innerHTML = `${(tel.freq_Hz || 50).toFixed(3)} <span class="kpi-unit">Hz</span>`;

    const load = tel.loading_pct || 0;
    el.kpiLoad.innerHTML = `${load.toFixed(1)} <span class="kpi-unit">%</span>`;
    el.kpiLoadCard.style.borderLeftColor = load > 100 ? "#DC2626" : (load > 80 ? "#D97706" : "#8DC63F");
}

function renderSubstationPills() {
    el.subPills.innerHTML = "";
    state.substations.forEach(sub => {
        const pill = document.createElement("div");
        const tel = state.telemetry[sub] || {};
        const isFault = (tel.label && tel.label !== "normal") || (tel.loading_pct > 100);

        pill.className = `sub-pill ${sub === state.selectedSubstation ? "active" : ""}`;
        pill.innerHTML = `<span class="sub-dot ${isFault ? "fault" : ""}"></span>${sub}`;

        pill.addEventListener("click", () => {
            state.selectedSubstation = sub;
            controlSim("set_substation");
            document.getElementById("trendSelectedSub").textContent = sub;
            updateDashboard();
        });
        el.subPills.appendChild(pill);
    });
}

function populateFaultTargets() {
    el.faultTargetSelect.innerHTML = "";
    state.substations.forEach(sub => {
        const opt = document.createElement("option");
        opt.value = sub;
        opt.textContent = sub;
        if (sub === state.selectedSubstation) opt.selected = true;
        el.faultTargetSelect.appendChild(opt);
    });
}

// ── SYSTEM STATE RENDERING ──────────────────────────────────────────────────
function renderSystemState() {
    if (!el.stateActiveSubBadge) return;
    const sub = state.selectedSubstation;
    el.stateActiveSubBadge.textContent = sub;

    const tel = state.telemetry[sub] || {};
    const aiRes = state.aiDiagnosis[sub] || {};

    // Use simulation ground-truth label (clears instantly when fault ends)
    // Fall back to AI prediction if simulation label unavailable
    const simLabel = tel.label || "normal";
    const aiLabel  = aiRes.pred_label || "normal";
    // Ground truth wins — only show AI label when sim also agrees it's a fault
    const displayLabel = (simLabel !== "normal") ? simLabel
                       : (aiLabel !== "normal" && (aiRes.confidence || 0) > 75) ? aiLabel
                       : "normal";

    if (displayLabel === "normal") {
        el.sysStateText.textContent = "Normal Operation";
        el.sysStateText.style.color = "#006837";
        el.sysStateSubText.textContent = "System operating within nominal parameters.";
        el.sysStateIcon.style.color = "#006837";
        el.sysStateIcon.innerHTML = `<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
    } else {
        const friendlyName = FAULT_LABELS[displayLabel] || displayLabel;
        const sev = AI_SEVERITY[displayLabel] || "WARNING";
        const color = sev === "CRITICAL" ? "#EF4444" : "#F59E0B";
        const srcLabel = (simLabel !== "normal") ? "Simulation" : `AI (${(aiRes.confidence||0).toFixed(1)}% conf)`;

        el.sysStateText.textContent = `FAULT DETECTED: ${friendlyName}`;
        el.sysStateText.style.color = color;
        el.sysStateSubText.textContent = `Source: ${srcLabel} — Substation: ${sub}`;
        el.sysStateIcon.style.color = color;
        el.sysStateIcon.innerHTML = `<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
    }
}

// ── TELEMETRY TABLE ─────────────────────────────────────────────────────────
function renderTelemetryTable() {
    let rowsHtml = "";

    // Use substations list; fall back to whatever keys telemetry returned
    const subs = (state.substations && state.substations.length > 0)
        ? state.substations
        : Object.keys(state.telemetry);

    if (subs.length === 0) {
        el.telemetryTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#475569;padding:20px;">Waiting for telemetry data…</td></tr>`;
        return;
    }

    subs.forEach(sub => {
        const tel = state.telemetry[sub] || {};
        const isSelected = sub === state.selectedSubstation;
        const lbl = tel.label || "normal";
        const load = tel.loading_pct || 0;

        let tagClass = "tag-normal";
        if (lbl !== "normal") tagClass = "tag-fault";
        else if (load > 80) tagClass = "tag-warn";

        rowsHtml += `
            <tr style="${isSelected ? 'background: #E2E8F0;' : ''}">
                <td style="font-weight: 700; color: ${isSelected ? '#006837' : '#0F172A'};">${sub}</td>
                <td>${(tel.V_hv_kV || 0).toFixed(2)}</td>
                <td>${(tel.V_lv_kV || 0).toFixed(3)}</td>
                <td>${(tel.I_hv_A || 0).toFixed(1)}</td>
                <td>${(tel.P_MW || 0).toFixed(3)}</td>
                <td>${(tel.Q_Mvar || 0).toFixed(3)}</td>
                <td style="color: ${load > 100 ? '#DC2626' : '#0F172A'};">${load.toFixed(1)}%</td>
                <td>${(tel.freq_Hz || 50).toFixed(3)}</td>
                <td class="${tagClass}">${FAULT_LABELS[lbl] || lbl}</td>
            </tr>
        `;
    });

    el.telemetryTableBody.innerHTML = rowsHtml;
}

// ── CHARTS ──────────────────────────────────────────────────────────────────
function initCharts() {
    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: '#475569', font: { size: 10 } } } },
        scales: {
            x: { ticks: { color: '#334155', font: { size: 9 } }, grid: { color: '#E2E8F0' } },
            y: { ticks: { color: '#475569', font: { size: 10 } }, grid: { color: '#E2E8F0' } }
        }
    };

    // Voltage
    state.charts.voltage = new Chart(document.getElementById("chartVoltage"), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'HV Bus (kV)', borderColor: '#006837', borderWidth: 2, data: [], pointRadius: 0 },
            { label: 'LV Bus (kV)', borderColor: '#008B46', borderWidth: 2, data: [], pointRadius: 0 }
        ]},
        options: commonOpts
    });

    // Current & Load
    state.charts.current = new Chart(document.getElementById("chartCurrent"), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Current (A)', borderColor: '#E11D48', borderWidth: 2, data: [], pointRadius: 0 },
            { label: 'Loading (%)', borderColor: '#D97706', borderWidth: 2, data: [], pointRadius: 0 }
        ]},
        options: commonOpts
    });

    // Power
    state.charts.power = new Chart(document.getElementById("chartPower"), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'P Active (MW)', borderColor: '#8DC63F', borderWidth: 2, data: [], pointRadius: 0 },
            { label: 'Q Reactive (Mvar)', borderColor: '#B4D43D', borderWidth: 2, data: [], pointRadius: 0 }
        ]},
        options: commonOpts
    });

    // Frequency
    state.charts.freq = new Chart(document.getElementById("chartFreq"), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Frequency (Hz)', borderColor: '#005C3A', borderWidth: 2, data: [], pointRadius: 0 }
        ]},
        options: commonOpts
    });

    // AI Probabilities Bar Chart
    state.charts.probs = new Chart(document.getElementById("chartProbabilities"), {
        type: 'bar',
        data: {
            labels: ["Normal", "LG Fault", "LL Fault", "Overload", "Voltage Sag", "Under-Freq", "Over-Freq", "Trafo Trip"],
            datasets: [{ label: 'Model Probability (%)', backgroundColor: '#006837', data: [100, 0, 0, 0, 0, 0, 0, 0] }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#475569', font: { size: 9 } }, grid: { display: false } },
                y: { min: 0, max: 100, ticks: { color: '#475569' }, grid: { color: '#E2E8F0' } }
            }
        }
    });

    // SHAP waterfall horizontal bar chart
    state.charts.shap = new Chart(document.getElementById("chartShap"), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'SHAP value',
                data: [],
                backgroundColor: [],
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` SHAP: ${ctx.parsed.x.toFixed(4)}  (${ctx.parsed.x >= 0 ? 'pushes toward' : 'pushes away from'} prediction)`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'SHAP value (impact on model output)', color: '#475569', font: { size: 10 } },
                    ticks: { color: '#334155' },
                    grid: { color: '#E2E8F0' }
                },
                y: { ticks: { color: '#0F172A', font: { size: 11, weight: '600' } }, grid: { display: false } }
            }
        }
    });
}

function renderCharts() {
    const hist = state.history || [];
    const labels = hist.map(h => `t+${h.t}`);

    // Update Voltage
    state.charts.voltage.data.labels = labels;
    state.charts.voltage.data.datasets[0].data = hist.map(h => h.V_hv_kV);
    state.charts.voltage.data.datasets[1].data = hist.map(h => h.V_lv_kV);
    state.charts.voltage.update();

    // Update Current
    state.charts.current.data.labels = labels;
    state.charts.current.data.datasets[0].data = hist.map(h => h.I_hv_A);
    state.charts.current.data.datasets[1].data = hist.map(h => h.loading_pct);
    state.charts.current.update();

    // Update Power
    state.charts.power.data.labels = labels;
    state.charts.power.data.datasets[0].data = hist.map(h => h.P_MW);
    state.charts.power.data.datasets[1].data = hist.map(h => h.Q_Mvar);
    state.charts.power.update();

    // Update Frequency
    state.charts.freq.data.labels = labels;
    state.charts.freq.data.datasets[0].data = hist.map(h => h.freq_Hz);
    state.charts.freq.update();
}

// ── SHAP CHART ──────────────────────────────────────────────────────────────
function renderShapChart() {
    const sd = state.shapData || {};
    const values = sd.values || {};
    const sub = sd.substation || state.selectedSubstation;

    if (el.shapSubBadge) el.shapSubBadge.textContent = sub;

    if (!Object.keys(values).length) return;

    // Sort features by absolute SHAP value descending, take top 10
    const sorted = Object.entries(values)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10)
        .reverse();  // reverse so largest is on top in horizontal bar

    const labels = sorted.map(([k]) => k);
    const data   = sorted.map(([, v]) => v);
    const colors = data.map(v => v >= 0 ? 'rgba(0,104,55,0.80)' : 'rgba(220,38,38,0.75)');

    state.charts.shap.data.labels = labels;
    state.charts.shap.data.datasets[0].data = data;
    state.charts.shap.data.datasets[0].backgroundColor = colors;
    state.charts.shap.update();
}

// ── AI DIAGNOSIS ─────────────────────────────────────────────────────────────
function renderAIDiagnosis() {
    const sub = state.selectedSubstation;
    const aiRes = state.aiDiagnosis[sub] || {};
    const tel = state.telemetry[sub] || {};

    const predLabel = aiRes.pred_label || "normal";
    const conf = aiRes.confidence || 100.0;
    const sev = AI_SEVERITY[predLabel] || "NONE";

    el.aiCard.className = `card ai-card ${sev.toLowerCase()}`;
    el.aiSubBadge.textContent = sub;
    el.aiPredState.textContent = FAULT_LABELS[predLabel] || predLabel;
    el.aiConfVal.textContent = `${conf.toFixed(1)}%`;
    el.aiProgressFill.style.width = `${conf}%`;
    el.aiSubName.textContent = sub;
    el.aiRelayStatus.textContent = tel.relay_status || "NORMAL";
    el.aiActionDesc.textContent = AI_RECOMMENDATIONS[predLabel] || "No protective action required.";

    // Update Probability Bar Chart
    if (aiRes.top_probs) {
        const probMap = aiRes.top_probs;
        const labels = ["normal", "lg_fault", "ll_fault", "overload", "voltage_sag", "under_frequency", "over_frequency", "transformer_trip"];
        const vals = labels.map(l => probMap[l] || 0.0);
        state.charts.probs.data.datasets[0].data = vals;
        state.charts.probs.update();
    }

    // Feature Inspector
    renderFeatureInspector(tel);
}

function renderFeatureInspector(tel) {
    const vHv = tel.V_hv_kV || 0;
    const vLv = tel.V_lv_kV || 0;
    const iHv = tel.I_hv_A || 0;
    const pMw = tel.P_MW || 0;
    const qMvar = tel.Q_Mvar || 0;
    const sMva = tel.S_MVA || 0;
    const pf = tel.PF || 0.95;
    const load = tel.loading_pct || 0;
    const freq = tel.freq_Hz || 50;

    // Computed per-unit ratios (estimated from telemetry)
    const vHvPu = (vHv / 60.0).toFixed(4);
    const vLvPu = (vLv / 5.5).toFixed(4);
    const vRatio = (vHv > 0 ? (vLv/5.5) / (vHv/60.0 + 1e-4) : 0).toFixed(4);
    const deltaVpu = ((vLv/5.5) - (vHv/60.0)).toFixed(4);

    const feats = [
        // Raw measurements
        { label: "V_hv_kV", val: vHv.toFixed(2), cat: "raw" },
        { label: "V_lv_kV", val: vLv.toFixed(3), cat: "raw" },
        { label: "I_hv_A", val: iHv.toFixed(1), cat: "raw" },
        { label: "P_MW", val: pMw.toFixed(3), cat: "raw" },
        { label: "Q_Mvar", val: qMvar.toFixed(3), cat: "raw" },
        { label: "S_MVA", val: sMva.toFixed(3), cat: "raw" },
        { label: "PF", val: pf.toFixed(3), cat: "raw" },
        { label: "loading_pct", val: load.toFixed(1) + "%", cat: "raw" },
        { label: "freq_Hz", val: freq.toFixed(3), cat: "raw" },
        { label: "Δfreq_Hz", val: (tel.delta_freq_Hz || 0).toFixed(4), cat: "raw" },
        { label: "ΔV_hv_kV", val: (tel.delta_V_hv || 0).toFixed(4), cat: "raw" },
        // Per-unit ratios (E1/E2 features)
        { label: "V_hv_pu", val: vHvPu, cat: "ratio" },
        { label: "V_lv_pu", val: vLvPu, cat: "ratio" },
        { label: "V_ratio", val: vRatio, cat: "ratio" },
        { label: "ΔV_pu", val: deltaVpu, cat: "ratio" },
        { label: "P_ratio", val: (tel.p_ratio || "—"), cat: "ratio" },
        { label: "PΔ_ratio", val: (tel.p_delta_ratio || "—"), cat: "ratio" },
        { label: "I_ratio", val: (tel.i_ratio || "—"), cat: "ratio" },
        { label: "S_ratio", val: (tel.s_ratio || "—"), cat: "ratio" },
        // Dynamic features
        { label: "ΔI_hv_A", val: (tel.delta_I_hv || 0).toFixed(3), cat: "dyn" },
        { label: "|ΔI_hv|", val: (tel.abs_delta_I_hv || 0).toFixed(3), cat: "dyn" },
        { label: "ΔP_MW", val: (tel.delta_P_MW || 0).toFixed(4), cat: "dyn" },
        { label: "ΔS_MVA", val: (tel.delta_S_MVA || 0).toFixed(4), cat: "dyn" },
        { label: "I²t_pu", val: (tel.i2t_pu || "—"), cat: "dyn" },
        // OHE substation
        { label: "Substation", val: (state.selectedSubstation || "").split(" ")[0], cat: "ohe" },
    ];

    const catColor = { raw: "#006837", ratio: "#0056AD", dyn: "#B45309", ohe: "#6B7280" };
    const catLabel = { raw: "Raw", ratio: "Ratio", dyn: "Dynamic", ohe: "OHE" };

    let gridHtml = "";
    feats.forEach(f => {
        const col = catColor[f.cat] || "#334155";
        gridHtml += `
            <div class="feature-tile" style="border-top: 3px solid ${col};">
                <div class="ft-lbl" style="color:${col};">${f.label}</div>
                <div class="ft-val">${f.val}</div>
            </div>
        `;
    });

    el.featureInspectorGrid.innerHTML = gridHtml;
}

// ── ALARMS & SOE ────────────────────────────────────────────────────────────
function renderAlarms() {
    const alarms = state.alarms || [];
    el.activeAlarmCountBadge.textContent = `${alarms.length} Active`;

    if (!alarms.length) {
        el.alarmsList.innerHTML = `<div class="empty-state">No active alarms. System operating normally.</div>`;
        return;
    }

    let html = "";
    alarms.slice(0, 15).forEach(a => {
        html += `
            <div class="alarm-item ${a.priority}">
                <div class="alarm-hdr">
                    <span class="alarm-time">${a.time}</span>
                    <span class="alarm-pri ${a.priority}">${a.priority}</span>
                </div>
                <div class="alarm-sub">${a.sub}</div>
                <div class="alarm-msg">${a.msg}</div>
            </div>
        `;
    });
    el.alarmsList.innerHTML = html;
}

function renderSOE() {
    const events = state.events || [];
    const filter = el.soeSearch.value.toLowerCase();

    const filtered = events.filter(e => e.event.toLowerCase().includes(filter));

    if (!filtered.length) {
        el.soeTableBody.innerHTML = `<tr><td colspan="2" style="text-align: center; color: #334155;">No events logged.</td></tr>`;
        return;
    }

    let html = "";
    filtered.slice(0, 40).forEach(e => {
        html += `
            <tr>
                <td class="soe-time">${e.time}</td>
                <td>${e.event}</td>
            </tr>
        `;
    });
    el.soeTableBody.innerHTML = html;
}
