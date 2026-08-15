import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const EMPTY_FILTERS = {
  risk_categories: [],
  departments: [],
  employee_types: [],
  genders: [],
  search: "",
};

const TABS = [
  ["risk", "◆ Attrition Risk"],
  ["explain", "✦ Explainable AI"],
  ["department", "▥ Department Risk Analysis"],
  ["engagement", "☻ Satisfaction & Engagement"],
  ["performance", "★ Performance & Tenure"],
];

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function avg(values) {
  const a = values.map(num).filter((v) => v !== null);
  return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
}

function App() {
  const [meta, setMeta] = useState({
    risk_levels: [],
    departments: [],
    employee_types: [],
    genders: [],
  });
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [rows, setRows] = useState([]);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [activeTab, setActiveTab] = useState("risk");
  const [recommendation, setRecommendation] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion] = useState("");
  const [loadingRecommendation, setLoadingRecommendation] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_BASE}/metadata`)
      .then((res) => setMeta(res.data))
      .catch((err) => console.error("Metadata error:", err));
  }, []);

  useEffect(() => {
    loadPredictions();
  }, [filters]);

  useEffect(() => {
    if (!selectedEmployee) {
      setExplainData(null);
      return;
    }
    loadExplainability(selectedEmployee);
  }, [selectedEmployee]);

  const loadPredictions = async () => {
    try {
      const params = new URLSearchParams();
      filters.risk_categories.forEach((v) => params.append("risk_categories", v));
      filters.departments.forEach((v) => params.append("departments", v));
      filters.employee_types.forEach((v) => params.append("employee_types", v));
      filters.genders.forEach((v) => params.append("genders", v));
      if (filters.search.trim()) params.append("search", filters.search.trim());

      const res = await axios.get(`${API_BASE}/predictions?${params.toString()}`);
      const records = res.data.records || [];
      setRows(records);

      // Do not auto-select the first employee. Keep only a manual selection
      // that is still present after filtering.
      setSelectedEmployee((current) => {
        if (!current) return null;
        return records.some(
          (r) => String(r.employeeid) === String(current.employeeid)
        )
          ? current
          : null;
      });
    } catch (error) {
      console.error("Prediction loading error:", error);
      setRows([]);
    }
  };

  const loadExplainability = async (employee) => {
    setExplainLoading(true);
    setExplainData(null);
    try {
      // If your backend already exposes a real SHAP endpoint, use it.
      const res = await axios.get(
        `${API_BASE}/explain/${encodeURIComponent(employee.employeeid)}`
      );
      setExplainData({ ...res.data, source: "SHAP" });
    } catch (error) {
      // The current backend does not expose a SHAP endpoint. We therefore
      // calculate transparent data-driven drivers from the employee values
      // relative to the currently filtered workforce. This is NOT SHAP.
      setExplainData({ source: "DATA", factors: buildDataDrivers(employee, rows) });
    } finally {
      setExplainLoading(false);
    }
  };

  const riskCounts = useMemo(() => {
    const c = { "High Risk": 0, "Medium Risk": 0, "Low Risk": 0 };
    rows.forEach((r) => {
      if (c[r.risk_category] !== undefined) c[r.risk_category]++;
    });
    return c;
  }, [rows]);

  const avgSatisfaction = useMemo(
    () => avg(rows.map((r) => r.satisfactionscore)),
    [rows]
  );
  const avgEngagement = useMemo(
    () => avg(rows.map((r) => r.engagementscore)),
    [rows]
  );
  const avgPerformance = useMemo(
    () => avg(rows.map((r) => r.performance_rating_numeric ?? r.performancescore)),
    [rows]
  );
  const avgTenure = useMemo(() => avg(rows.map((r) => r.tenure_years)), [rows]);
  const avgRisk = useMemo(
    () => avg(rows.map((r) => r.attrition_risk_score)),
    [rows]
  );

  const departmentData = useMemo(() => {
    const map = {};
    rows.forEach((r) => {
      const d = r.department_clean || r.departmenttype || "Unknown";
      if (!map[d]) map[d] = { name: d, count: 0, risk: [], high: 0, medium: 0, low: 0 };
      map[d].count++;
      const risk = num(r.attrition_risk_score);
      if (risk !== null) map[d].risk.push(risk);
      if (r.risk_category === "High Risk") map[d].high++;
      if (r.risk_category === "Medium Risk") map[d].medium++;
      if (r.risk_category === "Low Risk") map[d].low++;
    });
    return Object.values(map)
      .map((d) => ({ ...d, avgRisk: avg(d.risk) }))
      .sort((a, b) => b.avgRisk - a.avgRisk);
  }, [rows]);

  const employeePayload = () => {
    if (!selectedEmployee) return null;
    return {
      employee_data: {
        employee_id: selectedEmployee.employeeid,
        name: `${selectedEmployee.firstname || ""} ${selectedEmployee.lastname || ""}`.trim(),
        department: selectedEmployee.department_clean,
        title: selectedEmployee.title,
        gender: selectedEmployee.gender_clean || selectedEmployee.gender,
        employee_type: selectedEmployee.employeetype_clean || selectedEmployee.employeetype,
        satisfaction_score: selectedEmployee.satisfactionscore,
        engagement_score: selectedEmployee.engagementscore,
        work_life_balance: selectedEmployee.worklifebalancescore,
        performance_rating:
          selectedEmployee.performance_rating_numeric ?? selectedEmployee.performancescore,
        tenure_years: selectedEmployee.tenure_years,
      },
      risk_category: selectedEmployee.risk_category,
      risk_score: selectedEmployee.attrition_risk_score,
    };
  };

  const selectEmployee = (employee) => {
    setSelectedEmployee(employee);
    setRecommendation("");
    setChatHistory([]);
    setActiveTab("explain");
  };

  const clearFilters = () => setFilters({ ...EMPTY_FILTERS });

  const handleRecommendation = async () => {
    if (!selectedEmployee) return;
    try {
      setLoadingRecommendation(true);
      const res = await axios.post(`${API_BASE}/ai/recommendation`, employeePayload());
      setRecommendation(res.data.recommendation || "No recommendation returned.");
    } catch (error) {
      console.error(error);
      setRecommendation("Unable to generate the HR recommendation.");
    } finally {
      setLoadingRecommendation(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!selectedEmployee || !question.trim()) return;
    const q = question.trim();
    setQuestion("");
    setChatHistory((prev) => [...prev, { role: "user", content: q }]);
    try {
      setLoadingChat(true);
      const res = await axios.post(`${API_BASE}/ai/chat`, {
        ...employeePayload(),
        question: q,
      });
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: res.data.answer || "No answer returned." },
      ]);
    } catch (error) {
      console.error(error);
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I could not connect to the HR assistant." },
      ]);
    } finally {
      setLoadingChat(false);
    }
  };

  return (
    <div style={styles.app}>
      <header style={styles.topBar}>
        <div>
          <div style={styles.brand}>AI WORKFORCE</div>
          <div style={styles.brandSub}>Analytics & Predictive Intelligence</div>
        </div>
        <div style={styles.status}><span style={styles.statusDot} /> Backend Connected</div>
      </header>

      <div style={styles.layout}>
        <aside style={styles.sidebar}>
          <div style={styles.sidebarTitle}>Workforce Filters</div>
          <div style={styles.sidebarSubtitle}>Narrow the workforce and inspect employees individually.</div>

          <FilterGroup label="Predictive Attrition Risk" options={meta.risk_levels} value={filters.risk_categories}
            onChange={(v) => setFilters((p) => ({ ...p, risk_categories: v }))} />
          <FilterGroup label="Department" options={meta.departments} value={filters.departments}
            onChange={(v) => setFilters((p) => ({ ...p, departments: v }))} />
          <FilterGroup label="Employee Type" options={meta.employee_types} value={filters.employee_types}
            onChange={(v) => setFilters((p) => ({ ...p, employee_types: v }))} />
          <FilterGroup label="Gender" options={meta.genders} value={filters.genders}
            onChange={(v) => setFilters((p) => ({ ...p, genders: v }))} />

          <div style={styles.filterLabel}>Search Employee</div>
          <input value={filters.search} onChange={(e) => setFilters((p) => ({ ...p, search: e.target.value }))}
            placeholder="Name or department" style={styles.searchInput} />
          <button onClick={clearFilters} style={styles.clearButton}>Clear All Filters</button>

          <div style={styles.sidebarInfo}>
            <b>How it works</b>
            <div>1. Apply filters</div>
            <div>2. Open an employee</div>
            <div>3. Explore every analytics tab</div>
            <div>4. Ask the AI HR Assistant</div>
          </div>
        </aside>

        <main style={styles.main}>
          <section style={styles.hero}>
            <div style={styles.heroTitle}>AI Workforce Analytics & Predictive Intelligence</div>
            <div style={styles.heroSubtitle}>ML-powered attrition prediction, explainable workforce insights and an AI HR assistant.</div>
          </section>

          <section style={styles.kpiGrid}>
            <KpiCard label="Total Workforce" value={rows.length.toLocaleString()} />
            <KpiCard label="High-Risk Employees" value={riskCounts["High Risk"].toLocaleString()} accent="danger" />
            <KpiCard label="Medium-Risk Employees" value={riskCounts["Medium Risk"].toLocaleString()} accent="warning" />
            <KpiCard label="Low-Risk Employees" value={riskCounts["Low Risk"].toLocaleString()} accent="success" />
            <KpiCard label="Avg Satisfaction" value={`${avgSatisfaction.toFixed(2)} / 5`} />
            <KpiCard label="Avg Performance" value={`${avgPerformance.toFixed(2)} / 4`} />
            <KpiCard label="Avg Tenure" value={`${avgTenure.toFixed(1)} yrs`} />
          </section>

          <div style={styles.navStrip}>
            {TABS.map(([id, label]) => (
              <button key={id} onClick={() => setActiveTab(id)} style={activeTab === id ? styles.navButtonActive : styles.navButton}>{label}</button>
            ))}
          </div>

          {activeTab === "risk" && (
            <RiskTab rows={rows} riskCounts={riskCounts} selectedEmployee={selectedEmployee} onSelect={selectEmployee} />
          )}

          {activeTab === "department" && (
            <DepartmentTab data={departmentData} />
          )}

          {activeTab === "engagement" && (
            <EngagementTab rows={rows} avgSatisfaction={avgSatisfaction} avgEngagement={avgEngagement} />
          )}

          {activeTab === "performance" && (
            <PerformanceTab rows={rows} avgPerformance={avgPerformance} avgTenure={avgTenure} />
          )}

          {activeTab === "explain" && (
            <ExplainTab employee={selectedEmployee} data={explainData} loading={explainLoading}
              onRecommendation={handleRecommendation} recommendation={recommendation} loadingRecommendation={loadingRecommendation}
              chatHistory={chatHistory} question={question} setQuestion={setQuestion} onAsk={handleAskQuestion} loadingChat={loadingChat} />
          )}
        </main>
      </div>
    </div>
  );
}

function RiskTab({ rows, riskCounts, selectedEmployee, onSelect }) {
  return (
    <section style={styles.twoColumn}>
      <div style={styles.panel}>
        <PanelTitle title="Risk Category Breakdown" />
        <Plot data={[{ values: Object.values(riskCounts), labels: Object.keys(riskCounts), type: "pie", hole: 0.58,
          textinfo: "percent", hoverinfo: "label+value+percent" }]}
          layout={{ autosize: true, height: 360, margin: { l: 10, r: 10, t: 15, b: 25 }, paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dbeafe" }, legend: { orientation: "h", y: -0.02 } }}
          config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
      </div>
      <EmployeeDirectory rows={rows} selectedEmployee={selectedEmployee} onSelect={onSelect} />
    </section>
  );
}

function EmployeeDirectory({ rows, selectedEmployee, onSelect }) {
  return (
    <div style={styles.panel}>
      <PanelTitle title={`Individual Employee Attrition Risk Directory (${rows.length})`} />
      <div style={styles.tableHint}>Click any employee to open the profile. The profile is never auto-selected.</div>
      <div style={styles.tableWrap}>
        <table style={styles.table}><thead><tr>
          <th style={styles.th}>ID</th><th style={styles.th}>Name</th><th style={styles.th}>Department</th><th style={styles.th}>Risk</th><th style={styles.th}>Score</th>
        </tr></thead><tbody>
          {rows.map((row) => <tr key={row.employeeid} onClick={() => onSelect(row)} style={{ ...styles.tr,
            background: String(selectedEmployee?.employeeid) === String(row.employeeid) ? "#24334a" : "transparent" }}>
            <td style={styles.td}>{row.employeeid}</td>
            <td style={styles.td}>{row.firstname} {row.lastname}</td>
            <td style={styles.td}>{row.department_clean || row.departmenttype}</td>
            <td style={styles.td}><RiskBadge risk={row.risk_category} /></td>
            <td style={styles.td}>{num(row.attrition_risk_score)?.toFixed(2) ?? "N/A"}</td>
          </tr>)}
          {!rows.length && <tr><td colSpan="5" style={styles.emptyCell}>No employees match the filters.</td></tr>}
        </tbody></table>
      </div>
    </div>
  );
}

function DepartmentTab({ data }) {
  const top = data.slice(0, 20);
  return <section style={styles.stack}>
    <div style={styles.panel}>
      <PanelTitle title="Department Risk Analysis" />
      <div style={styles.chartGrid}>
        <Plot data={[{ x: top.map((d) => d.name), y: top.map((d) => d.avgRisk), type: "bar", marker: { color: "#38bdf8" },
          hovertemplate: "%{x}<br>Average risk: %{y:.3f}<extra></extra>" }]}
          layout={barLayout("Average Attrition Risk by Department", 430)} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
        <Plot data={[
          { x: top.map((d) => d.name), y: top.map((d) => d.high), name: "High Risk", type: "bar" },
          { x: top.map((d) => d.name), y: top.map((d) => d.medium), name: "Medium Risk", type: "bar" },
          { x: top.map((d) => d.name), y: top.map((d) => d.low), name: "Low Risk", type: "bar" },
        ]} layout={{ ...barLayout("Risk Distribution by Department", 430), barmode: "stack" }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
      </div>
    </div>
    <div style={styles.panel}>
      <PanelTitle title="Department Summary" />
      <div style={styles.summaryTable}>
        {top.map((d) => <div key={d.name} style={styles.summaryRow}>
          <span>{d.name}</span><span>{d.count} employees</span><RiskBadge risk={d.avgRisk >= 0.66 ? "High Risk" : d.avgRisk >= 0.33 ? "Medium Risk" : "Low Risk"} />
        </div>)}
      </div>
    </div>
  </section>;
}

function EngagementTab({ rows, avgSatisfaction, avgEngagement }) {
  const grouped = groupAverage(rows, "department_clean", ["satisfactionscore", "engagementscore"]);
  return <section style={styles.stack}>
    <div style={styles.kpiMiniGrid}>
      <KpiCard label="Average Satisfaction" value={`${avgSatisfaction.toFixed(2)} / 5`} />
      <KpiCard label="Average Engagement" value={`${avgEngagement.toFixed(2)} / 5`} />
    </div>
    <div style={styles.panel}>
      <PanelTitle title="Satisfaction & Engagement by Department" />
      <Plot data={[
        { x: grouped.map((d) => d.name), y: grouped.map((d) => d.satisfaction), name: "Satisfaction", type: "bar" },
        { x: grouped.map((d) => d.name), y: grouped.map((d) => d.engagement), name: "Engagement", type: "bar" },
      ]} layout={{ ...barLayout("Department Comparison", 450), barmode: "group", yaxis: { title: "Score", range: [0, 5] } }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
    </div>
    <div style={styles.panel}>
      <PanelTitle title="Employee Satisfaction vs Engagement" />
      <Plot data={[{ x: rows.map((r) => num(r.satisfactionscore)), y: rows.map((r) => num(r.engagementscore)), mode: "markers",
        type: "scatter", text: rows.map((r) => `${r.firstname || ""} ${r.lastname || ""}`), hovertemplate: "%{text}<br>Satisfaction: %{x}<br>Engagement: %{y}<extra></extra>" }]}
        layout={{ ...baseLayout(430), xaxis: { title: "Satisfaction Score", range: [0, 5] }, yaxis: { title: "Engagement Score", range: [0, 5] } }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
    </div>
  </section>;
}

function PerformanceTab({ rows, avgPerformance, avgTenure }) {
  return <section style={styles.stack}>
    <div style={styles.kpiMiniGrid}>
      <KpiCard label="Average Performance" value={`${avgPerformance.toFixed(2)} / 4`} />
      <KpiCard label="Average Tenure" value={`${avgTenure.toFixed(1)} yrs`} />
    </div>
    <div style={styles.panel}>
      <PanelTitle title="Performance & Tenure Analysis" />
      <Plot data={[{ x: rows.map((r) => num(r.tenure_years)), y: rows.map((r) => num(r.performance_rating_numeric ?? r.performancescore)), mode: "markers",
        type: "scatter", text: rows.map((r) => `${r.firstname || ""} ${r.lastname || ""}`), marker: { size: 7 },
        hovertemplate: "%{text}<br>Tenure: %{x:.1f} yrs<br>Performance: %{y}<extra></extra>" }]}
        layout={{ ...baseLayout(450), xaxis: { title: "Tenure (years)" }, yaxis: { title: "Performance Rating", range: [0, 4] } }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
    </div>
    <div style={styles.panel}>
      <PanelTitle title="Risk vs Tenure" />
      <Plot data={[{ x: rows.map((r) => num(r.tenure_years)), y: rows.map((r) => num(r.attrition_risk_score)), mode: "markers",
        type: "scatter", text: rows.map((r) => `${r.firstname || ""} ${r.lastname || ""}`), marker: { size: 7 },
        hovertemplate: "%{text}<br>Tenure: %{x:.1f} yrs<br>Risk: %{y:.3f}<extra></extra>" }]}
        layout={{ ...baseLayout(430), xaxis: { title: "Tenure (years)" }, yaxis: { title: "Attrition Risk Score" } }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} />
    </div>
  </section>;
}

function ExplainTab({ employee, data, loading, onRecommendation, recommendation, loadingRecommendation, chatHistory, question, setQuestion, onAsk, loadingChat }) {
  if (!employee) return <EmptyState title="Select an employee first" text="Open the Attrition Risk tab and click any employee. Then return here to see the employee-level explanation and use the AI HR Assistant." />;
  const factors = data?.factors || [];
  return <section style={styles.stack}>
    <div style={styles.panel}>
      <div style={styles.profileHeader}>
        <div><div style={styles.panelTitle}>Explainable AI — {employee.firstname} {employee.lastname}</div>
          <div style={styles.tableHint}>Risk: {employee.risk_category} · Score: {num(employee.attrition_risk_score)?.toFixed(4) ?? "N/A"}</div></div>
        <RiskBadge risk={employee.risk_category} />
      </div>
      {loading ? <div style={styles.loading}>Calculating explanation...</div> : (
        <>
          <div style={styles.explainSource}>{data?.source === "SHAP" ? "✓ Real SHAP explanation returned by backend" : "ℹ Data-driven explanation from employee values. Add /explain/{employeeid} to expose true SHAP values."}</div>
          {factors.length ? factors.map((f) => <div key={f.label} style={styles.factorRow}>
            <div style={styles.factorName}>{f.label}<span>{f.value}</span></div>
            <div style={styles.factorTrack}><div style={{ ...styles.factorBar, width: `${Math.min(100, Math.max(4, f.impact))}%`, background: f.direction === "risk" ? "#fb7185" : "#4ade80" }} /></div>
            <div style={styles.factorReason}>{f.reason}</div>
          </div>) : <div style={styles.emptyCell}>No explanation factors available.</div>}
        </>
      )}
    </div>

    <div style={styles.employeeGrid}>
      <div style={styles.profileCard}>
        <PanelTitle title="Employee Risk Profile" />
        <div style={styles.detailGrid}>
          <Detail label="Department" value={employee.department_clean} /><Detail label="Job Title" value={employee.title} />
          <Detail label="Satisfaction" value={employee.satisfactionscore} /><Detail label="Engagement" value={employee.engagementscore} />
          <Detail label="Work-Life Balance" value={employee.worklifebalancescore} /><Detail label="Performance" value={employee.performance_rating_numeric ?? employee.performancescore} />
          <Detail label="Tenure" value={`${employee.tenure_years ?? "N/A"} yrs`} /><Detail label="Employee Type" value={employee.employeetype_clean ?? employee.employeetype} />
        </div>
        <button onClick={onRecommendation} disabled={loadingRecommendation} style={styles.primaryButton}>{loadingRecommendation ? "Generating..." : "Generate AI HR Recommendation"}</button>
        {recommendation && <div style={styles.recommendationBox}><div style={styles.boxTitle}>AI HR Recommendation</div><div style={styles.aiText}>{recommendation}</div></div>}
      </div>

      <AssistantCard employee={employee} chatHistory={chatHistory} question={question} setQuestion={setQuestion} onAsk={onAsk} loadingChat={loadingChat} />
    </div>
  </section>;
}

function AssistantCard({ employee, chatHistory, question, setQuestion, onAsk, loadingChat }) {
  return <div style={styles.assistantCard}>
    <div style={styles.assistantHeader}><div style={styles.assistantIcon}>✦</div><div><div style={styles.assistantTitle}>AI HR Assistant</div><div style={styles.assistantStatus}>Context: {employee.firstname} {employee.lastname}</div></div></div>
    <div style={styles.contextBox}><b>{employee.risk_category}</b> · risk {num(employee.attrition_risk_score)?.toFixed(2) ?? "N/A"}<div style={styles.contextSmall}>{employee.department_clean} · Satisfaction {employee.satisfactionscore} · Engagement {employee.engagementscore}</div></div>
    <div style={styles.chatBox}>
      {!chatHistory.length ? <div style={styles.chatEmpty}>Ask why this employee is at risk, what HR should do, or how to improve retention.</div> : chatHistory.map((m, i) => <div key={i} style={{ ...styles.message, alignSelf: m.role === "user" ? "flex-end" : "flex-start", background: m.role === "user" ? "#075985" : "#1e293b" }}><div style={styles.messageRole}>{m.role === "user" ? "You" : "AI HR Assistant"}</div><div style={styles.aiText}>{m.content}</div></div>)}
      {loadingChat && <div style={styles.typing}>Assistant is thinking...</div>}
    </div>
    <div style={styles.quickQuestions}>{["Why is this employee at risk?", "What should HR do?", "How can we improve retention?"].map((q) => <button key={q} onClick={() => setQuestion(q)} style={styles.quickButton}>{q}</button>)}</div>
    <div style={styles.chatInputRow}><textarea value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onAsk(); } }} rows={3} placeholder="Ask the AI HR assistant..." style={styles.chatInput} /><button onClick={onAsk} disabled={loadingChat || !question.trim()} style={{ ...styles.sendButton, opacity: loadingChat || !question.trim() ? .5 : 1 }}>Send</button></div>
  </div>;
}

function buildDataDrivers(employee, cohort) {
  const definitions = [
    ["Satisfaction", "satisfactionscore", 5, "Low satisfaction can increase attrition risk."],
    ["Engagement", "engagementscore", 5, "Low engagement can increase attrition risk."],
    ["Work-Life Balance", "worklifebalancescore", 5, "Lower work-life balance can increase workforce risk."],
    ["Performance", "performance_rating_numeric", 4, "Performance is included in the model feature set."],
    ["Tenure", "tenure_years", null, "Tenure is included in the model feature set."],
  ];
  return definitions.map(([label, key, max, baseReason]) => {
    const value = num(employee[key] ?? (key === "performance_rating_numeric" ? employee.performancescore : null));
    const peers = cohort.map((r) => num(r[key] ?? (key === "performance_rating_numeric" ? r.performancescore : null))).filter((v) => v !== null);
    const peerAvg = avg(peers);
    let direction = "neutral", impact = 15, reason = `${baseReason} Cohort average: ${peerAvg.toFixed(2)}.`;
    if (value !== null && peerAvg > 0) {
      const diff = value - peerAvg;
      const lowIsRisk = key !== "tenure_years" || true;
      if (Math.abs(diff) > 0.05 * Math.max(1, Math.abs(peerAvg))) {
        direction = diff < 0 ? "risk" : "positive";
        if (key === "tenure_years") direction = diff < 0 ? "risk" : "positive";
        impact = Math.min(100, Math.max(10, Math.abs(diff) / Math.max(Math.abs(peerAvg), 1) * 100));
        reason = `${baseReason} Employee: ${value.toFixed(2)} vs cohort average ${peerAvg.toFixed(2)}.`;
      } else {
        reason = `Near the cohort average (${peerAvg.toFixed(2)}).`;
      }
    }
    return { label, value: value === null ? "N/A" : value.toFixed(2), impact, direction, reason };
  });
}

function groupAverage(rows, departmentKey, keys) {
  const m = {};
  rows.forEach((r) => {
    const name = r[departmentKey] || r.departmenttype || "Unknown";
    if (!m[name]) m[name] = { name, satisfaction: [], engagement: [] };
    m[name].satisfaction.push(r[keys[0]]);
    m[name].engagement.push(r[keys[1]]);
  });
  return Object.values(m).map((x) => ({ ...x, satisfaction: avg(x.satisfaction), engagement: avg(x.engagement) })).sort((a, b) => b.satisfaction - a.satisfaction).slice(0, 20);
}

function barLayout(title, height) {
  return { ...baseLayout(height), title: { text: title, font: { size: 14, color: "#b9d4ed" } }, xaxis: { tickangle: -45 }, yaxis: { title: "Employees / Risk" } };
}
function baseLayout(height) {
  return { autosize: true, height, margin: { l: 55, r: 20, t: 40, b: 100 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#dbeafe" }, legend: { orientation: "h" }, hovermode: "closest" };
}

function FilterGroup({ label, options, value, onChange }) {
  return <div style={styles.filterGroup}><div style={styles.filterLabel}>{label}</div><select multiple value={value} onChange={(e) => onChange(Array.from(e.target.selectedOptions, (o) => o.value))} style={styles.filterSelect}>{options.map((o) => <option key={o} value={o}>{o}</option>)}</select><div style={styles.filterHint}>{value.length ? `${value.length} selected` : "Ctrl + click for multiple"}</div></div>;
}
function KpiCard({ label, value, accent }) {
  const color = accent === "danger" ? "#fb7185" : accent === "warning" ? "#fbbf24" : accent === "success" ? "#4ade80" : "#38bdf8";
  return <div style={styles.kpiCard}><div style={styles.kpiLabel}>{label}</div><div style={{ ...styles.kpiValue, color }}>{value}</div></div>;
}
function PanelTitle({ title }) { return <div style={styles.panelTitle}>{title}</div>; }
function RiskBadge({ risk }) { const c = { "High Risk": ["#4c1721", "#fb7185"], "Medium Risk": ["#4a3510", "#fbbf24"], "Low Risk": ["#123d2a", "#4ade80"] }[risk] || ["#1e293b", "#cbd5e1"]; return <span style={{ ...styles.riskBadge, background: c[0], color: c[1] }}>{risk || "Unknown"}</span>; }
function Detail({ label, value }) { return <div style={styles.detail}><div style={styles.detailLabel}>{label}</div><div style={styles.detailValue}>{value === null || value === undefined || value === "" ? "N/A" : String(value)}</div></div>; }
function EmptyState({ title, text }) { return <div style={styles.panel}><div style={styles.noSelection}><div style={styles.noSelectionIcon}>◎</div><div style={styles.noSelectionTitle}>{title}</div><div style={styles.noSelectionText}>{text}</div></div></div>; }

const styles = {
  app: { minHeight: "100vh", background: "#08111f", color: "#e5eef9", fontFamily: "Inter, Segoe UI, Roboto, Arial, sans-serif" },
  topBar: { height: 64, padding: "0 28px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0b1627", borderBottom: "1px solid #1f334c", position: "sticky", top: 0, zIndex: 20 },
  brand: { color: "#38bdf8", fontWeight: 800, letterSpacing: 2, fontSize: 16 }, brandSub: { color: "#8da3bc", fontSize: 12, marginTop: 2 },
  status: { color: "#9fb3c8", fontSize: 12, display: "flex", gap: 8, alignItems: "center" }, statusDot: { width: 8, height: 8, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 10px rgba(74,222,128,.7)" },
  layout: { display: "grid", gridTemplateColumns: "270px minmax(0,1fr)", minHeight: "calc(100vh - 64px)" }, sidebar: { background: "#0b1524", borderRight: "1px solid #20344c", padding: 20, position: "sticky", top: 64, height: "calc(100vh - 64px)", overflowY: "auto" },
  sidebarTitle: { fontSize: 22, fontWeight: 750, marginBottom: 6 }, sidebarSubtitle: { fontSize: 12, lineHeight: 1.5, color: "#8298b2", marginBottom: 24 }, sidebarInfo: { marginTop: 20, padding: 12, background: "#0d1b2d", border: "1px solid #213a54", borderRadius: 10, color: "#7890a9", fontSize: 10, lineHeight: 1.7 },
  filterGroup: { marginBottom: 19 }, filterLabel: { fontSize: 13, fontWeight: 700, color: "#cbd8e7", marginBottom: 8 }, filterSelect: { width: "100%", minHeight: 105, borderRadius: 10, padding: 8, background: "#0d1a2c", color: "#e5eef9", border: "1px solid #2b405a" }, filterHint: { color: "#647b95", fontSize: 10, marginTop: 5 },
  searchInput: { width: "100%", boxSizing: "border-box", padding: "11px 12px", borderRadius: 10, background: "#0d1a2c", border: "1px solid #2b405a", color: "#fff", outline: "none" }, clearButton: { width: "100%", marginTop: 14, padding: "10px 12px", borderRadius: 9, border: "1px solid #31516e", background: "#13243a", color: "#9fdcff", cursor: "pointer" },
  main: { padding: 24, minWidth: 0 }, hero: { background: "linear-gradient(135deg,#14283f,#101d31 55%,#17273d)", border: "1px solid #29435e", borderRadius: 18, padding: "27px 30px", marginBottom: 18 }, heroTitle: { fontSize: 28, fontWeight: 800, color: "#38bdf8" }, heroSubtitle: { color: "#9ab0c7", marginTop: 8, fontSize: 13 },
  kpiGrid: { display: "grid", gridTemplateColumns: "repeat(7,minmax(120px,1fr))", gap: 12, marginBottom: 18 }, kpiMiniGrid: { display: "grid", gridTemplateColumns: "repeat(2,minmax(180px,1fr))", gap: 12 }, kpiCard: { background: "#142238", border: "1px solid #29425e", borderRadius: 14, padding: 15, minHeight: 92, boxSizing: "border-box" }, kpiLabel: { color: "#829ab4", fontSize: 10, fontWeight: 700, textTransform: "uppercase", lineHeight: 1.4 }, kpiValue: { fontSize: 23, fontWeight: 800, marginTop: 15 },
  navStrip: { display: "flex", flexWrap: "wrap", gap: 7, padding: "7px 0 13px", borderBottom: "1px solid #21354c", marginBottom: 18 }, navButton: { border: "none", background: "transparent", color: "#8399b1", padding: "8px 10px", borderRadius: 8, cursor: "pointer", fontSize: 12 }, navButtonActive: { border: "1px solid #1c6c96", background: "#0b3550", color: "#38bdf8", padding: "8px 10px", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 700 },
  twoColumn: { display: "grid", gridTemplateColumns: "0.85fr 1.15fr", gap: 18 }, stack: { display: "flex", flexDirection: "column", gap: 18 }, panel: { background: "#0e1929", border: "1px solid #253d58", borderRadius: 16, padding: 18, minWidth: 0 }, panelTitle: { color: "#38bdf8", fontSize: 16, fontWeight: 750, marginBottom: 12 }, tableHint: { color: "#7188a2", fontSize: 11, marginBottom: 10 },
  tableWrap: { maxHeight: 430, overflow: "auto", border: "1px solid #22384f", borderRadius: 10 }, table: { width: "100%", borderCollapse: "collapse", fontSize: 12 }, th: { position: "sticky", top: 0, background: "#14243a", color: "#9db2c9", textAlign: "left", padding: 10, borderBottom: "1px solid #2b425b", zIndex: 2 }, tr: { cursor: "pointer", borderBottom: "1px solid #1e3045" }, td: { padding: 10, color: "#dbe6f2" }, emptyCell: { padding: 30, textAlign: "center", color: "#7188a2" }, riskBadge: { display: "inline-block", padding: "5px 9px", borderRadius: 999, fontSize: 10, fontWeight: 800, whiteSpace: "nowrap" },
  chartGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }, summaryTable: { display: "flex", flexDirection: "column", gap: 5 }, summaryRow: { display: "grid", gridTemplateColumns: "1fr auto auto", gap: 14, alignItems: "center", padding: "9px 10px", borderBottom: "1px solid #1e3045", fontSize: 12 },
  profileHeader: { display: "flex", justifyContent: "space-between", alignItems: "flex-start" }, explainSource: { padding: 10, background: "#0b1727", border: "1px solid #213b55", borderRadius: 9, color: "#8da5bd", fontSize: 11, marginBottom: 14 }, factorRow: { marginBottom: 14 }, factorName: { display: "flex", justifyContent: "space-between", color: "#dce9f7", fontSize: 12, fontWeight: 700 }, factorTrack: { height: 8, borderRadius: 999, background: "#17263a", overflow: "hidden", marginTop: 6 }, factorBar: { height: "100%", borderRadius: 999 }, factorReason: { color: "#7188a2", fontSize: 10, marginTop: 5 }, loading: { color: "#38bdf8", padding: 25 },
  employeeGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }, profileCard: { background: "#132238", border: "1px solid #2a425d", borderRadius: 14, padding: 18 }, detailGrid: { display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 9, marginBottom: 16 }, detail: { background: "#0e1b2c", borderRadius: 9, padding: 10, border: "1px solid #20374f" }, detailLabel: { color: "#6f88a3", fontSize: 9, textTransform: "uppercase", fontWeight: 700 }, detailValue: { color: "#dbeafe", fontSize: 12, fontWeight: 650, marginTop: 4, wordBreak: "break-word" }, primaryButton: { width: "100%", padding: "12px 14px", border: "none", borderRadius: 10, background: "#0284c7", color: "#fff", fontWeight: 750, cursor: "pointer" }, recommendationBox: { marginTop: 14, padding: 14, background: "#0b1727", border: "1px solid #25405a", borderRadius: 11 }, boxTitle: { color: "#38bdf8", fontSize: 12, fontWeight: 800, marginBottom: 8 }, aiText: { whiteSpace: "pre-wrap", lineHeight: 1.6, fontSize: 12, color: "#d9e5f1" },
  assistantCard: { background: "#101d30", border: "1px solid #2b4865", borderRadius: 14, padding: 17, display: "flex", flexDirection: "column", minHeight: 470 }, assistantHeader: { display: "flex", alignItems: "center", gap: 11, paddingBottom: 13, borderBottom: "1px solid #243b54" }, assistantIcon: { width: 38, height: 38, borderRadius: 10, display: "grid", placeItems: "center", background: "#075985", color: "#bae6fd", fontSize: 20 }, assistantTitle: { fontSize: 16, fontWeight: 800 }, assistantStatus: { color: "#6f8aa5", fontSize: 10, marginTop: 3 }, contextBox: { marginTop: 13, padding: 11, borderRadius: 10, background: "#0b1727", border: "1px solid #203a55", fontSize: 12 }, contextSmall: { color: "#7891aa", fontSize: 10, marginTop: 4 }, chatBox: { flex: 1, minHeight: 190, maxHeight: 300, overflowY: "auto", marginTop: 12, display: "flex", flexDirection: "column", gap: 9, paddingRight: 3 }, chatEmpty: { color: "#7188a2", fontSize: 12, lineHeight: 1.65, padding: 14, borderRadius: 10, background: "#0b1727" }, message: { maxWidth: "88%", padding: "9px 11px", borderRadius: 11 }, messageRole: { fontSize: 9, color: "#9fc4df", fontWeight: 800, marginBottom: 5, textTransform: "uppercase" }, typing: { color: "#38bdf8", fontSize: 11, padding: 8 }, quickQuestions: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }, quickButton: { border: "1px solid #31516e", background: "#13243a", color: "#9fdcff", borderRadius: 999, padding: "6px 8px", fontSize: 9, cursor: "pointer" }, chatInputRow: { display: "grid", gridTemplateColumns: "1fr auto", gap: 8, marginTop: 10 }, chatInput: { resize: "none", width: "100%", boxSizing: "border-box", borderRadius: 10, border: "1px solid #2b465f", background: "#0a1625", color: "#e5eef9", padding: 10, outline: "none", fontFamily: "inherit", fontSize: 11 }, sendButton: { alignSelf: "stretch", minWidth: 65, border: "none", borderRadius: 10, background: "#0284c7", color: "#fff", fontWeight: 750, cursor: "pointer" },
  noSelection: { minHeight: 180, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", border: "1px dashed #2d4863", borderRadius: 14, background: "#0b1625" }, noSelectionIcon: { fontSize: 36, color: "#38bdf8" }, noSelectionTitle: { marginTop: 8, fontWeight: 750 }, noSelectionText: { marginTop: 7, maxWidth: 550, color: "#7188a2", fontSize: 12, lineHeight: 1.6 }
};

export default App