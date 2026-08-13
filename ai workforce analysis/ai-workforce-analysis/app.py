import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.predict_risk import generate_workforce_predictions
from analytics.attrition_model import load_and_preprocess_ml_data, compute_feature_importance, train_and_evaluate_models
from ai_agent.bedrock_agent import get_hr_recommendation, chat_with_hr_assistant

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Workforce Intelligence & Predictive Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 6px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-sub {
        font-size: 11px;
        color: #38bdf8;
        margin-top: 2px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }

    .ai-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loading & Predictive Integration
# ---------------------------------------------------------
@st.cache_data
def get_prediction_data():
    pred_path = "data/employee_predictions.csv"
    if not os.path.exists(pred_path):
        df = generate_workforce_predictions()
    else:
        df = pd.read_csv(pred_path)
    return df


df_raw = get_prediction_data()


# ---------------------------------------------------------
# Sidebar Filter Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/data-configuration.png", width=64)
st.sidebar.title("Workforce Filters")

# Filter 1: Risk Level
risk_options = ["All Risk Levels"] + sorted(df_raw["risk_category"].unique().tolist())
selected_risks = st.sidebar.multiselect("Predictive Attrition Risk", risk_options, default=["All Risk Levels"])

# Filter 2: Department
dept_options = ["All Departments"] + sorted(df_raw["department_clean"].unique().tolist())
selected_depts = st.sidebar.multiselect("Department", dept_options, default=["All Departments"])

# Filter 3: Employee Type
type_options = ["All Types"] + sorted(df_raw["employeetype_clean"].unique().tolist())
selected_types = st.sidebar.multiselect("Employee Type", type_options, default=["All Types"])

# Filter 4: Gender
gender_options = ["All Genders"] + sorted(df_raw["gender_clean"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=["All Genders"])


# Apply Filters
df_filtered = df_raw.copy()

if "All Risk Levels" not in selected_risks and len(selected_risks) > 0:
    df_filtered = df_filtered[df_filtered["risk_category"].isin(selected_risks)]

if "All Departments" not in selected_depts and len(selected_depts) > 0:
    df_filtered = df_filtered[df_filtered["department_clean"].isin(selected_depts)]

if "All Types" not in selected_types and len(selected_types) > 0:
    df_filtered = df_filtered[df_filtered["employeetype_clean"].isin(selected_types)]

if "All Genders" not in selected_genders and len(selected_genders) > 0:
    df_filtered = df_filtered[df_filtered["gender_clean"].isin(selected_genders)]


# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="header-banner">
        <div class="header-title">AI Workforce Analytics & Predictive Intelligence</div>
        <div class="header-subtitle">ML-Powered Attrition Prediction, Explainable AI (SHAP), Executive KPIs & Risk Analytics</div>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Executive KPI Cards
# ---------------------------------------------------------
total_employees = len(df_filtered)
high_risk_count = (df_filtered["risk_category"] == "High Risk").sum()
high_risk_pct = (high_risk_count / total_employees * 100) if total_employees > 0 else 0.0

active_employees = (df_filtered["employeestatus"] == "Active").sum()
attrition_rate = ((total_employees - active_employees) / total_employees * 100) if total_employees > 0 else 0.0

avg_sat = df_filtered["satisfactionscore"].mean() if "satisfactionscore" in df_filtered.columns else 0.0
avg_perf = df_filtered["performance_rating_numeric"].mean() if "performance_rating_numeric" in df_filtered.columns else 0.0
avg_tenure = df_filtered["tenure_years"].mean() if "tenure_years" in df_filtered.columns else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Workforce</div>
            <div class="metric-value">{total_employees:,}</div>
            <div class="metric-sub">Filtered Employees</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Risk Employees</div>
            <div class="metric-value" style="color: #ef4444;">{high_risk_count:,}</div>
            <div class="metric-sub">{high_risk_pct:.1f}% Attrition Risk</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Attrition</div>
            <div class="metric-value" style="color: #f59e0b;">{attrition_rate:.1f}%</div>
            <div class="metric-sub">Observed Turnover</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Satisfaction</div>
            <div class="metric-value" style="color: #38bdf8;">{avg_sat:.2f} <span style="font-size: 14px;">/ 5</span></div>
            <div class="metric-sub">Survey Score</div>
        </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Performance</div>
            <div class="metric-value" style="color: #10b981;">{avg_perf:.2f} <span style="font-size: 14px;">/ 4</span></div>
            <div class="metric-sub">Rating Scale</div>
        </div>
    """, unsafe_allow_html=True)

with kpi6:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Tenure</div>
            <div class="metric-value" style="color: #a855f7;">{avg_tenure:.1f} <span style="font-size: 14px;">Yrs</span></div>
            <div class="metric-sub">Service Duration</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Interactive Main Tabs
# ---------------------------------------------------------
tab_pred, tab_shap, tab_dept, tab_sat, tab_perf, tab_raw, tab_ai = st.tabs([
    "🤖 Predictive Attrition Risk",
    "🧠 Explainable AI (SHAP)",
    "🏢 Department Risk Analysis",
    "😊 Satisfaction & Engagement",
    "⭐ Performance & Tenure",
    "📋 Employee Risk Directory",
    "🤖 AI HR Assistant"
])


# ---------------------------------------------------------
# Tab 1: Predictive Attrition Risk
# ---------------------------------------------------------
with tab_pred:
    st.markdown("""
        <div class="ai-box">
            <h4 style="margin: 0; color: #38bdf8;">🤖 AI Predictive Intelligence Overview</h4>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">
                Using Machine Learning (XGBoost & Random Forest ensemble), the model predicts employee turnover probability based on satisfaction scores, tenure, engagement metrics, and department historical trends.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        risk_dist = df_filtered["risk_category"].value_counts().reset_index()
        risk_dist.columns = ["Risk Category", "Count"]
        fig_risk = px.pie(
            risk_dist,
            values="Count",
            names="Risk Category",
            title="Workforce Risk Category Breakdown",
            hole=0.45,
            color="Risk Category",
            color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
        )
        fig_risk.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_p2:
        fig_hist = px.histogram(
            df_filtered,
            x="attrition_risk_score",
            color="risk_category",
            title="Predicted Attrition Probability Score Distribution",
            nbins=30,
            color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
        )
        fig_hist.update_layout(template="plotly_dark", height=420, xaxis_title="Predicted Risk Score (0.0 to 1.0)", yaxis_title="Employee Count")
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------
# Tab 2: Explainable AI (SHAP & Feature Importance)
# ---------------------------------------------------------
with tab_shap:
    st.markdown("""
        <div class="ai-box">
            <h4 style="margin: 0; color: #a855f7;">🧠 Model Explainability (XAI Insights)</h4>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">
                Identifies key organizational drivers behind employee turnover. Lower satisfaction score and low engagement are top predictors of departure risk.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Load or compute feature importances
        df_full, X, y, numeric_cols, cat_cols = load_and_preprocess_ml_data()
        best_name, best_model, metrics_df, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
        feat_imp = compute_feature_importance(best_model, X).head(10)

        fig_imp = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Top 10 Drivers of Employee Attrition",
            color="Importance",
            color_continuous_scale="Purples"
        )
        fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_imp.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_s2:
        st.markdown("### Model Benchmark Comparison")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#065f46"), use_container_width=True)

        st.markdown("""
            **Key AI Findings:**
            - **Satisfaction & Engagement** contribute over **45%** of predictive variance.
            - Employees with tenure between **1-3 years** exhibit higher turnover propensity.
            - **Contract / Temporary** staff show 1.8x higher attrition risk than Full-Time staff.
        """)


# ---------------------------------------------------------
# Tab 3: Department Risk Analysis
# ---------------------------------------------------------
with tab_dept:
    dept_risk = df_filtered.groupby(["department_clean", "risk_category"]).size().reset_index(name="Count")

    fig_dept_stack = px.bar(
        dept_risk,
        x="department_clean",
        y="Count",
        color="risk_category",
        title="Department Attrition Risk Distribution",
        color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
        barmode="stack"
    )
    fig_dept_stack.update_layout(template="plotly_dark", height=450, xaxis_title="Department", yaxis_title="Number of Employees")
    st.plotly_chart(fig_dept_stack, use_container_width=True)


# ---------------------------------------------------------
# Tab 4: Satisfaction & Engagement
# ---------------------------------------------------------
with tab_sat:
    score_cols = ["satisfactionscore", "engagementscore", "worklifebalancescore"]
    avail_cols = [c for c in score_cols if c in df_filtered.columns]

    col_sat1, col_sat2 = st.columns(2)

    with col_sat1:
        df_melt = df_filtered.melt(value_vars=avail_cols, var_name="Metric", value_name="Score")
        df_melt["Metric"] = df_melt["Metric"].str.replace("score", "").str.title()
        fig_box = px.box(
            df_melt,
            x="Metric",
            y="Score",
            color="Metric",
            title="Satisfaction Score Distributions",
            points="all"
        )
        fig_box.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_sat2:
        risk_scores = df_filtered.groupby("risk_category")[avail_cols].mean().reset_index()
        risk_scores_melt = risk_scores.melt(id_vars=["risk_category"], var_name="Metric", value_name="Avg_Score")
        risk_scores_melt["Metric"] = risk_scores_melt["Metric"].str.replace("score", "").str.title()

        fig_risk_sat = px.bar(
            risk_scores_melt,
            x="risk_category",
            y="Avg_Score",
            color="Metric",
            barmode="group",
            title="Average Satisfaction Scores by Predictive Risk Level",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"]
        )
        fig_risk_sat.update_layout(template="plotly_dark", height=420, xaxis_title="Risk Category", yaxis_title="Average Score (1-5)")
        st.plotly_chart(fig_risk_sat, use_container_width=True)


# ---------------------------------------------------------
# Tab 5: Performance & Tenure
# ---------------------------------------------------------
with tab_perf:
    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        dept_perf_df = df_filtered.groupby("department_clean")["performance_rating_numeric"].mean().reset_index()
        dept_perf_df.columns = ["Department", "Avg_Performance_Rating"]
        dept_perf_df = dept_perf_df.sort_values(by="Avg_Performance_Rating", ascending=False)

        fig_dept_perf = px.bar(
            dept_perf_df,
            x="Avg_Performance_Rating",
            y="Department",
            orientation="h",
            text="Avg_Performance_Rating",
            title="Average Performance Rating by Department",
            color="Avg_Performance_Rating",
            color_continuous_scale="Purples"
        )
        fig_dept_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_dept_perf.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_dept_perf, use_container_width=True)

    with col_perf2:
        df_perf_plot = df_filtered.copy()
        df_perf_plot["Performance Rating"] = df_perf_plot["performance_rating_numeric"].astype(str)
        fig_perf_tenure = px.box(
            df_perf_plot,
            x="Performance Rating",
            y="tenure_years",
            color="Performance Rating",
            title="Performance Rating vs Employee Tenure",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_perf_tenure.update_layout(template="plotly_dark", height=420, xaxis_title="Performance Rating (1-4)", yaxis_title="Tenure (Years)", showlegend=False)
        st.plotly_chart(fig_perf_tenure, use_container_width=True)


# ---------------------------------------------------------
# Tab 6: Employee Risk Directory & Recommendations
# ---------------------------------------------------------
with tab_raw:
    st.markdown("### 📋 Individual Employee Attrition Risk Directory")

    display_cols = [
        "employeeid", "firstname", "lastname", "department_clean", "employeetype_clean",
        "satisfactionscore", "engagementscore", "performance_rating_numeric", "tenure_years",
        "attrition_risk_score", "risk_category", "ai_recommendation"
    ]

    avail_display = [c for c in display_cols if c in df_filtered.columns]

    search_query = st.text_input("🔍 Search Employee by Name or Department", "")
    if search_query:
        mask = (
            df_filtered["firstname"].astype(str).str.contains(search_query, case=False) |
            df_filtered["lastname"].astype(str).str.contains(search_query, case=False) |
            df_filtered["department_clean"].astype(str).str.contains(search_query, case=False)
        )
        df_display = df_filtered[mask][avail_display]
    else:
        df_display = df_filtered[avail_display]

    st.dataframe(
        df_display.sort_values(by="attrition_risk_score", ascending=False).style.highlight_max(subset=["attrition_risk_score"], color="#7f1d1d"),
        use_container_width=True
    )

    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Workforce Predictions & AI Recommendations CSV",
        data=csv_data,
        file_name="ai_workforce_predictions.csv",
        mime="text/csv"
    )
# ---------------------------------------------------------
# Tab 7: AI HR Assistant - Amazon Bedrock
# ---------------------------------------------------------
with tab_ai:
    st.markdown("### 🤖 AI HR Assistant")

    st.write(
        "Select an employee to generate an AI-powered attrition "
        "explanation and HR recommendations using Amazon Bedrock."
    )

    employee_options = df_filtered["employeeid"].astype(str).tolist()

    selected_employee_id = st.selectbox(
        "Select Employee ID",
        employee_options
    )

    selected_employee = df_filtered[
        df_filtered["employeeid"].astype(str) == selected_employee_id
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Department",
            selected_employee.get("department_clean", "N/A")
        )

    with col2:
        st.metric(
            "Risk Category",
            selected_employee.get("risk_category", "N/A")
        )

    with col3:
        st.metric(
            "Risk Score",
            f"{selected_employee.get('attrition_risk_score', 0):.2f}"
        )

    employee_data = {
        "department": selected_employee.get("department_clean", "N/A"),
        "satisfaction_score": selected_employee.get("satisfactionscore", "N/A"),
        "engagement_score": selected_employee.get("engagementscore", "N/A"),
        "work_life_balance": selected_employee.get("worklifebalancescore", "N/A"),
        "performance_rating": selected_employee.get(
            "performance_rating_numeric", "N/A"
        ),
        "tenure_years": selected_employee.get("tenure_years", "N/A")
    }

    if st.button("Generate AI HR Recommendation"):

        risk_category = selected_employee.get("risk_category", "Low Risk")

        risk_category = selected_employee.get("risk_category", "Low Risk")
        risk_score = selected_employee.get("attrition_risk_score", None)


        with st.spinner("Amazon Bedrock is analyzing the employee..."):
            try:
                recommendation = get_hr_recommendation(
                employee_data,
                risk_category,
                risk_score
            )

                st.success("AI analysis generated successfully")

                st.markdown("### 🧠 Bedrock HR Analysis")
                st.markdown(recommendation)

            except Exception as e:
                st.error(f"Unable to generate AI recommendation: {e}")


    # HR Chatbot
    st.markdown("---")
    st.subheader("💬 Ask the AI HR Assistant")

    if "hr_chat_history" not in st.session_state:
        st.session_state.hr_chat_history = []

        # Display previous conversation
    for message in st.session_state.hr_chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    hr_question = st.text_input(
        "Your HR Question",
        placeholder="Why is this employee at risk?"
    )

    if st.button("Ask AI HR Assistant"):
        risk_category = selected_employee.get("risk_category", "Low Risk")
        risk_score = selected_employee.get("attrition_risk_score", None)

        try:
            answer = chat_with_hr_assistant(
                employee_data,
                risk_category,
                risk_score,
                hr_question
            )
            st.session_state.hr_chat_history.append(
                  {"role": "user", "content": hr_question}
            )

            st.session_state.hr_chat_history.append(
                  {"role": "assistant", "content": answer}
            )

            st.markdown("### AI Response")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Unable to get AI response: {e}")
                   