import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set modern aesthetic design parameters
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_theme(style="whitegrid", palette="muted")

plt.rcParams.update({
    "font.sans-serif": "Segoe UI",
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc"
})


def fix_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes columns, fixing row shifting issues from exit date column.
    Extracts true department names and normalized status categories.
    """
    df = df.copy()

    # 1. Clean & Extract Department
    def extract_department(row):
        d_type = str(row.get("departmenttype", "")).strip()
        div = str(row.get("division", "")).strip()
        b_unit = str(row.get("businessunit", "")).strip()

        def is_valid_dept(val):
            val_lower = val.lower()
            if not val or val_lower in ["not applicable", "nan", "none", "0", "null", "1969-10-07"]:
                return False
            # Filter out random sentence descriptions
            if len(val) > 28 or "." in val or (len(val.split()) > 3 and not any(k in val_lower for k in ["software", "information", "customer"])):
                return False
            if any(char.isdigit() for char in val):
                return False
            return True

        if is_valid_dept(d_type):
            return d_type
        if is_valid_dept(div):
            return div
        if is_valid_dept(b_unit):
            return b_unit

        return "General Workforce"

    df["department_clean"] = df.apply(extract_department, axis=1)

    # 2. Clean & Extract Status
    def extract_status(row):
        status = str(row.get("employeestatus", "")).strip()
        if "Active" in status:
            return "Active"
        elif any(term in status for term in ["Terminated", "Resignation", "Voluntarily", "Involuntary", "Retirement"]):
            return "Terminated"
        elif "Future" in status:
            return "Future Start"

        # Check adjacent shifted columns
        for col in ["employeetype", "employeeclassificationtype", "terminationtype"]:
            val = str(row.get(col, ""))
            if "Active" in val:
                return "Active"
            elif any(term in val for term in ["Terminated", "Resignation", "Voluntarily", "Involuntary", "Retirement"]):
                return "Terminated"

        return "Active" if not status or status == "nan" else status

    df["status_clean"] = df.apply(extract_status, axis=1)

    # 3. Clean & Extract Employee Type
    def extract_emp_type(row):
        emp_type = str(row.get("employeetype", "")).strip()
        if emp_type in ["Full-Time", "Part-Time", "Contract"]:
            return emp_type
        for col in ["payzone", "employeeclassificationtype"]:
            val = str(row.get(col, "")).strip()
            if val in ["Full-Time", "Part-Time", "Contract"]:
                return val
        return "Full-Time"

    df["employeetype_clean"] = df.apply(extract_emp_type, axis=1)

    return df


def load_data(file_path: str = "data/clean_employee_data.csv") -> pd.DataFrame:
    """
    Loads clean employee dataset, fixes column alignments, and ensures numeric types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clean data file not found at path: {file_path}")

    df = pd.read_csv(file_path)
    df = fix_dataframe_columns(df)

    numeric_cols = [
        "satisfactionscore",
        "engagementscore",
        "worklifebalancescore",
        "performance_rating_numeric",
        "training_cost",
        "training_durationdays",
        "tenure_years"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ensure_output_dir(output_dir: str = "data/eda_results"):
    """
    Creates destination directory for EDA visual outputs.
    """
    os.makedirs(output_dir, exist_ok=True)


def analyze_workforce_overview(df: pd.DataFrame, output_dir: str = "data/eda_results") -> dict:
    """
    1. Workforce Overview: Total employees, status distribution, active vs terminated, type.
    """
    total_employees = len(df)
    status_dist = df["status_clean"].value_counts()
    type_dist = df["employeetype_clean"].value_counts()

    active_count = (df["status_clean"] == "Active").sum()
    terminated_count = (df["status_clean"] == "Terminated").sum()
    other_status_count = total_employees - active_count - terminated_count

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Employee Status Distribution
    palette_status = sns.color_palette("mako", len(status_dist))
    sns.barplot(x=status_dist.values, y=status_dist.index.astype(str), ax=axes[0], palette=palette_status)
    axes[0].set_title("Employee Status Distribution")
    axes[0].set_xlabel("Employee Count")
    axes[0].ticklabel_format(style='plain', axis='x')
    for i, v in enumerate(status_dist.values):
        pct = (v / total_employees) * 100
        axes[0].text(v + (max(status_dist.values) * 0.015), i, f"{v:,} ({pct:.1f}%)", va='center', fontweight='bold', fontsize=9)

    # Plot 2: Active vs Terminated (Donut Chart)
    categories = ["Active", "Terminated", "Other / Future"]
    counts = [active_count, terminated_count, max(0, other_status_count)]
    colors = ["#10b981", "#ef4444", "#3b82f6"]
    axes[1].pie(counts, labels=categories, autopct="%1.1f%%", colors=colors, startangle=140, pctdistance=0.75,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2), textprops={'fontweight': 'bold'})
    axes[1].set_title("Active vs Terminated Ratio")

    # Plot 3: Employee Type Distribution
    palette_type = sns.color_palette("Blues_r", len(type_dist))
    sns.barplot(x=type_dist.values, y=type_dist.index.astype(str), ax=axes[2], palette=palette_type)
    axes[2].set_title("Employee Type Distribution")
    axes[2].set_xlabel("Employee Count")
    axes[2].ticklabel_format(style='plain', axis='x')
    for i, v in enumerate(type_dist.values):
        pct = (v / total_employees) * 100
        axes[2].text(v + (max(type_dist.values) * 0.015), i, f"{v:,} ({pct:.1f}%)", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "1_workforce_overview.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 1. WORKFORCE OVERVIEW RESULTS")
    print("--------------------------------------------------")
    print(f"Total Employees: {total_employees}")
    print(f"Active Employees: {active_count} ({active_count/total_employees*100:.1f}%)")
    print(f"Terminated Employees: {terminated_count} ({terminated_count/total_employees*100:.1f}%)")
    print("Employee Status Distribution:\n", status_dist)
    print("Saved chart to:", chart_path)

    return {"total_employees": total_employees, "active_count": active_count, "terminated_count": terminated_count}


def analyze_department(df: pd.DataFrame, output_dir: str = "data/eda_results") -> pd.DataFrame:
    """
    2. Department Analysis:
       - Employee count by department
       - Department workforce distribution
       - Attrition rate (%) by department
    """
    dept_counts = df["department_clean"].value_counts()
    dept_pct = (dept_counts / len(df) * 100).round(2)

    is_terminated = df["status_clean"] == "Terminated"
    dept_attrition_count = df[is_terminated].groupby(df["department_clean"]).size().reindex(dept_counts.index, fill_value=0)
    dept_attrition_rate = ((dept_attrition_count / dept_counts) * 100).round(2)

    dept_summary = pd.DataFrame({
        "Employee_Count": dept_counts,
        "Workforce_Share_Pct": dept_pct,
        "Attrition_Count": dept_attrition_count,
        "Attrition_Rate_Pct": dept_attrition_rate
    }).sort_values(by="Employee_Count", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Employee Count by Department
    palette_dept = sns.color_palette("crest", len(dept_summary))
    sns.barplot(x=dept_summary["Employee_Count"], y=dept_summary.index.astype(str), ax=axes[0], palette=palette_dept)
    axes[0].set_title("Employee Count by Department")
    axes[0].set_xlabel("Number of Employees")
    axes[0].set_ylabel("Department")
    axes[0].ticklabel_format(style='plain', axis='x')
    max_c = max(dept_summary["Employee_Count"]) if len(dept_summary) > 0 else 10
    axes[0].set_xlim(0, max_c * 1.25)
    for i, v in enumerate(dept_summary["Employee_Count"]):
        axes[0].text(v + (max_c * 0.02), i, f"{v:,}", va='center', fontweight='bold', fontsize=9)

    # Subplot 2: Attrition Rate (%) by Department
    attrition_sorted = dept_summary.sort_values(by="Attrition_Rate_Pct", ascending=False)
    palette_att = sns.color_palette("flare", len(attrition_sorted))
    sns.barplot(x=attrition_sorted["Attrition_Rate_Pct"], y=attrition_sorted.index.astype(str), ax=axes[1], palette=palette_att)
    axes[1].set_title("Department Attrition Rate (%)")
    axes[1].set_xlabel("Attrition Rate (%)")
    axes[1].set_ylabel("")
    max_a = max(attrition_sorted["Attrition_Rate_Pct"]) if len(attrition_sorted) > 0 else 100
    axes[1].set_xlim(0, max(100, max_a * 1.2))
    for i, v in enumerate(attrition_sorted["Attrition_Rate_Pct"]):
        axes[1].text(v + 1.5, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "2_department_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 2. DEPARTMENT ANALYSIS RESULTS")
    print("--------------------------------------------------")
    print(dept_summary)
    highest_attrition_dept = attrition_sorted.index[0] if len(attrition_sorted) > 0 else "N/A"
    print(f"\nHighest Attrition Department: {highest_attrition_dept} ({attrition_sorted['Attrition_Rate_Pct'].iloc[0]}%)")
    print("Saved chart to:", chart_path)

    return dept_summary


def analyze_satisfaction(df: pd.DataFrame, output_dir: str = "data/eda_results") -> dict:
    """
    3. Employee Satisfaction Analysis:
       - satisfactionscore, engagementscore, worklifebalancescore
       - Relationship with employee status (Active vs Terminated)
    """
    score_cols = ["satisfactionscore", "engagementscore", "worklifebalancescore"]
    avail_cols = [c for c in score_cols if c in df.columns]

    stats = df[avail_cols].describe().T[["mean", "50%", "std", "min", "max"]].rename(columns={"50%": "median"})

    status_scores = (
        df.groupby("status_clean")[avail_cols].mean().round(2)
        if "status_clean" in df.columns else pd.DataFrame()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Distribution Boxplot
    df_melted = df.melt(value_vars=avail_cols, var_name="Metric", value_name="Score")
    df_melted["Metric"] = df_melted["Metric"].str.replace("score", "").str.title()
    sns.boxplot(x="Metric", y="Score", data=df_melted, ax=axes[0], palette="Set2", width=0.45)
    axes[0].set_title("Distribution of Employee Satisfaction Scores")
    axes[0].set_ylabel("Score (1 to 5)")
    axes[0].set_ylim(0, 5.5)

    # Subplot 2: Score Comparison by Status
    if not status_scores.empty:
        status_scores_renamed = status_scores.rename(columns=lambda c: c.replace("score", "").title())
        status_scores_renamed.plot(kind="barh", ax=axes[1], colormap="Set2", width=0.6)
        axes[1].set_title("Average Scores by Employee Status")
        axes[1].set_xlabel("Average Score (1 to 5)")
        axes[1].set_ylabel("Employee Status")
        axes[1].set_xlim(0, 5.5)
        axes[1].legend(title="Metrics", loc="lower right")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "3_satisfaction_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 3. EMPLOYEE SATISFACTION ANALYSIS RESULTS")
    print("--------------------------------------------------")
    print(stats.round(2))
    print("Saved chart to:", chart_path)

    return {"stats": stats, "status_scores": status_scores}


def analyze_performance(df: pd.DataFrame, output_dir: str = "data/eda_results") -> dict:
    """
    4. Performance Analysis:
       - High-performing departments
       - Relationship between performance and tenure
    """
    rating_col = "performance_rating_numeric" if "performance_rating_numeric" in df.columns else None

    dept_perf = pd.DataFrame()
    if rating_col:
        dept_perf = (
            df.groupby("department_clean")[rating_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Avg_Rating"})
            .sort_values(by="Avg_Rating", ascending=False)
        )

    corr = df[rating_col].corr(df["tenure_years"]) if rating_col and "tenure_years" in df.columns else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Average Performance by Department
    if not dept_perf.empty:
        palette_perf = sns.color_palette("viridis", len(dept_perf))
        sns.barplot(x=dept_perf["Avg_Rating"], y=dept_perf.index.astype(str), ax=axes[0], palette=palette_perf)
        axes[0].set_title("Average Performance Rating by Department")
        axes[0].set_xlabel("Avg Rating (1: PIP, 4: Exceeds)")
        axes[0].set_ylabel("Department")
        axes[0].set_xlim(0, 4.5)
        for i, v in enumerate(dept_perf["Avg_Rating"]):
            axes[0].text(v + 0.05, i, f"{v:.2f}", va='center', fontweight='bold', fontsize=9)

    # Subplot 2: Performance vs Tenure Boxplot
    if rating_col and "tenure_years" in df.columns:
        sns.boxplot(x=rating_col, y="tenure_years", data=df, ax=axes[1], palette="Blues")
        axes[1].set_title(f"Performance Rating vs Tenure (Corr: {corr:.3f})")
        axes[1].set_xlabel("Performance Rating Numeric (1 to 4)")
        axes[1].set_ylabel("Tenure (Years)")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "4_performance_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 4. PERFORMANCE ANALYSIS RESULTS")
    print("--------------------------------------------------")
    if not dept_perf.empty:
        print(dept_perf.round(2))
    print("Saved chart to:", chart_path)

    return {"dept_perf": dept_perf, "tenure_corr": corr}


def analyze_training(df: pd.DataFrame, output_dir: str = "data/eda_results") -> dict:
    """
    5. Training Analysis:
       - Training outcomes and performance impact
    """
    outcome_col = "training_outcome" if "training_outcome" in df.columns else None
    rating_col = "performance_rating_numeric" if "performance_rating_numeric" in df.columns else None

    outcome_dist = df[outcome_col].value_counts() if outcome_col else pd.Series()
    avg_cost = df["training_cost"].mean() if "training_cost" in df.columns else np.nan
    avg_duration = df["training_durationdays"].mean() if "training_durationdays" in df.columns else np.nan

    outcome_perf = pd.DataFrame()
    if outcome_col and rating_col:
        outcome_perf = (
            df.groupby(outcome_col)[rating_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Avg_Performance_Rating"})
            .sort_values(by="Avg_Performance_Rating", ascending=False)
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if outcome_col:
        palette_out = sns.color_palette("mako", len(outcome_dist))
        sns.barplot(x=outcome_dist.values, y=outcome_dist.index.astype(str), ax=axes[0], palette=palette_out)
        axes[0].set_title("Training Outcome Distribution")
        axes[0].set_xlabel("Employee Count")
        axes[0].set_ylabel("Outcome")
        axes[0].ticklabel_format(style='plain', axis='x')
        max_o = max(outcome_dist.values) if len(outcome_dist) > 0 else 10
        axes[0].set_xlim(0, max_o * 1.25)
        for i, v in enumerate(outcome_dist.values):
            axes[0].text(v + (max_o * 0.02), i, f"{v:,}", va='center', fontweight='bold', fontsize=9)

    if not outcome_perf.empty:
        palette_operf = sns.color_palette("Greens_r", len(outcome_perf))
        sns.barplot(x=outcome_perf["Avg_Performance_Rating"], y=outcome_perf.index.astype(str), ax=axes[1], palette=palette_operf)
        axes[1].set_title("Avg Performance Rating by Training Outcome")
        axes[1].set_xlabel("Avg Performance Rating (1 to 4)")
        axes[1].set_ylabel("")
        axes[1].set_xlim(0, 4.5)
        for i, v in enumerate(outcome_perf["Avg_Performance_Rating"]):
            axes[1].text(v + 0.05, i, f"{v:.2f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "5_training_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 5. TRAINING ANALYSIS RESULTS")
    print("--------------------------------------------------")
    print(f"Average Training Cost: ${avg_cost:.2f}" if not np.isnan(avg_cost) else "Cost: N/A")
    print(f"Average Training Duration: {avg_duration:.2f} days" if not np.isnan(avg_duration) else "Duration: N/A")
    print("Saved chart to:", chart_path)

    return {"outcome_dist": outcome_dist, "outcome_perf": outcome_perf}


def analyze_tenure(df: pd.DataFrame, output_dir: str = "data/eda_results") -> dict:
    """
    6. Tenure Analysis:
       - Tenure distribution and retention patterns
    """
    tenure_col = "tenure_years" if "tenure_years" in df.columns else None
    tenure_stats = df[tenure_col].describe() if tenure_col else pd.Series()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if tenure_col:
        sns.histplot(df[tenure_col].dropna(), kde=True, ax=axes[0], color="#2563eb", bins=25)
        axes[0].set_title("Employee Tenure Distribution (Years)")
        axes[0].set_xlabel("Tenure (Years)")
        axes[0].set_ylabel("Employee Count")

    if tenure_col and "status_clean" in df.columns:
        sns.boxplot(x=tenure_col, y="status_clean", data=df, ax=axes[1], palette="PuBu")
        axes[1].set_title("Employee Tenure by Status")
        axes[1].set_xlabel("Tenure (Years)")
        axes[1].set_ylabel("Employee Status")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "6_tenure_retention_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--------------------------------------------------")
    print(" 6. TENURE & RETENTION ANALYSIS RESULTS")
    print("--------------------------------------------------")
    print(tenure_stats.round(2))
    print("Saved chart to:", chart_path)

    return {"tenure_stats": tenure_stats}


def run_eda(file_path: str = "data/clean_employee_data.csv", output_dir: str = "data/eda_results"):
    """
    Main driver for Phase 7 Exploratory Data Analysis.
    """
    print("==================================================")
    print("   AI WORKFORCE ANALYTICS - PHASE 7: EDA")
    print("==================================================")

    df = load_data(file_path)
    ensure_output_dir(output_dir)

    analyze_workforce_overview(df, output_dir)
    analyze_department(df, output_dir)
    analyze_satisfaction(df, output_dir)
    analyze_performance(df, output_dir)
    analyze_training(df, output_dir)
    analyze_tenure(df, output_dir)

    print("\n==================================================")
    print("   PHASE 7 EDA COMPLETED SUCCESSFULLY!")
    print(f"   All high-resolution charts saved inside: '{output_dir}/'")
    print("==================================================\n")


if __name__ == "__main__":
    run_eda()
