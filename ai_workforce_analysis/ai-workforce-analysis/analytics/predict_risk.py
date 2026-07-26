import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from analytics.attrition_model import load_and_preprocess_ml_data, run_pipeline
except ModuleNotFoundError:
    from attrition_model import load_and_preprocess_ml_data, run_pipeline


def generate_workforce_predictions(
    data_path: str = "data/clean_employee_data.csv",
    model_dir: str = "models",
    output_path: str = "data/employee_predictions.csv"
) -> pd.DataFrame:
    """
    Loads clean employee data, runs prediction using the trained model artifact,
    computes employee-level attrition risk probabilities and risk categories (Low, Medium, High),
    and saves the enriched dataset to 'data/employee_predictions.csv'.
    """
    model_path = os.path.join(model_dir, "best_attrition_model.pkl")
    features_path = os.path.join(model_dir, "model_features.pkl")

    # If model artifact does not exist, run training pipeline first
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found. Training model pipeline first...")
        run_pipeline()

    model_data = joblib.load(model_path)
    model = model_data["model"]
    model_name = model_data["name"]
    feature_cols = joblib.load(features_path)

    # Load and preprocess dataset
    df, X, y, numeric_cols, cat_cols = load_and_preprocess_ml_data(data_path)

    # Reindex X to match saved feature set columns
    X_aligned = X.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Generate Attrition Risk Probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_aligned)[:, 1]
    else:
        probabilities = model.predict(X_aligned)

    df["attrition_risk_score"] = np.round(probabilities, 4)

    # Categorize Risk Scores based on calibrated distribution thresholds
    high_threshold = max(0.22, np.percentile(probabilities, 82))
    med_threshold = max(0.14, np.percentile(probabilities, 55))

    def assign_risk_category(prob):
        if prob >= high_threshold:
            return "High Risk"
        elif prob >= med_threshold:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_category"] = df["attrition_risk_score"].apply(assign_risk_category)

    # Add Actionable AI Recommendations
    def assign_ai_recommendation(row):
        risk = row["risk_category"]
        sat = row.get("satisfactionscore", 3)
        eng = row.get("engagementscore", 3)
        wlb = row.get("worklifebalancescore", 3)

        if risk == "High Risk":
            recs = []
            if sat <= 2:
                recs.append("Schedule 1-on-1 engagement review & compensation audit")
            if wlb <= 2:
                recs.append("Offer flexible work options / workload reduction")
            if eng <= 2:
                recs.append("Enroll in high-value mentorship / career growth initiative")
            return " | ".join(recs) if recs else "Immediate Retention Plan & Career Path Review"
        elif risk == "Medium Risk":
            return "Monitor engagement trends & offer proactive professional development"
        else:
            return "Maintain current retention strategies & recognize high performance"

    df["ai_recommendation"] = df.apply(assign_ai_recommendation, axis=1)

    # Save Enriched Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n--------------------------------------------------")
    print(" WORKFORCE RISK PREDICTIONS COMPLETED")
    print("--------------------------------------------------")
    print(f"Model Engine: {model_name}")
    print(f"Total Records Processed: {len(df):,}")
    print(f"High Risk Employees: {(df['risk_category'] == 'High Risk').sum():,} ({(df['risk_category'] == 'High Risk').mean()*100:.1f}%)")
    print(f"Medium Risk Employees: {(df['risk_category'] == 'Medium Risk').sum():,} ({(df['risk_category'] == 'Medium Risk').mean()*100:.1f}%)")
    print(f"Low Risk Employees: {(df['risk_category'] == 'Low Risk').sum():,} ({(df['risk_category'] == 'Low Risk').mean()*100:.1f}%)")
    print(f"Saved prediction dataset to: '{output_path}'")

    return df


if __name__ == "__main__":
    generate_workforce_predictions()
