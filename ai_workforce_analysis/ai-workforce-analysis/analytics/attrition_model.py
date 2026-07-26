import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def load_and_preprocess_ml_data(file_path: str = "data/clean_employee_data.csv"):
    """
    Loads dataset, cleans shifted columns, creates binary target 'is_attrited',
    and prepares encoded feature set X and target y.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clean dataset not found at path: {file_path}")

    df = pd.read_csv(file_path)

    # 1. Clean & Extract Department
    def extract_department(row):
        d_type = str(row.get("departmenttype", "")).strip()
        div = str(row.get("division", "")).strip()
        b_unit = str(row.get("businessunit", "")).strip()

        def is_valid_dept(val):
            val_lower = val.lower()
            if not val or val_lower in ["not applicable", "nan", "none", "0", "null", "1969-10-07"]:
                return False
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

    # 2. Clean & Extract Target (1 = Terminated, 0 = Active / Other)
    def extract_target(row):
        status = str(row.get("employeestatus", "")).strip()
        if any(term in status for term in ["Terminated", "Resignation", "Voluntarily", "Involuntary", "Retirement"]):
            return 1
        for col in ["employeetype", "employeeclassificationtype", "terminationtype"]:
            val = str(row.get(col, ""))
            if any(term in val for term in ["Terminated", "Resignation", "Voluntarily", "Involuntary", "Retirement"]):
                return 1
        return 0

    df["is_attrited"] = df.apply(extract_target, axis=1)

    # 3. Clean Employee Type & Gender
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
    df["gender_clean"] = df["gender"].fillna("Unspecified").astype(str) if "gender" in df.columns else "Unspecified"
    df["training_outcome_clean"] = df["training_outcome"].fillna("Completed").astype(str) if "training_outcome" in df.columns else "Completed"

    # Numeric Columns Imputation
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
            med = df[col].median()
            fill_val = med if not pd.isna(med) else 0.0
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = 0.0

    # Categorical Feature Selection & One-Hot Encoding
    cat_cols = ["department_clean", "employeetype_clean", "gender_clean", "training_outcome_clean"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)

    # Feature Matrix X and Target y
    X = pd.concat([df[numeric_cols], df_encoded], axis=1)
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y = df["is_attrited"].fillna(0).astype(int)

    return df, X, y, numeric_cols, cat_cols


def train_and_evaluate_models(X, y):
    """
    Splits dataset into train/test, trains Logistic Regression, Random Forest, and XGBoost,
    evaluates performance metrics, and selects the best model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, eval_metric="logloss")
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42)

    results = {}
    fitted_models = {}

    print("\n--------------------------------------------------")
    print(" MACHINE LEARNING MODEL EVALUATION RESULTS")
    print("--------------------------------------------------")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
        fitted_models[name] = model

        print(f"[{name}] -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Select Best Model based on ROC-AUC
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model = fitted_models[best_name]

    print("\n--------------------------------------------------")
    print(f" BEST PERFORMING MODEL SELECTED: {best_name}")
    print("--------------------------------------------------")

    return best_name, best_model, pd.DataFrame(results).T, X_train, X_test, y_train, y_test


def compute_feature_importance(best_model, X):
    """
    Computes feature importance rankings for Explainable AI (XAI).
    """
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(X.shape[1])

    feature_imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return feature_imp_df


def save_model_artifacts(best_model, best_name, feature_cols, output_dir: str = "models"):
    """
    Saves trained ML model and metadata artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_attrition_model.pkl")
    features_path = os.path.join(output_dir, "model_features.pkl")

    joblib.dump({"name": best_name, "model": best_model}, model_path)
    joblib.dump(feature_cols, features_path)

    print(f"Saved best model artifact to: '{model_path}'")
    print(f"Saved feature list artifact to: '{features_path}'")


def run_pipeline():
    """
    Main execution pipeline for Phase 9 ML training & evaluation.
    """
    df, X, y, numeric_cols, cat_cols = load_and_preprocess_ml_data()
    best_name, best_model, metrics_df, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    feature_imp_df = compute_feature_importance(best_model, X)
    save_model_artifacts(best_model, best_name, X.columns.tolist())

    return best_model, metrics_df, feature_imp_df


if __name__ == "__main__":
    run_pipeline()
