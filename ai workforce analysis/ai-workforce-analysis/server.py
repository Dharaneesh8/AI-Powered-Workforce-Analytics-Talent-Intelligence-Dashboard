from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import os

from ai_agent.bedrock_agent import (
    get_hr_recommendation,
    chat_with_hr_assistant
)


app = FastAPI(
    title="AI Workforce Analytics API"
)


# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "employee_predictions.csv"
)

EMPLOYEE_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean_employee_data.csv"
)


# Load data


def load_predictions():

    if not os.path.exists(PREDICTIONS_PATH):
        raise HTTPException(
            status_code=404,
            detail="employee_predictions.csv not found"
        )

    return pd.read_csv(PREDICTIONS_PATH)


def clean_records(df):

    # Convert NaN values to None
    df = df.astype(object)
    df = df.where(pd.notna(df), None)

    return df.to_dict(orient="records")


# Health


@app.get("/api/health")
def health():

    return {
        "status": "Backend is running"
    }


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

@app.get("/api/metadata")
def metadata():

    df = load_predictions()

    return {
        "risk_levels": sorted(
            df["risk_category"]
            .dropna()
            .unique()
            .tolist()
        ),

        "departments": sorted(
            df["department_clean"]
            .dropna()
            .unique()
            .tolist()
        ),

        "employee_types": sorted(
            df["employeetype_clean"]
            .dropna()
            .unique()
            .tolist()
        ),

        "genders": sorted(
            df["gender_clean"]
            .dropna()
            .unique()
            .tolist()
        )
    }


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

@app.get("/api/predictions")
def predictions(

    risk_categories: Optional[List[str]] = Query(None),

    departments: Optional[List[str]] = Query(None),

    employee_types: Optional[List[str]] = Query(None),

    genders: Optional[List[str]] = Query(None),

    search: Optional[str] = None
):

    df = load_predictions()

    # Risk filter
    if risk_categories:
        df = df[
            df["risk_category"].isin(risk_categories)
        ]

    # Department filter
    if departments:
        df = df[
            df["department_clean"].isin(departments)
        ]

    # Employee type filter
    if employee_types:
        df = df[
            df["employeetype_clean"].isin(employee_types)
        ]

    # Gender filter
    if genders:
        df = df[
            df["gender_clean"].isin(genders)
        ]

    # Search
    if search:

        search = search.lower()

        first_name = (
            df["firstname"]
            .fillna("")
            .astype(str)
        )

        last_name = (
            df["lastname"]
            .fillna("")
            .astype(str)
        )

        department = (
            df["department_clean"]
            .fillna("")
            .astype(str)
        )

        mask = (
            first_name.str.lower().str.contains(search, na=False)
            |
            last_name.str.lower().str.contains(search, na=False)
            |
            department.str.lower().str.contains(search, na=False)
        )

        df = df[mask]

    return {
        "count": len(df),
        "records": clean_records(df)
    }


# ---------------------------------------------------------
# AI Recommendation
# ---------------------------------------------------------

@app.post("/api/ai/recommendation")
def recommendation(payload: dict):

    try:

        employee_data = payload.get(
            "employee_data",
            {}
        )

        risk_category = payload.get(
            "risk_category"
        )

        risk_score = payload.get(
            "risk_score"
        )

        result = get_hr_recommendation(
            employee_data,
            risk_category,
            risk_score
        )

        return {
            "recommendation": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# AI Chat
# ---------------------------------------------------------

@app.post("/api/ai/chat")
def ai_chat(payload: dict):

    try:

        employee_data = payload.get(
            "employee_data",
            {}
        )

        risk_category = payload.get(
            "risk_category"
        )

        risk_score = payload.get(
            "risk_score"
        )

        question = payload.get(
            "question",
            ""
        )

        result = chat_with_hr_assistant(
            employee_data,
            risk_category,
            risk_score,
            question
        )

        return {
            "answer": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )