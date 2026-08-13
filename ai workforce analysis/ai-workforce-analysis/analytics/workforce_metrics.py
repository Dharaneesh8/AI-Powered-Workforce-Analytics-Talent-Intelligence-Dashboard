import pandas as pd
from database.athena_connection import get_connection


def get_employee_data():

    conn = get_connection()

    query = """
    SELECT *
    FROM employees_csv
    """

    df = pd.read_sql(query, conn)

    return df



def print_employee_overview():

    df = get_employee_data()

    print("\n========== DATASET OVERVIEW ==========")

    print("Total Rows:", len(df))
    print("Total Columns:", len(df.columns))


    print("\n========== EMPLOYEE STATUS ==========")
    print(df["employeestatus"].value_counts())


    print("\n========== DIVISION ==========")
    print(df["division"].value_counts())


    print("\n========== PERFORMANCE SCORE ==========")
    print(df["performancescore"].value_counts())


    print("\n========== EMPLOYEE RATING ==========")
    print(df["currentemployeerating"].value_counts())


    print("\n========== ENGAGEMENT SCORE ==========")
    print(df["engagementscore"].value_counts())


    print("\n========== SATISFACTION SCORE ==========")
    print(df["satisfactionscore"].value_counts())


    print("\n========== TRAINING OUTCOME ==========")
    print(df["training_outcome"].value_counts())



def workforce_summary():

    df = get_employee_data()


    # Convert numerical columns safely
    df["satisfactionscore"] = pd.to_numeric(
        df["satisfactionscore"],
        errors="coerce"
    )


    summary = {

        "Total Employees": len(df),


        "Total Divisions": df["division"].nunique(),


        "Average Satisfaction Score": round(
            df["satisfactionscore"].mean(),
            2
        ),


        "Gender Diversity Count": df["gender"].nunique(),


        "Training Programs": df["training_programname"].nunique(),


        "Average Training Duration Days": round(
            pd.to_numeric(
                df["training_durationdays"],
                errors="coerce"
            ).mean(),
            2
        )

    }


    return summary



if __name__ == "__main__":


    print_employee_overview()


    print("\n========== WORKFORCE SUMMARY ==========")

    result = workforce_summary()

    for key, value in result.items():

        print(f"{key}: {value}")