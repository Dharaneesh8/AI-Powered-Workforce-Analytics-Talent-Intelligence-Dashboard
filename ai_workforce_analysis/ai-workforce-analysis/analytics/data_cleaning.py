import pandas as pd
from database.athena_connection import get_connection


def clean_employee_data():

    # Load data from Athena
    conn = get_connection()

    query = """
    SELECT *
    FROM employees_csv
    """

    df = pd.read_sql(query, conn)

    print("Original Dataset Shape:")
    print(df.shape)


    # Remove completely invalid columns
    df.drop(
        columns=["dateofbirth"],
        inplace=True,
        errors="ignore"
    )


    # Convert numeric columns
    numeric_columns = [
        "satisfactionscore",
        "engagementscore",
        "worklifebalancescore",
        "training_durationdays",
        "training_cost",
        "recruitment_desiredsalary",
        "recruitment_yearsofexperience"
    ]


    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # Fill missing numeric values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(
                df[col].median()
            )


    # Convert date columns
    date_columns = [
        "startdate",
        "surveydate",
        "training_date",
        "recruitment_applicationdate"
    ]


    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )


    # Remove duplicate rows
    df.drop_duplicates(
        inplace=True
    )


    # Add performance score numeric mapping

    if "currentemployeerating" in df.columns:

        rating_map = {
            "Fully Meets": 3,
            "Exceeds": 4,
            "Needs Improvement": 2,
            "PIP": 1
        }

        df["performance_rating_numeric"] = (
            df["currentemployeerating"]
            .map(rating_map)
        )


    # Add employee tenure

    if "startdate" in df.columns:

        today = pd.Timestamp.today()

        df["tenure_years"] = (
            (today - df["startdate"])
            .dt.days / 365
        ).round(2)



    print("\nClean Dataset Shape:")
    print(df.shape)


    # Save cleaned dataset

    df.to_csv(
        "data/clean_employee_data.csv",
        index=False
    )


    print("\nClean dataset saved successfully!")

    print("\nMissing Values:")
    print(
        df.isnull()
        .sum()
        .head(20)
    )


    return df



if __name__ == "__main__":

    clean_employee_data()