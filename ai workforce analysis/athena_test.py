from pyathena import connect
import pandas as pd

conn = connect(
    s3_staging_dir="",
    region_name="",
    schema_name="mongodb_catalog"
)

query = """
SELECT *
FROM employees_csv
LIMIT 5
"""

df = pd.read_sql(query, conn)

print(df)
