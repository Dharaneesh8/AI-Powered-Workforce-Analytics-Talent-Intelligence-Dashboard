from pyathena import connect


def get_connection():

    conn = connect(
        s3_staging_dir="s3://workforce-analytics-data-infosys-virtual/athena-results/",
        region_name="eu-north-1",
        schema_name="mongodb_catalog"
    )

    return conn