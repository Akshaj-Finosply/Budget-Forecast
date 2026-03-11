from datetime import datetime
import numpy as np
import pandas as pd
import snowflake.connector
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from snowflake.connector.pandas_tools import write_pandas

import os
import logging

from configs.az_db_conn import connect_to_snowflake, insert_or_update_log, update_log_entry
from configs.snowflake_config import get_snowflake_config, get_snowflake_dsn, PYTHON_EXEC_PATH, SCRIPT_DIRECTORY_PATH
from configs.az_config import task_details

bar = "=" * 100

pd.options.display.float_format = '{:,.2f}'.format


# ---------------- LOGGING SETUP ----------------
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = os.path.join(
    log_directory,
    f"Budget_Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def log_and_print(message, level="info"):
    print(message)
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "debug":
        logging.debug(message)
    elif level == "warning":
        logging.warning(message)


log_and_print(bar)
log_and_print(bar)
log_and_print("Beginning Logging")

# ---------------- SNOWFLAKE CONFIG ----------------
config = get_snowflake_config()
dsn = get_snowflake_dsn(config)
SNOWFLAKE_CONFIG = {
    'account':   config['SNOWFLAKE_ACCOUNT'],
    'user':      config['SNOWFLAKE_USER'],
    'password':  config['SNOWFLAKE_PASSWORD'],
    'warehouse': config['SNOWFLAKE_WAREHOUSE'],
    'database':  config['SNOWFLAKE_DATABASE'],
    'schema':    config['SNOWFLAKE_SCHEMA'],
}

# Task metadata
task_details["task_name"] = "Budget Forecast"
client_id    = task_details["client_id"]
client_name  = task_details["client_name"]
task_name    = task_details["task_name"]
environment  = task_details["environment"]
current_date = datetime.now().strftime('%Y-%m-%d')

# Connect to Snowflake
try:
    log_and_print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    database = SNOWFLAKE_CONFIG['database']
    log_and_print("Connected to Snowflake successfully.")
except Exception as e:
    log_and_print(f"Failed to connect to Snowflake: {e}", level="error")
    raise


# ---------------- DATA FETCHING ----------------
def fetch_merged_costs(conn=None, CSP=None, current_month=None):
    if not conn or not CSP or not current_month:
        log_and_print("Missing required parameters to fetch chargeback costs: conn, CSP, or current_month", "error")
        raise ValueError("Missing required parameters to fetch chargeback costs: conn, CSP, or current_month")

    try:
        log_and_print("Starting the query execution.")
        with conn.cursor() as cs:
            query = """
                    WITH CHILD_BUS AS (
                        SELECT DISTINCT
                            ROOT_BU_ID,
                            BU_ID
                        FROM APP.VW_BUSINESS_UNITS_HIERARCHY
                        WHERE ROOT_BU_ID NOT IN (
                            SELECT DISTINCT FO_BUSINESS_UNIT_ID 
                            FROM APP.BUSINESS_UNITS
                            WHERE IS_ALL_BU = TRUE
                        )
                    ),
                    BU_CTE AS (
                        SELECT FO_BUSINESS_UNIT_ID, BUSINESS_UNIT_NAME
                        FROM APP.BUSINESS_UNITS
                        WHERE IS_UNALLOCATED_BU = FALSE
                    ),
                    ALL_BU AS (
                        SELECT FO_BUSINESS_UNIT_ID, BUSINESS_UNIT_NAME
                        FROM APP.BUSINESS_UNITS
                        WHERE IS_ALL_BU = TRUE
                    )

                    -- Combined query for total + BU-level spend
                    SELECT 
                        CBU.ROOT_BU_ID AS BU_ID,
                        BU.BUSINESS_UNIT_NAME,
                        A.MONTH,
                        SUM(A.COST) AS MONTHLY_SPENT
                    FROM APP.CHARGEBACK AS A
                    INNER JOIN CHILD_BUS AS CBU 
                        ON A.BUSINESS_UNIT_ID = CBU.BU_ID
                    INNER JOIN BU_CTE BU
                        ON CBU.ROOT_BU_ID = BU.FO_BUSINESS_UNIT_ID
                    WHERE UPPER(A.CSP) = %s
                            AND A.MONTH <> %s
                    GROUP BY 
                        CBU.ROOT_BU_ID,
                        BU.BUSINESS_UNIT_NAME,
                        A.MONTH

                    UNION ALL

                    -- Total costs for all business units
                    SELECT 
                        AB.FO_BUSINESS_UNIT_ID AS BU_ID,
                        AB.BUSINESS_UNIT_NAME,
                        A.MONTH,
                        SUM(A.COST) AS MONTHLY_SPENT
                    FROM APP.CHARGEBACK AS A
                    INNER JOIN ALL_BU AB ON 1 = 1
                    WHERE UPPER(A.CSP) = %s
                        AND A.MONTH <> %s
                    GROUP BY A.MONTH,
                             AB.FO_BUSINESS_UNIT_ID,
                             AB.BUSINESS_UNIT_NAME
                    ORDER BY BU_ID, MONTH;
                    """

            log_and_print(f"Executing query with CSP={CSP.upper()} and current_month={current_month}")
            log_and_print(f"Executing query: {query}")
            cs.execute(query, (CSP.upper(), current_month, CSP.upper(), current_month))

            log_and_print("Fetching results into a pandas DataFrame.")
            bu = cs.fetch_pandas_all()
            if bu.empty:
                log_and_print("No data returned from Snowflake query", "error")
                return pd.DataFrame()

            bu['MONTH'] = pd.to_datetime(bu['MONTH'].astype(str), format='%Y%m')
            bu['MONTHLY_SPENT'] = pd.to_numeric(bu['MONTHLY_SPENT'], errors='coerce')
            bu = bu.sort_values(["BU_ID", "MONTH"]).reset_index(drop=True)

            log_and_print("Query executed successfully and data processed.")
            return bu

    except Exception as e:
        log_and_print(f"An error occurred: {str(e)}", "error")
        raise ValueError(f"An error occurred: {str(e)}")


def fetch_bu_names(conn=None):
    if not conn:
        log_and_print("Missing required parameters to fetch bu names: conn", "error")
        raise ValueError("Missing required parameters to fetch bu names: conn")

    try:
        log_and_print("Starting the query execution.")
        with conn.cursor() as cs:
            query = """
                        SELECT 
                            FO_BUSINESS_UNIT_ID AS BU_ID, 
                            BUSINESS_UNIT_NAME,
                            CASE 
                                WHEN IS_ALL_BU = TRUE THEN 'TOTAL'
                                ELSE 'BU Level'
                            END AS LEVEL
                        FROM APP.BUSINESS_UNITS
                        WHERE IS_UNALLOCATED_BU = FALSE OR IS_ALL_BU = TRUE
                    """

            log_and_print(f"Executing query: {query}", "debug")
            cs.execute(query)

            log_and_print("Fetching results into a pandas DataFrame.")
            bu = cs.fetch_pandas_all()
            if bu.empty:
                log_and_print("No data returned from Snowflake query", "error")
                return pd.DataFrame()
            return bu

    except Exception as e:
        log_and_print(f"An error occurred: {str(e)}", "error")
        raise ValueError(f"An error occurred: {str(e)}")


# ---------------- FORECASTING LOGIC ----------------
def fit_and_forecast(df, forecast_period=12, is_log=False):
    """
    Fits a Holt-Winters model and returns a forecast Series.
    is_log=True means forecast uses LOG_SPENT column and converts back via expm1.
    """
    series_name = 'LOG_SPENT' if is_log else 'MONTHLY_SPENT'

    model = ExponentialSmoothing(
        df[series_name],
        trend='add',
        seasonal=None,
        damped_trend=True
    )

    fitted = model.fit(
        smoothing_level=0.6,
        smoothing_trend=0.5,
        damping_trend=0.8,
        optimized=False
    )

    last_month = df.index.max()
    forecast_index = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=forecast_period,
        freq='MS'
    )

    forecast = fitted.forecast(forecast_period)
    forecast.index = forecast_index

    if is_log:
        forecast = np.expm1(forecast)

    return forecast


def forecast_monthly_spent_long(bu_id, temp, periods=12):
    temp = temp.copy()
    temp.set_index('MONTH', inplace=True)
    temp.index = pd.to_datetime(temp.index)
    temp = temp.asfreq('MS')

    forecast = fit_and_forecast(temp, forecast_period=periods)

    recent_mean = (
        temp['MONTHLY_SPENT'].tail(3).mean()
        if len(temp) >= 3
        else temp['MONTHLY_SPENT'].mean()
    )

    if forecast.mean() < 0.6 * recent_mean or forecast.min() < 0.4 * recent_mean:
        temp['LOG_SPENT'] = np.log1p(temp['MONTHLY_SPENT'])
        forecast = fit_and_forecast(temp, forecast_period=periods, is_log=True)

        trend_slope = temp['MONTHLY_SPENT'].diff().mean()

        if trend_slope < 0 and temp['MONTHLY_SPENT'].iloc[-1] < (recent_mean * 0.5):
            floor = 0
        else:
            floor = max(
                temp['MONTHLY_SPENT'].iloc[-1] * 0.05,
                recent_mean * 0.02,
                0
            )

        forecast = forecast.clip(lower=floor)

    actual_df = (
        temp[['MONTHLY_SPENT']]
        .rename(columns={'MONTHLY_SPENT': 'SPEND'})
        .assign(IS_FORECAST=False)
    )

    forecast_df = (
        forecast.to_frame(name='SPEND')
        .assign(IS_FORECAST=True)
    )

    combined_df = pd.concat([actual_df, forecast_df], axis=0)
    combined_df['BU_ID'] = bu_id
    combined_df = combined_df.reset_index().rename(columns={'index': 'MONTH'})

    return combined_df


def complete_time_series(temp):
    """
    Ensures continuous monthly data for a single BU.
    Missing months are added with MONTHLY_SPENT = 0.
    """
    full_idx = pd.date_range(
        start=temp["MONTH"].min(),
        end=temp["MONTH"].max(),
        freq="MS"
    )

    completed = (
        temp.set_index("MONTH")
            .reindex(full_idx)
            .rename_axis("MONTH")
            .reset_index()
    )

    completed["BU_ID"] = temp["BU_ID"].iloc[0]
    completed["BUSINESS_UNIT_NAME"] = temp["BUSINESS_UNIT_NAME"].iloc[0]
    completed["MONTHLY_SPENT"] = completed["MONTHLY_SPENT"].fillna(0)

    return completed


def clip_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df.copy()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def forecast_linear_trend(bu_id, temp, periods=12):
    """
    Simple linear extrapolation for BUs with fewer than MIN_REQUIRED_MONTHS of data.
    Uses numpy polyfit (degree 1) over available points; falls back to flat if only 1 point.
    """
    temp = temp.copy().sort_values('MONTH').reset_index(drop=True)
    temp['MONTHLY_SPENT'] = temp['MONTHLY_SPENT'].clip(lower=0)

    x = np.arange(len(temp), dtype=float)
    y = temp['MONTHLY_SPENT'].values

    if len(temp) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y[0]) if len(y) > 0 else 0.0

    last_month = temp['MONTH'].max()
    forecast_index = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=periods,
        freq='MS'
    )

    forecast_values = np.maximum(
        0,
        intercept + slope * np.arange(len(temp), len(temp) + periods)
    )

    actual_df = (
        temp[['MONTH', 'MONTHLY_SPENT']]
        .set_axis(['MONTH', 'SPEND'], axis='columns')
        .assign(IS_FORECAST=False, BU_ID=bu_id)
    )

    forecast_df = pd.DataFrame({
        'MONTH': forecast_index,
        'SPEND': forecast_values,
        'IS_FORECAST': True,
        'BU_ID': bu_id
    })

    return pd.concat([actual_df, forecast_df], ignore_index=True)


# ---------------- SNOWFLAKE UPLOAD ----------------
def upload_forecast_to_snowflake(conn, df, csp, database):
    try:
        with conn.cursor() as cs:
            cs.execute(
                f"DELETE FROM {database}.ANALYTICS.BUDGET_FORECAST WHERE CSP = %s",
                (csp,)
            )
            log_and_print(f"Deleted {cs.rowcount} existing rows for CSP='{csp}' from {database}.ANALYTICS.BUDGET_FORECAST")

        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name='BUDGET_FORECAST',
            database=database,
            schema='ANALYTICS',
            overwrite=False
        )
        log_and_print(f"Uploaded {nrows} rows to {database}.ANALYTICS.BUDGET_FORECAST in {nchunks} chunk(s). Success={success}")
    except Exception as e:
        log_and_print(f"Failed to upload forecast to Snowflake: {e}", "error")
        raise


# ---------------- MAIN ----------------
def main(CSP=""):
    conn_data_db = None
    main_success = False

    try:
        CSP = CSP.upper()
        log_and_print(f"CSP: {CSP}")

        # Create task log entry
        conn_data_db = connect_to_snowflake(dsn)
        # insert_or_update_log(conn_data_db, client_id, client_name, task_name, environment, current_date)
        log_and_print("Task log entry created")

        MIN_REQUIRED_MONTHS = 6
        log_and_print(f"MIN_REQUIRED_MONTHS: {MIN_REQUIRED_MONTHS}")

        current_month = datetime.now().strftime("%Y%m")
        log_and_print(f"Current month: {current_month}")

        bu_cost_df = fetch_merged_costs(conn, CSP, current_month)
        if bu_cost_df.empty:
            log_and_print("No data returned from Snowflake query", "error")
            raise ValueError("No data returned from Snowflake query")

        log_and_print(f"Number of rows in bu_cost_df: {len(bu_cost_df)}")
        month_counts = (
            bu_cost_df.groupby("BU_ID")["MONTH"]
            .nunique()
            .reset_index()
            .rename(columns={"MONTH": "MONTH_COUNT"})
        )

        log_and_print(f"Number of rows in month_counts: {len(month_counts)}")
        log_and_print(str(month_counts))

        month_counts["IS_LONG_HISTORY"] = month_counts["MONTH_COUNT"] >= MIN_REQUIRED_MONTHS

        df_bu = fetch_bu_names(conn)
        if df_bu.empty:
            raise ValueError("Failed to fetch bu names")

        df_bu = df_bu.merge(
            month_counts[["BU_ID", "IS_LONG_HISTORY"]],
            on="BU_ID",
            how="left"
        )
        df_bu["IS_LONG_HISTORY"] = df_bu["IS_LONG_HISTORY"].where(df_bu["IS_LONG_HISTORY"].notna(), False).astype(bool)
        df_bu = df_bu.sort_values(["BU_ID"]).reset_index(drop=True)
        log_and_print(f"Number of rows in df_bu: {len(df_bu)}")
        log_and_print(str(df_bu))

        # Forecasting
        bu_list = df_bu.loc[df_bu['IS_LONG_HISTORY'], 'BU_ID'].to_list()
        forecast_periods = 12
        final_forecasts = []

        for bu_id in bu_list:
            row = df_bu[df_bu["BU_ID"] == bu_id].iloc[0]
            bu_name = row["BUSINESS_UNIT_NAME"]

            temp = bu_cost_df[bu_cost_df["BU_ID"] == bu_id].copy()
            temp["MONTH"] = pd.to_datetime(temp["MONTH"])
            temp = complete_time_series(temp)
            temp["MONTHLY_SPENT"] = temp["MONTHLY_SPENT"].clip(lower=0)

            if temp.empty:
                log_and_print(f"No data available for BU_ID: {bu_id}")
                continue

            if temp["MONTHLY_SPENT"].max() == 0:
                log_and_print(f"\n=== Forecast for {bu_name} (BU_ID: {bu_id}) ===")
                log_and_print("All historical values are zero → Forecasting zeros.")

                last_month = temp['MONTH'].max()
                forecast_index = pd.date_range(
                    start=last_month + pd.offsets.MonthBegin(1),
                    periods=forecast_periods,
                    freq='MS'
                )

                actual_df = (
                    temp[['MONTH', 'MONTHLY_SPENT']]
                    .set_axis(['MONTH', 'SPEND'], axis='columns')
                    .assign(IS_FORECAST=False, BU_ID=bu_id)
                )

                forecast_df = pd.DataFrame({
                    "MONTH": forecast_index,
                    "SPEND": 0,
                    "IS_FORECAST": True,
                    "BU_ID": bu_id
                })

                combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)
                final_forecasts.append(combined_df)
                continue

            temp = clip_outliers_iqr(temp, "MONTHLY_SPENT")

            log_and_print(f"\n=== Forecast for {bu_name} (BU_ID: {bu_id}) Processed Successfully ===")
            forecast = forecast_monthly_spent_long(bu_id=bu_id, temp=temp, periods=forecast_periods)
            final_forecasts.append(forecast)

        # Short history: linear trend for BUs below MIN_REQUIRED_MONTHS
        short_bu_list = df_bu.loc[~df_bu['IS_LONG_HISTORY'], 'BU_ID'].to_list()
        for bu_id in short_bu_list:
            row = df_bu[df_bu["BU_ID"] == bu_id].iloc[0]
            bu_name = row["BUSINESS_UNIT_NAME"]

            temp = bu_cost_df[bu_cost_df["BU_ID"] == bu_id].copy()
            if temp.empty:
                log_and_print(f"No data for BU_ID: {bu_id} ({bu_name}) - skipping linear trend")
                continue

            temp["MONTH"] = pd.to_datetime(temp["MONTH"])
            temp = complete_time_series(temp)
            temp["MONTHLY_SPENT"] = temp["MONTHLY_SPENT"].clip(lower=0)

            log_and_print(f"\n=== Linear Trend Forecast for {bu_name} (BU_ID: {bu_id}) ===")
            forecast = forecast_linear_trend(bu_id=bu_id, temp=temp, periods=forecast_periods)
            final_forecasts.append(forecast)

        if final_forecasts:
            combined_forecast_df = pd.concat(final_forecasts, ignore_index=True)
        else:
            combined_forecast_df = pd.DataFrame()

        log_and_print(f"Number of rows in combined_forecast_df: {len(combined_forecast_df)}")

        combined_forecast_df['CSP'] = CSP
        combined_forecast_df = (
            combined_forecast_df
            .assign(
                FO_BUSINESS_UNIT_ID=lambda df: df['BU_ID'].astype(int),
                YEAR_MONTH=lambda df: df['MONTH'].dt.strftime('%Y%m'),
                SPEND=lambda df: df['SPEND'].round(2),
                FORECAST_FLAG=lambda df: df['IS_FORECAST'].astype(bool),
            )
            [['CSP', 'FO_BUSINESS_UNIT_ID', 'YEAR_MONTH', 'SPEND', 'FORECAST_FLAG']]
        )
        log_and_print(f"combined_forecast_df shaped to BUDGET_FORECAST schema: {combined_forecast_df.shape}")
        log_and_print(str(combined_forecast_df.head()))

        upload_forecast_to_snowflake(conn, combined_forecast_df, CSP, database)

        # Update log entry to success
        update_log_entry(conn_data_db, client_id, task_name, environment, current_date, "success")
        log_and_print("Task completed and log updated")
        main_success = True

    except Exception as e:
        log_and_print(f"main() failed: {e}", "error")
        if conn_data_db:
            update_log_entry(conn_data_db, client_id, task_name, environment, current_date, "failed", str(e))
        main_success = False
        raise
    finally:
        if conn_data_db:
            conn_data_db.close()
        if conn:
            conn.close()
            log_and_print("Snowflake connection closed.")

    return main_success


if __name__ == "__main__":
    CSP = 'aws'
    main(CSP)
