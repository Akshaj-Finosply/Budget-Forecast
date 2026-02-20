from datetime import datetime
import numpy as np
import pandas as pd
import snowflake.connector
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from snowflake.connector.pandas_tools import write_pandas

from pathlib import Path
import sys 

bar = "="*100

project_root = Path().resolve()
print(project_root)
sys.path.append(str(project_root))

from env_map import customer_env_map

pd.options.display.float_format = '{:,.2f}'.format


import logging
log_file_name = f'Budget_Forecast{datetime.now().strftime("%Y-%m-%d")}.log' 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(log_file_name),
    ]
)

logger = logging.getLogger(__name__)  # Create a logger instance for this module

logger.info(bar)
logger.info(bar)
logger.info("Beginning Logging")
def connect_and_fetch_sf_credentials(customer_name):
    try:
        config = customer_env_map.get(customer_name)
        if not config:
            raise ValueError(f"No Snowflake configuration found for customer_id: {customer_name}")
        
        return snowflake.connector.connect(
            user=config['USER'],
            password=config['PASSWORD'],
            account=config['ACCOUNT'],
            warehouse=config['WAREHOUSE'],
            database=config['DATABASE'],
        )
    except Exception as e:
        logging.error(f"Snowflake connection error for '{customer_name}': {e}", exc_info=True)
        raise  
    


def fetch_merged_costs(conn=None, CSP=None, current_month=None):
    # Check if the connection, CSP, or current_month is not provided
    if not conn or not CSP or not current_month:
        logger.error("Missing required parameters to fetch chargeback costs: conn, CSP, or current_month")
        raise ValueError("Missing required parameters to fetch chargeback costs: conn, CSP, or current_month")
    
    try:
        # Using connection context manager to ensure proper cursor handling
        logger.info("Starting the query execution.")
        with conn.cursor() as cs:
            # Query with parameterized inputs to avoid SQL injection risks
            query = f"""
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
                    WHERE 
                        UPPER(A.CSP) = '{CSP}'
                        AND A.MONTH <> {current_month}
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
                    WHERE 
                        UPPER(A.CSP) = '{CSP}'
                        AND A.MONTH <> {current_month}
                    GROUP BY A.MONTH,
                             AB.FO_BUSINESS_UNIT_ID,
                             AB.BUSINESS_UNIT_NAME
                    ORDER BY BU_ID, MONTH;
                    """

            # Execute query with parameters
            logger.info(f"Executing query with CSP={CSP.upper()} and current_month={current_month}")
            logger.info(f"Executing query: {query}")
            cs.execute(query)

            # Fetch data into a pandas DataFrame
            logger.info("Fetching results into a pandas DataFrame.")
            bu = cs.fetch_pandas_all()
            if bu.empty:
                logger.error("No data returned from Snowflake query")
                return pd.DataFrame()  
            # Convert the 'MONTH' column to datetime and 'MONTHLY_SPENT' to numeric
            bu['MONTH'] = pd.to_datetime(bu['MONTH'].astype(str), format='%Y%m')
            bu['MONTHLY_SPENT'] = pd.to_numeric(bu['MONTHLY_SPENT'], errors='coerce')

            # Sort the DataFrame and reset index
            bu = bu.sort_values(["BU_ID", "MONTH"]).reset_index(drop=True)

            logger.info("Query executed successfully and data processed.")
            return bu

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise ValueError(f"An error occurred: {str(e)}")
    
 
def fetch_bu_names(conn=None):
    # Check if the connection, CSP, or current_month is not provided
    if not conn:
        logger.error("Missing required parameters to fetch bu names: conn")
        raise ValueError("Missing required parameters to fetch bu names: conn")
    
    try:
        # Using connection context manager to ensure proper cursor handling
        logger.info("Starting the query execution.")
        with conn.cursor() as cs:
            # Query with parameterized inputs to avoid SQL injection risks
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

            # Execute query with parameters
            logger.debug(f"Executing query: {query}")
            cs.execute(query)

            # Fetch data into a pandas DataFrame
            logger.info("Fetching results into a pandas DataFrame.")
            bu = cs.fetch_pandas_all()
            if bu.empty:
                logger.error("No data returned from Snowflake query")
                return pd.DataFrame()  
            return bu

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise ValueError(f"An error occurred: {str(e)}")
    


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
    
    # Build forecast index
    last_month = df.index.max()
    forecast_index = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=forecast_period,
        freq='MS'
    )

    forecast = fitted.forecast(forecast_period)
    forecast.index = forecast_index

    # Convert back to original scale if log model was used
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

    # Refit if forecast collapses too much
    if forecast.mean() < 0.6 * recent_mean or forecast.min() < 0.4 * recent_mean:

        temp['LOG_SPENT'] = np.log1p(temp['MONTHLY_SPENT'])
        forecast = fit_and_forecast(temp, forecast_period=periods, is_log=True)

        trend_slope = temp['MONTHLY_SPENT'].diff().mean()

        # Floor logic
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
    # Full monthly index for this BU
    full_idx = pd.date_range(
        start=temp["MONTH"].min(),
        end=temp["MONTH"].max(),
        freq="MS"
    )

    # Reindex on MONTH
    completed = (
        temp.set_index("MONTH")
            .reindex(full_idx)
            .rename_axis("MONTH")
            .reset_index()
    )

    # Fill BU ID and BU name
    completed["BU_ID"] = temp["BU_ID"].iloc[0]
    completed["BUSINESS_UNIT_NAME"] = temp["BUSINESS_UNIT_NAME"].iloc[0]

    # Fill missing spent with 0
    completed["MONTHLY_SPENT"] = completed["MONTHLY_SPENT"].fillna(0)

    return completed


def plot_time_series(df, bu_name, periods=12):
    plt.figure(figsize=(10, 6))

    # Separate actuals and forecast
    actual = df[~df['IS_FORECAST']]
    forecast = df[df['IS_FORECAST']]

    # Plot actuals
    plt.plot(
        actual['MONTH'],
        actual['SPEND'],
        label='Actual',
        marker='o'
    )

    # Plot forecast if it exists
    if not forecast.empty:
        plt.plot(
            forecast['MONTH'],
            forecast['SPEND'],
            label=f'Forecast (Next {periods} Months)',
            marker='x',
            linestyle='--'
        )

        max_ylim = max(
            actual['SPEND'].max() * 1.2,
            forecast['SPEND'].max() * 1.2,
            1
        )
    else:
        max_ylim = max(actual['SPEND'].max() * 1.2, 1)

    plt.title(f'{periods}-Month Forecast of Monthly Spent for {bu_name}')
    plt.xlabel('Month')
    plt.ylabel('Monthly Spent')
    plt.ylim(0, max_ylim)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def clip_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df.copy()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df



def upload_forecast_to_snowflake(conn, df, csp):
    try:
        with conn.cursor() as cs:
            cs.execute(
                "DELETE FROM ANALYTICS.BUDGET_FORECAST WHERE CSP = %s",
                (csp,)
            )
            logger.info(f"Deleted {cs.rowcount} existing rows for CSP='{csp}' from ANALYTICS.BUDGET_FORECAST")

        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name='BUDGET_FORECAST',
            database='DEV1_WEX',
            schema='ANALYTICS',
            overwrite=False
        )
        logger.info(f"Uploaded {nrows} rows to ANALYTICS.BUDGET_FORECAST in {nchunks} chunk(s). Success={success}")
    except Exception as e:
        logger.error(f"Failed to upload forecast to Snowflake: {e}", exc_info=True)
        raise


def main():
    conn = None
    try:
        customer = 'dev1-wex'
        customer = customer.lower()
        CSP = 'aws'
        CSP = CSP.upper()
        logger.info(f"Customer: {customer}")
        logger.info(f"CSP: {CSP}")
        logger.info("Connecting to Snowflake")
        conn = connect_and_fetch_sf_credentials(customer_name=customer)
        if not conn:
            raise ValueError("Failed to connect to Snowflake")
        # Config
        MIN_REQUIRED_MONTHS = 4
        logger.info(f"MIN_REQUIRED_MONTHS: {MIN_REQUIRED_MONTHS}")
        
        
        current_month = datetime.now().strftime("%Y%m")
        logger.info(f"Current month: {current_month}")
        
        bu_cost_df = fetch_merged_costs(conn, CSP, current_month)
        if bu_cost_df.empty:
            logger.error("No data returned from Snowflake query")
            raise ValueError("No data returned from Snowflake query")

            
        logger.info(f"Number of rows in bu_cost_df: {len(bu_cost_df)}")
        logger.info(bu_cost_df)
        month_counts = (
                            bu_cost_df.groupby("BU_ID")["MONTH"]
                            .nunique()
                            .reset_index()
                            .rename(columns={"MONTH": "MONTH_COUNT"})
                        )
        
        logger.info(f"Number of rows in month_counts: {len(month_counts)}")
        logger.info(month_counts)
        
        month_counts["IS_LONG_HISTORY"] = month_counts["MONTH_COUNT"] >= MIN_REQUIRED_MONTHS
        
        df_bu = fetch_bu_names(conn)
        if df_bu is False:
            raise ValueError("Failed to fetch bu names")
        
        df_bu =  df_bu.merge(
                    month_counts[["BU_ID", "IS_LONG_HISTORY"]],
                    on="BU_ID",
                    how="left"
                )
        df_bu["IS_LONG_HISTORY"] = df_bu["IS_LONG_HISTORY"].where(df_bu["IS_LONG_HISTORY"].notna(), False).astype(bool)
        df_bu = df_bu.sort_values(["BU_ID"]).reset_index(drop=True)
        logger.info(f"Number of rows in df_bu: {len(df_bu)}")
        logger.info(df_bu)

        # Forecasting 
        bu_list = df_bu.loc[df_bu['IS_LONG_HISTORY'], 'BU_ID'].to_list()
        forecast_periods = 12
        final_forecasts = []
        
        for bu_id in bu_list:
            row = df_bu[df_bu["BU_ID"] == bu_id].iloc[0]
            bu_name = row["BUSINESS_UNIT_NAME"]

            # Extract BU dataset
            temp = bu_cost_df[bu_cost_df["BU_ID"] == bu_id].copy()
            
            temp["MONTH"] = pd.to_datetime(temp["MONTH"])
            # Clean Data and complete Time Series
            temp = complete_time_series(temp)
            # --- SAFETY CHECKS ---
            
            #No Negative Values
            temp["MONTHLY_SPENT"] = temp["MONTHLY_SPENT"].clip(lower=0)
            # 1. Check if it is empty
            if temp.empty:
                print(f"No data available for BU_ID: {bu_id}")
                continue
            
            # 2. Check if the max value is Zero then give all future values zero  
            if temp["MONTHLY_SPENT"].max() == 0:
                print(f"\n=== Forecast for {bu_name} (BU_ID: {bu_id}) ===")
                print("All historical values are zero → Forecasting zeros.")

                last_month = temp['MONTH'].max()

                forecast_index = pd.date_range(
                    start=last_month + pd.offsets.MonthBegin(1),
                    periods=forecast_periods,
                    freq='MS'
                )

                actual_df = (
                    temp[['MONTH', 'MONTHLY_SPENT']]
                    .set_axis(['MONTH', 'SPEND'], axis='columns')
                    .assign(
                        IS_FORECAST=False,
                        BU_ID=bu_id
                    )
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
            # Clipping Outliers 
            temp= clip_outliers_iqr(temp, "MONTHLY_SPENT")

            # --- NORMAL FORECASTING ---
            print(f"\n=== Forecast for {bu_name} (BU_ID: {bu_id})  Processed Successfully===")
            forecast = forecast_monthly_spent_long(bu_id=bu_id, temp=temp, periods=forecast_periods)
            final_forecasts.append(forecast)
        
        if final_forecasts:
            combined_forecast_df = pd.concat(final_forecasts, ignore_index=True)
        else:
            combined_forecast_df = pd.DataFrame()
        logger.info(f"Number of rows in combined_forecast_df: {len(combined_forecast_df)}")
        logger.info(combined_forecast_df)
        
        combined_forecast_df['CSP'] = CSP

        # Reshape to match DEV1_WEX.ANALYTICS.BUDGET_FORECAST schema
        combined_forecast_df = (
            combined_forecast_df
            .assign(
                FO_BUSINESS_UNIT_ID=lambda df: df['BU_ID'].astype(int),
                YEAR_MONTH=lambda df: df['MONTH'].dt.strftime('%Y%m'),
                SPEND=lambda df: df['SPEND'].round(2),
                FORECAST_FLAG=lambda df: df['IS_FORECAST'].astype(int),
            )
            [['CSP', 'FO_BUSINESS_UNIT_ID', 'YEAR_MONTH', 'SPEND', 'FORECAST_FLAG']]
        )
        logger.info(f"combined_forecast_df shaped to BUDGET_FORECAST schema: {combined_forecast_df.shape}")
        logger.info(combined_forecast_df.head())

        upload_forecast_to_snowflake(conn, combined_forecast_df, CSP)

    except Exception as e:
        logger.error(f"main() failed: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Snowflake connection closed.")


if __name__ == "__main__":
    main()