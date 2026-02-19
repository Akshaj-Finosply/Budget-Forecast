from datetime import datetime
import numpy as np
import pandas as pd
import snowflake.connector
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


from pathlib import Path
import sys 

bar = "="*100

project_root = Path().resolve()
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
        return False
    
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

            # Convert the 'MONTH' column to datetime and 'MONTHLY_SPENT' to numeric
            bu['MONTH'] = pd.to_datetime(bu['MONTH'].astype(str), format='%Y%m')
            bu['MONTHLY_SPENT'] = pd.to_numeric(bu['MONTHLY_SPENT'], errors='coerce')

            # Sort the DataFrame and reset index
            bu = bu.sort_values(["BU_ID", "MONTH"]).reset_index(drop=True)

            logger.info("Query executed successfully and data processed.")
            return bu

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
 
def fetch_bu_names(conn=None):
    # Check if the connection, CSP, or current_month is not provided
    if not conn:
        logger.error("Missing required parameters to fetch bu names: conn")
        return False
    
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
            return bu

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
       
def main():
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
    if bu_cost_df is False:
        raise ValueError("Failed to fetch merged costs")
        
    logger.info(f"Number of rows in bu_cost_df: {len(bu_cost_df)}")
    logger.info(bu_cost_df)
    month_counts = (
                        bu_cost_df.groupby("BU_ID")["MONTH"]
                        .nunique()
                        .reset_index(name="MONTH_COUNT")
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


if __name__ == "__main__":
    main()