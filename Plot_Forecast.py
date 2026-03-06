import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import snowflake.connector
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from env_map import customer_env_map


def connect(customer_name):
    config = customer_env_map.get(customer_name)
    if not config:
        raise ValueError(f"No Snowflake configuration found for customer: {customer_name}")
    conn = snowflake.connector.connect(
        user=config['USER'],
        password=config['PASSWORD'],
        account=config['ACCOUNT'],
        warehouse=config['WAREHOUSE'],
        database=config['DATABASE'],
    )
    return conn, config['DATABASE']


def fetch_forecast(conn, database, csp):
    query = f"""
        SELECT
            f.FO_BUSINESS_UNIT_ID,
            b.BUSINESS_UNIT_NAME,
            f.YEAR_MONTH,
            f.SPEND,
            f.FORECAST_FLAG
        FROM {database}.ANALYTICS.BUDGET_FORECAST f
        LEFT JOIN {database}.APP.BUSINESS_UNITS b
            ON f.FO_BUSINESS_UNIT_ID = b.FO_BUSINESS_UNIT_ID
        WHERE UPPER(f.CSP) = %s
        ORDER BY f.FO_BUSINESS_UNIT_ID, f.YEAR_MONTH
    """
    with conn.cursor() as cs:
        cs.execute(query, (csp.upper(),))
        df = cs.fetch_pandas_all()

    if df.empty:
        raise ValueError(f"No forecast data found for CSP='{csp}'")

    df['MONTH'] = pd.to_datetime(df['YEAR_MONTH'].astype(str), format='%Y%m')
    df['SPEND'] = pd.to_numeric(df['SPEND'], errors='coerce')
    df['FORECAST_FLAG'] = df['FORECAST_FLAG'].astype(bool)
    return df


def plot_bu(bu_id, bu_name, bu_df, idx, total):
    actual = bu_df[~bu_df['FORECAST_FLAG']].sort_values('MONTH')
    forecast = bu_df[bu_df['FORECAST_FLAG']].sort_values('MONTH')

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    # Actual spend
    ax.plot(
        actual['MONTH'], actual['SPEND'],
        color='#4fc3f7', linewidth=2, marker='o', markersize=4,
        label='Actual'
    )

    # Forecast spend
    ax.plot(
        forecast['MONTH'], forecast['SPEND'],
        color='#ff8a65', linewidth=2, linestyle='--', marker='o', markersize=4,
        label='Forecast'
    )

    # Shaded forecast region
    if not forecast.empty:
        ax.axvspan(
            forecast['MONTH'].min(), forecast['MONTH'].max(),
            alpha=0.08, color='#ff8a65'
        )

    # Divider between actual and forecast
    if not actual.empty and not forecast.empty:
        boundary = actual['MONTH'].max()
        ax.axvline(x=boundary, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=0.6)

    # Formatting
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=35)

    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(axis='y', color='#333333', linewidth=0.6)

    ax.set_title(
        f'{bu_name}  (BU ID: {bu_id})',
        color='white', fontsize=13, fontweight='bold', pad=12
    )
    ax.set_ylabel('Monthly Spend', color='#cccccc')

    legend = ax.legend(facecolor='#1e1e2e', edgecolor='#444444', labelcolor='white')

    fig.text(
        0.99, 0.01,
        f'{idx} / {total}  —  Press Enter in the terminal to continue',
        ha='right', va='bottom', color='#666666', fontsize=8
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)


def main():
    parser = argparse.ArgumentParser(description="Plot Budget Forecast by BU from Snowflake.")
    parser.add_argument("--customer", type=str, default="dev1-wex", help="Customer name")
    parser.add_argument("--csp", type=str, default="aws", help="Cloud service provider (aws, azure, gcp)")
    args = parser.parse_args()

    conn, database = connect(args.customer.lower())
    try:
        df = fetch_forecast(conn, database, args.csp)
    finally:
        conn.close()

    bu_groups = list(df.groupby('FO_BUSINESS_UNIT_ID', sort=True))
    total = len(bu_groups)
    print(f"\nLoaded {total} BUs for CSP={args.csp.upper()}. Press Enter to step through each chart.\n")

    for idx, (bu_id, bu_df) in enumerate(bu_groups, start=1):
        bu_name = bu_df['BUSINESS_UNIT_NAME'].iloc[0] or f"BU {bu_id}"
        plot_bu(bu_id, bu_name, bu_df, idx, total)

        if idx < total:
            response = input(f"  [{idx}/{total}] {bu_name} — press Enter for next, q to quit: ").strip().lower()
            plt.close('all')
            if response == 'q':
                print("Exiting.")
                break
        else:
            print(f"  [{idx}/{total}] {bu_name} — last BU. Close the chart to exit.")
            plt.show(block=True)


if __name__ == "__main__":
    main()
