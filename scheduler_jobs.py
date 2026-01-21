# scheduler_jobs.py
import schedule
import time
from live_trading import run_daily_retrain_and_rebalance

WATCHLIST = ["GOOGL", "AAPL", "AMZN", "MSFT", "META", "NFLX", "TSLA"]

MODEL_PATH = "models/multi_asset_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.pkl"

def job_daily_retrain():
    run_daily_retrain_and_rebalance(WATCHLIST, MODEL_PATH, FEATURE_COLS_PATH)

def main():
    schedule.every().day.at("03:00").do(job_daily_retrain)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()

