# live_trading.py
import os
import time
import joblib
import logging
from typing import List, Dict
import pandas as pd

from data_loader import download_multi
from features import make_multi_asset_dataset, remove_bad_features
from model import predict_scores
from simulator import _apply_slippage_and_spread  # reuse price adjustment


logger = logging.getLogger(__name__)


class BrokerAPI:
    """
    Skeleton for Alpaca / Interactive Brokers integration.
    Implement connect(), get_positions(), submit_order(), etc.
    """

    def __init__(self, broker: str = "alpaca"):
        self.broker = broker

    def connect(self):
        # TODO: integrate alpaca-trade-api or ib_insync with API keys from env vars
        pass

    def get_positions(self) -> Dict[str, float]:
        return {}

    def submit_order(self, symbol: str, side: str, qty: float):
        logger.info(f"Submit {side} {qty} {symbol}")


def load_model(model_path: str):
    return joblib.load(model_path)


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def run_daily_retrain_and_rebalance(
    symbols: List[str],
    model_path: str,
    feature_cols_path: str
):
    """
    Example of daily job:
    - download latest data
    - recompute features / labels
    - retrain model
    - persist model + feature_cols.
    """
    from config import RunConfig, DataConfig, MLConfig, PortfolioConfig, RiskConfig
    from model import train_with_cv
    from features import make_multi_asset_dataset

    run_cfg = RunConfig(
        data=DataConfig(symbols=symbols, period="10y", interval="1d"),
        ml=MLConfig(task="classification"),
        portfolio=PortfolioConfig(),
        risk=RiskConfig(),
    )

    data_by_symbol = download_multi(symbols, period=run_cfg.data.period,
                                    interval=run_cfg.data.interval)
    df_all, feature_cols = make_multi_asset_dataset(
        data_by_symbol,
        horizon=run_cfg.ml.horizon,
        task="classification",
        threshold=run_cfg.ml.classification_threshold,
    )
    feature_cols = remove_bad_features(
        df_all, feature_cols, task="classification", target_col="y_class"
    )
    X = df_all[feature_cols].values
    y = df_all["y_class"].values

    res = train_with_cv(
        X, y, task="classification", model_type="ensemble", n_splits=5
    )
    save_model(res.model, model_path)
    save_model(feature_cols, feature_cols_path)
    logger.info("Daily retrain complete.")


def run_live_trading_loop(
    symbols: List[str],
    model_path: str,
    feature_cols_path: str,
    broker: BrokerAPI,
    poll_interval_sec: int = 60,
):
    """
    Very high-level example loop for paper/live trading:
    - Pull fresh prices (can be via broker or market data API).
    - Build features from recent window.
    - Load model + feature cols.
    - Generate signals, compute orders.
    - Submit via broker API.
    """
    model = load_model(model_path)
    feature_cols = load_model(feature_cols_path)

    broker.connect()

    while True:
        try:
            # In production you'd use a proper real-time feed instead of yfinance
            data_by_symbol = download_multi(symbols, period="60d", interval="5m")
            df_all, _ = make_multi_asset_dataset(
                data_by_symbol, horizon=1, task="classification"
            )
            X_live = df_all[feature_cols].values
            probs, _ = predict_scores(model, X_live, task="classification")
            df_all["prob_up"] = probs

            latest = df_all.groupby("symbol").tail(1).set_index("symbol")
            # TODO: decide orders based on prob_up and position sizing logic
            logger.info(f"Latest live signals:\n{latest[['prob_up']]}")

            # Example: very naive long/flat
            positions = broker.get_positions()
            for sym, row in latest.iterrows():
                p = row["prob_up"]
                price = row["Close"]
                target_side = "buy" if p > 0.55 else "sell"
                # Compute qty based on your position sizing.
                qty = 1.0  # placeholder
                if target_side == "buy":
                    broker.submit_order(sym, "buy", qty)
                else:
                    if positions.get(sym, 0) > 0:
                        broker.submit_order(sym, "sell", positions[sym])

        except Exception as e:
            logger.exception("Error in live trading loop: %s", e)

        time.sleep(poll_interval_sec)

