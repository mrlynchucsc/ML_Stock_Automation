# run_pipeline.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import download_intraday, download_multi
from features import (
    add_features,
    make_features_labels,
    make_multi_asset_dataset,
    remove_bad_features,
)
from model import (
    train_classifier,
    train_with_cv,
    predict_scores,
    _build_base_estimator,   # use same model builder as in model.py
)
from simulator import backtest_classifier, backtest_multi_asset
from config import RunConfig, DataConfig, MLConfig, PortfolioConfig, RiskConfig


# ===== ORIGINAL SINGLE-ASSET LOGIC (kept, with minimal change) ================

def search_thresholds(df_with_targets, clf, feature_cols, start_cash=700.0):
    """
    Grid-search buy/sell thresholds to maximize final equity for ONE symbol.
    """
    best_equity = -1.0
    best_params = (0.55, 0.45)

    for buy_th in [0.52, 0.55, 0.58, 0.60]:
        for sell_th in [0.40, 0.45, 0.48, 0.50]:
            if sell_th >= buy_th:
                continue

            bt_df = backtest_classifier(
                df_with_targets,
                clf,
                feature_cols=feature_cols,
                start_cash=start_cash,
                buy_threshold=buy_th,
                sell_threshold=sell_th,
                commission_per_trade=0.01,
            )
            end_equity = bt_df["equity"].iloc[-1]

            if end_equity > best_equity:
                best_equity = end_equity
                best_params = (buy_th, sell_th)

    return best_params[0], best_params[1], best_equity


def run_for_symbol(symbol: str, start_cash: float, period: str = "10y", interval: str = "1d"):
    """
    ORIGINAL-style pipeline for a single symbol.
    """
    print(f"\n===== Running pipeline for {symbol} =====")
    try:
        raw = download_intraday(symbol=symbol, period=period, interval=interval)
    except Exception as e:
        print(f"[{symbol}] Failed to download data: {e}")
        return None

    if len(raw) < 300:
        print(f"[{symbol}] Not enough data ({len(raw)} rows), skipping.")
        return None

    print(f"[{symbol}] Adding features...")
    feat_df = add_features(raw)

    print(f"[{symbol}] Building features and labels...")
    X, y, df_with_targets = make_features_labels(feat_df, horizon=1, threshold=0.0)

    if len(df_with_targets) < 300:
        print(f"[{symbol}] Not enough rows after feature/label building, skipping.")
        return None

    print(f"[{symbol}] Training classifier...")
    clf, _ = train_classifier(X, y)

    feature_cols = [
        "ret_1", "ret_3",
        "ma_fast", "ma_slow", "ma_ratio",
        "vol_10",
        "rsi_14",
        "vol_ratio",
    ]

    print(f"[{symbol}] Searching thresholds to maximize end equity...")
    best_buy, best_sell, best_equity = search_thresholds(
        df_with_targets, clf, feature_cols, start_cash=start_cash
    )
    print(
        f"[{symbol}] Best thresholds: buy>{best_buy:.3f}, "
        f"sell<{best_sell:.3f}, tuned end equity={best_equity:.2f}"
    )

    print(f"[{symbol}] Final backtest with optimized thresholds...")
    bt_df = backtest_classifier(
        df_with_targets,
        clf,
        feature_cols=feature_cols,
        start_cash=start_cash,
        buy_threshold=best_buy,
        sell_threshold=best_sell,
        commission_per_trade=0.01,
    )

    start_equity = bt_df["equity"].iloc[0]
    end_equity = bt_df["equity"].iloc[-1]
    total_return = (end_equity - start_equity) / start_equity * 100

    print(f"[{symbol}] Start equity: {start_equity:.2f}")
    print(f"[{symbol}] End equity:   {end_equity:.2f}")
    print(f"[{symbol}] Total return: {total_return:.2f}%")
    print(f"[{symbol}] Number of trades: {(bt_df['action'] != 'HOLD').sum()}")

    return bt_df


def run_single_asset_demo():
    symbols = [
        "GOOGL", "AAPL", "AMZN", "MSFT", "META",
        "NFLX", "TSLA", "IBM", "GE", "SNAP", "ARKK", "^GSPC",
    ]
    start_cash = 700.0
    period = "10y"
    interval = "1d"

    equity_df = pd.DataFrame()
    for sym in symbols:
        bt_df = run_for_symbol(sym, start_cash=start_cash, period=period, interval=interval)
        if bt_df is None:
            continue
        equity_df[sym] = bt_df["equity"]

    if equity_df.empty:
        print("No valid single-asset results to plot.")
        return

    plt.figure(figsize=(10, 6))
    for sym in equity_df.columns:
        plt.plot(equity_df.index, equity_df[sym], label=sym)
    plt.title(f"Single-Asset Equity Curves - ${start_cash} start, {period} {interval}")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===== WALK-FORWARD SIGNAL BUILDER (PER-TRADE TRAINING) =======================

def build_walkforward_signals(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    run_cfg: RunConfig,
    initial_train_end_date: str = "2015-01-01",
    max_test_years: int = 10,
    min_train_rows: int = 200,
) -> pd.DataFrame:
    """
    Build out-of-sample signals using *incremental* walk-forward training.

    - All data strictly before `initial_train_end_date` is used as the initial
      training set (if available).
    - Testing starts at `initial_train_end_date` (or the nearest later date
      if there is no data before that).
    - For each trading day in the test period:
        * Train a model on ALL data up to that day (expanding window).
        * Predict for that day's rows (all symbols) only.
        * Move forward one day and repeat.
    - Test horizon is limited to `max_test_years` after `initial_train_end_date`
      or truncated at the last available date.

    This is closer to "learn with each trade" while still using batch training.
    """
    df = df_all.sort_index().copy()

    # Decide which target/ML task we're using
    if run_cfg.ml.task in ("classification", "both") and "y_class" in df.columns:
        task = "classification"
        target_col = "y_class"
    elif "y_reg" in df.columns:
        task = "regression"
        target_col = "y_reg"
    else:
        raise ValueError("No suitable target column (y_class / y_reg) found in df_all.")

    # Convert date boundaries
    initial_train_end = pd.to_datetime(initial_train_end_date)
    unique_dates = df.index.unique().sort_values()

    if len(unique_dates) < 100:
        raise ValueError("Not enough dates for walk-forward (need at least 100).")

    # If we don't actually have any data before the requested train end date,
    # fall back to using the first ~20% of dates as the initial training window.
    if unique_dates[0] >= initial_train_end:
        fallback_train_size = max(int(0.2 * len(unique_dates)), 50)
        initial_train_end = unique_dates[fallback_train_size - 1]
        print(
            f"[WARN] No data before requested initial_train_end_date; "
            f"using first {fallback_train_size} dates as initial training "
            f"(train <= {initial_train_end.date()})"
        )

    test_start_date = initial_train_end
    test_end_date = min(
        test_start_date + pd.DateOffset(years=max_test_years),
        unique_dates[-1],
    )

    print(
        f"Walk-forward incremental mode:\n"
        f"  initial train <= {test_start_date.date()}\n"
        f"  test from {test_start_date.date()} to {test_end_date.date()}"
    )

    # Prepare prediction columns
    df["prob_up"] = np.nan
    df["exp_ret"] = np.nan

    # Loop over each test date and re-fit using all data up to that date
    for current_date in unique_dates:
        if current_date < test_start_date or current_date > test_end_date:
            continue

        train_mask = df.index < current_date
        test_mask = df.index == current_date

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values

        if len(X_train) < min_train_rows or len(X_test) == 0:
            continue

        # Build a fresh base estimator and pipeline (no CV here for speed)
        base_est = _build_base_estimator(task, model_type=run_cfg.ml.model_type)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_est),
        ])

        pipe.fit(X_train, y_train)
        probs, exp_rets = predict_scores(pipe, X_test, task=task)

        df.loc[test_mask, "prob_up"] = probs
        df.loc[test_mask, "exp_ret"] = exp_rets

    # Keep only rows where we have out-of-sample predictions
    signals = df[df["exp_ret"].notna()].copy()

    # Build MultiIndex (date, symbol)
    signals = signals.set_index([signals.index, "symbol"])
    signals.index = signals.index.set_names(["date", "symbol"])

    return signals


# ===== NEW MULTI-ASSET PIPELINE (WALK-FORWARD) ================================

def run_multi_asset_pipeline(run_cfg: RunConfig):
    """
    Multi-asset ML-driven pipeline with *incremental* walk-forward training:
    - downloads data for many symbols
    - builds advanced features & multi-asset dataset
    - trains models in an expanding window (re-fit for each trading day)
    - generates out-of-sample signals for all assets
    - runs multi-asset portfolio backtest on those signals
    """
    symbols = run_cfg.data.symbols
    print(f"\n=== MULTI-ASSET PIPELINE (walk-forward) for {len(symbols)} symbols ===")

    # 1) Download OHLCV for all symbols
    data_by_symbol = download_multi(
        symbols,
        period=run_cfg.data.period,
        interval=run_cfg.data.interval,
        n_jobs=-1,
    )

    # 2) Build feature/label dataset
    df_all, feature_cols = make_multi_asset_dataset(
        data_by_symbol,
        horizon=run_cfg.ml.horizon,
        task=run_cfg.ml.task if run_cfg.ml.task in ("classification", "regression") else "classification",
        threshold=run_cfg.ml.classification_threshold,
    )

    # 3) Feature selection (to reduce noise / leakage)
    if run_cfg.ml.task in ("classification", "both") and "y_class" in df_all.columns:
        sel_task = "classification"
        target_col = "y_class"
    else:
        sel_task = "regression"
        target_col = "y_reg"

    print(f"Feature selection: {len(feature_cols)} -> ", end="")
    feature_cols = remove_bad_features(
        df_all,
        feature_cols,
        task=sel_task,
        target_col=target_col,
    )
    print(len(feature_cols))

    # 4) Build walk-forward, out-of-sample signals (incremental training)
    signals = build_walkforward_signals(df_all, feature_cols, run_cfg)

    # 5) Run multi-asset backtest on those signals
    bt_df = backtest_multi_asset(
        signals,
        price_col="Close",
        prob_col="prob_up",
        exp_ret_col="exp_ret",
        start_cash=run_cfg.portfolio.start_cash,
        max_leverage=run_cfg.portfolio.max_leverage,
        max_position_pct=run_cfg.portfolio.max_position_pct,
        top_k_assets=run_cfg.portfolio.top_k_assets,
        commission_per_trade=run_cfg.portfolio.commission_per_trade,
        slippage_bps=run_cfg.portfolio.slippage_bps,
        spread_bps=run_cfg.portfolio.spread_bps,
    )

    # 6) Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(bt_df.index, bt_df["equity"], label="Portfolio")
    plt.title("Multi-Asset Portfolio Equity Curve (Walk-Forward)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bt_df, signals


def main():
    # Choose which mode to run:
    USE_MULTI_ASSET = True

    if not USE_MULTI_ASSET:
        run_single_asset_demo()
        return

    run_cfg = RunConfig(
        data=DataConfig(
            symbols=[
                "GOOGL", "AAPL", "AMZN", "MSFT", "META",
                "NFLX", "TSLA", "IBM", "GE", "SNAP", "ARKK", "^GSPC",
            ],
            period="20y",   # need pre-2015 data for your requested train/test split
            interval="1d",
        ),
        ml=MLConfig(
            horizon=1,
            classification_threshold=0.0,
            task="classification",  # or "regression"
            model_type="ensemble",  # rf / gb / xgb / ensemble
            use_optuna=False,
            n_splits=5,
        ),
        portfolio=PortfolioConfig(
            start_cash=10_000.0,
            max_leverage=2.0,
            max_position_pct=0.2,
            top_k_assets=5,
            commission_per_trade=0.01,
            slippage_bps=5.0,
            spread_bps=10.0,
            max_notional_per_trade=25_000.0,
        ),
        risk=RiskConfig(
            use_kelly=True,
            kelly_scale=0.5,
            target_vol_annual=0.20,
            max_drawdown_pct=0.3,
        ),
    )

    run_multi_asset_pipeline(run_cfg)


if __name__ == "__main__":
    main()

