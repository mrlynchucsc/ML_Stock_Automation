# Multi-Asset ML Trading Engine

This project is a modular **multi-asset machine learning trading system** built in Python.

It started as a single-stock backtest and has been refactored into a full **ML-driven trading engine** that:

- Ingests historical OHLCV data for **dozens of tickers** (via `yfinance`)
- Engineers technical features and builds a **multi-asset dataset**
- Trains **ensemble ML models** (RandomForest, GradientBoosting, XGBoost-optional)
- Uses **walk-forward, leakage-safe** training (expanding window)
- Ranks assets and allocates a **multi-asset portfolio** with realistic trading frictions
- Computes risk/return metrics and visualizes equity curves (with optional live animation)

> ⚠️ **Disclaimer:** This is a research and paper-trading system. It is not financial advice and does not guarantee future performance.

---

## Features

**Architecture & Data**

- Modular structure:
  - `data_loader.py` – data ingestion (per-symbol & multi-symbol)
  - `features.py` – technical feature engineering & multi-asset dataset builder
  - `model.py` – ML models, cross-validation, hyperparameter utilities
  - `simulator.py` – single-asset and multi-asset simulators w/ risk metrics
  - `run_pipeline.py` – end-to-end pipeline orchestration (single vs multi-asset)
  - `config.py` – typed configuration (`RunConfig`, `DataConfig`, etc.)
- Pulls OHLCV data via **yfinance** for configurable symbol universes.

**ML Logic**

- Supports **classification** (up/down) and **regression** (future returns).
- Base models:
  - RandomForest (`rf`)
  - GradientBoosting (`gb`)
  - XGBoost (`xgb`, optional)
  - Voting ensemble (`ensemble`) combining RF + GB (+ XGB if available).
- Uses `Pipeline(StandardScaler() + model)` for scaling & training.
- **TimeSeriesSplit** and **walk-forward** training for proper temporal validation.
- Optional **GridSearchCV** / Optuna (hyperparameter tuning).
- Basic **feature selection** and leakage checks.

**Multi-Asset Portfolio Simulation**

- Ranks assets by expected return / probability of upside.
- Allocates to **top-K assets** with:
  - Global **max leverage**
  - Per-position **max equity %**
  - Support for **fractional shares**
- Models execution:
  - Commissions per trade
  - Slippage (bps)
  - Bid/ask spread (bps)
- Tracks:
  - Equity, cash, gross & net exposure
  - Number of open positions
- Computes:
  - Sharpe, Sortino
  - Max drawdown
  - Calmar ratio
  - CAGR

**Walk-Forward Training**

- Uses **expanding-window** training:
  - Initial training window = all data before a configurable date (e.g. 2015-01-01)
  - For each subsequent trading day:
    - Train on all historical data strictly before that day
    - Predict signals for that day only
- This approximates “learning with each trade” and avoids using future data.

**Visualization**

- Single-asset mode: static per-symbol equity curves.
- Multi-asset mode:
  - Live-updating equity curve during backtest (optional)
  - Static final equity curve.

---

## Project Structure

```text
.
├─ config.py             # RunConfig, DataConfig, MLConfig, PortfolioConfig, RiskConfig
├─ data_loader.py        # yfinance download helpers (single & multi)
├─ features.py           # Feature engineering & multi-asset dataset assembly
├─ model.py              # Original train_classifier + new ML utilities
├─ simulator.py          # Single-asset backtester + multi-asset portfolio simulator
├─ run_pipeline.py       # Main entrypoint: single vs multi-asset pipelines
├─ requirements.txt      # Python dependencies (optional)
└─ README.md             # This file
```

You may also have additional notebooks, scripts, or logs depending on how you extend the project.

---

## Installation

### 1. Clone / copy the repo

Place it wherever you work, e.g.:

```bash
C:\Users\<you>\Desktop\stock
```

or on macOS/Linux:

```bash
~/projects/stock
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```bash
cd C:\Users\<you>\Desktop\stock
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**

```bash
cd ~/projects/stock
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise install manually:

```bash
pip install     yfinance     pandas     numpy     scikit-learn     matplotlib     xgboost     optuna
```

> `xgboost` and `optuna` are optional. If they’re not installed, the code will fall back to non-XGB / non-Optuna paths.

---

## Configuration

All high-level settings live in `config.py` and are instantiated in `run_pipeline.py` via `RunConfig`.

Example (simplified):

```python
# run_pipeline.py (inside main())

run_cfg = RunConfig(
    data=DataConfig(
        symbols=[
            "GOOGL", "AAPL", "AMZN", "MSFT", "META",
            "NFLX", "TSLA", "IBM", "GE", "SNAP", "ARKK", "^GSPC",
        ],
        period="20y",     # need pre-2015 data for walk-forward
        interval="1d",
    ),
    ml=MLConfig(
        horizon=1,
        classification_threshold=0.0,
        task="classification",   # or "regression"
        model_type="ensemble",   # "rf", "gb", "xgb", "ensemble"
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
```

### Key knobs

- **Universe**: `data.symbols` – pick what you want the model to trade.
- **History window**: `data.period` (e.g. `"10y"`, `"20y"`) and `data.interval` (`"1d"`).
- **Task**:
  - `"classification"` – predict probability of price going up over horizon.
  - `"regression"` – predict future return over horizon.
- **Model type**:
  - `"rf"` – RandomForest
  - `"gb"` – GradientBoosting
  - `"xgb"` – XGBoost (if available)
  - `"ensemble"` – voting ensemble of RF + GB (+ XGB).
- **Risk / allocation**:
  - `start_cash` – portfolio starting value.
  - `max_leverage` – total gross exposure / equity cap.
  - `max_position_pct` – max per-position share of equity.
  - `top_k_assets` – how many names to hold at once.
  - `commission_per_trade`, `slippage_bps`, `spread_bps` – trading frictions.

---

## Running the Pipelines

### A. Multi-Asset Walk-Forward Pipeline (recommended)

This is the core, “realistic” workflow.

1. **Ensure multi-asset mode is on**

Inside `main()` (in `run_pipeline.py`):

```python
USE_MULTI_ASSET = True
LIVE_PLOT = True  # optional, for live equity curve
```

2. **Run the pipeline**

From the project root (with your venv activated):

```bash
python run_pipeline.py
```

**What happens under the hood:**

1. `download_multi()` fetches OHLCV data for all configured symbols using `yfinance`.
2. `make_multi_asset_dataset()` in `features.py`:
   - Adds technical features per symbol
   - Aligns them into a single multi-asset dataframe with symbol labels
   - Builds targets (`y_class` / `y_reg`) based on your `ml.task` and horizon.
3. `remove_bad_features()` drops obviously noisy or leaky features.
4. `build_walkforward_signals()`:
   - Uses all data before a cutoff (default `2015-01-01`) as the initial training set.
   - For each trading day after that:
     - Trains an estimator on all prior data (expanding window).
     - Predicts signals (prob_up / exp_ret) for that day.
   - Only out-of-sample predictions are kept.
5. `backtest_multi_asset()` in `simulator.py`:
   - Ranks assets by `exp_ret`.
   - Allocates to top-K assets within leverage/position constraints.
   - Applies slippage, spread, and commissions.
   - Tracks equity and other portfolio stats.
6. Console prints:
   - Sharpe, Sortino, max drawdown, Calmar, CAGR.
7. Plots:
   - If `live_plot=True`, a live-updating equity curve as the backtest advances.
   - A final static equity curve at the end.

---

### B. Single-Asset Demo (original behavior)

This mode preserves your **original single-symbol training + threshold search + backtest**.

1. In `main()`:

```python
USE_MULTI_ASSET = False
```

2. Run:

```bash
python run_pipeline.py
```

3. The script will:

   - Loop over a list of symbols.
   - For each symbol:
     - Call `run_for_symbol()`:
       - Download data
       - Build features and labels
       - Train `RandomForestClassifier` with original `train_classifier()`
       - Grid-search buy/sell probability thresholds
       - Run `backtest_classifier()` with those thresholds
   - Plot all single-symbol equity curves on one figure.

---

## Example Output

After a multi-asset run, you should see in the console:

- A summary of:
  - Number of features before/after selection.
  - Walk-forward training dates.
  - Final performance metrics, e.g.:

  ```text
  === Multi-asset backtest performance ===
  sharpe:      0.85
  sortino:     1.21
  max_drawdown:-0.44
  calmar:      0.18
  cagr:        0.08
  ```

And a figure with the equity curve. Depending on your settings, you may see:

- A drawdown during difficult regimes (e.g., 2021–2022).
- Recovery and new highs as the model adapts to the new regime.
- Live animation of the curve “drawing itself in” if `live_plot=True`.

---

## Extending to Live Trading (Roadmap)

The codebase is structured so you can bolt on:

- **Real-time data ingestion** (e.g. websockets, REST polling).
- **Broker integrations**:
  - Alpaca API
  - Interactive Brokers (IBKR) via IB-insync.
- **Scheduling**:
  - Use `cron`, `APScheduler`, or a simple loop to perform:
    - Daily training / rebalancing
    - Intraday signal generation and execution
- **Dashboards**:
  - Streamlit, Dash, or a simple Flask app to visualize:
    - Current positions
    - Intraday PnL
    - Rolling performance metrics

Live trading hooks would typically be added as a new module, e.g. `brokers/alpaca_client.py` and `live_trading.py`, which translate portfolio target weights into real orders.

---

## Standard Operating Procedure (SOP)

A more formal step-by-step SOP (setup, run, interpret, troubleshoot) is provided as a PDF.

---

## Disclaimer

This repository is for **research and educational purposes**.  
Historical performance from backtests does **not** guarantee future results.  
Use caution and perform extensive paper trading and risk reviews before considering any real capital deployment.
