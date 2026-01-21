# simulator.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional

from metrics import summarize_performance
from regimes import detect_regimes


@dataclass
class PortfolioState:
    cash: float
    shares: float
    equity: float
    position: int  # 0 = flat, 1 = long


def backtest_classifier(
    df: pd.DataFrame,
    clf,
    feature_cols,
    start_cash: float = 50.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    commission_per_trade: float = 0.0,
    max_trades_per_window: int = 3,
    window_size: int = 5,
    pdt_equity_threshold: float = 25000.0
) -> pd.DataFrame:
    """
    ORIGINAL SINGLE-SYMBOL BACKTEST (unchanged in behavior).
    Uses fractional shares and PDT-like trade frequency cap below a threshold.
    """
    df_bt = df.copy()
    X = df_bt[feature_cols].values

    probs = clf.predict_proba(X)[:, 1]

    cash = start_cash
    shares = 0.0
    position = 0

    equity_curve = []
    positions = []
    actions = []
    p_ups = []

    close_prices = df_bt["Close"].values

    entry_indices = deque()  # BUY bar indices

    for i in range(len(df_bt)):
        price = close_prices[i]
        p_up = probs[i]
        p_ups.append(p_up)

        equity_before = cash + shares * price
        action = "HOLD"

        # PDT threshold logic
        if equity_before < pdt_equity_threshold:
            while entry_indices and i - entry_indices[0] >= window_size:
                entry_indices.popleft()
            recent_entries = len(entry_indices)
            can_enter = recent_entries < max_trades_per_window
        else:
            can_enter = True

        if position == 0 and p_up > buy_threshold and can_enter:
            available_cash = cash - commission_per_trade
            if available_cash > 0:
                shares_to_buy = available_cash / price
                if shares_to_buy * price > 0.01:
                    cost = shares_to_buy * price + commission_per_trade
                    cash -= cost
                    shares += shares_to_buy
                    position = 1
                    action = f"BUY {shares_to_buy:.4f}"
                    if equity_before < pdt_equity_threshold:
                        entry_indices.append(i)

        elif position == 1 and p_up < sell_threshold:
            if shares > 0:
                proceeds = shares * price - commission_per_trade
                cash += proceeds
                action = f"SELL {shares:.4f}"
                shares = 0.0
                position = 0

        equity = cash + shares * price
        equity_curve.append(equity)
        positions.append(position)
        actions.append(action)

    df_bt["equity"] = equity_curve
    df_bt["position"] = positions
    df_bt["action"] = actions
    df_bt["p_up"] = p_ups

    return df_bt


# === NEW MULTI-ASSET PORTFOLIO SIMULATOR ======================================

@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float


@dataclass
class MultiAssetState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0


def _apply_slippage_and_spread(
    price: float,
    side: str,
    slippage_bps: float,
    spread_bps: float
) -> float:
    """
    Adjust price for slippage + half-spread.
    side: "buy" or "sell"
    """
    slippage = slippage_bps / 10_000.0
    half_spread = spread_bps / 20_000.0  # half on each side
    if side == "buy":
        return price * (1 + slippage + half_spread)
    else:
        return price * (1 - slippage - half_spread)


def backtest_multi_asset(
    signals: pd.DataFrame,
    price_col: str = "Close",
    prob_col: str = "prob_up",
    exp_ret_col: str = "exp_ret",
    start_cash: float = 10_000.0,
    max_leverage: float = 2.0,
    max_position_pct: float = 0.2,
    top_k_assets: int = 5,
    commission_per_trade: float = 0.01,
    slippage_bps: float = 5.0,
    spread_bps: float = 10.0,
    stop_loss_pct: float = 0.1,
    take_profit_pct: float = 0.3,
    trailing_stop_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Multi-asset backtest:
    - signals: MultiIndex (date, symbol) DataFrame containing:
        - price_col (Close)
        - prob_col (probability of up, if classification; optional)
        - exp_ret_col (expected return from regression or proxy)
    - picks top_k_assets each day by score = prob * exp_ret (or exp_ret alone).
    - enforces leverage, position limits, commissions, slippage, spread.
    - applies basic risk management: stop loss, take profit, trailing stop.

    Returns signals_df enhanced with portfolio columns: equity, cash, position_value, etc.
    """
    if not isinstance(signals.index, pd.MultiIndex):
        raise ValueError("signals index must be MultiIndex (date, symbol)")

    signals = signals.sort_index().copy()

    dates = signals.index.get_level_values(0).unique()
    state = MultiAssetState(cash=start_cash, positions={})
    equity_series = []
    cash_series = []
    position_value_series = []
    date_list = []

    trailing_highs: Dict[str, float] = {}

    for dt in dates:
        day_slice = signals.xs(dt, level=0)
        # Score assets
        if prob_col in day_slice.columns and exp_ret_col in day_slice.columns:
            score = day_slice[prob_col].fillna(0.5) * day_slice[exp_ret_col].fillna(0)
        elif exp_ret_col in day_slice.columns:
            score = day_slice[exp_ret_col].fillna(0)
        else:
            raise ValueError("signals must contain exp_ret_col (and optionally prob_col).")

        day_slice = day_slice.copy()
        day_slice["score"] = score
        day_slice = day_slice.sort_values("score", ascending=False)

        # Calculate current equity
        position_value = 0.0
        for sym, pos in list(state.positions.items()):
            if sym in day_slice.index:
                price = day_slice.loc[sym, price_col]
            else:
                # if symbol missing data for this day, skip
                continue
            position_value += pos.shares * price

        equity = state.cash + position_value
        max_gross_exposure = equity * max_leverage

        # Risk management on existing positions (stop loss / TP / trailing stop)
        to_close = []
        for sym, pos in state.positions.items():
            if sym not in day_slice.index:
                continue
            price = day_slice.loc[sym, price_col]
            ret_since_entry = (price / pos.entry_price) - 1.0

            # Trailing high update
            if sym not in trailing_highs:
                trailing_highs[sym] = pos.entry_price
            trailing_highs[sym] = max(trailing_highs[sym], price)
            trailing_dd = (price / trailing_highs[sym]) - 1.0

            if ret_since_entry <= -stop_loss_pct:
                to_close.append(sym)
            elif ret_since_entry >= take_profit_pct:
                to_close.append(sym)
            elif trailing_dd <= -trailing_stop_pct:
                to_close.append(sym)

        # Close triggered positions
        for sym in to_close:
            if sym not in day_slice.index:
                continue
            price = day_slice.loc[sym, price_col]
            fill_price = _apply_slippage_and_spread(
                price, "sell", slippage_bps, spread_bps
            )
            pos = state.positions.pop(sym)
            proceeds = pos.shares * fill_price - commission_per_trade
            state.cash += proceeds

        # Recompute after closing
        position_value = 0.0
        for sym, pos in state.positions.items():
            if sym in day_slice.index:
                price = day_slice.loc[sym, price_col]
                position_value += pos.shares * price
        equity = state.cash + position_value
        max_gross_exposure = equity * max_leverage

        # Target new positions in top_k by score
        target_syms = day_slice.index[:top_k_assets]
        target_weights = {}
        if equity > 0:
            # equal weights across chosen assets subject to max_position_pct
            equal_w = min(1.0 / len(target_syms), max_position_pct)
            for sym in target_syms:
                target_weights[sym] = equal_w

        # Adjust positions toward target weights
        for sym in target_syms:
            if sym not in day_slice.index:
                continue
            price = day_slice.loc[sym, price_col]
            target_value = target_weights[sym] * equity
            curr_shares = state.positions.get(sym, Position(sym, 0.0, price)).shares
            curr_value = curr_shares * price
            delta_value = target_value - curr_value

            if abs(delta_value) < 1.0:  # skip dust
                continue

            if delta_value > 0:
                # Buy
                fill_price = _apply_slippage_and_spread(
                    price, "buy", slippage_bps, spread_bps
                )
                affordable_value = min(delta_value, max_gross_exposure - position_value)
                affordable_value = min(affordable_value, state.cash - commission_per_trade)
                if affordable_value <= 0:
                    continue
                shares_to_buy = affordable_value / fill_price
                cost = shares_to_buy * fill_price + commission_per_trade
                if cost <= state.cash:
                    state.cash -= cost
                    new_shares = curr_shares + shares_to_buy
                    state.positions[sym] = Position(
                        symbol=sym,
                        shares=new_shares,
                        entry_price=price if curr_shares == 0 else (
                            (curr_shares * pos.entry_price + shares_to_buy * price)
                            / new_shares
                        ),
                    )
            else:
                # Sell
                shares_to_sell = min(curr_shares, abs(delta_value) / price)
                if shares_to_sell <= 0:
                    continue
                fill_price = _apply_slippage_and_spread(
                    price, "sell", slippage_bps, spread_bps
                )
                proceeds = shares_to_sell * fill_price - commission_per_trade
                state.cash += proceeds
                remaining_shares = curr_shares - shares_to_sell
                if remaining_shares <= 0:
                    state.positions.pop(sym, None)
                else:
                    state.positions[sym] = Position(
                        symbol=sym,
                        shares=remaining_shares,
                        entry_price=state.positions[sym].entry_price,
                    )

        # End-of-day equity snapshot
        position_value = 0.0
        for sym, pos in state.positions.items():
            if sym in day_slice.index:
                price = day_slice.loc[sym, price_col]
                position_value += pos.shares * price
        equity = state.cash + position_value

        date_list.append(dt)
        equity_series.append(equity)
        cash_series.append(state.cash)
        position_value_series.append(position_value)

    equity_ser = pd.Series(equity_series, index=date_list, name="equity")
    cash_ser = pd.Series(cash_series, index=date_list, name="cash")
    pos_val_ser = pd.Series(position_value_series, index=date_list, name="position_value")

    summary = summarize_performance(equity_ser)
    print("=== Multi-asset backtest performance ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    regimes = detect_regimes(equity_ser)

    result_df = pd.concat([equity_ser, cash_ser, pos_val_ser, regimes], axis=1)
    return result_df

