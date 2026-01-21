# metrics.py
import numpy as np
import pandas as pd
from typing import Dict

TRADING_DAYS_PER_YEAR = 252

def compute_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    mu = excess.mean()
    sigma = excess.std()
    if sigma == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma)


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    mu = excess.mean()
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / dd_std)


def max_drawdown(equity: pd.Series) -> float:
    cum_max = equity.cummax()
    drawdowns = equity / cum_max - 1.0
    return float(drawdowns.min())


def calmar_ratio(equity: pd.Series) -> float:
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    cagr = (1 + total_return) ** (1 / years) - 1
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def summarize_performance(equity: pd.Series) -> Dict[str, float]:
    rets = compute_returns(equity)
    return {
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "cagr": (equity.iloc[-1] / equity.iloc[0])**(
            TRADING_DAYS_PER_YEAR / len(equity)
        ) - 1.0,
    }

