# regimes.py
import pandas as pd
import numpy as np

def detect_regimes(equity: pd.Series, window: int = 60) -> pd.Series:
    """
    Classify each point as bull/bear/sideways using rolling returns + vol.
    """
    rets = equity.pct_change()
    roll_ret = rets.rolling(window).mean()
    roll_vol = rets.rolling(window).std()

    regimes = []
    for r, v in zip(roll_ret, roll_vol):
        if np.isnan(r) or np.isnan(v):
            regimes.append("unknown")
        elif r > 0.001 and v < 0.02:
            regimes.append("bull")
        elif r < -0.001 and v > 0.02:
            regimes.append("bear")
        else:
            regimes.append("sideways")
    return pd.Series(regimes, index=equity.index, name="regime")

