# features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Literal
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ==== ORIGINAL FUNCTIONS (kept) ===============================================

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ORIGINAL: Add basic technical features.
    Assumes df has columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()

    # Returns
    df["ret_1"] = df["Close"].pct_change()
    df["ret_3"] = df["Close"].pct_change(3)

    # Moving averages
    df["ma_fast"] = df["Close"].rolling(5).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_fast"] / df["ma_slow"]

    # Volatility
    df["vol_10"] = df["ret_1"].rolling(10).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    # Volume features
    vol_ma_20 = df["Volume"].rolling(20).mean()
    df["vol_ma_20"] = vol_ma_20
    df["vol_ratio"] = df["Volume"] / vol_ma_20

    df.dropna(inplace=True)
    return df


def make_features_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0):
    """
    ORIGINAL: Build feature matrix X and label y.
    Label = 1 if future return over `horizon` bars > threshold, else 0.
    Returns: X, y, df_with_targets
    """
    df = df.copy()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    df["Close"] = close

    df["future_close"] = df["Close"].shift(-horizon)
    df["future_ret"] = (df["future_close"] - df["Close"]) / df["Close"]

    df.dropna(inplace=True)

    y = (df["future_ret"] > threshold).astype(int)

    feature_cols = [
        "ret_1", "ret_3",
        "ma_fast", "ma_slow", "ma_ratio",
        "vol_10",
        "rsi_14",
        "vol_ratio"
    ]

    X = df[feature_cols].values
    return X, y, df


# ==== NEW ADVANCED FEATURE ENGINEERING =======================================

ADVANCED_FEATURE_COLS = [
    # Original
    "ret_1", "ret_3",
    "ma_fast", "ma_slow", "ma_ratio",
    "vol_10", "rsi_14", "vol_ratio",
    # New
    "ret_5", "ret_10",
    "zscore_20",
    "hl_range",
    "log_volume",
    "vol_20",
    "vol_ratio_20",
    "drawdown_20",
]

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extends add_features with more features: momentum, volatility, z-score, etc.
    """
    df = add_features(df)  # keep original logic

    df = df.copy()
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    rolling_mean_20 = df["Close"].rolling(20).mean()
    rolling_std_20 = df["Close"].rolling(20).std()
    df["zscore_20"] = (df["Close"] - rolling_mean_20) / rolling_std_20

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    df["log_volume"] = np.log(df["Volume"] + 1)

    df["vol_20"] = df["ret_1"].rolling(20).std()
    vol_ma_20 = df["Volume"].rolling(20).mean()
    df["vol_ratio_20"] = df["Volume"] / vol_ma_20

    rolling_max = df["Close"].rolling(20).max()
    df["drawdown_20"] = df["Close"] / rolling_max - 1.0

    df.dropna(inplace=True)
    return df


def make_multi_asset_dataset(
    data_by_symbol: Dict[str, pd.DataFrame],
    horizon: int = 1,
    task: Literal["classification", "regression", "both"] = "both",
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a unified dataset from many symbols.
    Returns:
        df_all: index = DatetimeIndex, columns: features + 'symbol', 'future_ret',
                'y_class', 'y_reg' (depending on task).
        feature_cols: list of feature column names.
    """
    rows = []
    for sym, df in data_by_symbol.items():
        if len(df) < 50:
            continue
        feat = add_advanced_features(df)
        feat["symbol"] = sym

        feat["future_close"] = feat["Close"].shift(-horizon)
        feat["future_ret"] = (feat["future_close"] - feat["Close"]) / feat["Close"]

        feat.dropna(inplace=True)

        if task in ("classification", "both"):
            feat["y_class"] = (feat["future_ret"] > threshold).astype(int)
        if task in ("regression", "both"):
            feat["y_reg"] = feat["future_ret"]

        rows.append(feat)

    if not rows:
        raise ValueError("No valid data after feature generation.")

    df_all = pd.concat(rows, axis=0).sort_index()
    feature_cols = [col for col in df_all.columns
                    if col in ADVANCED_FEATURE_COLS]

    return df_all, feature_cols


def remove_bad_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    task: Literal["classification", "regression"] = "classification",
    target_col: str = "y_class",
    min_variance: float = 1e-6,
    importance_threshold: float = 0.0,
) -> List[str]:
    """
    Heuristic bad/noisy feature removal:
    - Drop near-zero variance features.
    - Train a quick RF model to compute feature importance and drop non-positive ones.
    Returns the filtered feature_cols.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    # 1) Variance threshold
    vt = VarianceThreshold(threshold=min_variance)
    X_vt = vt.fit_transform(X)
    kept_indices = np.where(vt.get_support())[0]
    kept_features = [feature_cols[i] for i in kept_indices]

    # 2) RF importance
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_vt, y)
    importances = model.feature_importances_
    final_features = [
        f for f, imp in zip(kept_features, importances) if imp > importance_threshold
    ]

    if not final_features:
        # fall back if we were too aggressive
        final_features = kept_features

    print(f"Feature selection: {len(feature_cols)} -> {len(final_features)}")
    return final_features

