# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    symbols: List[str]
    period: str = "10y"
    interval: str = "1d"
    intraday: bool = False

@dataclass
class MLConfig:
    horizon: int = 1
    classification_threshold: float = 0.0
    task: str = "classification"  # or "regression" or "both"
    model_type: str = "rf"        # "rf", "gb", "xgb", "ensemble"
    use_optuna: bool = False
    n_splits: int = 5

@dataclass
class PortfolioConfig:
    start_cash: float = 10_000.0
    max_leverage: float = 2.0
    max_position_pct: float = 0.2   # max 20% of equity per asset
    top_k_assets: int = 5
    commission_per_trade: float = 0.01
    slippage_bps: float = 5.0       # 0.05%
    spread_bps: float = 10.0        # 0.1%
    max_notional_per_trade: float = 25_000.0
    pdt_equity_threshold: float = 25_000.0

@dataclass
class RiskConfig:
    use_kelly: bool = True
    kelly_scale: float = 0.5        # scale down Kelly fraction
    target_vol_annual: float = 0.20
    max_drawdown_pct: float = 0.3   # cut risk if DD > 30%

@dataclass
class RunConfig:
    data: DataConfig
    ml: MLConfig
    portfolio: PortfolioConfig
    risk: RiskConfig

