# risk.py
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class KellyInputs:
    p: float          # win probability
    win_r: float      # avg win return
    loss_r: float     # avg loss return (abs value)


def kelly_fraction(inp: KellyInputs) -> float:
    """
    Basic Kelly fraction for a single bet.
    """
    p, win_r, loss_r = inp.p, inp.win_r, inp.loss_r
    q = 1 - p
    if loss_r <= 0:
        return 0.0
    f = (p / loss_r - q / win_r)
    return max(0.0, min(f, 1.0))


def volatility_target_weights(
    expected_vols: Dict[str, float],
    target_vol_annual: float = 0.20
) -> Dict[str, float]:
    """
    Simple vol-targeting: weight ~ 1 / vol, scaled to hit target portfolio vol.
    (Assumes independence between assets.)
    """
    inv_vol = {sym: 1.0 / max(v, 1e-6) for sym, v in expected_vols.items()}
    total_inv = sum(inv_vol.values())
    raw_weights = {sym: v / total_inv for sym, v in inv_vol.items()}
    # scaling to target_vol can be applied later; here we just normalize.
    return raw_weights

