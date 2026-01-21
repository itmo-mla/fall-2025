from __future__ import annotations
from dataclasses import dataclass

@dataclass
class FitConfig:
    # core training
    n_iter: int = 10000
    lr: float = 1e-2
    lambda_q: float = 0.01
    l2: float = 0.5
    seed: int = 42

    # momentum
    use_momentum: bool = False
    gamma: float = 0.9
    nesterov: bool = False

    # fastest step (russian-style)
    use_fastest_step: bool = False
    fastest_eps: float = 1e-12
    fastest_l2_in_denom: bool = True  # makes it more stable

    # margin sampling
    use_margin_sampling: bool = False
    margin_warmup: int = 0
    margin_temperature: float = 1.0
    margin_mix: float = 0.0          # mix with uniform distribution [0..1]
    eps_sampling: float = 1e-12      # (optional) small epsilon if you want it later
