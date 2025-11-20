import numpy as np
from typing import Callable

from .abc_model import ABCModel
from source.layers import LinearLayer
from source.activations import SignActivation
from source.losses import ABCLoss
from source.optimizers import ABCOptimizer
from source.regularizers import ABCRegularizer


class LinearClassificator(ABCModel):
    def __init__(self, in_features: int):
        super().__init__([
            LinearLayer(in_features, 1, True),
            SignActivation()
        ])
