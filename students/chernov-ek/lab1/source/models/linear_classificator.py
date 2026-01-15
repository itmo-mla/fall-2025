from source.models import ABCModel
from source.layers import LinearLayer
from source.activations import SignActivation


class LinearClassificator(ABCModel):
    def __init__(self, in_features: int):
        super().__init__([
            LinearLayer(in_features, 1, True),
            SignActivation()
        ])
