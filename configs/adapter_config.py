from typing import List, Literal

AdapterLocation = Literal["mlp", "conv", "both"]

class AdapterConfig:
    def __init__(
        self,
        place_on: AdapterLocation = "mlp",
        layer_mode: str = "all",      # "all", "every", "last_k", "custom"
        layer_indices: List[int] = None,
        every: int = 2,
        last_k: int = None,
    ):
        self.place_on = place_on
        self.layer_mode = layer_mode
        self.layer_indices = layer_indices
        self.every = every
        self.last_k = last_k
