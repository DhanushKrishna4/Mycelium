"""Versioned, shared transition constants (see docs/transition.md)."""

from dataclasses import asdict, dataclass

GRID_SIZE = 32
CHANNELS = 16
HIDDEN_WIDTH = 64
ALPHA_THRESHOLD = 0.1
UPDATE_PROBABILITY = 0.5
MAX_GRID_SIZE = 64
MAX_STATE_ABS = 1e6
RUNTIME_VERSION = "1.0.0"
WEIGHT_SHAPES = {"W1": (48, 64), "b1": (64,), "W2": (64, 16), "b2": (16,)}


@dataclass(frozen=True)
class ModelConfig:
    channels: int = CHANNELS
    hidden_width: int = HIDDEN_WIDTH
    perception_order: str = "state,gx,gy"
    weight_layout: str = "input,output"
    alpha_threshold: float = ALPHA_THRESHOLD
    sobel_normalization: int = 8
    padding: str = "zero"
    living_rule: str = "pre_and_post_3x3"

    def to_dict(self):
        return asdict(self)


ARCHITECTURE = ModelConfig().to_dict()
