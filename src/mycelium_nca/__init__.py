"""Mycelium's data-only NumPy runtime; importing this never imports PyTorch."""

__version__ = "1.0.0"

from .config import CHANNELS, GRID_SIZE, HIDDEN_WIDTH
from .numpy_sim import seed_state, step

__all__ = ["CHANNELS", "GRID_SIZE", "HIDDEN_WIDTH", "seed_state", "step"]
