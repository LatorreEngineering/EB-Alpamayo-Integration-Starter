"""
EB-Alpamayo-Integration-Starter

Professional integration toolkit for NVIDIA Alpamayo 1 Vision-Language-Action (VLA)
model in automotive research contexts.

This package provides clean, modular interfaces for:
- Loading and running inference with Alpamayo 1
- Parsing and visualizing reasoning traces
- Validating trajectories with safety checks
- Integrating with simulation environments (AlpaSim)

For research and experimentation purposes only.
NOT intended for production vehicles or safety-critical applications.
"""

__version__ = "0.1.0"
__author__ = "EB-Alpamayo Contributors"
__license__ = "Apache-2.0"

from src.alpamayo_wrapper import AlpamayoInference, InferenceResult
from src.reasoning_parser import ReasoningParser, ReasoningStep
from src.safety_checks import TrajectoryValidator, ValidationResult
from src.trajectory_viz import TrajectoryVisualizer

__all__ = [
    "AlpamayoInference",
    "InferenceResult",
    "ReasoningParser",
    "ReasoningStep",
    "TrajectoryValidator",
    "ValidationResult",
    "TrajectoryVisualizer",
]
