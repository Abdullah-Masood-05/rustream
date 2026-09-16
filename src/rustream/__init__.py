"""
rustream: Zero-copy multi-modal stream fusion engine for real-time AI pipelines.

Powered by vigilo-core, PyO3, and Maturin.
"""

from ._core import (
    __version__,
    BBox,
    Event,
    FaceDetection,
    Frame,
    FusionEngine,
    Gaze,
    HeadPose,
    ObjectDetection,
    Pipeline,
    Signals,
    Violation,
    create_synthetic_frame,
)

__all__ = [
    "__version__",
    "BBox",
    "Event",
    "FaceDetection",
    "Frame",
    "FusionEngine",
    "Gaze",
    "HeadPose",
    "ObjectDetection",
    "Pipeline",
    "Signals",
    "Violation",
    "create_synthetic_frame",
]
