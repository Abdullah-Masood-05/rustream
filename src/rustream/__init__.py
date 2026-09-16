"""
rustream: Backward-compatibility wrapper for vigilo_stream.
"""

from vigilo_stream import (
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
    download_models,
    MODEL_URLS,
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
    "download_models",
    "MODEL_URLS",
]
