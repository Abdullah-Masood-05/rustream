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
    detect_gpu_support,
    device_info,
    download_gpu_backend,
    enable_gpu,
    is_gpu_cached,
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
    "detect_gpu_support",
    "device_info",
    "download_gpu_backend",
    "enable_gpu",
    "is_gpu_cached",
]

