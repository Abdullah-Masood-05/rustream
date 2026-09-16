"""
vigilo_stream: Zero-copy multi-modal stream fusion engine for real-time AI pipelines.

Powered by vigilo-core, PyO3, and Maturin.
"""

from pathlib import Path
from typing import Optional
import urllib.request

from . import _core
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
    Signals,
    Violation,
    create_synthetic_frame,
)

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "headpose_mobilenetv3_small.onnx": "https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx",
    "mobileone_s0_gaze.onnx": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx",
    "yolox_nano.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
}


def download_models(dest_dir: str = "models", verbose: bool = True) -> Path:
    """Download default ONNX model weights into the specified directory."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        target = dest / filename
        if not target.exists():
            if verbose:
                print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, target)
        elif verbose:
            print(f"{filename} already exists, skipping.")
    return dest


class Pipeline:
    """High-performance multi-modal capture and detection pipeline.

    Wraps background capture workers (camera, video file, or image folder)
    and ONNX inference workers. By default, auto_download=True automatically
    downloads missing default model weights (such as YuNet face detection)
    into models_dir on first use.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        models_dir: str = "models",
        auto_download: bool = True,
    ):
        self.models_dir = models_dir
        if auto_download:
            dest = Path(models_dir)
            primary_model = dest / "face_detection_yunet_2023mar.onnx"
            if not primary_model.exists():
                download_models(dest_dir=models_dir, verbose=True)

        self._inner = _core.Pipeline(config_path=config_path, models_dir=models_dir)

    def start(self, source_spec: str):
        """Start processing from a video source specification (e.g. 'camera:0', 'file:clip.mp4')."""
        return self._inner.start(source_spec)

    def is_running(self) -> bool:
        """Check if the pipeline workers are currently running."""
        return self._inner.is_running()

    def poll_frame(self) -> Optional[Frame]:
        """Poll the most recent video frame (zero-copy RGB8 Frame)."""
        return self._inner.poll_frame()

    def snapshot(self) -> Optional[Signals]:
        """Poll the latest instantaneous detection signals without blocking."""
        return self._inner.snapshot()

    def events(self) -> list[Event]:
        """Drain and return all violation events produced since the last call."""
        return self._inner.events()

    def enrol(self):
        """Request face identity enrollment on the next frame with a detected face."""
        return self._inner.enrol()

    def is_enrolled(self) -> bool:
        """Check if a face reference has been enrolled."""
        return self._inner.is_enrolled()

    def stop(self):
        """Stop all background workers and release camera resources."""
        return self._inner.stop()

    def update_config(self, toml_str: str):
        """Hot-reload threshold configuration."""
        return self._inner.update_config(toml_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __repr__(self) -> str:
        return repr(self._inner)


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
