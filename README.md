# rustream

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.80+-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-1.15-purple)](https://github.com/PyO3/maturin)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Zero-copy multi-modal stream fusion engine for real-time AI pipelines in Python.

`rustream` provides Python bindings for the stream fusion engine in [`vigilo-core`](https://github.com/Abdullah-Masood-05/vigilo-core). It gives Python vision and proctoring pipelines direct access to video frames and temporal rule evaluation without copying memory across the FFI boundary.

- Zero-copy buffer sharing: Frame memory allocated in Rust is exposed directly to NumPy and PyTorch through `__array_interface__` and the buffer protocol.
- Lock-free frame exchange: Capture workers publish frames through `ArcSwap` slots, discarding stale frames automatically instead of building queues.
- Deterministic temporal fusion: The `FusionEngine` processes detection signals through configurable hysteresis bands, hold timers, and score accumulators. Given the same input, replay produces identical events.
- Multimodal detection: Wraps the `vigilo-core` inference pipeline for face detection (YuNet), head pose (MobileNetV3), gaze estimation (MobileGaze), object detection (YOLOX-Nano), and identity matching (ArcFace).

## Quick start

```python
import rustream
import numpy as np

# 1. Zero-copy frame operations
frame = rustream.create_synthetic_frame(1280, 720, seq=1, r=255, g=0, b=0)
print(frame.width, frame.height, frame.shape) # 1280 720 (720, 1280, 3)

# Expose Rust memory directly as a NumPy array without copying
arr = np.asarray(frame)
assert arr.__array_interface__["data"][0] == frame.__array_interface__["data"][0]

# 2. Vision and proctoring pipeline
with rustream.Pipeline(models_dir="models") as pipe:
    pipe.start("camera:0")  # Accepts "camera:0", "file:clip.mp4", or "dir:frames/"

    while pipe.is_running():
        frame = pipe.poll_frame()
        if frame:
            img = np.asarray(frame)

        snapshot = pipe.snapshot()
        if snapshot:
            print(f"Faces: {snapshot.face_count}, Pose: {snapshot.head_pose}")

        events = pipe.events()
        for event in events:
            print(f"Violation: {event}")

# 3. Headless deterministic stream fusion
engine = rustream.FusionEngine()
events = engine.replay("recorded_session.jsonl")
print(f"Replayed session produced {len(events)} events.")
```

## Architecture

```
Camera / Video File / Image Directory
                 │
                 ▼
  FrameSource (DirectShow / FFmpeg)
                 │
                 ▼
     ArcSwap Latest-Frame Slot  ◄── Zero-copy pointer sharing with NumPy
        ┌────────┴────────┐
        ▼                 ▼
   Face Worker      Object Worker
  YuNet+Pose+Gaze     YOLOX-Nano
        └────────┬────────┘
                 ▼
              Signals ──► FusionEngine ──► Events / Violations
```

## Building from source

Requirements:
- Rust 1.80 or newer
- Python 3.9 or newer
- C++ build tools (MSVC on Windows, GCC/Clang on Linux and macOS)

```bash
# Set up a virtual environment and install build tools
uv venv
uv pip install maturin pytest numpy

# Build and install the extension into the active environment
uv run maturin develop

# Run the test suite
uv run pytest -v tests/
```

## Release notes

### v0.1.0

- Initial release of `rustream` targeting Python 3.9 through 3.13.
- Implemented `Frame` with `__array_interface__` and `memoryview()` support for zero-copy NumPy interop.
- Implemented `FusionEngine` with single-frame stepping and deterministic JSONL log replay.
- Implemented `Pipeline` context manager wrapping camera capture, detection workers, and event polling.
- Added data bindings for `BBox`, `FaceDetection`, `HeadPose`, `Gaze`, `ObjectDetection`, `Signals`, `Violation`, and `Event`.
- Multi-platform CI testing across Windows, Ubuntu, and macOS.

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.
