# rustream

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.80+-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-1.15-purple)](https://github.com/PyO3/maturin)
[![Licence](https://img.shields.io/badge/licence-AGPL--3.0-blue)](LICENSE)

**Zero-copy multi-modal stream fusion engine for real-time AI pipelines.**

`rustream` extracts the lock-free sensor fusion architecture from [Vigilo](https://github.com/Abdullah-Masood-05/Vigilo) / [`vigilo-core`](https://github.com/Abdullah-Masood-05/vigilo-core) into a high-performance Python package — built for the exact moment where Python's GIL and serialization overhead break real-time AI pipelines.

- **0 bytes copied across FFI**: Frame memory is owned by Rust and shared directly with NumPy and PyTorch via Python's Buffer Protocol and `__array_interface__`.
- **60 Hz+ target fusion rate**: Lock-free triple-buffered ring buffers via `ArcSwap`.
- **Deterministic temporal fusion**: Pure temporal logic (`FusionEngine`) with asymmetric enter/exit hysteresis bands, hold timers, and decaying score accumulators.
- **Multimodal ONNX pipeline**: Integrated models for face detection (YuNet), head pose (MobileNetV3), gaze & eye-in-head (MobileGaze), prohibited objects (YOLOX-Nano), and biometric identity (ArcFace).

---

## Quickstart

```python
import rustream
import numpy as np

# 1. Zero-copy frame operations
frame = rustream.create_synthetic_frame(1280, 720, seq=1, r=255, g=0, b=0)
print(frame.width, frame.height, frame.shape) # 1280 720 (720, 1280, 3)

# Zero-copy conversion to NumPy ndarray (shares Rust memory pointer)
arr = np.asarray(frame)
assert arr.__array_interface__["data"][0] == frame.__array_interface__["data"][0]

# 2. Real-time vision pipeline
with rustream.Pipeline(models_dir="models") as pipe:
    pipe.start("camera:0")  # Or "file:video.mp4", "dir:frames_folder"
    
    while pipe.is_running():
        frame = pipe.poll_frame()
        if frame:
            img = np.asarray(frame)
            
        snapshot = pipe.snapshot()
        if snapshot:
            print(f"Faces: {snapshot.face_count}, Pose: {snapshot.head_pose}")
            
        events = pipe.events()
        for event in events:
            print(f"Violation detected: {event}")

# 3. Headless deterministic stream fusion
engine = rustream.FusionEngine()
events = engine.replay("recorded_session.jsonl")
print(f"Replayed session generated {len(events)} events.")
```

---

## Architecture

```
[Camera / RTSP / Video File]
          │
          ▼
   [FrameSource (DirectShow / FFmpeg)]
          │
          ▼
   [ArcSwap Triple-Buffered Bus]  ◄── Zero-copy pointer sharing with Python
   ┌──────┴──────┐
   ▼             ▼
[Face Worker]  [Object Worker]
YuNet+Pose+Gaze  YOLOX-Nano
   └──────┬──────┘
          ▼
     [Signals] ──► [FusionEngine (Pure Deterministic)] ──► [Events / Violations]
```

---

## Development & Building

`rustream` uses [`uv`](https://github.com/astral-sh/uv) and [`maturin`](https://github.com/PyO3/maturin):

```bash
# Setup virtual environment and install dev dependencies
uv venv
uv pip install pytest numpy maturin

# Build and install editable extension
maturin develop

# Run tests
uv run pytest -v tests/
```

---

## License

GNU Affero General Public License v3.0 (`AGPL-3.0-only`). See [LICENSE](LICENSE) for details.
