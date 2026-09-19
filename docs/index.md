---
layout: home

hero:
  name: "Vigilo Stream"
  text: "Real-time AI stream fusion in Python"
  tagline: "Zero-copy video buffers, multi-worker inference, and deterministic temporal fusion backed by Rust."
  image:
    src: /logo.svg
    alt: Vigilo Stream Logo
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Python API reference
      link: /api/pipeline
    - theme: alt
      text: GitHub
      link: https://github.com/Abdullah-Masood-05/vigilo-stream

features:
  - icon: ⚡
    title: Zero-copy buffer sharing
    details: Frame memory allocated in Rust is exposed directly to NumPy and PyTorch through the buffer protocol and __array_interface__ without copying bytes across FFI.
  - icon: 🔄
    title: Lock-free frame exchange
    details: Capture workers publish frames through ArcSwap atomic slots. Processing threads read the latest frame without locks, while stale frames drop automatically.
  - icon: ⏱
    title: Deterministic temporal fusion
    details: Hysteresis bands, hold timers, and score accumulators convert instantaneous model signals into stable violations. Given identical input, replay produces identical events.
  - icon: 👁
    title: Multi-modal vision pipeline
    details: Built-in inference workers for face detection (YuNet), head pose (MobileNetV3), gaze estimation (MobileGaze), object detection (YOLOX-Nano), and identity matching (ArcFace).
  - icon: 🎯
    title: Cadence-separated workers
    details: High-frequency models run at 30 Hz for smooth tracking, while heavy object detection runs at 1 Hz in the background to preserve compute.
  - icon: 📊
    title: Live OpenCV integration
    details: Direct interop with OpenCV for real-time HUD rendering with corner reticles, 5 facial landmarks, 3D pose gizmos, gaze rays, and violation status pills.
---

<div class="vp-doc" style="max-width: 960px; margin: 40px auto 0;">

## Quick installation

::: code-group
```bash [pip]
pip install vigilo-stream
```

```bash [uv]
uv add vigilo-stream
```

```bash [poetry]
poetry add vigilo-stream
```
:::

## Example: Live camera with temporal fusion

```python
import vigilo_stream
import numpy as np

# Initialize pipeline with automatic model download on first run
with vigilo_stream.Pipeline(models_dir="models", auto_download=True) as pipe:
    pipe.start("camera:0")  # Accepts camera index, video file, or frame folder

    while pipe.is_running():
        # Zero-copy frame access: raw Rust memory exposed to NumPy
        frame = pipe.poll_frame()
        if frame:
            img = np.asarray(frame)

        # Instantaneous model output
        snapshot = pipe.snapshot()
        if snapshot:
            print(f"Faces: {snapshot.face_count}, Pose: {snapshot.head_pose}")

        # Temporal violation events from fusion engine
        for event in pipe.events():
            print(f"Event: {event.event_type} -> {event.violation}")
```

</div>
