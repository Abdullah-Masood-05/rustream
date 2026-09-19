# Configuration schema

`vigilo-stream` can be configured using a TOML configuration file passed to `Pipeline(config_path="config.toml")` or updated dynamically with `pipe.update_config(toml_str)`.

## Complete TOML schema

```toml
[capture]
# Capture resolution
width = 1280
height = 720
fps = 30.0

[models]
# Model paths (auto-filled if models exist in models_dir)
face = "models/face_detection_yunet_2023mar.onnx"
pose = "models/headpose_mobilenetv3_small.onnx"
gaze = "models/mobileone_s0_gaze.onnx"
objects = "models/yolox_nano.onnx"
identity = "models/w600k_mbf.onnx"

[models.thresholds]
# Minimum confidence scores to accept detections
face_score = 0.70
object_score = 0.45
identity_match = 0.50

[cadence]
# Worker execution intervals in frames
face_interval = 1       # Run face + pose + gaze every 1 frame (~30 Hz)
objects_interval = 30   # Run object detection every 30 frames (~1 Hz)
identity_interval = 15  # Run identity verification every 15 frames (~2 Hz)

[fusion.head_pose]
# Angular limits in degrees
yaw_threshold_deg = 25.0
pitch_threshold_deg = 20.0
# Hold timers in milliseconds
onset_hold_ms = 800
release_hold_ms = 400

[fusion.gaze]
yaw_threshold_rad = 0.35
pitch_threshold_rad = 0.30
onset_hold_ms = 700
release_hold_ms = 350

[fusion.no_face]
onset_hold_ms = 1000
release_hold_ms = 200

[fusion.multiple_faces]
onset_hold_ms = 500
release_hold_ms = 300

[fusion.prohibited_objects]
# Object labels to flag as violations
monitored = ["cell phone", "laptop", "book"]
onset_hold_ms = 1000
release_hold_ms = 500
```

## Section details

### `[capture]`
Controls hardware capture parameters:
- `width` (*int*): Target horizontal resolution in pixels.
- `height` (*int*): Target vertical resolution in pixels.
- `fps` (*float*): Target capture framerate.

### `[cadence]`
Controls how often background worker threads process frames. This separation allows heavy models (e.g. YOLOX-Nano) to run without degrading smooth 30 Hz face tracking.

### `[fusion.*]`
Defines the temporal rules evaluated by `FusionEngine`:
- `onset_hold_ms`: Milliseconds the condition must persist before a violation is declared.
- `release_hold_ms`: Milliseconds the condition must remain absent before a violation is ended.
