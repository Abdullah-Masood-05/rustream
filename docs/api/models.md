# Model weights & helpers

Utilities for managing, downloading, and verifying ONNX model files.

## Helper functions

### `download_models(target_dir="models")`
Downloads all missing ONNX model weights from their official upstream repositories.

- **Parameters**:
  - `target_dir` (*str*, optional): Target directory where model files are saved. Created automatically if it does not exist. Defaults to `"models"`.
- **Returns**: `None`
- **Raises**:
  - `RuntimeError`: If any model download fails or times out.

```python
import vigilo_stream

# Download all missing models into the default ./models directory
vigilo_stream.download_models()

# Download into a custom path
vigilo_stream.download_models("/opt/models")
```

---

## Constants

### `MODEL_URLS`
A dictionary mapping each model filename to its official download URL:

```python
import vigilo_stream

for filename, url in vigilo_stream.MODEL_URLS.items():
    print(f"{filename} -> {url}")
```

| Filename | Source repository |
| :--- | :--- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo) |
| `headpose_mobilenetv3_small.onnx` | [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation) |
| `mobileone_s0_gaze.onnx` | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) |
| `yolox_nano.onnx` | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) |
| `w600k_mbf.onnx` | [deepinsight/insightface](https://github.com/deepinsight/insightface) |
