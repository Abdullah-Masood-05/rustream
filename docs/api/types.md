# Detection & signal types

Data classes representing bounding boxes, facial landmarks, head pose orientation, gaze directions, detected objects, and frame signal snapshots.

## `BBox`
Axis-aligned 2D bounding box in source frame pixel coordinates.

```python
class BBox:
    def __init__(self, x: float, y: float, w: float, h: float) -> None: ...
    x: float
    y: float
    w: float
    h: float
    area: float
    center: Tuple[float, float]
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `x`: Top-left X coordinate.
  - `y`: Top-left Y coordinate.
  - `w`: Width in pixels.
  - `h`: Height in pixels.
  - `area`: Bounding box area (`w * h`).
  - `center`: Tuple `(cx, cy)` representing the box midpoint.
- **Methods**:
  - `to_dict()`: Serializes properties into a Python dictionary.

---

## `FaceDetection`
A detected human face with confidence score and 5 facial keypoint landmarks.

```python
class FaceDetection:
    def __init__(
        self,
        bbox: BBox,
        score: float = 1.0,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> None: ...
    bbox: BBox
    score: float
    landmarks: List[Tuple[float, float]]
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `bbox`: The face `BBox`.
  - `score`: Detection confidence score between `0.0` and `1.0`.
  - `landmarks`: List of 5 coordinates `[(x, y), ...]` in fixed order:
    1. Right eye
    2. Left eye
    3. Nose tip
    4. Right mouth corner
    5. Left mouth corner
- **Methods**:
  - `to_dict()`: Serializes detection into a dictionary.

---

## `HeadPose`
Head pose Euler angles in degrees using the aerospace yaw-pitch-roll convention.

```python
class HeadPose:
    def __init__(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> None: ...
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `yaw_deg`: Horizontal rotation (-90 to +90 degrees). Negative is turning right from subject POV, positive is turning left.
  - `pitch_deg`: Vertical nod (-90 to +90 degrees). Positive is looking up, negative is looking down.
  - `roll_deg`: Lateral tilt (-180 to +180 degrees).
- **Methods**:
  - `to_dict()`: Serializes angles into a dictionary.

---

## `Gaze`
Gaze direction in radians and relative eye-in-head deflection.

```python
class Gaze:
    def __init__(
        self,
        yaw_rad: float = 0.0,
        pitch_rad: float = 0.0,
        eye_yaw_rad: Optional[float] = None,
        eye_pitch_rad: Optional[float] = None,
    ) -> None: ...
    yaw_rad: float
    pitch_rad: float
    eye_yaw_rad: Optional[float]
    eye_pitch_rad: Optional[float]
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `yaw_rad`: Overall gaze horizontal angle in radians.
  - `pitch_rad`: Overall gaze vertical angle in radians.
  - `eye_yaw_rad`: Eyeball deflection relative to head orientation.
  - `eye_pitch_rad`: Eyeball vertical deflection relative to head orientation.
- **Methods**:
  - `to_dict()`: Serializes gaze vectors into a dictionary.

---

## `ObjectDetection`
A detected prohibited object (e.g. cell phone, laptop, book).

```python
class ObjectDetection:
    def __init__(self, label: str, score: float, bbox: BBox, class_id: int = 0) -> None: ...
    class_id: int
    label: str
    score: float
    bbox: BBox
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `class_id`: COCO class identifier.
  - `label`: Human-readable label (e.g. `"cell phone"`, `"laptop"`, `"book"`).
  - `score`: Confidence score between `0.0` and `1.0`.
  - `bbox`: Object `BBox`.
- **Methods**:
  - `to_dict()`: Serializes object detection into a dictionary.

---

## `Signals`
Instantaneous snapshot of all models' outputs for a single frame.

```python
class Signals:
    def __init__(
        self,
        seq: int = 0,
        t_ms: int = 0,
        faces: Optional[List[FaceDetection]] = None,
        head_pose: Optional[HeadPose] = None,
        gaze: Optional[Gaze] = None,
        objects: Optional[List[ObjectDetection]] = None,
        identity_match: Optional[f32] = None,
    ) -> None: ...
    seq: int
    t_ms: int
    faces: List[FaceDetection]
    face_count: int
    head_pose: Optional[HeadPose]
    gaze: Optional[Gaze]
    objects: List[ObjectDetection]
    identity_match: Optional[float]
    def to_dict(self) -> dict: ...
```

- **Properties**:
  - `seq`: Frame sequence number.
  - `t_ms`: Timestamp in milliseconds from session start.
  - `faces`: List of detected faces.
  - `face_count`: Helper returning `len(faces)`.
  - `head_pose`: Primary face head pose, or `None`.
  - `gaze`: Primary face gaze vector, or `None`.
  - `objects`: List of detected prohibited objects.
  - `identity_match`: Cosine similarity score (0.0 to 1.0) against enrolled face reference, or `None`.
- **Methods**:
  - `to_dict()`: Serializes full frame snapshot into a nested dictionary.
