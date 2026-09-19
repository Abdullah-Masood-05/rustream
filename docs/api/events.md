# Violations & events

Temporal proctoring events emitted by the fusion engine and pipeline.

## `Violation`
A temporal violation produced by evaluating signals across sliding time windows.

```python
class Violation:
    kind: str
    severity: str
    confidence: float
    t_start_ms: int
    t_end_ms: Optional[int]
    subject: Optional[str]
    def to_dict(self) -> dict: ...
```

### Properties

- **`kind`** (*str*): The violation category string:
  - `"no_face"`: No human face detected in frame.
  - `"multiple_faces"`: More than one face detected simultaneously.
  - `"prohibited_object"`: Unauthorized device detected (cell phone, book, laptop).
  - `"head_turned_away"`: Head pose yaw or pitch exceeded angular thresholds.
  - `"gaze_off_screen"`: Gaze direction deflected away from the screen area.
  - `"identity_mismatch"`: Current face does not match enrolled reference.
  - `"signal_lost"`: Camera feed disconnected or stalled.
- **`severity`** (*str*): Violation severity rating: `"low"`, `"medium"`, `"high"`, or `"critical"`.
- **`confidence`** (*float*): Evaluated confidence score (0.0 to 1.0).
- **`t_start_ms`** (*int*): Timestamp in milliseconds when the violation condition first started.
- **`t_end_ms`** (*Optional[int]*): Timestamp in milliseconds when the violation ended, or `None` if ongoing.
- **`subject`** (*Optional[str]*): Context string (e.g. `"cell phone"` for prohibited objects).

### Methods

- **`to_dict()`**: Serializes the violation into a dictionary.

---

## `Event`
A discrete lifecycle or alert event emitted by the pipeline.

```python
class Event:
    event_type: str
    violation: Optional[Violation]
    message: Optional[str]
    def to_dict(self) -> dict: ...
```

### Properties

- **`event_type`** (*str*):
  - `"ViolationStarted"`: A violation condition persisted past its onset hold timer. `violation` contains the active `Violation`.
  - `"ViolationEnded"`: An active violation ceased and cleared its release hold timer. `violation` contains the closed `Violation` with `t_end_ms` set.
  - `"CalibrationProgress"`: Head/gaze baseline calibration progress. `message` contains progress percentage.
  - `"CalibrationComplete"`: Baseline calibration finished.
  - `"Degraded"`: System entered degraded mode (e.g. low framerate or missing frames).
  - `"Recovered"`: System recovered from degraded mode.
- **`violation`** (*Optional[Violation]*): Associated violation object for `ViolationStarted` and `ViolationEnded`.
- **`message`** (*Optional[str]*): Informational text for calibration or degraded events.

### Methods

- **`to_dict()`**: Serializes the event into a dictionary.
