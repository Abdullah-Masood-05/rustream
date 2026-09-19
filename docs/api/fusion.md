# FusionEngine

`vigilo_stream.FusionEngine` is a headless, deterministic temporal fusion engine. It processes instantaneous detection signals through configurable hysteresis bands, hold timers, and score accumulators to produce stable violation events.

```python
class FusionEngine:
    def __init__(self, config_toml: Optional[str] = None) -> None: ...
```

## Constructor

### `FusionEngine(...)`
Initializes a new fusion engine instance.

- **Parameters**:
  - `config_toml` (*Optional[str]*): A valid TOML string containing fusion rules and thresholds. If `None`, default fusion thresholds are used.
- **Raises**:
  - `ValueError`: If `config_toml` contains invalid TOML syntax.

---

## Methods

### `step(signals)`
Advances the fusion state machine by evaluating a single frame's detection signals.

- **Parameters**:
  - `signals` (*Signals*): The instantaneous detection signals for the current frame.
- **Returns**: `List[Event]`: Any new violation events triggered on this frame step (e.g. `ViolationStarted`, `ViolationEnded`).

```python
import vigilo_stream

engine = vigilo_stream.FusionEngine()
signals = vigilo_stream.Signals(seq=1, t_ms=33, faces=[], head_pose=None, gaze=None, objects=[])

events = engine.step(signals)
```

### `replay(jsonl_path)`
Replays an entire recorded session from a `.jsonl` file offline and returns all evaluated events.

- **Parameters**:
  - `jsonl_path` (*str*): Path to a JSONL file containing lines of serialized `Signals` records.
- **Returns**: `List[Event]`: All temporal events generated during the replayed timeline.
- **Raises**:
  - `IOError`: If the file cannot be opened or read.
  - `ValueError`: If the file contains invalid JSON lines.

```python
engine = vigilo_stream.FusionEngine()
events = engine.replay("recorded_session.jsonl")

print(f"Replay evaluated {len(events)} events.")
```

### `config()`
Retrieves the current fusion configuration as a TOML formatted string.

- **Returns**: `str`

```python
print(engine.config())
```
