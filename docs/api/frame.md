# Frame & buffer sharing

`vigilo_stream.Frame` represents an uncompressed video frame held in Rust memory. It exposes its raw memory address directly to Python using the buffer protocol and NumPy's `__array_interface__`.

```python
class Frame:
    width: int
    height: int
    seq: int
    shape: Tuple[int, int, int]
```

## Properties

### `width`
The frame width in pixels.
- **Type**: `int`

### `height`
The frame height in pixels.
- **Type**: `int`

### `seq`
The monotonically increasing frame sequence number assigned at capture time.
- **Type**: `int`

### `shape`
The 3D tensor shape `(height, width, channels)` for NumPy compatibility. Channels is always `3` for RGB24.
- **Type**: `Tuple[int, int, int]`

---

## Methods

### `to_numpy()`
Creates a NumPy array referencing the underlying Rust frame buffer without copying memory.

- **Returns**: `numpy.ndarray` with shape `(height, width, 3)` and dtype `uint8`.

```python
arr = frame.to_numpy()
```

::: tip Alternative usage
You can also pass `frame` directly to `numpy.asarray()`:
```python
arr = np.asarray(frame)
```
Both approaches use `__array_interface__` and share the identical memory pointer.
:::

---

## Protocols

### `__array_interface__`
Dictionary describing the memory layout for NumPy array construction:
- `data`: `(memory_address, read_only)`
- `shape`: `(height, width, 3)`
- `typestr`: `"|u1"` (unsigned 8-bit integer)
- `version`: `3`

### Buffer protocol
Implements the C-level Python buffer protocol, allowing `memoryview(frame)` to expose the raw bytes directly.

---

## Helper functions

### `create_synthetic_frame(width, height, seq=1, r=0, g=0, b=0)`
Constructs a synthetic RGB24 frame in Rust memory for testing, benchmarking, and headless validation.

- **Parameters**:
  - `width` (*int*): Frame width in pixels.
  - `height` (*int*): Frame height in pixels.
  - `seq` (*int*, optional): Frame sequence number. Defaults to `1`.
  - `r` (*int*, optional): Red color fill (0-255). Defaults to `0`.
  - `g` (*int*, optional): Green color fill (0-255). Defaults to `0`.
  - `b` (*int*, optional): Blue color fill (0-255). Defaults to `0`.
- **Returns**: `Frame`

```python
import vigilo_stream
import numpy as np

# Create a 720p pure green frame
frame = vigilo_stream.create_synthetic_frame(1280, 720, seq=1, r=0, g=255, b=0)
arr = np.asarray(frame)

assert arr.shape == (720, 1280, 3)
assert arr[0, 0, 1] == 255
```
