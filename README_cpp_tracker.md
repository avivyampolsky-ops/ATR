# C++ Tracker Python API (ATRVMDTrackerCpp)

This repo exposes a pure‑C++ tracking pipeline to Python via pybind11.
The class is `ATRVMDTrackerCpp` and lives in `python_modules._register_detect_cpp`.

## Build

From the repo root:

```bash
./compile_pybind_cpp.sh
```

This builds the pybind modules (including `_register_detect_cpp`) into `python_modules/`.

## Usage

```python
import numpy as np
from python_modules._register_detect_cpp import ATRVMDTrackerCpp

# reference frame (uint8, HxW or HxWx3)
reference = np.zeros((480, 640, 3), dtype=np.uint8)

# By default, it loads config.yaml from the current working directory.
tracker = ATRVMDTrackerCpp(reference)

frame = np.zeros((480, 640, 3), dtype=np.uint8)
tracks = tracker.process_frame(frame)

# Optional accessors
registered = tracker.get_last_registered_frame()
mask = tracker.get_last_mask()
```

You can also pass an explicit config path:

```python
tracker = ATRVMDTrackerCpp(reference, "path/to/config.yaml")
```

Or pass a dict with dot-notation keys (same as config.yaml):

```python
cfg = {"general.use_cuda": True, "registration.mode": "translation"}
tracker = ATRVMDTrackerCpp(reference, cfg)
```

## Notes

- `general.use_cuda` must be `True`. The pure C++ pipeline currently requires CUDA.
- `registration.reference_window_frames` can be passed directly. If it is missing or <= 0,
  it is computed from `registration.reference_window_ms` and `debug.fps`.
