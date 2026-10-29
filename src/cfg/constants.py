"""Select a detector consistently in the controller and inherited subprocesses."""
import os

_family = os.environ.get("LLMGE_DETECTOR", "yolo11")
if _family == "yolo11":
    from .constants_fred_yolo11 import *
elif _family == "faster_rcnn":
    from .constants_fred_faster_rcnn import *
elif _family == "faster_rcnn_scratch":
    from .constants_fred_faster_rcnn_scratch import *
else:
    raise ValueError(f"Unsupported LLMGE_DETECTOR: {_family}")
