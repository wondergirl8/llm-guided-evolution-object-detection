"""Event-frame renderer evolved by LLM-GE (FRED_SEED_FAMILY=renderer).

phase1_detection/render_runner.py loads this file, draws every frame of the
challenging search split with it, trains the fixed YOLO11n seed on those
frames and scores it. Fitness is (mAP50, mAP50:95, WINDOW_US): accurate frames
from a short window are best, because a frame can be produced sooner.

Contract, checked by the runner:
    WINDOW_US  int, 1000 to 50000: how many microseconds of events a frame uses.
    render(events, window_us) -> numpy uint8 array of shape (720, 1280, 3).
        events is a numpy structured array holding only the events inside the
        window, sorted by time, with fields
            x  uint16, 0..1279    column
            y  uint16, 0..719     row
            p  int16,  1 or 0     1 = pixel got brighter, 0 = pixel got darker
            t  int64,  -window_us < t < 0, microseconds before the frame time
        window_us is WINDOW_US. The labels are fixed, so the drone must stay
        where it is in the image: no flipping, cropping, shifting or resizing.

The seed reproduces FRED's released 33.3 ms frames exactly.
"""

import numpy as np

WIDTH, HEIGHT = 1280, 720

# --OPTION--
# FRED's released frames: 33.3 ms of events, each pixel coloured by its latest
# event (white = brighter, blue = darker) on Metavision's dark background.
WINDOW_US = 33333


def render(events, window_us):
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = (30, 37, 52)
    colors = np.where((events["p"] == 1)[:, None], (255, 255, 255), (64, 126, 200))
    # Fancy assignment keeps the last write per pixel, i.e. the latest event.
    image[events["y"], events["x"]] = colors
    return image
