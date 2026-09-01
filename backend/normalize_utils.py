"""
normalize_utils.py

Wrist-relative, scale-invariant normalization for MediaPipe hand landmarks.
Use this both to fix existing saved data AND at live inference time —
they must match, or the model sees inconsistent input.
"""

import numpy as np


def normalize_landmarks(flat):
    """
    flat: list/array of 63 floats (21 landmarks x [x, y, z]), raw MediaPipe output.
    Returns a wrist-relative, scale-invariant version of the same 63 floats.
    """
    pts = np.array(flat, dtype=np.float32).reshape(21, 3)

    # Translate: wrist (landmark 0) becomes the origin
    wrist = pts[0].copy()
    pts -= wrist

    # Scale: normalize by wrist-to-middle-MCP distance (landmark 9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero on bad detections

    pts /= scale
    return pts.flatten().tolist()