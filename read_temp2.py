"""
read_temp2.py

Minimal adapter to use 'grayscale.py' as a library without modifying its code.

Exposes:
    - select_roi(monitor=None) -> (x, y, w, h)
    - read_temperature_from_roi(x, y, w, h) -> float (°C) | None
"""

import math
import grayscale as cap


def select_roi(monitor: int | None = None):
    """
    Open a ROI selection window for the specified monitor.

    Args:
        monitor (int | None): Monitor index (1-based, per MSS). 
                              If None, uses cap.MONITOR_INDEX from config.

    Returns:
        tuple[int, int, int, int]: Absolute ROI coordinates (x, y, w, h)
                                   in the virtual desktop space.
    """
    mon_idx = cap.MONITOR_INDEX if monitor is None else int(monitor)
    return cap._select_roi_on_monitor(mon_idx)


def read_temperature_from_roi(x: int, y: int, w: int, h: int):
    """
    Capture a ROI, find the maximum grayscale pixel, and map it to temperature.

    Args:
        x, y (int): Absolute top-left coordinates of ROI in virtual desktop space.
        w, h (int): ROI width and height in pixels.

    Returns:
        float | None: Temperature in °C, or None if capture/mapping fails.
    """
    try:
        # Capture ROI as BGR image
        roi_bgr = cap._mss_region(int(x), int(y), int(w), int(h))

        # Find maximum grayscale value (filtered for gray pixels)
        max_val, _, _ = cap.find_max_pixel_and_coord(roi_bgr)

        # Map grayscale value to temperature using config constants
        T = cap.linmap(max_val, cap.VMIN, cap.VMAX, cap.TMIN, cap.TMAX)

        # Validate result: return None if invalid (NaN or infinite)
        if T is None or (isinstance(T, float) and (math.isnan(T) or math.isinf(T))):
            return None

        return float(T)

    except Exception:
        # Return None on any capture or processing error
        return None
