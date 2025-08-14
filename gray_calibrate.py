import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pyautogui as pag

from read_temp import select_roi


def _grab_bgr_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    snap = pag.screenshot(region=(x, y, w, h))
    arr = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
    return arr


def _average_along_thin_axis(img_bgr: np.ndarray, vertical: bool) -> np.ndarray:
    if vertical:
        # Vertical colorbar: average across width -> vector over height
        line = img_bgr.mean(axis=1)  # (H,3)
    else:
        # Horizontal colorbar: average across height -> vector over width
        line = img_bgr.mean(axis=0)  # (W,3)
    gray = cv2.cvtColor(line.astype(np.uint8)[None, :, :], cv2.COLOR_BGR2GRAY)[0]
    # smooth a bit to reduce dithering/noise
    gray = cv2.GaussianBlur(gray, (5, 1) if vertical else (1, 5), 0)
    return gray.astype(np.float32)


def build_intensity_to_temp_lut(colorbar_bgr: np.ndarray,
                                T_min: float,
                                T_max: float,
                                vertical: bool,
                                hot_at_start: bool) -> np.ndarray:
    line = _average_along_thin_axis(colorbar_bgr, vertical=vertical)  # shape (L,)
    L = line.shape[0]
    pos = np.linspace(0.0, 1.0, L, dtype=np.float32)  # 0=start (top or left), 1=end

    # Map intensity(0..255) -> position(0..1) via nearest neighbor on measured line
    # Ensure intensities are within [0,255]
    intensities = np.clip(line, 0, 255)

    lut_pos = np.zeros(256, dtype=np.float32)
    # For speed, build an index from rounded measured intensity to best position
    # Multiple measured samples may share the same rounded intensity; take the median pos
    idxs = [[] for _ in range(256)]
    for i, val in enumerate(intensities):
        r = int(round(val))
        r = 0 if r < 0 else (255 if r > 255 else r)
        idxs[r].append(pos[i])
    # Fill positions where we saw that intensity; others interpolate later
    for k in range(256):
        if idxs[k]:
            lut_pos[k] = float(np.median(idxs[k]))
        else:
            lut_pos[k] = np.nan

    # Interpolate missing positions (monotone fill)
    valid = np.where(~np.isnan(lut_pos))[0]
    if len(valid) < 2:
        raise RuntimeError("Not enough gradient variation detected in colorbar ROI to build LUT.")
    lut_pos = np.interp(np.arange(256), valid, lut_pos[valid])

    # Convert position -> temperature
    if hot_at_start:
        # pos=0 is hot end
        temps = T_max - lut_pos * (T_max - T_min)
    else:
        # pos=0 is cold end
        temps = T_min + lut_pos * (T_max - T_min)

    return temps.astype(np.float32)  # index by grayscale value 0..255


def main():
    print("\nGrayscale colorbar calibration (one-time).\n")
    try:
        T_min = float(input("Enter T_min [default 150]: ") or 150)
        T_max = float(input("Enter T_max [default 350]: ") or 350)
    except Exception:
        print("Invalid input; using defaults T_min=150, T_max=350")
        T_min, T_max = 150.0, 350.0

    print("\nSelect the COLORBAR ROI (draw a rectangle around the grayscale colorbar).")
    x, y, w, h = select_roi()
    colorbar = _grab_bgr_region(x, y, w, h)

    vertical_guess = h >= w
    orientation = input(f"Is the colorbar vertical? [Y/n] (default {'Y' if vertical_guess else 'N'}): ")
    if orientation.strip().lower() in ("n", "no"):
        vertical = False
    elif orientation.strip().lower() in ("y", "yes", ""):
        vertical = True
    else:
        vertical = vertical_guess

    hot_prompt = "Is the TOP (if vertical) or LEFT (if horizontal) the HOT end? [Y/n]: "
    hot_resp = input(hot_prompt)
    hot_at_start = not (hot_resp.strip().lower() in ("n", "no"))

    print("\nBuilding LUT...")
    lut = build_intensity_to_temp_lut(colorbar, T_min, T_max, vertical=vertical, hot_at_start=hot_at_start)

    calib_dir = os.path.join(os.path.dirname(__file__), "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"grayscale_lut_{int(T_min)}_{int(T_max)}"
    path_npz = os.path.join(calib_dir, f"{base}.npz")
    path_csv = os.path.join(calib_dir, f"{base}.csv")

    # Save NPZ with metadata and CSV for inspection
    np.savez(path_npz,
             lut=lut,
             T_min=np.float32(T_min),
             T_max=np.float32(T_max),
             vertical=np.bool_(vertical),
             hot_at_start=np.bool_(hot_at_start),
             created=stamp,
             roi=np.array([x, y, w, h], dtype=np.int32))

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        f.write("gray,temperature_c\n")
        for i, t in enumerate(lut):
            f.write(f"{i},{t:.6f}\n")

    print(f"Saved calibration: {path_npz}\nAlso wrote CSV preview: {path_csv}")


if __name__ == "__main__":
    main()


