import os
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui as pag


def load_lut(path_npz: str) -> dict:
    data = np.load(path_npz)
    # Ensure dtypes
    lut = data["lut"].astype(np.float32)
    T_min = float(data["T_min"]) if "T_min" in data else None
    T_max = float(data["T_max"]) if "T_max" in data else None
    vertical = bool(data["vertical"]) if "vertical" in data else True
    hot_at_start = bool(data["hot_at_start"]) if "hot_at_start" in data else True
    return {
        "lut": lut,
        "T_min": T_min,
        "T_max": T_max,
        "vertical": vertical,
        "hot_at_start": hot_at_start,
    }


def grab_gray_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    snap = pag.screenshot(region=(x, y, w, h))
    bgr = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def gray_to_temp(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # gray uint8 -> temperature using LUT
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    temps = lut[gray]
    return temps


def read_max_temp_from_roi(x: int, y: int, w: int, h: int, lut: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    gray = grab_gray_region(x, y, w, h)
    temps = gray_to_temp(gray, lut)
    idx = int(np.nanargmax(temps))
    yy, xx = np.unravel_index(idx, temps.shape)
    return float(temps[yy, xx]), (int(yy), int(xx))


def demo_once(npz_path: str, x: int, y: int, w: int, h: int):
    cfg = load_lut(npz_path)
    t, (yy, xx) = read_max_temp_from_roi(x, y, w, h, cfg["lut"])
    print(f"Max temp: {t:.2f} °C at (y={yy}, x={xx})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python gray_temp.py calibration/grayscale_lut_150_350.npz x y w h")
        sys.exit(1)
    npz_path = sys.argv[1]
    x, y, w, h = map(int, sys.argv[2:6])
    demo_once(npz_path, x, y, w, h)


