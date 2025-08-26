"""
grayscale.py

This script captures a specific screen region or monitor,
filters for pure grayscale pixels, finds the maximum pixel value,
and maps it to a temperature using a linear mapping.
It can run once or in a continuous loop.

Author: NICOLAS MUNOZ
Date: YYYY-MM-DD
"""

import time
import mss
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =====================================================
# ================= CONFIGURATION =====================
# Select ROI method: True = select with mouse, False = fixed coordinates
USE_MOUSE = False

# Monitor index (1 = primary, 2 = secondary, etc.)
MONITOR_INDEX = 2  # Change to 1 or 2

# Fixed ROI (only used if USE_MOUSE = False)
# Absolute coordinates in virtual desktop space
ROI_X, ROI_Y, ROI_W, ROI_H =1065,-806,135,245 #  332,-979,1479,698 #-1591, 115, 1449, 699

# Run mode: = continuous loop, False = one-shot capture
RUN_LOOP = False
FPS = 30.0  # Capture frequency in Hz (only for RUN_LOOP = True)

# Grayscale-to-temperature mapping (linear)
VMIN, VMAX = 0.0, 255.0
TMIN, TMAX = 0.0, 350.0

# Tolerance for considering a pixel pure gray (R≈G≈B)
GRAY_TOL = 12
# =====================================================


# -------------------- Screen Capture Helpers --------------------
def _mss_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Capture an absolute region (x, y, w, h) and return as BGR image."""
    with mss.mss() as sct:
        raw = np.array(sct.grab({"left": x, "top": y, "width": w, "height": h}))
    return raw[..., :3]  # Remove alpha channel


def _list_monitors():
    """Return a list of detected monitors (mss format)."""
    with mss.mss() as sct:
        return sct.monitors  # Index 0 = virtual desktop, 1+ = physical monitors


def _grab_monitor(monitor_idx: int):
    """Capture an entire monitor and return (BGR image, region dict)."""
    mons = _list_monitors()
    if monitor_idx < 1 or monitor_idx >= len(mons):
        raise ValueError(f"Invalid monitor {monitor_idx}. Available: 1..{len(mons)-1}")
    region = mons[monitor_idx]
    with mss.mss() as sct:
        raw = np.array(sct.grab(region))
    return raw[..., :3], region


def _select_roi_on_monitor(monitor_idx: int):
    """Let the user select a ROI with the mouse on the chosen monitor."""
    img_bgr, mon = _grab_monitor(monitor_idx)
    win = "Select ROI (ENTER: confirm, ESC: cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img_bgr)
    x, y, w, h = cv2.selectROI(win, img_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    if w == 0 or h == 0:
        raise RuntimeError("ROI selection cancelled or empty.")

    # Convert to absolute coordinates in virtual desktop space
    x_abs = int(mon["left"] + x)
    y_abs = int(mon["top"] + y)
    return (x_abs, y_abs, int(w), int(h))


# -------------------- Analysis Functions --------------------
def find_max_pixel_and_coord(bgr_img: np.ndarray):
    """
    Convert to grayscale, but keep only pixels where R≈G≈B within GRAY_TOL.
    Other pixels are set to 0.
    """
    b, g, r = cv2.split(bgr_img)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    mask = (maxc - minc) <= GRAY_TOL

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_masked = np.where(mask, gray, 0).astype(np.uint8)

    _, max_val, _, max_loc = cv2.minMaxLoc(gray_masked)
    return int(max_val), (int(max_loc[0]), int(max_loc[1])), gray_masked


def linmap(value: float, vmin: float, vmax: float, tmin: float, tmax: float) -> float:
    """Map value linearly from range [vmin, vmax] to [tmin, tmax]."""
    if vmax == vmin:
        return float('nan')
    alpha = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    return tmin + alpha * (tmax - tmin)


def show_popup(roi_bgr, max_val, max_xy, t_value, vmin, vmax, tmin, tmax):
    """Display ROI image with max pixel marked and grayscale scale bar."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    mx, my = max_xy

    fig = plt.figure(figsize=(8, 4.8), dpi=100)
    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)

    # ROI with marker
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(roi_rgb, origin='upper')
    ax_img.scatter([mx], [my], s=120, facecolors='none', edgecolors='red', linewidths=2)
    ax_img.set_title("ROI (max pixel marked)")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    info = f"Max: {max_val} (x={mx}, y={my})\nT={t_value:.2f} °C\n" \
           f"Map: [{vmin},{vmax}] → [{tmin}°C,{tmax}°C]"
    ax_img.text(0.02, 0.98, info, transform=ax_img.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=6))

    # Grayscale bar
    ax_bar = fig.add_subplot(gs[0, 1])
    grad = np.linspace(0, 255, 256, dtype=np.uint8)[:, None]
    grad_img = np.repeat(grad, 40, axis=1)
    ax_bar.imshow(grad_img, cmap='gray', vmin=0, vmax=255,
                  extent=[0, 1, 0, 255], aspect='auto', origin='lower')
    ax_bar.set_title("Scale (0–255)")
    ax_bar.set_xlim(0, 1); ax_bar.set_xticks([])
    ax_bar.set_ylabel("Gray level")
    ax_bar.set_yticks([0, 64, 128, 192, 255])
    ax_bar.axhline(int(np.clip(max_val, 0, 255)), linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.show()


# -------------------- Execution Modes --------------------
def one_shot(roi_abs):
    """Capture ROI once and display max pixel info."""
    x, y, w, h = roi_abs
    roi_bgr = _mss_region(x, y, w, h)
    max_val, (mx, my), _ = find_max_pixel_and_coord(roi_bgr)
    T = linmap(max_val, VMIN, VMAX, TMIN, TMAX)
    show_popup(roi_bgr, max_val, (mx, my), T, VMIN, VMAX, TMIN, TMAX)

    print(f"ROI abs: x={x} y={y} w={w} h={h}")
    print(f"Max pixel: {max_val} at (x={mx}, y={my}) (ROI coords)")
    print(f"Temperature: {T:.3f} °C  (map [{VMIN},{VMAX}] → [{TMIN},{TMAX}] °C)")


def loop_mode(roi_abs):
    """Continuously capture ROI and print max pixel info in real-time."""
    period = 1.0 / max(FPS, 0.001)
    x, y, w, h = roi_abs
    try:
        while True:
            t0 = time.time()
            roi_bgr = _mss_region(x, y, w, h)
            max_val, (mx, my), _ = find_max_pixel_and_coord(roi_bgr)
            T = linmap(max_val, VMIN, VMAX, TMIN, TMAX)
            print(f"T={T:.2f}°C  max={max_val}  pos=({mx},{my})   ", end="\r", flush=True)
            sleep_time = period - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[STOP]")


# -------------------- Main Entry Point --------------------
def main():
    """Main function: configure ROI and start capture."""
    # Show detected monitors
    mons = _list_monitors()
    print(f"[mss] Detected monitors: {len(mons)-1}")
    for i in range(1, len(mons)):
        m = mons[i]
        print(f"  Monitor {i}: left={m['left']} top={m['top']} "
              f"width={m['width']} height={m['height']}")

    # Get ROI based on configuration
    roi_abs = _select_roi_on_monitor(MONITOR_INDEX) if USE_MOUSE else (ROI_X, ROI_Y, ROI_W, ROI_H)
    print(f"[CONFIG] USE_MOUSE={USE_MOUSE}  MONITOR_INDEX={MONITOR_INDEX}  RUN_LOOP={RUN_LOOP}  FPS={FPS}")
    print(f"[ROI abs] x={roi_abs[0]} y={roi_abs[1]} w={roi_abs[2]} h={roi_abs[3]}")

    # Run in selected mode
    loop_mode(roi_abs) if RUN_LOOP else one_shot(roi_abs)


if __name__ == "__main__":
    main()
