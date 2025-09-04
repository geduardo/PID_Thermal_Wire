"""
Manual PWM Control with Background Temperature Reading

Reads temperature from a selected ROI using OCR in a background thread,
while allowing manual PWM control via Arduino. Plots temperature over time
and logs data to CSV. Threshold crossing is detected and annotated.
"""

import time
import serial
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from collections import deque
from matplotlib.widgets import TextBox
from threading import Thread, Event
from time import perf_counter
from read_temp2 import select_roi, read_temperature_from_roi
from datetime import datetime


# ---------------- CONFIGURATION ----------------
INTERVAL     = 0.01    # UI/plot loop pacing (seconds)
PORT         = "COM3"  # Arduino serial port
PWM0         = 0     # Initial PWM (%)
THRESHOLD_C  = 300.0   # Temperature threshold to measure time until threshold(°C)

# Background worker pacing
OCR_PERIOD   = 0.08    # ~12.5 Hz OCR updates
PLOT_PERIOD  = 0.10    # ~10 Hz plot updates
PRINT_PERIOD = 0.10    # ~10 Hz console prints


# ---------------- FILTER ----------------
class OutlierFilter:
    """Rejects sudden unrealistic jumps in values."""
    def __init__(self, max_delta=60):
        self.prev = None
        self.max_delta = max_delta

    def update(self, v):
        if self.prev is not None and abs(v - self.prev) > self.max_delta:
            return self.prev
        self.prev = v
        return v


def clamp_uint8(v):
    """Clamp value to [0, 255] and return as int."""
    return max(0, min(255, int(v)))


# ---------------- MAIN ----------------
def main():
    # --- ROI selection ---
    # Uncomment to select ROI interactively:
    # x, y, w, h = select_roi()
    x, y, w, h = 1841, 139, 46, 30  # Fixed ROI

    # --- Arduino serial connection ---
    ser = serial.Serial(PORT, 9600, timeout=1)
    time.sleep(2)  # Allow Arduino reset

    # --- CSV logging setup ---
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_PWM_CTE.csv"
    )
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s", "temp_c", "setpoint_c", "pwm", "error_c"])
    tlog0 = time.time()
    _last_flush = time.time()

    # --- Plot setup ---
    outlier = OutlierFilter(max_delta=70)
    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)

    temps, t_axis = deque(maxlen=200), deque(maxlen=200)
    t0 = time.time()
    lt, = ax.plot([], [], 'b', label='Temperature')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("°C")
    ax.set_title("Manual PWM Control")
    ax.legend()

    # --- PWM control via TextBox ---
    pwm = clamp_uint8(PWM0)
    axbox = plt.axes([0.30, 0.05, 0.15, 0.07])
    textbox = TextBox(axbox, "PWM (0-100%)", initial=str(pwm))

    def submit(text):
        nonlocal pwm
        try:
            pwm = clamp_uint8(float(text))
            print(f"\nPWM changed to {pwm} ({pwm / 2.55:3.0f}%)")
        except ValueError:
            print("\n[WARN] Invalid number.")

    textbox.on_submit(submit)

    # --- Threshold crossing detection ---
    crossing_time_rel = None
    crossing_time_abs = None

    # --- Background OCR worker ---
    latest = {"temp": None, "ts": 0.0}
    stop_evt = Event()

    def ocr_worker():
        """Performs OCR at fixed cadence and stores latest value."""
        while not stop_evt.is_set():
            tstart = perf_counter()
            tmp = read_temperature_from_roi(x, y, w, h) 
            if tmp is not None:
                latest["temp"] = tmp
                latest["ts"] = time.time()
            dt = perf_counter() - tstart
            remain = max(0.0, OCR_PERIOD - dt)
            stop_evt.wait(remain)

    thr = Thread(target=ocr_worker, name="OCRWorker", daemon=True)
    thr.start()

    # --- Timers for throttling ---
    next_plot = time.time()
    next_print = time.time()

    try:
        while plt.fignum_exists(fig.number):
            now = time.time()
            t = now - t0

            # Send PWM to Arduino (manual control)
            ser.write(bytes([pwm]))

            # Get latest OCR reading
            temp = latest["temp"]

            if temp is not None:
                tf = outlier.update(temp)

                # Detect threshold crossing
                if crossing_time_rel is None and tf >= THRESHOLD_C:
                    crossing_time_rel = t
                    crossing_time_abs = now
                    ax.axvline(crossing_time_rel, linestyle='--', linewidth=1.5, color='g')
                    ax.annotate(f"T>{THRESHOLD_C:.0f}°C @ {crossing_time_rel:.2f}s",
                                xy=(crossing_time_rel, tf),
                                xytext=(crossing_time_rel + 2, tf),
                                arrowprops=dict(arrowstyle='->', lw=1.0))

                # Log to CSV
                t_el = time.time() - tlog0
                csv_writer.writerow([
                    f"{t_el:.4f}", f"{tf:.4f}", "", pwm, ""
                ])
                if time.time() - _last_flush >= 1.0:
                    csv_file.flush()
                    _last_flush = time.time()

                # Throttled console output
                if now >= next_print:
                    print(f"\rT={tf:6.2f} °C | PWM={pwm:3d} ({pwm / 2.55:3.0f}%)", end="")
                    next_print = now + PRINT_PERIOD

                # Add data for plotting
                temps.append(tf)
                t_axis.append(t)
            else:
                if now >= next_print:
                    print(f"\rT=  --.- °C | PWM={pwm:3d} ({pwm / 2.55:3.0f}%)", end="")
                    next_print = now + PRINT_PERIOD

            # Throttled plot update
            if now >= next_plot and len(t_axis) > 1:
                lt.set_data(t_axis, temps)
                ax.set_xlim(max(0, t_axis[0]), t_axis[-1] + 1)
                ymin = min(temps) - 2 if temps else 0
                ymax = max(temps) + 2 if temps else 10
                ax.set_ylim(ymin, ymax)
                fig.canvas.draw()
                fig.canvas.flush_events()
                next_plot = now + PLOT_PERIOD

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Stop worker and clean up
        try:
            ser.write(bytes([0]))
            print("\nPWM = 0%")
        except Exception:
            pass

        stop_evt.set()
        thr.join(timeout=1.0)
        ser.close()

        try:
            csv_file.flush()
            csv_file.close()
            print(f"Saved CSV: {log_path}")
        except Exception:
            pass

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
