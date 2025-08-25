"""
PID Temperature Control (threaded reader + throttled plot)

- Temperature is read in a background thread at a fixed cadence (READ_PERIOD).
- The main loop only consumes the latest value and throttles:
    * plot updates (PLOT_PERIOD)
    * console prints (PRINT_PERIOD)
- Preheat and PID behavior remain the same.

Author: NICOLAS MUNOZ
"""

import time, serial, matplotlib.pyplot as plt, numpy as np, csv, os
from collections import deque
from matplotlib.widgets import TextBox
from threading import Thread, Event
from time import perf_counter
from read_temp2 import select_roi, read_temperature_from_roi
from datetime import datetime

# ---------------- CONFIG ----------------
INTERVAL      = 0.1   # main loop pacing (seconds) - small for responsive UI
PORT          = "COM3"
SETPOINT0     = 200.0   # °C

USE_MOUSE_ROI = False
ROI_X, ROI_Y, ROI_W, ROI_H = -1591, 115, 1449, 699

# Background worker pacing (independent from UI loop)
READ_PERIOD   = 0.02    # ~50 Hz temperature sampling in the worker thread
PLOT_PERIOD   = 0.10    # ~10 Hz plot updates
PRINT_PERIOD  = 0.10    # ~10 Hz console prints


# ---------------- FILTER ----------------
class OutlierFilter:
    """Simple jump filter to ignore unrealistic spikes."""
    def __init__(self, max_delta=80):
        self.prev = None; self.max_delta = max_delta
    def update(self, v):
        if self.prev is not None and abs(v - self.prev) > self.max_delta:
            return self.prev
        self.prev = v; return v


# ---------------- PID ----------------
class PID:
    """Basic PID controller with output clamping."""
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(0,255)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._last_error = self._integral = 0.0
        self._last_time  = None
        self.output_limits = output_limits
    def update(self, pv):
        now = time.time()
        e = self.setpoint - pv
        dt = now - self._last_time if self._last_time else 0.0
        d = (e - self._last_error)/dt if (self._last_time and dt>0) else 0.0
        self._integral += e*dt
        out = self.Kp*e + self.Ki*self._integral + self.Kd*d
        lo, hi = self.output_limits; out = max(lo, min(hi, out))
        self._last_error, self._last_time = e, now
        return int(out), e


def main():
    # ---- ROI ----
    if USE_MOUSE_ROI:
        x, y, w, h = select_roi()                 # pick live (uses monitor from grayscale config)
    else:
        x, y, w, h = ROI_X, ROI_Y, ROI_W, ROI_H   # fixed ROI

    # ---- PID ----
    kp, ki, kd = 0.75, 0.78, 0.0
    pid = PID(kp, ki, kd, SETPOINT0)

    # ---- CSV logging ----
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_T_control_loop_2.csv")
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s", "temp_c", "setpoint_c", "pwm", "error_c"])
    tlog0 = time.time()
    _last_flush = time.time()

    # ---- Arduino ----
    ser = serial.Serial(PORT, 9600, timeout=1)
    time.sleep(1)

    def fmt_T(t):
        return f"{t:6.2f} °C" if t is not None else "-- °C"

    # --------------- Background temperature reader ---------------
    latest = {"temp": None, "ts": 0.0}
    stop_evt = Event()

    def reader_worker():
        """Reads temperature at a fixed cadence and stores the latest value."""
        while not stop_evt.is_set():
            t0 = perf_counter()
            try:
                T = read_temperature_from_roi(x, y, w, h)
                if T is not None:
                    latest["temp"] = T
                    latest["ts"]   = time.time()
            except Exception:
                # keep running; ignore sporadic read errors
                pass
            # keep cadence
            dt = perf_counter() - t0
            remain = max(0.0, READ_PERIOD - dt)
            stop_evt.wait(remain)

    thr = Thread(target=reader_worker, name="TempReader", daemon=True)
    thr.start()
    # -------------------------------------------------------------

    # -------- PREHEAT: drive constant PWM until T >= target --------
    PWM_PREHEAT    = int(1 * 255)   # 100% duty
    PREHEAT_TARGET = 150.0          # °C
    print(f"\n[PREHEAT] {PWM_PREHEAT/2.55:.0f}% until T ≥ {PREHEAT_TARGET:.0f} °C...")

    ser.write(bytes([PWM_PREHEAT]))
    try:
        while True:
            temp = latest["temp"]                   # non-blocking read
            ser.write(bytes([PWM_PREHEAT]))         # keep preheat active

            if temp is not None:
                # log preheat sample
                t_el = time.time() - tlog0
                err_pre = pid.setpoint - temp
                csv_writer.writerow([f"{t_el:.4f}", f"{temp:.4f}", f"{pid.setpoint:.2f}", PWM_PREHEAT, f"{err_pre:.4f}"])
                if time.time() - _last_flush >= 1.0:
                    csv_file.flush(); _last_flush = time.time()
                if temp >= PREHEAT_TARGET:
                    break
            time.sleep(0.0025)

        print("\n[PREHEAT] FINISHED → PID")

        # ---- Plot setup ----
        outlier = OutlierFilter(max_delta=80)
        plt.ion(); fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.18)

        temps, t_axis = deque(maxlen=400), deque(maxlen=400)  # slightly larger buffer
        t0 = time.time()
        lt, = ax.plot([], [], 'b', label='Temperature')
        ls, = ax.plot([], [], 'r--', label='SETPOINT')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("°C")
        ax.set_title("PID control (threaded read, throttled plot)"); ax.legend()

        # ---- Setpoint TextBox ----
        axbox = plt.axes([0.3, 0.05, 0.15, 0.07])
        textbox = TextBox(axbox, "Setpoint °C (100-300!)", initial=str(SETPOINT0))

        def submit(text):
            try:
                pid.setpoint = float(text)
                print(f"\n Setpoint changed to {pid.setpoint:.1f} °C")
            except ValueError:
                print("\n[WARN] Invalid number.")
            if t_axis:
                ls.set_data(t_axis, [pid.setpoint]*len(t_axis))
                ymin = min(min(temps), pid.setpoint)-2
                ymax = max(max(temps), pid.setpoint)+2
                ax.set_ylim(ymin, ymax)
                fig.canvas.draw_idle()
        textbox.on_submit(submit)

        # ---- throttling timers ----
        next_plot  = time.time()
        next_print = time.time()

        # ---- PID loop ----
        while plt.fignum_exists(fig.number):
            now = time.time()
            t   = now - t0

            # read latest temp (non-blocking)
            temp = latest["temp"]

            if temp is not None:
                tf = outlier.update(temp)
                pwm, err = pid.update(tf)
                ser.write(bytes([pwm]))

                # console print (throttled)
                if now >= next_print:
                    print(f"\rT={fmt_T(temp)} | e={err:6.2f} °C | PWM={pwm:3d} ({pwm/2.55:3.0f}%)", end="")
                    next_print = now + PRINT_PERIOD

                # store for plotting
                temps.append(tf); t_axis.append(t)

                # log control loop sample (filtered temp)
                t_el = time.time() - tlog0
                csv_writer.writerow([f"{t_el:.4f}", f"{tf:.4f}", f"{pid.setpoint:.2f}", pwm, f"{err:.4f}"])
                if time.time() - _last_flush >= 1.0:
                    csv_file.flush(); _last_flush = time.time()
            else:
                if now >= next_print:
                    print("\rT=  -- °C", end="")
                    next_print = now + PRINT_PERIOD

            # plot update (throttled)
            if now >= next_plot and len(t_axis) > 1:
                lt.set_data(t_axis, temps)
                ax.set_xlim(max(0, t_axis[0]), t_axis[-1]+1)
                ymin = min(min(temps), pid.setpoint)-2 if temps else 0
                ymax = max(max(temps), pid.setpoint)+2 if temps else 10
                ax.set_ylim(ymin, ymax)
                ls.set_data(t_axis, [pid.setpoint]*len(t_axis))
                fig.canvas.draw(); fig.canvas.flush_events()
                next_plot = now + PLOT_PERIOD

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # stop worker + safety off
        try:
            ser.write(bytes([0])); print("\nPWM = 0%")
        except Exception:
            pass
        stop_evt.set(); thr.join(timeout=1.0)
        ser.close()

        try:
            csv_file.flush(); csv_file.close()
            print(f"Saved CSV: {log_path}")
        except Exception:
            pass

        plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
