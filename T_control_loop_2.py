import time, serial, matplotlib.pyplot as plt, numpy as np, csv, os
from collections import deque
from matplotlib.widgets import TextBox
from read_temp import select_roi, read_temperature_from_roi
from datetime import datetime

INTERVAL  = 0.033   # s
PORT      = "COM3"
SETPOINT0 = 200.0   # ºC 

#x=1443, y=138, w=363, h=286
# ---------------- FILTER ------------------
class OutlierFilter:
    def __init__(self, max_delta=80):
        self.prev = None; self.max_delta = max_delta
    def update(self, v):
        if self.prev is not None and abs(v - self.prev) > self.max_delta:
            return self.prev
        self.prev = v; return v

# ---------------- PID ------------------
class PID:
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

    # --- ROI ---
    #x, y, w, h = select_roi()
    x,y,w,h=1841,139,46,30
    # --- PID ---
    kp = 0.75 #2.4277   0.75  1.5    
    ki = 0.78    # 0.8972  0.78  1.8
    kd = 0  #0.1999  0  0.13

    pid = PID(kp,ki,kd, SETPOINT0)

    # --- CSV logging setup ---
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_T_control_loop_2.csv"
    )
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s", "temp_c", "setpoint_c", "pwm", "error_c"])
    tlog0 = time.time()
    _last_flush = time.time()
    # --- Arduino ---
    ser = serial.Serial(PORT, 9600, timeout=1); time.sleep(1)
    def fmt_T(t):
        return f"{t:6.2f} °C" if t is not None else "-- °C"

    # --------- PREHEATING HASTA T >= 150°C ----------
    PWM_PREHEAT     = int(1 * 255)   # 20% duty
    PREHEAT_TARGET  = 170.0             # ºC (umbral de lectura de la cámara)

    print(f"\n[PREHEAT] {PWM_PREHEAT/2.55:.0f}% hasta T ≥ {PREHEAT_TARGET:.0f} °C...")
    ser.write(bytes([PWM_PREHEAT]))
    t_start = time.time()
    hits = 0

    while True:
        temp = read_temperature_from_roi(x, y, w, h)
        # Mantén el PWM de preheat mandando el valor periódicamente
        ser.write(bytes([PWM_PREHEAT]))

        if temp is not None:
            # log preheat sample
            t_el = time.time() - tlog0
            err_pre = pid.setpoint - temp
            csv_writer.writerow([
                f"{t_el:.4f}", f"{temp:.4f}", f"{pid.setpoint:.2f}", PWM_PREHEAT, f"{err_pre:.4f}"
            ])
            if time.time() - _last_flush >= 1.0:
                csv_file.flush(); _last_flush = time.time()
            if temp >= PREHEAT_TARGET:
                break
        time.sleep(0.0025)

    print("\n[PREHEAT] FINISHED → PID")

    outlier = OutlierFilter(max_delta=80)
    plt.ion(); fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)

    temps, t_axis = deque(maxlen=200), deque(maxlen=200); t0 = time.time()
    lt, = ax.plot([], [], 'b', label='Temperature')
    ls, = ax.plot([], [], 'r--', label='SETPOINT')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("°C")
    ax.set_title("PID control"); ax.legend()

    # -------- Setpoint TextBox  --------
    axbox = plt.axes([0.3, 0.05, 0.15, 0.07])
    textbox = TextBox(axbox, "Setpoint °C (100-250!)", initial=str(SETPOINT0))

    def submit(text):
        try:
            new_sp = float(text)
            pid.setpoint = new_sp
            print(f"\n Setpoint changed to {new_sp:.1f} °C")
        except ValueError:
            print("\n[WARN] No valid number.")

        if t_axis:
            ls.set_data(t_axis, [pid.setpoint]*len(t_axis))
            ymin = min(min(temps), pid.setpoint)-2
            ymax = max(max(temps), pid.setpoint)+2
            ax.set_ylim(ymin, ymax)
            fig.canvas.draw_idle()
    textbox.on_submit(submit)

    try:
        while plt.fignum_exists(fig.number):
            t = time.time()-t0
            temp = read_temperature_from_roi(x, y, w, h)
            if temp is not None:
                tf = outlier.update(temp)
                pwm, err = pid.update(tf)
                ser.write(bytes([pwm]))
                print(f"\rT={fmt_T(temp)} | e={err:6.2f} °C | PWM={pwm:3d} ({pwm/2.55:3.0f}%)", end="")
                temps.append(tf); t_axis.append(t)
                ls.set_data(t_axis, [pid.setpoint]*len(t_axis))
                # log control loop sample (filtered temp)
                t_el = time.time() - tlog0
                csv_writer.writerow([
                    f"{t_el:.4f}", f"{tf:.4f}", f"{pid.setpoint:.2f}", pwm, f"{err:.4f}"
                ])
                if time.time() - _last_flush >= 1.0:
                    csv_file.flush(); _last_flush = time.time()
            else:
                print("\rT=  -- °C", end="")


            if len(t_axis)>1:
                lt.set_data(t_axis, temps)
                ax.set_xlim(max(0, t_axis[0]), t_axis[-1]+1)
                ymin = min(min(temps), pid.setpoint)-2
                ymax = max(max(temps), pid.setpoint)+2
                ax.set_ylim(ymin, ymax)
                fig.canvas.draw(); fig.canvas.flush_events()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nInterrumpted.")
    finally:
        ser.write(bytes([0])); print("\nPWM = 0%"); ser.close()
        try:
            csv_file.flush(); csv_file.close()
            print(f"Saved CSV: {log_path}")
        except Exception:
            pass
        plt.ioff(); plt.show()

if __name__ == "__main__":
    main()