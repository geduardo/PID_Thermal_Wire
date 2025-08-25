"""
Control de Temperatura: Preheat + PID + Stop (con gráfica en vivo + CSV)

Fases:
1) Preheat: PWM = 100% hasta T >= SETPOINT
2) PID: durante PID_DURATION segundos
3) Stop: PWM = 0, cerrar gráfica y terminar
4) Cierre por ventana del plot → también apaga y guarda CSV

Autor: Nicolas Muñoz
"""

import time
import os
import csv
import serial
import matplotlib.pyplot as plt
from collections import deque
from read_temp2 import select_roi, read_temperature_from_roi

# ---------------- CONFIG ----------------
PORT          = "COM3"
SETPOINT      = 280.0      # °C
PID_DURATION  = 30.0       # s
LOOP_DT       = 0.03       # s (periodo de control/plot)

USE_MOUSE_ROI = False
ROI_X, ROI_Y, ROI_W, ROI_H = 332, -979, 1479, 698

# ---------------- PID ----------------
class PID:
    """PID básico con salida limitada 0..255."""
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(0, 255)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.lo, self.hi = output_limits

    def update(self, pv):
        now = time.time()
        e = self.setpoint - pv
        dt = (now - self.last_time) if self.last_time else 0.0
        de_dt = (e - self.last_error) / dt if dt > 0 else 0.0
        self.integral += e * dt
        u = self.Kp * e + self.Ki * self.integral + self.Kd * de_dt
        u = max(self.lo, min(self.hi, u))
        self.last_error, self.last_time = e, now
        return int(u)

# ---- CSV logging with auto-numbering ----
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

def get_next_file_number(base_name):
    existing = [f for f in os.listdir(logs_dir)
                if f.startswith(base_name) and f.endswith(".csv")]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split("_")[-1].split(".")[0]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1

base_name   = f"T{int(SETPOINT)}_{int(PID_DURATION)}"  # p.ej. T170_30
file_number = get_next_file_number(base_name)
log_path    = os.path.join(logs_dir, f"{base_name}_{file_number}.csv")

csv_file   = open(log_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time_s", "temp_c", "setpoint_c", "pwm", "error_c"])
tlog0 = time.time()
_last_flush = time.time()

# ---------------- MAIN ----------------
def main():
    # ROI
    roi = select_roi() if USE_MOUSE_ROI else (ROI_X, ROI_Y, ROI_W, ROI_H)

    # Serial
    ser = serial.Serial(PORT, 9600, timeout=1)
    time.sleep(1)

    # Plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Preheat (100%) → PID → Stop")
    line_temp, = ax.plot([], [], label="Temperature")
    line_setp, = ax.plot([], [], "r--", label="Setpoint")
    ax.legend()

    t0 = time.time()
    t_buf = deque(maxlen=4000)
    T_buf = deque(maxlen=4000)

    def update_plot(t_now, T_now, setpoint):
        t_buf.append(t_now)
        T_buf.append(T_now)
        line_temp.set_data(t_buf, T_buf)
        line_setp.set_data(t_buf, [setpoint] * len(t_buf))
        ax.set_xlim(0, max(10, t_now + 1))
        ymin = min(T_buf) - 5 if T_buf else setpoint - 10
        ymax = max(T_buf) + 5 if T_buf else setpoint + 10
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    def log_row(temp, setpoint, pwm):
        global _last_flush
        t_el = time.time() - tlog0
        err = (setpoint - temp) if (temp is not None) else ""
        csv_writer.writerow([f"{t_el:.4f}",
                             f"{temp:.4f}" if temp is not None else "",
                             f"{setpoint:.2f}",
                             int(pwm),
                             f"{err:.4f}" if temp is not None else ""])
        # flush cada ~1s
        if time.time() - _last_flush >= 1.0:
            csv_file.flush()
            _last_flush = time.time()

    try:
        # ---------- PREHEAT ----------
        print("[PREHEAT] PWM=100% hasta T >= SETPOINT...")
        while True:
            if not plt.fignum_exists(fig.number):
                print("\n[STOP] Ventana cerrada durante preheat.")
                return
            T = read_temperature_from_roi(*roi)
            t_now = time.time() - t0

            pwm = 255  # 100%
            ser.write(bytes([pwm]))

            if T is not None:
                update_plot(t_now, T, SETPOINT)
                log_row(T, SETPOINT, pwm)
                print(f"T={T:6.1f} °C (preheat)", end="\r")
                if T >= SETPOINT:
                    break

            time.sleep(LOOP_DT)

        # ---------- PID ----------
        print("\n[PID] Ejecutando durante %.1f s..." % PID_DURATION)
        pid = PID(Kp=0.75, Ki=0.78, Kd=0.0, setpoint=SETPOINT)
        start_pid = time.time()

        while (time.time() - start_pid) < PID_DURATION:
            if not plt.fignum_exists(fig.number):
                print("\n[STOP] Ventana cerrada durante PID.")
                return
            T = read_temperature_from_roi(*roi)
            t_now = time.time() - t0

            if T is not None:
                pwm = pid.update(T)
                ser.write(bytes([pwm]))
                update_plot(t_now, T, SETPOINT)
                log_row(T, SETPOINT, pwm)
                print(f"T={T:6.1f} °C | PWM={pwm:3d}", end="\r")

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("\n[STOP] Interrumpido por usuario.")

    finally:
        # ---------- STOP ----------
        try:
            ser.write(bytes([0]))
            ser.close()
        except Exception:
            pass
        try:
            csv_file.flush()
            csv_file.close()
            print(f"\n[CSV] Guardado: {log_path}")
        except Exception:
            pass
        plt.close('all')
        print("\n[FIN] PWM=0%, gráfica cerrada, programa terminado.")

if __name__ == "__main__":
    main()
