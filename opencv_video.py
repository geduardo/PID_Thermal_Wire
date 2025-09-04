import time
import csv
from collections import deque
import numpy as np
import cv2
import mss
import os
from datetime import datetime

# ================== CONFIG ==================
OUTPUT_CSV = "deformation_log1.csv"
SMOOTH_WINDOW = 5          # tamaño media móvil para suavizar (frames)
PIXELS_PER_UNIT = None     # ej.: px/mm si conoces la escala; si no, deja None
UNIT_NAME = "mm"           # etiqueta de unidad si defines PIXELS_PER_UNIT
INITIAL_MONITOR = 1        # 1 = monitor principal (mss usa 1-index)
# ============================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_contiguous_bgr(img_bgra):
    # img_bgra viene de mss (BGRA)
    img = img_bgra[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)

def get_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("Tu OpenCV no tiene CSRT. Instala opencv-contrib-python.")

def grab_screen(sct, monitor):
    img = np.array(sct.grab(monitor))
    return ensure_contiguous_bgr(img)

def select_roi_window(title, frame):
    roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    if roi == (0,0,0,0):
        return None
    return tuple(map(int, roi))  # x,y,w,h

def center_of(bbox):
    x,y,w,h = bbox
    return (int(x + w/2), int(y + h/2))

def draw_hud(frame, p1, p2, d_px, d0_px, paused, ok1, ok2):
    color1 = (0,255,0) if ok1 else (0,0,255)
    color2 = (0,255,0) if ok2 else (0,0,255)
    cv2.circle(frame, p1, 5, color1, -1)
    cv2.circle(frame, p2, 5, color2, -1)
    cv2.line(frame, p1, p2, (255,255,255), 2)

    txt = [f"d = {d_px:.2f} px"]
    if PIXELS_PER_UNIT:
        d_unit = d_px / PIXELS_PER_UNIT
        txt.append(f"d = {d_unit:.3f} {UNIT_NAME}")
    if d0_px is not None and d0_px > 1e-6:
        eps = (d_px - d0_px) / d0_px
        txt.append(f"ε = {eps:+.5f}")
    if paused:
        txt.append("[PAUSA]")

    y0 = 24
    for t in txt:
        cv2.putText(frame, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        y0 += 26

    cv2.putText(frame, "a: area  r: re-seleccionar puntos  s: guardar CSV  SPACE: pausa  q: salir",
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, "a: area  r: re-seleccionar puntos  s: guardar CSV  SPACE: pausa  q: salir",
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def _compose_filename(base, tag=None):
    name, ext = os.path.splitext(base)
    if tag:
        return f"{name}_{tag}{ext}"
    return base

def save_csv(log, path, autosave=False):
    if not log:
        return None
    out_path = path
    if autosave:
        out_path = _compose_filename(path, f"autosave_{timestamp()}")
    # si no es autosave y ya existe, guarda con sufijo timestamp para no pisar
    elif os.path.exists(path):
        out_path = _compose_filename(path, timestamp())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["t_sec", "d_px", "epsilon"]
        if PIXELS_PER_UNIT:
            header.insert(2, f"d_{UNIT_NAME}")
        w.writerow(header)
        for row in log:
            # row: (t, d_px, d_unit or None, eps)
            if PIXELS_PER_UNIT:
                w.writerow([f"{row[0]:.3f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.8f}"])
            else:
                w.writerow([f"{row[0]:.3f}", f"{row[1]:.6f}", f"{row[3]:.8f}"])
    return out_path

def main():
    sct = mss.mss()
    monitor = sct.monitors[INITIAL_MONITOR]
    crop = dict(monitor)

    frame = grab_screen(sct, monitor)
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    trackers = None
    d0_px = None
    paused = False
    dist_smooth = deque(maxlen=SMOOTH_WINDOW)
    t0 = time.time()
    log = []  # (t, d_px, d_unit?, eps)

    def select_area():
        nonlocal crop
        snap = grab_screen(sct, monitor)
        roi = select_roi_window("Selecciona AREA de pantalla", snap)
        if roi:
            x,y,w,h = roi
            crop = {'left': monitor['left'] + x, 'top': monitor['top'] + y, 'width': w, 'height': h}

    def select_points_and_init():
        nonlocal trackers, d0_px, dist_smooth
        dist_smooth.clear()
        d0_px = None
        snap = grab_screen(sct, crop)
        if snap.size == 0:
            return
        r1 = select_roi_window("ROI punto 1", snap)
        if r1 is None:
            return
        r2 = select_roi_window("ROI punto 2", snap)
        if r2 is None:
            return
        t1 = get_tracker()
        t2 = get_tracker()
        ok1 = t1.init(snap, r1)
        ok2 = t2.init(snap, r2)
        if not (ok1 and ok2):
            print("No se pudo inicializar el tracker. Reintenta.")
            trackers = None
            return
        trackers = (t1, t2)

    # Primer paso: pedir área y puntos
    select_area()
    select_points_and_init()

    saved_on_loss = False
    saved_on_exit = False

    try:
        while True:
            # Si la ventana se ha cerrado por el usuario, salimos
            if cv2.getWindowProperty("Live", cv2.WND_PROP_VISIBLE) < 1:
                break

            if not paused:
                frame = grab_screen(sct, crop)

                if trackers is not None:
                    t1, t2 = trackers
                    ok1, box1 = t1.update(frame)
                    ok2, box2 = t2.update(frame)

                    if ok1 and ok2:
                        p1 = center_of(tuple(map(int, box1)))
                        p2 = center_of(tuple(map(int, box2)))
                        d_px = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                        dist_smooth.append(d_px)
                        d_px_sm = np.mean(dist_smooth) if len(dist_smooth) else d_px

                        if d0_px is None and d_px_sm > 1e-6:
                            d0_px = d_px_sm

                        eps = (d_px_sm - d0_px)/d0_px if (d0_px and d0_px > 1e-6) else 0.0
                        draw_hud(frame, p1, p2, d_px_sm, d0_px, paused, ok1, ok2)

                        t = time.time() - t0
                        if PIXELS_PER_UNIT:
                            d_unit = d_px_sm / PIXELS_PER_UNIT
                            log.append((t, d_px_sm, d_unit, eps))
                        else:
                            log.append((t, d_px_sm, None, eps))

            cv2.imshow("Live", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Salida explícita -> guardado final en finally
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                paused = True
                select_area()
                select_points_and_init()
                paused = False
                saved_on_loss = False
            elif key == ord('r'):
                paused = True
                select_points_and_init()
                paused = False
                saved_on_loss = False
            elif key == ord('s'):
                try:
                    path = save_csv(log, OUTPUT_CSV, autosave=False)
                    if path:
                        print(f"[SAVE] Guardado: {path} ({len(log)} muestras)")
                except Exception as e:
                    print("Error guardando CSV:", e)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Interrumpido por el usuario (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # Guardado final garantizado
        if not saved_on_exit:
            path = save_csv(log, OUTPUT_CSV, autosave=False)
            if path:
                print(f"[FINAL] Guardado final: {path} ({len(log)} muestras)")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
