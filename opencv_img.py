import csv
import os
from datetime import datetime
import time

import cv2
import numpy as np
import mss

# ================== CONFIG ==================
OUTPUT_CSV = "elongaciones.csv"
PIXELS_PER_UNIT = None   # ej. 3.20 px/mm. Si no conoces la escala, deja None
UNIT_NAME = "mm"
INITIAL_MONITOR = 1      # 1 = monitor principal (mss usa 1-index)
# ============================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_contiguous_bgr(img_bgra):
    img = img_bgra[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)

def grab_screen(sct, monitor):
    return ensure_contiguous_bgr(np.array(sct.grab(monitor)))

def save_elongations(dist_list, path):
    if not dist_list:
        return None
    L0 = dist_list[0]
    eps = [ (L - L0)/L0 for L in dist_list ]

    out_path = path if not os.path.exists(path) else f"{os.path.splitext(path)[0]}_{timestamp()}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["idx", "L_px", "epsilon"]
        if PIXELS_PER_UNIT:
            header.insert(2, f"L_{UNIT_NAME}")
        w.writerow(header)
        for i, (L, e) in enumerate(zip(dist_list, eps), start=1):
            if PIXELS_PER_UNIT:
                w.writerow([i, f"{L:.6f}", f"{L/PIXELS_PER_UNIT:.6f}", f"{e:.8f}"])
            else:
                w.writerow([i, f"{L:.6f}", f"{e:.8f}"])
    return out_path

def main():
    sct = mss.mss()
    crop = dict(sct.monitors[INITIAL_MONITOR])  # toda la pantalla del monitor elegido

    window = "Medición por fotogramas (clic: P1,P2 | Enter/Space: siguiente captura | u: rehacer | q: guardar y salir)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    distances = []  # lista de L en píxeles
    p1, p2 = None, None
    have_distance = False

    def on_mouse(event, x, y, flags, param):
        nonlocal p1, p2, have_distance
        if event == cv2.EVENT_LBUTTONDOWN:
            if p1 is None:
                p1 = (x, y)
                p2 = None
                have_distance = False
            elif p2 is None:
                p2 = (x, y)
                have_distance = True

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            # 1) Captura única (frame estático)
            frame = grab_screen(sct, crop)
            base = frame.copy()
            p1, p2, have_distance = None, None, False

            # Bucle de interacción sobre esta captura hasta que pulse Enter/Space o 'q'
            while True:
                canvas = base.copy()

                # Dibujar instrucciones
                hud1 = "Clic izquierdo: marcar P1 y P2 | u: rehacer | Enter/Space: nueva captura | q: terminar"
                cv2.putText(canvas, hud1, (10, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(canvas, hud1, (10, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255,255,255), 2, cv2.LINE_AA)

                # Dibujar puntos/medida actual
                if p1 is not None:
                    cv2.circle(canvas, p1, 6, (0, 255, 0), -1)
                if p2 is not None:
                    cv2.circle(canvas, p2, 6, (0, 255, 0), -1)
                if p1 is not None and p2 is not None:
                    cv2.line(canvas, p1, p2, (0, 255, 0), 2)
                    L = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                    txts = [f"L = {L:.3f} px"]
                    if PIXELS_PER_UNIT:
                        txts.append(f"L = {L/PIXELS_PER_UNIT:.3f} {UNIT_NAME}")
                    if distances:
                        L0 = distances[0]
                        eps = (L - L0)/L0 if L0 > 1e-9 else 0.0
                        txts.append(f"epsilon = {eps:+.6f}")
                    y0 = 30
                    for tline in txts:
                        cv2.putText(canvas, tline, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(canvas, tline, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255,255,255), 2, cv2.LINE_AA)
                        y0 += 32

                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF

                # ¿ventana cerrada?
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    key = ord('q')

                if key in (13, 32):  # Enter o Space -> nueva captura
                    # Si ya hay medida en esta captura, la registramos
                    if have_distance and p1 is not None and p2 is not None:
                        L = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                        distances.append(L)
                    break
                elif key == ord('u'):
                    # Rehacer selección en esta captura
                    p1, p2, have_distance = None, None, False
                elif key == ord('q'):
                    # Si hay medida tomada en esta captura y aún no añadida, la guardamos
                    if have_distance and p1 is not None and p2 is not None:
                        L = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                        distances.append(L)
                    # salir del bucle principal
                    raise SystemExit

        # (No se llega aquí por el raise, pero por claridad)
    except SystemExit:
        pass
    finally:
        cv2.destroyAllWindows()
        path = save_elongations(distances, OUTPUT_CSV)
        if path:
            print(f"[OK] Guardado CSV: {path}  (n={len(distances)} medidas)")
        else:
            print("[INFO] No se generó CSV (sin medidas).")

if __name__ == "__main__":
    main()
