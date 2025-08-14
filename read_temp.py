# File: read_temperature.py 
# pip install pyautogui pillow easyocr opencv-python numpy

import cv2
import numpy as np
import pyautogui as pag
import easyocr
import re
import time

# Parámetros OCR
DIGIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
reader = easyocr.Reader(['en'], gpu=True)

def select_roi():
    """
    Permite al usuario dibujar con el ratón la ROI sobre la pantalla.
    Devuelve tupla (x, y, w, h).
    """
    screen = pag.screenshot()
    img_orig = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    img_display = img_orig.copy()
    roi = None
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, roi, img_display
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_display = img_orig.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Seleccione ROI", img_display)

    cv2.namedWindow("Seleccione ROI")
    cv2.setMouseCallback("Seleccione ROI", mouse_callback)
    print("Dibuje un recuadro arrastrando el ratón y luego presione cualquier tecla...")
    while True:
        cv2.imshow("Seleccione ROI", img_display)
        key = cv2.waitKey(1)
        if roi and key != -1:
            break

    cv2.destroyAllWindows()
    x, y, w, h = roi
    print(f"ROI seleccionada: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h

def read_temperature_from_roi(x, y, w, h) -> float | None:
    """
    Hace captura de pantalla de la ROI y aplica OCR para extraer un número.
    Devuelve float con la temperatura, o None si no se detecta.
    """
    snap = pag.screenshot(region=(x, y, w, h))
    arr = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
    results = reader.readtext(arr, allowlist='0123456789.,')
    for _, text, _ in results:
        m = DIGIT_RE.search(text.replace(',', '.'))                         
        if m:
            num_str = m.group(1).replace(',', '.')
            if '.' not in num_str and num_str.isdigit() and len(num_str) >= 2:
                num_str = num_str[:-1] + '.' + num_str[-1]
            return float(num_str)

    return None


if __name__ == "__main__":
    # 1. Selecciona ROI
    x, y, w, h = select_roi()
    print("Iniciando lectura de temperatura. Pulsa Ctrl+C para salir.")


    try:
        while True:
            temp = read_temperature_from_roi(x, y, w, h)
            if temp is not None:
                print(f"\rTemperatura: {temp:.2f} ºC", end="", flush=True)
            else:
                print("\rTemperatura: --.- ºC", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nLectura interrumpida por el usuario.")
