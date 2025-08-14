# File: read_temperature.py
# pip install pyautogui pillow easyocr opencv-python numpy

import cv2
import numpy as np
import pyautogui as pag
import easyocr
import re
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Parámetros OCR
DIGIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
reader = easyocr.Reader(["en"], gpu=True)


class ROISelector:
    def __init__(self, screenshot_pil):
        self.screenshot = screenshot_pil
        self.roi = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Create main window
        self.root = tk.Tk()
        self.root.title("Select ROI - Draw rectangle around the area")
        self.root.attributes("-topmost", True)

        # Get screen dimensions and resize image if too large
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        img_width, img_height = self.screenshot.size
        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        if img_width > max_width or img_height > max_height:
            # Calculate scaling factor to fit screen
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            self.scale = min(scale_x, scale_y)

            new_width = int(img_width * self.scale)
            new_height = int(img_height * self.scale)
            display_image = self.screenshot.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        else:
            self.scale = 1.0
            display_image = self.screenshot

        self.photo = ImageTk.PhotoImage(display_image)

        # Create canvas
        self.canvas = tk.Canvas(
            self.root, width=display_image.width, height=display_image.height
        )
        self.canvas.pack()

        # Display image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

        # Add instructions
        instruction_frame = tk.Frame(self.root)
        instruction_frame.pack(pady=5)

        tk.Label(
            instruction_frame,
            text="Draw a rectangle around the region of interest, then click 'Confirm'",
            font=("Arial", 10),
        ).pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        tk.Button(
            button_frame,
            text="Confirm Selection",
            command=self.confirm_selection,
            bg="green",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_selection,
            bg="red",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)

        # Center window
        self.root.update_idletasks()
        x = (screen_width // 2) - (self.root.winfo_width() // 2)
        y = (screen_height // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

    def start_selection(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def update_selection(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="red", width=2
        )

    def end_selection(self, event):
        if self.start_x is not None and self.start_y is not None:
            # Calculate ROI coordinates (convert back to original image coordinates)
            x1 = int(min(self.start_x, event.x) / self.scale)
            y1 = int(min(self.start_y, event.y) / self.scale)
            x2 = int(max(self.start_x, event.x) / self.scale)
            y2 = int(max(self.start_y, event.y) / self.scale)

            w = x2 - x1
            h = y2 - y1

            if w > 10 and h > 10:  # Minimum size check
                self.roi = (x1, y1, w, h)

    def confirm_selection(self):
        if self.roi:
            self.root.quit()
        else:
            messagebox.showwarning("No Selection", "Please draw a rectangle first!")

    def cancel_selection(self):
        self.roi = None
        self.root.quit()

    def get_roi(self):
        self.root.mainloop()
        self.root.destroy()
        return self.roi


def select_roi():
    """
    Permite al usuario dibujar con el ratón la ROI sobre la pantalla usando tkinter.
    Devuelve tupla (x, y, w, h).
    """
    print("Taking screenshot for ROI selection...")
    screen = pag.screenshot()

    # Convert to PIL Image for tkinter
    screenshot_pil = screen  # pyautogui.screenshot() already returns PIL Image

    print("Opening ROI selection window...")
    print(
        "Instructions: Draw a rectangle around the region of interest, then click 'Confirm'"
    )

    roi_selector = ROISelector(screenshot_pil)
    roi = roi_selector.get_roi()

    if roi is None:
        print("ROI selection cancelled.")
        raise RuntimeError("ROI selection was cancelled by user")

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
    results = reader.readtext(arr, allowlist="0123456789.,")
    for _, text, _ in results:
        m = DIGIT_RE.search(text.replace(",", "."))
        if m:
            num_str = m.group(1).replace(",", ".")
            if "." not in num_str and num_str.isdigit() and len(num_str) >= 2:
                num_str = num_str[:-1] + "." + num_str[-1]
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
