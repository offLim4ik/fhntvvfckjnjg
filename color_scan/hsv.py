import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Нужен Pillow: pip install pillow")

APP_TITLE = "HSV Scanner (Min/Max)"
ACCENT = "#7C4DFF"
BG = "#0B0F17"
CARD = "#101827"
FG = "#E6EAF2"
MUTED = "#9AA4B2"
DANGER = "#FF4D6D"

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))

def build_mask(hsv, low, high):
    lh, ls, lv = int(low[0]), int(low[1]), int(low[2])
    uh, us, uv = int(high[0]), int(high[1]), int(high[2])

    ls = clamp(ls, 0, 255); lv = clamp(lv, 0, 255)
    us = clamp(us, 0, 255); uv = clamp(uv, 0, 255)

    if lh <= uh:
        return cv2.inRange(hsv, (lh, ls, lv), (uh, us, uv))
    m1 = cv2.inRange(hsv, (0,  ls, lv), (uh, us, uv))
    m2 = cv2.inRange(hsv, (lh, ls, lv), (179, us, uv))
    return cv2.bitwise_or(m1, m2)

def auto_fit_hsv_from_roi(frame_bgr, roi, q_lo=5, q_hi=95):
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    H = hsv[..., 0].astype(np.int32).reshape(-1)
    S = hsv[..., 1].astype(np.int32).reshape(-1)
    V = hsv[..., 2].astype(np.int32).reshape(-1)

    h_lo = int(np.percentile(H, q_lo))
    h_hi = int(np.percentile(H, q_hi))

    if (h_hi - h_lo) > 90:
        H2 = H.copy()
        H2[H2 < 90] += 180
        h2_lo = int(np.percentile(H2, q_lo))
        h2_hi = int(np.percentile(H2, q_hi))
        lh = h2_lo % 180
        uh = h2_hi % 180
    else:
        lh, uh = h_lo, h_hi

    ls = int(np.percentile(S, q_lo))
    us = int(np.percentile(S, q_hi))
    lv = int(np.percentile(V, q_lo))
    uv = int(np.percentile(V, q_hi))

    low = np.array([clamp(lh, 0, 179), clamp(ls, 0, 255), clamp(lv, 0, 255)], dtype=np.uint8)
    high = np.array([clamp(uh, 0, 179), clamp(us, 0, 255), clamp(uv, 0, 255)], dtype=np.uint8)
    return low, high

def make_snippet(low, high, name="Color"):
    low_list = [int(low[0]), int(low[1]), int(low[2])]
    high_list = [int(high[0]), int(high[1]), int(high[2])]
    # формируем “как ты показывал”
    snippet = (
        "COLORS_HSV = {\n"
        f'    "{name}": [\n'
        f"        (np.array({low_list}), np.array({high_list})),\n"
        "    ],\n"
        "}\n"
    )
    return snippet

class HSVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=BG)
        self.minsize(1240, 760)

        self.cap = None
        self.source_mode = "none"   # none | image | stream
        self.paused = False
        self.frozen_frame = None
        self.last_frame = None

        self.roi_canvas = None
        self.dragging = False
        self.drag_start = None

        self.frame_w = 1
        self.frame_h = 1
        self.draw_w = 1
        self.draw_h = 1
        self.draw_x0 = 0
        self.draw_y0 = 0

        self.sliders = {}

        self._build_style()
        self._build_ui()

        self.after(10, self._loop)

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=FG)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED)
        style.configure("Title.TLabel", background=CARD, foreground=FG, font=("Segoe UI Semibold", 13))
        style.configure("TButton", padding=10, background="#1B2434", foreground=FG, borderwidth=0)
        style.map("TButton", background=[("active", "#23324A")])
        style.configure("Accent.TButton", padding=10, background=ACCENT, foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#6B3DFF")])
        style.configure("Danger.TButton", padding=10, background=DANGER, foreground="white", borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#FF2F56")])
        style.configure("TScale", background=CARD)

    def _build_ui(self):
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True, padx=14, pady=14)

        left = ttk.Frame(root, style="Card.TFrame")
        left.pack(side="left", fill="y", padx=(0, 12))

        right = ttk.Frame(root, style="Card.TFrame")
        right.pack(side="right", fill="both", expand=True)

        header = ttk.Frame(left, style="Card.TFrame")
        header.pack(fill="x", padx=14, pady=(14, 10))
        ttk.Label(header, text="HSV Scanner", style="Title.TLabel").pack(anchor="w")
        self.status_lbl = ttk.Label(header, text="Нет источника. Открой файл или включи камеру.", style="Muted.TLabel")
        self.status_lbl.pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(left, style="Card.TFrame")
        btns.pack(fill="x", padx=14, pady=(0, 10))

        row1 = ttk.Frame(btns, style="Card.TFrame")
        row1.pack(fill="x")
        ttk.Button(row1, text="Открыть файл", style="Accent.TButton", command=self.open_file)\
            .pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row1, text="Камера", style="TButton", command=self.use_camera)\
            .pack(side="left", fill="x", expand=True)

        row2 = ttk.Frame(btns, style="Card.TFrame")
        row2.pack(fill="x", pady=(8, 0))
        self.pause_btn = ttk.Button(row2, text="Пауза", style="TButton", command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row2, text="Сброс ROI", style="TButton", command=self.reset_roi_and_sliders)\
            .pack(side="left", fill="x", expand=True)

        row3 = ttk.Frame(btns, style="Card.TFrame")
        row3.pack(fill="x", pady=(8, 0))
        ttk.Button(row3, text="Auto-fit по ROI", style="Accent.TButton", command=self.autofit_roi)\
            .pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row3, text="Копировать", style="TButton", command=self.copy_to_clipboard)\
            .pack(side="left", fill="x", expand=True)

        # Плашка вывода
        out_card = ttk.Frame(left, style="Card.TFrame")
        out_card.pack(fill="both", expand=False, padx=14, pady=(0, 14))

        ttk.Label(out_card, text="Вывод (готовый Python)", style="Title.TLabel").pack(anchor="w", pady=(0, 6))
        name_row = ttk.Frame(out_card, style="Card.TFrame")
        name_row.pack(fill="x", pady=(0, 6))
        ttk.Label(name_row, text="Имя:", style="Muted.TLabel").pack(side="left")
        self.color_name_var = tk.StringVar(value="Color")
        name_entry = ttk.Entry(name_row, textvariable=self.color_name_var)
        name_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.out_text = tk.Text(out_card, height=9, wrap="none", bg="#0A0E16", fg=FG,
                                insertbackground=FG, relief="flat", bd=0)
        self.out_text.pack(fill="x", expand=False)
        self.out_text.configure(state="disabled")

        # ----- sliders
        sliders = ttk.Frame(left, style="Card.TFrame")
        sliders.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        ttk.Label(sliders, text="Диапазон HSV (min/max)", style="Title.TLabel").pack(anchor="w", pady=(0, 8))

        self.vars = {
            "LH": tk.IntVar(value=0),
            "LS": tk.IntVar(value=0),
            "LV": tk.IntVar(value=0),
            "UH": tk.IntVar(value=179),
            "US": tk.IntVar(value=255),
            "UV": tk.IntVar(value=255),
            "Blur": tk.IntVar(value=0),
            "Erode": tk.IntVar(value=0),
            "Dilate": tk.IntVar(value=0),
            "Qlo": tk.IntVar(value=5),
            "Qhi": tk.IntVar(value=95),
        }

        self._add_scale(sliders, "LH", 0, 179, "LOW H")
        self._add_scale(sliders, "LS", 0, 255, "LOW S")
        self._add_scale(sliders, "LV", 0, 255, "LOW V")
        self._add_scale(sliders, "UH", 0, 179, "HIGH H")
        self._add_scale(sliders, "US", 0, 255, "HIGH S")
        self._add_scale(sliders, "UV", 0, 255, "HIGH V")

        ttk.Separator(sliders, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(sliders, text="Стабилизация маски", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        self._add_scale(sliders, "Blur", 0, 20, "Blur")
        self._add_scale(sliders, "Erode", 0, 10, "Erode")
        self._add_scale(sliders, "Dilate", 0, 10, "Dilate")

        ttk.Separator(sliders, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(sliders, text="Auto-fit (процентили)", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        self._add_scale(sliders, "Qlo", 0, 30, "Q low")
        self._add_scale(sliders, "Qhi", 51, 100, "Q high")

        # ----- preview
        topbar = ttk.Frame(right, style="Card.TFrame")
        topbar.pack(fill="x", padx=14, pady=(14, 8))
        ttk.Label(
            topbar,
            text="ROI: ЛКМ + тянуть. Auto-fit двигает ползунки. Копировать кладёт готовый код в буфер.",
            style="Muted.TLabel"
        ).pack(anchor="w")

        preview = ttk.Frame(right, style="Card.TFrame")
        preview.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.canvas = tk.Canvas(preview, bg="#05070C", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._roi_start)
        self.canvas.bind("<B1-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._roi_end)

        self._tk_img = None

        # при изменении имени/ползунков — обновляем вывод
        self.color_name_var.trace_add("write", lambda *_: self.update_output_box())
        for k in ["LH","LS","LV","UH","US","UV","Blur","Erode","Dilate","Qlo","Qhi"]:
            self.vars[k].trace_add("write", lambda *_: self.update_output_box())

        self.update_output_box()

    def _add_scale(self, parent, key, mn, mx, label):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)

        ttk.Label(row, text=label, style="Muted.TLabel").pack(side="left")
        ttk.Label(row, textvariable=self.vars[key], style="Muted.TLabel").pack(side="right")

        scale = ttk.Scale(
            parent, from_=mn, to=mx, orient="horizontal",
            command=lambda v, k=key: self.vars[k].set(int(float(v)))
        )
        scale.set(self.vars[key].get())
        scale.pack(fill="x", pady=(0, 6))
        self.sliders[key] = scale

    def set_slider(self, key, value):
        value = int(value)
        self.vars[key].set(value)
        if key in self.sliders:
            self.sliders[key].set(value)

    def set_sliders_bulk(self, mapping: dict):
        for k, v in mapping.items():
            self.set_slider(k, v)

    def update_output_box(self):
        low = np.array([self.vars["LH"].get(), self.vars["LS"].get(), self.vars["LV"].get()], dtype=np.uint8)
        high = np.array([self.vars["UH"].get(), self.vars["US"].get(), self.vars["UV"].get()], dtype=np.uint8)
        name = (self.color_name_var.get() or "Color").strip()
        snippet = make_snippet(low, high, name=name)

        self.out_text.configure(state="normal")
        self.out_text.delete("1.0", "end")
        self.out_text.insert("1.0", snippet)
        self.out_text.configure(state="disabled")

    def copy_to_clipboard(self):
        low = np.array([self.vars["LH"].get(), self.vars["LS"].get(), self.vars["LV"].get()], dtype=np.uint8)
        high = np.array([self.vars["UH"].get(), self.vars["US"].get(), self.vars["UV"].get()], dtype=np.uint8)
        name = (self.color_name_var.get() or "Color").strip()
        snippet = make_snippet(low, high, name=name)

        self.clipboard_clear()
        self.clipboard_append(snippet)
        self.update()  # чтобы буфер точно записался
        self._set_status("Скопировано в буфер обмена ✅")

    def _set_status(self, text):
        self.status_lbl.config(text=text)

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выбери изображение или видео",
            filetypes=[
                ("Media files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        self.reset_roi_and_sliders(reset_range=False)
        self.paused = False
        self.frozen_frame = None

        if ext in IMG_EXT:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Ошибка", "Не удалось прочитать изображение.")
                return
            self._release_cap()
            self.source_mode = "image"
            self.frozen_frame = img
            self.last_frame = img
            self.paused = True
            self.pause_btn.config(state="disabled", text="Пауза")
            self._set_status(f"Изображение: {os.path.basename(path)} (кадр фиксирован)")
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео.")
            return
        self._release_cap()
        self.cap = cap
        self.source_mode = "stream"
        self.paused = False
        self.frozen_frame = None
        self.pause_btn.config(state="normal", text="Пауза")
        self._set_status(f"Видео: {os.path.basename(path)}")

    def use_camera(self):
        self.reset_roi_and_sliders(reset_range=False)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру (index=0).")
            return
        self._release_cap()
        self.cap = cap
        self.source_mode = "stream"
        self.paused = False
        self.frozen_frame = None
        self.pause_btn.config(state="normal", text="Пауза")
        self._set_status("Камера: index=0")

    def toggle_pause(self):
        if self.source_mode != "stream":
            return
        self.paused = not self.paused
        if self.paused:
            if self.last_frame is not None:
                self.frozen_frame = self.last_frame.copy()
            self.pause_btn.config(text="Продолжить")
            self._set_status("Пауза (можно крутить диапазон)")
        else:
            self.frozen_frame = None
            self.pause_btn.config(text="Пауза")

    def reset_roi_and_sliders(self, reset_range=True):
        self.roi_canvas = None
        if reset_range:
            self.set_sliders_bulk({
                "LH": 0, "LS": 0, "LV": 0,
                "UH": 179, "US": 255, "UV": 255,
                "Blur": 0, "Erode": 0, "Dilate": 0,
                "Qlo": 5, "Qhi": 95,
            })
            self._set_status("ROI сброшен + диапазон сброшен в дефолт.")
        self.update_output_box()

    # ROI mouse
    def _roi_start(self, e):
        self.dragging = True
        self.drag_start = (e.x, e.y)
        self.roi_canvas = (e.x, e.y, e.x, e.y)

    def _roi_drag(self, e):
        if not self.dragging or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        self.roi_canvas = (x1, y1, e.x, e.y)

    def _roi_end(self, e):
        if not self.dragging or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        self.roi_canvas = (x1, y1, e.x, e.y)
        self.dragging = False
        self.drag_start = None

    def _roi_canvas_to_frame(self):
        if self.roi_canvas is None:
            return None
        x1, y1, x2, y2 = self.roi_canvas
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1 = clamp(x1, self.draw_x0, self.draw_x0 + self.draw_w)
        x2 = clamp(x2, self.draw_x0, self.draw_x0 + self.draw_w)
        y1 = clamp(y1, self.draw_y0, self.draw_y0 + self.draw_h)
        y2 = clamp(y2, self.draw_y0, self.draw_y0 + self.draw_h)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None

        nx1 = (x1 - self.draw_x0) / max(1, self.draw_w)
        nx2 = (x2 - self.draw_x0) / max(1, self.draw_w)
        ny1 = (y1 - self.draw_y0) / max(1, self.draw_h)
        ny2 = (y2 - self.draw_y0) / max(1, self.draw_h)

        fx1 = clamp(nx1 * self.frame_w, 0, self.frame_w - 1)
        fx2 = clamp(nx2 * self.frame_w, 0, self.frame_w - 1)
        fy1 = clamp(ny1 * self.frame_h, 0, self.frame_h - 1)
        fy2 = clamp(ny2 * self.frame_h, 0, self.frame_h - 1)

        fx1, fx2 = sorted([fx1, fx2])
        fy1, fy2 = sorted([fy1, fy2])

        if (fx2 - fx1) < 2 or (fy2 - fy1) < 2:
            return None
        return (fx1, fy1, fx2, fy2)

    def autofit_roi(self):
        frame = self.frozen_frame if (self.paused and self.frozen_frame is not None) else self.last_frame
        if frame is None:
            messagebox.showinfo("ROI", "Сначала открой файл или включи камеру.")
            return

        roi_frame = self._roi_canvas_to_frame()
        if roi_frame is None:
            messagebox.showinfo("ROI", "Выдели ROI мышкой по картинке (ЛКМ + тянуть).")
            return

        qlo = clamp(self.vars["Qlo"].get(), 0, 49)
        qhi = clamp(self.vars["Qhi"].get(), 51, 100)
        low, high = auto_fit_hsv_from_roi(frame, roi_frame, q_lo=qlo, q_hi=qhi)

        self.set_sliders_bulk({
            "LH": int(low[0]), "LS": int(low[1]), "LV": int(low[2]),
            "UH": int(high[0]), "US": int(high[1]), "UV": int(high[2]),
        })

        self._set_status(f"Auto-fit: LOW={tuple(low)} HIGH={tuple(high)}")
        self.update_output_box()

    # Render / loop
    def _get_frame(self):
        if self.source_mode == "image":
            return self.frozen_frame
        if self.source_mode == "stream":
            if self.paused and self.frozen_frame is not None:
                return self.frozen_frame
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.last_frame = frame
            return frame
        return None

    def _compose_preview(self, frame):
        low = np.array([self.vars["LH"].get(), self.vars["LS"].get(), self.vars["LV"].get()], dtype=np.uint8)
        high = np.array([self.vars["UH"].get(), self.vars["US"].get(), self.vars["UV"].get()], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = build_mask(hsv, low, high)

        blur = self.vars["Blur"].get()
        if blur > 0:
            k = 2 * blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        kernel = np.ones((3, 3), np.uint8)
        er = self.vars["Erode"].get()
        di = self.vars["Dilate"].get()
        if er > 0:
            mask = cv2.erode(mask, kernel, iterations=er)
        if di > 0:
            mask = cv2.dilate(mask, kernel, iterations=di)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask_small = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return result, mask_small

    def _draw_to_canvas(self, result_bgr, mask_bgr):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        self.frame_h, self.frame_w = result_bgr.shape[:2]

        scale = min(cw / self.frame_w, ch / self.frame_h)
        self.draw_w = int(self.frame_w * scale)
        self.draw_h = int(self.frame_h * scale)
        self.draw_x0 = (cw - self.draw_w) // 2
        self.draw_y0 = (ch - self.draw_h) // 2

        resized = cv2.resize(result_bgr, (self.draw_w, self.draw_h), interpolation=cv2.INTER_AREA)

        mw = max(160, self.draw_w // 4)
        mh = max(90, self.draw_h // 4)
        mask_r = cv2.resize(mask_bgr, (mw, mh), interpolation=cv2.INTER_NEAREST)

        ox = clamp(self.draw_w - mw - 12, 0, self.draw_w - mw)
        oy = clamp(self.draw_h - mh - 12, 0, self.draw_h - mh)

        overlay = resized.copy()
        cv2.rectangle(overlay, (ox - 6, oy - 6), (ox + mw + 6, oy + mh + 6), (10, 14, 24), -1)
        overlay[oy:oy + mh, ox:ox + mw] = mask_r

        status = "IMAGE" if self.source_mode == "image" else ("PAUSED" if self.paused else "LIVE")
        low = (self.vars["LH"].get(), self.vars["LS"].get(), self.vars["LV"].get())
        high = (self.vars["UH"].get(), self.vars["US"].get(), self.vars["UV"].get())
        text1 = f"{status}   LOW={low}   HIGH={high}"
        cv2.putText(overlay, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if self.roi_canvas is not None:
            x1, y1, x2, y2 = self.roi_canvas
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            dx1 = clamp(x1 - self.draw_x0, 0, self.draw_w - 1)
            dx2 = clamp(x2 - self.draw_x0, 0, self.draw_w - 1)
            dy1 = clamp(y1 - self.draw_y0, 0, self.draw_h - 1)
            dy2 = clamp(y2 - self.draw_y0, 0, self.draw_h - 1)
            if (dx2 - dx1) > 2 and (dy2 - dy1) > 2:
                cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (124, 77, 255), 2)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.draw_x0, self.draw_y0, anchor="nw", image=self._tk_img)

    def _loop(self):
        frame = self._get_frame()
        if frame is not None:
            result, mask = self._compose_preview(frame)
            self._draw_to_canvas(result, mask)
        self.after(16, self._loop)

if __name__ == "__main__":
    app = HSVApp()
    app.mainloop()
