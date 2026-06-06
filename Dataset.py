import cv2
import os
import sys
import time
import threading

#  Tkinter 
import tkinter as tk

# Pillow (required for Tkinter image display) 
try:
    from PIL import Image, ImageTk
except ImportError:
    print("\n[ERROR] Pillow is not installed.")
    print("        Fix:  pip install Pillow\n")
    sys.exit(1)

import numpy as np

# ============================================================================
# CONFIG
# ============================================================================
DATASET_DIR        = "dataset"
FRAMES_PER_STUDENT = 100
FRAME_SKIP         = 2           # run detector every N frames (keeps FPS high)
MIN_FACE_SIZE      = 60          # ignore faces smaller than this (px)
CAPTURE_W          = 1280
CAPTURE_H          = 720
FACE_CROP_SIZE     = 112         # ArcFace standard

#  Palette 
BG       = "#0d0d0d"
CARD     = "#16213e"
ACCENT   = "#00e5ff"
GREEN    = "#00e676"
RED      = "#ff1744"
ORANGE   = "#ff9100"
WHITE    = "#f0f0f0"
MUTED    = "#616161"

# ============================================================================
# CAMERA BACKEND
# ============================================================================
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    print("[WARN] picamera2 not found — will fall back to cv2.VideoCapture")


def open_camera():
    """
    Returns (cam_object, mode_string) or (None, None) on failure.
    mode_string is 'picamera2' or 'cv2'.
    """
    if HAS_PICAMERA2:
        try:
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (CAPTURE_W, CAPTURE_H), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            cam.configure(cfg)
            cam.start()
            time.sleep(1.5)          # let auto-exposure settle
            return cam, "picamera2"
        except Exception as e:
            print(f"[WARN] picamera2 failed: {e} — trying cv2 fallback")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, "cv2"


def grab_frame(cam, mode):
    """Returns (ok: bool, bgr_frame: ndarray | None)"""
    if mode == "picamera2":
        arr = cam.capture_array()          # shape (H, W, 3) RGB
        if arr is None:
            return False, None
        return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cam.read()


def close_camera(cam, mode):
    if cam is None:
        return
    try:
        if mode == "picamera2":
            cam.stop()
        else:
            cap.release()
    except Exception:
        pass


# ============================================================================
# FACE-DETECTION MODELS
# (loaded once in a background thread while the user fills in the name)
# ============================================================================
try:
    from insightface.app import FaceAnalysis
    USE_INSIGHTFACE = True
except ImportError:
    USE_INSIGHTFACE = False
    print("[WARN] insightface not found — using Haar Cascade fallback")

try:
    from ultralytics import YOLO
    USE_YOLO = True
except ImportError:
    USE_YOLO = False

_face_app    = None
_haar        = None
_yolo_model  = None
_models_evt  = threading.Event()   # set when loading is done


def _load_models_bg():
    global _face_app, _haar, _yolo_model
    try:
        if USE_INSIGHTFACE:
            _face_app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"]
            )
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
        else:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _haar = cv2.CascadeClassifier(path)

        if USE_YOLO:
            _yolo_model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[WARN] Model load error: {e}")
    finally:
        _models_evt.set()


def detect_faces_in_frame(bgr):
    """Returns list of (crop_bgr, (x1,y1,x2,y2)) for every face found."""
    if USE_INSIGHTFACE and _face_app is not None:
        faces = _face_app.get(bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue
            if hasattr(f, "norm_crop") and f.norm_crop is not None:
                crop = f.norm_crop
            else:
                crop = bgr[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
            out.append((crop, (x1, y1, x2, y2)))
        return out

    if _haar is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        dets = _haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        out = []
        for (x, y, w, h) in (dets if len(dets) else []):
            crop = bgr[y:y + h, x:x + w]
            crop = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
            out.append((crop, (x, y, x + w, y + h)))
        return out

    return []   # models not ready yet


def get_person_roi(bgr):
    """Tight ROI around the dominant detected person; fallback = full frame."""
    if not USE_YOLO or _yolo_model is None:
        return 0, 0, bgr.shape[1], bgr.shape[0]
    res = _yolo_model(bgr, classes=[0], conf=0.4, verbose=False)
    best, best_area = None, 0
    for r in res:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            a = (x2 - x1) * (y2 - y1)
            if a > best_area:
                best_area = a
                best = (x1, y1, x2, y2)
    if best:
        pad = 30
        h, w = bgr.shape[:2]
        return (max(0, best[0]-pad), max(0, best[1]-pad),
                min(w, best[2]+pad), min(h, best[3]+pad))
    return 0, 0, bgr.shape[1], bgr.shape[0]


# ============================================================================
# TKINTER APPLICATION
# ============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance System — Dataset Collection")
        self.configure(bg=BG)

        # Maximise window
        try:
            self.attributes("-zoomed", True)      # Linux / RPi OS
        except Exception:
            try:
                self.state("zoomed")               # Windows
            except Exception:
                self.geometry("1280x720")

        # Camera state
        self._cam        = None
        self._cam_mode   = None
        self._running    = False
        self._saved      = 0
        self._frame_ctr  = 0
        self._student    = ""
        self._save_dir   = ""
        self._after_id   = None
        self._photo_ref  = None  # keep Tk PhotoImage alive

        self.protocol("WM_DELETE_WINDOW", self._quit)

        # Start model loading immediately in background
        threading.Thread(target=_load_models_bg, daemon=True).start()

        # Launch the newly added Main Screen instead of going straight to entry
        self._show_main_screen()

    # =========================================================  SCREENS   ===

    def _clear_screen(self):
        for w in self.winfo_children():
            w.destroy()

    # ── SCREEN 0: Main Screen Options (New Feature) ──────────────────────────
    def _show_main_screen(self):
        self._stop_camera()
        self._clear_screen()

        wrap = tk.Frame(self, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="🎓  LIVE ATTENDANCE SYSTEM",
                 bg=BG, fg=ACCENT,
                 font=("Helvetica", 32, "bold")).pack(pady=(0, 6))

        tk.Label(wrap, text="Dataset Generation Panel",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 16)).pack(pady=(0, 50))

        card = tk.Frame(wrap, bg=CARD,
                        highlightbackground=ACCENT, highlightthickness=2,
                        padx=60, pady=50)
        card.pack()

        tk.Button(
            card,
            text="➕   Add New Student",
            command=self._show_name_screen,
            bg=GREEN, fg="#000000",
            font=("Helvetica", 16, "bold"),
            relief="flat", padx=40, pady=18,
            width=20,
            cursor="hand2",
            activebackground="#00c853",
            activeforeground="#000000",
        ).pack(pady=(0, 20))

        tk.Button(
            card,
            text="❌   Exit Program",
            command=self._quit,
            bg=RED, fg=WHITE,
            font=("Helvetica", 16, "bold"),
            relief="flat", padx=40, pady=18,
            width=20,
            cursor="hand2",
            activebackground="#d50000",
            activeforeground=WHITE,
        ).pack()

    # ── SCREEN 1: Name entry ─────────────────────────────────────────────────
    def _show_name_screen(self):
        self._stop_camera()
        self._clear_screen()

        # Centre container
        wrap = tk.Frame(self, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="🎓  LIVE ATTENDANCE SYSTEM",
                 bg=BG, fg=ACCENT,
                 font=("Helvetica", 30, "bold")).pack(pady=(0, 6))

        tk.Label(wrap, text="Face Dataset Collection",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 14)).pack(pady=(0, 46))

        card = tk.Frame(wrap, bg=CARD,
                        highlightbackground=ACCENT, highlightthickness=2,
                        padx=50, pady=44)
        card.pack()

        tk.Label(card, text="Student Name",
                 bg=CARD, fg=WHITE,
                 font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))

        self._name_var = tk.StringVar()
        self._entry = tk.Entry(
            card,
            textvariable=self._name_var,
            font=("Helvetica", 20),
            bg="#0d1b2a", fg=WHITE,
            insertbackground=ACCENT,
            relief="flat",
            width=24,
            highlightthickness=2,
            highlightbackground=MUTED,
            highlightcolor=ACCENT,
        )
        self._entry.pack(ipady=11, pady=(0, 8))
        self._entry.focus_set()
        self._entry.bind("<Return>", lambda _e: self._on_start())

        tk.Label(card,
                 text="Use underscore for spaces  (e.g. John_Doe)",
                 bg=CARD, fg=MUTED,
                 font=("Helvetica", 10)).pack(anchor="w", pady=(0, 20))

        self._err_var = tk.StringVar()
        tk.Label(card, textvariable=self._err_var,
                 bg=CARD, fg=RED,
                 font=("Helvetica", 11)).pack(pady=(0, 14))

        tk.Button(
            card,
            text="▶    Start Capture",
            command=self._on_start,
            bg=ACCENT, fg="#000000",
            font=("Helvetica", 15, "bold"),
            relief="flat", padx=20, pady=14,
            cursor="hand2",
            activebackground="#00b8d4",
            activeforeground="#000000",
        ).pack(fill="x", pady=(0, 10))
        
        # Added a back button to quickly return to the Home page
        tk.Button(
            card,
            text="⬅  Back to Menu",
            command=self._show_main_screen,
            bg=MUTED, fg=WHITE,
            font=("Helvetica", 12),
            relief="flat", padx=10, pady=8,
            cursor="hand2",
            activebackground="#757575",
            activeforeground=WHITE,
        ).pack(fill="x")

        tk.Label(wrap,
                 text="Stand in front of the camera · slowly move head  LEFT ← → RIGHT · UP ↑ ↓ DOWN",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 10),
                 wraplength=520).pack(pady=(32, 0))

    # ── SCREEN 2: Camera feed ────────────────────────────────────────────────
    def _show_camera_screen(self):
        self._clear_screen()

        # ── Top bar ──
        top = tk.Frame(self, bg=CARD, height=60)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(top, text=f"  📷  Capturing:  {self._student}",
                 bg=CARD, fg=ACCENT,
                 font=("Helvetica", 15, "bold")).pack(side="left",
                                                       padx=20, pady=10)

        self._status_lbl = tk.Label(top, text="Initialising…",
                                    bg=CARD, fg=WHITE,
                                    font=("Helvetica", 12))
        self._status_lbl.pack(side="right", padx=24)

        # ── Camera canvas (fills all middle space) ──
        self._canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        # ── Bottom bar ──
        bot = tk.Frame(self, bg=CARD, height=72)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        left = tk.Frame(bot, bg=CARD)
        left.pack(side="left", fill="x", expand=True, padx=24, pady=12)

        tk.Label(left, text="Progress",
                 bg=CARD, fg=MUTED,
                 font=("Helvetica", 9)).pack(anchor="w")

        self._pbar = tk.Canvas(left, bg=MUTED, height=16,
                               highlightthickness=0)
        self._pbar.pack(fill="x", pady=(4, 0))

        self._count_lbl = tk.Label(bot,
                                   text=f"0 / {FRAMES_PER_STUDENT}",
                                   bg=CARD, fg=GREEN,
                                   font=("Helvetica", 20, "bold"))
        self._count_lbl.pack(side="right", padx=32)

        # Kick off camera init after Tk lays out the widgets
        self.after(200, self._init_camera)

    # ── SCREEN 3: Done ───────────────────────────────────────────────────────
    def _show_done_screen(self):
        self._stop_camera()
        self._clear_screen()

        wrap = tk.Frame(self, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="✅", bg=BG, fg=GREEN,
                 font=("Helvetica", 64)).pack()

        tk.Label(wrap,
                 text=f"Captured  {self._saved}  images",
                 bg=BG, fg=GREEN,
                 font=("Helvetica", 26, "bold")).pack(pady=(8, 4))

        tk.Label(wrap,
                 text=f"Student :  {self._student}",
                 bg=BG, fg=WHITE,
                 font=("Helvetica", 15)).pack()

        tk.Label(wrap,
                 text=f"Saved to :  dataset/{self._student}/",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 11)).pack(pady=(4, 48))

        tk.Button(
            wrap,
            text="➕   Add Next Student",
            command=self._show_name_screen,
            bg=GREEN, fg="#000000",
            font=("Helvetica", 16, "bold"),
            relief="flat", padx=50, pady=18,
            cursor="hand2",
            activebackground="#00c853",
        ).pack(fill="x", pady=(0, 16))

        tk.Button(
            wrap,
            text="⏹   Finish & Exit",
            command=self._quit,
            bg=MUTED, fg=WHITE,
            font=("Helvetica", 13),
            relief="flat", padx=50, pady=12,
            cursor="hand2",
            activebackground="#757575",
        ).pack(fill="x")

    # =========================================================  LOGIC  ======

    def _on_start(self):
        raw  = self._name_var.get().strip()
        name = raw.replace(" ", "_")

        if not name:
            self._err_var.set("⚠  Name cannot be empty.")
            return
        bad = [c for c in name if not (c.isalnum() or c == "_")]
        if bad:
            self._err_var.set(f"⚠  Invalid characters: {''.join(set(bad))}")
            return

        self._student  = name
        self._save_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(self._save_dir, exist_ok=True)

        self._saved     = 0
        self._frame_ctr = 0
        self._show_camera_screen()

    def _init_camera(self):
        """Called once after camera screen layout is ready."""
        if not _models_evt.is_set():
            self._status_lbl.config(text="⏳  Loading AI models…")
            self.after(300, self._init_camera)
            return

        self._status_lbl.config(text="⏳  Opening camera…")
        self.update_idletasks()

        cam, mode = open_camera()
        if cam is None:
            self._status_lbl.config(text="❌  Camera not found! Check connection.")
            return

        self._cam      = cam
        self._cam_mode = mode
        self._running  = True
        self._status_lbl.config(text="● LIVE  —  face the camera")
        self._loop()

    def _loop(self):
        """
        Main camera loop — runs entirely in the Tk main thread via after().
        """
        if not self._running:
            return

        ok, frame = grab_frame(self._cam, self._cam_mode)
        if not ok or frame is None:
            self._after_id = self.after(5, self._loop)
            return

        self._frame_ctr += 1
        display = frame.copy()

        # ── Face detection every FRAME_SKIP frames ──
        if self._frame_ctr % FRAME_SKIP == 0 and self._saved < FRAMES_PER_STUDENT:
            px1, py1, px2, py2 = get_person_roi(frame)
            region = frame[py1:py2, px1:px2]
            if region.size > 0:
                for (crop, (fx1, fy1, fx2, fy2)) in detect_faces_in_frame(region):
                    # Absolute coords on full frame
                    ax1, ay1 = px1 + fx1, py1 + fy1
                    ax2, ay2 = px1 + fx2, py1 + fy2

                    # Green detection box on display
                    cv2.rectangle(display, (ax1, ay1), (ax2, ay2),
                                  (0, 230, 80), 2)

                    # Save crop
                    fpath = os.path.join(
                        self._save_dir,
                        f"{self._student}_{self._saved:04d}.jpg"
                    )
                    cv2.imwrite(fpath, crop)
                    self._saved += 1
                    if self._saved >= FRAMES_PER_STUDENT:
                        break

        # ── HUD overlay ──
        self._draw_hud(display)

        # ── Push frame to Tkinter canvas ──
        self._render_frame(display)

        # ── Update progress widgets ──
        self._refresh_progress()

        # ── Done? ──
        if self._saved >= FRAMES_PER_STUDENT:
            self._running = False
            self.after(500, self._show_done_screen)
            return

        # Schedule next frame
        self._after_id = self.after(1, self._loop)

    # ───────────────────────────────────────────  helpers

    def _draw_hud(self, bgr):
        h, w = bgr.shape[:2]
        pct  = int(self._saved / FRAMES_PER_STUDENT * 100)

        # Semi-transparent dark strip at top
        strip = bgr[:50, :].copy()
        cv2.rectangle(strip, (0, 0), (w, 50), (0, 0, 0), -1)
        bgr[:50, :] = cv2.addWeighted(strip, 0.6, bgr[:50, :], 0.4, 0)

        cv2.putText(
            bgr,
            f"Student: {self._student}   |   Captured: {self._saved}/{FRAMES_PER_STUDENT}   ({pct}%)",
            (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
            (0, 230, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            bgr,
            "Move head:  LEFT  ←  →  RIGHT  |  UP  ↑  ↓  DOWN",
            (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (180, 180, 180), 1, cv2.LINE_AA
        )

    def _render_frame(self, bgr):
        """Scale BGR frame to fill the canvas, convert to PhotoImage."""
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        fh, fw = bgr.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo   = ImageTk.PhotoImage(pil_img)

        self._photo_ref = photo          # prevent garbage collection
        xo = (cw - nw) // 2
        yo = (ch - nh) // 2
        self._canvas.delete("all")
        self._canvas.create_image(xo, yo, anchor="nw", image=photo)

    def _refresh_progress(self):
        pct = self._saved / FRAMES_PER_STUDENT
        self._count_lbl.config(text=f"{self._saved} / {FRAMES_PER_STUDENT}")

        pw = self._pbar.winfo_width()
        if pw < 2:
            return
        fw = int(pw * pct)
        colour = GREEN if pct < 0.85 else ACCENT
        self._pbar.delete("all")
        if fw > 0:
            self._pbar.create_rectangle(0, 0, fw, 16, fill=colour, outline="")

    def _stop_camera(self):
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._cam is not None:
            close_camera(self._cam, self._cam_mode)
            self._cam      = None
            self._cam_mode = None
        
    def _quit(self):
        self._stop_camera()
        self.destroy()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    app = App()
    app.mainloop()
