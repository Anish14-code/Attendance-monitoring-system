from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
import signal
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    import libcamera
except ImportError:
    sys.exit("[ERROR] Picamera2 / libcamera dependencies missing.")

try:
    from hailo_platform import (
        ConfigureParams,
        FormatType,
        HEF,
        HailoSchedulingAlgorithm,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
except ImportError:
    sys.exit("[ERROR] HailoRT components missing. Check Hat software initialization.")

# Paths Configuration
BASE_DIR            = Path("/home/dps/attendance/pratham1")
Path1               = BASE_DIR / "csv_files"
DEFAULT_ENCODINGS   = BASE_DIR / "encodings.pkl"
DEFAULT_SCRFD_HEF   = BASE_DIR / "models1" / "scrfd_2.5g_h8l.hef"      
DEFAULT_ARCFACE_HEF = BASE_DIR / "models1" / "arcface_r50.hef"

# Model Resolution Constants
SCRFD_SIZE    = 640
ARCFACE_SIZE  = 112
SCRFD_STRIDES = (8, 16, 32)

REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

running = True

def signal_handler(sig, frame):
    global running
    print("\n[SYSTEM] Intercepted termination payload. Clearing allocations...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@dataclass
class Face:
    box:       np.ndarray   
    landmarks: np.ndarray   
    score:     float        


class HailoModel:
    _shared_device: Any  = None
    _shared_count:  int  = 0

    def __init__(self, hef_path: Path, input_format: str = "uint8") -> None:
        if not hef_path.exists():
            sys.exit(f"[ERROR] HEF not found: {hef_path}")

        self.device = self._get_shared_device()
        self.hef    = HEF(str(hef_path))
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group  = self.device.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()

        fmt        = FormatType.UINT8 if input_format == "uint8" else FormatType.FLOAT32
        in_params  = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=fmt)
        out_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.input_name   = self.hef.get_input_vstream_infos()[0].name
        self.output_names = [o.name for o in self.hef.get_output_vstream_infos()]

        self.activation = self.network_group.activate(self.network_params)
        self.activation.__enter__()
        self.infer_pipe = InferVStreams(self.network_group, in_params, out_params)
        self.infer_pipe.__enter__()

    @classmethod
    def _get_shared_device(cls) -> Any:
        if cls._shared_device is None:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            cls._shared_device = VDevice(params)
        cls._shared_count += 1
        return cls._shared_device

    @classmethod
    def _release_shared_device(cls) -> None:
        cls._shared_count = max(0, cls._shared_count - 1)
        if cls._shared_count == 0 and cls._shared_device is not None:
            try:
                cls._shared_device.release()
            except Exception:
                pass
            cls._shared_device = None

    def run(self, image: np.ndarray) -> dict[str, Any]:
        batch = np.expand_dims(np.ascontiguousarray(image), axis=0)
        return self.infer_pipe.infer({self.input_name: batch})

    def close(self) -> None:
        for obj in (getattr(self, "infer_pipe", None), getattr(self, "activation", None)):
            if obj is not None:
                try:
                    obj.__exit__(None, None, None)
                except Exception:
                    pass
        self._release_shared_device()


def preprocess_scrfd(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w   = bgr.shape[:2]
    scale  = min(SCRFD_SIZE / w, SCRFD_SIZE / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((SCRFD_SIZE, SCRFD_SIZE, 3), dtype=np.uint8)
    pad_x   = (SCRFD_SIZE - new_w) // 2
    pad_y   = (SCRFD_SIZE - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, pad_x, pad_y


def collect_tensors(outputs: dict[str, Any], element_count: int) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values(): visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value: visit(item)
            return
        arr = np.squeeze(np.asarray(value, dtype=np.float32))
        if arr.size == 0: return
        if element_count == 1:
            if arr.ndim == 1 or arr.shape[-1] <= 3: tensors.append(arr.reshape(-1, 1))
        elif arr.ndim >= 1 and arr.shape[-1] % element_count == 0:
            tensors.append(arr.reshape(-1, element_count))
    visit(outputs)
    tensors.sort(key=lambda x: x.shape[0], reverse=True)
    return tensors


def closest_by_rows(tensors: list[np.ndarray], rows: int) -> np.ndarray | None:
    if not tensors: return None
    return min(tensors, key=lambda x: abs(x.shape[0] - rows))


def infer_stride(rows: int) -> int:
    expected = {12800: 8, 3200: 16, 800: 32}
    if rows in expected: return expected[rows]
    return min(SCRFD_STRIDES, key=lambda s: abs(rows - ((SCRFD_SIZE // s) ** 2) * 2))


def anchor_centers(stride: int, rows: int) -> np.ndarray:
    feat    = SCRFD_SIZE // stride
    anchors = max(1, rows // (feat * feat))
    y, x    = np.mgrid[:feat, :feat]
    centers = np.stack(((x + 0.5) * stride, (y + 0.5) * stride), axis=-1).reshape(-1, 2)
    centers = np.repeat(centers, anchors, axis=0)
    if len(centers) < rows: centers = np.resize(centers, (rows, 2))
    return centers[:rows].astype(np.float32)


def distance_to_box(center: np.ndarray, pred: np.ndarray, stride: int) -> np.ndarray:
    vals = pred.astype(np.float32)
    if np.max(np.abs(vals)) < 40.0: vals = vals * stride
    x, y = center
    return np.array([x - vals[0], y - vals[1], x + vals[2], y + vals[3]], dtype=np.float32)


def distance_to_landmarks(center: np.ndarray, pred: np.ndarray, stride: int) -> np.ndarray:
    vals = pred.reshape(5, 2).astype(np.float32)
    if np.max(np.abs(vals)) < 40.0: vals = vals * stride
    return vals + center.reshape(1, 2)


def decode_scrfd(outputs: dict[str, Any], score_thresh: float, nms_thresh: float, frame_w: int, frame_h: int, pad_x: int, pad_y: int, scale: float) -> list[Face]:
    score_tensors = collect_tensors(outputs, 1)
    box_tensors   = collect_tensors(outputs, 4)
    lm_tensors    = collect_tensors(outputs, 10)

    faces: list[Face] = []
    for score_t in score_tensors:
        box_t = closest_by_rows(box_tensors, len(score_t))
        lm_t  = closest_by_rows(lm_tensors, len(score_t))
        if box_t is None or lm_t is None: continue

        stride  = infer_stride(len(score_t))
        centers = anchor_centers(stride, len(score_t))
        scores  = score_t.reshape(-1)
        if scores.min() < 0.0 or scores.max() > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        keep = np.where(scores >= score_thresh)[0]
        for idx in keep:
            box = distance_to_box(centers[idx], box_t[idx], stride)
            lms = distance_to_landmarks(centers[idx], lm_t[idx], stride)

            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            lms[:, 0]   = (lms[:, 0] - pad_x) / scale
            lms[:, 1]   = (lms[:, 1] - pad_y) / scale

            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_h - 1)

            if not (np.any(lms < 0) or np.any(lms[:, 0] >= frame_w) or np.any(lms[:, 1] >= frame_h)):
                faces.append(Face(box=box.astype(int), landmarks=lms, score=float(scores[idx])))

    if not faces: return []
    order = np.argsort([f.score for f in faces])[::-1]
    keep_faces: list[Face] = []
    while len(order) > 0:
        i = int(order[0])
        keep_faces.append(faces[i])
        if len(order) == 1: break
        rest = order[1:]
        ious = []
        for j in rest:
            xA, yA = max(faces[i].box[0], faces[j].box[0]), max(faces[i].box[1], faces[j].box[1])
            xB, yB = min(faces[i].box[2], faces[j].box[2]), min(faces[i].box[3], faces[j].box[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            a_area = (faces[i].box[2]-faces[i].box[0]) * (faces[i].box[3]-faces[i].box[1])
            b_area = (faces[j].box[2]-faces[j].box[0]) * (faces[j].box[3]-faces[j].box[1])
            ious.append(inter / float(a_area + b_area - inter + 1e-6))
        order = rest[np.array(ious) <= nms_thresh]
    return keep_faces


def align_face(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), REFERENCE_LANDMARKS, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        x1, y1 = np.min(landmarks, axis=0).astype(int)
        x2, y2 = np.max(landmarks, axis=0).astype(int)
        crop = bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0: return np.zeros((ARCFACE_SIZE, ARCFACE_SIZE, 3), dtype=np.uint8)
        return cv2.resize(crop, (ARCFACE_SIZE, ARCFACE_SIZE))
    return cv2.warpAffine(bgr, M, (ARCFACE_SIZE, ARCFACE_SIZE), flags=cv2.INTER_LINEAR)


def extract_embedding(arcface: HailoModel, aligned_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    outputs = arcface.run(rgb)
    feats = outputs.get("fc1") or next(iter(outputs.values()))
    feats = np.squeeze(np.asarray(feats, dtype=np.float32))
    
    norm = np.linalg.norm(feats)
    if norm > 1e-6:
        feats = feats / norm
    return feats


def csv_writer_worker(io_queue: queue.Queue, output_path: Path, headers: list[str]):
    while True:
        data = io_queue.get()
        if data is None:
            break
        rows_to_save = data
        try:
            with open(output_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows_to_save)
        except Exception as e:
            print(f"[IO ERROR] Failed asynchronous commit: {e}")
        io_queue.task_done()


def main() -> None:
    global running
    parser = argparse.ArgumentParser()
    # OPTIMIZATION: Minor score calibration adjustment for better dynamic classroom captures
    parser.add_argument("--threshold", type=float, default=0.44, help="Cosine alignment threshold")
    parser.add_argument("--min-margin", type=float, default=0.15, help="Confidence margin threshold")
    parser.add_argument("--det-thresh", type=float, default=0.35, help="SCRFD object threshold (relaxed for distant tiny faces)")
    args = parser.parse_args()

    if not DEFAULT_ENCODINGS.exists():
        sys.exit(f"[ERROR] Database file missing at {DEFAULT_ENCODINGS}")

    with DEFAULT_ENCODINGS.open("rb") as fh:
        known = pickle.load(fh)
    known_names = sorted(list(known.keys()))
    print(f"[SYSTEM] Loaded Profiles: {known_names}")

    db_embeddings_list = []
    db_mapping_indices = []
    for name_idx, name in enumerate(known_names):
        for emb in known[name]:
            db_embeddings_list.append(emb)
            db_mapping_indices.append(name_idx)
            
    db_matrix = np.array(db_embeddings_list, dtype=np.float32)  
    db_map = np.array(db_mapping_indices, dtype=int)

    file_timestamp = datetime.now().strftime("%H.%M.%S__%Y.%m.%d_")
    attendance_file = Path1 / f"attendance_{file_timestamp}.csv"

    headers = ["Name", "Status"]
    csv_rows = [[name, "Absent/Not Marked"] for name in known_names]

    with open(attendance_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)
    print(f"[SYSTEM] Generated session file: {attendance_file}")

    io_queue = queue.Queue()
    io_thread = threading.Thread(target=csv_writer_worker, args=(io_queue, attendance_file, headers), daemon=True)
    io_thread.start()

    marked_attendance: set[str] = set()

    scrfd    = HailoModel(DEFAULT_SCRFD_HEF, "uint8")
    arcface  = HailoModel(DEFAULT_ARCFACE_HEF, "uint8")

    cam = Picamera2()
    # OPTIMIZATION: Configure camera to pull complete native 12MP high-resolution stills
    cfg = cam.create_still_configuration(main={"size": (4608, 2592), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    
    try:
        cam.set_controls({"AfMode": libcamera.controls.AfModeEnum.Manual, "LensPosition": 0.45})
        print("[SYSTEM] Fixed hyperfocal focus locked matrix deployed.")
    except Exception as ex:
        print(f"[WARN] Focus override error: {ex}. Proceeding with stock logic.")

    print("[SYSTEM] Interval-Driven Burst Mode active. Checking classroom every 5 seconds...")

    try:
        while running:
            start_tick = time.time()
            
            # Grab a high-density raw 12-Megapixel still artifact
            arr = cam.capture_array()
            if arr is None: 
                continue

            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            fh_h, fh_w = frame.shape[:2]

            # OPTIMIZATION: 3x3 High-Density Matrix Tiling Pass (9 overlapping regions)
            # Dividing 4608x2592 into a 3x3 matrix keeps each slice in a wide aspect ratio.
            # This completely stops downscale pixel distortion for far away students.
            w_step, h_step = fh_w // 3, fh_h // 3
            overlap_x, overlap_y = 200, 150  # Wide margins to securely trap faces on boundaries
            
            tiles = []
            for r in range(3):
                for c in range(3):
                    y_start = max(0, r * h_step - (overlap_y if r > 0 else 0))
                    y_end   = min(fh_h, (r + 1) * h_step + (overlap_y if r < 2 else 0))
                    x_start = max(0, c * w_step - (overlap_x if c > 0 else 0))
                    x_end   = min(fh_w, (c + 1) * w_step + (overlap_x if c < 2 else 0))
                    
                    crop_segment = frame[y_start:y_end, x_start:x_end]
                    tiles.append((crop_segment, x_start, y_start))

            all_detected_faces = []

            # Stream each tile through the Hailo accelerator
            for tile_img, offset_x, offset_y in tiles:
                if tile_img.size == 0:
                    continue
                th_h, th_w = tile_img.shape[:2]
                tensor, scale, pad_x, pad_y = preprocess_scrfd(tile_img)
                outputs = scrfd.run(tensor)
                tile_faces = decode_scrfd(outputs, args.det_thresh, 0.45, th_w, th_h, pad_x, pad_y, scale)
                
                for face in tile_faces:
                    face.box[0] += offset_x
                    face.box[2] += offset_x
                    face.box[1] += offset_y
                    face.box[3] += offset_y
                    face.landmarks[:, 0] += offset_x
                    face.landmarks[:, 1] += offset_y
                    all_detected_faces.append(face)

            # Clean duplicate edge counts with a Non-Maximum Suppression filter
            if len(all_detected_faces) > 1:
                all_detected_faces.sort(key=lambda x: x.score, reverse=True)
                filtered_faces = []
                for f in all_detected_faces:
                    keep = True
                    for k in filtered_faces:
                        xA, yA = max(f.box[0], k.box[0]), max(f.box[1], k.box[1])
                        xB, yB = min(f.box[2], k.box[2]), min(f.box[3], k.box[3])
                        inter = max(0, xB - xA) * max(0, yB - yA)
                        a_area = (f.box[2]-f.box[0]) * (f.box[3]-f.box[1])
                        b_area = (k.box[2]-k.box[0]) * (k.box[3]-k.box[1])
                        if (inter / float(a_area + b_area - inter + 1e-6)) > 0.40:
                            keep = False
                            break
                    if keep:
                        filtered_faces.append(f)
                faces = filtered_faces
            else:
                faces = all_detected_faces

            file_needs_update = False

            # Extract embeddings and cross-verify with database matrices
            for face in faces:
                aligned = align_face(frame, face.landmarks)
                emb = extract_embedding(arcface, aligned)

                # High-speed matrix multiplication pass
                all_similarities = np.dot(db_matrix, emb)
                
                best_name = "Unknown"
                best_score = 0.0
                second_score = 0.0

                for name_idx, name in enumerate(known_names):
                    student_mask = (db_map == name_idx)
                    max_sim = float(np.max(all_similarities[student_mask]))
                    if max_sim > best_score:
                        second_score = best_score
                        best_score = max_sim
                        best_name = name
                    elif max_sim > second_score:
                        second_score = max_sim

                margin = best_score - second_score
                
                # Check threshold gate assignments
                if best_score >= args.threshold and margin >= args.min_margin:
                    if best_name not in marked_attendance:
                        marked_attendance.add(best_name)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[BURST RECOGNIZED] Verified Match: {best_name} | Cosine Score: {best_score:.2f}")
                        
                        for row in csv_rows:
                            if row[0] == best_name:
                                row[1] = f"Present ({timestamp})"
                                file_needs_update = True
                                break

            if file_needs_update:
                io_queue.put(list(csv_rows))

            # OPTIMIZATION: Compute exact processing execution cost and sleep for the remainder of the 10-second interval
            execution_cost = time.time() - start_tick
            sleep_duration = max(0.1, 5.0 - execution_cost)
            
            # Non-blocking shutdown listener break inside downtime loops
            for _ in range(int(sleep_duration * 20)):
                if not running: 
                    break
                time.sleep(0.05)

    finally:
        print("[SYSTEM] Closing active peripheral instances...")
        io_queue.put(None)  
        io_thread.join(timeout=2.0)
        cam.stop()
        scrfd.close()
        arcface.close()
        print("[SYSTEM] Deployment terminated cleanly.")


if __name__ == "__main__":
    main()
