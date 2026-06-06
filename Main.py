from __future__ import annotations

import argparse
import collections
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("[ERROR] Picamera2 not found.")

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
    sys.exit("[ERROR] hailo_platform not found. Activate/install HailoRT first.")

#  Paths 
BASE_DIR            = Path("/home/dps/attendance/codes2/pratham1")
DEFAULT_ENCODINGS   = BASE_DIR / "encodings.pkl"
DEFAULT_SCRFD_HEF   = BASE_DIR / "models1" / "scrfd_2.5g_h8l.hef"      
DEFAULT_ARCFACE_HEF = BASE_DIR / "models1" / "arcface_r50.hef"

#  Constants 
SCRFD_SIZE    = 640
ARCFACE_SIZE  = 112
SCRFD_STRIDES = (8, 16, 32)
SMOOTH_FRAMES = 5

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


@dataclass
class Face:
    box:       np.ndarray   
    landmarks: np.ndarray   
    score:     float        


@dataclass
class TrackedFace:
    scores:     dict[str, collections.deque] = field(default_factory=dict)
    last_name:  str   = "Unknown"
    last_score: float = 0.0

    def update(self, scores: dict[str, float], known_names: list[str]) -> None:
        for name in known_names:
            if name not in self.scores:
                self.scores[name] = collections.deque(maxlen=SMOOTH_FRAMES)
            self.scores[name].append(scores.get(name, 0.0))

    def smoothed_scores(self) -> dict[str, float]:
        if not self.scores:
            return {}
        return {name: float(np.mean(list(buf))) for name, buf in self.scores.items()}


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

            # Keep validation light to avoid skipping rotated faces live
            if not (np.any(lms < 0) or np.any(lms[:, 0] >= frame_w) or np.any(lms[:, 1] >= frame_h)):
                faces.append(Face(box=box.astype(int), landmarks=lms, score=float(scores[idx])))

    # Implement NMS
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
    
    # Enforce unit normalization mapping
    norm = np.linalg.norm(feats)
    if norm > 1e-6:
        feats = feats / norm
    return feats


class FaceSmoother:
    def __init__(self) -> None:
        self.tracks: list[TrackedFace] = []

    def update(self, current_faces: list[Face], raw_scores_list: list[dict[str, float]], known_names: list[str]) -> list[tuple[Face, str, float, float]]:
        new_tracks: list[TrackedFace] = []
        results: list[tuple[Face, str, float, float]] = []

        for face, scores in zip(current_faces, raw_scores_list):
            best_track = None
            best_iou   = 0.2

            for t in self.tracks:
                if hasattr(t, "last_box"):
                    boxA, boxB = t.last_box, face.box
                    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    ovr = inter / float((boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter + 1e-6)
                    if ovr > best_iou:
                        best_iou = ovr
                        best_track = t

            if best_track is None: best_track = TrackedFace()
            best_track.update(scores, known_names)
            best_track.last_box = face.box
            new_tracks.append(best_track)

            smoothed = best_track.smoothed_scores()
            if smoothed:
                sorted_identities = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
                best_name, best_score = sorted_identities[0]
                margin = (best_score - sorted_identities[1][1]) if len(sorted_identities) > 1 else best_score
            else:
                best_name, best_score, margin = "Unknown", 0.0, 0.0

            results.append((face, best_name, best_score, margin))

        self.tracks = new_tracks
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.38, help="Cosine alignment threshold")
    parser.add_argument("--min-margin", type=float, default=0.10, help="Confidence margin threshold")
    parser.add_argument("--det-thresh", type=float, default=0.45, help="SCRFD object threshold")
    args = parser.parse_args()

    if not DEFAULT_ENCODINGS.exists():
        sys.exit(f"[ERROR] Database file missing at {DEFAULT_ENCODINGS}")

    with DEFAULT_ENCODINGS.open("rb") as fh:
        known = pickle.load(fh)
    known_names = list(known.keys())
    print(f"[SYSTEM] Loaded Profiles: {known_names}")

    scrfd    = HailoModel(DEFAULT_SCRFD_HEF, "uint8")
    arcface  = HailoModel(DEFAULT_ARCFACE_HEF, "uint8")
    smoother = FaceSmoother()

    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()

    smooth_fps = 30.0
    last_time  = time.time()

    try:
        while True:
            arr = cam.capture_array()
            if arr is None: continue

            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            fh_h, fh_w = frame.shape[:2]

            tensor, scale, pad_x, pad_y = preprocess_scrfd(frame)
            outputs = scrfd.run(tensor)
            faces = decode_scrfd(outputs, args.det_thresh, 0.45, fh_w, fh_h, pad_x, pad_y, scale)

            raw_scores_list = []
            for face in faces:
                aligned = align_face(frame, face.landmarks)
                emb = extract_embedding(arcface, aligned)

                scores = {}
                for name, centroid in known.items():
                    scores[name] = float(np.dot(emb, centroid))
                raw_scores_list.append(scores)

            smoothed_results = smoother.update(faces, raw_scores_list, known_names)
            now        = time.time()
            smooth_fps = 0.9 * smooth_fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time  = now

            recognized_count = 0
            for face, name, score, margin in smoothed_results:
                display_name = name
                if score < args.threshold or margin < args.min_margin:
                    display_name = "Unknown"
                else:
                    recognized_count += 1

                # Draw to view frame
                x1, y1, x2, y2 = face.box
                color = (50, 50, 240) if display_name == "Unknown" else (76, 230, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"{display_name} ({score:.2f})" if display_name != "Unknown" else "Unknown Face"
                cv2.putText(frame, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Overlay stats HUD panel
            cv2.rectangle(frame, (0, 0), (520, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {smooth_fps:.1f} | Tracking: {len(faces)} faces", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            cv2.putText(frame, f"Recognized: {recognized_count} | Match Limit: {args.threshold:.2f}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow("Live Attendance Monitoring System", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        scrfd.close()
        arcface.close()
        cam.stop()


if __name__ == "__main__":
    main()
