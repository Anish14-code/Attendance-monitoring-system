from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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

# Paths
BASE_DIR    = Path("/home/dps/attendance/codes2/pratham1")
DATASET_DIR = BASE_DIR / "dataset"
ARCFACE_HEF = BASE_DIR / "models1" / "arcface_r50.hef"
OUTPUT_FILE = BASE_DIR / "encodings.pkl"
REPORT_FILE = BASE_DIR / "encoding_report.txt"

# Constants
ARCFACE_SIZE = 112
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTLIER_STD_FACTOR = 1.5
INTER_CLASS_WARN_THRESHOLD = 0.55


class HailoModel:
    def __init__(self, hef_path: Path, format_type: FormatType, device: VDevice) -> None:
        if not hef_path.exists():
            sys.exit(f"[ERROR] HEF not found: {hef_path}")

        self.device = device
        self.hef = HEF(str(hef_path))
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()

        in_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=format_type
        )
        out_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        in_info = self.hef.get_input_vstream_infos()[0]
        self.input_name   = in_info.name
        self.input_shape  = tuple(in_info.shape)
        self.output_infos = self.hef.get_output_vstream_infos()
        self.output_names = [o.name for o in self.output_infos]

        self.activation = self.network_group.activate(self.network_params)
        self.activation.__enter__()
        self.infer_pipe = InferVStreams(self.network_group, in_params, out_params)
        self.infer_pipe.__enter__()

        print(f"[HAILO] Loaded model: {hef_path.name}")

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


def extract_embedding(arcface: HailoModel, crop_bgr: np.ndarray) -> np.ndarray:
    # Ensure it perfectly matches standard dimensions
    if crop_bgr.shape[:2] != (ARCFACE_SIZE, ARCFACE_SIZE):
        crop_bgr = cv2.resize(crop_bgr, (ARCFACE_SIZE, ARCFACE_SIZE), interpolation=cv2.INTER_LINEAR)
        
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    outputs = arcface.run(rgb)
    
    # Extract feature map tensor
    feats = outputs.get("fc1") or next(iter(outputs.values()))
    feats = np.squeeze(np.asarray(feats, dtype=np.float32))
    
    # Apply precise Unit-Norm Scaling
    norm = np.linalg.norm(feats)
    if norm > 1e-6:
        feats = feats / norm
    return feats


def build_student_encoding(
    name: str, embeddings: list[np.ndarray], status_counts: dict[str, int]
) -> tuple[np.ndarray, list[str]]:
    arr = np.array(embeddings)
    initial_centroid = np.mean(arr, axis=0)
    
    # Calculate cosine distance metrics
    similarities = np.dot(arr, initial_centroid) / (
        np.linalg.norm(arr, axis=1) * np.linalg.norm(initial_centroid) + 1e-8
    )
    mean_sim = float(np.mean(similarities))
    std_sim  = float(np.std(similarities))

    thresh = mean_sim - (OUTLIER_STD_FACTOR * std_sim)
    valid_idx = np.where(similarities >= thresh)[0]
    
    if len(valid_idx) == 0:
        valid_idx = np.arange(len(embeddings))

    filtered_embeddings = arr[valid_idx]
    final_centroid = np.mean(filtered_embeddings, axis=0)
    
    # Re-normalize centroid vector after mean calculation 
    centroid_norm = np.linalg.norm(final_centroid)
    if centroid_norm > 1e-6:
        final_centroid = final_centroid / centroid_norm

    report = [
        f"=== {name} ===",
        f"  Total processed files   : {status_counts['processed']}",
        f"  Failed reads / errors    : {status_counts['failed']}",
        f"  Initial uniformity mean : {mean_sim:.4f}, std={std_sim:.4f}",
        f"  Outliers dropped        : {len(embeddings) - len(valid_idx)}",
        f"  Final accepted frames   : {len(valid_idx)}",
        "",
    ]
    return final_centroid, report


def print_inter_class_similarity(database: dict[str, np.ndarray]) -> None:
    names = list(database.keys())
    if len(names) < 2:
        return
    print("\n[ANALYSIS] Cross-Class Similarity Profile:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = float(np.dot(database[names[i]], database[names[j]]))
            print(f"   Similarity ({names[i]} <-> {names[j]}): {sim:.4f}")
            if sim > INTER_CLASS_WARN_THRESHOLD:
                print(f"   [!] WARNING: Profiles '{names[i]}' and '{names[j]}' are extremely close.")


def main() -> None:
    if not DATASET_DIR.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {DATASET_DIR}")

    print("[HAILO] Initializing Shared Hardware Device Context...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    shared_device = VDevice(params)

    # We only need ArcFace here since dataset is pre-cropped
    arcface = HailoModel(ARCFACE_HEF, FormatType.UINT8, shared_device)

    database: dict[str, np.ndarray] = {}
    full_report: list[str] = []

    try:
        student_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
        for sdir in student_dirs:
            student_name = sdir.name
            print(f"\n[PROCESSING] Extracting database targets for: {student_name}")

            status_counts = {"processed": 0, "failed": 0}
            image_embeddings: list[np.ndarray] = []

            img_paths = [p for p in sdir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if not img_paths:
                continue

            for ipath in img_paths:
                frame = cv2.imread(str(ipath))
                if frame is None:
                    status_counts["failed"] += 1
                    continue
                
                emb = extract_embedding(arcface, frame)
                image_embeddings.append(emb)
                status_counts["processed"] += 1

            if not image_embeddings:
                continue

            centroid, report_lines = build_student_encoding(
                student_name, image_embeddings, status_counts
            )
            database[student_name] = centroid
            full_report.extend(report_lines)

    finally:
        arcface.close()
        try:
            shared_device.release()
            print("[HAILO] VDevice context closed cleanly.")
        except Exception:
            pass

    if not database:
        sys.exit("[ERROR] Zero encodings built successfully.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("wb") as fh:
        pickle.dump(database, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with REPORT_FILE.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(full_report))

    print(f"\n[SAVED] {len(database)} students registered -> {OUTPUT_FILE}")
    print_inter_class_similarity(database)


if __name__ == "__main__":
    main()
