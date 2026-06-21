#!/usr/bin/env python3
"""
=============================================================================
LIVE ATTENDANCE SYSTEM - SCRIPT 2: ENCODING GENERATION
=============================================================================
Hardware : RPi5 + AI HAT+ (13 TOPS / Hailo-8L)
Model    : ArcFace R50 (arcface_r50.hef) on Hailo AI HAT

Key improvements over v1
────────────────────────
• ArcFace input quality  : crops saved at 224×224 are downscaled to 112 with
                           INTER_AREA — sharper than resizing from a tiny crop
• Tighter outlier filter : std factor reduced 1.5 → 1.2, removing more
                           drifted embeddings without gutting the gallery
• Per-embedding re-norm  : explicit L2 normalisation after HEF output (already
                           present, kept and verified)
• Lower inter-class warn : threshold lowered 0.65 → 0.55 — catches confusable
                           faces at a safer margin
• Minimum gallery size   : if filtered gallery < 5 embeddings, outlier
                           filtering is skipped entirely (small dataset guard)
• Saves shape logged     : report now records the embedding matrix shape
• No functional regressions — all original report / pickle outputs preserved
=============================================================================
"""

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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/dps/attendance/pratham1")
DATASET_DIR = BASE_DIR / "dataset"
ARCFACE_HEF = BASE_DIR / "models1" / "arcface_r50.hef"
OUTPUT_FILE = BASE_DIR / "encodings.pkl"
REPORT_FILE = BASE_DIR / "encoding_report.txt"

# ── Constants ─────────────────────────────────────────────────────────────────
ARCFACE_SIZE = 112
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ↓ tighter than v1 (was 1.5) — prunes more outlier crops
OUTLIER_STD_FACTOR = 1.2

# ↓ lower than v1 (was 0.65) — warns earlier about potentially confused pairs
INTER_CLASS_WARN_THRESHOLD = 0.55

# Minimum valid embeddings to keep outlier filtering active.
# Below this the dataset is too small to estimate a reliable centroid,
# so we keep everything.
MIN_GALLERY_FOR_FILTERING = 5


# ============================================================================
# HAILO MODEL WRAPPER
# ============================================================================
class HailoModel:
    def __init__(self, hef_path: Path, format_type: FormatType, device: VDevice) -> None:
        if not hef_path.exists():
            sys.exit(f"[ERROR] HEF not found: {hef_path}")

        self.device = device
        self.hef    = HEF(str(hef_path))
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group  = self.device.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()

        in_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=format_type
        )
        out_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        in_info           = self.hef.get_input_vstream_infos()[0]
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


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================
def extract_embedding(arcface: HailoModel, crop_bgr: np.ndarray) -> np.ndarray:
    """
    Resize crop to ARCFACE_SIZE × ARCFACE_SIZE, run ArcFace, return unit vector.

    Improvement: if the crop is larger than ARCFACE_SIZE (true for 224-px crops
    saved by the updated dataset script), INTER_AREA downscaling is used, which
    is the highest-quality filter for shrinking — reduces aliasing and preserves
    fine facial texture better than the INTER_LINEAR default.
    """
    h, w = crop_bgr.shape[:2]
    if h != ARCFACE_SIZE or w != ARCFACE_SIZE:
        # INTER_AREA for shrinking (224→112), INTER_CUBIC for enlarging
        interp   = cv2.INTER_AREA if (h > ARCFACE_SIZE or w > ARCFACE_SIZE) \
                   else cv2.INTER_CUBIC
        crop_bgr = cv2.resize(crop_bgr, (ARCFACE_SIZE, ARCFACE_SIZE),
                              interpolation=interp)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    outputs = arcface.run(rgb)

    feats = outputs.get("fc1") or next(iter(outputs.values()))
    feats = np.squeeze(np.asarray(feats, dtype=np.float32))

    # Explicit L2 normalisation → unit vector (cosine sim == dot product)
    norm = np.linalg.norm(feats)
    if norm > 1e-6:
        feats = feats / norm
    return feats


# ============================================================================
# GALLERY BUILDER — outlier filtering + multi-embedding retention
# ============================================================================
def build_student_encoding(
    name: str,
    embeddings: list[np.ndarray],
    status_counts: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """
    Filters outliers from the embedding pool, then retains ALL valid individual
    embeddings (not a single averaged centroid).  A richer gallery means the
    recogniser can match frontal, left-profile, right-profile, etc.

    Guard: if the pool is too small (<MIN_GALLERY_FOR_FILTERING) outlier
    filtering is skipped — the centroid estimate would be unreliable anyway.
    """
    arr = np.array(embeddings, dtype=np.float32)  # (N, 512)

    # Compute centroid of the full pool (already unit vectors, so just mean+renorm)
    centroid = np.mean(arr, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-6:
        centroid = centroid / centroid_norm

    # Cosine similarities of each image vs centroid
    # Since arr rows and centroid are unit vectors: sim = dot(row, centroid)
    similarities = arr @ centroid             # shape (N,)
    mean_sim = float(np.mean(similarities))
    std_sim  = float(np.std(similarities))

    n_initial = len(embeddings)

    if n_initial >= MIN_GALLERY_FOR_FILTERING:
        thresh    = mean_sim - OUTLIER_STD_FACTOR * std_sim
        valid_idx = np.where(similarities >= thresh)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(n_initial)  # safety: keep all
    else:
        valid_idx = np.arange(n_initial)
        print(f"  [WARN] {name}: only {n_initial} embeddings — skipping outlier filter")

    filtered = arr[valid_idx]

    report = [
        f"=== {name} ===",
        f"  Total processed files      : {status_counts['processed']}",
        f"  Failed reads / errors      : {status_counts['failed']}",
        f"  Centroid similarity mean   : {mean_sim:.4f},  std={std_sim:.4f}",
        f"  Outlier threshold          : {mean_sim - OUTLIER_STD_FACTOR * std_sim:.4f}",
        f"  Outliers dropped           : {n_initial - len(valid_idx)}",
        f"  Final saved embeddings     : {len(filtered)}  "
        f"(shape {filtered.shape})",
        "",
    ]
    return filtered, report


# ============================================================================
# INTER-CLASS SIMILARITY ANALYSIS
# ============================================================================
def print_inter_class_similarity(database: dict[str, np.ndarray]) -> None:
    """
    Reports the *maximum* pairwise similarity between any embedding of person A
    and any embedding of person B.  A high value means at least one image of A
    looks very similar to at least one image of B — a likely confusion point.

    Threshold lowered to 0.55 (was 0.65) to surface potential issues earlier.
    """
    names = list(database.keys())
    if len(names) < 2:
        return

    print("\n[ANALYSIS] Cross-Class Max Similarity Profile:")
    warnings = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # (N_i, 512) @ (512, N_j) → (N_i, N_j) pairwise cosine similarities
            sim_matrix = database[names[i]] @ database[names[j]].T
            max_sim    = float(np.max(sim_matrix))
            mean_sim   = float(np.mean(sim_matrix))

            flag = ""
            if max_sim > INTER_CLASS_WARN_THRESHOLD:
                flag = "  ⚠ WARNING: potential confusion"
                warnings += 1

            print(
                f"   {names[i]} <-> {names[j]}:  "
                f"max={max_sim:.4f}  mean={mean_sim:.4f}{flag}"
            )

    if warnings == 0:
        print("   [OK] All class pairs are well-separated.")
    else:
        print(f"\n   [{warnings} pair(s) flagged] — "
              "Consider re-enrolling the flagged students with more pose variety.")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    if not DATASET_DIR.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {DATASET_DIR}")

    print("[HAILO] Initialising shared hardware device context…")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    shared_device = VDevice(params)

    arcface = HailoModel(ARCFACE_HEF, FormatType.UINT8, shared_device)

    database:    dict[str, np.ndarray] = {}
    full_report: list[str]             = []

    try:
        student_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())

        for sdir in student_dirs:
            student_name = sdir.name
            img_paths    = [p for p in sdir.iterdir()
                            if p.suffix.lower() in IMAGE_EXTS]

            if not img_paths:
                print(f"[WARN] No valid images in: {student_name}  — skipping.")
                continue

            print(f"\n[PROCESSING] {student_name}  ({len(img_paths)} images)")
            status_counts     = {"processed": 0, "failed": 0}
            image_embeddings: list[np.ndarray] = []

            for ipath in img_paths:
                frame = cv2.imread(str(ipath))
                if frame is None:
                    status_counts["failed"] += 1
                    continue

                emb = extract_embedding(arcface, frame)
                image_embeddings.append(emb)
                status_counts["processed"] += 1

            if not image_embeddings:
                print(f"[ERROR] All images failed for: {student_name}")
                continue

            embeddings_array, report_lines = build_student_encoding(
                student_name, image_embeddings, status_counts
            )
            database[student_name] = embeddings_array
            full_report.extend(report_lines)

            print(f"   → kept {len(embeddings_array)} / {len(image_embeddings)} embeddings")

    finally:
        arcface.close()
        try:
            shared_device.release()
            print("[HAILO] VDevice context closed cleanly.")
        except Exception:
            pass

    if not database:
        sys.exit("[ERROR] Zero encodings built successfully.")

    # ── Persist ──────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("wb") as fh:
        pickle.dump(database, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with REPORT_FILE.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(full_report))

    print(f"\n[SAVED] {len(database)} student(s) registered → {OUTPUT_FILE}")
    print(f"[REPORT] Written to {REPORT_FILE}")

    print_inter_class_similarity(database)


if __name__ == "__main__":
    main()
