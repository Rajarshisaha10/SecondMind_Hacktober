"""Repeatable evaluation benchmark for SecondMind Reference-Image Embeddings.

Evaluates precision, recall, false-positive rate, missed detections, and inference latency
across controlled test scenarios with rotations, scale changes, lighting variations, and distractors.
"""

import os
import sys
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import psutil

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.stores.memory_store import MemoryStore
from backend.stores.object_store import ObjectStore
from backend.visual.encoder import get_visual_encoder
from backend.visual.matcher import MatchStatus, SimilarityMatcher
from backend.visual.proposals import CandidateRegionProposalGenerator
from backend.visual.service import VisualRetrievalService


def create_glasses_asset(color=(30, 30, 30)) -> np.ndarray:
    """Generate synthetic pair of reading glasses."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Left lens and frame
    cv2.rectangle(img, (15, 35), (45, 65), color, 4)
    cv2.rectangle(img, (18, 38), (42, 62), (200, 220, 240), -1)
    # Right lens and frame
    cv2.rectangle(img, (55, 35), (85, 65), color, 4)
    cv2.rectangle(img, (58, 38), (82, 62), (200, 220, 240), -1)
    # Bridge and temples
    cv2.line(img, (45, 50), (55, 50), color, 3)
    cv2.line(img, (15, 45), (5, 30), color, 2)
    cv2.line(img, (85, 45), (95, 30), color, 2)
    return img


def create_keys_asset() -> np.ndarray:
    """Generate synthetic house key with brass head and green fob."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Key ring
    cv2.circle(img, (35, 40), 18, (180, 180, 180), 3)
    # Green fob
    cv2.rectangle(img, (15, 20), (35, 45), (30, 180, 30), -1)
    # Brass key blade and teeth
    cv2.rectangle(img, (35, 38), (85, 44), (50, 180, 220), -1)
    cv2.rectangle(img, (60, 44), (66, 52), (50, 180, 220), -1)
    cv2.rectangle(img, (72, 44), (78, 55), (50, 180, 220), -1)
    return img


def create_pill_bottle_asset() -> np.ndarray:
    """Generate synthetic prescription pill bottle."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Orange bottle body
    cv2.rectangle(img, (30, 30), (70, 85), (20, 120, 230), -1)
    # White child-resistant cap
    cv2.rectangle(img, (26, 18), (74, 30), (240, 240, 240), -1)
    # White prescription label
    cv2.rectangle(img, (32, 42), (68, 70), (255, 255, 255), -1)
    cv2.line(img, (36, 50), (64, 50), (0, 0, 0), 1)
    cv2.line(img, (36, 58), (56, 58), (0, 0, 0), 1)
    return img


def create_wallet_asset() -> np.ndarray:
    """Generate synthetic leather wallet."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Brown leather body
    cv2.rectangle(img, (20, 25), (80, 75), (42, 68, 115), -1)
    # Stitching detail
    cv2.rectangle(img, (24, 29), (76, 71), (60, 90, 140), 1)
    # Metal emblem
    cv2.circle(img, (70, 65), 4, (120, 200, 220), -1)
    return img


def create_distractor_mug() -> np.ndarray:
    """Generate distractor coffee mug."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 80), (180, 60, 60), -1)
    cv2.ellipse(img, (72, 55), (10, 16), 0, -90, 90, (180, 60, 60), 3)
    return img


def apply_transformation(asset: np.ndarray, angle: float = 0, scale: float = 1.0, brightness: int = 0) -> np.ndarray:
    """Apply affine rotation, scaling, and lighting shift to test robust matching."""
    h, w = asset.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    transformed = cv2.warpAffine(asset, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if brightness != 0:
        transformed = np.clip(transformed.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return transformed


def place_in_scene(
    scene_w: int, scene_h: int, asset: np.ndarray, pos_x: int, pos_y: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Place an object asset into a simulated room camera frame."""
    frame = np.full((scene_h, scene_w, 3), 40, dtype=np.uint8)
    # Add room background surface (table top)
    cv2.rectangle(frame, (40, 60), (scene_w - 40, scene_h - 40), (80, 70, 65), -1)

    ah, aw = asset.shape[:2]
    x2, y2 = min(scene_w, pos_x + aw), min(scene_h, pos_y + ah)
    crop_w, crop_h = x2 - pos_x, y2 - pos_y

    mask = (asset[:crop_h, :crop_w] > 0).any(axis=2)
    frame_region = frame[pos_y:y2, pos_x:x2]
    frame_region[mask] = asset[:crop_h, :crop_w][mask]
    frame[pos_y:y2, pos_x:x2] = frame_region

    return frame, (pos_x, pos_y, crop_w, crop_h)


def run_evaluation() -> Dict[str, Any]:
    """Execute complete automated benchmark evaluation across 20 labeled trials."""
    print("=" * 70)
    print("SecondMind Reference-Image Retrieval — Evaluation Benchmark")
    print("=" * 70)

    # Initialize isolated environment
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "eval.db"
    obj_store = ObjectStore(db_path=db_path)
    mem_store = MemoryStore(db_path=db_path)
    encoder = get_visual_encoder()
    matcher = SimilarityMatcher(
        strong_threshold=0.70,
        weak_threshold=0.45,
        ambiguity_margin=0.05,
        enable_local_verification=True,
    )
    service = VisualRetrievalService(
        object_store=obj_store,
        memory_store=mem_store,
        encoder=encoder,
        matcher=matcher,
    )
    service.ref_dir = Path(tmp_dir) / "refs"
    service.ref_dir.mkdir(parents=True, exist_ok=True)

    # 1. Register Reference Objects
    print("\n[Step 1] Registering reference objects...")
    reg_items = [
        ("reading glasses", "glasses", create_glasses_asset()),
        ("house keys", "keys", create_keys_asset()),
        ("medication bottle", "medication", create_pill_bottle_asset()),
        ("leather wallet", "personal", create_wallet_asset()),
    ]
    registered_ids = {}
    for name, cat, asset in reg_items:
        _, buf = cv2.imencode(".jpg", asset)
        obj = service.register_object(name=name, category=cat, image_bytes=buf.tobytes(), filename=f"{cat}.jpg")
        registered_ids[name] = obj["id"]
        # Add slight variation as second reference image
        var_asset = apply_transformation(asset, angle=5, brightness=10)
        _, vbuf = cv2.imencode(".jpg", var_asset)
        service.add_reference_to_object(obj["id"], vbuf.tobytes(), f"{cat}_v2.jpg")
        print(f"  ✓ Registered '{name}' ({cat}) with 2 reference embeddings")

    # 2. Build Labeled Evaluation Trials
    print("\n[Step 2] Building labeled test dataset (12 Positive, 8 Negative trials)...")
    trials = []

    # Positive Trials (Target present under realistic conditions)
    trials.append({"name": "Glasses Baseline (Center)", "target": "reading glasses", "expected": True, "asset": create_glasses_asset(), "pos": (260, 180)})
    trials.append({"name": "Glasses Rotated +15deg", "target": "reading glasses", "expected": True, "asset": apply_transformation(create_glasses_asset(), angle=15), "pos": (120, 100)})
    trials.append({"name": "Glasses Dim Lighting (-25)", "target": "reading glasses", "expected": True, "asset": apply_transformation(create_glasses_asset(), brightness=-25), "pos": (400, 260)})

    trials.append({"name": "Keys Baseline (Right)", "target": "house keys", "expected": True, "asset": create_keys_asset(), "pos": (440, 200)})
    trials.append({"name": "Keys Rotated -15deg", "target": "house keys", "expected": True, "asset": apply_transformation(create_keys_asset(), angle=-15), "pos": (160, 280)})
    trials.append({"name": "Keys Scaled Down 0.85x", "target": "house keys", "expected": True, "asset": apply_transformation(create_keys_asset(), scale=0.85), "pos": (300, 140)})

    trials.append({"name": "Med Bottle Baseline (Left)", "target": "medication bottle", "expected": True, "asset": create_pill_bottle_asset(), "pos": (100, 180)})
    trials.append({"name": "Med Bottle Bright (+25)", "target": "medication bottle", "expected": True, "asset": apply_transformation(create_pill_bottle_asset(), brightness=25), "pos": (320, 220)})
    trials.append({"name": "Med Bottle Rotated +10deg", "target": "medication bottle", "expected": True, "asset": apply_transformation(create_pill_bottle_asset(), angle=10), "pos": (200, 120)})

    trials.append({"name": "Wallet Baseline (Center)", "target": "leather wallet", "expected": True, "asset": create_wallet_asset(), "pos": (270, 210)})
    trials.append({"name": "Wallet Scaled Up 1.15x", "target": "leather wallet", "expected": True, "asset": apply_transformation(create_wallet_asset(), scale=1.15), "pos": (380, 160)})
    trials.append({"name": "Wallet Rotated -10deg", "target": "leather wallet", "expected": True, "asset": apply_transformation(create_wallet_asset(), angle=-10), "pos": (140, 260)})

    # Negative Trials (Target not present / distractors)
    trials.append({"name": "Empty Room / Clean Table", "target": "reading glasses", "expected": False, "asset": None, "pos": None})
    trials.append({"name": "Distractor Coffee Mug", "target": "reading glasses", "expected": False, "asset": create_distractor_mug(), "pos": (260, 180)})
    trials.append({"name": "Distractor Coffee Mug vs Keys", "target": "house keys", "expected": False, "asset": create_distractor_mug(), "pos": (300, 200)})
    trials.append({"name": "Distractor Coffee Mug vs Med", "target": "medication bottle", "expected": False, "asset": create_distractor_mug(), "pos": (180, 160)})
    trials.append({"name": "Unregistered Dark Box", "target": "leather wallet", "expected": False, "asset": np.full((80, 80, 3), 150, dtype=np.uint8), "pos": (320, 220)})
    trials.append({"name": "Negative: Blank Dim Surface", "target": "house keys", "expected": False, "asset": None, "pos": None})
    trials.append({"name": "Negative: Blue Circle Pattern", "target": "medication bottle", "expected": False, "asset": cv2.circle(np.zeros((90, 90, 3), dtype=np.uint8), (45, 45), 35, (255, 100, 50), -1), "pos": (220, 180)})
    trials.append({"name": "Negative: Empty Scene vs Wallet", "target": "leather wallet", "expected": False, "asset": None, "pos": None})

    # 3. Execute Benchmark Trials
    print("\n[Step 3] Running inference and measuring accuracy and latencies...")
    tp, fp, tn, fn = 0, 0, 0, 0
    latencies: List[float] = []

    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    trial_records = []
    for idx, t in enumerate(trials, 1):
        if t["asset"] is not None:
            frame, box = place_in_scene(640, 480, t["asset"], t["pos"][0], t["pos"][1])
        else:
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (40, 60), (600, 440), (80, 70, 65), -1)

        t0 = time.perf_counter()
        target_oid = registered_ids.get(t["target"])
        result = service.search_frame(
            frame=frame,
            target_object_id=target_oid,
            save_observation=False,
            max_proposals=16,
        )
        total_lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_lat)

        detected = result.is_confident and (result.best_match is not None) and (result.best_match.object_name == t["target"])

        if t["expected"] and detected:
            tp += 1
            outcome = "TP (Hit)"
        elif t["expected"] and not detected:
            fn += 1
            outcome = "FN (Miss)"
        elif not t["expected"] and not detected:
            tn += 1
            outcome = "TN (Correct Rejection)"
        else:
            fp += 1
            outcome = "FP (False Alarm)"

        top_sim = result.top_similarity
        status_str = result.status.value
        trial_records.append((idx, t["name"], outcome, f"{top_sim:.3f}", status_str, f"{total_lat:.1f}ms"))

    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    # 4. Compute Statistical Metrics
    total_pos = tp + fn
    total_neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0
    specificity = tn / total_neg if total_neg > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0

    avg_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))

    # 5. Display Benchmark Summary Table
    print("\n" + "=" * 70)
    print(f"{'#':<3} {'Trial Scenario':<34} {'Result':<22} {'Sim':<7} {'Status':<14} {'Latency':<8}")
    print("-" * 70)
    for rec in trial_records:
        print(f"{rec[0]:<3} {rec[1]:<34} {rec[2]:<22} {rec[3]:<7} {rec[4]:<14} {rec[5]:<8}")
    print("=" * 70)

    print("\n--- METRIC SUMMARY ---")
    print(f"Total Trials:              {len(trials)} (12 Positive, 8 Negative)")
    print(f"True Positives (TP):       {tp}")
    print(f"True Negatives (TN):       {tn}")
    print(f"False Positives (FP):      {fp}")
    print(f"False Negatives (FN):      {fn}")
    print(f"Precision:                 {precision * 100:.1f}%")
    print(f"Recall (Sensitivity):      {recall * 100:.1f}%")
    print(f"Specificity:               {specificity * 100:.1f}%")
    print(f"F1-Score:                  {f1 * 100:.1f}%")
    print(f"False Positive Rate (FPR): {fpr * 100:.1f}%")
    print(f"Missed Detection Rate:     {fnr * 100:.1f}%")

    print("\n--- PERFORMANCE & HARDWARE PROFILING ---")
    print(f"Encoder Model:             {service.encoder.model_name} (dim={service.encoder.dimension})")
    print(f"Device:                    {getattr(service.encoder, 'device_str', 'CPU')}")
    print(f"Avg Latency per Frame:     {avg_lat:.2f} ms")
    print(f"P50 Median Latency:        {p50_lat:.2f} ms")
    print(f"P95 Latency:               {p95_lat:.2f} ms")
    print(f"Min / Max Latency:         {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Process Memory (RSS):      {mem_after_mb:.1f} MB (Delta: +{mem_after_mb - mem_before_mb:.2f} MB)")
    print("=" * 70)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "memory_mb": mem_after_mb,
    }


if __name__ == "__main__":
    run_evaluation()
