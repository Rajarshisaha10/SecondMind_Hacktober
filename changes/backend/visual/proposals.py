# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1: Visual Proposal Generator
# Relative Path: backend/visual/proposals.py
# Summary: Object proposal generation using edge/contour saliency, selective search or sliding window grid crops for localized instance retrieval in full camera frames.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Candidate object region proposal generator for visual localization.

Generates localized candidate bounding boxes using motion diffing, edge/saliency contouring,
and multi-scale spatial tiling with Non-Maximum Suppression (NMS).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box representation with spatial naming utilities."""

    x: int
    y: int
    width: int
    height: int
    score: float = 1.0
    source: str = "proposal"

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "score": round(float(self.score), 3),
            "source": self.source,
        }

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        inter_x1 = max(self.x, other.x)
        inter_y1 = max(self.y, other.y)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        self_area = self.width * self.height
        other_area = other.width * other.height
        union_area = self_area + other_area - inter_area
        return float(inter_area) / float(union_area) if union_area > 0 else 0.0

    def get_spatial_label(self, frame_w: int, frame_h: int) -> str:
        """Generate human-readable natural language spatial description of the region."""
        center_x = self.x + self.width / 2.0
        center_y = self.y + self.height / 2.0

        rel_x = center_x / max(frame_w, 1)
        rel_y = center_y / max(frame_h, 1)

        # Horizontal classification
        if rel_x < 0.35:
            h_pos = "left side"
        elif rel_x > 0.65:
            h_pos = "right side"
        else:
            h_pos = "center"

        # Vertical classification
        if rel_y < 0.35:
            v_pos = "upper"
        elif rel_y > 0.65:
            v_pos = "lower"
        else:
            v_pos = "middle"

        if h_pos == "center" and v_pos == "middle":
            return "center area"
        return f"{v_pos} {h_pos}"


class CandidateRegionProposalGenerator:
    """Generates candidate bounding boxes from incoming camera frames for localized search."""

    def __init__(
        self,
        min_box_dim: int = 40,
        max_box_area_ratio: float = 0.85,
        nms_iou_threshold: float = 0.55,
    ) -> None:
        self.min_box_dim = min_box_dim
        self.max_box_area_ratio = max_box_area_ratio
        self.nms_iou_threshold = nms_iou_threshold

    def generate_proposals(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        max_proposals: int = 16,
    ) -> List[Tuple[BoundingBox, np.ndarray]]:
        """Generate candidate bounding boxes and extract corresponding image crops.

        Args:
            frame: Current BGR camera frame.
            prev_frame: Optional previous BGR frame for motion difference proposals.
            max_proposals: Maximum number of candidate crops to return after NMS.

        Returns:
            List of (BoundingBox, cropped_bgr_image) tuples.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        frame_area = float(h * w)
        raw_boxes: List[BoundingBox] = []

        # 1. Motion / Difference Contours (highest priority when motion occurs)
        if prev_frame is not None and prev_frame.shape == frame.shape:
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_curr, gray_prev)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Expand box slightly to include object boundary context
                pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
                bx = max(0, bx - pad_x)
                by = max(0, by - pad_y)
                bw = min(w - bx, bw + 2 * pad_x)
                bh = min(h - by, bh + 2 * pad_y)

                if bw >= self.min_box_dim and bh >= self.min_box_dim and (bw * bh) < (self.max_box_area_ratio * frame_area):
                    raw_boxes.append(BoundingBox(x=bx, y=by, width=bw, height=bh, score=0.9, source="motion"))

        # 2. Edge and Saliency Contours (static foreground object localization)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh) if bh > 0 else 1.0
            if (
                bw >= self.min_box_dim
                and bh >= self.min_box_dim
                and (bw * bh) < (self.max_box_area_ratio * frame_area)
                and 0.15 <= aspect <= 6.5
            ):
                raw_boxes.append(BoundingBox(x=bx, y=by, width=bw, height=bh, score=0.75, source="contour"))

        # 3. Multi-scale spatial tiling (ensures small/medium object coverage even in clutter)
        # Quadrants with 20% overlap
        half_w, half_h = int(w * 0.55), int(h * 0.55)
        step_x, step_y = int(w * 0.45), int(h * 0.45)
        for gy in range(0, h - step_y, step_y):
            for gx in range(0, w - step_x, step_x):
                qw = min(w - gx, half_w)
                qh = min(h - gy, half_h)
                raw_boxes.append(BoundingBox(x=gx, y=gy, width=qw, height=qh, score=0.5, source="spatial_quadrant"))

        # Central table/counter focus crop
        cw, ch = int(w * 0.6), int(h * 0.6)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        raw_boxes.append(BoundingBox(x=cx, y=cy, width=cw, height=ch, score=0.6, source="center_focus"))

        # 4. Non-Maximum Suppression (NMS)
        deduped_boxes = self._apply_nms(raw_boxes, max_proposals=max_proposals)

        # 5. Extract Crops with safety bounds checking
        crops: List[Tuple[BoundingBox, np.ndarray]] = []
        for box in deduped_boxes:
            crop = frame[box.y : box.y2, box.x : box.x2]
            if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                crops.append((box, crop))

        return crops

    def _apply_nms(self, boxes: List[BoundingBox], max_proposals: int) -> List[BoundingBox]:
        """Filter overlapping proposal boxes based on IoU threshold."""
        if not boxes:
            return []

        # Sort by score descending, then by area descending
        sorted_boxes = sorted(boxes, key=lambda b: (b.score, b.width * b.height), reverse=True)
        selected: List[BoundingBox] = []

        for candidate in sorted_boxes:
            keep = True
            for existing in selected:
                if candidate.iou(existing) > self.nms_iou_threshold:
                    keep = False
                    break
            if keep:
                selected.append(candidate)
                if len(selected) >= max_proposals:
                    break

        return selected
