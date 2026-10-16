#!/usr/bin/env python3
"""Second Mind — Live OpenCV Vision Tracker & Heads-Up Display (HUD).

Real-time laptop camera tracking with OpenCV motion proposals, contour saliency,
visual similarity matching against registered objects, and interactive live registration.
"""

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import settings
from backend.constants import Confidence
from backend.main import tts_wrapper, visual_service
from backend.visual.matcher import MatchStatus
from backend.visual.proposals import BoundingBox


class OpenCVLiveTrackerHUD:
    """Manages real-time OpenCV camera capture, HUD rendering, and object tracking."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 960,
        height: int = 540,
        detect_interval: int = 3,
    ) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.detect_interval = detect_interval

        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.show_proposals: bool = True
        self.last_fps: float = 0.0
        self.fps_timer: float = time.time()
        self.fps_frames: int = 0

        # Detection cache
        self.last_result = None
        self.last_status_msg: str = "Ready. Point camera at registered objects."
        self.last_status_color: Tuple[int, int, int] = (200, 200, 200)
        self.active_proposals: List[BoundingBox] = []

    def start_camera(self) -> bool:
        """Initialize camera stream and set resolution."""
        print(f"[OpenCV] Initializing camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[OpenCV] Camera active. Target resolution: {self.width}x{self.height}")
        return True

    def stop_camera(self) -> None:
        """Release camera and close OpenCV windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[OpenCV] Camera released and windows closed.")

    def _draw_hud_header(self, canvas: np.ndarray, reg_count: int) -> None:
        """Draw top banner with status and metrics."""
        h, w = canvas.shape[:2]
        header_h = 48

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 24, 32), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, header_h), (w, header_h), (37, 99, 235), 2)

        # Title
        cv2.putText(
            canvas,
            "SECOND MIND // OPENCV REAL-TIME HUD",
            (16, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Registry info
        reg_text = f"Registry: {reg_count} Objects"
        cv2.putText(
            canvas,
            reg_text,
            (w - 360, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (147, 197, 253),
            1,
            cv2.LINE_AA,
        )

        # FPS
        fps_text = f"FPS: {self.last_fps:.1f}"
        cv2.putText(
            canvas,
            fps_text,
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )

    def _draw_hud_footer(self, canvas: np.ndarray) -> None:
        """Draw bottom bar with keyboard shortcuts and current status."""
        h, w = canvas.shape[:2]
        footer_h = 56

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - footer_h), (w, h), (18, 24, 32), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (51, 65, 85), 1)

        # Status text
        cv2.putText(
            canvas,
            f"STATUS: {self.last_status_msg}",
            (16, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.last_status_color,
            1,
            cv2.LINE_AA,
        )

        # Shortcuts
        shortcuts = "[R] Register  |  [S] Snap & Speak  |  [M] Proposals  |  [Q] Quit"
        cv2.putText(
            canvas,
            shortcuts,
            (16, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

    def _draw_bounding_box_corners(
        self,
        img: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int] = (34, 197, 94),
        thickness: int = 2,
        corner_len: int = 18,
    ) -> None:
        """Draw tactical corner bracket styling around bounding box."""
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Main bounding rectangle (thin)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        # Top-Left corner
        cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(img, (x, y), (x, y + corner_len), color, thickness)

        # Top-Right corner
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)

        # Bottom-Left corner
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)

        # Bottom-Right corner
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    def register_object_interactive(self, current_frame: np.ndarray) -> None:
        """Capture object from live camera and register into OpenCV Object Registry."""
        h, w = current_frame.shape[:2]
        print("\n" + "=" * 62)
        print("   OPENCV INTERACTIVE OBJECT REGISTRATION")
        print("=" * 62)

        name = input("Enter object name (e.g. Reading Glasses, Keys, Pill Box): ").strip()
        if not name:
            print("[!] Registration cancelled: Name cannot be empty.")
            return

        category = input("Enter category [personal/glasses/keys/medication/wallet] (default personal): ").strip() or "personal"
        description = input("Enter brief description (optional): ").strip()

        # Crop center region or full frame
        cw, ch = int(w * 0.6), int(h * 0.6)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        crop = current_frame[cy : cy + ch, cx : cx + cw]

        _, buf = cv2.imencode(".jpg", crop)
        print(f"\n[OpenCV] Computing feature descriptors & registering '{name}'...")
        obj = visual_service.register_object(
            name=name,
            category=category,
            description=description,
            image_bytes=buf.tobytes(),
            filename=f"{name.lower().replace(' ', '_')}.jpg",
        )

        msg = f"Successfully registered '{name}' in OpenCV Object Registry!"
        print(f"[✓] {msg}")
        self.last_status_msg = f"Registered: {name} ({category})"
        self.last_status_color = (34, 197, 94)

        tts_wrapper.speak(f"Successfully registered {name} in your OpenCV Object Registry.")

    def snap_and_speak(self, current_frame: np.ndarray) -> None:
        """Immediately scan current camera frame and speak match via laptop speaker."""
        print("\n[OpenCV] Running high-precision scene scan...")
        self.last_status_msg = "Scanning scene..."
        self.last_status_color = (234, 179, 8)

        res = visual_service.search_frame(frame=current_frame, save_observation=True)
        if res.is_confident and res.best_match:
            bm = res.best_match
            print(f"[✓] Detected '{bm.object_name}' ({bm.similarity_score * 100:.1f}%) on {bm.spatial_location}")
            self.last_status_msg = f"Detected: {bm.object_name} ({bm.similarity_score * 100:.0f}%) on {bm.spatial_location}"
            self.last_status_color = (34, 197, 94)
            msg = f"I detected your {bm.object_name} on the {bm.spatial_location}."
            tts_wrapper.speak(msg)
        else:
            print("[i] No registered object detected in current view.")
            self.last_status_msg = "Scan complete: No registered objects detected."
            self.last_status_color = (148, 163, 184)
            tts_wrapper.speak("I scanned the room, but did not detect any registered objects.")

    def run(self) -> None:
        """Main loop: capture frame, perform tracking, render HUD, handle key input."""
        if not self.start_camera():
            return

        cv2.namedWindow("Second Mind - OpenCV Live Tracker", cv2.WINDOW_AUTOSIZE)
        print("\n" + "=" * 65)
        print("  SECOND MIND — OPENCV LIVE TRACKER & HUD ACTIVE")
        print("=" * 65)
        print("  Controls:")
        print("    [R] - Register New Object from Live Frame")
        print("    [S] - Snap Scene & Speak Result out loud")
        print("    [M] - Toggle Motion / Saliency Region Proposals")
        print("    [Q] / [ESC] - Quit Window")
        print("=" * 65 + "\n")

        prev_frame: Optional[np.ndarray] = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[!] Failed to grab frame from camera.")
                    break

                # Frame resize for consistent processing
                frame = cv2.resize(frame, (self.width, self.height))
                display_frame = frame.copy()
                self.frame_count += 1

                # Calculate live FPS
                self.fps_frames += 1
                now = time.time()
                elapsed = now - self.fps_timer
                if elapsed >= 1.0:
                    self.last_fps = self.fps_frames / elapsed
                    self.fps_frames = 0
                    self.fps_timer = now

                # Run object search every N frames
                if self.frame_count % self.detect_interval == 0:
                    try:
                        self.last_result = visual_service.search_frame(
                            frame=frame,
                            prev_frame=prev_frame,
                            save_observation=True,
                            max_proposals=10,
                        )
                        # Extract proposals for rendering
                        self.active_proposals = visual_service.proposal_generator.generate_proposals(
                            frame=frame, prev_frame=prev_frame, max_proposals=6
                        )
                    except Exception as e:
                        self.last_status_msg = f"Detection Error: {str(e)[:40]}"
                        self.last_status_color = (239, 68, 68)

                prev_frame = frame.copy()

                # 1. Render active candidate proposals (soft cyan / yellow boxes)
                if self.show_proposals and self.active_proposals:
                    for prop in self.active_proposals:
                        bx, by, bw, bh = prop.x, prop.y, prop.width, prop.height
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (220, 200, 40), 1, cv2.LINE_AA)
                        cv2.putText(
                            display_frame,
                            "proposal",
                            (bx, max(15, by - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (220, 200, 40),
                            1,
                            cv2.LINE_AA,
                        )

                # 2. Render confident detected object
                if self.last_result and self.last_result.is_confident and self.last_result.best_match:
                    bm = self.last_result.best_match
                    bbox = bm.bounding_box
                    self._draw_bounding_box_corners(display_frame, bbox, color=(34, 197, 94), thickness=3)

                    # Label badge
                    label = f"{bm.object_name.upper()}  [{bm.similarity_score * 100:.0f}%]"
                    loc_text = f"Location: {bm.spatial_location}"

                    # Background badge rectangle
                    tag_x, tag_y = bbox.x, max(30, bbox.y - 10)
                    cv2.rectangle(display_frame, (tag_x, tag_y - 24), (tag_x + 220, tag_y + 14), (16, 24, 32), -1)
                    cv2.rectangle(display_frame, (tag_x, tag_y - 24), (tag_x + 220, tag_y + 14), (34, 197, 94), 1)

                    cv2.putText(
                        display_frame,
                        label,
                        (tag_x + 8, tag_y - 8),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.48,
                        (34, 197, 94),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_frame,
                        loc_text,
                        (tag_x + 8, tag_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    self.last_status_msg = f"FOUND: {bm.object_name} on {bm.spatial_location} ({bm.similarity_score * 100:.0f}%)"
                    self.last_status_color = (34, 197, 94)

                # 3. Draw HUD Header and Footer
                reg_count = len(visual_service.object_store.list_objects())
                self._draw_hud_header(display_frame, reg_count)
                self._draw_hud_footer(display_frame)

                # 4. Display Window
                cv2.imshow("Second Mind - OpenCV Live Tracker", display_frame)

                # Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 27 is ESC
                    print("\n[OpenCV] Exit requested by user.")
                    break
                elif key in [ord("r"), ord("R")]:
                    self.register_object_interactive(frame)
                elif key in [ord("s"), ord("S")]:
                    self.snap_and_speak(frame)
                elif key in [ord("m"), ord("M")]:
                    self.show_proposals = not self.show_proposals
                    state_str = "ENABLED" if self.show_proposals else "DISABLED"
                    print(f"[OpenCV] Candidate proposals visualization: {state_str}")
                    self.last_status_msg = f"Proposals display {state_str}"
                    self.last_status_color = (147, 197, 253)

        finally:
            self.stop_camera()


def main():
    parser = argparse.ArgumentParser(description="Second Mind OpenCV Live Vision Tracker HUD")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Window width (default: 960)")
    parser.add_argument("--height", type=int, default=540, help="Window height (default: 540)")
    parser.add_argument("--interval", type=int, default=3, help="Detection frame interval (default: 3)")
    args = parser.parse_args()

    tracker = OpenCVLiveTrackerHUD(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        detect_interval=args.interval,
    )
    tracker.run()


if __name__ == "__main__":
    main()
