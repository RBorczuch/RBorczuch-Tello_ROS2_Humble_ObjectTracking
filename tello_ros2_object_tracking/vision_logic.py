# tello_ros2_object_tracking/vision_logic.py

import cv2
import math
import time
import threading
from threading import Lock
import numpy as np

# --- Constants ---
RESIZED_WIDTH, RESIZED_HEIGHT = 960, 720
DEFAULT_ROI_SIZE, MIN_ROI_SIZE, DEFAULT_MAX_ROI_SIZE = 50, 25, 300
ROI_ADJUST_STEP = 10
MIN_TRACK_DURATION, MIN_TRACK_SCORE = 0.5, 0.30
FONT_SCALE, FONT_THICKNESS, LINE_SPACING = 0.6, 1, 25
STATUS_TEXT_POS, CONTROL_MODE_TEXT_POS, RECORDING_TEXT_POS = (10, 30), (10, 140), (10, 200)
COLOR_WHITE, COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_BLUE = (255, 255, 255), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)
REID_INTERVAL_FRAMES, REID_FAILURE_FRAME_LIMIT = 5, 240
SIFT_UPDATE_SCORE_THRESHOLD, SIFT_MATCH_RATIO, SIFT_MIN_GOOD_MATCHES = 70, 0.75, 10

class TrackingData:
    """Thread-safe class to store shared tracking and control state."""
    def __init__(self):
        self.lock = Lock()
        self.status = "Lost"
        self.dx, self.dy = 0, 0
        self.distance, self.angle, self.score = 0.0, 0.0, 0.0
        self.roi_height = 0
        self.control_mode, self.forward_enabled = "Manual", False

class VitTrack:
    """A wrapper for the OpenCV TrackerVit."""
    def __init__(self, model_path: str):
        params = cv2.TrackerVit_Params()
        params.net = model_path
        self.model = cv2.TrackerVit_create(params)
    def init(self, image, roi): self.model.init(image, roi)
    def infer(self, image):
        found, bbox = self.model.update(image)
        return found, bbox, self.model.getTrackingScore()

class VideoProcessingLogic:
    """Handles all image processing, tracking, re-identification, and drawing operations."""
    def __init__(self, model_path: str):
        self.tracking_data = TrackingData()
        self.model_path = model_path
        self.tracker, self.tracking_enabled, self.tracking_start_time = None, False, None
        self.mouse_x, self.mouse_y = 0, 0
        self.roi_size = DEFAULT_ROI_SIZE
        self.initialization_request, self.initialization_lock = None, Lock()
        self.sift_template = None
        self.frame_number, self.reid_fail_count = 0, 0
        self.reid_thread_running, self.reid_lock = False, Lock()

    def set_mouse_callback(self):
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = (x - self.roi_size // 2, y - self.roi_size // 2, self.roi_size, self.roi_size)
            with self.initialization_lock: self.initialization_request = bbox
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled, self.tracker, self.sift_template = False, None, None
            with self.tracking_data.lock: self.tracking_data.status = "Lost"

    def adjust_roi_size(self, delta: int):
        self.roi_size = max(MIN_ROI_SIZE, min(self.roi_size + delta, DEFAULT_MAX_ROI_SIZE))

    def draw_text(self, img, lines: list[str], start_x: int, start_y: int):
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * LINE_SPACING)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def _run_reid(self, frame):
        sift = cv2.SIFT_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
        success = False
        if des_frame is not None and self.sift_template and self.sift_template[1] is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.sift_template[1], des_frame, k=2)
            good = [m for m, n in matches if m.distance < SIFT_MATCH_RATIO * n.distance]
            if len(good) > SIFT_MIN_GOOD_MATCHES:
                src_pts = np.float32([self.sift_template[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    x, y, w, h = self.sift_template[2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(corners, M)
                    new_bbox = cv2.boundingRect(dst)
                    with self.reid_lock:
                        self.tracker = VitTrack(self.model_path)
                        self.tracker.init(frame, new_bbox)
                        self.tracking_enabled, self.tracking_start_time = True, time.time()
                    success = True
        self.reid_thread_running = False
        if success: self.reid_fail_count = 0

    def process_frame(self, frame, original_frame):
        if self.tracking_enabled and self.tracker is not None:
            found, bbox, score = self.tracker.infer(frame)
            if found:
                x, y, w, h = map(int, bbox)
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                cx, cy = x + w // 2, y + h // 2
                
                # --- OSTATECZNA KOREKTA LOGIKI BŁĘDU ---
                dx = center_x - cx
                dy = cy - center_y # To jest poprawna wersja z kodu referencyjnego

                dist = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))
                
                with self.tracking_data.lock:
                    self.tracking_data.status, self.tracking_data.dx, self.tracking_data.dy = "Tracking", dx, dy
                    self.tracking_data.distance, self.tracking_data.angle = dist, angle
                    self.tracking_data.score, self.tracking_data.roi_height = score, h
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
                cv2.line(frame, (center_x, center_y), (cx, cy), COLOR_YELLOW, 1)

                if time.time() - self.tracking_start_time > MIN_TRACK_DURATION and score < MIN_TRACK_SCORE:
                    found = False
                
                if found and score > SIFT_UPDATE_SCORE_THRESHOLD:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size != 0:
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None)
                        self.sift_template = (kp, des, bbox)
            if not found:
                self.tracking_enabled, self.tracker = False, None
                with self.tracking_data.lock: self.tracking_data.status = "Lost"

        elif not self.tracking_enabled and self.sift_template is not None:
            self.frame_number += 1
            if (self.frame_number % REID_INTERVAL_FRAMES == 0) and not self.reid_thread_running:
                self.reid_thread_running = True
                threading.Thread(target=self._run_reid, args=(original_frame.copy(),), daemon=True).start()
                self.reid_fail_count += REID_INTERVAL_FRAMES
            
            status_text = "Status: Re-identification"
            if self.reid_fail_count >= REID_FAILURE_FRAME_LIMIT:
                status_text = "Status: Lost (Re-ID failed)"
                self.sift_template = None
            with self.tracking_data.lock: self.tracking_data.status = "Re-identification"
            self.draw_text(frame, [status_text], STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])

        if self.tracking_enabled:
            with self.tracking_data.lock:
                td = self.tracking_data
                info = [f"Status: {td.status}", f"Score: {td.score:.2f}", f"dx: {td.dx}px, dy: {td.dy}px"]
                self.draw_text(frame, info, STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])
        elif not self.sift_template:
            self.draw_text(frame, ["Status: Lost"], STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])

        with self.tracking_data.lock:
            mode_text = f"Mode: {self.tracking_data.control_mode}"
            if self.tracking_data.control_mode == "Autonomous":
                mode_text += " (Forward)" if self.tracking_data.forward_enabled else " (Follow)"
        self.draw_text(frame, [mode_text], CONTROL_MODE_TEXT_POS[0], CONTROL_MODE_TEXT_POS[1])
        
        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 3, COLOR_RED, -1)
        cv2.rectangle(frame, (self.mouse_x - self.roi_size//2, self.mouse_y - self.roi_size//2),
                      (self.mouse_x + self.roi_size//2, self.mouse_y + self.roi_size//2), COLOR_BLUE, 1)
        return frame