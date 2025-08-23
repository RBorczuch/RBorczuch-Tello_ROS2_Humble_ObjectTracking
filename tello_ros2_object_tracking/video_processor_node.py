
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tello_ros2_object_tracking.msg import TrackedObject # Niestandardowy komunikat

# Poniższy kod jest adaptacją oryginalnych plików process_video.py i tracking_data.py
# zintegrowaną w jeden samodzielny węzeł ROS2.
import numpy as np
import math
import time
import threading
import os
from threading import Lock

# ==============================================================================
# SEKCJA 1: Kod zaadaptowany z oryginalnych plików (logika przetwarzania)
# ==============================================================================

# -----------------------------
# Globalne stałe
# -----------------------------
RESIZED_WIDTH = 480
RESIZED_HEIGHT = 360

DEFAULT_ROI_SIZE = 50
MIN_ROI_SIZE = 25
DEFAULT_MAX_ROI_SIZE = 200

REID_INTERVAL = 5
REID_FAILURE_LIMIT = 240

SIFT_UPDATE_SCORE_THRESHOLD = 70
SIFT_MATCH_RATIO = 0.75
SIFT_MIN_GOOD_MATCHES = 10

MIN_TRACK_DURATION = 0.5
MIN_TRACK_SCORE = 0.30

IMG_MARGIN = 10

FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_SPACING = 20

STATUS_TEXT_POS = (10, 20)
REID_TEXT_POS = (10, 60)
CONTROL_MODE_TEXT_POS = (10, 140)
RECORDING_TEXT_POS = (10, 180)

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

FPS_TEXT_OFFSET_Y = -10

class TrackingData:
    """
    Przechowuje współdzielone wartości śledzenia z blokadą wątku.
    W kontekście ROS2, jest to wewnętrzna klasa do zarządzania stanem.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Lost"
        self.dx = 0
        self.dy = 0
        self.distance = 0.0
        self.angle = 0.0
        self.score = 0.0
        self.control_mode = "Manual"  # Tryb jest teraz zarządzany przez węzeł kontrolera
        self.roi_height = 0
        self.forward_enabled = False

class VitTrack:
    """
    Wrapper dla OpenCV TrackerVit.
    """
    def __init__(self, model_path, backend_id=0, target_id=0):
        self.model_path = model_path
        self.backend_id = backend_id
        self.target_id = target_id

        self.params = cv2.TrackerVit_Params()
        self.params.net = self.model_path
        self.params.backend = self.backend_id
        self.params.target = self.target_id

        self.model = cv2.TrackerVit_create(self.params)

    def init(self, image, roi):
        self.model.init(image, roi)

    def infer(self, image):
        found, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return found, bbox, score

class VideoProcessingLogic:
    """
    Zawiera całą logikę przetwarzania obrazu z oryginalnej klasy VideoProcessor.
    Została przemianowana, aby uniknąć konfliktu nazw z węzłem ROS2.
    """
    STATUS_TRACKING = "Status: Tracking"
    STATUS_REID = "Status: Re-identification"
    STATUS_LOST = "Status: Lost"

    def __init__(self, tracking_data, model_path='vittrack.onnx'):
        self.tracking_data = tracking_data
        self.model_path = model_path

        self.frame_lock = Lock()
        self.frame = None

        self.tracker = None
        self.tracking_enabled = False
        self.tracking_start_time = None

        self.mouse_x = 0
        self.mouse_y = 0
        self.new_bbox = None
        self.roi_size = DEFAULT_ROI_SIZE
        self.min_roi_size = MIN_ROI_SIZE
        self.max_roi_size = None

        self.sift_template = None
        self.reid_interval = REID_INTERVAL
        self.frame_number = 0
        self.reid_fail_count = 0
        self.reid_thread_running = False
        self.reid_lock = Lock()

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 960, 720)
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _overlay_status(self, text, pos=STATUS_TEXT_POS):
        cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x - self.roi_size // 2, y - self.roi_size // 2
            self.new_bbox = (x1, y1, self.roi_size, self.roi_size)
            self.tracking_enabled = True
            self.tracker = VitTrack(self.model_path)
            with self.frame_lock:
                if self.frame is not None:
                    self.tracker.init(self.frame, self.new_bbox)
                    roi = self.frame[y1:y1 + self.roi_size, x1:x1 + self.roi_size]
                    if roi.size != 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(gray_roi, None)
                        self.sift_template = (kp, des, self.new_bbox)
            self.tracking_start_time = time.time()
            self.frame_number = 0
            self.reid_fail_count = 0
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None
            self.sift_template = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta_size = 10 if flags > 0 else -10
            if self.max_roi_size is None: self.max_roi_size = DEFAULT_MAX_ROI_SIZE
            self.roi_size = max(MIN_ROI_SIZE, min(self.roi_size + delta_size, self.max_roi_size))

    def draw_text(self, img, lines, start_x, start_y, **kwargs):
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * LINE_SPACING)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def draw_rectangle(self, img, bbox, color=COLOR_GREEN, thickness=1):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=COLOR_BLUE, thickness=1):
        half_size = size // 2
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(img.shape[1], x + half_size), min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_angle(self, dx, dy):
        dist = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        return dist, angle + 360 if angle < 0 else angle

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
                        self.tracking_enabled = True
                        self.tracking_start_time = time.time()
                    success = True
        self.reid_thread_running = False
        if success: self.reid_fail_count = 0

    def process_frame(self, frame):
        with self.frame_lock: self.frame = frame
        img_h, img_w = self.frame.shape[:2]
        if self.max_roi_size is None: self.max_roi_size = min(img_h, img_w)
        center_x, center_y = img_w // 2, img_h // 2
        cv2.circle(self.frame, (center_x, center_y), 2, (0, 0, 255), -1)
        frame_copy = self.frame.copy()

        if self.tracking_enabled and self.tracker is not None:
            found, bbox, score = self.tracker.infer(frame_copy)
            if found:
                x, y, w, h = map(int, bbox)
                if w >= img_w - IMG_MARGIN or h >= img_h - IMG_MARGIN:
                    self._overlay_status(self.STATUS_REID, STATUS_TEXT_POS)
                    self.tracking_enabled = False; self.tracker = None
                    with self.tracking_data.lock: self.tracking_data.status = self.STATUS_REID
                else:
                    cx, cy = self.draw_rectangle(frame_copy, bbox)
                    cv2.line(frame_copy, (center_x, center_y), (cx, cy), COLOR_YELLOW, 1)
                    dx, dy = center_x - cx, cy - center_y
                    dist, angle = self.calculate_distance_angle(dx, dy)
                    info = [self.STATUS_TRACKING, f"Score: {score:.2f}", f"dx: {dx}px", f"dy: {dy}px"]
                    self.draw_text(frame_copy, info, STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])
                    with self.tracking_data.lock:
                        self.tracking_data.status = self.STATUS_TRACKING
                        self.tracking_data.dx, self.tracking_data.dy = dx, dy
                        self.tracking_data.distance, self.tracking_data.angle = dist, angle
                        self.tracking_data.score, self.tracking_data.roi_height = score, h
                    if (time.time() - self.tracking_start_time > MIN_TRACK_DURATION) and (score < MIN_TRACK_SCORE):
                        self.tracking_enabled = False; self.tracker = None; self.tracking_start_time = None
                        with self.tracking_data.lock: self.tracking_data.status = self.STATUS_LOST
            else:
                self.tracking_enabled = False; self.tracker = None; self.tracking_start_time = None
                with self.tracking_data.lock: self.tracking_data.status = self.STATUS_LOST
            self.frame = frame_copy
            if found and score > SIFT_UPDATE_SCORE_THRESHOLD:
                x, y, w, h = map(int, bbox)
                if w > 0 and h > 0:
                    roi = frame_copy[y:y + h, x:x + w]
                    if roi.size != 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(gray_roi, None)
                        self.sift_template = (kp, des, bbox)

        if not self.tracking_enabled and self.sift_template is not None:
            self.frame_number += 1
            if (self.frame_number % self.reid_interval == 0) and not self.reid_thread_running:
                self.reid_thread_running = True
                threading.Thread(target=self._run_reid, args=(self.frame.copy(),)).start()
                self.reid_fail_count += self.reid_interval
                with self.tracking_data.lock: self.tracking_data.status = self.STATUS_REID
                if self.reid_fail_count >= REID_FAILURE_LIMIT:
                    self.sift_template = None; self.tracking_enabled = False
                    with self.tracking_data.lock: self.tracking_data.status = self.STATUS_LOST
        
        self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)
        return self.frame


# ==============================================================================
# SEKCJA 2: Węzeł ROS2
# ==============================================================================

class VideoProcessorNode(Node):
    def __init__(self):
        super().__init__('video_processor')
        self.get_logger().info("Inicjalizacja węzła Video Processor...")

        # Wczytaj ścieżkę do modelu z parametrów, jeśli istnieje
        self.declare_parameter('model_path', 'vittrack.onnx')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f"Używanie modelu śledzącego z: {model_path}")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Plik modelu nie został znaleziony w '{model_path}'! Upewnij się, że znajduje się w odpowiednim katalogu.")
            rclpy.shutdown()
            return

        # Utwórz instancję naszej logiki przetwarzania obrazu
        self.processing_logic = VideoProcessingLogic(
            tracking_data=TrackingData(),  # Używa wewnętrznej instancji do zarządzania stanem
            model_path=model_path
        )

        self.bridge = CvBridge()

        # Wydawcy
        self.processed_image_pub = self.create_publisher(Image, 'object_tracking/image_processed', 10)
        self.tracking_data_pub = self.create_publisher(TrackedObject, 'object_tracking/data', 10)

        # Subskrybent
        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 10)

        # Do obliczania FPS
        self.start_time = time.time()
        self.frame_count = 0

        self.get_logger().info("Węzeł Video Processor jest gotowy.")

    def image_callback(self, msg: Image):
        try:
            # Węzeł tello_driver publikuje obrazy w formacie RGB8
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu z ROS na CV2: {e}")
            return

        # Przeskaluj obraz do stałego rozmiaru na potrzeby przetwarzania
        resized_frame = cv2.resize(cv_frame, (RESIZED_WIDTH, RESIZED_HEIGHT))

        # Użyj logiki z oryginalnego pliku do przetworzenia klatki
        processed_frame = self.processing_logic.process_frame(resized_frame)

        # --- Opublikuj dane śledzenia ---
        tracking_msg = TrackedObject()
        with self.processing_logic.tracking_data.lock:
            tracking_msg.status = self.processing_logic.tracking_data.status
            tracking_msg.dx = float(self.processing_logic.tracking_data.dx)
            tracking_msg.dy = float(self.processing_logic.tracking_data.dy)
            tracking_msg.distance = self.processing_logic.tracking_data.distance
            tracking_msg.angle = self.processing_logic.tracking_data.angle
            tracking_msg.score = self.processing_logic.tracking_data.score
            tracking_msg.roi_height = self.processing_logic.tracking_data.roi_height
        
        self.tracking_data_pub.publish(tracking_msg)

        # --- Dodaj licznik FPS do wyświetlanego obrazu ---
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            processed_frame, f"FPS: {fps:.2f}",
            (10, processed_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1
        )

        # --- Opublikuj przetworzony obraz ---
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            processed_image_msg.header = msg.header  # Zachowaj oryginalny znacznik czasu
            self.processed_image_pub.publish(processed_image_msg)
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji przetworzonego obrazu na komunikat ROS: {e}")

        # --- Wyświetl okno (zachowanie oryginalnej funkcjonalności) ---
        cv2.imshow("Tracking", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Klawisz ESC
            self.get_logger().info("Naciśnięto ESC, zamykanie węzła.")
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    video_processor_node = VideoProcessorNode()
    try:
        rclpy.spin(video_processor_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Upewnij się, że zasoby są zwalniane przy zamykaniu
        cv2.destroyAllWindows()
        if rclpy.ok():
            video_processor_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()