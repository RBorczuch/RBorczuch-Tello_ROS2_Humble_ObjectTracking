import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np
import math
import time
import threading
import os
import datetime
from threading import Lock

# ==============================================================================
# SEKCJA 1: Logika przetwarzania wideo i śledzenia
# ==============================================================================

RESIZED_WIDTH = 960
RESIZED_HEIGHT = 720

# Stałe
DEFAULT_ROI_SIZE = 50
MIN_ROI_SIZE = 25
DEFAULT_MAX_ROI_SIZE = 300
ROI_ADJUST_STEP = 10
MIN_TRACK_DURATION = 0.5
MIN_TRACK_SCORE = 0.30
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LINE_SPACING = 25
STATUS_TEXT_POS = (10, 30)
RECORDING_TEXT_POS = (10, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

class TrackingData:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Lost"
        self.dx, self.dy = 0, 0
        self.score, self.roi_height = 0.0, 0

class VitTrack:
    def __init__(self, model_path):
        params = cv2.TrackerVit_Params()
        params.net = model_path
        self.model = cv2.TrackerVit_create(params)
    def init(self, image, roi): self.model.init(image, roi)
    def infer(self, image):
        found, bbox = self.model.update(image)
        return found, bbox, self.model.getTrackingScore()

class VideoProcessingLogic:
    def __init__(self, model_path):
        self.tracking_data = TrackingData()
        self.model_path = model_path
        self.tracker, self.tracking_enabled, self.tracking_start_time = None, False, None
        self.mouse_x, self.mouse_y = 0, 0
        self.roi_size = DEFAULT_ROI_SIZE
        self.initialization_request = None
        self.initialization_lock = Lock()

    def set_mouse_callback(self):
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = (x - self.roi_size // 2, y - self.roi_size // 2, self.roi_size, self.roi_size)
            with self.initialization_lock:
                self.initialization_request = bbox
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled = False
            self.tracker = None
            with self.tracking_data.lock: self.tracking_data.status = "Lost"

    def adjust_roi_size(self, delta):
        self.roi_size = max(MIN_ROI_SIZE, min(self.roi_size + delta, DEFAULT_MAX_ROI_SIZE))

    def draw_text(self, img, lines, start_x, start_y):
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * LINE_SPACING)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def process_frame(self, frame):
        if self.tracking_enabled and self.tracker is not None:
            found, bbox, score = self.tracker.infer(frame)
            if found:
                x, y, w, h = map(int, bbox)
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                cx, cy = x + w // 2, y + h // 2
                dx, dy = center_x - cx, cy - center_y
                dist = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))
                
                # ZMIANA: Przywrócono szczegółowe napisy
                info = [
                    f"Status: Tracking",
                    f"Score: {score:.2f}",
                    f"dx: {dx}px, dy: {dy}px",
                    f"Dystans: {dist:.1f}px",
                    f"Kat: {angle:.1f} deg"
                ]
                self.draw_text(frame, info, STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])

                with self.tracking_data.lock:
                    self.tracking_data.status, self.tracking_data.dx, self.tracking_data.dy = "Tracking", dx, dy
                    self.tracking_data.score, self.tracking_data.roi_height = score, h
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
                cv2.line(frame, (center_x, center_y), (cx, cy), COLOR_YELLOW, 1)
                
                if time.time() - self.tracking_start_time > MIN_TRACK_DURATION and score < MIN_TRACK_SCORE:
                    found = False
            
            if not found:
                self.tracking_enabled = False
                self.tracker = None
                with self.tracking_data.lock: self.tracking_data.status = "Lost"
        
        # ZMIANA: Dodano wyświetlanie statusu "Lost", gdy nie śledzimy
        if not self.tracking_enabled:
            self.draw_text(frame, ["Status: Lost"], STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])

        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 3, COLOR_RED, -1)
        cv2.rectangle(frame, (self.mouse_x - self.roi_size//2, self.mouse_y - self.roi_size//2),
                      (self.mouse_x + self.roi_size//2, self.mouse_y + self.roi_size//2), COLOR_BLUE, 1)
        return frame

# ==============================================================================
# SEKCJA 2: Węzeł ROS2
# ==============================================================================

class VideoProcessorNode(Node):
    def __init__(self):
        super().__init__('video_processor')
        self.get_logger().info("Inicjalizacja węzła Video Processor...")
        self.declare_parameter('model_path', 'vittrack.onnx')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.exists(model_path):
            self.get_logger().error(f"Plik modelu '{model_path}' nie znaleziony!")
            rclpy.shutdown(); return
        
        self.processing_logic = VideoProcessingLogic(model_path=model_path)
        self.bridge = CvBridge()
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", RESIZED_WIDTH, RESIZED_HEIGHT)
        self.processing_logic.set_mouse_callback()
        self.status_pub = self.create_publisher(String, 'object_tracking/status', 10)
        self.control_error_pub = self.create_publisher(Vector3, 'object_tracking/control_error', 10)
        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        
        self.latest_frame, self.frame_lock = None, Lock()
        self.processed_frame, self.processed_frame_lock = None, Lock()
        self.is_running = True
        self.frame_count, self.start_time = 0, time.time()
        self.recording, self.video_writer = False, None
        self.recordings_folder = "recordings"
        os.makedirs(self.recordings_folder, exist_ok=True)
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()
        self.display_timer = self.create_timer(1.0 / 60.0, self.display_loop)
        self.get_logger().info("Węzeł gotowy. Sterowanie odbywa się w oknie 'Tracking'.")

    def image_callback(self, msg: Image):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_frame_bgr = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)
            with self.frame_lock: self.latest_frame = cv_frame_bgr
        except Exception as e: self.get_logger().error(f"Błąd konwersji obrazu: {e}")

    def processing_loop(self):
        # ZMIANA: Ustawienie docelowego czasu klatki na 60 FPS
        target_frame_duration = 1.0 / 60.0

        while self.is_running and rclpy.ok():
            loop_start_time = time.time()

            bbox_to_init = None
            with self.processing_logic.initialization_lock:
                if self.processing_logic.initialization_request:
                    bbox_to_init = self.processing_logic.initialization_request
                    self.processing_logic.initialization_request = None

            if bbox_to_init:
                self.get_logger().info("Inicjalizacja śledzenia w tle...")
                frame_for_init = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame_for_init = cv2.resize(self.latest_frame.copy(), (RESIZED_WIDTH, RESIZED_HEIGHT))
                
                if frame_for_init is not None:
                    self.processing_logic.tracker = VitTrack(self.processing_logic.model_path)
                    self.processing_logic.tracker.init(frame_for_init, bbox_to_init)
                    self.processing_logic.tracking_enabled = True
                    self.processing_logic.tracking_start_time = time.time()
                    self.get_logger().info("Inicjalizacja zakończona.")
                else:
                    self.get_logger().warn("Brak klatki do inicjalizacji.")

            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                resized = cv2.resize(frame_to_process, (RESIZED_WIDTH, RESIZED_HEIGHT))
                processed = self.processing_logic.process_frame(resized)
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                self.processing_logic.draw_text(processed, [f"FPS: {fps:.2f}"], 10, processed.shape[0] - 30)
                rec_text = f"Nagrywanie: {'ON' if self.recording else 'OFF'} (klawisz 'n')"
                self.processing_logic.draw_text(processed, [rec_text], RECORDING_TEXT_POS[0], RECORDING_TEXT_POS[1])

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(processed)

                with self.processing_logic.tracking_data.lock:
                    status_msg = String(data=self.processing_logic.tracking_data.status)
                    control_msg = Vector3(
                        x=float(self.processing_logic.tracking_data.dx),
                        y=float(self.processing_logic.tracking_data.dy),
                        z=float(self.processing_logic.tracking_data.roi_height)
                    )
                    self.status_pub.publish(status_msg)
                    self.control_error_pub.publish(control_msg)
                
                with self.processed_frame_lock:
                    self.processed_frame = processed
            
            # ZMIANA: Logika ograniczania FPS
            elapsed_time = time.time() - loop_start_time
            sleep_time = target_frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def display_loop(self):
        frame_to_show = None
        with self.processed_frame_lock:
            if self.processed_frame is not None: frame_to_show = self.processed_frame
        
        if frame_to_show is not None:
            is_initializing = False
            with self.processing_logic.initialization_lock:
                if self.processing_logic.initialization_request is not None: is_initializing = True
            
            if is_initializing:
                 self.processing_logic.draw_text(frame_to_show, ["Inicjalizacja..."], 
                                                 int(RESIZED_WIDTH/2) - 50, int(RESIZED_HEIGHT/2))
            cv2.imshow("Tracking", frame_to_show)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: self.is_running = False; self.destroy_node()
        elif key == ord('1'): self.processing_logic.adjust_roi_size(-ROI_ADJUST_STEP)
        elif key == ord('2'): self.processing_logic.adjust_roi_size(ROI_ADJUST_STEP)
        elif key == ord('n'): self.toggle_recording()

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recordings_folder, f"tello_{RESIZED_HEIGHT}p_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (RESIZED_WIDTH, RESIZED_HEIGHT))
            self.get_logger().info(f"Rozpoczęto nagrywanie: {filename}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info("Zakończono nagrywanie.")

    def on_shutdown(self):
        self.is_running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        if self.video_writer is not None: self.video_writer.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Wątek przetwarzania i zasoby OpenCV zwolnione.")

def main(args=None):
    rclpy.init(args=args)
    video_processor_node = VideoProcessorNode()
    try: rclpy.spin(video_processor_node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
    finally:
        video_processor_node.on_shutdown()
        if rclpy.ok(): video_processor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()