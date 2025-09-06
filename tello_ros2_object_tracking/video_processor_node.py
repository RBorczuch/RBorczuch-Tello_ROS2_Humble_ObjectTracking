# tello_ros2_object_tracking/video_processor_node.py

import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import time
import threading
import os
import datetime
from threading import Lock
from .vision_logic import (VideoProcessingLogic, VitTrack, ROI_ADJUST_STEP, 
                           RESIZED_WIDTH, RESIZED_HEIGHT, RECORDING_TEXT_POS)

class VideoProcessorNode(Node):
    def __init__(self):
        super().__init__('video_processor')
        # ... (constructor initialization as before)
        self.declare_parameter('model_path', 'vittrack.onnx')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.exists(model_path):
            self.get_logger().error(f"Tracker model file not found at '{model_path}'!")
            rclpy.shutdown(); return
        
        self.processing_logic = VideoProcessingLogic(model_path=model_path)
        self.bridge = CvBridge()
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", RESIZED_WIDTH, RESIZED_HEIGHT)
        self.processing_logic.set_mouse_callback()
        self.status_pub = self.create_publisher(String, 'object_tracking/status', 10)
        self.control_error_pub = self.create_publisher(Vector3, 'object_tracking/control_error', 10)
        self.image_sub = self.create_subscription(Image, 'image_raw', self._image_callback, 10)
        # NEW: Subscriber for controller state
        self.create_subscription(String, 'tello/control_state', self._control_state_callback, 10)
        
        self.latest_frame, self.frame_lock = None, Lock()
        self.processed_frame, self.processed_frame_lock = None, Lock()
        self.is_running = True
        # ... (rest of constructor)
        self.frame_count, self.start_time = 0, time.time()
        self.recording, self.video_writer = False, None
        self.recordings_folder = "recordings"
        os.makedirs(self.recordings_folder, exist_ok=True)
        threading.Thread(target=self._processing_loop, daemon=True).start()
        self.create_timer(1.0 / 60.0, self._display_loop)
        self.get_logger().info("Video Processor node ready.")

    def _control_state_callback(self, msg: String):
        """Receives control state and updates the vision logic's data object."""
        try:
            mode, fwd_str = msg.data.split(',')
            with self.processing_logic.tracking_data.lock:
                self.processing_logic.tracking_data.control_mode = mode
                self.processing_logic.tracking_data.forward_enabled = (fwd_str == 'True')
        except ValueError:
            self.get_logger().warn(f"Received invalid control state message: {msg.data}")

    def _processing_loop(self):
        target_frame_duration = 1.0 / 60.0
        while self.is_running:
            # ... (as before)
            loop_start_time = time.time()
            bbox_to_init = None
            with self.processing_logic.initialization_lock:
                if self.processing_logic.initialization_request:
                    bbox_to_init = self.processing_logic.initialization_request
                    self.processing_logic.initialization_request = None
            if bbox_to_init: self._initialize_tracker(bbox_to_init)
            
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None: frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None:
                # Pass the original frame for potential re-id
                processed = self.processing_logic.process_frame(frame_to_process, frame_to_process)
                self._add_overlays_and_record(processed)
                self._publish_tracking_data()
                with self.processed_frame_lock: self.processed_frame = processed
            
            elapsed = time.time() - loop_start_time
            sleep_time = target_frame_duration - elapsed
            if sleep_time > 0: time.sleep(sleep_time)
            
    # ... (rest of the file is the same as the previous refactored version)
    def _image_callback(self, msg: Image):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_frame_bgr = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)
            with self.frame_lock:
                self.latest_frame = cv_frame_bgr
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    def _initialize_tracker(self, bbox: tuple):
        self.get_logger().info("Initializing tracker in background...")
        frame_for_init = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_for_init = cv2.resize(self.latest_frame.copy(), (RESIZED_WIDTH, RESIZED_HEIGHT))
        
        if frame_for_init is not None:
            self.processing_logic.tracker = VitTrack(self.processing_logic.model_path)
            self.processing_logic.tracker.init(frame_for_init, bbox)
            self.processing_logic.tracking_enabled = True
            self.processing_logic.tracking_start_time = time.time()
            self.get_logger().info("Tracker initialized.")
        else:
            self.get_logger().warn("No frame available for tracker initialization.")

    def _add_overlays_and_record(self, frame):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        self.processing_logic.draw_text(frame, [f"FPS: {fps:.2f}"], 10, frame.shape[0] - 30)
        
        rec_text = f"Recording: {'ON' if self.recording else 'OFF'} (Press 'n')"
        self.processing_logic.draw_text(frame, [rec_text], RECORDING_TEXT_POS[0], RECORDING_TEXT_POS[1])

        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def _publish_tracking_data(self):
        with self.processing_logic.tracking_data.lock:
            status_msg = String(data=self.processing_logic.tracking_data.status)
            control_msg = Vector3(
                x=float(self.processing_logic.tracking_data.dx),
                y=float(self.processing_logic.tracking_data.dy),
                z=float(self.processing_logic.tracking_data.roi_height)
            )
            self.status_pub.publish(status_msg)
            self.control_error_pub.publish(control_msg)
    
    def _display_loop(self):
        frame_to_show = None
        with self.processed_frame_lock:
            if self.processed_frame is not None:
                frame_to_show = self.processed_frame
        
        if frame_to_show is not None:
            is_initializing = False
            with self.processing_logic.initialization_lock:
                if self.processing_logic.initialization_request is not None:
                    is_initializing = True
            
            if is_initializing:
                 self.processing_logic.draw_text(frame_to_show, ["Initializing..."], 
                                                 int(RESIZED_WIDTH/2) - 50, int(RESIZED_HEIGHT/2))
            cv2.imshow("Tracking", frame_to_show)
        
        self._handle_keyboard()
        
    def _handle_keyboard(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            self.is_running = False
            self.destroy_node()
        elif key == ord('1'): self.processing_logic.adjust_roi_size(-ROI_ADJUST_STEP)
        elif key == ord('2'): self.processing_logic.adjust_roi_size(ROI_ADJUST_STEP)
        elif key == ord('n'): self._toggle_recording()

    def _toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(self.recordings_folder, f"tello_{RESIZED_HEIGHT}p_{ts}.mp4")
            self.video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (RESIZED_WIDTH, RESIZED_HEIGHT))
            self.get_logger().info(f"Recording started: {fn}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info("Recording stopped.")

    def on_shutdown(self):
        self.is_running = False
        time.sleep(0.1)
        if self.video_writer is not None: self.video_writer.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Video processor resources released.")
        
def main(args=None):
    rclpy.init(args=args)
    node = VideoProcessorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.on_shutdown()
        if rclpy.ok(): node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()