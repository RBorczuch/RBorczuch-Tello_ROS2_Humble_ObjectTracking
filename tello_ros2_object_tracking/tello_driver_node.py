import rclpy
from rclpy.node import Node
from djitellopy import Tello
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from tello_ros2_object_tracking.msg import TelloStatus
import time

class TelloDriverNode(Node):
    def __init__(self):
        super().__init__('tello_driver')
        self.get_logger().info("Inicjalizacja węzła Tello Driver...")

        self.tello = Tello()
        self.bridge = CvBridge()

        # Inicjalizacja drona
        try:
            self.tello.connect()
            self.get_logger().info(f"Bateria: {self.tello.get_battery()}%")
            self.tello.streamon()
            self.get_logger().info("Strumień wideo włączony.")
        except Exception as e:
            self.get_logger().error(f"Nie udało się zainicjalizować Tello: {e}")
            rclpy.shutdown()
            return

        # Wydawcy
        self.image_publisher = self.create_publisher(Image, 'image_raw', 10)
        self.status_publisher = self.create_publisher(TelloStatus, 'tello/status', 10)

        # Subskrybenci
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, 'tello/cmd_vel', self.cmd_vel_callback, 10)
        self.control_subscriber = self.create_subscription(
            String, 'tello/control', self.control_callback, 10)

        # Timery do publikowania danych
        self.video_timer = self.create_timer(1/30.0, self.publish_video_frame) # 30 FPS
        self.status_timer = self.create_timer(0.1, self.publish_status) # 10 Hz

        self.get_logger().info("Węzeł Tello Driver jest gotowy.")

    def cmd_vel_callback(self, msg):
        # Komunikaty Twist: linear.x (przód/tył), linear.y (lewo/prawo), linear.z (góra/dół), angular.z (obrót)
        # Tello: send_rc_control(lewo_prawo, przód_tył, góra_dół, obrót)
        self.tello.send_rc_control(
            int(msg.linear.y),
            int(msg.linear.x),
            int(msg.linear.z),
            int(msg.angular.z)
        )

    def control_callback(self, msg):
        if msg.data == 'takeoff':
            self.get_logger().info("Startowanie...")
            self.tello.takeoff()
        elif msg.data == 'land':
            self.get_logger().info("Lądowanie...")
            self.tello.land()

    def publish_video_frame(self):
        frame_read = self.tello.get_frame_read()
        if frame_read.frame is not None:
            # Tello zwraca klatki w formacie RGB
            frame = frame_read.frame
            # Konwertuj do wiadomości ROS i opublikuj
            ros_image = self.bridge.cv2_to_imgmsg(frame, "rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.image_publisher.publish(ros_image)

    def publish_status(self):
        try:
            state = self.tello.get_current_state()
            status_msg = TelloStatus()
            status_msg.battery = int(state.get('bat', 0))
            status_msg.temperature_low = int(state.get('templ', 0))
            status_msg.temperature_high = int(state.get('temph', 0))
            status_msg.flight_time = int(state.get('time', 0))
            status_msg.barometer = float(state.get('baro', 0.0))
            status_msg.pitch = int(state.get('pitch', 0))
            status_msg.roll = int(state.get('roll', 0))
            status_msg.yaw = int(state.get('yaw', 0))
            status_msg.vgx = int(state.get('vgx', 0))
            status_msg.vgy = int(state.get('vgy', 0))
            status_msg.vgz = int(state.get('vgz', 0))
            status_msg.tof_distance = int(state.get('tof', 0))
            status_msg.height = int(state.get('h', 0))
            status_msg.agx = float(state.get('agx', 0.0))
            status_msg.agy = float(state.get('agy', 0.0))
            status_msg.agz = float(state.get('agz', 0.0))
            
            # Wi-Fi jest osobnym zapytaniem
            # status_msg.wifi_snr = int(self.tello.query_wifi_signal_noise_ratio())
            
            self.status_publisher.publish(status_msg)
        except Exception as e:
            self.get_logger().warn(f"Nie udało się pobrać statusu Tello: {e}")

    def on_shutdown(self):
        self.get_logger().info("Zamykanie węzła Tello Driver...")
        try:
            if self.tello.is_flying:
                self.tello.land()
        except Exception as e:
            self.get_logger().warn(f"Nie udało się wylądować: {e}")
        finally:
            self.tello.streamoff()
            self.tello.end()

def main(args=None):
    rclpy.init(args=args)
    tello_driver_node = TelloDriverNode()
    try:
        rclpy.spin(tello_driver_node)
    except KeyboardInterrupt:
        pass
    finally:
        tello_driver_node.on_shutdown()
        tello_driver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()```

**Plik: `tello_ros2_object_tracking/video_processor_node.py`**
Ten węzeł subskrybuje surowy obraz, przetwarza go i publikuje dane śledzenia oraz przetworzony obraz.

```python
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tello_ros2_object_tracking.msg import TrackedObject
import numpy as np
import math
import time
from threading import Lock

# Import oryginalnej logiki z process_video.py
from .process_video import VitTrack, VideoProcessor

class VideoProcessorNode(VideoProcessor, Node):
    def __init__(self):
        # Inicjalizacja węzła ROS2
        Node.__init__(self, 'video_processor')
        self.get_logger().info("Inicjalizacja węzła Video Processor...")

        # Inicjalizacja logiki przetwarzania wideo (klasy nadrzędnej)
        # Tworzymy fałszywy obiekt tracking_data, ponieważ stan jest teraz zarządzany wewnątrz tej klasy
        class DummyTrackingData:
            def __init__(self):
                self.lock = Lock()
                self.status = "Lost"
                self.dx = 0; self.dy = 0; self.distance = 0.0; self.angle = 0.0
                self.score = 0.0; self.roi_height = 0
                self.control_mode = "Manual"; self.forward_enabled = False
        
        VideoProcessor.__init__(self, tracking_data=DummyTrackingData())

        self.bridge = CvBridge()

        # Wydawcy
        self.processed_image_pub = self.create_publisher(Image, 'object_tracking/image_processed', 10)
        self.tracking_data_pub = self.create_publisher(TrackedObject, 'object_tracking/data', 10)

        # Subskrybenci
        self.image_sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10)

        self.get_logger().info("Węzeł Video Processor jest gotowy.")

    def image_callback(self, msg):
        try:
            # Konwertuj wiadomość ROS na klatkę OpenCV (BGR)
            # Tello Driver publikuje jako RGB8
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu: {e}")
            return

        # Użyj logiki z oryginalnego pliku do przetworzenia klatki
        processed_frame = self.process_frame(cv_frame)

        # Opublikuj przetworzony obraz
        processed_image_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
        processed_image_msg.header.stamp = self.get_clock().now().to_msg()
        self.processed_image_pub.publish(processed_image_msg)
        
        # Opublikuj dane śledzenia
        tracking_msg = TrackedObject()
        with self.tracking_data.lock:
            tracking_msg.status = self.tracking_data.status
            tracking_msg.dx = float(self.tracking_data.dx)
            tracking_msg.dy = float(self.tracking_data.dy)
            tracking_msg.distance = self.tracking_data.distance
            tracking_msg.angle = self.tracking_data.angle
            tracking_msg.score = self.tracking_data.score
            tracking_msg.roi_height = self.tracking_data.roi_height
        
        self.tracking_data_pub.publish(tracking_msg)

        # Pokaż okno (oryginalna funkcjonalność)
        cv2.imshow("Tracking", processed_frame)
        key = cv2.waitKey(1)
        if key == 27: # ESC
             self.get_logger().info("Naciśnięto ESC, zamykanie.")
             cv2.destroyAllWindows()
             rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    video_processor_node = VideoProcessorNode()
    rclpy.spin(video_processor_node)
    video_processor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()