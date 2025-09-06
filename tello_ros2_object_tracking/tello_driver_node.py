import rclpy
from rclpy.node import Node
from djitellopy import Tello
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, BatteryState, Temperature
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int16
import time
import threading
import math
import numpy as np

def euler_to_quaternion(yaw, pitch, roll):
    """ Konwertuje kąty Eulera (w radianach) na kwaternion. """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = [0] * 4
    q[0] = cy * cp * sr - sy * sp * cr  # x
    q[1] = sy * cp * sr + cy * sp * cr  # y
    q[2] = sy * cp * cr - cy * sp * sr  # z
    q[3] = cy * cp * cr + sy * sp * sr  # w
    return q

class TelloDriverNode(Node):
    def __init__(self):
        super().__init__('tello_driver')
        self.get_logger().info("Inicjalizacja zoptymalizowanego węzła Tello Driver...")

        # --- Połączenie z dronem ---
        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info(f"Bateria: {self.tello.get_battery()}%")
        self.tello.streamon()
        self.get_logger().info("Strumień wideo włączony.")
        
        self.bridge = CvBridge()
        
        # --- Wydawcy (Publishers) ---
        # Dane wysokiej częstotliwości
        self.pub_image_raw = self.create_publisher(Image, 'image_raw', 10)
        self.pub_imu = self.create_publisher(Imu, 'tello/imu', 10)
        self.pub_tof = self.create_publisher(Int16, 'tello/tof', 10)
        # Dane niskiej częstotliwości
        self.pub_battery = self.create_publisher(BatteryState, 'tello/battery', 10)
        self.pub_temperature = self.create_publisher(Temperature, 'tello/temperature', 10)
        
        # --- Subskrybenci (Subscribers) ---
        self.create_subscription(Twist, 'tello/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(String, 'tello/control', self.control_callback, 10)

        # --- Wątki do pobierania danych ---
        self.is_running = True
        self.state_lock = threading.Lock()
        self.current_state = {}

        self.video_thread = threading.Thread(target=self._video_thread_loop)
        self.fast_telemetry_thread = threading.Thread(target=self._fast_telemetry_loop)
        self.slow_telemetry_thread = threading.Thread(target=self._slow_telemetry_loop)
        
        self.video_thread.start()
        self.fast_telemetry_thread.start()
        self.slow_telemetry_thread.start()

        self.get_logger().info("Węzeł Tello Driver jest gotowy. Wątki danych uruchomione.")

    def cmd_vel_callback(self, msg):
        self.tello.send_rc_control(
            int(msg.linear.y), int(msg.linear.x),
            int(msg.linear.z), int(msg.angular.z)
        )

    def control_callback(self, msg):
        if msg.data == 'takeoff': self.tello.takeoff()
        elif msg.data == 'land': self.tello.land()

    def _video_thread_loop(self):
        """ Pętla wątku wideo, działa z maksymalną możliwą prędkością. """
        frame_read = self.tello.get_frame_read()
        while self.is_running:
            frame = frame_read.frame
            if frame is not None:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "rgb8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                self.pub_image_raw.publish(ros_image)
            time.sleep(1.0 / 30.0) # Ograniczenie do ~30 FPS

    def _fast_telemetry_loop(self):
        """ Pętla wątku telemetrii szybkiej, działa z częstotliwością ~10 Hz. """
        while self.is_running:
            try:
                # Pobierz cały stan naraz - to jest bardziej wydajne
                state = self.tello.get_current_state()
                with self.state_lock:
                    self.current_state = state

                # --- Publikacja danych IMU ---
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = "tello_imu_link" # Można dostosować
                
                # Kwaternion orientacji
                # djitellopy zwraca kąty w stopniach
                deg_to_rad = math.pi / 180.0
                q = euler_to_quaternion(
                    state['yaw'] * deg_to_rad,
                    state['pitch'] * deg_to_rad,
                    state['roll'] * deg_to_rad
                )
                imu_msg.orientation.x = q[0]
                imu_msg.orientation.y = q[1]
                imu_msg.orientation.z = q[2]
                imu_msg.orientation.w = q[3]

                # Przyspieszenie liniowe (w m/s^2)
                # djitellopy zwraca w 'g' * 100, więc dzielimy przez 100 i mnożymy przez 9.81
                g = 9.80665
                imu_msg.linear_acceleration.x = state['agx'] * g / 1000.0
                imu_msg.linear_acceleration.y = state['agy'] * g / 1000.0
                imu_msg.linear_acceleration.z = state['agz'] * g / 1000.0
                self.pub_imu.publish(imu_msg)

                # --- Publikacja danych ToF ---
                tof_msg = Int16(data=int(state['tof']))
                self.pub_tof.publish(tof_msg)

            except Exception as e:
                self.get_logger().warn(f"Błąd w pętli szybkiej telemetrii: {e}")

            time.sleep(1.0 / 10.0) # Celuj w 10 Hz

    def _slow_telemetry_loop(self):
        """ Pętla wątku telemetrii wolnej, działa z częstotliwością ~1 Hz. """
        while self.is_running:
            with self.state_lock:
                state = self.current_state.copy()

            if not state: # Jeśli stan nie został jeszcze pobrany
                time.sleep(1.0)
                continue

            try:
                # --- Publikacja stanu baterii ---
                battery_msg = BatteryState()
                battery_msg.header.stamp = self.get_clock().now().to_msg()
                battery_msg.percentage = float(state['bat'])
                battery_msg.present = True
                self.pub_battery.publish(battery_msg)

                # --- Publikacja temperatury ---
                temp_msg = Temperature()
                temp_msg.header.stamp = self.get_clock().now().to_msg()
                # Używamy średniej z najniższej i najwyższej temperatury
                temp_avg = (state['templ'] + state['temph']) / 2.0
                temp_msg.temperature = temp_avg
                self.pub_temperature.publish(temp_msg)

            except Exception as e:
                self.get_logger().warn(f"Błąd w pętli wolnej telemetrii: {e}")

            time.sleep(1.0) # Celuj w 1 Hz

    def on_shutdown(self):
        self.get_logger().info("Zamykanie węzła Tello Driver...")
        self.is_running = False
        self.video_thread.join(timeout=2)
        self.fast_telemetry_thread.join(timeout=2)
        self.slow_telemetry_thread.join(timeout=2)
        
        if self.tello.is_flying: self.tello.land()
        self.tello.streamoff()
        self.tello.end()

def main(args=None):
    rclpy.init(args=args)
    tello_driver_node = TelloDriverNode()
    try:
        rclpy.spin(tello_driver_node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        tello_driver_node.on_shutdown()
        tello_driver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()