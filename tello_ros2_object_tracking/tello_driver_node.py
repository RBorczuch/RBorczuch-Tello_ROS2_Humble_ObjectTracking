# tello_ros2_object_tracking/tello_driver_node.py

import rclpy
from rclpy.node import Node
from djitellopy import Tello
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, BatteryState, Temperature
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int16
import time
import threading
import math
from .utils import euler_to_quaternion

class TelloDriverNode(Node):
    """
    Manages the connection to the Tello drone and publishes its data.
    It uses a multi-threaded approach to fetch different data types at
    appropriate frequencies, ensuring high-priority data like video is not
    blocked by low-priority data like battery status.
    """
    def __init__(self):
        super().__init__('tello_driver')

        # --- Drone Connection ---
        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info(f"Battery: {self.tello.get_battery()}%")
        self.tello.streamon()
        self.get_logger().info("Video stream enabled.")
        self.bridge = CvBridge()
        
        # --- Publishers ---
        self.pub_image_raw = self.create_publisher(Image, 'image_raw', 10)
        self.pub_imu = self.create_publisher(Imu, 'tello/imu', 10)
        self.pub_tof = self.create_publisher(Int16, 'tello/tof', 10)
        self.pub_battery = self.create_publisher(BatteryState, 'tello/battery', 10)
        self.pub_temperature = self.create_publisher(Temperature, 'tello/temperature', 10)
        
        # --- Subscribers ---
        self.create_subscription(Twist, 'tello/cmd_vel', self._cmd_vel_callback, 10)
        self.create_subscription(String, 'tello/control', self._control_callback, 10)

        # --- Data Fetching Threads ---
        self.is_running = True
        self.state_lock = threading.Lock()
        self.current_state = {}

        threading.Thread(target=self._video_thread_loop, daemon=True).start()
        threading.Thread(target=self._fast_telemetry_loop, daemon=True).start()
        threading.Thread(target=self._slow_telemetry_loop, daemon=True).start()

        self.get_logger().info("Tello Driver node is ready. Data threads started.")

    def _cmd_vel_callback(self, msg: Twist):
        self.tello.send_rc_control(
            int(msg.linear.y), int(msg.linear.x),
            int(msg.linear.z), int(msg.angular.z)
        )

    def _control_callback(self, msg: String):
        if msg.data == 'takeoff': self.tello.takeoff()
        elif msg.data == 'land': self.tello.land()

    def _video_thread_loop(self):
        """High-frequency loop for video frames (~30 Hz)."""
        frame_read = self.tello.get_frame_read()
        while self.is_running:
            frame = frame_read.frame
            if frame is not None:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "rgb8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                self.pub_image_raw.publish(ros_image)
            time.sleep(1.0 / 30.0)

    def _fast_telemetry_loop(self):
        """Medium-frequency loop for critical flight data (~10 Hz)."""
        while self.is_running:
            try:
                state = self.tello.get_current_state()
                with self.state_lock:
                    self.current_state = state

                # Publish IMU data
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = "tello_imu_link"
                
                deg_to_rad = math.pi / 180.0
                q = euler_to_quaternion(
                    state['yaw'] * deg_to_rad, state['pitch'] * deg_to_rad, state['roll'] * deg_to_rad
                )
                imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w = q
                
                # Acceleration in m/s^2 (Tello returns cm/s^2, so divide by 100)
                imu_msg.linear_acceleration.x = float(state['agx']) / 100.0
                imu_msg.linear_acceleration.y = float(state['agy']) / 100.0
                imu_msg.linear_acceleration.z = float(state['agz']) / 100.0
                self.pub_imu.publish(imu_msg)

                # Publish ToF distance
                self.pub_tof.publish(Int16(data=int(state['tof'])))
            except Exception as e:
                self.get_logger().warn(f"Error in fast telemetry loop: {e}")
            time.sleep(1.0 / 10.0)

    def _slow_telemetry_loop(self):
        """Low-frequency loop for non-critical data (~1 Hz)."""
        while self.is_running:
            with self.state_lock:
                state = self.current_state.copy()
            if not state:
                time.sleep(1.0)
                continue

            try:
                # Publish battery state
                battery_msg = BatteryState()
                battery_msg.header.stamp = self.get_clock().now().to_msg()
                battery_msg.percentage = float(state['bat'])
                battery_msg.present = True
                self.pub_battery.publish(battery_msg)

                # Publish temperature
                temp_msg = Temperature()
                temp_msg.header.stamp = self.get_clock().now().to_msg()
                temp_msg.temperature = (state['templ'] + state['temph']) / 2.0
                self.pub_temperature.publish(temp_msg)
            except Exception as e:
                self.get_logger().warn(f"Error in slow telemetry loop: {e}")
            time.sleep(1.0)

    def on_shutdown(self):
        """Ensures a safe shutdown of the drone and threads."""
        self.get_logger().info("Shutting down Tello Driver node...")
        self.is_running = False
        time.sleep(0.5) # Give threads a moment to exit their loops
        
        if self.tello.is_flying:
            self.get_logger().info("Drone is flying, landing now.")
            self.tello.land()
        
        self.tello.streamoff()
        self.tello.end()
        self.get_logger().info("Connection to Tello closed.")

def main(args=None):
    rclpy.init(args=args)
    node = TelloDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()