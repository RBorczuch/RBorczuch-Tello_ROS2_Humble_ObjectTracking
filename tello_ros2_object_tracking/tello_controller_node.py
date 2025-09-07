# tello_ros2_object_tracking/tello_controller_node.py

import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
from .PID import PIDController
import sys, select, tty, termios
import numpy as np

class KeyboardReader:
    def __init__(self): self.settings = termios.tcgetattr(sys.stdin)
    def __enter__(self):
        tty.setraw(sys.stdin.fileno())
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
    def read_key(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []): return sys.stdin.read(1)
        return None

# --- Constants ---
LOOP_SLEEP = 0.05
MODE_SWITCH_COOLDOWN = 0.3
ERROR_THRESHOLD = 20
DEAD_ZONE = 5

# ==========================================================
# ZMIANA: Przywrócono wartości PID identyczne z działającym kodem referencyjnym.
# To jest ostateczna i kluczowa poprawka.
# ==========================================================
PID_CONFIG = {
    'yaw':      {'kp': 0.2, 'ki': 0.01, 'kd': 0.0},
    'vertical': {'kp': 0.2, 'ki': 0.01, 'kd': 0.0},
    'forward':  {'kp': 0.4, 'ki': 0.0, 'kd': 0.0, 'setpoint': 120}
}
VELOCITY_CONFIG = { 'initial': 50, 'min': 10, 'max': 100, 'step': 10 }

class TelloControllerNode(Node):
    def __init__(self, key_reader: KeyboardReader):
        super().__init__('tello_controller')
        self.key_reader = key_reader
        self.cmd_vel_publisher = self.create_publisher(Twist, 'tello/cmd_vel', 10)
        self.control_publisher = self.create_publisher(String, 'tello/control', 10)
        self.state_publisher = self.create_publisher(String, 'tello/control_state', 10)
        self.create_subscription(String, 'object_tracking/status', self._status_callback, 10)
        self.create_subscription(Vector3, 'object_tracking/control_error', self._control_error_callback, 10)
        self.pids = {name: PIDController(**params) for name, params in PID_CONFIG.items()}
        self.velocity = VELOCITY_CONFIG['initial']
        self.last_mode_switch = 0
        self.control_mode = "Manual"
        self.forward_enabled = False
        self.tracking_status = "Lost"
        self.control_error = None
        self.create_timer(LOOP_SLEEP, self._control_loop)
        self._display_controls()
        self.get_logger().info("Tello Controller node is ready.")
        
    def _status_callback(self, msg: String): self.tracking_status = msg.data
    def _control_error_callback(self, msg: Vector3): self.control_error = msg

    def _control_loop(self):
        key = self.key_reader.read_key()
        if key and self._handle_command_keys(key):
            self._publish_state()
            return
        if self.control_mode == "Manual":
            self._process_manual_control(key)
        else:
            if key and key.lower() in 'wasdrfeq':
                 self._switch_to_manual("Manual input override.")
            else:
                 self._process_autonomous_control()
        self._publish_state()

    def _publish_state(self):
        state_str = f"{self.control_mode},{self.forward_enabled}"
        self.state_publisher.publish(String(data=state_str))

    def _process_manual_control(self, key: str | None):
        cmd = Twist()
        vel = float(self.velocity)
        key_map = {
            'w': ('linear', 'x', 1),  's': ('linear', 'x', -1),
            'a': ('linear', 'y', -1), 'd': ('linear', 'y', 1),
            'r': ('linear', 'z', 1),  'f': ('linear', 'z', -1),
            'q': ('angular', 'z', -1),'e': ('angular', 'z', 1),
        }
        if key and key.lower() in key_map:
            axis_type, axis_name, sign = key_map[key.lower()]
            setattr(getattr(cmd, axis_type), axis_name, vel * sign)
        self.cmd_vel_publisher.publish(cmd)

    def _process_autonomous_control(self):
        if self.tracking_status != "Tracking" or self.control_error is None:
            self._switch_to_manual("Target lost.")
            return
            
        dx, dy, roi_height = self.control_error.x, self.control_error.y, self.control_error.z

        x_vel = self._calculate_pid_output('forward', roi_height) if self.forward_enabled else 0.0
        yaw_vel = self._calculate_pid_output('yaw', dx, ERROR_THRESHOLD)
        z_vel = self._calculate_pid_output('vertical', dy, ERROR_THRESHOLD)

        cmd = Twist()
        cmd.linear.x = float(x_vel)
        cmd.linear.y = 0.0
        cmd.linear.z = float(z_vel)
        cmd.angular.z = float(yaw_vel)
        self.cmd_vel_publisher.publish(cmd)

    def _calculate_pid_output(self, pid_key: str, error: float, threshold: float = 0) -> float:
        if abs(error) < max(threshold, DEAD_ZONE):
            self.pids[pid_key].reset()
            return 0.0
        return self.pids[pid_key].compute(error)

    def _handle_command_keys(self, key: str) -> bool:
        key_lower = key.lower()
        now = time.time()
        if key_lower in 'tl':
            cmd = 'takeoff' if key_lower == 't' else 'land'
            self.control_publisher.publish(String(data=cmd))
        elif key in ',.':
            step = VELOCITY_CONFIG['step'] * (-1 if key == ',' else 1)
            self.velocity = np.clip(self.velocity + step, VELOCITY_CONFIG['min'], VELOCITY_CONFIG['max'])
            self.get_logger().info(f"Velocity set to: {self.velocity}")
        elif key == ' ':
            if now - self.last_mode_switch > MODE_SWITCH_COOLDOWN:
                self.control_mode = "Autonomous" if self.control_mode == "Manual" else "Manual"
                self.get_logger().info(f"Switched to {self.control_mode} mode.")
                self.last_mode_switch = now
        elif key_lower == 'x':
             if self.control_mode == "Autonomous" and now - self.last_mode_switch > MODE_SWITCH_COOLDOWN:
                self.forward_enabled = not self.forward_enabled
                self.get_logger().info(f"Forward movement: {'ENABLED' if self.forward_enabled else 'DISABLED'}")
                self.last_mode_switch = now
        elif key == '\x03' or key_lower == 'p':
            self.get_logger().info("Shutdown requested.")
            self.cmd_vel_publisher.publish(Twist())
            rclpy.shutdown()
        else: return False
        return True

    def _switch_to_manual(self, reason: str):
        if self.control_mode == "Autonomous":
            self.control_mode = "Manual"
            self.get_logger().warn(f"Switching to Manual mode: {reason}")
            self.cmd_vel_publisher.publish(Twist())
    
    def _display_controls(self):
        self.get_logger().info(
            "\n" + "="*40 +
            "\n Tello Controller Interface" +
            "\n----------------------------------------" +
            "\n Controls:" +
            "\n   W/S: Forward/Backward | A/D: Left/Right" +
            "\n   R/F: Up/Down          | Q/E: Rotate" +
            "\n Commands:" +
            "\n   T/L: Takeoff/Land     | ,/. : Dec/Inc Speed" +
            "\n   SPACE: Toggle Manual/Autonomous mode" +
            "\n   X (in Auto): Toggle forward movement" +
            "\n   P or CTRL+C: Quit" +
            "\n" + "="*40)

def main(args=None):
    rclpy.init(args=args)
    with KeyboardReader() as key_reader:
        node = TelloControllerNode(key_reader)
        try: rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
        finally: node.destroy_node()
    rclpy.shutdown()
    print("\nTerminal settings restored.")

if __name__ == '__main__':
    main()