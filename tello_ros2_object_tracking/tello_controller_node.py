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
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

# Stałe
LOOP_SLEEP = 0.05
MODE_SWITCH_SLEEP = 0.3
ERROR_THRESHOLD = 20
DEAD_ZONE = 5
PID_CONFIG = {
    'yaw': {'kp': 0.25, 'ki': 0.01, 'kd': 0.1},
    'vertical': {'kp': 0.3, 'ki': 0.01, 'kd': 0.1},
    'forward': {'kp': 0.4, 'ki': 0.0, 'kd': 0.1, 'setpoint': 120}
}
VELOCITY_CONFIG = { 'initial': 50, 'min': 10, 'max': 100, 'step': 10 }

class TelloControllerNode(Node):
    def __init__(self, key_reader: KeyboardReader):
        super().__init__('tello_controller')
        self.key_reader = key_reader

        self.cmd_vel_publisher = self.create_publisher(Twist, 'tello/cmd_vel', 10)
        self.control_publisher = self.create_publisher(String, 'tello/control', 10)

        # Subskrybenci danych śledzenia
        self.create_subscription(String, 'object_tracking/status', self.status_callback, 10)
        self.create_subscription(Vector3, 'object_tracking/control_error', self.control_error_callback, 10)

        self.pids = {
            'yaw': PIDController(**PID_CONFIG['yaw']),
            'vertical': PIDController(**PID_CONFIG['vertical']),
            'forward': PIDController(**PID_CONFIG['forward'])
        }
        self.velocity = VELOCITY_CONFIG['initial']
        self.last_mode_switch = 0
        self.control_mode = "Manual"
        self.forward_enabled = False
        
        # Zmienne przechowujące stan śledzenia
        self.tracking_status = "Lost"
        self.control_error = None # Będzie przechowywać wiadomość Vector3

        self.timer = self.create_timer(LOOP_SLEEP, self.control_loop)
        self._display_controls()
        self.get_logger().info("Węzeł Tello Controller jest gotowy.")
        
    def status_callback(self, msg):
        self.tracking_status = msg.data

    def control_error_callback(self, msg):
        self.control_error = msg

    def control_loop(self):
        key = self.key_reader.read_key()
        if key:
            if self._handle_non_movement_keys(key): return

        if self.control_mode == "Manual":
            self._process_manual_control(key)
        else: # Tryb autonomiczny
            if key and key.lower() in 'wasdrfqe':
                 self.get_logger().info("Wykryto ręczne sterowanie, przełączanie na tryb manualny.")
                 self._switch_to_manual()
            else:
                 self._process_autonomous_control()

    def _process_manual_control(self, key):
        cmd = Twist()
        vel = float(self.velocity)
        key_map = {'w': ('x', vel), 's': ('x', -vel), 'a': ('y', vel), 'd': ('y', -vel),
                   'r': ('z', vel), 'f': ('z', -vel)}
        if key and key.lower() in key_map:
            axis, value = key_map[key.lower()]
            setattr(cmd.linear, axis, value)
        
        key_map_angular = {'q': -vel, 'e': vel}
        if key and key.lower() in key_map_angular:
            cmd.angular.z = float(key_map_angular[key.lower()])
            
        self.cmd_vel_publisher.publish(cmd)

    def _process_autonomous_control(self):
        if self.tracking_status != "Tracking" or self.control_error is None:
            self._switch_to_manual()
            return
            
        dx = self.control_error.x
        dy = self.control_error.y
        roi_height = self.control_error.z

        yaw_vel = -self._calculate_pid_output('yaw', dx, ERROR_THRESHOLD) 
        z_vel = -self._calculate_pid_output('vertical', dy, ERROR_THRESHOLD)
        x_vel = self._calculate_pid_output('forward', roi_height) if self.forward_enabled else 0.0

        cmd = Twist()
        cmd.linear.x = float(x_vel)
        cmd.linear.z = float(z_vel)
        cmd.angular.z = float(yaw_vel)
        self.cmd_vel_publisher.publish(cmd)

    def _calculate_pid_output(self, pid_key, error, threshold=0):
        if abs(error) < max(threshold, DEAD_ZONE):
            self.pids[pid_key].reset()
            return 0.0
        clamped_error = np.clip(error, -500, 500)
        return self.pids[pid_key].compute(clamped_error)

    def _handle_non_movement_keys(self, key):
        if key.lower() in 'tl':
            cmd = 'takeoff' if key.lower() == 't' else 'land'
            self.get_logger().info(f"Polecenie: {cmd.upper()}")
            self.control_publisher.publish(String(data=cmd))
            return True
        elif key in ',.':
            step = VELOCITY_CONFIG['step'] * (-1 if key == ',' else 1)
            self.velocity = np.clip(self.velocity + step, VELOCITY_CONFIG['min'], VELOCITY_CONFIG['max'])
            self.get_logger().info(f"Prędkość: {self.velocity}")
            return True
        elif key == ' ':
            if time.time() - self.last_mode_switch > MODE_SWITCH_SLEEP:
                self.control_mode = "Autonomous" if self.control_mode == "Manual" else "Manual"
                self.get_logger().info(f"Tryb: {self.control_mode}")
                self.last_mode_switch = time.time()
            return True
        elif key.lower() == 's' and self.control_mode == "Autonomous":
             if time.time() - self.last_mode_switch > MODE_SWITCH_SLEEP:
                self.forward_enabled = not self.forward_enabled
                self.get_logger().info(f"Celowanie do przodu: {'WŁ' if self.forward_enabled else 'WYŁ'}")
                self.last_mode_switch = time.time()
             return True
        elif key == '\x03' or key.lower() == 'p':
            self.get_logger().info("Wyjście...")
            self.cmd_vel_publisher.publish(Twist()) # Zatrzymaj drona przed wyjściem
            rclpy.shutdown()
            return True
        return False

    def _switch_to_manual(self):
        if self.control_mode == "Autonomous":
            self.control_mode = "Manual"
            self.get_logger().info("Cel utracony lub tryb przerwany - Przełączanie na tryb manualny")
            self.cmd_vel_publisher.publish(Twist()) # Zatrzymaj ruch

    def _display_controls(self):
        self.get_logger().info("\n" + "="*35 +
            "\nSterowanie Dronem Tello:" +
            "\n  W/S: Przód/Tył | A/D: Lewo/Prawo" +
            "\n  R/F: Góra/Dół  | Q/E: Obrót" +
            "\n-----------------------------------" +
            "\n  T/L: Start/Lądowanie" +
            "\n  ,/.: Zmniejsz/Zwiększ prędkość" +
            "\n  SPACJA: Przełącz tryb Manual/Auto" +
            "\n  S (w Auto): Przełącz ruch przód/tył" +
            "\n  P lub CTRL+C: Wyjdź" +
            "\n" + "="*35)

def main(args=None):
    rclpy.init(args=args)
    with KeyboardReader() as key_reader:
        node = TelloControllerNode(key_reader)
        try: rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
        finally:
            node.destroy_node()
    rclpy.shutdown()
    print("\nUstawienia terminala przywrócone.")

if __name__ == '__main__':
    main()