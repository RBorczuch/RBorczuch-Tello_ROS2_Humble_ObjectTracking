

import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from tello_ros2_object_tracking.msg import TrackedObject
from .PID import PIDController

# Importy do obsługi terminala bez zewnętrznych bibliotek
import sys
import select
import tty
import termios

# ==============================================================================
# SEKCJA 1: Klasa pomocnicza do odczytu klawiatury
# ==============================================================================

class KeyboardReader:
    """
    Menedżer kontekstu do nieblokującego odczytu pojedynczych znaków
    z terminala. Zapewnia przywrócenie ustawień terminala po zakończeniu.
    """
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        # Przełącz terminal w tryb "surowy" (raw mode)
        tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Przywróć oryginalne ustawienia terminala
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def read_key(self):
        """Odczytuje pojedynczy klawisz, jeśli jest dostępny, w przeciwnym razie zwraca None."""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

# ==============================================================================
# SEKCJA 2: Logika węzła kontrolera
# ==============================================================================

# Stałe z oryginalnego pliku
LOOP_SLEEP = 0.05
MODE_SWITCH_SLEEP = 0.3
ERROR_THRESHOLD = 20
DEAD_ZONE = 5
PID_CONFIG = {
    'yaw': {'kp': 0.2, 'ki': 0.01, 'kd': 0.0},
    'vertical': {'kp': 0.2, 'ki': 0.01, 'kd': 0.0},
    'forward': {'kp': 0.4, 'ki': 0.0, 'kd': 0.0, 'setpoint': 120}
}
VELOCITY_CONFIG = { 'initial': 30, 'min': 10, 'max': 100, 'step': 5 }

class TelloControllerNode(Node):
    def __init__(self, key_reader: KeyboardReader):
        super().__init__('tello_controller')
        self.get_logger().info("Inicjalizacja węzła Tello Controller...")
        
        # Przechowaj instancję czytnika klawiatury
        self.key_reader = key_reader

        # Wydawcy
        self.cmd_vel_publisher = self.create_publisher(Twist, 'tello/cmd_vel', 10)
        self.control_publisher = self.create_publisher(String, 'tello/control', 10)

        # Subskrybenci
        self.tracking_subscriber = self.create_subscription(
            TrackedObject, 'object_tracking/data', self.tracking_callback, 10)

        # Inicjalizacja stanu i PID
        self.pids = {
            'yaw': PIDController(**PID_CONFIG['yaw']),
            'vertical': PIDController(**PID_CONFIG['vertical']),
            'forward': PIDController(**PID_CONFIG['forward'])
        }
        self.velocity = VELOCITY_CONFIG['initial']
        self.last_mode_switch = 0
        
        self.control_mode = "Manual"
        self.forward_enabled = False
        self.tracking_status = "Lost"
        self.tracking_data = None

        self.timer = self.create_timer(LOOP_SLEEP, self.control_loop)
        self._display_controls()
        self.get_logger().info("Węzeł Tello Controller jest gotowy. Terminal jest w trybie odczytu klawiszy.")
        
    def tracking_callback(self, msg):
        self.tracking_data = msg
        self.tracking_status = msg.status

    def control_loop(self):
        # Odczytaj pojedynczy klawisz (jeśli został naciśnięty)
        key = self.key_reader.read_key()

        if key:
            # Obsługa poleceń, które nie są ruchem
            if self._handle_takeoff_and_landing(key): pass
            elif self._handle_mode_switches(key): pass
            elif self._handle_velocity_adjustment(key): pass
            elif key == '\x03' or key.lower() == 'p': # CTRL+C lub 'p' do wyjścia
                self.get_logger().info("Wykryto polecenie wyjścia. Zamykanie...")
                rclpy.shutdown()
                return

        # Zdecyduj o trybie sterowania
        if self.control_mode == "Manual":
            self._process_manual_control(key)
        else: # Tryb autonomiczny
            # W trybie autonomicznym, jeśli naciśnięto jakikolwiek klawisz ruchu, przełącz na manual
            if key and key.lower() in 'wasdrfqe':
                 self.get_logger().info("Wykryto ręczne sterowanie, przełączanie na tryb manualny.")
                 self._switch_to_manual()
            else:
                 self._process_autonomous_control()

    def _process_manual_control(self, key):
        cmd = Twist()
        vel = float(self.velocity)
        
        key_map = {
            'w': ('linear', 'x', vel),   's': ('linear', 'x', -vel),
            'a': ('linear', 'y', vel),   'd': ('linear', 'y', -vel),
            'r': ('linear', 'z', vel),   'f': ('linear', 'z', -vel),
            'q': ('angular', 'z', -vel), 'e': ('angular', 'z', vel),
        }
        
        if key and key.lower() in key_map:
            axis_type, axis_name, value = key_map[key.lower()]
            setattr(getattr(cmd, axis_type), axis_name, value)

        # Publikujemy komendę przy każdym naciśnięciu klawisza.
        # Jeśli klawisz nie został naciśnięty (key is None), wysyłana jest pusta komenda Twist,
        # co efektywnie zatrzymuje drona.
        self.cmd_vel_publisher.publish(cmd)

    def _process_autonomous_control(self):
        if self.tracking_status == "Lost" or self.tracking_data is None:
            self._switch_to_manual()
            return
            
        dx = self.tracking_data.dx
        dy = self.tracking_data.dy
        roi_height = self.tracking_data.roi_height

        x_vel = self._calculate_pid_output('forward', roi_height) if self.forward_enabled else 0.0
        yaw_vel = self._calculate_pid_output('yaw', dx, ERROR_THRESHOLD)
        z_vel = self._calculate_pid_output('vertical', dy, ERROR_THRESHOLD)

        cmd = Twist()
        cmd.linear.x = float(x_vel)
        cmd.linear.z = float(z_vel)
        cmd.angular.z = float(yaw_vel)
        
        self.cmd_vel_publisher.publish(cmd)

    def _calculate_pid_output(self, pid_key, error, threshold=0):
        if abs(error) < max(threshold, DEAD_ZONE):
            self.pids[pid_key].reset()
            return 0.0
        return self.pids[pid_key].compute(error)

    def _handle_velocity_adjustment(self, key):
        if key == ',': # Znak <
            self.velocity = max(VELOCITY_CONFIG['min'], self.velocity - VELOCITY_CONFIG['step'])
            self.get_logger().info(f"Prędkość: {self.velocity} cm/s")
            return True
        elif key == '.': # Znak >
            self.velocity = min(VELOCITY_CONFIG['max'], self.velocity + VELOCITY_CONFIG['step'])
            self.get_logger().info(f"Prędkość: {self.velocity} cm/s")
            return True
        return False

    def _handle_mode_switches(self, key):
        now = time.time()
        if now - self.last_mode_switch < MODE_SWITCH_SLEEP: return False

        if key == ' ':
            self.control_mode = "Autonomous" if self.control_mode == "Manual" else "Manual"
            self.get_logger().info(f"Tryb: {self.control_mode}")
            self.last_mode_switch = now
            return True

        if key.lower() == 's' and self.control_mode == "Autonomous":
            self.forward_enabled = not self.forward_enabled
            self.get_logger().info(f"Celowanie do przodu: {'WŁ' if self.forward_enabled else 'WYŁ'}")
            self.last_mode_switch = now
            return True
        return False

    def _switch_to_manual(self):
        self.control_mode = "Manual"
        self.get_logger().info("Cel utracony - Przełączanie na tryb manualny")
        self.cmd_vel_publisher.publish(Twist()) # Zatrzymaj ruch

    def _handle_takeoff_and_landing(self, key):
        if key.lower() == 't':
            self.get_logger().info("Inicjacja startu")
            self.control_publisher.publish(String(data='takeoff'))
            return True
        elif key.lower() == 'l':
            self.get_logger().info("Inicjacja lądowania")
            self.control_publisher.publish(String(data='land'))
            return True
        return False

    def _display_controls(self):
        controls = [
            "\n" + "="*30,
            "Sterowanie Dronem Tello:",
            "  W/S: Przód/Tył", "  A/D: Lewo/Prawo", "  R/F: Góra/Dół",
            "  Q/E: Obrót Lewo/Prawo",
            "------------------------------",
            "  T/L: Start/Lądowanie",
            "  ,/.: Zmniejsz/Zwiększ prędkość",
            "  SPACJA: Przełącz tryb Manualny/Autonomiczny",
            "  S (w trybie Auto): Przełącz celowanie do przodu",
            "------------------------------",
            "  P lub CTRL+C: Wyjdź",
            "="*30
        ]
        self.get_logger().info("\n".join(controls))

def main(args=None):
    rclpy.init(args=args)
    
    # Użyj menedżera kontekstu, aby bezpiecznie zarządzać stanem terminala
    with KeyboardReader() as key_reader:
        tello_controller_node = TelloControllerNode(key_reader)
        try:
            rclpy.spin(tello_controller_node)
        except KeyboardInterrupt:
            pass
        finally:
            tello_controller_node.destroy_node()
            
    # Po wyjściu z bloku 'with', stan terminala jest automatycznie przywracany
    rclpy.shutdown()
    print("\nUstawienia terminala przywrócone. Do widzenia!")

if __name__ == '__main__':
    main()