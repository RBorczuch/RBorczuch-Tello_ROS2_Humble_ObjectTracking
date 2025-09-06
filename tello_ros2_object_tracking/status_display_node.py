import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState, Temperature
from std_msgs.msg import Int16
import sys
from threading import Lock
import math

def quaternion_to_euler(q_x, q_y, q_z, q_w):
    """ Konwertuje kwaternion na kąty Eulera (w stopniach). """
    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw_z = math.atan2(t3, t4)
    
    return math.degrees(yaw_z), math.degrees(pitch_y), math.degrees(roll_x)

class StatusDisplayNode(Node):
    def __init__(self):
        super().__init__('status_display')
        
        self._telemetry_data = {}
        self._lock = Lock()
        self.last_msg_lines = 0

        # Subskrybuj nowe tematy
        self.create_subscription(BatteryState, 'tello/battery', self.battery_callback, 10)
        self.create_subscription(Temperature, 'tello/temperature', self.temperature_callback, 10)
        self.create_subscription(Imu, 'tello/imu', self.imu_callback, 10)
        self.create_subscription(Int16, 'tello/tof', self.tof_callback, 10)

        self.get_logger().info("Węzeł wyświetlania telemetrii jest gotowy.")
        self.create_timer(0.5, self.display_telemetry)

    # --- Definicje callbacków ---
    def battery_callback(self, msg):
        with self._lock: self._telemetry_data['battery'] = f"{msg.percentage:.0f}%"
    def temperature_callback(self, msg):
        with self._lock: self._telemetry_data['temperature'] = f"{msg.temperature:.1f}°C"
    def tof_callback(self, msg):
        with self._lock: self._telemetry_data['tof'] = f"{msg.data} cm"
    def imu_callback(self, msg):
        with self._lock:
            q = msg.orientation
            yaw, pitch, roll = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self._telemetry_data['orientation'] = f"{pitch:.0f}° / {roll:.0f}° / {yaw:.0f}°"
            acc = msg.linear_acceleration
            self._telemetry_data['acceleration'] = f"{acc.x:.2f} / {acc.y:.2f} / {acc.z:.2f} m/s²"

    def display_telemetry(self):
        with self._lock:
            if self.last_msg_lines > 0:
                sys.stdout.write(f"\x1b[{self.last_msg_lines}A")
                sys.stdout.write("\x1b[J")

            labels = {
                "Stan Baterii": 'battery', "Temperatura": 'temperature',
                "Odległość (ToF)": 'tof',
                "Orientacja (Pitch/Roll/Yaw)": 'orientation',
                "Akceleracja (X/Y/Z)": 'acceleration'
            }
            
            output_string = "======= TELEMETRIA DRONA TELLO (Zoptymalizowana) =======\n"
            for label, key in labels.items():
                value = self._telemetry_data.get(key, 'Czekam na dane...')
                output_string += f"{label:<28}: {value}\n"
            output_string += "========================================================"
            
            print(output_string, flush=True)
            self.last_msg_lines = output_string.count('\n') + 1

def main(args=None):
    rclpy.init(args=args)
    status_display_node = StatusDisplayNode()
    try: rclpy.spin(status_display_node)
    except KeyboardInterrupt: pass
    finally:
        status_display_node.destroy_node()
        rclpy.shutdown()
        print("\nWyłączono wyświetlanie telemetrii.")

if __name__ == '__main__':
    main()