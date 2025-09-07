# tello_ros2_object_tracking/status_display_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState, Temperature
from std_msgs.msg import Int16, String, Float32
from geometry_msgs.msg import Vector3
import tkinter as tk
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
from .utils import quaternion_to_euler

class StatusDisplayNode(Node):
    """
    Creates a comprehensive GUI using Tkinter and Matplotlib to display
    real-time drone telemetry and tracking data received from ROS 2 topics.
    """
    def __init__(self):
        super().__init__('status_display')
        
        self.state_data = {}
        self.state_lock = Lock()
        self.start_time = time.time()

        self.buffer = self._initialize_buffers()
        self.plot_data = self._initialize_buffers()
        self.position_data = {"x": [0], "y": [0], "z": [0]}

        self.root = tk.Tk()
        self.root.title("Tello ROS 2 Dashboard")
        self.root.geometry("2000x900")
        self._setup_gui_layout()
        self.units, self.state_labels = self._define_sections_and_units()
        
        self._setup_subscribers()

    def _setup_gui_layout(self):
        self.root.grid_columnconfigure(0, weight=1); self.root.grid_columnconfigure(1, weight=2); self.root.grid_columnconfigure(2, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        left_frame = tk.Frame(self.root); left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1); left_frame.columnconfigure(1, weight=1)
        self.left_col_frame = tk.Frame(left_frame); self.left_col_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.right_col_frame = tk.Frame(left_frame); self.right_col_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        right_frame = tk.Frame(self.root); right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.fig, self.axes = plt.subplots(3, 2); self.fig.tight_layout(pad=3)
        (self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6) = self.axes
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_3d_frame = tk.Frame(self.root); right_3d_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.fig_3d = plt.figure(); self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=right_3d_frame); self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _define_sections_and_units(self):
        sections = {
            "Battery": ["bat"], "Temperature": ["temp"], "Time": ["time"], "Pressure": ["baro"],
            "Tracking": ["status", "score"], "Object Distance": ["dx", "dy", "distance", "angle"],
            "Orientation": ["pitch", "roll", "yaw"], "Velocity": ["vgx", "vgy", "vgz"],
            "Height/Distance": ["tof", "h"], "Accelerometer": ["agx", "agy", "agz"]
        }
        units = {
            "bat": "%", "temp": "°C", "time": "s", "baro": "hPa", "pitch": "°", "roll": "°", "yaw": "°",
            "vgx": "cm/s", "vgy": "cm/s", "vgz": "cm/s", "tof": "cm", "h": "cm", "agx": "m/s²", "agy": "m/s²", "agz": "m/s²",
            "dx": "px", "dy": "px", "distance": "px", "angle": "°", "score": ""
        }
        labels = {}
        left_cols = ["Battery", "Temperature", "Time", "Pressure", "Tracking"]
        right_cols = ["Orientation", "Velocity", "Height/Distance", "Accelerometer", "Object Distance"]

        for col_frame, col_sections in [(self.left_col_frame, left_cols), (self.right_col_frame, right_cols)]:
            for section_name in col_sections:
                frame = tk.LabelFrame(col_frame, text=section_name, font=("Arial", 12))
                frame.pack(fill="x", padx=5, pady=5)
                for key in sections[section_name]:
                    label = tk.Label(frame, text=f"{key}: --", font=("Arial", 11)); label.pack(anchor="w", padx=10, pady=2)
                    labels[key] = label
        return units, labels

    def _initialize_buffers(self):
        return {k: [] for k in ["local_time", "vgx", "vgy", "vgz", "pitch", "roll", "yaw", "h",
                                 "agx", "agy", "agz", "dx", "dy", "distance", "angle"]}

    def _setup_subscribers(self):
        self.create_subscription(BatteryState, 'tello/battery', lambda msg: self._update_state('bat', msg.percentage), 10)
        self.create_subscription(Temperature, 'tello/temperature', lambda msg: self._update_state('temp', msg.temperature), 10)
        self.create_subscription(Int16, 'tello/tof', lambda msg: self._update_state('tof', msg.data), 10)
        self.create_subscription(Int16, 'tello/height', lambda msg: self._update_state('h', msg.data), 10)
        self.create_subscription(Float32, 'tello/barometer', lambda msg: self._update_state('baro', msg.data), 10)
        self.create_subscription(Int16, 'tello/flight_time', lambda msg: self._update_state('time', msg.data), 10)
        self.create_subscription(Imu, 'tello/imu', self._imu_callback, 10)
        self.create_subscription(Vector3, 'tello/velocity', self._velocity_callback, 10)
        self.create_subscription(String, 'object_tracking/status', lambda msg: self._update_state('status', msg.data), 10)
        self.create_subscription(Vector3, 'object_tracking/control_error', lambda msg: self._update_state_dict({'dx': msg.x, 'dy': msg.y}), 10)
        self.create_subscription(Vector3, 'object_tracking/info', lambda msg: self._update_state_dict({'distance': msg.x, 'angle': msg.y, 'score': msg.z}), 10)
    
    def _update_state(self, key, value):
        with self.state_lock: self.state_data[key] = value

    def _update_state_dict(self, data: dict):
        with self.state_lock: self.state_data.update(data)

    def _imu_callback(self, msg: Imu):
        q = msg.orientation
        yaw, pitch, roll = quaternion_to_euler(q.x, q.y, q.z, q.w)
        acc = msg.linear_acceleration
        self._update_state_dict({'pitch': pitch, 'roll': roll, 'yaw': yaw, 'agx': acc.x, 'agy': acc.y, 'agz': acc.z})

    def _velocity_callback(self, msg: Vector3):
        self._update_state_dict({'vgx': msg.x, 'vgy': msg.y, 'vgz': msg.z})

    def _update_gui_labels(self):
        with self.state_lock: state_copy = self.state_data.copy()
        for key, value in state_copy.items():
            if key in self.state_labels:
                unit = self.units.get(key, "")
                text = f"{key}: {value:.2f} {unit}" if isinstance(value, float) else f"{key}: {value} {unit}"
                self.state_labels[key].config(text=text)
        self.root.after(100, self._update_gui_labels)

    def _buffer_data_for_plots(self):
        with self.state_lock: state_copy = self.state_data.copy()
        
        current_time = time.time() - self.start_time
        self.buffer["local_time"].append(current_time)
        for key in self.buffer.keys():
            if key != "local_time": self.buffer[key].append(state_copy.get(key, 0))
        
        self._update_position(state_copy)
        self.root.after(100, self._buffer_data_for_plots)

    def _update_position(self, state):
        required = ["vgx", "vgy", "vgz", "pitch", "roll", "yaw"]
        if not all(k in state for k in required): return
        
        v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) * 0.01 # to m/s
        roll, pitch, yaw = np.radians(state["roll"]), np.radians(state["pitch"]), np.radians(state["yaw"])
        R = self._get_rotation_matrix(roll, pitch, yaw)
        v_world = R @ v_body
        dt = 0.1 # Assumed interval of this method
        
        self.position_data["x"].append(self.position_data["x"][-1] + v_world[0] * dt * 100) # back to cm
        self.position_data["y"].append(self.position_data["y"][-1] + v_world[1] * dt * 100)
        self.position_data["z"].append(self.position_data["z"][-1] + v_world[2] * dt * 100)

    def _update_plots(self, _):
        for key in self.buffer:
            self.plot_data[key].extend(self.buffer[key])
            self.buffer[key].clear()
        if not self.plot_data["local_time"]: return

        time_window = self.plot_data["local_time"][-1] - 10
        indices = [i for i, t in enumerate(self.plot_data["local_time"]) if t >= time_window]
        for key in self.plot_data: self.plot_data[key] = [self.plot_data[key][i] for i in indices]

        # Update 2D plots
        axes_map = {
            self.ax1: (["vgx", "vgy", "vgz"], "Velocity (cm/s)"),
            self.ax2: (["pitch", "roll", "yaw"], "Orientation (°)"),
            self.ax3: (["h"], "Height (cm)"),
            self.ax4: (["agx", "agy", "agz"], "Acceleration (m/s²)"),
            self.ax5: (["dx", "dy", "distance"], "Object Offset (px)"),
            self.ax6: (["angle"], "Object Angle (°)"),
        }
        for ax, (keys, title) in axes_map.items():
            ax.clear(); ax.set_title(title)
            for key in keys: ax.plot(self.plot_data["local_time"], self.plot_data[key], label=key)
            ax.legend(loc='upper right')
        self.canvas.draw()

        # Update 3D plot
        self.ax_3d.clear()
        self.ax_3d.plot(self.position_data["x"], self.position_data["y"], self.position_data["z"], label="Trajectory")
        self.ax_3d.set_title("Estimated Odometry"); self.ax_3d.set_xlabel("X (cm)"); self.ax_3d.set_ylabel("Y (cm)"); self.ax_3d.set_zlabel("Z (cm)")
        self.ax_3d.invert_zaxis()
        self.ax_3d.legend()
        self.canvas_3d.draw()

    def _get_rotation_matrix(self, roll, pitch, yaw):
        R_r = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        R_p = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        R_y = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        return R_y @ R_p @ R_r

    def run_gui(self):
        self._update_gui_labels()
        self._buffer_data_for_plots()
        self.ani = animation.FuncAnimation(self.fig, self._update_plots, interval=500, cache_frame_data=False)
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    status_display_node = StatusDisplayNode()
    spin_thread = Thread(target=rclpy.spin, args=(status_display_node,), daemon=True)
    spin_thread.start()
    try: status_display_node.run_gui()
    except (KeyboardInterrupt, SystemExit): pass
    finally:
        if rclpy.ok():
            status_display_node.destroy_node()
            rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()