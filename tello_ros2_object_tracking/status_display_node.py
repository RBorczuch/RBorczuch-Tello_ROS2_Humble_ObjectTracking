# tello_ros2_object_tracking/status_display_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState, Temperature
from std_msgs.msg import Int16, String
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
    real-time drone telemetry and tracking data.
    """
    def __init__(self):
        super().__init__('status_display')
        
        self.state_data = {}
        self.state_lock = Lock()
        self.start_time = time.time()

        # Buffers for plotting
        self.buffer = self._initialize_buffers()
        self.plot_data = self._initialize_buffers()
        self.position_data = {"x": [0], "y": [0], "z": [0]}

        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("Tello ROS 2 Dashboard")
        self.root.geometry("2000x900")
        self._setup_gui_layout()
        self.units, self.state_labels = self._define_sections_and_units()
        
        # Subscribers
        self._setup_subscribers()

    def _setup_gui_layout(self):
        """Configures the main Tkinter window grid and frames."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left panel for text data
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        self.left_col_frame = tk.Frame(left_frame)
        self.left_col_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.right_col_frame = tk.Frame(left_frame)
        self.right_col_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Center panel for 2D plots
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2)
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel for 3D odometry plot
        right_3d_frame = tk.Frame(self.root)
        right_3d_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=right_3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _define_sections_and_units(self):
        """Creates and places all the text labels in the GUI."""
        sections = {
            "Battery": ["bat"], "Temperature": ["temp"], "Distance (ToF)": ["tof"],
            "Tracking": ["status", "score"], "Object Offset": ["dx", "dy"],
            "Orientation": ["pitch", "roll", "yaw"], "Acceleration": ["agx", "agy", "agz"]
        }
        units = {
            "bat": "%", "temp": "°C", "tof": "cm", "pitch": "°", "roll": "°", "yaw": "°",
            "agx": "m/s²", "agy": "m/s²", "agz": "m/s²", "dx": "px", "dy": "px", "score": ""
        }
        labels = {}
        
        # Distribute sections between two columns
        left_cols = ["Battery", "Temperature", "Distance (ToF)", "Tracking"]
        right_cols = ["Orientation", "Acceleration", "Object Offset"]

        for section_name in left_cols:
            frame = tk.LabelFrame(self.left_col_frame, text=section_name, font=("Arial", 12))
            frame.pack(fill="x", padx=5, pady=5)
            for key in sections[section_name]:
                label = tk.Label(frame, text=f"{key}: --", font=("Arial", 11))
                label.pack(anchor="w", padx=10, pady=2)
                labels[key] = label
        
        for section_name in right_cols:
            frame = tk.LabelFrame(self.right_col_frame, text=section_name, font=("Arial", 12))
            frame.pack(fill="x", padx=5, pady=5)
            for key in sections[section_name]:
                label = tk.Label(frame, text=f"{key}: --", font=("Arial", 11))
                label.pack(anchor="w", padx=10, pady=2)
                labels[key] = label

        return units, labels

    def _initialize_buffers(self):
        """Initializes dictionaries to buffer data for plotting."""
        return { "local_time": [], "pitch": [], "roll": [], "yaw": [], "agx": [], "agy": [], "agz": [], "dx": [], "dy": [] }

    def _setup_subscribers(self):
        """Sets up all ROS 2 subscribers."""
        self.create_subscription(BatteryState, 'tello/battery', lambda msg: self._update_state_value('bat', msg.percentage), 10)
        self.create_subscription(Temperature, 'tello/temperature', lambda msg: self._update_state_value('temp', msg.temperature), 10)
        self.create_subscription(Int16, 'tello/tof', lambda msg: self._update_state_value('tof', msg.data), 10)
        self.create_subscription(Imu, 'tello/imu', self._imu_callback, 10)
        self.create_subscription(String, 'object_tracking/status', lambda msg: self._update_state_value('status', msg.data), 10)
        self.create_subscription(Vector3, 'object_tracking/control_error', self._tracking_callback, 10)
        self.get_logger().info("Subscribers are set up.")
    
    def _update_state_value(self, key, value):
        with self.state_lock:
            self.state_data[key] = value

    def _imu_callback(self, msg: Imu):
        with self.state_lock:
            q = msg.orientation
            yaw, pitch, roll = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self.state_data.update({'pitch': pitch, 'roll': roll, 'yaw': yaw})
            acc = msg.linear_acceleration
            self.state_data.update({'agx': acc.x, 'agy': acc.y, 'agz': acc.z})

    def _tracking_callback(self, msg: Vector3):
        with self.state_lock:
            self.state_data.update({'dx': msg.x, 'dy': msg.y})

    def _update_gui_labels(self):
        """Periodically updates the text labels in the Tkinter window."""
        with self.state_lock:
            state_copy = self.state_data.copy()

        for key, value in state_copy.items():
            if key in self.state_labels:
                unit = self.units.get(key, "")
                text = f"{key}: {value:.2f} {unit}" if isinstance(value, float) else f"{key}: {value} {unit}"
                self.state_labels[key].config(text=text)

        self.root.after(100, self._update_gui_labels) # Schedule next update

    def _buffer_data_for_plots(self):
        """Periodically buffers the current state for plotting."""
        with self.state_lock:
            state_copy = self.state_data.copy()
        
        current_time = time.time() - self.start_time
        self.buffer["local_time"].append(current_time)
        for key in self.buffer.keys():
            if key != "local_time":
                self.buffer[key].append(state_copy.get(key, 0))
        
        # Simple odometry update
        # This is a very rough estimation and should not be used for navigation
        if all(k in state_copy for k in ['pitch', 'roll', 'yaw']):
             # Simplified: assumes velocity is proportional to pitch/roll, which is not accurate
            dt = 0.1
            vx = -state_copy.get('pitch', 0) * dt
            vy = state_copy.get('roll', 0) * dt
            vz = 0 # Cannot be determined from this data
            self.position_data["x"].append(self.position_data["x"][-1] + vx)
            self.position_data["y"].append(self.position_data["y"][-1] + vy)
            self.position_data["z"].append(self.position_data["z"][-1] + vz)

        self.root.after(100, self._buffer_data_for_plots)

    def _update_plots(self, _):
        """Matplotlib animation callback to redraw plots."""
        # Move buffered data to plot_data and trim to a 10-second window
        for key in self.buffer:
            self.plot_data[key].extend(self.buffer[key])
            self.buffer[key].clear()
        
        if not self.plot_data["local_time"]: return

        time_window = self.plot_data["local_time"][-1] - 10
        indices = [i for i, t in enumerate(self.plot_data["local_time"]) if t >= time_window]
        for key in self.plot_data:
            self.plot_data[key] = [self.plot_data[key][i] for i in indices]

        # Update 2D plots
        axes_map = {
            self.ax1: (["pitch", "roll", "yaw"], "Orientation (°/s)"),
            self.ax2: (["agx", "agy", "agz"], "Acceleration (m/s²)"),
            self.ax3: (["dx"], "Object X Offset (px)"),
            self.ax4: (["dy"], "Object Y Offset (px)"),
        }
        for ax, (keys, title) in axes_map.items():
            ax.clear()
            ax.set_title(title)
            for key in keys:
                ax.plot(self.plot_data["local_time"], self.plot_data[key], label=key)
            ax.legend(loc='upper right')
        self.canvas.draw()

        # Update 3D plot
        self.ax_3d.clear()
        self.ax_3d.plot(self.position_data["x"], self.position_data["y"], self.position_data["z"], label="Trajectory")
        self.ax_3d.set_title("Estimated Odometry")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.legend()
        self.canvas_3d.draw()

    def run_gui(self):
        """Starts the GUI and all periodic updates."""
        self._update_gui_labels()
        self._buffer_data_for_plots()
        
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=500, cache_frame_data=False
        )
        
        self.get_logger().info("Starting Tkinter main loop.")
        self.root.mainloop()
        self.get_logger().info("Tkinter main loop has finished.")

def main(args=None):
    rclpy.init(args=args)
    status_display_node = StatusDisplayNode()
    
    # Run rclpy.spin() in a separate thread to not block the GUI
    spin_thread = Thread(target=rclpy.spin, args=(status_display_node,), daemon=True)
    spin_thread.start()

    try:
        status_display_node.run_gui()
    except (KeyboardInterrupt, SystemExit):
        status_display_node.get_logger().info('GUI closed.')
    finally:
        status_display_node.get_logger().info('Shutting down ROS node.')
        if rclpy.ok():
            status_display_node.destroy_node()
            rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()