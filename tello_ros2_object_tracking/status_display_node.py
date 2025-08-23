# Copyright 2025 Google LLC.
# SPDX-License-Identifier: Apache-2.0

import rclpy
from rclpy.node import Node
import tkinter as tk
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np

# Import niestandardowych komunikatów ROS2
from tello_ros2_object_tracking.msg import TelloStatus
from tello_ros2_object_tracking.msg import TrackedObject

# ==============================================================================
# SEKCJA 1: Kod zaadaptowany z oryginalnego pliku status_display.py
# ==============================================================================

class StatusDisplayLogic:
    """
    Zawiera całą logikę GUI i rysowania wykresów z oryginalnego pliku StatusDisplay.
    Została przemianowana, aby uniknąć konfliktu nazw z węzłem ROS2.
    Nie przyjmuje już obiektów 'tello' ani 'tracking_data', ponieważ dane
    są dostarczane przez subskrybentów ROS2.
    """
    def __init__(self):
        self.state_data_lock = Lock()
        self.state_data = {}

        self.root = tk.Tk()
        self.root.title("Status Drona (ROS2)")
        self.root.geometry("2000x900")

        # Konfiguracja siatki (grid)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=2)
        self.root.grid_rowconfigure(0, weight=1)

        # Inicjalizacja paneli
        self._initialize_left_panel()
        self._initialize_right_panel()
        self._initialize_3d_panel()

        # Definicja etykiet i buforów
        self.sections, self.units, self.state_labels = self._define_sections_and_units()
        self.buffer = self._initialize_buffers()
        self.plot_data = self._initialize_buffers()
        self.position_data = {"x": [0], "y": [0], "z": [0]}
        
    def _initialize_left_panel(self):
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1); left_frame.columnconfigure(1, weight=1)
        left_frame.rowconfigure(0, weight=1)
        self.left_col_frame = tk.Frame(left_frame); self.left_col_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.right_col_frame = tk.Frame(left_frame); self.right_col_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    def _initialize_right_panel(self):
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _initialize_3d_panel(self):
        right_3d_frame = tk.Frame(self.root)
        right_3d_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.ax_3d.set_title("Odometria (X=Przód, Y=Lewo, Z=Dół)")
        self.ax_3d.set_xlabel("X (cm)"); self.ax_3d.set_ylabel("Y (cm)"); self.ax_3d.set_zlabel("Z (cm)")
        self.ax_3d.invert_zaxis()
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=right_3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _define_sections_and_units(self):
        sections = {
            "Bateria": ["bat"], "Temperatura": ["templ", "temph"], "WiFi": ["wifi"],
            "Czas": ["time"], "Ciśnienie": ["baro"], "Śledzenie": ["status", "score"],
            "Odległość od obiektu": ["dx", "dy", "distance", "angle"],
            "Orientacja": ["pitch", "roll", "yaw"], "Prędkość": ["vgx", "vgy", "vgz"],
            "Wysokość i odległość": ["tof", "h"], "Akcelerometr": ["agx", "agy", "agz"],
        }
        units = {
            "bat": "%", "templ": "°C", "temph": "°C", "wifi": "dBm", "pitch": "°", "roll": "°", "yaw": "°",
            "vgx": "cm/s", "vgy": "cm/s", "vgz": "cm/s", "time": "s", "tof": "cm", "h": "cm",
            "baro": "hPa", "agx": "m/s²", "agy": "m/s²", "agz": "m/s²", "dx": "px", "dy": "px",
            "distance": "px", "angle": "°", "score": ""
        }
        state_labels = {}
        left_sections = ["Bateria", "Temperatura", "WiFi", "Czas", "Ciśnienie", "Śledzenie"]
        right_sections = ["Orientacja", "Prędkość", "Wysokość i odległość", "Akcelerometr", "Odległość od obiektu"]

        for section_name, frame_parent in [(left_sections, self.left_col_frame), (right_sections, self.right_col_frame)]:
            for section in section_name:
                frame = tk.LabelFrame(frame_parent, text=section, font=("Arial", 12))
                frame.pack(fill="both", expand=True, padx=5, pady=5)
                for key in sections[section]:
                    label = tk.Label(frame, text=f"{key}: -- {units.get(key, '')}", font=("Arial", 12))
                    label.pack(anchor="w", padx=10, pady=2)
                    state_labels[key] = label
        return sections, units, state_labels

    def _initialize_buffers(self):
        return {
            "local_time": [], "vgx": [], "vgy": [], "vgz": [], "pitch": [], "roll": [], "yaw": [], "h": [],
            "agx": [], "agy": [], "agz": [], "dx": [], "dy": [], "distance": [], "angle": [],
        }

    def _compensate_gravity(self, state):
        pitch_rad, roll_rad = np.radians(state.get("pitch", 0)), np.radians(state.get("roll", 0))
        agx, agy, agz = state.get("agx", 0), state.get("agy", 0), state.get("agz", 0)
        
        # Przybliżona kompensacja grawitacji. Właściwa wymagałaby macierzy obrotu.
        g = 9.81
        state["agx"] = agx - g * np.sin(pitch_rad)
        state["agy"] = agy + g * np.sin(roll_rad) * np.cos(pitch_rad)
        state["agz"] = agz + g * np.cos(roll_rad) * np.cos(pitch_rad)

    def update_labels(self):
        with self.state_data_lock:
            state_copy = self.state_data.copy()
        for key, value in state_copy.items():
            if key in self.state_labels:
                unit = self.units.get(key, "")
                text = f"{key}: {value:.2f} {unit}" if isinstance(value, float) else f"{key}: {value} {unit}"
                self.state_labels[key].config(text=text)
        self.root.after(100, self.update_labels)

    def _buffer_data(self, state):
        current_time = time.time() - self.start_time
        self.buffer["local_time"].append(current_time)
        for key in self.buffer.keys():
            if key != "local_time":
                self.buffer[key].append(state.get(key, 0))

    def _update_position(self, state):
        # Ta funkcja pozostaje taka sama, używając danych ze stanu
        # ... (skopiowano z oryginalnego pliku)
        if not all(k in state for k in ["vgx", "vgy", "vgz", "pitch", "roll", "yaw"]): return
        v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) * 0.01
        roll_rad, pitch_rad, yaw_rad = np.radians(state["roll"]), np.radians(state["pitch"]), np.radians(state["yaw"])
        R = self._get_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
        v_world = R @ v_body
        dt = 0.05
        self.position_data["x"].append(self.position_data["x"][-1] + v_world[0] * dt * 100)
        self.position_data["y"].append(self.position_data["y"][-1] + v_world[1] * dt * 100)
        self.position_data["z"].append(self.position_data["z"][-1] + v_world[2] * dt * 100)

    def _get_rotation_matrix(self, roll, pitch, yaw):
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return R_yaw @ R_pitch @ R_roll

    def update_plots(self, _):
        self._update_2d_plots()
        self._update_3d_plot()

    def _update_2d_plots(self):
        if not self.buffer["local_time"]: return
        for key in self.buffer: self.plot_data[key].extend(self.buffer[key]); self.buffer[key] = []
        latest_time = self.plot_data["local_time"][-1]
        indices = [i for i, t in enumerate(self.plot_data["local_time"]) if t >= latest_time - 10]
        for key in self.plot_data: self.plot_data[key] = [self.plot_data[key][i] for i in indices]

        axes_map = {
            self.ax1: (["vgx", "vgy", "vgz"], "Prędkość w czasie"),
            self.ax2: (["pitch", "roll", "yaw"], "Orientacja w czasie"),
            self.ax3: (["h"], "Wysokość w czasie"),
            self.ax4: (["agx", "agy", "agz"], "Przyspieszenie w czasie"),
            self.ax5: (["dx", "dy", "distance"], "Odległość od obiektu w czasie"),
            self.ax6: (["angle"], "Kąt w czasie"),
        }
        for ax, (keys, title) in axes_map.items():
            ax.clear(); ax.set_title(title)
            for key in keys: ax.plot(self.plot_data["local_time"], self.plot_data[key], label=key)
            ax.legend()
        self.canvas.draw()

    def _update_3d_plot(self):
        self.ax_3d.clear()
        self.ax_3d.plot(self.position_data["x"], self.position_data["y"], self.position_data["z"], label="Trajektoria")
        # Dodaj wektory orientacji, jeśli chcesz
        self.ax_3d.set_title("Odometria"); self.ax_3d.set_xlabel("X (cm)"); self.ax_3d.set_ylabel("Y (cm)"); self.ax_3d.set_zlabel("Z (cm)")
        self.ax_3d.invert_zaxis(); self.ax_3d.legend()
        self.canvas_3d.draw()

# ==============================================================================
# SEKCJA 2: Węzeł ROS2
# ==============================================================================

class StatusDisplayNode(Node, StatusDisplayLogic):
    def __init__(self):
        # Inicjalizuj oba rodziców: Node i GUI
        Node.__init__(self, 'status_display')
        StatusDisplayLogic.__init__(self)

        # Subskrybenci
        self.create_subscription(TelloStatus, 'tello/status', self.tello_status_callback, 10)
        self.create_subscription(TrackedObject, 'object_tracking/data', self.tracking_data_callback, 10)

        self.get_logger().info("Węzeł Status Display jest gotowy.")

    def tello_status_callback(self, msg: TelloStatus):
        """Odbiera dane telemetryczne drona i aktualizuje stan."""
        with self.state_data_lock:
            self.state_data['bat'] = msg.battery
            self.state_data['templ'] = msg.temperature_low
            self.state_data['temph'] = msg.temperature_high
            self.state_data['time'] = msg.flight_time
            self.state_data['baro'] = msg.barometer
            self.state_data['pitch'] = msg.pitch
            self.state_data['roll'] = msg.roll
            self.state_data['yaw'] = msg.yaw
            self.state_data['vgx'] = msg.vgx
            self.state_data['vgy'] = msg.vgy
            self.state_data['vgz'] = msg.vgz
            self.state_data['tof'] = msg.tof_distance
            self.state_data['h'] = msg.height
            self.state_data['agx'] = msg.agx
            self.state_data['agy'] = msg.agy
            self.state_data['agz'] = msg.agz

        # Przetwarzaj i buforuj nowe dane
        state_copy = self.state_data.copy()
        self._compensate_gravity(state_copy)
        self._buffer_data(state_copy)
        self._update_position(state_copy)

    def tracking_data_callback(self, msg: TrackedObject):
        """Odbiera dane śledzenia obiektu i aktualizuje stan."""
        with self.state_data_lock:
            self.state_data['status'] = msg.status
            self.state_data['score'] = msg.score
            self.state_data['dx'] = msg.dx
            self.state_data['dy'] = msg.dy
            self.state_data['distance'] = msg.distance
            self.state_data['angle'] = msg.angle

    def run_gui(self):
        """Uruchamia GUI i animację Matplotlib."""
        self.start_time = time.time()
        self.update_labels() # Rozpoczyna cykliczną aktualizację etykiet
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, interval=200, cache_frame_data=False
        )
        self.root.mainloop() # Uruchamia pętlę główną Tkinter

def main(args=None):
    rclpy.init(args=args)
    status_display_node = StatusDisplayNode()
    
    # Uruchom rclpy.spin w osobnym wątku, aby nie blokować pętli Tkinter
    spin_thread = Thread(target=rclpy.spin, args=(status_display_node,), daemon=True)
    spin_thread.start()

    try:
        # Główny wątek uruchamia GUI
        status_display_node.run_gui()
    except KeyboardInterrupt:
        status_display_node.get_logger().info('Przerwano przez użytkownika (KeyboardInterrupt)')
    finally:
        status_display_node.get_logger().info('Zamykanie węzła Status Display...')
        if rclpy.ok():
            status_display_node.destroy_node()
            rclpy.shutdown()
        spin_thread.join() # Poczekaj na zakończenie wątku spin

if __name__ == '__main__':
    main()