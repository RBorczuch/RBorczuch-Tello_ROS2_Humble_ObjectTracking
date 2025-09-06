# tello_ros2_object_tracking/PID.py

import time

class PIDController:
    """
    An optimized PID controller with anti-windup protection and conditional integration.
    """
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.0,
                 setpoint: float = 0.0, sample_time: float = 0.05,
                 output_limits: tuple[float, float] = (-100.0, 100.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits
        
        self.min_output, self.max_output = self.output_limits
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()
        self._last_output = 0.0

    def compute(self, current_value: float) -> float:
        """Calculate the PID output for a given current value."""
        now = time.monotonic()
        dt = now - self._last_time
        
        if dt < self.sample_time:
            return self._last_output

        error = self.setpoint - current_value
        d_error = error - self._last_error
        
        p_term = self.kp * error
        d_term = self.kd * d_error / dt if dt > 1e-6 else 0.0
        
        # Calculate potential integral term before clamping
        potential_integral = self._integral + (self.ki * error * dt)

        # Calculate pre-clamped output
        output = p_term + potential_integral + d_term

        # Clamp output to defined limits
        clamped_output = max(self.min_output, min(output, self.max_output))

        # Anti-windup: only update the integral if the output is not saturated
        if self.min_output < output < self.max_output:
            self._integral = potential_integral
        
        self._last_error = error
        self._last_time = now
        self._last_output = clamped_output
        return clamped_output

    def reset(self) -> None:
        """Resets the controller's internal state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()
        self._last_output = 0.0