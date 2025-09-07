import time

# Constants
DEFAULT_KP = 0.5
DEFAULT_KI = 0.0
DEFAULT_KD = 0.0
DEFAULT_SETPOINT = 0.0
DEFAULT_SAMPLE_TIME = 0.05
DEFAULT_OUTPUT_LIMITS = (-100, 100)
INITIAL_OUTPUT = 0
MIN_DT = 1e-6  # Minimum delta time to avoid division by zero

class PIDController:
    """Optimized PID controller with anti-windup protection and conditional integration"""
    def __init__(self, kp=DEFAULT_KP, ki=DEFAULT_KI, kd=DEFAULT_KD,
                 setpoint=DEFAULT_SETPOINT, sample_time=DEFAULT_SAMPLE_TIME,
                 output_limits=DEFAULT_OUTPUT_LIMITS):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits
        
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
        self._last_output = 0.0

    def compute(self, current_value):
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            return INITIAL_OUTPUT

        dt = now - self._last_time
        if dt < self.sample_time:
            return self._last_output

        error = self.setpoint - current_value
        d_error = error - self._last_error
        
        # Calculate components
        p_term = self.kp * error
        d_term = self.kd * d_error / max(dt, MIN_DT)
        
        # Conditional integration anti-windup
        output_unclamped = p_term + self._integral + d_term
        output = max(min(output_unclamped, self.output_limits[1]), self.output_limits[0])
        
        # Only integrate if not saturated
        if output == output_unclamped or self.ki == 0:
            self._integral += self.ki * error * dt
        else:
            # Back-calculate integral to prevent windup
            self._integral = output - p_term - d_term

        self._last_error = error
        self._last_time = now
        self._last_output = output
        return output

    def reset(self):
        """Reset controller state"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
        self._last_output = 0.0