import numpy as np

class KalmanFilter:
    """A lightweight, high-performance 1D/Multi-Dimensional Kalman Filter."""
    def __init__(self, process_noise=1e-4, measurement_noise=1e-3, error_cov=1.0):
        self.q = process_noise  # Process noise covariance
        self.r = measurement_noise  # Measurement noise covariance
        self.x = None  # Estimated state
        self.p = error_cov  # Error covariance/uncertainty

    def update(self, measurement):
        if self.x is None:
            if isinstance(measurement, (list, tuple, np.ndarray)):
                self.x = np.array(measurement, dtype=np.float32)
                self.p = np.eye(len(self.x)) * self.p
            else:
                self.x = float(measurement)
                self.p = float(self.p)
            return self.x
            
        # Vector case
        if isinstance(self.x, np.ndarray):
            meas = np.array(measurement, dtype=np.float32)
            # Predict step (State transition matrix is identity, control input is zero)
            p_pred = self.p + np.eye(len(self.x)) * self.q
            # Update step
            k = p_pred @ np.linalg.inv(p_pred + np.eye(len(self.x)) * self.r)  # Kalman Gain
            self.x = self.x + k @ (meas - self.x)
            self.p = (np.eye(len(self.x)) - k) @ p_pred
        # Scalar case
        else:
            meas = float(measurement)
            # Predict step
            p_pred = self.p + self.q
            # Update step
            k = p_pred / (p_pred + self.r)  # Kalman Gain
            self.x = self.x + k * (meas - self.x)
            self.p = (1.0 - k) * p_pred
            
        return self.x

    def reset(self):
        self.x = None
        self.p = 1.0
