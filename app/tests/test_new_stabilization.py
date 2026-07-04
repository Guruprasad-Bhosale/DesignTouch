import os
import sys
import time
import numpy as np
import cv2
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.kalman_filter import KalmanFilter
from app.modules.filter_mode.filter_registry import FilterRegistry
from app.modules.sign_language.sign_language_module import SignLanguageModule
from app.services.camera_service import CameraService
from app.core.interfaces import ITrackingService, IStorageService, IConfigService
from app.core.service_manager import ServiceManager

class MockTrackingService:
    def __init__(self):
        self.smoothed_landmarks = {"Left": {}, "Right": {}}
        self.hand_present = {"Left": False, "Right": False}

class MockStorageService:
    def get_landmarks_dataset(self, version):
        return []

class MockConfigService:
    def get(self, key, default=None):
        if key == "sign_language_model":
            return "nonexistent_model.pkl"
        return default

class TestNewStabilization(unittest.TestCase):
    def test_kalman_filter_scalar(self):
        kf = KalmanFilter(process_noise=0.1, measurement_noise=0.5)
        # First measurement sets initial state
        val1 = kf.update(10.0)
        self.assertEqual(val1, 10.0)
        
        # Second measurement should be smoothed
        val2 = kf.update(12.0)
        self.assertTrue(10.0 < val2 < 12.0)
        
        kf.reset()
        self.assertIsNone(kf.x)

    def test_kalman_filter_vector(self):
        kf = KalmanFilter(process_noise=0.1, measurement_noise=0.5)
        pos1 = kf.update([10.0, 20.0])
        np.testing.assert_array_equal(pos1, np.array([10.0, 20.0], dtype=np.float32))
        
        pos2 = kf.update([12.0, 22.0])
        self.assertTrue(10.0 < pos2[0] < 12.0)
        self.assertTrue(20.0 < pos2[1] < 22.0)

    def test_filter_registry(self):
        registry = FilterRegistry()
        # Verify it lists files dynamically (or defaults to fallbacks if empty)
        filters = registry.filters
        self.assertTrue(len(filters) > 0)
        self.assertIn("ThermalVision", filters)

    def test_sign_language_rate_limiting(self):
        # Setup mocks
        tracker = MockTrackingService()
        storage = MockStorageService()
        config = MockConfigService()
        
        ServiceManager.register(ITrackingService, tracker)
        ServiceManager.register(IStorageService, storage)
        ServiceManager.register(IConfigService, config)
        
        sl_module = SignLanguageModule()
        sl_module.initialize(ServiceManager)
        
        # Mock right hand present for prediction
        tracker.hand_present["Right"] = True
        landmarks = {}
        for idx in range(21):
            landmarks[idx] = np.array([float(idx)*0.01, float(idx)*0.02, 0.1])
        tracker.smoothed_landmarks["Right"] = landmarks
        
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Mock prediction to keep track of calls
        predict_calls = 0
        original_predict = sl_module.classifier.predict
        
        def mock_predict(features):
            nonlocal predict_calls
            predict_calls += 1
            return "A", 0.90
            
        sl_module.classifier.predict = mock_predict
        
        # Call process_frame multiple times rapidly
        t_start = time.time()
        for _ in range(5):
            sl_module.process_frame(frame, {})
            
        # Since t_elapsed < 50ms, predict should only be called once (first time)
        self.assertEqual(predict_calls, 1)
        
        # Advance time by 60ms and call again
        sl_module._last_predict_time -= 0.06
        sl_module.process_frame(frame, {})
        self.assertEqual(predict_calls, 2)
        
        # Clean up
        sl_module.classifier.predict = original_predict

    def test_camera_service_fallback(self):
        # Create camera service pointing to a non-existent camera index or simulated failure
        cam = CameraService(camera_idx=999, target_fps=100)
        
        # Mock cv2.VideoCapture to simulate failed reads
        class MockVideoCapture:
            def __init__(self, idx):
                pass
            def set(self, prop, val):
                pass
            def get(self, prop):
                return 0
            def read(self):
                # Always fail
                return False, None
            def isOpened(self):
                return True
            def release(self):
                pass
                
        original_videocapture = cv2.VideoCapture
        cv2.VideoCapture = MockVideoCapture
        
        try:
            cam.start()
            # Wait briefly to let the thread execute a few iterations
            time.sleep(0.15)
            # The service should fail 10 times and switch to mock camera stream
            self.assertTrue(cam.use_mock)
            self.assertTrue(cam.grabbed)
            self.assertIsNotNone(cam.frame)
            cam.stop()
        finally:
            cv2.VideoCapture = original_videocapture

if __name__ == "__main__":
    import cv2
    unittest.main()
