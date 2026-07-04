import numpy as np
import time
from app.modules.base_module import BaseModule
from app.core.service_manager import ServiceManager
from app.core.interfaces import ITrackingService, IConfigService
from app.modules.filter_mode.filter_registry import FilterRegistry

class FilterModule(BaseModule):
    def __init__(self):
        self._service_manager = None
        self._tracker = None
        self._config = None
        self._registry = None
        self._current_effect_idx = 0
        self._last_swipe_time = 0.0
        self._swipe_cooldown_ms = 1000
        
        from app.core.kalman_filter import KalmanFilter
        self.corner_kfs = [KalmanFilter(process_noise=1e-5, measurement_noise=5e-3) for _ in range(4)]

    def initialize(self, service_manager) -> bool:
        self._service_manager = service_manager
        try:
            self._tracker = service_manager.get(ITrackingService)
            self._config = service_manager.get(IConfigService)
            
            # Load shader registry
            self._registry = FilterRegistry()
            
            # Sync cooldown with configuration
            self._swipe_cooldown_ms = self._config.get("filter_cooldown_ms", 1000)
            
            # Match current effect index from config or set default
            self._current_effect_idx = 0
            
            print("[FilterModule] Initialized successfully.")
            return True
        except Exception as e:
            print(f"[FilterModule Error] Initialization failed: {e}")
            return False

    @property
    def current_effect_name(self) -> str:
        filters = self._registry.filters
        if filters:
            return filters[self._current_effect_idx % len(filters)]
        return "ThermalVision"

    def process_frame(self, frame, tracking_data: dict) -> np.ndarray:
        """
        The CPU fallback rendering or drawing of instructions is orchestrated here.
        GPU rendering coordinates are processed in the QOpenGLWidget viewport using get_panel_geometry().
        """
        # Read gesture swipe events from tracker
        t_now = time.time()
        if hasattr(self._tracker, "swipe_event") and self._tracker.swipe_event:
            if (t_now - self._last_swipe_time) * 1000 > self._swipe_cooldown_ms:
                direction = self._tracker.swipe_event
                self.handle_gesture(f"swipe_{direction.lower()}")
                self._last_swipe_time = t_now
                
        return frame

    def handle_gesture(self, gesture_event: str):
        filters = self._registry.filters
        if not filters:
            return
            
        if gesture_event == "swipe_right":
            self._current_effect_idx = (self._current_effect_idx + 1) % len(filters)
            print(f"[FilterModule] Swiped Right. Active filter: {self.current_effect_name}")
        elif gesture_event == "swipe_left":
            self._current_effect_idx = (self._current_effect_idx - 1) % len(filters)
            print(f"[FilterModule] Swiped Left. Active filter: {self.current_effect_name}")

    def get_panel_geometry(self, width: int, height: int) -> dict:
        """
        Calculates normalized corners of the quadrilateral reality filter panel:
        Top Left (Left Index=8), Bottom Left (Left Thumb=4),
        Top Right (Right Index=8), Bottom Right (Right Thumb=4).
        """
        landmarks = self._tracker.smoothed_landmarks
        presence = self._tracker.hand_present
        
        # Check both hands present with required joints
        if presence["Left"] and presence["Right"]:
            left = landmarks["Left"]
            right = landmarks["Right"]
            
            required = [4, 8]
            if all(j in left for j in required) and all(j in right for j in required):
                # Mirroring adjustments are removed because the input frame is already flipped
                tl = np.array([left[8][0], left[8][1]], dtype=np.float32)
                bl = np.array([left[4][0], left[4][1]], dtype=np.float32)
                tr = np.array([right[8][0], right[8][1]], dtype=np.float32)
                br = np.array([right[4][0], right[4][1]], dtype=np.float32)
                
                # Format required: TopLeft, TopRight, BottomRight, BottomLeft (clockwise)
                corners = np.array([tl, tr, br, bl], dtype=np.float32)
                
                # Apply Kalman filtering to each corner
                filtered_corners = []
                for i in range(4):
                    filtered_corners.append(self.corner_kfs[i].update(corners[i]))
                corners = np.array(filtered_corners, dtype=np.float32)
                
                return {
                    "id": "reality_quad",
                    "corners": corners,
                    "effect": self.current_effect_name,
                    "alpha": getattr(self._tracker, "panel_alpha", 1.0)
                }
        else:
            # Reset Kalman filters on hand loss to prevent lagging transition jumps
            for kf in self.corner_kfs:
                kf.reset()
                
        return None

    def get_ui_widget(self):
        """No custom module settings widget needed, using main app settings."""
        return None

    def deinitialize(self):
        print("[FilterModule] Deinitialized.")
