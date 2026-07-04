import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.core.interfaces import ITrackingService

class TrackingService(ITrackingService):
    def __init__(self, model_relative_path="hand_landmarker.task"):
        self.lock = threading.RLock()
        self.started = False
        self.thread = None
        self.detector = None
        self._model_path = model_relative_path
        
        # Exponential Moving Average for joint landmarks
        self.ema_alpha = 0.75
        self._smoothed_landmarks = {"Left": {}, "Right": {}}
        self._hand_present = {"Left": False, "Right": False}
        
        # UI Cursor System (Right Hand Index Tip)
        self.cursor_window_size = 3
        self.cursor_history_x = deque(maxlen=self.cursor_window_size)
        self.cursor_history_y = deque(maxlen=self.cursor_window_size)
        self._cursor_pos = (0, 0)
        self.sensitivity = 1.8
        
        from app.core.kalman_filter import KalmanFilter
        self.cursor_kf = KalmanFilter(process_noise=0.03, measurement_noise=0.5)
        
        # Selection System (Left Hand Index + Thumb Pinch)
        self.selection_threshold = 0.035
        self.release_threshold = 0.055
        self._is_pinched = False
        self._click_state = False
        
        # Gesture States
        self._gesture_states = {
            "Left": {"pinch": False, "fist": False, "open": False},
            "Right": {"pinch": False, "fist": False, "open": False}
        }
        
        # Accordion Band Alpha (occlusion fading)
        self.panel_alpha = 0.0
        
        # Swipe and Fist state parameters
        self._palm_center_history = []
        self._swipe_cooldown = 0.0
        self._last_switch_time = 0.0
        self._swipe_event = None  # "Left" or "Right"
        self._gesture_lock_active = False
        self._lock_active_frames = 0
        self._lock_grace_frames = 0
        
        # Model path resolution
        self._resolve_model_path()

    def _resolve_model_path(self):
        if not os.path.isabs(self._model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            abs_path = os.path.join(base_dir, self._model_path)
            if os.path.exists(abs_path):
                self._model_path = abs_path
            else:
                # Search project root
                parent_dir = os.path.dirname(base_dir)
                root_path = os.path.join(parent_dir, self._model_path)
                if os.path.exists(root_path):
                    self._model_path = root_path
                    
        print(f"[TrackingService] Resolved landmarker path to: {self._model_path}")

    def start(self):
        with self.lock:
            if self.started:
                return self
                
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"MediaPipe task model file not found at: {self._model_path}")
                
            base_options = python.BaseOptions(model_asset_path=self._model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.45
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self._start_time = time.time()
            self.started = True
            print("[TrackingService] MediaPipe HandLandmarker started successfully.")
            return self

    def process_frame(self, frame):
        if not self.started or self.detector is None:
            return
            
        h, w, c = frame.shape
        # MediaPipe expects RGB image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        
        # Process landmarks
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        current_presence = {"Left": False, "Right": False}
        raw_landmarks = {"Left": None, "Right": None}
        
        if results.hand_landmarks and results.handedness:
            for hand_idx, hand_lms in enumerate(results.hand_landmarks):
                handedness_list = results.handedness[hand_idx]
                if handedness_list:
                    category = handedness_list[0]
                    label = getattr(category, 'category_name', getattr(category, 'label', 'Left'))
                    # Standardize classification label naming
                    label = label.capitalize()
                    current_presence[label] = True
                    raw_landmarks[label] = hand_lms
                    
        # Rate-limited landmark diagnostic output (every 2.0 seconds)
        t_now = time.time()
        if not hasattr(self, "_last_diag_print_time"):
            self._last_diag_print_time = 0.0
        if t_now - self._last_diag_print_time >= 2.0:
            printed_any = False
            for side in ["Left", "Right"]:
                if current_presence[side] and raw_landmarks[side] is not None:
                    lms_list = raw_landmarks[side]
                    num_landmarks = len(lms_list)
                    print(f"[TrackingService Diagnostic] {side} hand detected. Count: {num_landmarks} landmarks.")
                    printed_any = True
            if printed_any:
                self._last_diag_print_time = t_now

        with self.lock:
            new_presence = dict(self._hand_present)
            new_landmarks = {
                "Left": dict(self._smoothed_landmarks["Left"]),
                "Right": dict(self._smoothed_landmarks["Right"])
            }
            new_gesture_states = {
                "Left": dict(self._gesture_states["Left"]),
                "Right": dict(self._gesture_states["Right"])
            }
            
            for side in ["Left", "Right"]:
                new_presence[side] = current_presence[side]
                if current_presence[side] and raw_landmarks[side] is not None:
                    new_side_landmarks = {}
                    for idx in range(21):
                        raw_p = np.array([raw_landmarks[side][idx].x, raw_landmarks[side][idx].y, raw_landmarks[side][idx].z])
                        if idx in new_landmarks[side]:
                            prev_p = new_landmarks[side][idx]
                            # EMA smoothing of raw landmark coordinates
                            new_side_landmarks[idx] = self.ema_alpha * raw_p + (1.0 - self.ema_alpha) * prev_p
                        else:
                            new_side_landmarks[idx] = raw_p
                    new_landmarks[side] = new_side_landmarks
                else:
                    new_landmarks[side] = {}
                    new_gesture_states[side] = {"pinch": False, "fist": False, "open": False}
            
            self._hand_present = new_presence
            self._smoothed_landmarks = new_landmarks
            self._gesture_states = new_gesture_states
            
            # Panel occlusion fading transition
            if self._hand_present["Left"] and self._hand_present["Right"]:
                self.panel_alpha = min(1.0, self.panel_alpha + 0.1)
            else:
                self.panel_alpha = max(0.0, self.panel_alpha - 0.1)
                
            self._detect_gestures(w, h)

    def _dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _check_hand_pose(self, side):
        if not self._hand_present[side] or len(self._smoothed_landmarks[side]) < 21:
            return False, False
            
        lms = self._smoothed_landmarks[side]
        v0, v5, v17 = lms[0], lms[5], lms[17]
        
        # Check if palm faces the camera
        u = v5 - v0
        w_vec = v17 - v0
        normal = np.cross(w_vec, u) if side == "Right" else np.cross(u, w_vec)
        norm_val = np.linalg.norm(normal)
        normal_unit = normal / norm_val if norm_val > 0 else np.array([0.0, 0.0, 0.0])
        is_facing_camera = normal_unit[2] < -0.55
        
        # Check open fingers
        t_tip, t_ip, t_mcp = lms[4], lms[3], lms[2]
        dist_tip_wrist = np.linalg.norm(t_tip - v0)
        dist_mcp_wrist = np.linalg.norm(t_mcp - v0)
        is_thumb_open = dist_tip_wrist > dist_mcp_wrist * 1.12
        
        is_index_open = np.linalg.norm(lms[8] - v0) > np.linalg.norm(lms[6] - v0) * 1.15
        is_middle_open = np.linalg.norm(lms[12] - v0) > np.linalg.norm(lms[10] - v0) * 1.15
        is_ring_open = np.linalg.norm(lms[16] - v0) > np.linalg.norm(lms[14] - v0) * 1.15
        is_pinky_open = np.linalg.norm(lms[20] - v0) > np.linalg.norm(lms[18] - v0) * 1.15
        
        all_open = is_thumb_open and is_index_open and is_middle_open and is_ring_open and is_pinky_open
        return is_facing_camera, all_open

    def _detect_gestures(self, w, h):
        t_now = time.time()
        
        # --- Dual Palm Swipe Detection (Lock verification with 5-frame debounce) ---
        is_posture_valid = False
        if self._hand_present["Left"] and self._hand_present["Right"]:
            l_facing, l_open = self._check_hand_pose("Left")
            r_facing, r_open = self._check_hand_pose("Right")
            if l_facing and l_open and r_facing and r_open:
                is_posture_valid = True
                
        if is_posture_valid:
            self._lock_active_frames = min(10, self._lock_active_frames + 1)
            self._lock_grace_frames = 8  # Refresh grace window
        else:
            self._lock_active_frames = 0
            if self._lock_grace_frames > 0:
                self._lock_grace_frames -= 1
                
        # Lock is active if debounced OR we are in the grace window during swipe
        lock_active = (self._lock_active_frames >= 5) or (self._gesture_lock_active and self._lock_grace_frames > 0)
        self._gesture_lock_active = lock_active
        
        if self._hand_present["Left"] and self._hand_present["Right"]:
            l_lms = self._smoothed_landmarks["Left"]
            r_lms = self._smoothed_landmarks["Right"]
            if all(j in l_lms for j in [0, 5, 17]) and all(j in r_lms for j in [0, 5, 17]):
                left_center = (l_lms[0] + l_lms[5] + l_lms[17]) / 3.0
                right_center = (r_lms[0] + r_lms[5] + r_lms[17]) / 3.0
                # Since camera coordinates range 0 to 1, map x coordinates to pixels
                self._palm_center_history.append((t_now, left_center[0] * w, right_center[0] * w))
            
        # Prune history to last 500 ms
        self._palm_center_history = [item for item in self._palm_center_history if t_now - item[0] <= 0.5]
        
        # Check swipe
        self._swipe_event = None
        if self._gesture_lock_active and t_now - self._swipe_cooldown > 1.0:
            if len(self._palm_center_history) > 1:
                curr_left_x = self._palm_center_history[-1][1]
                curr_right_x = self._palm_center_history[-1][2]
                
                for t_hist, hist_left_x, hist_right_x in self._palm_center_history[:-1]:
                    dx_left = curr_left_x - hist_left_x
                    dx_right = curr_right_x - hist_right_x
                    
                    # Swipe right
                    if dx_left > 120.0 and dx_right > 120.0:
                        self._swipe_event = "Right"
                        self._swipe_cooldown = t_now
                        self._last_switch_time = t_now
                        self._palm_center_history.clear()
                        self._lock_grace_frames = 0
                        self._gesture_lock_active = False
                        print("[TrackingService] Swipe Right detected!")
                        break
                    # Swipe left
                    elif dx_left < -120.0 and dx_right < -120.0:
                        self._swipe_event = "Left"
                        self._swipe_cooldown = t_now
                        self._last_switch_time = t_now
                        self._palm_center_history.clear()
                        self._lock_grace_frames = 0
                        self._gesture_lock_active = False
                        self._palm_center_history.clear()
                        print("[TrackingService] Swipe Left detected!")
                        break

        # --- Finger states for individual hands ---
        new_gesture_states = {
            "Left": dict(self._gesture_states["Left"]),
            "Right": dict(self._gesture_states["Right"])
        }
        
        for side in ["Left", "Right"]:
            if not self._hand_present[side] or len(self._smoothed_landmarks[side]) < 21:
                continue
                
            lms = self._smoothed_landmarks[side]
            w_pt, t_tip, i_tip, m_tip, r_tip, p_tip = lms[0], lms[4], lms[8], lms[12], lms[16], lms[20]
            i_mcp, m_mcp, r_mcp, p_mcp = lms[5], lms[9], lms[13], lms[17]
            i_pip, m_pip, r_pip, p_pip = lms[6], lms[10], lms[14], lms[18]
            
            # Pinch (Index tip + Thumb tip)
            pinch_d = self._dist(t_tip, i_tip)
            new_gesture_states[side]["pinch"] = pinch_d < 0.045
            
            # Fist (Fingers folded)
            new_gesture_states[side]["fist"] = (
                self._dist(i_tip, w_pt) < self._dist(i_mcp, w_pt) and
                self._dist(m_tip, w_pt) < self._dist(m_mcp, w_pt) and
                self._dist(r_tip, w_pt) < self._dist(r_mcp, w_pt) and
                self._dist(p_tip, w_pt) < self._dist(p_mcp, w_pt)
            )
            
            # Open Palm
            new_gesture_states[side]["open"] = (
                self._dist(i_tip, w_pt) > self._dist(i_pip, w_pt) * 1.15 and
                self._dist(m_tip, w_pt) > self._dist(m_pip, w_pt) * 1.15 and
                self._dist(r_tip, w_pt) > self._dist(r_pip, w_pt) * 1.15 and
                self._dist(p_tip, w_pt) > self._dist(p_pip, w_pt) * 1.15
            )
            
        self._gesture_states = new_gesture_states
        
        # --- UI Virtual Cursor (Right Hand Index tip = landmark 8) ---
        if self._hand_present["Right"] and 8 in self._smoothed_landmarks["Right"]:
            raw_index = self._smoothed_landmarks["Right"][8]
            # Mirror x-coordinate is removed because the input frame is already flipped
            screen_x_raw = raw_index[0]
            screen_y_raw = raw_index[1]
            
            # Map with sensitivity stretching centered around 0.5
            mapped_x = w * (0.5 + (screen_x_raw - 0.5) * self.sensitivity)
            mapped_y = h * (0.5 + (screen_y_raw - 0.5) * self.sensitivity)
            
            # Clamp inside bounds
            mapped_x = float(np.clip(mapped_x, 0, w - 1))
            mapped_y = float(np.clip(mapped_y, 0, h - 1))
            
            # Feed to moving average sliding buffer
            self.cursor_history_x.append(mapped_x)
            self.cursor_history_y.append(mapped_y)
            
            # Simple Moving Average calculation
            sma_x = sum(self.cursor_history_x) / len(self.cursor_history_x)
            sma_y = sum(self.cursor_history_y) / len(self.cursor_history_y)
            
            # Kalman Filter smoothing
            smoothed_pos = self.cursor_kf.update(np.array([sma_x, sma_y], dtype=np.float32))
            self._cursor_pos = (int(smoothed_pos[0]), int(smoothed_pos[1]))
        else:
            self.cursor_kf.reset()
            
        # --- Selection click trigger (Left hand pinch 4 + 8) ---
        self._click_state = False
        if self._hand_present["Left"] and 4 in self._smoothed_landmarks["Left"] and 8 in self._smoothed_landmarks["Left"]:
            left_thumb = self._smoothed_landmarks["Left"][4]
            left_index = self._smoothed_landmarks["Left"][8]
            pinch_dist = self._dist(left_thumb, left_index)
            
            if not self._is_pinched:
                if pinch_dist < self.selection_threshold:
                    self._is_pinched = True
                    self._click_state = True  # Click triggered (rising edge)
                    print(f"[TrackingService] Click! Pinch distance: {pinch_dist:.4f}")
            else:
                if pinch_dist > self.release_threshold:
                    self._is_pinched = False
                    print(f"[TrackingService] Pinch released! Distance: {pinch_dist:.4f}")

    @property
    def smoothed_landmarks(self) -> dict:
        with self.lock:
            # Return copy to prevent thread data races on concurrent modifications
            return {side: dict(coords) for side, coords in self._smoothed_landmarks.items()}

    @property
    def hand_present(self) -> dict:
        with self.lock:
            # Return copy to prevent thread data races
            return dict(self._hand_present)

    @property
    def cursor_position(self) -> tuple:
        with self.lock:
            return self._cursor_pos

    @property
    def click_state(self) -> bool:
        with self.lock:
            return self._click_state

    @property
    def gesture_states(self) -> dict:
        with self.lock:
            # Return copy to prevent thread data races
            return {side: dict(states) for side, states in self._gesture_states.items()}

    @property
    def swipe_event(self) -> str:
        with self.lock:
            return self._swipe_event

    @property
    def last_switch_time(self) -> float:
        with self.lock:
            return self._last_switch_time

    def check_freeze(self) -> bool:
        with self.lock:
            left_freeze = self._hand_present["Left"] and self._gesture_states["Left"]["fist"]
            right_freeze = self._hand_present["Right"] and self._gesture_states["Right"]["fist"]
            # Trigger freeze frame if either hand is closed, but not both
            if left_freeze and right_freeze:
                return False
            return left_freeze or right_freeze

    def stop(self):
        with self.lock:
            self.started = False
            if self.detector:
                try:
                    self.detector.close()
                except Exception:
                    pass
                self.detector = None
            print("[TrackingService] Tracker closed and resources released.")
