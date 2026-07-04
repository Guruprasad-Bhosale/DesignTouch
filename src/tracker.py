import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=2, detection_con=0.45, tracking_con=0.45):
        # Reentrant lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize the modern MediaPipe HandLandmarker Task
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file 'hand_landmarker.task' not found at {model_path}.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Use VIDEO mode for optimized temporal hand tracking
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=tracking_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()
        
        # Exponential Moving Average smoothing factor
        self.ema_alpha = 0.5
        self._smoothed_landmarks = {"Left": {}, "Right": {}}
        
        # Hand presence states for occlusion handling
        self._hand_present = {"Left": False, "Right": False}
        self.last_presence = {"Left": False, "Right": False}
        self._panel_alpha = 0.0  # Smooth transition alpha
        
        # Gesture states
        self._gesture_states = {
            "Left": {"pinch": False, "fist": False, "open": False},
            "Right": {"pinch": False, "fist": False, "open": False}
        }
        
        # Gesture history for swipe detection
        self.wrist_history = {"Left": [], "Right": []}
        self.history_len = 10
        self._swipe_cooldown = 0.0
        self._pinch_cooldown = 0.0
        self._double_fist_cooldown = 0.0
        self._was_both_fist = False
        
        # Spawned filters (floating panels)
        self._spawned_panels = []
        
        from app.core.kalman_filter import KalmanFilter
        self.corner_kfs = [KalmanFilter(process_noise=1e-5, measurement_noise=5e-3) for _ in range(4)]
        self.next_spawn_id = 1
        
        # Available effects cycle loaded dynamically
        from app.modules.filter_mode.filter_registry import FilterRegistry
        registry = FilterRegistry()
        self.effects = registry.filters
        self._current_effect_idx = 0
        self._last_switch_time = 0.0
        self._gesture_lock_active = False
        self._palm_center_history = []
        self._lock_active_frames = 0
        self._lock_grace_frames = 0

    # --- Thread Safe Properties ---
    
    @property
    def hand_present(self):
        with self.lock:
            return self._hand_present
            
    @hand_present.setter
    def hand_present(self, val):
        with self.lock:
            self._hand_present = val

    @property
    def smoothed_landmarks(self):
        with self.lock:
            return self._smoothed_landmarks
            
    @smoothed_landmarks.setter
    def smoothed_landmarks(self, val):
        with self.lock:
            self._smoothed_landmarks = val

    @property
    def gesture_states(self):
        with self.lock:
            return self._gesture_states
            
    @gesture_states.setter
    def gesture_states(self, val):
        with self.lock:
            self._gesture_states = val

    @property
    def spawned_panels(self):
        with self.lock:
            return self._spawned_panels
            
    @spawned_panels.setter
    def spawned_panels(self, val):
        with self.lock:
            self._spawned_panels = val

    @property
    def current_effect_idx(self):
        with self.lock:
            return self._current_effect_idx
            
    @current_effect_idx.setter
    def current_effect_idx(self, val):
        with self.lock:
            self._current_effect_idx = val

    @property
    def last_switch_time(self):
        with self.lock:
            return self._last_switch_time
            
    @last_switch_time.setter
    def last_switch_time(self, val):
        with self.lock:
            self._last_switch_time = val

    @property
    def panel_alpha(self):
        with self.lock:
            return self._panel_alpha
            
    @panel_alpha.setter
    def panel_alpha(self, val):
        with self.lock:
            self._panel_alpha = val

    @property
    def double_fist_cooldown(self):
        with self.lock:
            return self._double_fist_cooldown
            
    @double_fist_cooldown.setter
    def double_fist_cooldown(self, val):
        with self.lock:
            self._double_fist_cooldown = val

    @property
    def was_both_fist(self):
        with self.lock:
            return self._was_both_fist
            
    @was_both_fist.setter
    def was_both_fist(self, val):
        with self.lock:
            self._was_both_fist = val

    @property
    def swipe_cooldown(self):
        with self.lock:
            return self._swipe_cooldown
            
    @swipe_cooldown.setter
    def swipe_cooldown(self, val):
        with self.lock:
            self._swipe_cooldown = val

    @property
    def pinch_cooldown(self):
        with self.lock:
            return self._pinch_cooldown
            
    @pinch_cooldown.setter
    def pinch_cooldown(self, val):
        with self.lock:
            self._pinch_cooldown = val

    @property
    def palm_center_history(self):
        with self.lock:
            return self._palm_center_history
            
    @palm_center_history.setter
    def palm_center_history(self, val):
        with self.lock:
            self._palm_center_history = val

    @property
    def gesture_lock_active(self):
        with self.lock:
            return self._gesture_lock_active
            
    @gesture_lock_active.setter
    def gesture_lock_active(self, val):
        with self.lock:
            self._gesture_lock_active = val

    # --- End Properties ---

    def _dist(self, p1, p2):
        """Helper to calculate distance between two 3D landmarks."""
        return np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z]))

    def process_frame(self, frame):
        """Processes the OpenCV frame, runs MediaPipe, and updates tracker state."""
        h, w, c = frame.shape
        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate monotonic timestamp in milliseconds
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        
        # Process and detect landmarks using Video running mode
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        # Reset presence flags for this frame
        current_presence = {"Left": False, "Right": False}
        raw_landmarks = {"Left": None, "Right": None}
        
        if results.hand_landmarks and results.handedness:
            for hand_idx, hand_lms in enumerate(results.hand_landmarks):
                # Retrieve handedness classification
                handedness_list = results.handedness[hand_idx]
                if handedness_list:
                    category = handedness_list[0]
                    label = getattr(category, 'category_name', getattr(category, 'label', 'Left'))
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
                    print(f"[HandTracker Diagnostic] {side} hand detected. Count: {num_landmarks} landmarks.")
                    if num_landmarks == 21:
                        print(f"{'ID':<5} | {'X':<10} | {'Y':<10} | {'Visibility':<10}")
                        print("-" * 45)
                        for idx, lm in enumerate(lms_list):
                            vis = getattr(lm, "visibility", 1.0)
                            if vis is None:
                                vis = 1.0
                            print(f"{idx:<5} | {lm.x:<10.4f} | {lm.y:<10.4f} | {vis:<10.4f}")

                        printed_any = True
            if printed_any:
                self._last_diag_print_time = t_now

        with self.lock:
            # We construct new states to perform reference swapping
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
                    # Smooth landmarks into new inner dictionary
                    new_side_landmarks = {}
                    for idx in range(21):
                        raw_p = np.array([raw_landmarks[side][idx].x, raw_landmarks[side][idx].y, raw_landmarks[side][idx].z])
                        if idx in new_landmarks[side]:
                            prev_p = new_landmarks[side][idx]
                            new_side_landmarks[idx] = self.ema_alpha * raw_p + (1.0 - self.ema_alpha) * prev_p
                        else:
                            new_side_landmarks[idx] = raw_p
                    new_landmarks[side] = new_side_landmarks
                else:
                    new_landmarks[side] = {}
                    self.wrist_history[side].clear()
                    new_gesture_states[side] = {"pinch": False, "fist": False, "open": False}
            
            # Atomic assignments
            self._hand_present = new_presence
            self._smoothed_landmarks = new_landmarks
            self._gesture_states = new_gesture_states

            # Print state changes for debugging
            if (self._hand_present["Left"] != self.last_presence["Left"] or 
                self._hand_present["Right"] != self.last_presence["Right"]):
                print(f"[Tracker Debug] Hand presence changed - Left: {self._hand_present['Left']}, Right: {self._hand_present['Right']}")
                self.last_presence["Left"] = self._hand_present["Left"]
                self.last_presence["Right"] = self._hand_present["Right"]

            # Update occlusion fade alpha
            if self._hand_present["Left"] and self._hand_present["Right"]:
                self._panel_alpha = min(1.0, self._panel_alpha + 0.1)
            else:
                self._panel_alpha = max(0.0, self._panel_alpha - 0.1)
                
            # Detect gestures and update state
            self._detect_gestures(w, h)
        
    def _check_hand_pose(self, side):
        # Already under lock when called from _detect_gestures
        if not self._hand_present[side] or len(self._smoothed_landmarks[side]) < 21:
            return False, False
        
        lms = self._smoothed_landmarks[side]
        v0 = lms[0]
        v5 = lms[5]
        v17 = lms[17]
        
        # 1. Palm Normal facing camera
        u = v5 - v0
        w_vec = v17 - v0
        if side == "Right":
            normal = np.cross(w_vec, u)
        else:
            normal = np.cross(u, w_vec)
            
        norm_val = np.linalg.norm(normal)
        normal_unit = normal / norm_val if norm_val > 0 else np.array([0.0, 0.0, 0.0])
        is_facing_camera = normal_unit[2] < -0.55
        
        # 2. Fingers open check
        t_tip_val = lms[4]
        t_ip_val = lms[3]
        t_mcp_val = lms[2]
        
        if np.linalg.norm(t_tip_val - t_ip_val) < 1e-4:
            is_thumb_open = np.linalg.norm(t_tip_val - v0) > 0.08
        else:
            dist_tip_wrist = np.linalg.norm(t_tip_val - v0)
            dist_mcp_wrist = np.linalg.norm(t_mcp_val - v0)
            is_thumb_open = dist_tip_wrist > dist_mcp_wrist * 1.12
            
        is_index_open = np.linalg.norm(lms[8] - v0) > np.linalg.norm(lms[6] - v0) * 1.15
        is_middle_open = np.linalg.norm(lms[12] - v0) > np.linalg.norm(lms[10] - v0) * 1.15
        is_ring_open = np.linalg.norm(lms[16] - v0) > np.linalg.norm(lms[14] - v0) * 1.15
        is_pinky_open = np.linalg.norm(lms[20] - v0) > np.linalg.norm(lms[18] - v0) * 1.15
        
        all_open = is_thumb_open and is_index_open and is_middle_open and is_ring_open and is_pinky_open
        
        return is_facing_camera, all_open

    def _detect_gestures(self, w, h):
        """Processes landmarks to detect gestures. (Invoked under lock)"""
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
                
        lock_active = (self._lock_active_frames >= 5) or (self._gesture_lock_active and self._lock_grace_frames > 0)
        self._gesture_lock_active = lock_active
        
        # Track palm centers in pixels if both hands are detected
        if self._hand_present["Left"] and self._hand_present["Right"] and len(self._smoothed_landmarks["Left"]) >= 21 and len(self._smoothed_landmarks["Right"]) >= 21:
            l_lms = self._smoothed_landmarks["Left"]
            r_lms = self._smoothed_landmarks["Right"]
            
            left_center = (l_lms[0] + l_lms[5] + l_lms[17]) / 3.0
            right_center = (r_lms[0] + r_lms[5] + r_lms[17]) / 3.0
            
            left_px_x = left_center[0] * w
            right_px_x = right_center[0] * w
            
            new_palm_history = list(self._palm_center_history)
            new_palm_history.append((t_now, left_px_x, right_px_x))
            self._palm_center_history = new_palm_history
            
        # Prune history to last 500 milliseconds
        self._palm_center_history = [item for item in self._palm_center_history if t_now - item[0] <= 0.5]
        
        # Check two palm swipe action
        if self._gesture_lock_active and t_now - self._swipe_cooldown > 1.0:
            if len(self._palm_center_history) > 1:
                curr_left_x = self._palm_center_history[-1][1]
                curr_right_x = self._palm_center_history[-1][2]
                
                for t_hist, hist_left_x, hist_right_x in self._palm_center_history[:-1]:
                    dx_left = curr_left_x - hist_left_x
                    dx_right = curr_right_x - hist_right_x
                    
                    if dx_left > 120.0 and dx_right > 120.0:
                        self._current_effect_idx = (self._current_effect_idx + 1) % len(self.effects)
                        self._swipe_cooldown = t_now
                        self._last_switch_time = t_now
                        self._palm_center_history = []
                        self._lock_grace_frames = 0
                        self._gesture_lock_active = False
                        print(f"[Tracker] Swipe Right! Filter changed to: {self.effects[self._current_effect_idx]}")
                        break
                    elif dx_left < -120.0 and dx_right < -120.0:
                        self._current_effect_idx = (self._current_effect_idx - 1) % len(self.effects)
                        self._swipe_cooldown = t_now
                        self._last_switch_time = t_now
                        self._palm_center_history = []
                        self._lock_grace_frames = 0
                        self._gesture_lock_active = False
                        print(f"[Tracker] Swipe Left! Filter changed to: {self.effects[self._current_effect_idx]}")
                        break

        # Keep existing single hand tracking gestures
        new_gesture_states = {
            "Left": dict(self._gesture_states["Left"]),
            "Right": dict(self._gesture_states["Right"])
        }
        
        for side in ["Left", "Right"]:
            if not self._hand_present[side] or len(self._smoothed_landmarks[side]) < 21:
                continue
                
            landmarks = self._smoothed_landmarks[side]
            
            class Pt:
                def __init__(self, arr):
                    self.x, self.y, self.z = arr[0], arr[1], arr[2]
            
            def get_pt(idx):
                return Pt(landmarks[idx])
                
            w_pt = get_pt(0)
            t_tip = get_pt(4)
            i_tip = get_pt(8)
            m_tip = get_pt(12)
            r_tip = get_pt(16)
            p_tip = get_pt(20)
            
            i_mcp = get_pt(5)
            m_mcp = get_pt(9)
            r_mcp = get_pt(13)
            p_mcp = get_pt(17)
            
            i_pip = get_pt(6)
            m_pip = get_pt(10)
            r_pip = get_pt(14)
            p_pip = get_pt(18)
            
            # 1. Detect Pinch
            pinch_dist = self._dist(t_tip, i_tip)
            is_pinching = pinch_dist < 0.045
            
            if is_pinching and not new_gesture_states[side]["pinch"]:
                if t_now - self._pinch_cooldown > 0.8:
                    self._pinch_cooldown = t_now
                    
            new_gesture_states[side]["pinch"] = is_pinching
            
            # 2. Detect Fist
            is_fist = (
                self._dist(i_tip, w_pt) < self._dist(i_mcp, w_pt) and
                self._dist(m_tip, w_pt) < self._dist(m_mcp, w_pt) and
                self._dist(r_tip, w_pt) < self._dist(r_mcp, w_pt) and
                self._dist(p_tip, w_pt) < self._dist(p_mcp, w_pt)
            )
            new_gesture_states[side]["fist"] = is_fist
            
            # 3. Detect Open Palm
            is_open = (
                self._dist(i_tip, w_pt) > self._dist(i_pip, w_pt) * 1.15 and
                self._dist(m_tip, w_pt) > self._dist(m_pip, w_pt) * 1.15 and
                self._dist(r_tip, w_pt) > self._dist(r_pip, w_pt) * 1.15 and
                self._dist(p_tip, w_pt) > self._dist(p_pip, w_pt) * 1.15
            )
            new_gesture_states[side]["open"] = is_open
            
            if self._hand_present["Left"] and self._hand_present["Right"]:
                if new_gesture_states["Left"]["open"] and new_gesture_states["Right"]["open"]:
                    self._spawned_panels = []
            elif new_gesture_states[side]["open"]:
                self._spawned_panels = []

        self._gesture_states = new_gesture_states

        # Check two hand fist action for changing filters
        both_fist = (self._hand_present["Left"] and self._gesture_states["Left"]["fist"] and
                     self._hand_present["Right"] and self._gesture_states["Right"]["fist"])
        
        if both_fist:
            if not self._was_both_fist and (t_now - self._double_fist_cooldown > 1.0):
                self._current_effect_idx = (self._current_effect_idx + 1) % len(self.effects)
                self._last_switch_time = t_now
                self._double_fist_cooldown = t_now
                print(f"[Tracker] Double Fist! Filter changed to: {self.effects[self._current_effect_idx]}")
            self._was_both_fist = True
        else:
            self._was_both_fist = False

    def _spawn_filter(self, x, y, screen_w, screen_h):
        """Spawns a floating rectangular panel at the normalized coordinates (x, y)."""
        with self.lock:
            size_x = 0.20
            size_y = 0.20 * (screen_w / screen_h)
            
            p0 = [x - size_x/2.0, y - size_y/2.0]
            p1 = [x + size_x/2.0, y - size_y/2.0]
            p2 = [x + size_x/2.0, y + size_y/2.0]
            p3 = [x - size_x/2.0, y + size_y/2.0]
            
            effect = self.effects[(self.next_spawn_id) % len(self.effects)]
            
            new_panels = list(self._spawned_panels)
            new_panels.append({
                "id": self.next_spawn_id,
                "center": (x, y),
                "size": (size_x, size_y),
                "effect": effect,
                "corners": np.array([p0, p1, p2, p3], dtype=np.float32)
            })
            self._spawned_panels = new_panels
            self.next_spawn_id += 1

    def get_panels(self, screen_w, screen_h):
        """Returns coordinate details and effects for all active panels."""
        with self.lock:
            panels = []
            
            # Include all permanently spawned panels first
            for sp in self._spawned_panels:
                panels.append({
                    "id": f"spawned_{sp['id']}",
                    "corners": sp["corners"],
                    "effect": sp["effect"],
                    "alpha": 1.0
                })
                
            t_now = time.time()
                
            if self._hand_present["Left"] and self._hand_present["Right"] and len(self._smoothed_landmarks["Left"]) > 0 and len(self._smoothed_landmarks["Right"]) > 0:
                left = self._smoothed_landmarks["Left"]
                right = self._smoothed_landmarks["Right"]
                
                required_joints = [4, 8]
                if all(j in left for j in required_joints) and all(j in right for j in required_joints):
                    tl1 = [left[8][0], left[8][1]]
                    tr1 = [right[8][0], right[8][1]]
                    br1 = [right[4][0], right[4][1]]
                    bl1 = [left[4][0], left[4][1]]
                    
                    raw_corners = np.array([tl1, tr1, br1, bl1], dtype=np.float32)
                    filtered_corners = []
                    for i in range(4):
                        filtered_corners.append(self.corner_kfs[i].update(raw_corners[i]))
                    corners = np.array(filtered_corners, dtype=np.float32)
                    
                    e = self.effects[self._current_effect_idx]
                    
                    panels.append(
                        {"id": "band_1", "corners": corners, "effect": e, "alpha": self._panel_alpha}
                    )
                    
                    # --- Steady Hold Detection Logic ---
                    if not hasattr(self, '_band_corner_history'):
                        self._band_corner_history = []
                        
                    self._band_corner_history.append((t_now, corners.copy()))
                    
                    # Prune history to last 3.5 seconds max to keep memory lean
                    self._band_corner_history = [item for item in self._band_corner_history if t_now - item[0] <= 3.5]
                    
                    if len(self._band_corner_history) > 10:
                        t_oldest, c_oldest = self._band_corner_history[0]
                        
                        if t_now - t_oldest >= 2.9: # roughly 3 seconds
                            # Check variance of all corners
                            max_dist = 0.0
                            for hist_t, hist_c in self._band_corner_history:
                                dists = np.linalg.norm(hist_c - c_oldest, axis=1)
                                max_dist = max(max_dist, np.max(dists))
                                
                            # If the hands haven't moved more than ~4% of the screen over 3 seconds
                            if max_dist < 0.04:
                                print(f"[Tracker] Filter Held Steady! Pinning '{e}' permanently.")
                                new_panels = list(self._spawned_panels)
                                new_panels.append({
                                    "id": self.next_spawn_id,
                                    "center": (0, 0),
                                    "size": (0, 0),
                                    "effect": e,
                                    "corners": corners.copy()
                                })
                                self._spawned_panels = new_panels
                                self.next_spawn_id += 1
                                # Clear history so we don't pin it a hundred times
                                self._band_corner_history = []
            else:
                for kf in self.corner_kfs:
                    kf.reset()
                if hasattr(self, '_band_corner_history'):
                    self._band_corner_history.clear()
                    
            return panels

    def check_freeze(self):
        """Returns True if a fist is active on either hand (but not both), signaling a freeze-frame."""
        with self.lock:
            left_freeze = self._hand_present["Left"] and self._gesture_states["Left"]["fist"]
            right_freeze = self._hand_present["Right"] and self._gesture_states["Right"]["fist"]
            if left_freeze and right_freeze:
                return False
            return left_freeze or right_freeze

    def clear_spawned_panels(self):
        """Thread-safely clears the spawned filters."""
        with self.lock:
            self._spawned_panels = []

    def close(self):
        """Closes the MediaPipe detector session to release assets."""
        if hasattr(self, "detector"):
            try:
                self.detector.close()
            except Exception:
                pass
