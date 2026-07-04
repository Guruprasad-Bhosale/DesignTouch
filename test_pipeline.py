import numpy as np
import time
from src.tracker import HandTracker

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def test_tracker_pipeline():
    print("[Test] Initializing HandTracker...")
    tracker = HandTracker()
    
    # Create dummy black frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("[Test] Processing blank frame (no hands)...")
    tracker.process_frame(frame)
    assert tracker.hand_present["Left"] == False
    assert tracker.hand_present["Right"] == False
    assert tracker.panel_alpha == 0.0
    print("[Pass] Zero hands state verified.")
    
    # Helper to create mock landmarks for a hand
    # Extended open hand profile
    def get_mock_landmarks(wrist_x, wrist_y, is_fist=False, is_left=False):
        landmarks = []
        # Wrist (0)
        landmarks.append(MockLandmark(wrist_x, wrist_y))
        
        # Thumb: 1, 2, 3, 4
        thumb_d = 0.03 if is_fist else 0.1
        # Mirror thumb offset for left hand
        thumb_offset_x = 0.05 if is_left else -0.05
        for i in range(4):
            landmarks.append(MockLandmark(wrist_x + thumb_offset_x, wrist_y - thumb_d))
            
        # Fingers: Index, Middle, Ring, Pinky
        # Mirror finger offsets for left hand
        if is_left:
            offsets = [0.06, 0.03, 0.0, -0.03]
        else:
            offsets = [-0.03, 0.0, 0.03, 0.06]
            
        for offset in offsets:
            mcp_y = wrist_y - 0.08
            pip_y = wrist_y - 0.12
            dip_y = wrist_y - 0.15
            tip_y = wrist_y - 0.05 if is_fist else wrist_y - 0.18
            
            landmarks.append(MockLandmark(wrist_x + offset, mcp_y)) # MCP
            landmarks.append(MockLandmark(wrist_x + offset, pip_y)) # PIP
            landmarks.append(MockLandmark(wrist_x + offset, dip_y)) # DIP
            landmarks.append(MockLandmark(wrist_x + offset, tip_y)) # Tip
            
        return landmarks

    # Mock process_frame with injected hand landmarks
    # Since we want to bypass MediaPipe's camera capture for testing, we manually set hand tracking parameters:
    print("[Test] Simulating both hands visible (open palms)...")
    
    # We will simulate 15 successive frames of both hands active to test EMA smoothing and alpha fade-in
    for frame_idx in range(15):
        current_presence = {"Left": True, "Right": True}
        raw_landmarks = {
            "Left": get_mock_landmarks(0.2, 0.5, is_fist=False, is_left=True),
            "Right": get_mock_landmarks(0.8, 0.5, is_fist=False, is_left=False)
        }
        
        # Manually invoke tracking logic updates
        for side in ["Left", "Right"]:
            tracker.hand_present[side] = current_presence[side]
            for idx in range(21):
                raw_p = np.array([raw_landmarks[side][idx].x, raw_landmarks[side][idx].y, raw_landmarks[side][idx].z])
                if idx in tracker.smoothed_landmarks[side]:
                    prev_p = tracker.smoothed_landmarks[side][idx]
                    tracker.smoothed_landmarks[side][idx] = tracker.ema_alpha * raw_p + (1.0 - tracker.ema_alpha) * prev_p
                else:
                    tracker.smoothed_landmarks[side][idx] = raw_p
                    
        # Update presence and alpha
        if tracker.hand_present["Left"] and tracker.hand_present["Right"]:
            tracker.panel_alpha = min(1.0, tracker.panel_alpha + 0.1)
        else:
            tracker.panel_alpha = max(0.0, tracker.panel_alpha - 0.1)
            
        # Trigger internal gesture detection
        tracker._detect_gestures(1280, 720)
        
    print(f"[Test] Current Panel Alpha: {tracker.panel_alpha:.2f}")
    assert tracker.panel_alpha > 0.9  # Should fade in completely
    # Retrieve panels
    panels = tracker.get_panels(1280, 720)
    print(f"[Test] Active Panels: {len(panels)}")
    assert len(panels) == 1  # Should only generate the index-thumb band
    assert panels[0]["id"] == "band_1"
    print("[Pass] Accordion band generation (index-thumb only) and fade-in verified.")
    
    # Test Double Fist Gesture (Filter Switch + No Freeze)
    print("[Test] Simulating Double Fist gesture (both hands fist)...")
    tracker.current_effect_idx = 0
    tracker.double_fist_cooldown = 0.0
    tracker.was_both_fist = False
    
    raw_landmarks_double_fist = {
        "Left": get_mock_landmarks(0.2, 0.5, is_fist=True, is_left=True),
        "Right": get_mock_landmarks(0.8, 0.5, is_fist=True, is_left=False)
    }
    
    for side in ["Left", "Right"]:
        tracker.hand_present[side] = True
        for idx in range(21):
            raw_p = np.array([raw_landmarks_double_fist[side][idx].x, raw_landmarks_double_fist[side][idx].y, raw_landmarks_double_fist[side][idx].z])
            tracker.smoothed_landmarks[side][idx] = raw_p
            
    tracker._detect_gestures(1280, 720)
    is_frozen = tracker.check_freeze()
    print(f"[Test] Double Fist - Freeze State Active: {is_frozen}")
    assert is_frozen == False  # Double fist should not trigger freeze
    print(f"[Test] Double Fist - New Filter Index: {tracker.current_effect_idx}")
    assert tracker.current_effect_idx == 1  # Should cycle from 0 to 1
    print("[Pass] Double fist filter switch and no-freeze verified.")

    # Test Single Fist Gesture (Freeze)
    print("[Test] Simulating Single Fist gesture (Left hand fist, Right hand open)...")
    raw_landmarks_single_fist = {
        "Left": get_mock_landmarks(0.2, 0.5, is_fist=True, is_left=True),
        "Right": get_mock_landmarks(0.8, 0.5, is_fist=False, is_left=False)
    }
    
    for side in ["Left", "Right"]:
        tracker.hand_present[side] = True
        for idx in range(21):
            raw_p = np.array([raw_landmarks_single_fist[side][idx].x, raw_landmarks_single_fist[side][idx].y, raw_landmarks_single_fist[side][idx].z])
            tracker.smoothed_landmarks[side][idx] = raw_p
            
    tracker._detect_gestures(1280, 720)
    is_frozen = tracker.check_freeze()
    print(f"[Test] Single Fist - Freeze State Active: {is_frozen}")
    assert is_frozen == True  # Single fist should freeze
    print("[Pass] Single fist freeze frame gesture verified.")
    
    # Test Open Palm Reset
    print("[Test] Simulating Open Palm (re-extending fingers)...")
    tracker.spawned_panels.clear()
    # Spawn a dummy floating panel to test if it clears
    tracker._spawn_filter(0.5, 0.5, 1280, 720)
    assert len(tracker.spawned_panels) == 1
    
    raw_landmarks_open = {
        "Left": get_mock_landmarks(0.2, 0.5, is_fist=False, is_left=True),
        "Right": get_mock_landmarks(0.8, 0.5, is_fist=False, is_left=False)
    }
    for side in ["Left", "Right"]:
        for idx in range(21):
            raw_p = np.array([raw_landmarks_open[side][idx].x, raw_landmarks_open[side][idx].y, raw_landmarks_open[side][idx].z])
            tracker.smoothed_landmarks[side][idx] = raw_p
            
    tracker._detect_gestures(1280, 720)
    print(f"[Test] Spawned Panels Count: {len(tracker.spawned_panels)}")
    assert len(tracker.spawned_panels) == 0  # Should be cleared by open palm
    print("[Pass] Open palm reset state verified.")
    
    # Test Two Palm Filter Switch Gesture (Swipe Left)
    print("[Test] Simulating Two Palm Filter Switch gesture (Swipe Left)...")
    tracker.current_effect_idx = 0
    tracker.swipe_cooldown = 0.0
    tracker.palm_center_history.clear()
    
    # Frame 1: Hands open and facing camera at start position
    raw_lms1 = {
        "Left": get_mock_landmarks(0.3, 0.5, is_fist=False, is_left=True),
        "Right": get_mock_landmarks(0.7, 0.5, is_fist=False, is_left=False)
    }
    for side in ["Left", "Right"]:
        tracker.hand_present[side] = True
        for idx in range(21):
            raw_p = np.array([raw_lms1[side][idx].x, raw_lms1[side][idx].y, raw_lms1[side][idx].z])
            tracker.smoothed_landmarks[side][idx] = raw_p
            
    tracker._detect_gestures(1280, 720)
    assert tracker.gesture_lock_active == True
    assert len(tracker.palm_center_history) == 1
    
    # Frame 2: Move hands to the left (by 0.2 normalized width, which is 256 pixels)
    raw_lms2 = {
        "Left": get_mock_landmarks(0.1, 0.5, is_fist=False, is_left=True),
        "Right": get_mock_landmarks(0.5, 0.5, is_fist=False, is_left=False)
    }
    for side in ["Left", "Right"]:
        tracker.hand_present[side] = True
        for idx in range(21):
            raw_p = np.array([raw_lms2[side][idx].x, raw_lms2[side][idx].y, raw_lms2[side][idx].z])
            tracker.smoothed_landmarks[side][idx] = raw_p
            
    tracker._detect_gestures(1280, 720)
    print(f"[Test] New active filter index: {tracker.current_effect_idx}")
    # Since we swiped left, current_effect_idx should cycle backward: 0 -> 9 (which is len(effects) - 1)
    assert tracker.current_effect_idx == len(tracker.effects) - 1
    print("[Pass] Two Palm Filter Switch Swipe Left verified.")
    
    print("\n--- ALL PIPELINE TESTS PASSED SUCCESSFULY ---")

if __name__ == "__main__":
    test_tracker_pipeline()
