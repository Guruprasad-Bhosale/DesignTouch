from pythonosc.udp_client import SimpleUDPClient
import numpy as np

class OSCSender:
    def __init__(self, ip="127.0.0.1", port=9000, enabled=True):
        self.ip = ip
        self.port = port
        self.enabled = enabled
        self.client = None
        
        if self.enabled:
            try:
                self.client = SimpleUDPClient(self.ip, self.port)
                print(f"[Info] OSC Sender initialized on {self.ip}:{self.port}")
            except Exception as e:
                print(f"[Warning] Failed to initialize OSC UDP Client: {e}")
                self.enabled = False

    def send_tracking_data(self, tracker):
        """Packs and streams coordinates and gestures over OSC to TouchDesigner."""
        if not self.enabled or not self.client:
            return
            
        try:
            # 1. Send hand presence flags
            self.client.send_message("/hand/left/present", int(tracker.hand_present["Left"]))
            self.client.send_message("/hand/right/present", int(tracker.hand_present["Right"]))
            
            # 2. Send hand gesture strings
            for side in ["Left", "Right"]:
                if tracker.hand_present[side]:
                    # Determine active gesture name
                    active_gesture = "None"
                    if tracker.gesture_states[side]["fist"]:
                        active_gesture = "Fist"
                    elif tracker.gesture_states[side]["open"]:
                        active_gesture = "OpenPalm"
                    elif tracker.gesture_states[side]["pinch"]:
                        active_gesture = "Pinch"
                        
                    self.client.send_message(f"/hand/{side.lower()}/gesture", active_gesture)
            
            # 3. Send fingertip landmarks (smooth)
            # Sends raw float array: [x4, y4, z4, x8, y8, z8, ...]
            # Fingertips indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
            fingertip_ids = [4, 8, 12, 16, 20]
            for side in ["Left", "Right"]:
                if tracker.hand_present[side] and len(tracker.smoothed_landmarks[side]) > 0:
                    arr = []
                    for idx in fingertip_ids:
                        if idx in tracker.smoothed_landmarks[side]:
                            arr.extend(tracker.smoothed_landmarks[side][idx].tolist())
                    if arr:
                        self.client.send_message(f"/hand/{side.lower()}/fingertips", arr)
                        
            # 4. Send accordion panels corners coordinates (if active)
            # Coordinates are sent as flat arrays of 8 floats: [x0, y0, x1, y1, x2, y2, x3, y3]
            panels = tracker.get_panels(1280, 720)
            self.client.send_message("/panels/count", len(panels))
            
            for p in panels:
                p_id = p["id"]
                corners = p["corners"].flatten().tolist()
                effect = p["effect"]
                alpha = p["alpha"]
                
                # Send OSC message for each panel
                # Address format: /panels/<id>/corners, /panels/<id>/effect, /panels/<id>/alpha
                self.client.send_message(f"/panels/{p_id}/corners", corners)
                self.client.send_message(f"/panels/{p_id}/effect", effect)
                self.client.send_message(f"/panels/{p_id}/alpha", alpha)
                
            # Send general system parameters
            self.client.send_message("/system/freeze", int(tracker.check_freeze()))
            
        except Exception as e:
            # Silence sending errors to prevent breaking the 60 FPS main loop
            pass
