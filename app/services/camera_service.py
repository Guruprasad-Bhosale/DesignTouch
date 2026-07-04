import cv2
import threading
import time
import numpy as np
from app.core.interfaces import ICameraService

class CameraService(ICameraService):
    def __init__(self, camera_idx=0, width=1280, height=720, target_fps=60):
        self._camera_idx = camera_idx
        self._target_width = width
        self._target_height = height
        self._target_fps = target_fps
        
        self.stream = None
        self.started = False
        self.thread = None
        self.read_lock = threading.Lock()
        
        self.frame = None
        self.grabbed = False
        self.frame_id = 0
        self.use_mock = False
        
        # FPS calculation variables
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self._actual_fps = 0.0
        self._consecutive_failures = 0

    def start(self):
        if self.started:
            return self
            
        print(f"[CameraService] Attempting to open webcam index {self._camera_idx}...")
        self.stream = cv2.VideoCapture(self._camera_idx)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        
        self._actual_w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.grabbed, self.frame = self.stream.read()
        
        if not self.grabbed or self.frame is None:
            print(f"[CameraService Warning] Failed to open webcam at index {self._camera_idx}.")
            print("[CameraService Info] Initializing mock camera stream fallback.")
            self.use_mock = True
            self._actual_w = self._target_width
            self._actual_h = self._target_height
            self.grabbed = True
            self.frame = self._generate_mock_frame(0)
            
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        t_interval = 1.0 / self._target_fps
        frame_idx = 0
        
        while self.started:
            t_start = time.time()
            if self.use_mock:
                frame_idx += 1
                frame = self._generate_mock_frame(frame_idx)
                grabbed = True
            else:
                grabbed, frame = self.stream.read()
                
            if grabbed and frame is not None:
                self._consecutive_failures = 0
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_id += 1
                    
                # FPS Calculation
                self.fps_counter += 1
                now = time.time()
                if now - self.fps_start_time >= 1.0:
                    self._actual_fps = self.fps_counter / (now - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = now
            else:
                if not self.use_mock:
                    self._consecutive_failures += 1
                    print(f"[CameraService Warning] Failed to read frame from webcam. Consecutive failures: {self._consecutive_failures}")
                    if self._consecutive_failures >= 10:
                        print("[CameraService Error] 10 consecutive frame read failures. Falling back to mock camera stream.")
                        self.use_mock = True
                        self._actual_w = self._target_width
                        self._actual_h = self._target_height
                        self.grabbed = True
                        self.frame = self._generate_mock_frame(0)
                time.sleep(0.001)
                
            t_elapsed = time.time() - t_start
            sleep_time = max(0, t_interval - t_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame, self.frame_id

    def stop(self):
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stream and self.stream.isOpened():
            self.stream.release()
        print("[CameraService] Stream stopped and cleaned up.")

    @property
    def width(self) -> int:
        return self._actual_w

    @property
    def height(self) -> int:
        return self._actual_h

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    def _generate_mock_frame(self, frame_idx) -> np.ndarray:
        """Generates a dynamic placeholder frame simulating a real camera feed."""
        w, h = self._target_width, self._target_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw dynamic abstract grid lines
        grid_size = 80
        t = time.time()
        offset = int((t * 40) % grid_size)
        
        # Horizontal lines
        for y in range(offset, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (30, 25, 20), 1)
        # Vertical lines
        for x in range(offset, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (30, 25, 20), 1)
            
        # Draw a bouncing neon circle to represent tracking target or webcam movement
        speed_x = int(np.sin(t * 1.5) * (w / 3))
        speed_y = int(np.cos(t * 2.2) * (h / 3))
        center_x = w // 2 + speed_x
        center_y = h // 2 + speed_y
        
        # Neon cyan circle (representing a hand)
        cv2.circle(frame, (center_x, center_y), 45, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), 50, (255, 200, 0), 2, lineType=cv2.LINE_AA)
        
        # Add labels and mock indicator
        cv2.putText(frame, "MOCK CAMERA STREAM ACTIVE", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "NO WEBCAM CAPTURE DETECTED", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Resolution: {w}x{h} | Frame: {frame_idx}", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, lineType=cv2.LINE_AA)
        
        return frame
