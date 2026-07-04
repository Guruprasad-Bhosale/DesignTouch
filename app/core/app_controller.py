import time
import cv2
import threading
import numpy as np
from PyQt6.QtCore import QObject, QTimer

from app.core.interfaces import (
    ICameraService, ITrackingService, IStorageService,
    IConfigService, IModuleService
)
from app.core.service_manager import ServiceManager
from app.services.camera_service import CameraService
from app.services.tracking_service import TrackingService
from app.services.storage_service import StorageService
from app.services.config_service import ConfigService
from app.services.module_service import ModuleService

class AppController(QObject):
    def __init__(self, app_model):
        super().__init__()
        self._model = app_model
        self.running = False
        
        # Service placeholders
        self.config_service = None
        self.storage_service = None
        self.camera_service = None
        self.tracking_service = None
        self.module_service = None
        
        # Async threads
        self.tracking_thread = None
        
        # Fallback modes for missing assets
        self.manual_mouse_mode = False
        self.warning_message = None
        
        # Timer for polling updates to UI
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._on_ui_tick)
        
        self._init_services()

    def _init_services(self):
        # 1. Start Config and SQLite Services
        self.config_service = ConfigService()
        self.storage_service = StorageService()
        
        # Register in ServiceManager
        ServiceManager.register(IConfigService, self.config_service)
        ServiceManager.register(IStorageService, self.storage_service)
        
        # 2. Setup camera params
        cam_idx = self.config_service.get("camera_index", 0)
        fps_cap = self.config_service.get("fps_cap", 60)
        self.camera_service = CameraService(camera_idx=cam_idx, target_fps=fps_cap)
        ServiceManager.register(ICameraService, self.camera_service)
        
        # 3. Setup hand tracking
        self.tracking_service = TrackingService("hand_landmarker.task")
        # Sync sensitivity/threshold config parameters
        self.tracking_service.sensitivity = self.config_service.get("cursor_sensitivity", 1.5)
        self.tracking_service.selection_threshold = self.config_service.get("selection_threshold", 0.035)
        self.tracking_service.release_threshold = self.config_service.get("release_threshold", 0.055)
        ServiceManager.register(ITrackingService, self.tracking_service)
        
        # 4. Setup Dynamic module manager
        enabled_mods = self.config_service.get("enabled_modules", ["filter_mode", "sign_language"])
        self.module_service = ModuleService(ServiceManager, enabled_mods)
        ServiceManager.register(IModuleService, self.module_service)

    def start(self):
        print("[AppController] Starting GestureVerse backend engines...")
        self.running = True
        
        # 1. Start camera capture thread
        self.camera_service.start()
        
        # 2. Start tracking engines
        try:
            self.tracking_service.start()
        except Exception as e:
            print(f"[AppController Warning] Failed to start tracking service: {e}")
            self.manual_mouse_mode = True
            self.warning_message = "WARNING: MediaPipe model not found. Running in Mouse-Control Mode."
            # Set started to False in service so we don't try to call it
            self.tracking_service.started = False
        
        # 3. Load dynamic modules
        self.module_service.load_modules()
        
        # 4. Start background MediaPipe tracking loop thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        # 5. Start main UI ticks timer (aligned to 60 FPS = ~16 ms)
        fps_cap = self.config_service.get("fps_cap", 60)
        self.ui_timer.start(int(1000 / fps_cap))
        
        print("[AppController] Background threads and UI timers active.")

    def _tracking_loop(self):
        """Webcam frame consumer executing MediaPipe in the background."""
        last_processed_id = -1
        while self.running:
            if not self.tracking_service.started:
                time.sleep(0.05)
                continue
            success, frame, frame_id = self.camera_service.read()
            if success and frame is not None:
                if frame_id > last_processed_id:
                    flipped_frame = cv2.flip(frame, 1)
                    self.tracking_service.process_frame(flipped_frame)
                    last_processed_id = frame_id
                else:
                    time.sleep(0.002)
            else:
                time.sleep(0.002)

    def _on_ui_tick(self):
        """Ticks in alignment with GUI frames to coordinate Model state updates."""
        if not self.running:
            return
            
        success, frame, _ = self.camera_service.read()
        if not success or frame is None:
            return
            
        # 1. Calculate freeze frame state
        is_frozen = False if self.manual_mouse_mode else self.tracking_service.check_freeze()
        self._model.freeze_frame_active = is_frozen
        
        if not is_frozen or self._model.last_unfrozen_frame is None:
            self._model.last_unfrozen_frame = frame.copy()
            
        render_frame = self._model.last_unfrozen_frame if is_frozen else frame
        
        # Mirror the frame horizontally for standard display and text overlay preservation
        mirrored_frame = cv2.flip(render_frame, 1)
        
        # 2. Bundle tracking output
        if self.manual_mouse_mode:
            tracking_data = {
                "hand_present": {"Left": False, "Right": False},
                "smoothed_landmarks": {"Left": {}, "Right": {}},
                "cursor_pos": (0, 0),
                "click_state": False,
                "gesture_states": {
                    "Left": {"pinch": False, "fist": False, "open": False},
                    "Right": {"pinch": False, "fist": False, "open": False}
                },
                "warning_message": self.warning_message
            }
        else:
            tracking_data = {
                "hand_present": self.tracking_service.hand_present,
                "smoothed_landmarks": self.tracking_service.smoothed_landmarks,
                "cursor_pos": self.tracking_service.cursor_position,
                "click_state": self.tracking_service.click_state,
                "gesture_states": self.tracking_service.gesture_states
            }
        
        # 3. Process frame through active module pipeline
        active_mod = self._model.active_module_name
        module_instance = self.module_service.get_module(active_mod)
        
        if module_instance:
            # Module processes mirrored frame (CPU drawing falls here, GPU mapping coordinates returned)
            processed = module_instance.process_frame(mirrored_frame, tracking_data)
            self._model.camera_frame = processed
        else:
            self._model.camera_frame = mirrored_frame
            
        # Push tracking data to observers to trigger hover/click checks
        self._model.tracking_data = tracking_data

    def switch_module(self, module_name: str):
        """Changes the active execution space (menu, filter mode, or sign interpreter)."""
        print(f"[AppController] Switching execution space: {self._model.active_module_name} ➔ {module_name}")
        self._model.active_module_name = module_name

    def get_active_panels_geometry(self) -> list:
        """Assembles perspective warp quadrilateral dimensions if Filter Mode is active."""
        active_mod = self._model.active_module_name
        module_instance = self.module_service.get_module(active_mod)
        
        # Check if FilterModule
        if module_instance and hasattr(module_instance, "get_panel_geometry"):
            w, h = self.camera_service.width, self.camera_service.height
            geom = module_instance.get_panel_geometry(w, h)
            if geom:
                return [geom]
        return []

    # --- Sign Language Module interfaces forwarders ---

    def start_dataset_collection(self, label: str):
        sl_mod = self.module_service.get_module("sign_language")
        if sl_mod:
            sl_mod.trigger_collection(label)

    def stop_dataset_collection(self):
        sl_mod = self.module_service.get_module("sign_language")
        if sl_mod:
            sl_mod.stop_collection()

    def train_sign_model(self) -> bool:
        sl_mod = self.module_service.get_module("sign_language")
        if sl_mod:
            return sl_mod.train_model()
        return False

    # --- Config Reload Override ---

    def reload_settings(self):
        """Called when settings dialog is saved to override runtime configs."""
        self.config_service.load()
        
        # Reload camera settings
        cam_idx = self.config_service.get("camera_index", 0)
        fps_cap = self.config_service.get("fps_cap", 60)
        
        if self.camera_service._camera_idx != cam_idx:
            print(f"[AppController] Re-initializing camera to index {cam_idx}...")
            self.camera_service.stop()
            self.camera_service._camera_idx = cam_idx
            self.camera_service._target_fps = fps_cap
            self.camera_service.start()
            
        # Reload tracking settings
        self.tracking_service.sensitivity = self.config_service.get("cursor_sensitivity", 1.5)
        self.tracking_service.selection_threshold = self.config_service.get("selection_threshold", 0.035)
        self.tracking_service.release_threshold = self.config_service.get("release_threshold", 0.055)
        
        # Reset UI poll interval
        self.ui_timer.setInterval(int(1000 / fps_cap))
        print("[AppController] Runtime parameters updated successfully.")

    @property
    def camera_width(self) -> int:
        return self.camera_service.width

    @property
    def camera_height(self) -> int:
        return self.camera_service.height

    def stop(self):
        print("[AppController] Deactivating subsystems and background threads...")
        self.running = False
        self.ui_timer.stop()
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            
        self.module_service.deinitialize_all()
        self.tracking_service.stop()
        self.camera_service.stop()
        print("[AppController] Systems deactivated complete.")
