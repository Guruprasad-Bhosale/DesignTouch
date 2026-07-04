import numpy as np

class AppStateModel:
    def __init__(self):
        self._active_module_name = "floating_menu"
        self._camera_frame = None
        self._mirrored_frame = None
        self._tracking_data = {
            "hand_present": {"Left": False, "Right": False},
            "smoothed_landmarks": {"Left": {}, "Right": {}},
            "cursor_pos": (0, 0),
            "click_state": False,
            "gesture_states": {
                "Left": {"pinch": False, "fist": False, "open": False},
                "Right": {"pinch": False, "fist": False, "open": False}
            }
        }
        self._freeze_frame_active = False
        self._last_unfrozen_frame = None
        self._current_fps = 0.0
        
        # Observers list
        self._observers = []

    def register_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister_observer(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, event_type: str, data=None):
        for observer in self._observers:
            try:
                observer.on_state_changed(event_type, data)
            except Exception as e:
                print(f"[Model Warning] Observer failed to process event '{event_type}': {e}")

    # --- Getters & Setters ---

    @property
    def active_module_name(self) -> str:
        return self._active_module_name

    @active_module_name.setter
    def active_module_name(self, value: str):
        if self._active_module_name != value:
            self._active_module_name = value
            self.notify_observers("active_module_changed", value)

    @property
    def camera_frame(self) -> np.ndarray:
        return self._camera_frame

    @camera_frame.setter
    def camera_frame(self, frame: np.ndarray):
        self._camera_frame = frame
        if frame is not None:
            # The frame is already mirrored horizontally in AppController
            self._mirrored_frame = np.ascontiguousarray(frame)
        else:
            self._mirrored_frame = None
        self.notify_observers("camera_frame_updated")

    @property
    def mirrored_frame(self) -> np.ndarray:
        return self._mirrored_frame

    @property
    def tracking_data(self) -> dict:
        return self._tracking_data

    @tracking_data.setter
    def tracking_data(self, data: dict):
        self._tracking_data = data
        self.notify_observers("tracking_data_updated", data)

    @property
    def freeze_frame_active(self) -> bool:
        return self._freeze_frame_active

    @freeze_frame_active.setter
    def freeze_frame_active(self, value: bool):
        if self._freeze_frame_active != value:
            self._freeze_frame_active = value
            self.notify_observers("freeze_frame_changed", value)

    @property
    def last_unfrozen_frame(self) -> np.ndarray:
        return self._last_unfrozen_frame

    @last_unfrozen_frame.setter
    def last_unfrozen_frame(self, frame: np.ndarray):
        self._last_unfrozen_frame = frame

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @current_fps.setter
    def current_fps(self, value: float):
        if self._current_fps != value:
            self._current_fps = value
            self.notify_observers("fps_updated", value)
