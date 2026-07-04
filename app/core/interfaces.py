from abc import ABC, abstractmethod
import numpy as np

class ICameraService(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def read(self):
        """Returns (success, frame, frame_id)"""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def actual_fps(self) -> float:
        pass


class ITrackingService(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def process_frame(self, frame):
        pass

    @property
    @abstractmethod
    def smoothed_landmarks(self) -> dict:
        """Returns {'Left': {idx: np.array}, 'Right': {idx: np.array}}"""
        pass

    @property
    @abstractmethod
    def hand_present(self) -> dict:
        """Returns {'Left': bool, 'Right': bool}"""
        pass

    @property
    @abstractmethod
    def cursor_position(self) -> tuple:
        """Returns smoothed (x, y) in screen coordinates (0 to width, 0 to height)"""
        pass

    @property
    @abstractmethod
    def click_state(self) -> bool:
        """Returns True if a click is triggered on this frame"""
        pass

    @property
    @abstractmethod
    def gesture_states(self) -> dict:
        """Returns {'Left': {'pinch': bool, 'fist': bool, 'open': bool}, ...}"""
        pass

    @abstractmethod
    def check_freeze(self) -> bool:
        pass


class IStorageService(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_setting(self, key: str, default=None):
        pass

    @abstractmethod
    def set_setting(self, key: str, value):
        pass

    @abstractmethod
    def save_landmarks(self, label: str, landmarks: list, version: str):
        pass

    @abstractmethod
    def get_landmarks_dataset(self, version: str) -> list:
        """Returns list of tuples (label, x0, y0, z0, ... x20, y20, z20)"""
        pass

    @abstractmethod
    def get_dataset_versions(self) -> list:
        pass


class IConfigService(ABC):
    @abstractmethod
    def load(self) -> dict:
        pass

    @abstractmethod
    def save(self, config_data: dict):
        pass

    @abstractmethod
    def get(self, key: str, default=None):
        pass

    @abstractmethod
    def set(self, key: str, value):
        pass


class IModuleService(ABC):
    @abstractmethod
    def load_modules(self):
        pass

    @abstractmethod
    def get_module(self, name: str):
        pass

    @property
    @abstractmethod
    def available_modules(self) -> list:
        pass


class BaseModule(ABC):
    """Abstract base class for all pluggable feature modules."""
    
    @abstractmethod
    def initialize(self, service_manager) -> bool:
        """Set up the module with required services."""
        pass

    @abstractmethod
    def process_frame(self, frame, tracking_data: dict) -> np.ndarray:
        """Process and return the frame with effects applied."""
        pass

    @abstractmethod
    def handle_gesture(self, gesture_event: str):
        """Respond to visual gestures."""
        pass

    @abstractmethod
    def get_ui_widget(self):
        """Return a custom QWidget for settings/controls, if any, else None."""
        pass

    @abstractmethod
    def deinitialize(self):
        """Clean up resources."""
        pass
