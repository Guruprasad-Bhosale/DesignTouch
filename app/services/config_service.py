import json
import os
from app.core.interfaces import IConfigService

class ConfigService(IConfigService):
    def __init__(self, config_path="app/config/config.json"):
        self._config_path = config_path
        self._config_data = {}
        self.load()

    def load(self) -> dict:
        if not os.path.exists(self._config_path):
            # Try finding it relative to project base directory if path is relative
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            alt_path = os.path.join(base_dir, self._config_path)
            if os.path.exists(alt_path):
                self._config_path = alt_path
                
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    self._config_data = json.load(f)
                print(f"[ConfigService] Config loaded successfully from {self._config_path}")
            except Exception as e:
                print(f"[ConfigService Error] Failed to read config file: {e}")
                self._load_defaults()
        else:
            print(f"[ConfigService Warning] Config file not found at {self._config_path}. Loading defaults.")
            self._load_defaults()
            self.save(self._config_data)
            
        return self._config_data

    def _load_defaults(self):
        self._config_data = {
            "camera_index": 0,
            "fps_cap": 60,
            "cursor_sensitivity": 1.5,
            "selection_threshold": 0.035,
            "release_threshold": 0.055,
            "filter_cooldown_ms": 1000,
            "enabled_modules": ["filter_mode", "sign_language"],
            "enabled_filters": [
                "ThermalVision", "NightVision", "CyberpunkGlow", "EdgeDetection",
                "PixelSort", "WaterDistortion", "CRTMonitor", "RGB_Split",
                "Hologram", "Galaxy_Portal"
            ],
            "sign_language_model": "app/models/sign_language_cls.pkl",
            "dataset_version": "v1.0"
        }

    def save(self, config_data: dict = None):
        if config_data is not None:
            self._config_data = config_data
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        try:
            with open(self._config_path, "w") as f:
                json.dump(self._config_data, f, indent=2)
            print(f"[ConfigService] Config saved successfully to {self._config_path}")
        except Exception as e:
            print(f"[ConfigService Error] Failed to write config file: {e}")

    def get(self, key: str, default=None):
        return self._config_data.get(key, default)

    def set(self, key: str, value):
        self._config_data[key] = value
        self.save()
