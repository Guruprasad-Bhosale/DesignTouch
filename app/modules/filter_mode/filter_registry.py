import os

class FilterRegistry:
    def __init__(self, shaders_dir="app/shaders"):
        self._shaders_dir = shaders_dir
        self._filters = []
        self._resolve_shaders_dir()
        self.scan_filters()

    def _resolve_shaders_dir(self):
        if not os.path.isabs(self._shaders_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            abs_path = os.path.join(base_dir, self._shaders_dir)
            if os.path.exists(abs_path):
                self._shaders_dir = abs_path
                
        print(f"[FilterRegistry] Shaders directory path resolved to: {self._shaders_dir}")

    def scan_filters(self):
        """Scans the shaders folder and registers all fragment shaders (.frag)."""
        self._filters = []
        if os.path.exists(self._shaders_dir):
            for file in os.listdir(self._shaders_dir):
                if file.endswith(".frag"):
                    # Effect name is the file name without extension
                    effect_name = file[:-5]
                    self._filters.append(effect_name)
            self._filters.sort()
                    
        # Fallbacks in case directory is empty/not found
        if not self._filters:
            self._filters = [
                "ThermalVision", "NightVision", "CyberpunkGlow", "EdgeDetection",
                "PixelSort", "WaterDistortion", "CRTMonitor", "RGB_Split",
                "Hologram", "Galaxy_Portal"
            ]
            
        print(f"[FilterRegistry] Registered filters: {self._filters}")

    @property
    def filters(self) -> list:
        return self._filters

    def get_shader_path(self, effect_name: str) -> str:
        return os.path.join(self._shaders_dir, f"{effect_name}.frag")

    def get_shader_source(self, effect_name: str) -> str:
        path = self.get_shader_path(effect_name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                print(f"[FilterRegistry Error] Failed to read shader file '{effect_name}': {e}")
        return ""
