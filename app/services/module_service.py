import importlib
import os
from app.core.interfaces import IModuleService, BaseModule

class ModuleService(IModuleService):
    def __init__(self, service_manager, enabled_module_names: list):
        self._service_manager = service_manager
        self._enabled_names = enabled_module_names
        self._loaded_modules = {}
        self._available_modules = []
        
        self._scan_available_modules()

    def _scan_available_modules(self):
        """Scans the app/modules folder for subdirectories representing pluggable modules."""
        modules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules")
        if os.path.exists(modules_dir):
            for entry in os.scandir(modules_dir):
                if entry.is_dir() and not entry.name.startswith("__"):
                    self._available_modules.append(entry.name)
        else:
            # Fallback if path scanning fails
            self._available_modules = ["filter_mode", "sign_language"]
        print(f"[ModuleService] Scanned available modules: {self._available_modules}")

    def load_modules(self):
        """Loads and initializes enabled modules dynamically."""
        for name in self._enabled_names:
            if name in self._loaded_modules:
                continue
                
            try:
                print(f"[ModuleService] Dynamically loading module '{name}'...")
                # Map folder name to module class file
                # e.g., modules/filter_mode/ -> app.modules.filter_mode.filter_module
                if name == "filter_mode":
                    module_path = "app.modules.filter_mode.filter_module"
                    class_name = "FilterModule"
                elif name == "sign_language":
                    module_path = "app.modules.sign_language.sign_language_module"
                    class_name = "SignLanguageModule"
                else:
                    # Generic mapping fallback
                    module_path = f"app.modules.{name}.{name}_module"
                    # CamelCase formatting
                    class_name = "".join(part.capitalize() for part in name.split("_")) + "Module"
                
                module_lib = importlib.import_module(module_path)
                module_class = getattr(module_lib, class_name)
                
                # Instantiate and initialize module
                instance = module_class()
                success = instance.initialize(self._service_manager)
                if success:
                    self._loaded_modules[name] = instance
                    print(f"[ModuleService] Module '{name}' successfully loaded and initialized.")
                else:
                    print(f"[ModuleService Error] Failed to initialize module '{name}'")
                    
            except Exception as e:
                print(f"[ModuleService Error] Failed to load module '{name}': {e}")

    def get_module(self, name: str) -> BaseModule:
        return self._loaded_modules.get(name)

    @property
    def available_modules(self) -> list:
        return self._available_modules

    @property
    def loaded_modules(self) -> dict:
        return self._loaded_modules
        
    def deinitialize_all(self):
        for name, instance in self._loaded_modules.items():
            try:
                instance.deinitialize()
                print(f"[ModuleService] Module '{name}' deinitialized.")
            except Exception as e:
                print(f"[ModuleService Error] Failed to deinitialize module '{name}': {e}")
        self._loaded_modules.clear()
