import sys
import os

# Adjust path so imports work correctly when running from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from app.core.app_model import AppStateModel
from app.core.app_controller import AppController
from app.core.service_manager import ServiceManager
from app.ui.main_window import MainWindow

def main():
    print("[Launcher] Starting GestureVerse Application Bootloader...")
    
    # 1. Initialize Qt Application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 2. Setup MVC elements
    model = AppStateModel()
    controller = AppController(model)
    
    # 3. Create Main Window
    window = MainWindow(ServiceManager, model, controller)
    
    # 4. Start controller polling loop and background threads
    controller.start()
    
    # 5. Open display
    window.show()
    
    print("[Launcher] GUI window visible. Entering Qt event loop.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
