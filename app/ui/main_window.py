import sys
import os
import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QKeyEvent, QIcon

from app.ui.camera_view import CameraView
from app.ui.floating_menu import FloatingMenu
from app.ui.settings_dialog import SettingsDialog
from app.core.interfaces import IConfigService, IStorageService

class TrainingWorker(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, controller):
        super().__init__()
        self._controller = controller
        
    def run(self):
        try:
            success = self._controller.train_sign_model()
            if success:
                self.finished.emit(True, "RandomForest fit successful! Model reloaded.")
            else:
                self.finished.emit(False, "Error: fitting failed. Check dataset samples.")
        except Exception as e:
            self.finished.emit(False, f"Exception during training: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self, service_manager, app_model, app_controller):
        super().__init__()
        self._service_manager = service_manager
        self._model = app_model
        self._controller = app_controller
        
        self.setWindowTitle("GestureVerse - Interactive AI Space")
        self.resize(1280, 720)
        self.setMinimumSize(1024, 576)
        
        # Sync dark theme color palette
        self.setStyleSheet("background-color: #0c0a0f;")
        
        self.init_ui()
        
        # Register observer on the app state model
        self._model.register_observer(self)
        
        # Tracks last widget hover to handle Leave/Enter events cleanly
        self._last_hovered_widget = None
        
        # Dwell select/hover selection timing variables
        self._hover_start_time = 0.0
        self._current_hovered_btn = None
        self._dwell_time_threshold = 3.0 # 3 seconds

    def init_ui(self):
        # 1. Base Container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Absolute layout for floating widgets
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 2. Main OpenGL / Fallback Camera View (fills window)
        self.camera_view = CameraView(self.central_widget)
        self.layout.addWidget(self.camera_view)
        
        # 3. Floating Menu Overlay (instantiated as child of camera view)
        self.floating_menu = FloatingMenu(self.camera_view)
        
        # 4. Floating Back to Menu button (futuristic layout in top-left)
        self.btn_back = QPushButton("◀ MAIN MENU", self.camera_view)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 16, 26, 220);
                color: #00ffff;
                border: 2px solid rgba(0, 255, 255, 150);
                border-radius: 6px;
                padding: 10px 15px;
                font-weight: bold;
                font-family: 'Consolas';
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 255, 60);
                border-color: #00ffff;
                color: #ffffff;
            }
        """)
        self.btn_back.setGeometry(20, 20, 140, 40)
        self.btn_back.clicked.connect(self._on_back_to_menu)
        self.btn_back.hide()  # Hidden initially (since menu is active)
        
        # Connect menu actions
        self.floating_menu.filter_mode_selected.connect(self._on_filter_mode)
        self.floating_menu.sign_language_selected.connect(self._on_sign_language)
        self.floating_menu.settings_selected.connect(self._on_settings)
        self.floating_menu.exit_selected.connect(self.close)
        
        # Center menu overlay
        self._reposition_menu()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_menu()

    def _reposition_menu(self):
        # Center the floating menu panel inside the viewport
        mw = self.floating_menu.width()
        mh = self.floating_menu.height()
        cw = self.camera_view.width()
        ch = self.camera_view.height()
        
        # Position centered
        self.floating_menu.setGeometry(
            (cw - mw) // 2,
            (ch - mh) // 2,
            mw, mh
        )
        self.floating_menu.raise_()  # Ensure menu is drawn on top of OpenGL widget

    def _on_filter_mode(self):
        self._controller.switch_module("filter_mode")

    def _on_sign_language(self):
        self._controller.switch_module("sign_language")

    def _on_settings(self):
        # Open system settings dialog
        dialog = SettingsDialog(self._service_manager, self)
        
        # Connect training triggers
        dialog.start_collection_requested.connect(self._controller.start_dataset_collection)
        dialog.stop_collection_requested.connect(self._controller.stop_dataset_collection)
        
        def handle_training():
            self._training_worker = TrainingWorker(self._controller)
            self._training_worker.finished.connect(dialog.set_training_status)
            self._training_worker.start()
                
        dialog.train_model_requested.connect(handle_training)
        
        # Sync sensitivity values back if saved
        def sync_config():
            self._controller.reload_settings()
            
        dialog.config_updated.connect(sync_config)
        dialog.exec()

    def _on_back_to_menu(self):
        self._controller.switch_module("floating_menu")

    # --- Observer state change callback ---
    
    def on_state_changed(self, event_type: str, data=None):
        if event_type == "active_module_changed":
            module_name = data
            self.camera_view.active_module_name = module_name
            
            if module_name == "floating_menu":
                self.floating_menu.show()
                self.floating_menu.raise_()
                self.btn_back.hide()
            else:
                self.floating_menu.hide()
                self.btn_back.show()
                self.btn_back.raise_()
                
            self._reposition_menu()
            
        elif event_type == "camera_frame_updated":
            # Pass updated webcam buffers down to openGL widgets
            frame = self._model.mirrored_frame
            tracking_data = self._model.tracking_data
            
            # Map virtual cursor inputs to Qt GUI events (runs first to inject dwell progress)
            self._dispatch_virtual_cursor_events()
            
            # Retrieve active panels geometry
            panels = self._controller.get_active_panels_geometry()
            self.camera_view.update_frame_data(frame, tracking_data, panels)

    def _dispatch_virtual_cursor_events(self):
        """Translates index tip and pinch tracking inputs to native hover and click Qt events."""
        tracking = self._model.tracking_data
        if not tracking or "warning_message" in tracking or getattr(self._controller, "manual_mouse_mode", False):
            return
            
        # Coordinates mapped in tracking service (relative to frame dimensions)
        cx, cy = tracking.get("cursor_pos", (0, 0))
        is_click = tracking.get("click_state", False)
        
        # Translate to viewport local points
        fw, fh = self._controller.camera_width, self._controller.camera_height
        if fw > 0 and fh > 0:
            local_x = int(cx * self.width() / fw)
            local_y = int(cy * self.height() / fh)
        else:
            return
            
        local_point = QPoint(local_x, local_y)
        global_point = self.mapToGlobal(local_point)
        
        # Find QPushButton target child under cursor for hover dwell selection logic
        target_child = self.childAt(local_point)
        hovered_btn = None
        temp = target_child
        while temp:
            if isinstance(temp, QPushButton):
                hovered_btn = temp
                break
            temp = temp.parentWidget()
            
        # Dwell Selection Logic (Hover over buttons for 3 seconds triggers click)
        dwell_progress = 0.0
        if hovered_btn:
            if hovered_btn != self._current_hovered_btn:
                self._current_hovered_btn = hovered_btn
                self._hover_start_time = time.time()
            else:
                elapsed = time.time() - self._hover_start_time
                if elapsed < 0.0:
                    # Hover dwell locked after trigger until cursor leaves button
                    dwell_progress = 1.0
                elif elapsed >= self._dwell_time_threshold:
                    is_click = True  # Force virtual click trigger
                    self._hover_start_time = time.time() - 999.0  # Set lock state
                    dwell_progress = 1.0
                    print(f"[UI Virtual Cursor] Dwell selection triggered on: {hovered_btn.text()}")
                else:
                    dwell_progress = elapsed / self._dwell_time_threshold
        else:
            self._current_hovered_btn = None
            self._hover_start_time = 0.0
            
        tracking["dwell_progress"] = dwell_progress
        
        # Map parameters to QPointF to satisfy PyQt6 QMouseEvent signature
        local_point_f = QPointF(local_point)
        global_point_f = QPointF(global_point)
        
        # 1. Hover Move simulation posted directly to MainWindow
        move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            local_point_f,
            global_point_f,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier
        )
        QApplication.postEvent(self, move_event)
        
        # 2. Virtual Pinch Mouse Button simulation posted directly to MainWindow
        if is_click:
            press_event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                local_point_f,
                global_point_f,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier
            )
            QApplication.postEvent(self, press_event)
            
            # Send immediate release event to complete select action
            release_event = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                local_point_f,
                global_point_f,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier
            )
            QApplication.postEvent(self, release_event)

    def keyPressEvent(self, event: QKeyEvent):
        # Support ESC key to return to menu or exit
        if event.key() == Qt.Key.Key_Escape:
            if self._model.active_module_name != "floating_menu":
                self._on_back_to_menu()
            else:
                self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        # Shut down services cleanly
        self._controller.stop()
        super().closeEvent(event)
