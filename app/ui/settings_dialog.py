from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QDoubleSpinBox, QCheckBox, QPushButton, QTabWidget, QWidget, QListWidget, QListWidgetItem, QInputDialog
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from app.core.interfaces import IConfigService, IStorageService

class SettingsDialog(QDialog):
    # Signals for config changes
    config_updated = pyqtSignal()
    start_collection_requested = pyqtSignal(str)
    stop_collection_requested = pyqtSignal()
    train_model_requested = pyqtSignal()

    def __init__(self, service_manager, parent=None):
        super().__init__(parent)
        self._service_manager = service_manager
        self._config = service_manager.get(IConfigService)
        self._storage = service_manager.get(IStorageService)
        
        self.setWindowTitle("SYSTEM SETTINGS - GESTUREVERSE")
        self.setMinimumSize(520, 420)
        self.setWindowFlags(Qt.WindowType.Dialog)
        
        # Capture timer for recording countdown
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self._on_record_tick)
        self.countdown = 0
        
        self.init_ui()

    def init_ui(self):
        self.setObjectName("SettingsDialog")
        # Cyberpunk slate dark-mode styling
        self.setStyleSheet("""
            QDialog#SettingsDialog {
                background-color: #121016;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid rgba(0, 255, 255, 100);
                background: #181520;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1b1824;
                color: #b0b0b0;
                border: 1px solid rgba(255, 255, 255, 20);
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #181520;
                color: #00ffff;
                border-color: rgba(0, 255, 255, 100);
                border-bottom-color: #181520;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #242030;
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 4px;
                padding: 6px;
                min-width: 100px;
            }
            QComboBox:focus, QDoubleSpinBox:focus {
                border-color: #00ffff;
            }
            QCheckBox {
                color: #e0e0e0;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #242030;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #00ffff;
                border-color: #00ffff;
            }
            QPushButton {
                background-color: rgba(30, 25, 40, 255);
                color: #e0e0e0;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 255, 40);
                border-color: #00ffff;
                color: #ffffff;
            }
            QListWidget {
                background-color: #1a1822;
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 4px;
                color: #ffffff;
            }
        """)

        main_layout = QVBoxLayout()
        tabs = QTabWidget()
        
        # --- TAB 1: System Settings ---
        system_tab = QWidget()
        system_layout = QGridLayout()
        system_layout.setContentsMargins(20, 20, 20, 20)
        system_layout.setSpacing(15)
        
        # Camera selection
        system_layout.addWidget(QLabel("Webcam Device:"), 0, 0)
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["0 (Default)", "1", "2", "3"])
        self.cam_combo.setCurrentIndex(self._config.get("camera_index", 0))
        system_layout.addWidget(self.cam_combo, 0, 1)
        
        # FPS selection
        system_layout.addWidget(QLabel("Target Frame Rate:"), 1, 0)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["30 FPS", "60 FPS", "90 FPS"])
        current_fps = self._config.get("fps_cap", 60)
        if current_fps == 30: self.fps_combo.setCurrentIndex(0)
        elif current_fps == 60: self.fps_combo.setCurrentIndex(1)
        elif current_fps == 90: self.fps_combo.setCurrentIndex(2)
        system_layout.addWidget(self.fps_combo, 1, 1)
        
        # Sensitivity
        system_layout.addWidget(QLabel("Cursor Sensitivity:"), 2, 0)
        self.sens_spin = QDoubleSpinBox()
        self.sens_spin.setRange(0.5, 4.0)
        self.sens_spin.setSingleStep(0.1)
        self.sens_spin.setValue(self._config.get("cursor_sensitivity", 1.5))
        system_layout.addWidget(self.sens_spin, 2, 1)
        
        # Pinch Thresholds
        system_layout.addWidget(QLabel("Pinch Select Threshold:"), 3, 0)
        self.pinch_spin = QDoubleSpinBox()
        self.pinch_spin.setRange(0.01, 0.10)
        self.pinch_spin.setSingleStep(0.005)
        self.pinch_spin.setDecimals(3)
        self.pinch_spin.setValue(self._config.get("selection_threshold", 0.035))
        system_layout.addWidget(self.pinch_spin, 3, 1)
        
        system_layout.addWidget(QLabel("Pinch Release Threshold:"), 4, 0)
        self.release_spin = QDoubleSpinBox()
        self.release_spin.setRange(0.02, 0.15)
        self.release_spin.setSingleStep(0.005)
        self.release_spin.setDecimals(3)
        self.release_spin.setValue(self._config.get("release_threshold", 0.055))
        system_layout.addWidget(self.release_spin, 4, 1)
        
        system_tab.setLayout(system_layout)
        tabs.addTab(system_tab, "SYSTEM")
        
        # --- TAB 2: Filter Options ---
        filter_tab = QWidget()
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(15, 15, 15, 15)
        
        filter_layout.addWidget(QLabel("Manage Active Reality Shaders:"))
        
        self.filter_list = QListWidget()
        from app.modules.filter_mode.filter_registry import FilterRegistry
        registry = FilterRegistry()
        all_filters = registry.filters
        enabled_filters = self._config.get("enabled_filters", all_filters)
        
        for name in all_filters:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if name in enabled_filters:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.filter_list.addItem(item)
            
        filter_layout.addWidget(self.filter_list)
        filter_tab.setLayout(filter_layout)
        tabs.addTab(filter_tab, "REALITY SHADERS")
        
        # --- TAB 3: Sign Language Model Settings ---
        sl_tab = QWidget()
        sl_layout = QVBoxLayout()
        sl_layout.setContentsMargins(15, 15, 15, 15)
        sl_layout.setSpacing(12)
        
        # Dataset version
        ver_layout = QHBoxLayout()
        ver_layout.addWidget(QLabel("Dataset Version Tag:"))
        self.ver_label = QLabel(self._config.get("dataset_version", "v1.0"))
        self.ver_label.setStyleSheet("font-weight: bold; color: #00ffff;")
        ver_layout.addWidget(self.ver_label)
        self.btn_change_ver = QPushButton("NEW VERSION")
        self.btn_change_ver.clicked.connect(self._on_change_ver)
        ver_layout.addWidget(self.btn_change_ver)
        sl_layout.addLayout(ver_layout)
        
        # Training controls
        ctrl_layout = QGridLayout()
        ctrl_layout.addWidget(QLabel("Recording Label (A-Z):"), 0, 0)
        self.label_combo = QComboBox()
        self.label_combo.addItems([chr(i) for i in range(65, 91)]) # A-Z
        ctrl_layout.addWidget(self.label_combo, 0, 1)
        
        self.btn_capture = QPushButton("START RECORDING (5s)")
        self.btn_capture.clicked.connect(self._on_capture)
        ctrl_layout.addWidget(self.btn_capture, 1, 0, 1, 2)
        
        self.btn_train = QPushButton("TRAIN SCLEARN MODEL")
        self.btn_train.clicked.connect(self._on_train)
        self.btn_train.setStyleSheet("background-color: rgba(0, 255, 255, 20); border-color: rgba(0, 255, 255, 150);")
        ctrl_layout.addWidget(self.btn_train, 2, 0, 1, 2)
        
        self.status_label = QLabel("Ready to capture landmarks.")
        self.status_label.setStyleSheet("color: #b0b0b0;")
        ctrl_layout.addWidget(self.status_label, 3, 0, 1, 2)
        
        sl_layout.addLayout(ctrl_layout)
        sl_tab.setLayout(sl_layout)
        tabs.addTab(sl_tab, "SIGN LANGUAGE DATA")
        
        main_layout.addWidget(tabs)
        
        # Save / Cancel bottom buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_save = QPushButton("SAVE CHANGES")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setStyleSheet("background-color: #00ffff; color: #000000; font-weight: bold;")
        self.btn_cancel = QPushButton("CANCEL")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_save)
        main_layout.addLayout(btn_box)
        
        self.setLayout(main_layout)

    def _on_change_ver(self):
        new_ver, ok = QInputDialog.getText(self, "Dataset Version", "Enter new version tag (e.g. v1.1):")
        if ok and new_ver.strip():
            self.ver_label.setText(new_ver.strip())

    def _on_capture(self):
        if self.btn_capture.text().startswith("START"):
            label = self.label_combo.currentText()
            self.start_collection_requested.emit(label)
            
            # Start timer countdown
            self.countdown = 5  # 5 seconds capture
            self.btn_capture.setText("RECORDING... (5s)")
            self.btn_capture.setEnabled(False)
            self.status_label.setText(f"Hold '{label}' gesture in front of camera...")
            self.record_timer.start(1000)
            
    def _on_record_tick(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.btn_capture.setText(f"RECORDING... ({self.countdown}s)")
        else:
            self.record_timer.stop()
            self.stop_collection_requested.emit()
            self.btn_capture.setEnabled(True)
            self.btn_capture.setText("START RECORDING (5s)")
            self.status_label.setText("Landmarks capture complete! Sample saved to CSV and SQLite.")

    def _on_train(self):
        self.status_label.setText("Training model... please wait.")
        self.btn_train.setEnabled(False)
        # Give UI thread a moment to update display
        QTimer.singleShot(500, self._trigger_training)

    def _trigger_training(self):
        self.train_model_requested.emit()

    def set_training_status(self, success: bool, message: str):
        self.btn_train.setEnabled(True)
        color = "#a0ffa0" if success else "#ffa0a0"
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

    def _on_save(self):
        # 1. Update config service values
        self._config.set("camera_index", self.cam_combo.currentIndex())
        
        fps_opts = [30, 60, 90]
        self._config.set("fps_cap", fps_opts[self.fps_combo.currentIndex()])
        self._config.set("cursor_sensitivity", self.sens_spin.value())
        self._config.set("selection_threshold", self.pinch_spin.value())
        self._config.set("release_threshold", self.release_spin.value())
        self._config.set("dataset_version", self.ver_label.text())
        
        # Active filters checkbox processing
        enabled_shs = []
        for i in range(self.filter_list.count()):
            item = self.filter_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                enabled_shs.append(item.text())
        self._config.set("enabled_filters", enabled_shs)
        
        self.config_updated.emit()
        self.accept()
