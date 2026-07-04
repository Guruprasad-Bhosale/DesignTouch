from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

class FloatingMenu(QWidget):
    # Signals for menu clicks
    filter_mode_selected = pyqtSignal()
    sign_language_selected = pyqtSignal()
    settings_selected = pyqtSignal()
    exit_selected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(320, 360)
        
        self.init_ui()

    def init_ui(self):
        # Semi-transparent dark background with neon cyan border
        self.setObjectName("FloatingMenu")
        self.setStyleSheet("""
            QWidget#FloatingMenu {
                background-color: rgba(20, 16, 26, 225);
                border: 2px solid rgba(0, 255, 255, 180);
                border-radius: 20px;
            }
            QLabel {
                color: #00ffff;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QPushButton {
                background-color: rgba(30, 25, 40, 180);
                color: #e0e0e0;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 220px;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 255, 45);
                border: 2px solid rgba(0, 255, 255, 255);
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: rgba(0, 255, 255, 90);
                border: 2px solid rgba(0, 255, 255, 255);
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(18)
        
        # Futuristic HUD Header
        self.title_label = QLabel("GESTUREVERSE")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Consolas", 18, QFont.Weight.ExtraBold)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)
        
        # Menu Options
        self.btn_filter = QPushButton("INTERACTIVE FILTERS")
        self.btn_sign = QPushButton("SIGN LANGUAGE INTERPRETER")
        self.btn_settings = QPushButton("SETTINGS")
        self.btn_exit = QPushButton("EXIT SYSTEM")
        
        # Connect signals
        self.btn_filter.clicked.connect(self.filter_mode_selected.emit)
        self.btn_sign.clicked.connect(self.sign_language_selected.emit)
        self.btn_settings.clicked.connect(self.settings_selected.emit)
        self.btn_exit.clicked.connect(self.exit_selected.emit)
        
        layout.addWidget(self.btn_filter)
        layout.addWidget(self.btn_sign)
        layout.addWidget(self.btn_settings)
        layout.addWidget(self.btn_exit)
        
        self.setLayout(layout)
        
        # Add a subtle glowing shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 255, 255, 100))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
