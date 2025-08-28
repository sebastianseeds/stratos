"""
Earth visualization controls for STRATOS
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QSlider, QLabel,
    QPushButton, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class EarthControlsWidget(QWidget):
    """Widget for Earth-specific controls"""
    
    # Signals
    opacity_changed = pyqtSignal(int)
    grid_toggled = pyqtSignal(bool)
    reset_camera = pyqtSignal()
    
    def __init__(self, config, parent=None):
        """Initialize Earth controls"""
        super().__init__(parent)
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the controls UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Earth opacity slider
        layout.addWidget(QLabel("Earth Opacity:"))
        
        self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.earth_opacity_slider.setRange(0, 100)
        self.earth_opacity_slider.setValue(int(self.config.EARTH_DEFAULT_OPACITY * 100))
        self.earth_opacity_slider.setMaximumWidth(150)
        self.earth_opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.earth_opacity_slider)
        
        self.earth_opacity_label = QLabel(f"{int(self.config.EARTH_DEFAULT_OPACITY * 100)}%")
        self.earth_opacity_label.setMinimumWidth(40)
        layout.addWidget(self.earth_opacity_label)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox("Show Lat/Long Grid")
        self.grid_checkbox.setChecked(False)
        self.grid_checkbox.stateChanged.connect(self._on_grid_toggled)
        layout.addWidget(self.grid_checkbox)
        
        # Reset camera button
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.setMaximumWidth(100)
        self.reset_camera_button.clicked.connect(self.reset_camera.emit)
        layout.addWidget(self.reset_camera_button)
        
        layout.addStretch()
        
        # Apply styling
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 180);
                border: 1px solid #666;
                border-radius: 5px;
            }
            QLabel, QCheckBox {
                color: white;
            }
            QPushButton {
                background-color: rgba(70, 70, 70, 200);
                color: white;
                border: 1px solid #888;
                border-radius: 3px;
                padding: 3px 10px;
            }
            QPushButton:hover {
                background-color: rgba(90, 90, 90, 200);
            }
        """)
    
    def _on_opacity_changed(self, value):
        """Handle opacity slider change"""
        self.earth_opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(value)
    
    def _on_grid_toggled(self, state):
        """Handle grid checkbox toggle"""
        self.grid_toggled.emit(state == 2)  # Qt.Checked = 2
    
    def get_opacity(self):
        """Get current opacity value (0-100)"""
        return self.earth_opacity_slider.value()
    
    def is_grid_enabled(self):
        """Check if grid is enabled"""
        return self.grid_checkbox.isChecked()
