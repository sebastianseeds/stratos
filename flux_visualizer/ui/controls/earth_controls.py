# flux_visualizer/ui/controls/earth_controls.py
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
    orbital_paths_toggled = pyqtSignal(bool)
    trails_toggled = pyqtSignal(bool)
    satellite_size_changed = pyqtSignal(int)
    
    def __init__(self, config, parent=None):
        """Initialize Earth controls"""
        super().__init__(parent)
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the controls UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(15)
        
        # Earth opacity slider
        earth_label = QLabel("Earth Opacity:")
        earth_label.setStyleSheet("color: #aaa; font-size: 12px;")  # Lighter gray
        layout.addWidget(earth_label)
        
        self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.earth_opacity_slider.setRange(0, 100)
        self.earth_opacity_slider.setValue(int(self.config.EARTH_DEFAULT_OPACITY * 100))
        self.earth_opacity_slider.setMaximumWidth(100)
        self.earth_opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.earth_opacity_slider)
        
        self.earth_opacity_label = QLabel(f"{int(self.config.EARTH_DEFAULT_OPACITY * 100)}%")
        self.earth_opacity_label.setStyleSheet("color: #ddd; font-size: 12px; min-width: 35px;")
        layout.addWidget(self.earth_opacity_label)
        
        # Separator
        separator1 = QLabel("|")
        separator1.setStyleSheet("color: #555; font-size: 16px;")  # Lighter separator
        layout.addWidget(separator1)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox("Show Lat/Long Grid")
        self.grid_checkbox.setChecked(False)
        self.grid_checkbox.stateChanged.connect(self._on_grid_toggled)
        layout.addWidget(self.grid_checkbox)
        
        # Hide orbital paths checkbox
        self.hide_orbital_paths = QCheckBox("Hide Orbital Paths")
        self.hide_orbital_paths.setChecked(False)
        self.hide_orbital_paths.stateChanged.connect(self._on_orbital_paths_toggled)
        layout.addWidget(self.hide_orbital_paths)
        
        # Show trails checkbox
        self.show_trails = QCheckBox("Show Trails")
        self.show_trails.setChecked(True)
        self.show_trails.stateChanged.connect(self._on_trails_toggled)
        layout.addWidget(self.show_trails)
        
        # Separator
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #555; font-size: 16px;")
        layout.addWidget(separator2)
        
        # Satellite size slider
        sat_label = QLabel("Satellite Size:")
        sat_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(sat_label)
        
        self.satellite_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.satellite_size_slider.setRange(100, 2000)
        self.satellite_size_slider.setValue(500)
        self.satellite_size_slider.setMaximumWidth(80)
        self.satellite_size_slider.valueChanged.connect(self._on_satellite_size_changed)
        layout.addWidget(self.satellite_size_slider)
        
        self.satellite_size_label = QLabel("500km")
        self.satellite_size_label.setStyleSheet("color: #ddd; font-size: 12px; min-width: 50px;")
        layout.addWidget(self.satellite_size_label)
        
        layout.addStretch()
        
        # Apply flat styling - transparent background to blend with parent
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
            QCheckBox {
                color: #cccccc;
                font-size: 12px;
                spacing: 5px;
                background-color: transparent;
            }
            QCheckBox:hover {
                color: white;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #666;
                background-color: rgba(42, 42, 42, 180);
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 1px solid #45a049;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #999;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(58, 58, 58, 180);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #777;
                border: 1px solid #666;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #999;
                border: 1px solid #aaa;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 2px;
            }
        """)
        
        self.setFixedHeight(30)
    
    def _on_opacity_changed(self, value):
        """Handle opacity slider change"""
        self.earth_opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(value)
    
    def _on_grid_toggled(self, state):
        """Handle grid checkbox toggle"""
        self.grid_toggled.emit(state == 2)
    
    def _on_orbital_paths_toggled(self, state):
        """Handle orbital paths checkbox toggle"""
        hide_paths = state == 2
        show_paths = not hide_paths
        self.orbital_paths_toggled.emit(show_paths)
    
    def _on_trails_toggled(self, state):
        """Handle trails checkbox toggle"""
        self.trails_toggled.emit(state == 2)
    
    def _on_satellite_size_changed(self, value):
        """Handle satellite size slider change"""
        self.satellite_size_label.setText(f"{value}km")
        self.satellite_size_changed.emit(value)
    
    def get_opacity(self):
        """Get current opacity value (0-100)"""
        return self.earth_opacity_slider.value()
    
    def is_grid_enabled(self):
        """Check if grid is enabled"""
        return self.grid_checkbox.isChecked()
    
    def are_orbital_paths_visible(self):
        """Check if orbital paths are visible"""
        return not self.hide_orbital_paths.isChecked()
    
    def are_trails_visible(self):
        """Check if trails are visible"""
        return self.show_trails.isChecked()
    
    def get_satellite_size(self):
        """Get current satellite size in km"""
        return self.satellite_size_slider.value()
