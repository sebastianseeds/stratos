"""
File widget for displaying loaded data files
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QCheckBox, QLabel, 
    QPushButton, QComboBox
)


class LoadedFileWidget(QWidget):
    """Widget for displaying a loaded file with checkbox and type-specific controls"""
    
    def __init__(self, filename, file_type="flux", parent=None):
        """
        Initialize the loaded file widget.
        
        Args:
            filename: Path to the loaded file
            file_type: Type of file ("flux" or "orbital")
            parent: Parent widget
        """
        super().__init__(parent)
        self.filename = filename
        self.file_type = file_type
        self.vtk_data = None
        self.orbital_data = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI"""
        # Create horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Checkbox for visibility toggle
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)
        
        # Filename label (truncated if too long)
        display_name = Path(self.filename).name
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        self.label = QLabel(display_name)
        self.label.setToolTip(self.filename)  # Full path on hover
        layout.addWidget(self.label, 1)  # Stretch factor 1
        
        # Type-specific controls
        if self.file_type == "flux":
            # Particle type dropdown for flux files
            self.particle_combo = QComboBox()
            self.particle_combo.addItems([
                "electron",
                "proton",
                "alpha",
                "heavy_ion",
                "neutron",
                "gamma",
                "cosmic_ray"
            ])
            self.particle_combo.setMinimumWidth(100)
            layout.addWidget(self.particle_combo)
        else:
            # Color selector for orbital files
            self.color_combo = QComboBox()
            self.color_combo.addItems([
                "Yellow",
                "Cyan",
                "Magenta",
                "Green",
                "Orange",
                "White",
                "Red",
                "Blue"
            ])
            self.color_combo.setMinimumWidth(80)
            layout.addWidget(self.color_combo)
        
        # Remove button (X)
        self.remove_button = QPushButton("×")
        self.remove_button.setMaximumWidth(20)
        self.remove_button.setStyleSheet("""
            QPushButton {
                color: red;
                font-weight: bold;
                border: none;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 30);
            }
        """)
        layout.addWidget(self.remove_button)
    
    def get_particle_type(self):
        """Get selected particle type (flux files only)"""
        if self.file_type == "flux" and hasattr(self, 'particle_combo'):
            return self.particle_combo.currentText()
        return None
    
    def get_color(self):
        """Get selected color (orbital files only)"""
        if self.file_type == "orbital" and hasattr(self, 'color_combo'):
            return self.color_combo.currentText()
        return None
    
    def is_checked(self):
        """Check if the file is enabled for visualization"""
        return self.checkbox.isChecked()
    
    def get_display_name(self):
        """Get the display name for this file"""
        return self.label.text()
