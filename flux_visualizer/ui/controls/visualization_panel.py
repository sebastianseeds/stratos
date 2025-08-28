"""
Visualization control panel for STRATOS
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSlider, QLabel, QWidget, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class VisualizationPanel(QGroupBox):
    """Panel for visualization controls"""
    
    # Signals
    mode_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    scale_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(int)
    settings_changed = pyqtSignal()
    
    def __init__(self, config, parent=None):
        """Initialize the visualization panel"""
        super().__init__("Field Visualization", parent)
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        self.main_layout = QVBoxLayout(self)
        
        # Visualization mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems([
            "Point Cloud",
            "Volume Rendering",
            "Isosurfaces",
            "Wireframe",
            "Slice Planes",
            "Surface with Edges"
        ])
        self.viz_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.viz_mode_combo)
        self.main_layout.addLayout(mode_layout)
        
        # Dynamic content area
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_widget)
        self.main_layout.addWidget(self.dynamic_widget)
        
        # Initialize with default mode
        self._update_dynamic_controls()
    
    def _on_mode_changed(self, mode):
        """Handle visualization mode change"""
        self._update_dynamic_controls()
        self.mode_changed.emit(mode)
    
    def _update_dynamic_controls(self):
        """Update controls based on selected mode"""
        # Clear existing widgets
        self._clear_layout(self.dynamic_layout)
        
        viz_mode = self.viz_mode_combo.currentText()
        
        # Common controls for all modes
        common_layout = QFormLayout()
        
        # Colormap selector
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.config.AVAILABLE_COLORMAPS)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        common_layout.addRow("Colormap:", self.colormap_combo)
        
        # Scale mode
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        self.scale_combo.currentTextChanged.connect(self.scale_changed.emit)
        common_layout.addRow("Scale:", self.scale_combo)
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("70%")
        opacity_layout.addWidget(self.opacity_label)
        common_layout.addRow("Opacity:", opacity_layout)
        
        self.dynamic_layout.addLayout(common_layout)
        
        # Add mode-specific controls
        if viz_mode == "Point Cloud":
            self._add_point_cloud_controls()
        elif viz_mode == "Volume Rendering":
            self._add_volume_controls()
        elif viz_mode == "Isosurfaces":
            self._add_isosurface_controls()
        elif viz_mode == "Slice Planes":
            self._add_slice_controls()
    
    def _add_point_cloud_controls(self):
        """Add point cloud specific controls"""
        pc_layout = QFormLayout()
        
        # Point density
        density_layout = QHBoxLayout()
        self.point_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_density_slider.setRange(self.config.MIN_POINT_DENSITY, 
                                          self.config.MAX_POINT_DENSITY)
        self.point_density_slider.setValue(self.config.DEFAULT_POINT_DENSITY)
        self.point_density_slider.valueChanged.connect(self._on_density_changed)
        density_layout.addWidget(self.point_density_slider)
        self.density_label = QLabel(f"{self.config.DEFAULT_POINT_DENSITY}")
        density_layout.addWidget(self.density_label)
        pc_layout.addRow("Density:", density_layout)
        
        # Point size
        size_layout = QHBoxLayout()
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(self.config.MIN_POINT_SIZE, 
                                       self.config.MAX_POINT_SIZE)
        self.point_size_slider.setValue(self.config.DEFAULT_POINT_SIZE)
        self.point_size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.point_size_slider)
        self.size_label = QLabel(f"{self.config.DEFAULT_POINT_SIZE}m")
        size_layout.addWidget(self.size_label)
        pc_layout.addRow("Size:", size_layout)
        
        self.dynamic_layout.addLayout(pc_layout)
    
    def _add_volume_controls(self):
        """Add volume rendering specific controls"""
        vol_layout = QFormLayout()
        
        # Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("50%")
        threshold_layout.addWidget(self.threshold_label)
        vol_layout.addRow("Threshold:", threshold_layout)
        
        # Quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Medium", "High Quality"])
        self.quality_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        vol_layout.addRow("Quality:", self.quality_combo)
        
        self.dynamic_layout.addLayout(vol_layout)
    
    def _add_isosurface_controls(self):
        """Add isosurface specific controls"""
        iso_layout = QFormLayout()
        
        # Number of surfaces
        self.num_surfaces_spin = QSpinBox()
        self.num_surfaces_spin.setRange(1, 10)
        self.num_surfaces_spin.setValue(3)
        self.num_surfaces_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        iso_layout.addRow("Surfaces:", self.num_surfaces_spin)
        
        # Surface levels
        level_layout = QHBoxLayout()
        self.min_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_level_slider.setRange(0, 100)
        self.min_level_slider.setValue(20)
        self.min_level_slider.valueChanged.connect(lambda: self.settings_changed.emit())
        level_layout.addWidget(QLabel("Min:"))
        level_layout.addWidget(self.min_level_slider)
        
        self.max_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_level_slider.setRange(0, 100)
        self.max_level_slider.setValue(80)
        self.max_level_slider.valueChanged.connect(lambda: self.settings_changed.emit())
        level_layout.addWidget(QLabel("Max:"))
        level_layout.addWidget(self.max_level_slider)
        
        iso_layout.addRow("Levels:", level_layout)
        
        self.dynamic_layout.addLayout(iso_layout)
    
    def _add_slice_controls(self):
        """Add slice plane specific controls"""
        slice_layout = QFormLayout()
        
        # Slice axis
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems(["X-Axis", "Y-Axis", "Z-Axis"])
        self.slice_axis_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        slice_layout.addRow("Axis:", self.slice_axis_combo)
        
        # Slice position
        pos_layout = QHBoxLayout()
        self.slice_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_pos_slider.setRange(0, 100)
        self.slice_pos_slider.setValue(50)
        self.slice_pos_slider.valueChanged.connect(self._on_slice_pos_changed)
        pos_layout.addWidget(self.slice_pos_slider)
        self.slice_pos_label = QLabel("50%")
        pos_layout.addWidget(self.slice_pos_label)
        slice_layout.addRow("Position:", pos_layout)
        
        # Multiple slices
        self.multi_slice_check = QCheckBox("Multiple Slices")
        self.multi_slice_check.stateChanged.connect(lambda: self.settings_changed.emit())
        slice_layout.addRow("", self.multi_slice_check)
        
        self.dynamic_layout.addLayout(slice_layout)
    
    def _on_opacity_changed(self, value):
        """Handle opacity change"""
        self.opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(value)
    
    def _on_density_changed(self, value):
        """Handle density change"""
        self.density_label.setText(str(value))
        self.settings_changed.emit()
    
    def _on_size_changed(self, value):
        """Handle size change"""
        self.size_label.setText(f"{value}m")
        self.settings_changed.emit()
    
    def _on_threshold_changed(self, value):
        """Handle threshold change"""
        self.threshold_label.setText(f"{value}%")
        self.settings_changed.emit()
    
    def _on_slice_pos_changed(self, value):
        """Handle slice position change"""
        self.slice_pos_label.setText(f"{value}%")
        self.settings_changed.emit()
    
    def _clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
    
    def get_current_mode(self):
        """Get current visualization mode"""
        return self.viz_mode_combo.currentText()
    
    def get_colormap(self):
        """Get current colormap"""
        if hasattr(self, 'colormap_combo'):
            return self.colormap_combo.currentText()
        return "Viridis"
    
    def get_scale(self):
        """Get current scale mode"""
        if hasattr(self, 'scale_combo'):
            return self.scale_combo.currentText()
        return "Linear"
    
    def get_opacity(self):
        """Get current opacity value (0-100)"""
        if hasattr(self, 'opacity_slider'):
            return self.opacity_slider.value()
        return 70
