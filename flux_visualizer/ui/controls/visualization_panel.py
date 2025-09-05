"""
Visualization control panel for STRATOS
"""

print("LOADING visualization_panel.py - CHANGES SHOULD BE VISIBLE NOW")

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSlider, QLabel, QWidget, QCheckBox, QSpinBox, QLineEdit, QSizePolicy
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
        self.loaded_files = {}  # Track loaded files
        
        # Set size policy to be more flexible
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(4)  # Tighter vertical spacing
        self.main_layout.setContentsMargins(8, 8, 8, 8)  # Tighter margins
        
        # Flux files status (simple label, no dropdown)
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Files:"))
        self.files_status_label = QLabel("No flux files loaded")
        self.files_status_label.setStyleSheet("font-weight: bold; color: #666;")
        file_layout.addWidget(self.files_status_label)
        
        self.main_layout.addLayout(file_layout)
        
        # Visualization mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems([
            "Point Cloud",
            "Isosurfaces",
            "Slice Planes"
        ])
        self.viz_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.viz_mode_combo)
        self.main_layout.addLayout(mode_layout)
        
        # Dynamic content area
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_widget)
        self.dynamic_layout.setSpacing(3)  # Compact spacing between dynamic elements
        self.main_layout.addWidget(self.dynamic_widget)
        
        # Initialize with default mode
        self._update_dynamic_controls()
    
    def _on_mode_changed(self, mode):
        """Handle visualization mode change"""
        self._update_dynamic_controls()
        self.mode_changed.emit(mode)
    
    
    def _on_colormap_combo_changed(self, text):
        """Handle colormap combo change"""
        self.colormap_changed.emit(text)
    
    def _update_dynamic_controls(self):
        """Update controls based on selected mode"""
        # Clear existing widgets
        self._clear_layout(self.dynamic_layout)
        
        viz_mode = self.viz_mode_combo.currentText()
        
        # Common controls for all modes with aligned labels
        common_layout = QVBoxLayout()
        
        # Colormap selector
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.config.AVAILABLE_COLORMAPS)
        self.colormap_combo.currentTextChanged.connect(lambda text: self._on_colormap_combo_changed(text))
        colormap_row = self._create_aligned_row("Colormap", self.colormap_combo)
        common_layout.addLayout(colormap_row)
        
        # Scale mode
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        self.scale_combo.currentTextChanged.connect(self.scale_changed.emit)
        scale_row = self._create_aligned_row("Scale", self.scale_combo)
        common_layout.addLayout(scale_row)
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        # Remove fixed width - let it scale with available space
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("70%")
        self.opacity_label.setMinimumWidth(40)
        opacity_layout.addWidget(self.opacity_label)
        opacity_row = self._create_aligned_row("Opacity", opacity_layout)
        common_layout.addLayout(opacity_row)
        
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
        
        # Add flux range controls for all modes
        self._add_flux_range_controls()
    
    def _add_point_cloud_controls(self):
        """Add point cloud specific controls"""
        pc_layout = QVBoxLayout()
        
        # Point density
        density_layout = QHBoxLayout()
        self.point_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_density_slider.setRange(self.config.MIN_POINT_DENSITY, 
                                          self.config.MAX_POINT_DENSITY)
        self.point_density_slider.setValue(self.config.DEFAULT_POINT_DENSITY)
        # Remove fixed width - let it scale with available space
        self.point_density_slider.valueChanged.connect(self._on_density_changed)
        density_layout.addWidget(self.point_density_slider)
        self.density_label = QLabel(f"{self.config.DEFAULT_POINT_DENSITY}")
        self.density_label.setMinimumWidth(40)
        density_layout.addWidget(self.density_label)
        
        # Add tooltip for density
        self.point_density_slider.setToolTip(
            "Maximum number of points to display from the flux field data.\n"
            "Higher values show more detail but may impact performance."
        )
        
        density_row = self._create_aligned_row("Density", density_layout)
        pc_layout.addLayout(density_row)
        
        # Point size
        size_layout = QHBoxLayout()
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(self.config.MIN_POINT_SIZE, 
                                       self.config.MAX_POINT_SIZE)
        self.point_size_slider.setValue(self.config.DEFAULT_POINT_SIZE)
        # Remove fixed width - let it scale with available space
        self.point_size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.point_size_slider)
        self.size_label = QLabel(f"{self.config.DEFAULT_POINT_SIZE}km")
        self.size_label.setMinimumWidth(40)
        size_layout.addWidget(self.size_label)
        
        # Add tooltip for size
        self.point_size_slider.setToolTip("Radius of each point sphere in kilometers")
        
        size_row = self._create_aligned_row("Size", size_layout)
        pc_layout.addLayout(size_row)
        
        self.dynamic_layout.addLayout(pc_layout)
    
    def _add_volume_controls(self):
        """Add volume rendering specific controls"""
        vol_layout = QFormLayout()
        
        # Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        # Remove fixed width - let it scale with available space
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("50%")
        self.threshold_label.setMinimumWidth(40)
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
        iso_layout = QVBoxLayout()
        
        # Isosurface style selector
        style_row = self._create_aligned_row("Style", None)  # We'll add the combo manually
        self.isosurface_style_combo = QComboBox()
        self.isosurface_style_combo.addItems([
            "Single Isosurface",
            "Multiple Isosurface"
        ])
        self.isosurface_style_combo.currentTextChanged.connect(self._on_isosurface_style_changed)
        # Replace the None placeholder with the actual combo
        style_row.addWidget(self.isosurface_style_combo)
        iso_layout.addLayout(style_row)
        
        # Single Isosurface Controls (initially visible)
        self.single_iso_controls = QWidget()
        single_layout = QVBoxLayout(self.single_iso_controls)
        single_layout.setContentsMargins(0, 3, 0, 3)  # Tighter margins
        
        # Title for the control
        iso_title = QLabel("Contour Level:")
        iso_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        single_layout.addWidget(iso_title)
        
        # Slider gets full width (like volume threshold)
        self.isosurface_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.isosurface_level_slider.setRange(10, 90)
        self.isosurface_level_slider.setValue(50)
        self.isosurface_level_slider.valueChanged.connect(self._on_isosurface_level_changed)
        # Remove fixed width - let it scale with available space
        self.isosurface_level_slider.setToolTip("Adjust the contour level for single isosurface")
        single_layout.addWidget(self.isosurface_level_slider)
        
        # Label below the slider (centered)
        self.isosurface_level_label = QLabel("50th flux percentile")
        self.isosurface_level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.isosurface_level_label.setStyleSheet("color: #5C6A72; font-size: 11px; margin-top: 2px;")
        single_layout.addWidget(self.isosurface_level_label)
        
        iso_layout.addWidget(self.single_iso_controls)
        
        # Multiple Isosurface Controls (initially hidden)
        self.multi_iso_controls = QWidget()
        multi_layout = QVBoxLayout(self.multi_iso_controls)
        multi_layout.setContentsMargins(0, 3, 0, 3)  # Tighter margins
        
        # Title
        multi_title = QLabel("Multiple Isosurface Levels:")
        multi_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        multi_layout.addWidget(multi_title)
        
        # Create multiple level controls
        self.multi_level_controls = []
        default_percents = [20, 40, 60, 80]
        for idx, pct in enumerate(default_percents, start=1):
            level_row = QHBoxLayout()
            
            cb = QCheckBox(f"Level {idx}")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda: self.settings_changed.emit())
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(10, 90)
            slider.setValue(pct)
            slider.valueChanged.connect(lambda: self.settings_changed.emit())
            
            value_label = QLabel(f"{pct}th %")
            value_label.setMinimumWidth(60)
            value_label.setStyleSheet("color: #5C6A72; font-size: 10px;")
            
            # Connect slider to label update
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}th %"))
            
            self.multi_level_controls.append((cb, slider, value_label))
            
            level_row.addWidget(cb)
            level_row.addWidget(slider)
            level_row.addWidget(value_label)
            multi_layout.addLayout(level_row)
        
        iso_layout.addWidget(self.multi_iso_controls)
        
        # Initially hide multiple controls
        self.multi_iso_controls.setVisible(False)
        
        self.dynamic_layout.addLayout(iso_layout)
    
    def _add_slice_controls(self):
        """Add slice plane specific controls"""
        slice_layout = QVBoxLayout()
        
        # Slice style selector (similar to isosurfaces)
        style_row = self._create_aligned_row("Style", None)
        self.slice_style_combo = QComboBox()
        self.slice_style_combo.addItems([
            "Single Slice",
            "Three Plane Slice"
        ])
        self.slice_style_combo.currentTextChanged.connect(self._on_slice_style_changed)
        style_row.addWidget(self.slice_style_combo)
        slice_layout.addLayout(style_row)
        
        # Single Slice Controls (initially visible)
        self.single_slice_controls = QWidget()
        single_slice_layout = QVBoxLayout(self.single_slice_controls)
        single_slice_layout.setContentsMargins(0, 3, 0, 3)  # Tighter margins
        
        # Slice axis/orientation for single slice
        axis_row = self._create_aligned_row("Orientation", None)
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems([
            "X-Axis (YZ Plane)",
            "Y-Axis (XZ Plane)", 
            "Z-Axis (XY Plane)"
        ])
        self.slice_axis_combo.setCurrentText("Z-Axis (XY Plane)")
        self.slice_axis_combo.currentTextChanged.connect(self._on_slice_axis_changed)
        axis_row.addWidget(self.slice_axis_combo)
        single_slice_layout.addLayout(axis_row)
        
        # Slice position with label
        pos_title = QLabel("Slice Position:")
        pos_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px; margin-top: 5px;")
        single_slice_layout.addWidget(pos_title)
        
        # Position slider
        self.slice_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_pos_slider.setRange(0, 100)
        self.slice_pos_slider.setValue(50)
        # Remove fixed width - let it scale with available space
        self.slice_pos_slider.valueChanged.connect(self._on_slice_pos_changed)
        # Also emit settings changed to trigger main window updates
        self.slice_pos_slider.valueChanged.connect(lambda: self.settings_changed.emit())
        self.slice_pos_slider.setToolTip("Adjust the position of the slice plane")
        single_slice_layout.addWidget(self.slice_pos_slider)
        
        # Position label (shows coordinate)
        self.slice_pos_label = QLabel("Center (0 km)")
        self.slice_pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_pos_label.setStyleSheet("color: #5C6A72; font-size: 11px; margin-top: 2px;")
        single_slice_layout.addWidget(self.slice_pos_label)
        
        slice_layout.addWidget(self.single_slice_controls)
        
        # Three Plane Slice Controls (initially hidden)
        self.three_plane_controls = QWidget()
        three_plane_layout = QVBoxLayout(self.three_plane_controls)
        three_plane_layout.setContentsMargins(0, 3, 0, 3)  # Tighter margins
        
        # Title and description
        three_plane_title = QLabel("Three Orthogonal Slices:")
        three_plane_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        three_plane_layout.addWidget(three_plane_title)
        
        three_plane_desc = QLabel("XY, XZ, and YZ planes intersecting at center")
        three_plane_desc.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 5px;")
        three_plane_layout.addWidget(three_plane_desc)
        
        # Position controls for three planes
        three_pos_title = QLabel("Intersection Position:")
        three_pos_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px; margin-top: 5px;")
        three_plane_layout.addWidget(three_pos_title)
        
        # X Position
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.three_plane_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.three_plane_x_slider.setRange(0, 100)
        self.three_plane_x_slider.setValue(50)
        self.three_plane_x_slider.valueChanged.connect(self._on_three_plane_changed)
        x_layout.addWidget(self.three_plane_x_slider)
        self.three_plane_x_label = QLabel("50%")
        self.three_plane_x_label.setMinimumWidth(35)
        x_layout.addWidget(self.three_plane_x_label)
        three_plane_layout.addLayout(x_layout)
        
        # Y Position
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.three_plane_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.three_plane_y_slider.setRange(0, 100)
        self.three_plane_y_slider.setValue(50)
        self.three_plane_y_slider.valueChanged.connect(self._on_three_plane_changed)
        y_layout.addWidget(self.three_plane_y_slider)
        self.three_plane_y_label = QLabel("50%")
        self.three_plane_y_label.setMinimumWidth(35)
        y_layout.addWidget(self.three_plane_y_label)
        three_plane_layout.addLayout(y_layout)
        
        # Z Position  
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.three_plane_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.three_plane_z_slider.setRange(0, 100)
        self.three_plane_z_slider.setValue(50)
        self.three_plane_z_slider.valueChanged.connect(self._on_three_plane_changed)
        z_layout.addWidget(self.three_plane_z_slider)
        self.three_plane_z_label = QLabel("50%")
        self.three_plane_z_label.setMinimumWidth(35)
        z_layout.addWidget(self.three_plane_z_label)
        three_plane_layout.addLayout(z_layout)
        
        slice_layout.addWidget(self.three_plane_controls)
        
        # Initially hide three-plane controls
        self.three_plane_controls.setVisible(False)
        
        self.dynamic_layout.addLayout(slice_layout)
    
    def _add_flux_range_controls(self):
        """Add flux range controls for all modes"""
        # Add separator
        separator = QWidget()
        separator.setFixedHeight(10)
        self.dynamic_layout.addWidget(separator)
        
        # Flux range layout
        flux_layout = QVBoxLayout()
        
        # Min/Max flux controls side by side
        range_layout = QHBoxLayout()
        
        # Min flux
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Min Flux:"))
        self.min_flux_input = QLineEdit("1e-3")  # Default minimum: 0.001 particles/cm²/s
        self.min_flux_input.setPlaceholderText("e.g., 1e2")
        self.min_flux_input.setMinimumWidth(90)
        self.min_flux_input.editingFinished.connect(self._on_flux_range_changed)
        min_layout.addWidget(self.min_flux_input)
        range_layout.addLayout(min_layout)
        
        # Max flux  
        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Max Flux:"))
        self.max_flux_input = QLineEdit("Auto")  # Auto-detect from data
        self.max_flux_input.setPlaceholderText("e.g., 1e8 or Auto")
        self.max_flux_input.setMinimumWidth(90)
        self.max_flux_input.editingFinished.connect(self._on_flux_range_changed)
        max_layout.addWidget(self.max_flux_input)
        range_layout.addLayout(max_layout)
        
        flux_range_row = self._create_aligned_row("Flux Range", range_layout)
        flux_layout.addLayout(flux_range_row)
        
        # Info label
        info_text = "Zero flux values will be set to Min Flux for log scale"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        info_label.setWordWrap(True)
        info_row = self._create_aligned_row("", info_label)
        flux_layout.addLayout(info_row)
        
        self.dynamic_layout.addLayout(flux_layout)
    
    def _create_aligned_row(self, label_text, widget_or_layout):
        """Create a row with aligned labels and colons
        
        Args:
            label_text: Text for the label (without colon)
            widget_or_layout: Widget or layout to place next to label
            
        Returns:
            QHBoxLayout with aligned label and widget
        """
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 1, 0, 1)  # Tighter vertical margins on rows
        
        # Create label with fixed width for alignment - reduce for tighter spacing
        label = QLabel(label_text)
        label.setMinimumWidth(70)  # Reduced width for more compact layout
        label.setMaximumWidth(70)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(label)
        
        # Add colon as separate label for perfect alignment
        colon_label = QLabel(":")
        colon_label.setMinimumWidth(6)
        colon_label.setMaximumWidth(6)
        colon_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(colon_label)
        
        # Add some spacing - reduced for tighter layout
        row_layout.addSpacing(4)
        
        # Add the widget or layout
        if hasattr(widget_or_layout, 'addWidget'):  # It's a layout
            row_layout.addLayout(widget_or_layout)
        else:  # It's a widget
            row_layout.addWidget(widget_or_layout)
        
        return row_layout
    
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
        self.size_label.setText(f"{value}km")
        self.settings_changed.emit()
    
    def _on_threshold_changed(self, value):
        """Handle threshold change"""
        self.threshold_label.setText(f"{value}%")
        self.settings_changed.emit()
    
    def _on_slice_pos_changed(self, value):
        """Handle slice position change"""
        print(f"DEBUG: _on_slice_pos_changed called with value={value}")
        print(f"DEBUG: hasattr data_bounds = {hasattr(self, 'data_bounds')}")
        
        # Update label with km values if data bounds are available
        if hasattr(self, 'data_bounds'):
            print(f"DEBUG: data_bounds = {self.data_bounds}")
            self._update_single_slice_label(value)
        else:
            print("DEBUG: Using fallback percentage display")
            # Fallback to percentage display when no data bounds available
            if value == 50:
                self.slice_pos_label.setText("Center (0 km)")
            else:
                offset_percent = value - 50
                self.slice_pos_label.setText(f"{value}% ({offset_percent:+.0f}% from center)")
            # Keep the original generic tooltip when no bounds are available
            print(f"DEBUG: Setting fallback tooltip for value {value}")
            self.slice_pos_slider.setToolTip("Adjust the position of the slice plane")
        self.settings_changed.emit()
    
    def _on_slice_axis_changed(self, axis_text):
        """Handle slice axis change"""
        # Reset position to center when axis changes
        self.slice_pos_slider.setValue(50)
        # Update the label with new axis bounds if available
        if hasattr(self, 'data_bounds'):
            self._update_single_slice_label(50)
        self.settings_changed.emit()
    
    def _on_slice_style_changed(self, style):
        """Handle slice style change"""
        # Show/hide controls based on style
        is_single = (style == "Single Slice")
        self.single_slice_controls.setVisible(is_single)
        self.three_plane_controls.setVisible(not is_single)
        self.settings_changed.emit()
    
    def _on_three_plane_changed(self):
        """Handle three-plane position changes"""
        # Update labels with km values if data bounds are available
        if hasattr(self, 'data_bounds'):
            self._update_three_plane_labels()
        else:
            # Fallback to percentage display
            if hasattr(self, 'three_plane_x_slider'):
                x_val = self.three_plane_x_slider.value()
                self.three_plane_x_label.setText(f"{x_val}%")
            
            if hasattr(self, 'three_plane_y_slider'):
                y_val = self.three_plane_y_slider.value()
                self.three_plane_y_label.setText(f"{y_val}%")
                
            if hasattr(self, 'three_plane_z_slider'):
                z_val = self.three_plane_z_slider.value()
                self.three_plane_z_label.setText(f"{z_val}%")
        
        self.settings_changed.emit()
    
    def _on_flux_range_changed(self):
        """Handle flux range change"""
        self.settings_changed.emit()
    
    def _on_isosurface_style_changed(self, style):
        """Handle isosurface style change"""
        # Show/hide controls based on style
        is_single = (style == "Single Isosurface")
        self.single_iso_controls.setVisible(is_single)
        self.multi_iso_controls.setVisible(not is_single)
        self.settings_changed.emit()
    
    def _on_isosurface_level_changed(self, level_percent):
        """Handle single isosurface level change"""
        # Update label
        self.isosurface_level_label.setText(f"{level_percent}th flux percentile")
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
    
    def set_colormap(self, colormap_name):
        """Set the current colormap"""
        if hasattr(self, 'colormap_combo'):
            index = self.colormap_combo.findText(colormap_name)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)
    
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
    
    def add_flux_file(self, file_path):
        """Add a flux file to the selector
        
        Args:
            file_path: Path to the flux file
        """
        from pathlib import Path
        
        # Store the file
        self.loaded_files[file_path] = Path(file_path).name
        
        # Update UI
        self._update_file_selector()
    
    def remove_flux_file(self, file_path):
        """Remove a flux file from the selector
        
        Args:
            file_path: Path to the flux file
        """
        if file_path in self.loaded_files:
            del self.loaded_files[file_path]
            self._update_file_selector()
    
    def _update_file_selector(self):
        """Update the file selector display"""
        num_files = len(self.loaded_files)
        
        if num_files == 0:
            self.files_status_label.setText("No flux files loaded")
        elif num_files == 1:
            file_path = list(self.loaded_files.keys())[0]
            display_name = self.loaded_files[file_path]
            self.files_status_label.setText(f"1 file: {display_name}")
        else:
            self.files_status_label.setText(f"{num_files} files loaded")
    
    def get_loaded_files(self):
        """Get all loaded file paths
        
        Returns:
            list: List of all loaded file paths
        """
        return list(self.loaded_files.keys())
    
    def get_min_flux(self):
        """Get the minimum flux threshold
        
        Returns:
            float: Minimum flux value, or None if invalid
        """
        if hasattr(self, 'min_flux_input'):
            try:
                text = self.min_flux_input.text().strip()
                if text:
                    return float(text)
            except ValueError:
                pass
        return 1e-3  # Default minimum: 0.001 particles/cm²/s
    
    def get_max_flux(self):
        """Get the maximum flux threshold
        
        Returns:
            float: Maximum flux value, or None for auto
        """
        if hasattr(self, 'max_flux_input'):
            text = self.max_flux_input.text().strip().lower()
            if text and text != "auto":
                try:
                    return float(text)
                except ValueError:
                    pass
        return None  # Auto-detect from data
    
    def get_isosurface_style(self):
        """Get the current isosurface style
        
        Returns:
            str: "Single Isosurface" or "Multiple Isosurface"
        """
        if hasattr(self, 'isosurface_style_combo'):
            return self.isosurface_style_combo.currentText()
        return "Single Isosurface"  # Default
    
    def get_isosurface_level(self):
        """Get the single isosurface level percentage
        
        Returns:
            int: Level percentage (10-90)
        """
        if hasattr(self, 'isosurface_level_slider'):
            return self.isosurface_level_slider.value()
        return 50  # Default
    
    def get_multiple_isosurface_levels(self):
        """Get the multiple isosurface levels
        
        Returns:
            list: List of enabled level percentages
        """
        levels = []
        if hasattr(self, 'multi_level_controls'):
            for checkbox, slider, label in self.multi_level_controls:
                if checkbox.isChecked():
                    levels.append(slider.value())
        return levels if levels else [20, 40, 60, 80]  # Default levels
    
    def get_slice_axis(self):
        """Get the current slice axis/orientation
        
        Returns:
            str: Axis text like "X-Axis (YZ Plane)"
        """
        if hasattr(self, 'slice_axis_combo'):
            return self.slice_axis_combo.currentText()
        return "Z-Axis (XY Plane)"  # Default
    
    def get_slice_position(self):
        """Get the slice position percentage
        
        Returns:
            int: Position percentage (0-100)
        """
        if hasattr(self, 'slice_pos_slider'):
            return self.slice_pos_slider.value()
        return 50  # Default center
    
    def get_slice_style(self):
        """Get the current slice style
        
        Returns:
            str: "Single Slice" or "Three Plane Slice"
        """
        if hasattr(self, 'slice_style_combo'):
            return self.slice_style_combo.currentText()
        return "Single Slice"  # Default
    
    def get_three_plane_positions(self):
        """Get the three-plane intersection positions
        
        Returns:
            tuple: (x_percent, y_percent, z_percent)
        """
        if hasattr(self, 'three_plane_x_slider'):
            x_pos = self.three_plane_x_slider.value()
            y_pos = self.three_plane_y_slider.value()
            z_pos = self.three_plane_z_slider.value()
            return (x_pos, y_pos, z_pos)
        return (50, 50, 50)  # Default center
    
    def update_slice_bounds(self, data_bounds):
        """Update slice controls with actual data bounds for km display
        
        Args:
            data_bounds: tuple of (xmin, xmax, ymin, ymax, zmin, zmax) in km
        """
        self.data_bounds = data_bounds
        
        # Update single slice position label if it exists
        if hasattr(self, 'slice_pos_slider'):
            self._update_single_slice_label(self.slice_pos_slider.value())
        
        # Update three-plane labels if they exist
        if hasattr(self, 'three_plane_x_slider'):
            self._update_three_plane_labels()
    
    def _update_single_slice_label(self, value):
        """Update single slice position label with km values"""
        print(f"DEBUG: _update_single_slice_label called with value={value}")
        
        if not hasattr(self, 'data_bounds') or not hasattr(self, 'slice_axis_combo'):
            print("DEBUG: Missing data_bounds or slice_axis_combo")
            return
            
        axis_text = self.slice_axis_combo.currentText()
        print(f"DEBUG: axis_text = {axis_text}")
        
        # Determine which axis we're slicing along
        if "X-Axis" in axis_text:
            # YZ Plane - slicing along X axis
            min_coord, max_coord = self.data_bounds[0], self.data_bounds[1]
            axis_name = "X"
        elif "Y-Axis" in axis_text:
            # XZ Plane - slicing along Y axis  
            min_coord, max_coord = self.data_bounds[2], self.data_bounds[3]
            axis_name = "Y"
        else:
            # XY Plane - slicing along Z axis
            min_coord, max_coord = self.data_bounds[4], self.data_bounds[5]
            axis_name = "Z"
        
        # Convert percentage to actual coordinate
        coord_range = max_coord - min_coord
        actual_coord = min_coord + (value / 100.0) * coord_range
        
        print(f"DEBUG: axis={axis_name}, min={min_coord}, max={max_coord}, actual_coord={actual_coord}")
        
        # Update label
        if abs(actual_coord) < 0.1:
            self.slice_pos_label.setText("Center (0 km)")
        else:
            self.slice_pos_label.setText(f"{value}% ({actual_coord:+.1f} km)")
        
        # Update tooltip with ONLY the position in km on the orthogonal axis
        tooltip_text = f"{actual_coord:.1f} km"
        print(f"DEBUG: Setting tooltip to: {tooltip_text}")
        self.slice_pos_slider.setToolTip(tooltip_text)
    
    def _update_three_plane_labels(self):
        """Update three-plane position labels with km values"""
        if not hasattr(self, 'data_bounds'):
            return
            
        # X position (in km)
        x_val = self.three_plane_x_slider.value()
        x_range = self.data_bounds[1] - self.data_bounds[0]
        x_coord = self.data_bounds[0] + (x_val / 100.0) * x_range
        if abs(x_coord) < 0.1:
            self.three_plane_x_label.setText("0 km")
        else:
            self.three_plane_x_label.setText(f"{x_coord:+.0f} km")
        
        # Y position (in km)
        y_val = self.three_plane_y_slider.value()
        y_range = self.data_bounds[3] - self.data_bounds[2]
        y_coord = self.data_bounds[2] + (y_val / 100.0) * y_range
        if abs(y_coord) < 0.1:
            self.three_plane_y_label.setText("0 km")
        else:
            self.three_plane_y_label.setText(f"{y_coord:+.0f} km")
        
        # Z position (in km)
        z_val = self.three_plane_z_slider.value()
        z_range = self.data_bounds[5] - self.data_bounds[4]
        z_coord = self.data_bounds[4] + (z_val / 100.0) * z_range
        if abs(z_coord) < 0.1:
            self.three_plane_z_label.setText("0 km")
        else:
            self.three_plane_z_label.setText(f"{z_coord:+.0f} km")
