"""
Visualization control panel for STRATOS
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSlider, QLabel, QWidget, QCheckBox, QSpinBox, QLineEdit
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
    selected_file_changed = pyqtSignal(str)
    
    def __init__(self, config, parent=None):
        """Initialize the visualization panel"""
        super().__init__("Field Visualization", parent)
        self.config = config
        self.loaded_files = {}  # Track loaded files
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        self.main_layout = QVBoxLayout(self)
        
        # File selector (initially hidden)
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.setVisible(False)  # Hidden when no files or single file
        self.file_combo.currentTextChanged.connect(self._on_file_changed)
        file_layout.addWidget(self.file_combo)
        
        # Current file label (shown when single file)
        self.current_file_label = QLabel("No flux files loaded")
        self.current_file_label.setStyleSheet("font-weight: bold; color: #666;")
        file_layout.addWidget(self.current_file_label)
        
        self.main_layout.addLayout(file_layout)
        
        # Visualization mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems([
            "Point Cloud",
            "Isosurfaces"
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
    
    def _on_file_changed(self, file_path):
        """Handle file selection change"""
        if file_path:
            self.selected_file_changed.emit(file_path)
    
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
        self.opacity_slider.setMinimumWidth(200)  # Standardized width
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
        self.point_density_slider.setMinimumWidth(200)  # Standardized width
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
        self.point_size_slider.setMinimumWidth(200)  # Standardized width
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
        self.threshold_slider.setMinimumWidth(200)  # Standardized width
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
        single_layout.setContentsMargins(0, 5, 0, 5)
        
        # Title for the control
        iso_title = QLabel("Contour Level:")
        iso_title.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        single_layout.addWidget(iso_title)
        
        # Slider gets full width (like volume threshold)
        self.isosurface_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.isosurface_level_slider.setRange(10, 90)
        self.isosurface_level_slider.setValue(50)
        self.isosurface_level_slider.valueChanged.connect(self._on_isosurface_level_changed)
        self.isosurface_level_slider.setMinimumWidth(200)
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
        multi_layout.setContentsMargins(0, 5, 0, 5)
        
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
        self.slice_pos_slider.setMinimumWidth(200)  # Standardized width
        self.slice_pos_slider.valueChanged.connect(self._on_slice_pos_changed)
        pos_layout.addWidget(self.slice_pos_slider)
        self.slice_pos_label = QLabel("50%")
        self.slice_pos_label.setMinimumWidth(40)
        pos_layout.addWidget(self.slice_pos_label)
        slice_layout.addRow("Position:", pos_layout)
        
        # Multiple slices
        self.multi_slice_check = QCheckBox("Multiple Slices")
        self.multi_slice_check.stateChanged.connect(lambda: self.settings_changed.emit())
        slice_layout.addRow("", self.multi_slice_check)
        
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
        row_layout.setContentsMargins(0, 2, 0, 2)
        
        # Create label with fixed width for alignment
        label = QLabel(label_text)
        label.setMinimumWidth(80)  # Fixed width for alignment
        label.setMaximumWidth(80)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(label)
        
        # Add colon as separate label for perfect alignment
        colon_label = QLabel(":")
        colon_label.setMinimumWidth(8)
        colon_label.setMaximumWidth(8)
        colon_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(colon_label)
        
        # Add some spacing
        row_layout.addSpacing(6)
        
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
        self.slice_pos_label.setText(f"{value}%")
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
            # No files loaded
            self.file_combo.setVisible(False)
            self.current_file_label.setVisible(True)
            self.current_file_label.setText("No flux files loaded")
            
        elif num_files == 1:
            # Single file - show as label
            self.file_combo.setVisible(False) 
            self.current_file_label.setVisible(True)
            file_path = list(self.loaded_files.keys())[0]
            display_name = self.loaded_files[file_path]
            self.current_file_label.setText(f"File: {display_name}")
            
        else:
            # Multiple files - show dropdown
            self.current_file_label.setVisible(False)
            self.file_combo.setVisible(True)
            
            # Update combo box items
            self.file_combo.clear()
            for file_path, display_name in self.loaded_files.items():
                self.file_combo.addItem(display_name, file_path)
    
    def get_selected_file(self):
        """Get the currently selected file path
        
        Returns:
            str: Path to selected file, or None if no files
        """
        if len(self.loaded_files) == 0:
            return None
        elif len(self.loaded_files) == 1:
            return list(self.loaded_files.keys())[0]
        else:
            current_index = self.file_combo.currentIndex()
            if current_index >= 0:
                return self.file_combo.itemData(current_index)
            return None
    
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
