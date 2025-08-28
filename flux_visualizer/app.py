# app.py
"""
STRATOS - Space Trajectory Radiation Analysis Toolkit for Orbital Simulation
Main application with dynamic UI panels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import vtk
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QFormLayout, QSplitter, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QFrame
)
from PyQt6.QtCore import QTimer, Qt

# Import your refactored components
from config import Config
from core import OrbitalPoint
from data_io import VTKDataLoader, OrbitalDataLoader  
from visualization import ColorManager
from scene import EarthRenderer
from analysis import FluxAnalyzer

# VTK-Qt integration
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ElectronFluxVisualizerApp(QMainWindow):
    """
    Main STRATOS application with dynamic UI panels
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.config = Config()
        self.color_manager = ColorManager()
        self.flux_analyzer = FluxAnalyzer()
        
        # Data
        self.vtk_data = None
        self.orbital_path = []
        self.current_time_index = 0
        self.is_playing = False
        
        # Current states
        self.current_data_type = "Electron Flux"
        self.current_viz_mode = "Point Cloud"
        self.current_analysis_mode = "Flux Analysis"
        
        # Dynamic UI storage
        self.data_type_widgets = {}
        self.viz_mode_widgets = {}
        self.analysis_widgets = {}
        
        # Setup UI
        self.setup_ui()
        
        # Setup VTK
        self.setup_vtk()
        
        # Setup Earth using new EarthRenderer
        self.earth_renderer = EarthRenderer(self.renderer)
        self.earth_renderer.create_earth()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)
        
        # Connect signals
        self.connect_signals()
        
        # Initial camera setup
        self.reset_camera()
        
        # Initialize dynamic panels
        self.update_data_panel()
        self.update_viz_panel()
        self.update_analysis_panel()
        
    def setup_ui(self):
        """Setup the user interface with dynamic panels"""
        self.setWindowTitle(f"{self.config.APP_NAME} - v1.0")
        self.setGeometry(
            self.config.MAIN_WINDOW_START_X,
            self.config.MAIN_WINDOW_START_Y,
            self.config.MAIN_WINDOW_WIDTH,
            self.config.MAIN_WINDOW_HEIGHT
        )
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left: VTK Widget
        vtk_frame = QWidget()
        vtk_layout = QVBoxLayout(vtk_frame)
        
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout.addWidget(self.vtk_widget)
        
        # Earth controls below VTK widget
        earth_controls = self.create_earth_controls()
        vtk_layout.addWidget(earth_controls)
        
        splitter.addWidget(vtk_frame)
        
        # Right: Control Panel with dynamic sections
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(self.config.CONTROL_PANEL_MAX_WIDTH)
        control_panel.setMinimumWidth(self.config.CONTROL_PANEL_MIN_WIDTH)
        splitter.addWidget(control_panel)
        
        # Set splitter sizes
        splitter.setSizes([
            self.config.VTK_WIDGET_DEFAULT_WIDTH,
            self.config.CONTROL_PANEL_DEFAULT_WIDTH
        ])
    
    def create_earth_controls(self):
        """Create Earth-specific controls"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Earth opacity slider
        layout.addWidget(QLabel("Earth Opacity:"))
        
        self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.earth_opacity_slider.setRange(0, 100)
        self.earth_opacity_slider.setValue(int(self.config.EARTH_DEFAULT_OPACITY * 100))
        self.earth_opacity_slider.setMaximumWidth(150)
        layout.addWidget(self.earth_opacity_slider)
        
        self.earth_opacity_label = QLabel(f"{int(self.config.EARTH_DEFAULT_OPACITY * 100)}%")
        self.earth_opacity_label.setMinimumWidth(40)
        layout.addWidget(self.earth_opacity_label)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox("Show Lat/Long Grid")
        self.grid_checkbox.setChecked(False)
        layout.addWidget(self.grid_checkbox)
        
        # Reset camera button
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.setMaximumWidth(100)
        layout.addWidget(self.reset_camera_button)
        
        layout.addStretch()
        
        # Style
        widget.setStyleSheet("""
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
        
        return widget
    
    def create_control_panel(self):
        """Create the main control panel with dynamic sections"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # === 1. DATA LOADING PANEL (Dynamic based on data type) ===
        self.data_group = QGroupBox("Data Loading")
        self.data_layout = QVBoxLayout(self.data_group)
        
        # Data type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Data Type:"))
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "Electron Flux",
            "Proton Flux",
            "Heavy Ion Flux",
            "Combined Radiation",
            "Electric Field",
            "Magnetic Field"
        ])
        type_layout.addWidget(self.data_type_combo)
        self.data_layout.addLayout(type_layout)
        
        # Dynamic content area for data-specific controls
        self.data_dynamic_widget = QWidget()
        self.data_dynamic_layout = QVBoxLayout(self.data_dynamic_widget)
        self.data_layout.addWidget(self.data_dynamic_widget)
        
        layout.addWidget(self.data_group)
        
        # === 2. ANIMATION CONTROLS (Static) ===
        anim_group = QGroupBox("Animation Controls")
        anim_layout = QVBoxLayout(anim_group)
        
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        anim_layout.addLayout(button_layout)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setEnabled(False)
        anim_layout.addWidget(QLabel("Time:"))
        anim_layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("No data loaded")
        anim_layout.addWidget(self.time_label)
        
        # Animation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 1000)
        self.speed_slider.setValue(100)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("100 ms")
        speed_layout.addWidget(self.speed_label)
        anim_layout.addLayout(speed_layout)
        
        layout.addWidget(anim_group)
        
        # === 3. FIELD VISUALIZATION (Dynamic based on mode) ===
        self.viz_group = QGroupBox("Field Visualization")
        self.viz_layout = QVBoxLayout(self.viz_group)
        
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
        mode_layout.addWidget(self.viz_mode_combo)
        self.viz_layout.addLayout(mode_layout)
        
        # Dynamic content area for viz-specific controls
        self.viz_dynamic_widget = QWidget()
        self.viz_dynamic_layout = QVBoxLayout(self.viz_dynamic_widget)
        self.viz_layout.addWidget(self.viz_dynamic_widget)
        
        layout.addWidget(self.viz_group)
        
        # === 4. ANALYSIS (Dynamic based on analysis type) ===
        self.analysis_group = QGroupBox("Analysis")
        self.analysis_layout = QVBoxLayout(self.analysis_group)
        
        # Analysis mode selector
        analysis_mode_layout = QHBoxLayout()
        analysis_mode_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems([
            "Flux Analysis",
            "Spectrum Analysis",
            "Trajectory Statistics"
        ])
        analysis_mode_layout.addWidget(self.analysis_mode_combo)
        self.analysis_layout.addLayout(analysis_mode_layout)
        
        # Dynamic content area for analysis-specific controls
        self.analysis_dynamic_widget = QWidget()
        self.analysis_dynamic_layout = QVBoxLayout(self.analysis_dynamic_widget)
        self.analysis_layout.addWidget(self.analysis_dynamic_widget)
        
        layout.addWidget(self.analysis_group)
        
        # === 5. STATUS ===
        self.status_label = QLabel("Ready - Select data type and load files")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                padding: 10px; 
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        return panel
    
    def update_data_panel(self):
        """Update data panel based on selected data type"""
        # Clear existing widgets
        self.clear_layout(self.data_dynamic_layout)
        
        data_type = self.data_type_combo.currentText()
        
        # Common controls for all types
        self.load_vtk_button = QPushButton(f"Load {data_type} VTK")
        self.load_orbit_button = QPushButton("Load Orbital CSV")
        
        self.data_dynamic_layout.addWidget(self.load_vtk_button)
        self.data_dynamic_layout.addWidget(self.load_orbit_button)
        
        # Add type-specific controls
        if "Flux" in data_type:
            # Energy range for flux data
            energy_layout = QFormLayout()
            
            self.min_energy_spin = QDoubleSpinBox()
            self.min_energy_spin.setRange(0.01, 10000)
            self.min_energy_spin.setValue(10)
            self.min_energy_spin.setSuffix(" keV")
            energy_layout.addRow("Min Energy:", self.min_energy_spin)
            
            self.max_energy_spin = QDoubleSpinBox()
            self.max_energy_spin.setRange(0.01, 10000)
            self.max_energy_spin.setValue(10000)
            self.max_energy_spin.setSuffix(" keV")
            energy_layout.addRow("Max Energy:", self.max_energy_spin)
            
            self.data_dynamic_layout.addLayout(energy_layout)
            
        elif "Field" in data_type:
            # Field component selector
            field_layout = QHBoxLayout()
            field_layout.addWidget(QLabel("Component:"))
            self.field_component_combo = QComboBox()
            self.field_component_combo.addItems(["Magnitude", "X", "Y", "Z"])
            field_layout.addWidget(self.field_component_combo)
            self.data_dynamic_layout.addLayout(field_layout)
        
        # Re-connect signals
        self.load_vtk_button.clicked.connect(self.load_vtk_data)
        self.load_orbit_button.clicked.connect(self.load_orbital_data)
    
    def update_viz_panel(self):
        """Update visualization panel based on selected mode"""
        # Clear existing widgets
        self.clear_layout(self.viz_dynamic_layout)
        
        viz_mode = self.viz_mode_combo.currentText()
        
        # Common controls for all modes
        common_layout = QFormLayout()
        
        # Colormap selector
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.config.AVAILABLE_COLORMAPS)
        common_layout.addRow("Colormap:", self.colormap_combo)
        
        # Scale mode
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        common_layout.addRow("Scale:", self.scale_combo)
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("70%")
        opacity_layout.addWidget(self.opacity_label)
        common_layout.addRow("Opacity:", opacity_layout)
        
        self.viz_dynamic_layout.addLayout(common_layout)
        
        # Add mode-specific controls
        if viz_mode == "Point Cloud":
            pc_layout = QFormLayout()
            
            # Point density
            density_layout = QHBoxLayout()
            self.point_density_slider = QSlider(Qt.Orientation.Horizontal)
            self.point_density_slider.setRange(self.config.MIN_POINT_DENSITY, 
                                              self.config.MAX_POINT_DENSITY)
            self.point_density_slider.setValue(self.config.DEFAULT_POINT_DENSITY)
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
            size_layout.addWidget(self.point_size_slider)
            self.size_label = QLabel(f"{self.config.DEFAULT_POINT_SIZE}m")
            size_layout.addWidget(self.size_label)
            pc_layout.addRow("Size:", size_layout)
            
            self.viz_dynamic_layout.addLayout(pc_layout)
            
        elif viz_mode == "Volume Rendering":
            vol_layout = QFormLayout()
            
            # Threshold
            threshold_layout = QHBoxLayout()
            self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
            self.threshold_slider.setRange(0, 100)
            self.threshold_slider.setValue(50)
            threshold_layout.addWidget(self.threshold_slider)
            self.threshold_label = QLabel("50%")
            threshold_layout.addWidget(self.threshold_label)
            vol_layout.addRow("Threshold:", threshold_layout)
            
            # Quality
            self.quality_combo = QComboBox()
            self.quality_combo.addItems(["Fast", "Medium", "High Quality"])
            vol_layout.addRow("Quality:", self.quality_combo)
            
            self.viz_dynamic_layout.addLayout(vol_layout)
            
        elif viz_mode == "Isosurfaces":
            iso_layout = QFormLayout()
            
            # Number of surfaces
            self.num_surfaces_spin = QSpinBox()
            self.num_surfaces_spin.setRange(1, 10)
            self.num_surfaces_spin.setValue(3)
            iso_layout.addRow("Surfaces:", self.num_surfaces_spin)
            
            # Surface levels
            level_layout = QHBoxLayout()
            self.min_level_slider = QSlider(Qt.Orientation.Horizontal)
            self.min_level_slider.setRange(0, 100)
            self.min_level_slider.setValue(20)
            level_layout.addWidget(QLabel("Min:"))
            level_layout.addWidget(self.min_level_slider)
            
            self.max_level_slider = QSlider(Qt.Orientation.Horizontal)
            self.max_level_slider.setRange(0, 100)
            self.max_level_slider.setValue(80)
            level_layout.addWidget(QLabel("Max:"))
            level_layout.addWidget(self.max_level_slider)
            
            iso_layout.addRow("Levels:", level_layout)
            
            self.viz_dynamic_layout.addLayout(iso_layout)
            
        elif viz_mode == "Slice Planes":
            slice_layout = QFormLayout()
            
            # Slice axis
            self.slice_axis_combo = QComboBox()
            self.slice_axis_combo.addItems(["X-Axis", "Y-Axis", "Z-Axis"])
            slice_layout.addRow("Axis:", self.slice_axis_combo)
            
            # Slice position
            pos_layout = QHBoxLayout()
            self.slice_pos_slider = QSlider(Qt.Orientation.Horizontal)
            self.slice_pos_slider.setRange(0, 100)
            self.slice_pos_slider.setValue(50)
            pos_layout.addWidget(self.slice_pos_slider)
            self.slice_pos_label = QLabel("50%")
            pos_layout.addWidget(self.slice_pos_label)
            slice_layout.addRow("Position:", pos_layout)
            
            # Multiple slices
            self.multi_slice_check = QCheckBox("Multiple Slices")
            slice_layout.addRow("", self.multi_slice_check)
            
            self.viz_dynamic_layout.addLayout(slice_layout)
        
        # Connect signals for dynamic widgets
        if hasattr(self, 'opacity_slider'):
            self.opacity_slider.valueChanged.connect(
                lambda v: self.opacity_label.setText(f"{v}%"))
        if hasattr(self, 'point_density_slider'):
            self.point_density_slider.valueChanged.connect(
                lambda v: self.density_label.setText(str(v)))
        if hasattr(self, 'point_size_slider'):
            self.point_size_slider.valueChanged.connect(
                lambda v: self.size_label.setText(f"{v}m"))
        if hasattr(self, 'threshold_slider'):
            self.threshold_slider.valueChanged.connect(
                lambda v: self.threshold_label.setText(f"{v}%"))
        if hasattr(self, 'slice_pos_slider'):
            self.slice_pos_slider.valueChanged.connect(
                lambda v: self.slice_pos_label.setText(f"{v}%"))
    
    def update_analysis_panel(self):
        """Update analysis panel based on selected analysis type"""
        # Clear existing widgets
        self.clear_layout(self.analysis_dynamic_layout)
        
        analysis_type = self.analysis_mode_combo.currentText()
        
        if analysis_type == "Flux Analysis":
            flux_layout = QFormLayout()
            
            # Cross section
            self.cross_section_spin = QDoubleSpinBox()
            self.cross_section_spin.setRange(0.1, 100.0)
            self.cross_section_spin.setValue(1.0)
            self.cross_section_spin.setSuffix(" m")
            flux_layout.addRow("Cross Section:", self.cross_section_spin)
            
            # Integration time
            self.integration_spin = QDoubleSpinBox()
            self.integration_spin.setRange(0.1, 1000.0)
            self.integration_spin.setValue(1.0)
            self.integration_spin.setSuffix(" s")
            flux_layout.addRow("Integration:", self.integration_spin)
            
            self.analysis_dynamic_layout.addLayout(flux_layout)
            
            # Analysis windows
            self.analysis_dynamic_layout.addWidget(QLabel("Analysis Windows:"))
            
            self.flux_time_button = QPushButton("Flux vs Time")
            self.peak_analysis_button = QPushButton("Peak Exposure Analysis")
            self.dose_calc_button = QPushButton("Dose Calculator")
            
            self.analysis_dynamic_layout.addWidget(self.flux_time_button)
            self.analysis_dynamic_layout.addWidget(self.peak_analysis_button)
            self.analysis_dynamic_layout.addWidget(self.dose_calc_button)
            
            # Connect to flux analyzer
            if hasattr(self, 'cross_section_spin'):
                self.cross_section_spin.valueChanged.connect(
                    lambda v: self.flux_analyzer.set_cross_section(v))
            
        elif analysis_type == "Spectrum Analysis":
            spectrum_layout = QFormLayout()
            
            # Energy bins
            self.energy_bins_spin = QSpinBox()
            self.energy_bins_spin.setRange(10, 200)
            self.energy_bins_spin.setValue(50)
            spectrum_layout.addRow("Energy Bins:", self.energy_bins_spin)
            
            # Energy range
            energy_range_layout = QHBoxLayout()
            self.min_energy_input = QLineEdit("10")
            self.max_energy_input = QLineEdit("10000")
            energy_range_layout.addWidget(QLabel("Min:"))
            energy_range_layout.addWidget(self.min_energy_input)
            energy_range_layout.addWidget(QLabel("keV Max:"))
            energy_range_layout.addWidget(self.max_energy_input)
            energy_range_layout.addWidget(QLabel("keV"))
            spectrum_layout.addRow("Energy Range:", energy_range_layout)
            
            self.analysis_dynamic_layout.addLayout(spectrum_layout)
            
            # Analysis windows
            self.analysis_dynamic_layout.addWidget(QLabel("Spectrum Windows:"))
            
            self.energy_spectrum_button = QPushButton("Energy Spectrum")
            self.pitch_angle_button = QPushButton("Pitch Angle Distribution")
            self.phase_space_button = QPushButton("Phase Space Plot")
            
            self.analysis_dynamic_layout.addWidget(self.energy_spectrum_button)
            self.analysis_dynamic_layout.addWidget(self.pitch_angle_button)
            self.analysis_dynamic_layout.addWidget(self.phase_space_button)
            
        elif analysis_type == "Trajectory Statistics":
            traj_layout = QFormLayout()
            
            # Statistics options
            self.include_velocity_check = QCheckBox("Include Velocity Analysis")
            self.include_altitude_check = QCheckBox("Include Altitude Profile")
            traj_layout.addRow("Options:", self.include_velocity_check)
            traj_layout.addRow("", self.include_altitude_check)
            
            self.analysis_dynamic_layout.addLayout(traj_layout)
            
            # Analysis windows
            self.analysis_dynamic_layout.addWidget(QLabel("Trajectory Windows:"))
            
            self.orbit_stats_button = QPushButton("Orbital Statistics")
            self.ground_track_button = QPushButton("Ground Track")
            self.altitude_profile_button = QPushButton("Altitude Profile")
            
            self.analysis_dynamic_layout.addWidget(self.orbit_stats_button)
            self.analysis_dynamic_layout.addWidget(self.ground_track_button)
            self.analysis_dynamic_layout.addWidget(self.altitude_profile_button)
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
    
    def setup_vtk(self):
        """Setup VTK rendering pipeline"""
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(*self.config.RENDERER_BACKGROUND_COLOR)
        
        # Setup camera
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(*self.config.CAMERA_DEFAULT_POSITION)
        self.camera.SetFocalPoint(*self.config.CAMERA_DEFAULT_FOCAL_POINT)
        self.camera.SetViewUp(*self.config.CAMERA_DEFAULT_VIEW_UP)
        
        # Initialize interactor
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
    
    def connect_signals(self):
        """Connect UI signals to methods"""
        # Data panel
        self.data_type_combo.currentTextChanged.connect(self.update_data_panel)
        
        # Visualization panel
        self.viz_mode_combo.currentTextChanged.connect(self.update_viz_panel)
        
        # Analysis panel
        self.analysis_mode_combo.currentTextChanged.connect(self.update_analysis_panel)
        
        # Animation
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.speed_slider.valueChanged.connect(self.update_animation_speed)
        
        # Earth controls
        self.earth_opacity_slider.valueChanged.connect(self.update_earth_opacity)
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        self.reset_camera_button.clicked.connect(self.reset_camera)
    
    def load_vtk_data(self):
        """Load VTK data using VTKDataLoader"""
        file_filter = VTKDataLoader.get_file_filter()
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Load {self.current_data_type} Data", "", file_filter
        )
        
        if not file_path:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(25)
            
            # Use VTKDataLoader
            self.vtk_data = VTKDataLoader.load(file_path)
            
            self.progress_bar.setValue(50)
            
            # Get info
            info = VTKDataLoader.get_data_info(self.vtk_data)
            
            # Update flux analyzer
            self.flux_analyzer.set_vtk_data(self.vtk_data)
            
            # Create simple visualization
            self.create_field_visualization()
            
            self.progress_bar.setValue(100)
            
            # Update status
            self.status_label.setText(
                f"✓ Loaded {self.current_data_type}: {info['type']}\n"
                f"Points: {info['num_points']:,} | "
                f"Scalar: {info['scalar_name']}"
            )
            
            # Reset camera to show data
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load VTK:\n{str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def load_orbital_data(self):
        """Load orbital data using OrbitalDataLoader"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Orbital Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Use OrbitalDataLoader
            self.orbital_path = OrbitalDataLoader.load_csv(file_path)
            
            # Get info
            info = OrbitalDataLoader.get_trajectory_info(self.orbital_path)
            
            # Update flux analyzer
            self.flux_analyzer.set_orbital_data(self.orbital_path)
            
            # Create orbital path visualization
            self.create_orbital_visualization()
            
            # Setup time slider
            self.time_slider.setMaximum(len(self.orbital_path) - 1)
            self.time_slider.setEnabled(True)
            
            # Update status
            self.status_label.setText(
                f"✓ Loaded Orbit: {info['num_points']} points\n"
                f"Duration: {info['time_span']:.2f} hours\n"
                f"Altitude: {info['altitude_min']:.0f}-{info['altitude_max']:.0f} km"
            )
            
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load orbit:\n{str(e)}")
    
    def create_field_visualization(self):
        """Create simple point cloud visualization of VTK data"""
        if not self.vtk_data:
            return
        
        # Remove existing field actor if any
        if hasattr(self, 'field_actor'):
            self.renderer.RemoveActor(self.field_actor)
        
        # Simple threshold filter
        scalar_range = self.vtk_data.GetScalarRange()
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(self.vtk_data)
        threshold.SetLowerThreshold(scalar_range[1] * 0.01)  # Show top 99%
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
        threshold.Update()
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(threshold.GetOutputPort())
        mapper.SetScalarRange(scalar_range)
        
        # Use ColorManager for lookup table
        colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else "Viridis"
        scale_mode = self.scale_combo.currentText() if hasattr(self, 'scale_combo') else "Linear"
        lut = self.color_manager.create_lookup_table_with_scale(
            colormap_name, scale_mode, scalar_range
        )
        mapper.SetLookupTable(lut)
        
        # Create actor
        self.field_actor = vtk.vtkActor()
        self.field_actor.SetMapper(mapper)
        self.field_actor.GetProperty().SetPointSize(5)
        
        # Set opacity if available
        if hasattr(self, 'opacity_slider'):
            self.field_actor.GetProperty().SetOpacity(self.opacity_slider.value() / 100.0)
        
        self.renderer.AddActor(self.field_actor)
    
    def create_orbital_visualization(self):
        """Create orbital path and satellite"""
        if not self.orbital_path:
            return
        
        # Remove existing if any
        if hasattr(self, 'path_actor'):
            self.renderer.RemoveActor(self.path_actor)
        if hasattr(self, 'satellite_actor'):
            self.renderer.RemoveActor(self.satellite_actor)
        
        # Create path polyline
        points = vtk.vtkPoints()
        for point in self.orbital_path:
            points.InsertNextPoint(point.x, point.y, point.z)
        
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(self.orbital_path))
        for i in range(len(self.orbital_path)):
            polyline.GetPointIds().SetId(i, i)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        self.path_actor = vtk.vtkActor()
        self.path_actor.SetMapper(mapper)
        self.path_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
        self.path_actor.GetProperty().SetLineWidth(2.0)
        
        self.renderer.AddActor(self.path_actor)
        
        # Create satellite sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(500.0)  # 500 km for visibility
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        
        self.satellite_actor = vtk.vtkActor()
        self.satellite_actor.SetMapper(sphere_mapper)
        self.satellite_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        
        # Set initial position
        if self.orbital_path:
            first_point = self.orbital_path[0]
            self.satellite_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        self.renderer.AddActor(self.satellite_actor)
    
    def update_earth_opacity(self, value):
        """Update Earth opacity using EarthRenderer"""
        opacity = value / 100.0
        self.earth_opacity_label.setText(f"{value}%")
        self.earth_renderer.set_opacity(opacity)
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_grid(self, state):
        """Toggle lat/long grid using EarthRenderer"""
        self.earth_renderer.toggle_grid(state == 2)  # Qt.Checked = 2
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera(self):
        """Reset camera to default position"""
        self.camera.SetPosition(*self.config.CAMERA_DEFAULT_POSITION)
        self.camera.SetFocalPoint(*self.config.CAMERA_DEFAULT_FOCAL_POINT)
        self.camera.SetViewUp(*self.config.CAMERA_DEFAULT_VIEW_UP)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_animation_speed(self, value):
        """Update animation speed"""
        self.speed_label.setText(f"{value} ms")
        if self.is_playing:
            self.animation_timer.setInterval(value)
    
    def start_animation(self):
        """Start animation"""
        if not self.orbital_path:
            QMessageBox.warning(self, "Warning", "Please load orbital data first")
            return
        
        self.is_playing = True
        speed = self.speed_slider.value()
        self.animation_timer.start(speed)
        
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
    
    def pause_animation(self):
        """Pause animation"""
        self.is_playing = False
        self.animation_timer.stop()
        
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
    
    def stop_animation(self):
        """Stop animation"""
        self.is_playing = False
        self.animation_timer.stop()
        self.current_time_index = 0
        self.update_satellite_position()
        
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
    
    def animation_step(self):
        """Animation step"""
        if self.current_time_index >= len(self.orbital_path) - 1:
            self.current_time_index = 0
        else:
            self.current_time_index += 1
        
        self.time_slider.setValue(self.current_time_index)
        self.update_satellite_position()
    
    def on_time_slider_changed(self, value):
        """Handle time slider change"""
        self.current_time_index = value
        self.update_satellite_position()
    
    def update_satellite_position(self):
        """Update satellite position"""
        if not self.orbital_path or not hasattr(self, 'satellite_actor'):
            return
        
        if 0 <= self.current_time_index < len(self.orbital_path):
            point = self.orbital_path[self.current_time_index]
            self.satellite_actor.SetPosition(point.x, point.y, point.z)
            
            # Update time label
            hours = point.time
            h = int(hours)
            m = int((hours - h) * 60)
            s = int(((hours - h) * 60 - m) * 60)
            self.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
            
            # If flux analysis is active, show current flux
            if self.current_analysis_mode == "Flux Analysis" and self.vtk_data:
                flux = self.flux_analyzer.analyze_flux_at_point(point)
                self.status_label.setText(
                    f"Time: {point.time:.2f}h | Alt: {point.altitude:.1f}km\n"
                    f"Flux: {flux:.2e} particles/s"
                )
            
            self.vtk_widget.GetRenderWindow().Render()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties from Config
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName(Config.ORGANIZATION)
    
    # Create and show main window
    window = ElectronFluxVisualizerApp()
    window.show()
    
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
