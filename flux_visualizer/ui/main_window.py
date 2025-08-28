"""
Main window for STRATOS application
"""

import numpy as np
import vtk
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QProgressBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt

# Import configuration and core modules
from config import Config
from core import OrbitalPoint
from data_io import VTKDataLoader, OrbitalDataLoader
from visualization import ColorManager
from scene import EarthRenderer, OrbitalRenderer, SatelliteRenderer
from analysis import FluxAnalyzer

# Import UI components
from .widgets import LoadedFileWidget
from .controls import (
    DataLoadingPanel, AnimationControlPanel,
    VisualizationPanel, AnalysisPanel, EarthControlsWidget
)

# VTK-Qt integration
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ElectronFluxVisualizerApp(QMainWindow):
    """Main STRATOS application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.config = Config()
        self.color_manager = ColorManager()
        self.flux_analyzer = FluxAnalyzer()
        
        # Data storage
        self.loaded_files = {}  # Dict of filename -> LoadedFileWidget
        self.vtk_data_dict = {}  # Dict of filename -> vtk_data
        self.loaded_orbital_files = {}  # Dict of filename -> LoadedFileWidget
        self.orbital_data_dict = {}  # Dict of filename -> orbital_data
        self.current_time_index = 0
        self.is_playing = False
        
        # Setup UI
        self._setup_ui()
        
        # Setup VTK
        self._setup_vtk()

        # Setup axes widget
        self._setup_axes_widget()
        
        # Setup Earth
        self.earth_renderer = EarthRenderer(self.renderer)
        self.earth_renderer.create_earth()
        
        # ADDED: Setup orbital and satellite renderers
        self.orbital_renderer = OrbitalRenderer(self.renderer)
        self.satellite_renderer = SatelliteRenderer(self.renderer)
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._animation_step)
        
        # Connect signals
        self._connect_signals()
        
        # Initial camera setup
        self._reset_camera()
    
    def _setup_ui(self):
        """Setup the user interface"""
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
        
        # Create a container for VTK widget with overlay buttons
        vtk_container = QWidget()
        vtk_container_layout = QVBoxLayout(vtk_container)
        vtk_container_layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(vtk_container)
        vtk_container_layout.addWidget(self.vtk_widget)

        # Create view buttons overlay - NEW
        self._create_view_buttons_overlay()

        vtk_layout.addWidget(vtk_container)

        # Earth controls below VTK widget (will be updated for flat style)
        self.earth_controls = EarthControlsWidget(self.config)
        vtk_layout.addWidget(self.earth_controls)
        
        # Earth controls below VTK widget
        self.earth_controls = EarthControlsWidget(self.config)
        vtk_layout.addWidget(self.earth_controls)
        
        splitter.addWidget(vtk_frame)
        
        # Right: Control Panel
        control_panel = self._create_control_panel()
        control_panel.setMaximumWidth(self.config.CONTROL_PANEL_MAX_WIDTH)
        control_panel.setMinimumWidth(self.config.CONTROL_PANEL_MIN_WIDTH)
        splitter.addWidget(control_panel)
        
        # Set splitter sizes
        splitter.setSizes([
            self.config.VTK_WIDGET_DEFAULT_WIDTH,
            self.config.CONTROL_PANEL_DEFAULT_WIDTH
        ])

    def _create_view_buttons_overlay(self):
        """Create view buttons overlaid on VTK widget"""
        from PyQt6.QtWidgets import QPushButton

        # Create container widget for buttons
        self.view_buttons_widget = QWidget(self.vtk_widget)
        self.view_buttons_widget.setGeometry(10, 10, 150, 35)

        # Create horizontal layout
        layout = QHBoxLayout(self.view_buttons_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Create view label
        view_label = QLabel("View:")
        view_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        layout.addWidget(view_label)

        # X button (red like X axis)
        self.view_x_button = QPushButton("X")
        self.view_x_button.setMaximumSize(25, 25)
        self.view_x_button.clicked.connect(self._snap_to_x_axis)
        self.view_x_button.setToolTip("View YZ plane (from X axis)")
        self.view_x_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(200, 50, 50, 180);
                color: white;
                border: 1px solid white;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 70, 70, 200);
            }
        """)
        layout.addWidget(self.view_x_button)

        # Y button (green like Y axis)
        self.view_y_button = QPushButton("Y")
        self.view_y_button.setMaximumSize(25, 25)
        self.view_y_button.clicked.connect(self._snap_to_y_axis)
        self.view_y_button.setToolTip("View XZ plane (from Y axis)")
        self.view_y_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 200, 50, 180);
                color: white;
                border: 1px solid white;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(70, 255, 70, 200);
            }
        """)
        layout.addWidget(self.view_y_button)

        # Z button (blue like Z axis)
        self.view_z_button = QPushButton("Z")
        self.view_z_button.setMaximumSize(25, 25)
        self.view_z_button.clicked.connect(self._snap_to_z_axis)
        self.view_z_button.setToolTip("View XY plane (from Z axis)")
        self.view_z_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 50, 200, 180);
                color: white;
                border: 1px solid white;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(70, 70, 255, 200);
            }
        """)
        layout.addWidget(self.view_z_button)

        layout.addStretch()

        # Style the container
        self.view_buttons_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 150);
                border-radius: 5px;
            }
        """)

        # Make sure buttons are on top
        self.view_buttons_widget.raise_()
        
    def _create_control_panel(self):
        """Create the main control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data loading panel
        self.data_panel = DataLoadingPanel()
        layout.addWidget(self.data_panel)
        
        # Animation controls
        self.animation_panel = AnimationControlPanel()
        layout.addWidget(self.animation_panel)
        
        # Visualization panel
        self.viz_panel = VisualizationPanel(self.config)
        layout.addWidget(self.viz_panel)
        
        # Analysis panel
        self.analysis_panel = AnalysisPanel()
        layout.addWidget(self.analysis_panel)
        
        # Status label
        self.status_label = QLabel("Ready - Load flux and orbital files")
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

    def _setup_axes_widget(self):
        """Setup coordinate axes widget in corner"""
        # Create axes actor
        self.axes_actor = vtk.vtkAxesActor()

        # Set axis labels
        self.axes_actor.SetXAxisLabelText("X")
        self.axes_actor.SetYAxisLabelText("Y")
        self.axes_actor.SetZAxisLabelText("Z")

        # Set axis lengths
        self.axes_actor.SetTotalLength(1.0, 1.0, 1.0)

        # Create orientation widget
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(self.axes_actor)
        self.orientation_widget.SetInteractor(self.vtk_widget)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.15, 0.15)  # Bottom left corner
        self.orientation_widget.EnabledOn()
        self.orientation_widget.InteractiveOff()  # Don't allow user to move it
    
    def _setup_vtk(self):
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
    
    def _connect_signals(self):
        """Connect UI signals to methods"""
        # Data panel
        self.data_panel.load_flux_clicked.connect(self._load_flux_files)
        self.data_panel.load_orbital_clicked.connect(self._load_orbital_files)
        
        # Animation panel
        self.animation_panel.play_clicked.connect(self._start_animation)
        self.animation_panel.pause_clicked.connect(self._pause_animation)
        self.animation_panel.stop_clicked.connect(self._stop_animation)
        self.animation_panel.time_changed.connect(self._on_time_changed)
        self.animation_panel.speed_changed.connect(self._update_animation_speed)
        
        # Visualization panel
        self.viz_panel.mode_changed.connect(self._on_viz_mode_changed)
        self.viz_panel.colormap_changed.connect(self._update_visualization)
        self.viz_panel.scale_changed.connect(self._update_visualization)
        self.viz_panel.opacity_changed.connect(self._update_visualization)
        self.viz_panel.settings_changed.connect(self._update_visualization)
        
        # Analysis panel
        self.analysis_panel.settings_changed.connect(self._on_analysis_settings_changed)
        
        # Earth controls
        self.earth_controls.opacity_changed.connect(self._update_earth_opacity)
        self.earth_controls.grid_toggled.connect(self._toggle_grid)
        self.earth_controls.orbital_paths_toggled.connect(self._toggle_orbital_paths)
        self.earth_controls.trails_toggled.connect(self._toggle_trails)
        self.earth_controls.satellite_size_changed.connect(self._update_satellite_size)

    def _update_satellite_size(self, size_km):
        """Update size of all satellites"""
        if hasattr(self, 'satellite_renderer'):
            self.satellite_renderer.update_all_satellite_sizes(size_km)
            self.vtk_widget.GetRenderWindow().Render()

    def _calculate_optimal_camera_distance(self):
        """Calculate optimal camera distance based on loaded data"""
        max_distance = self.config.CAMERA_DEFAULT_DISTANCE if hasattr(self.config, 'CAMERA_DEFAULT_DISTANCE') else 30000

        # Check orbital data bounds
        if self.orbital_data_dict:
            for orbital_data in self.orbital_data_dict.values():
                for point in orbital_data:
                    distance = np.sqrt(point.x**2 + point.y**2 + point.z**2)
                    max_distance = max(max_distance, distance)

        # Check VTK data bounds
        if self.vtk_data_dict:
            for vtk_data in self.vtk_data_dict.values():
                bounds = vtk_data.GetBounds()
                data_max = max(
                    abs(bounds[0]), abs(bounds[1]),
                    abs(bounds[2]), abs(bounds[3]),
                    abs(bounds[4]), abs(bounds[5])
                )
                max_distance = max(max_distance, data_max)

        # Add 50% margin for comfortable viewing
        return max_distance * 1.5

    def _snap_to_x_axis(self):
        """Snap camera to view YZ plane from X axis"""
        distance = self._calculate_optimal_camera_distance()
        self.camera.SetPosition(distance, 0, 0)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 0, 1)  # Z is up
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _snap_to_y_axis(self):
        """Snap camera to view XZ plane from Y axis"""
        distance = self._calculate_optimal_camera_distance()
        self.camera.SetPosition(0, distance, 0)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 0, 1)  # Z is up
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _snap_to_z_axis(self):
        """Snap camera to view XY plane from Z axis"""
        distance = self._calculate_optimal_camera_distance()
        self.camera.SetPosition(0, 0, distance)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 1, 0)  # Y is up
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        
    def _load_flux_files(self):
        """Load one or more VTK flux files"""
        file_filter = VTKDataLoader.get_file_filter()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Flux File(s)", "", file_filter
        )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            self._add_flux_file(file_path)
    
    def _add_flux_file(self, file_path):
        """Add a single flux file"""
        try:
            # Check if already loaded
            if file_path in self.loaded_files:
                QMessageBox.information(self, "Info", 
                    f"File already loaded:\n{Path(file_path).name}")
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(25)
            
            # Load VTK data
            vtk_data = VTKDataLoader.load(file_path)
            
            self.progress_bar.setValue(50)
            
            # Store VTK data
            self.vtk_data_dict[file_path] = vtk_data
            
            # Create file widget
            file_widget = LoadedFileWidget(file_path, file_type="flux")
            file_widget.vtk_data = vtk_data
            
            # Connect signals
            file_widget.remove_button.clicked.connect(
                lambda: self._remove_flux_file(file_path))
            file_widget.checkbox.stateChanged.connect(self._update_visualization)
            if hasattr(file_widget, 'particle_combo'):
                file_widget.particle_combo.currentTextChanged.connect(
                    self._update_visualization)
            
            # Add to panel
            self.data_panel.add_flux_file_widget(file_widget)
            self.loaded_files[file_path] = file_widget
            
            self.progress_bar.setValue(75)
            
            # Update visualization
            self._update_visualization()
            
            self.progress_bar.setValue(100)
            
            # Update status
            info = VTKDataLoader.get_data_info(vtk_data)
            num_files = len(self.loaded_files)
            self.status_label.setText(
                f"✓ Loaded {num_files} file(s)\n"
                f"Latest: {Path(file_path).name}\n"
                f"Points: {info['num_points']:,}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Failed to load {Path(file_path).name}:\n{str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def _remove_flux_file(self, file_path):
        """Remove a flux file"""
        if file_path in self.loaded_files:
            # Remove widget
            widget = self.loaded_files[file_path]
            self.data_panel.remove_flux_file_widget(widget)
            widget.deleteLater()
            
            # Remove from dictionaries
            del self.loaded_files[file_path]
            del self.vtk_data_dict[file_path]
            
            # Remove actor if exists
            actor_key = f"flux_actor_{file_path}"
            if hasattr(self, actor_key):
                actor = getattr(self, actor_key)
                self.renderer.RemoveActor(actor)
                delattr(self, actor_key)
            
            # Update visualization
            self._update_visualization()
            
            # Update status
            num_files = len(self.loaded_files)
            if num_files > 0:
                self.status_label.setText(f"✓ {num_files} file(s) loaded")
            else:
                self.status_label.setText("Ready - Load flux and orbital files")
            
            self.vtk_widget.GetRenderWindow().Render()
    
    def _load_orbital_files(self):
        """Load one or more orbital CSV files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Orbital File(s)", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            self._add_orbital_file(file_path)
    
    def _add_orbital_file(self, file_path):
        """Add a single orbital file"""
        try:
            # Check if already loaded
            if file_path in self.loaded_orbital_files:
                QMessageBox.information(self, "Info", 
                    f"File already loaded:\n{Path(file_path).name}")
                return
            
            # Load orbital data
            orbital_data = OrbitalDataLoader.load_csv(file_path)
            
            # Store orbital data
            self.orbital_data_dict[file_path] = orbital_data
            
            # Create file widget
            file_widget = LoadedFileWidget(file_path, file_type="orbital")
            file_widget.orbital_data = orbital_data
            
            # Connect signals
            file_widget.remove_button.clicked.connect(
                lambda: self._remove_orbital_file(file_path))
            file_widget.checkbox.stateChanged.connect(self._update_orbital_visualization)
            if hasattr(file_widget, 'color_combo'):
                file_widget.color_combo.currentTextChanged.connect(
                    self._update_orbital_visualization)
            
            # Add to panel
            self.data_panel.add_orbital_file_widget(file_widget)
            self.loaded_orbital_files[file_path] = file_widget
            
            # Update visualization
            self._update_orbital_visualization()
            
            # Setup time slider if this is the first orbital file
            if len(self.loaded_orbital_files) == 1:
                self.animation_panel.set_time_range(len(orbital_data) - 1)
            
            # Get info
            info = OrbitalDataLoader.get_trajectory_info(orbital_data)
            
            # Update status
            num_files = len(self.loaded_orbital_files)
            self.status_label.setText(
                f"✓ Loaded {num_files} orbital file(s)\n"
                f"Latest: {Path(file_path).name}\n"
                f"Duration: {info['time_span']:.2f}h | "
                f"Alt: {info['altitude_min']:.0f}-{info['altitude_max']:.0f}km"
            )
            
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load orbit:\n{str(e)}")
    
    def _remove_orbital_file(self, file_path):
        """Remove an orbital file - UPDATED TO USE NEW RENDERERS"""
        if file_path in self.loaded_orbital_files:
            # Remove widget
            widget = self.loaded_orbital_files[file_path]
            self.data_panel.remove_orbital_file_widget(widget)
            widget.deleteLater()
            
            # Remove from dictionaries
            del self.loaded_orbital_files[file_path]
            del self.orbital_data_dict[file_path]
            
            # UPDATED: Use new renderers to remove
            self.orbital_renderer.remove_path(file_path)
            self.satellite_renderer.remove_satellite(file_path)
            
            # Update time slider
            if len(self.loaded_orbital_files) == 0:
                self.animation_panel.time_slider.setEnabled(False)
                self.animation_panel.set_time_label("No data loaded")
            
            # Update status
            num_files = len(self.loaded_orbital_files)
            if num_files > 0:
                self.status_label.setText(f"✓ {num_files} orbital file(s) loaded")
            else:
                self.status_label.setText("Ready - Load flux and orbital files")
            
            self.vtk_widget.GetRenderWindow().Render()
    
    def _update_visualization(self):
        """Update field visualization for all loaded files"""
        # Remove all existing flux actors
        for file_path in list(self.vtk_data_dict.keys()):
            actor_key = f"flux_actor_{file_path}"
            if hasattr(self, actor_key):
                actor = getattr(self, actor_key)
                self.renderer.RemoveActor(actor)
                delattr(self, actor_key)
        
        # Create actors for checked files
        for file_path, file_widget in self.loaded_files.items():
            if file_widget.is_checked():
                self._create_field_visualization_for_file(file_path, file_widget)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def _create_field_visualization_for_file(self, file_path, file_widget):
        """Create visualization for a specific flux file"""
        vtk_data = self.vtk_data_dict[file_path]
        particle_type = file_widget.get_particle_type()
        
        # Simple threshold filter
        scalar_range = vtk_data.GetScalarRange()
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(vtk_data)
        threshold.SetLowerThreshold(scalar_range[1] * 0.01)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
        threshold.Update()
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(threshold.GetOutputPort())
        mapper.SetScalarRange(scalar_range)
        
        # Get colormap settings
        colormap_name = self.viz_panel.get_colormap()
        scale_mode = self.viz_panel.get_scale()
        
        # Particle-specific colormaps
        particle_colormaps = {
            "electron": "Viridis",
            "proton": "Plasma",
            "alpha": "Inferno",
            "heavy_ion": "Magma",
            "neutron": "Cividis",
            "gamma": "Turbo",
            "cosmic_ray": "Twilight"
        }
        
        if particle_type in particle_colormaps:
            colormap_name = particle_colormaps[particle_type]
        
        lut = self.color_manager.create_lookup_table_with_scale(
            colormap_name, scale_mode, scalar_range
        )
        mapper.SetLookupTable(lut)
        
        # Create actor
        actor_key = f"flux_actor_{file_path}"
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(5)
        actor.GetProperty().SetOpacity(self.viz_panel.get_opacity() / 100.0)
        
        setattr(self, actor_key, actor)
        self.renderer.AddActor(actor)
    
    def _update_orbital_visualization(self):
        """Update orbital visualization for all loaded files"""
        # Clear and recreate all orbital visualizations
        for file_path, file_widget in self.loaded_orbital_files.items():
            if file_widget.is_checked():
                self._create_orbital_visualization_for_file(file_path, file_widget)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def _create_orbital_visualization_for_file(self, file_path, file_widget):
        """Create orbital path and satellite for a specific file"""
        orbital_data = self.orbital_data_dict[file_path]
        color_name = file_widget.get_color()
        
        # Color mapping
        color_map = {
            "Yellow": (1.0, 1.0, 0.0),
            "Cyan": (0.0, 1.0, 1.0),
            "Magenta": (1.0, 0.0, 1.0),
            "Green": (0.0, 1.0, 0.0),
            "Orange": (1.0, 0.5, 0.0),
            "White": (1.0, 1.0, 1.0),
            "Red": (1.0, 0.0, 0.0),
            "Blue": (0.0, 0.0, 1.0)
        }
        color = color_map.get(color_name, (1.0, 1.0, 0.0))
        
        self.orbital_renderer.create_orbital_path(orbital_data, color, file_path)
        
        if orbital_data:
            first_point = orbital_data[0]
            initial_position = (first_point.x, first_point.y, first_point.z)
            sat_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
            self.satellite_renderer.create_satellite(
                initial_position, sat_color, 500.0, file_path
            )
    
    def _update_earth_opacity(self, value):
        """Update Earth opacity"""
        opacity = value / 100.0
        self.earth_renderer.set_opacity(opacity)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _toggle_grid(self, enabled):
        """Toggle lat/long grid"""
        self.earth_renderer.toggle_grid(enabled)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _toggle_orbital_paths(self, visible):
        """Toggle orbital path visibility"""
        self.orbital_renderer.toggle_visibility(visible)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _toggle_trails(self, visible):
        """Toggle satellite trail visibility"""
        self.satellite_renderer.toggle_trail_visibility(visible)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _reset_camera(self):
        """Reset camera to default position - also make it dynamic"""
        distance = self._calculate_optimal_camera_distance()

        # Use config for angles if available, otherwise default
        if hasattr(self.config, 'CAMERA_DEFAULT_POSITION'):
            # Scale the default position by the calculated distance
            default_pos = self.config.CAMERA_DEFAULT_POSITION
            default_distance = np.sqrt(sum(x**2 for x in default_pos))
            scale_factor = distance / default_distance if default_distance > 0 else 1.0

            self.camera.SetPosition(
                default_pos[0] * scale_factor,
                default_pos[1] * scale_factor,
                default_pos[2] * scale_factor
            )
        else:
            # Default isometric view
            self.camera.SetPosition(
                distance * 0.7,
                distance * 0.7,
                distance * 0.5
            )

        self.camera.SetFocalPoint(*self.config.CAMERA_DEFAULT_FOCAL_POINT)
        self.camera.SetViewUp(*self.config.CAMERA_DEFAULT_VIEW_UP)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
    
    def _start_animation(self):
        """Start animation"""
        if not self.orbital_data_dict:
            QMessageBox.warning(self, "Warning", "Please load orbital data first")
            return
        
        self.is_playing = True
        speed = self.animation_panel.get_speed()
        self.animation_timer.start(speed)
        
        self.animation_panel.set_animation_playing(True)
    
    def _pause_animation(self):
        """Pause animation"""
        self.is_playing = False
        self.animation_timer.stop()
        
        self.animation_panel.set_animation_playing(False)
    
    def _stop_animation(self):
        """Stop animation"""
        self.is_playing = False
        self.animation_timer.stop()
        self.current_time_index = 0
        
        # ADDED: Clear all trails when stopping
        self.satellite_renderer.clear_all_trails()
        
        self._update_satellite_positions()
        
        self.animation_panel.set_animation_playing(False)
    
    def _animation_step(self):
        """Animation step"""
        if self.orbital_data_dict:
            # Use the longest orbital data for animation length
            max_length = max(len(data) for data in self.orbital_data_dict.values())
            
            if self.current_time_index >= max_length - 1:
                self.current_time_index = 0
                # ADDED: Clear trails when looping
                self.satellite_renderer.clear_all_trails()
            else:
                self.current_time_index += 1
            
            self.animation_panel.set_time_value(self.current_time_index)
            self._update_satellite_positions()
    
    def _on_time_changed(self, value):
        """Handle time slider change"""
        old_index = self.current_time_index
        self.current_time_index = value
        
        # Clear trails if jumping backwards or far forward
        if value < old_index or abs(value - old_index) > 10:
            self.satellite_renderer.clear_all_trails()
        
        self._update_satellite_positions()
    
    def _update_animation_speed(self, value):
        """Update animation speed"""
        if self.is_playing:
            self.animation_timer.setInterval(value)
    
    def _update_satellite_positions(self):
        """Update satellite positions for all loaded orbits"""
        # Update each satellite based on its orbital data
        for file_path, file_widget in self.loaded_orbital_files.items():
            if not file_widget.is_checked():
                continue
            
            orbital_data = self.orbital_data_dict[file_path]
            
            if 0 <= self.current_time_index < len(orbital_data):
                point = orbital_data[self.current_time_index]
                position = (point.x, point.y, point.z)
                
                self.satellite_renderer.update_satellite_position(position, file_path)
        
        # Update time label if we have any orbital data
        if self.orbital_data_dict:
            # Use first orbital file for time display
            first_orbital = list(self.orbital_data_dict.values())[0]
            if 0 <= self.current_time_index < len(first_orbital):
                point = first_orbital[self.current_time_index]
                hours = point.time
                h = int(hours)
                m = int((hours - h) * 60)
                s = int(((hours - h) * 60 - m) * 60)
                self.animation_panel.set_time_label(f"{h:02d}:{m:02d}:{s:02d}")
                
                # Update status with flux if analysis is active
                if self.analysis_panel.get_current_mode() == "Flux Analysis" and self.vtk_data_dict:
                    total_flux = 0
                    for file_path, file_widget in self.loaded_files.items():
                        if file_widget.is_checked():
                            vtk_data = self.vtk_data_dict[file_path]
                            self.flux_analyzer.set_vtk_data(vtk_data)
                            flux = self.flux_analyzer.analyze_flux_at_point(point)
                            total_flux += flux
                    
                    self.status_label.setText(
                        f"Time: {point.time:.2f}h | Alt: {point.altitude:.1f}km\n"
                        f"Total Flux: {total_flux:.2e} particles/s | "
                        f"{len(self.loaded_orbital_files)} orbit(s)"
                    )
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def _on_viz_mode_changed(self, mode):
        """Handle visualization mode change"""
        # Mode-specific visualization updates would go here
        self._update_visualization()
    
    def _on_analysis_settings_changed(self):
        """Handle analysis settings change"""
        # Update flux analyzer with new settings
        if self.analysis_panel.get_current_mode() == "Flux Analysis":
            cross_section = self.analysis_panel.get_cross_section()
            self.flux_analyzer.set_cross_section(cross_section)
