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
from scene import EarthRenderer
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
        
        # Setup Earth
        self.earth_renderer = EarthRenderer(self.renderer)
        self.earth_renderer.create_earth()
        
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
        
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout.addWidget(self.vtk_widget)
        
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
        # Connect analysis window buttons when implemented
        
        # Earth controls
        self.earth_controls.opacity_changed.connect(self._update_earth_opacity)
        self.earth_controls.grid_toggled.connect(self._toggle_grid)
        self.earth_controls.reset_camera.connect(self._reset_camera)
    
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
        """Remove an orbital file"""
        if file_path in self.loaded_orbital_files:
            # Remove widget
            widget = self.loaded_orbital_files[file_path]
            self.data_panel.remove_orbital_file_widget(widget)
            widget.deleteLater()
            
            # Remove from dictionaries
            del self.loaded_orbital_files[file_path]
            del self.orbital_data_dict[file_path]
            
            # Remove actors
            self._remove_orbital_actors(file_path)
            
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
    
    def _remove_orbital_actors(self, file_path):
        """Remove orbital visualization actors"""
        path_actor_key = f"path_actor_{file_path}"
        sat_actor_key = f"satellite_actor_{file_path}"
        
        if hasattr(self, path_actor_key):
            actor = getattr(self, path_actor_key)
            self.renderer.RemoveActor(actor)
            delattr(self, path_actor_key)
        
        if hasattr(self, sat_actor_key):
            actor = getattr(self, sat_actor_key)
            self.renderer.RemoveActor(actor)
            delattr(self, sat_actor_key)
    
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
        # Remove all existing orbital actors
        for file_path in list(self.orbital_data_dict.keys()):
            self._remove_orbital_actors(file_path)
        
        # Create actors for checked files
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
        
        # Create path polyline
        points = vtk.vtkPoints()
        for point in orbital_data:
            points.InsertNextPoint(point.x, point.y, point.z)
        
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(orbital_data))
        for i in range(len(orbital_data)):
            polyline.GetPointIds().SetId(i, i)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        path_actor = vtk.vtkActor()
        path_actor.SetMapper(mapper)
        path_actor.GetProperty().SetColor(color)
        path_actor.GetProperty().SetLineWidth(2.0)
        
        # Store actor reference
        path_actor_key = f"path_actor_{file_path}"
        setattr(self, path_actor_key, path_actor)
        self.renderer.AddActor(path_actor)
        
        # Create satellite sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(500.0)  # 500 km for visibility
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        
        satellite_actor = vtk.vtkActor()
        satellite_actor.SetMapper(sphere_mapper)
        # Use slightly darker version of path color for satellite
        satellite_actor.GetProperty().SetColor(
            color[0] * 0.7, 
            color[1] * 0.7, 
            color[2] * 0.7
        )
        
        # Set initial position
        if orbital_data:
            first_point = orbital_data[0]
            satellite_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        # Store actor reference
        sat_actor_key = f"satellite_actor_{file_path}"
        setattr(self, sat_actor_key, satellite_actor)
        self.renderer.AddActor(satellite_actor)
    
    def _update_earth_opacity(self, value):
        """Update Earth opacity"""
        opacity = value / 100.0
        self.earth_renderer.set_opacity(opacity)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _toggle_grid(self, enabled):
        """Toggle lat/long grid"""
        self.earth_renderer.toggle_grid(enabled)
        self.vtk_widget.GetRenderWindow().Render()
    
    def _reset_camera(self):
        """Reset camera to default position"""
        self.camera.SetPosition(*self.config.CAMERA_DEFAULT_POSITION)
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
        self._update_satellite_positions()
        
        self.animation_panel.set_animation_playing(False)
    
    def _animation_step(self):
        """Animation step"""
        if self.orbital_data_dict:
            # Use the longest orbital data for animation length
            max_length = max(len(data) for data in self.orbital_data_dict.values())
            
            if self.current_time_index >= max_length - 1:
                self.current_time_index = 0
            else:
                self.current_time_index += 1
            
            self.animation_panel.set_time_value(self.current_time_index)
            self._update_satellite_positions()
    
    def _on_time_changed(self, value):
        """Handle time slider change"""
        self.current_time_index = value
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
                
            sat_actor_key = f"satellite_actor_{file_path}"
            if not hasattr(self, sat_actor_key):
                continue
                
            orbital_data = self.orbital_data_dict[file_path]
            
            if 0 <= self.current_time_index < len(orbital_data):
                point = orbital_data[self.current_time_index]
                actor = getattr(self, sat_actor_key)
                actor.SetPosition(point.x, point.y, point.z)
        
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
