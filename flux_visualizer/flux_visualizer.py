#!/usr/bin/env python3
"""
Electron Flux Orbital Visualizer
A scientific visualization tool for analyzing electron flux data along orbital paths.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QDoubleSpinBox, QFileDialog, 
    QMessageBox, QProgressBar, QGroupBox, QFormLayout, QSpinBox,
    QSplitter, QFrame, QComboBox, QCheckBox, QLineEdit
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# VTK-Qt integration
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class OrbitalPoint:
    """Data structure to hold orbital position data"""
    def __init__(self, time, x, y, z, vx=0, vy=0, vz=0):
        self.time = time  # hours
        self.x, self.y, self.z = x, y, z  # position in km
        self.vx, self.vy, self.vz = vx, vy, vz  # velocity in km/s
        self.phi = np.arctan2(y, x)  # azimuthal angle
        if self.phi < 0:
            self.phi += 2 * np.pi

class FluxAnalyzer:
    """Handles flux analysis calculations"""
    
    def __init__(self):
        self.vtk_data = None
        self.orbital_path = []
        self.cross_section_radius = 1.0  # meters
        
    def set_vtk_data(self, vtk_data):
        """Set the VTK dataset containing flux field"""
        self.vtk_data = vtk_data
        
    def set_orbital_data(self, orbital_points):
        """Set the orbital path data"""
        self.orbital_path = orbital_points
        
    def set_cross_section(self, radius_meters):
        """Set object cross-sectional radius"""
        self.cross_section_radius = radius_meters
        
    def analyze_flux_at_point(self, orbital_point):
        """Analyze flux at a specific orbital point"""
        if not self.vtk_data:
            return 0.0
            
        # Create a probe to sample the field at this point
        probe = vtk.vtkProbeFilter()
        
        # Create a point to probe
        points = vtk.vtkPoints()
        points.InsertNextPoint(orbital_point.x, orbital_point.y, orbital_point.z)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        probe.SetInputData(polydata)
        probe.SetSourceData(self.vtk_data)
        probe.Update()
        
        # Get the flux value
        result = probe.GetOutput()
        if result.GetNumberOfPoints() > 0:
            scalar_array = result.GetPointData().GetScalars()
            if scalar_array:
                flux_value = scalar_array.GetValue(0)
                # Calculate integrated flux through cross-sectional area
                area = np.pi * (self.cross_section_radius ** 2)
                return flux_value * area
                
        return 0.0

class SlicePlotWindow(QMainWindow):
    """Window for displaying orbital slice visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orbital Slice Visualization")
        self.setGeometry(100, 100, 800, 600)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)
        
        # VTK pipeline
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.2)  # Dark blue
        
        # Slice plane and cutter
        self.slice_plane = vtk.vtkPlane()
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(self.slice_plane)
        
        # Slice actor
        self.slice_mapper = vtk.vtkDataSetMapper()
        self.slice_mapper.SetInputConnection(self.cutter.GetOutputPort())
        
        self.slice_actor = vtk.vtkActor()
        self.slice_actor.SetMapper(self.slice_mapper)
        self.renderer.AddActor(self.slice_actor)
        
        # Object representation (small sphere)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(50.0)  # 50 km for visibility
        
        object_mapper = vtk.vtkPolyDataMapper()
        object_mapper.SetInputConnection(sphere.GetOutputPort())
        
        self.object_actor = vtk.vtkActor()
        self.object_actor.SetMapper(object_mapper)
        self.object_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        self.renderer.AddActor(self.object_actor)
        
        # Scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.slice_mapper.GetLookupTable())
        self.scalar_bar.SetTitle("Electron Flux")
        self.renderer.AddActor2D(self.scalar_bar)
        
    def update_slice(self, phi_angle, vtk_data):
        """Update the slice at constant phi angle"""
        if vtk_data is None:
            return
            
        # Set slice plane for constant phi
        nx = -np.sin(phi_angle)
        ny = np.cos(phi_angle) 
        nz = 0.0
        
        self.slice_plane.SetNormal(nx, ny, nz)
        self.slice_plane.SetOrigin(0.0, 0.0, 0.0)
        
        self.cutter.SetInputData(vtk_data)
        self.cutter.Update()
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def set_object_position(self, x, y, z):
        """Update object position"""
        self.object_actor.SetPosition(x, y, z)
        self.vtk_widget.GetRenderWindow().Render()

class SpectrumPlotWindow(QMainWindow):
    """Window for energy spectrum plotting with real-time updates"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Energy Spectrum - Real-time")
        self.setGeometry(200, 200, 700, 500)
        
        # Reference to parent for data access
        self.parent_app = parent
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        
        # Energy bins for electron spectrum (keV)
        self.energy_bins = np.logspace(1, 4, 50)  # 10 keV to 10 MeV, 50 bins
        self.energy_centers = (self.energy_bins[:-1] + self.energy_bins[1:]) / 2
        
        # Initialize plot
        self.setup_plot()
        
        # Status info
        self.info_label = QLabel("Spectrum updates in real-time during animation")
        self.info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Current spectrum data
        self.current_spectrum = np.zeros_like(self.energy_centers)
        
    def setup_plot(self):
        """Setup the spectrum plot with proper styling"""
        self.ax.clear()
        
        # Create the spectrum line plot
        self.spectrum_line, = self.ax.loglog(self.energy_centers, 
                                           np.ones_like(self.energy_centers) * 1e-10, 
                                           'b-', linewidth=2, label='Electron Flux')
        
        # Styling
        self.ax.set_xlabel('Energy (keV)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Flux (particles/cm²/s/keV)', fontsize=12, fontweight='bold')
        self.ax.set_title('Electron Energy Spectrum at Satellite Position', fontsize=14, fontweight='bold')
        
        # Set axis limits
        self.ax.set_xlim(10, 10000)  # 10 keV to 10 MeV
        self.ax.set_ylim(1e-2, 1e8)   # Reasonable flux range
        
        # Grid and legend
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Tight layout
        self.figure.tight_layout()
        self.canvas.draw()
        
    def calculate_energy_spectrum(self, total_flux, altitude_km):
        """Calculate realistic energy spectrum from total flux and altitude"""
        try:
            if total_flux <= 0:
                return np.zeros_like(self.energy_centers)
            
            # Van Allen belt energy spectrum model
            # Different spectral shapes for different regions
            
            if altitude_km < 1000:  # Low Earth orbit
                # Harder spectrum, more high-energy particles
                spectral_index = -2.5
                e_char = 100  # keV
            elif 1000 <= altitude_km < 5000:  # Inner Van Allen belt region
                # Softer spectrum with exponential cutoff
                spectral_index = -1.8
                e_char = 200  # keV
            elif 5000 <= altitude_km < 15000:  # Peak Van Allen belt
                # Very hard spectrum, lots of relativistic electrons
                spectral_index = -1.2
                e_char = 500  # keV
            else:  # Outer belt and beyond
                # Soft spectrum
                spectral_index = -3.0
                e_char = 50   # keV
            
            # Calculate differential flux: dJ/dE = J0 * (E/E0)^spectral_index * exp(-E/E_char)
            E0 = 100  # Reference energy in keV
            
            # Power law with exponential cutoff
            differential_flux = (self.energy_centers / E0) ** spectral_index * np.exp(-self.energy_centers / e_char)
            
            # Normalize to match total flux
            # Total flux ≈ integral of differential flux over energy
            total_calculated = np.trapz(differential_flux, self.energy_centers)
            if total_calculated > 0:
                differential_flux *= (total_flux / total_calculated)
            
            # Add some realistic spectral features
            # Electron cyclotron resonance enhancement around 100-300 keV
            if 1000 <= altitude_km <= 10000:
                resonance_energy = 200  # keV
                resonance_width = 100   # keV
                resonance_enhancement = 2.0 * np.exp(-((self.energy_centers - resonance_energy) / resonance_width)**2)
                differential_flux *= (1 + 0.3 * resonance_enhancement)
            
            return np.maximum(differential_flux, 1e-10)  # Floor to avoid zeros
            
        except Exception as e:
            print(f"Error calculating energy spectrum: {e}")
            return np.ones_like(self.energy_centers) * 1e-6
    
    def update_spectrum_from_satellite_position(self):
        """Update spectrum based on current satellite position"""
        try:
            if not self.parent_app:
                return
                
            # Get current satellite position and flux
            if (hasattr(self.parent_app, 'orbital_path') and 
                self.parent_app.orbital_path and
                0 <= self.parent_app.current_time_index < len(self.parent_app.orbital_path)):
                
                current_point = self.parent_app.orbital_path[self.parent_app.current_time_index]
                
                # Calculate altitude
                distance_from_earth = np.sqrt(current_point.x**2 + current_point.y**2 + current_point.z**2)
                altitude_km = distance_from_earth - 6371  # Earth radius
                
                # Get total flux at this position
                total_flux = self.parent_app.flux_analyzer.analyze_flux_at_point(current_point)
                
                # Calculate energy spectrum
                self.current_spectrum = self.calculate_energy_spectrum(total_flux, altitude_km)
                
                # Update plot
                self.spectrum_line.set_ydata(self.current_spectrum)
                
                # Update info
                self.info_label.setText(
                    f"Time: {current_point.time:.2f}h | Alt: {altitude_km:.1f}km | "
                    f"Total Flux: {total_flux:.2e} particles/cm²/s"
                )
                
                # Redraw
                self.canvas.draw_idle()
                
            else:
                # No satellite data
                self.info_label.setText("No satellite position data available")
                
        except Exception as e:
            print(f"Error updating spectrum: {e}")
            self.info_label.setText(f"Error updating spectrum: {e}")
    
    # def update_spectrum(self, energy_bins=None, spectrum_data=None):
    #     """Legacy method for compatibility - redirects to satellite-based update"""
    #     if energy_bins is not None and spectrum_data is not None:
    #         # Direct spectrum data provided
    #         try:
    #             self.spectrum_line.set_xdata(energy_bins)
    #             self.spectrum_line.set_ydata(spectrum_data)
    #             self.ax.relim()
    #             self.ax.autoscale_view()
    #             self.canvas.draw_idle()
    #             self.info_label.setText(f"Updated with {len(spectrum_data)} energy bins")
    #         except:
    #             pass
    #     else:
    #         # Use satellite position
    #         self.update_spectrum_from_satellite_position()
    
    # def closeEvent(self, event):
    #     """Handle window close event"""
    #     event.accept()

class FluxTimePlotWindow(QMainWindow):
    """Window for flux vs time plotting"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flux vs Time")
        self.setGeometry(300, 300, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        self.status_label = QLabel("Flux vs time visualization\n(Matplotlib integration coming soon)")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.flux_data = []
        self.time_data = []
        
    def add_flux_point(self, time, flux):
        """Add a new flux measurement point"""
        self.time_data.append(time)
        self.flux_data.append(flux)
        self.status_label.setText(f"Flux data: {len(self.flux_data)} points\n"
                                f"Latest: {flux:.2e} particles/s at t={time:.2f}h")
        
    def clear_data(self):
        """Clear all flux data"""
        self.flux_data.clear()
        self.time_data.clear()
        self.status_label.setText("Flux vs time visualization\n(Data cleared)")

class ElectronFluxVisualizer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        """Constructor - FIXED timer initialization"""
        super().__init__()
        print("DEBUG: ElectronFluxVisualizer.__init__() called")
        
        self.setWindowTitle("Electron Flux Orbital Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data
        self.vtk_data = None
        self.orbital_path = []
        self.current_time_index = 0
        self.is_playing = False
        
        # Analysis
        self.flux_analyzer = FluxAnalyzer()
        
        # Plot windows
        self.slice_window = None
        self.spectrum_window = None
        self.flux_time_window = None
        
        # Animation timer - FIXED initialization
        print("DEBUG: Creating animation timer...")
        self.animation_timer = QTimer(self)  # CRITICAL: Pass parent
        print(f"DEBUG: Animation timer created with parent: {self.animation_timer.parent()}")
        
        # IMMEDIATELY connect the timer signal
        print("DEBUG: Connecting timer signal...")
        self.animation_timer.timeout.connect(self.animation_step)
        print("DEBUG: Timer signal connected to animation_step")
        
        # Set up UI and VTK
        self.setup_ui()
        self.setup_vtk()
        self.setup_visualization_controls()
        self.add_debug_button_to_controls()
        
        print("DEBUG: ElectronFluxVisualizer.__init__() completed")

    def debug_renderer_state(self):
        """Debug method to check renderer state"""
        print("\n=== RENDERER DEBUG INFO ===")
        
        # Check actors
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor_count = 0
        while actors.GetNextActor():
            actor_count += 1
        print(f"Total actors: {actor_count}")
        
        # Check volumes
        volumes = self.renderer.GetVolumes()
        volumes.InitTraversal()
        volume_count = 0
        while volumes.GetNextVolume():
            volume_count += 1
        print(f"Total volumes: {volume_count}")
        
        # Check view props
        props = self.renderer.GetViewProps()
        props.InitTraversal()
        prop_count = 0
        while props.GetNextProp():
            prop_count += 1
        print(f"Total view props: {prop_count}")
        
        # Camera info
        camera = self.renderer.GetActiveCamera()
        print(f"Camera position: {camera.GetPosition()}")
        print(f"Camera focal point: {camera.GetFocalPoint()}")
        print(f"Camera view up: {camera.GetViewUp()}")
        
        # Data bounds if available
        if hasattr(self, 'vtk_data') and self.vtk_data:
            bounds = self.vtk_data.GetBounds()
            print(f"Data bounds: X({bounds[0]:.1f}, {bounds[1]:.1f}) Y({bounds[2]:.1f}, {bounds[3]:.1f}) Z({bounds[4]:.1f}, {bounds[5]:.1f})")
        
        print("=========================\n")

    def debug_scalar_ranges(self):
        """Debug method to check all the ranges and identify NaN source - FIXED"""
        print("\n=== SCALAR RANGE DEBUG ===")
        
        if hasattr(self, 'vtk_data') and self.vtk_data:
            raw_range = self.vtk_data.GetScalarRange()
            print(f"Raw VTK data range: {raw_range}")
        
        if hasattr(self, 'current_scalar_range'):
            print(f"Stored current_scalar_range: {self.current_scalar_range}")
        
        if hasattr(self, 'field_actor') and self.field_actor:
            mapper = self.field_actor.GetMapper()
            if mapper:
                mapper_range = mapper.GetScalarRange()
                print(f"Mapper range: {mapper_range}")
                
                lut = mapper.GetLookupTable()
                if lut:
                    lut_range = lut.GetRange()
                    print(f"LUT range: {lut_range}")
                    
                    # Test some values - FIXED method call
                    test_val = (lut_range[0] + lut_range[1]) / 2
                    test_color = [0, 0, 0]  # Output array
                    lut.GetColor(test_val, test_color)  # Correct VTK call
                    print(f"Test color at {test_val}: {test_color}")
        
        if hasattr(self, 'scalar_bar') and self.scalar_bar:
            sb_lut = self.scalar_bar.GetLookupTable()
            if sb_lut:
                sb_range = sb_lut.GetRange()
                print(f"Scalar bar LUT range: {sb_range}")
        
        cutoff = getattr(self, 'current_flux_cutoff', 1e-5)
        print(f"Current flux cutoff: {cutoff}")
        
        # Check if zero values are causing issues
        if hasattr(self, 'current_scalar_range'):
            min_val, max_val = self.current_scalar_range
            if min_val == 0:
                print("WARNING: Data starts at 0.0 - this breaks logarithmic scaling!")
                non_zero_min = max_val * 1e-6  # Suggested minimum
                print(f"Suggested minimum for log scale: {non_zero_min:.2e}")
        
        print("========================\n")

        
    def test_simple_visualization(self):
        """Test method to create a simple visible object"""
        print("=== TESTING SIMPLE VISUALIZATION ===")
        
        if not self.vtk_data:
            print("No VTK data available")
            return
        
        try:
            # Clear everything first
            #self.clear_field_visualization()
            
            # Create a simple sphere at the center of the data
            bounds = self.vtk_data.GetBounds()
            center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
            radius = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) / 10
            
            print(f"Creating test sphere at {center} with radius {radius}")
            
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(center)
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Bright red
            actor.GetProperty().SetOpacity(1.0)
            
            self.renderer.AddActor(actor)
            
            # Store as field actor for cleanup
            self.field_actor = actor
            
            # Reset camera and render
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            print("Test sphere created and rendered")
            
        except Exception as e:
            print(f"Error in test visualization: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_ui(self):
        """Setup the user interface with wider control panel"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - VTK 3D visualization
        vtk_frame = QFrame()
        vtk_layout = QVBoxLayout(vtk_frame)
        
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout.addWidget(self.vtk_widget)
        
        splitter.addWidget(vtk_frame)

        self.create_earth_opacity_control(vtk_layout)
        
        # Right panel - controls (WIDER)
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        
        # File loading group
        file_group = QGroupBox("Data Loading")
        file_layout = QFormLayout(file_group)
        
        self.load_vtk_button = QPushButton("Load VTK Data")
        self.load_orbit_button = QPushButton("Load Orbital CSV")
        
        file_layout.addRow(self.load_vtk_button)
        file_layout.addRow(self.load_orbit_button)
        
        self.control_layout.addWidget(file_group)
        
        # Animation controls group
        anim_group = QGroupBox("Animation Controls")
        anim_layout = QVBoxLayout(anim_group)
        
        # Play/pause/stop buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        anim_layout.addLayout(button_layout)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        anim_layout.addWidget(QLabel("Time:"))
        anim_layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("00:00:00")
        anim_layout.addWidget(self.time_label)
        
        # Speed control
        speed_layout = QFormLayout()
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 1000)
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.setSuffix(" ms")
        speed_layout.addRow("Animation Speed:", self.speed_spinbox)
        anim_layout.addLayout(speed_layout)
        
        self.control_layout.addWidget(anim_group)
        
        # Analysis parameters group
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout(analysis_group)
        
        self.cross_section_spinbox = QDoubleSpinBox()
        self.cross_section_spinbox.setRange(0.1, 100.0)
        self.cross_section_spinbox.setValue(1.0)
        self.cross_section_spinbox.setSingleStep(0.1)
        self.cross_section_spinbox.setSuffix(" m")
        
        analysis_layout.addRow("Cross Section Radius:", self.cross_section_spinbox)
        self.control_layout.addWidget(analysis_group)
        
        # Plot windows group
        plots_group = QGroupBox("Plot Windows")
        plots_layout = QVBoxLayout(plots_group)
        
        self.show_slice_button = QPushButton("Show Slice Plot")
        self.show_spectrum_button = QPushButton("Show Spectrum Plot")
        self.show_flux_time_button = QPushButton("Show Flux vs Time")
        
        plots_layout.addWidget(self.show_slice_button)
        plots_layout.addWidget(self.show_spectrum_button)
        plots_layout.addWidget(self.show_flux_time_button)
        
        self.control_layout.addWidget(plots_group)

        debug_group = QGroupBox("Debug Controls")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_button = QPushButton("Debug Renderer State")
        self.debug_button.clicked.connect(self.debug_renderer_state)
        debug_layout.addWidget(self.debug_button)
        
        self.test_viz_button = QPushButton("Test Simple Visualization")
        self.test_viz_button.clicked.connect(self.test_simple_visualization)
        debug_layout.addWidget(self.test_viz_button)
        
        # Status
        self.status_label = QLabel("Ready - Load VTK data and orbital CSV to begin")
        self.status_label.setWordWrap(True)
        self.control_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.control_layout.addWidget(self.progress_bar)
        
        self.control_layout.addStretch()
        
        # WIDER control panel (was 350, now 450)
        control_panel.setMaximumWidth(450)
        control_panel.setMinimumWidth(450)
        splitter.addWidget(control_panel)
        
        # Set splitter proportions (adjusted for wider panel)
        splitter.setSizes([750, 450])
        
        self.connect_signals()
        
    def setup_vtk(self):
        """Setup VTK 3D visualization - REVERTED TO ORIGINAL"""
        print("=== SETTING UP VTK RENDERER ===")
        
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.05, 0.05, 0.15)  # Dark blue background
        
        # Camera setup for Earth-centered view
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(20000, 20000, 10000)  # km
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        
        print("Renderer and camera setup complete")
        
        # Create Earth representation ONCE, with error handling
        print(">>> About to call create_earth_representation...")
        earth_success = self.create_earth_representation()
        print(f">>> create_earth_representation returned: {earth_success} (type: {type(earth_success)})")
        
        if earth_success:
            print("Earth created successfully")
        else:
            print("WARNING: Earth creation failed, continuing without Earth")
        
        # Create finite starfield background
        print("Creating finite starfield background...")
        starfield_success = self.create_starfield_background()
        if starfield_success:
            print("Starfield background created successfully")
        else:
            print("WARNING: Starfield creation failed")
        
        # Add coordinate axes
        print("Setting up coordinate axes...")
        try:
            self.setup_coordinate_axes()
            print("Coordinate axes setup complete")
        except Exception as e:
            print(f"WARNING: Coordinate axes setup failed: {e}")
        
        print("=== VTK SETUP COMPLETE ===")

    def setup_coordinate_axes(self):
        """Setup dynamic coordinate axes with snap-to-axis controls"""
        # Create axes actor with proper labels and units
        self.axes_actor = vtk.vtkAxesActor()
        
        # Set axis labels with units
        self.axes_actor.SetXAxisLabelText("X (km)")
        self.axes_actor.SetYAxisLabelText("Y (km)")
        self.axes_actor.SetZAxisLabelText("Z (km)")
        
        # Style the axes
        self.axes_actor.SetTotalLength(5000, 5000, 5000)  # 5000 km length
        self.axes_actor.SetShaftTypeToLine()
        self.axes_actor.SetAxisLabels(True)
        
        # Improve text properties
        for axis_label in [self.axes_actor.GetXAxisCaptionActor2D(),
                          self.axes_actor.GetYAxisCaptionActor2D(),
                          self.axes_actor.GetZAxisCaptionActor2D()]:
            axis_label.GetTextActor().SetTextScaleModeToNone()
            axis_label.GetCaptionTextProperty().SetFontSize(12)
            axis_label.GetCaptionTextProperty().SetColor(1, 1, 1)
            axis_label.GetCaptionTextProperty().BoldOn()
        
        # Create orientation widget to show axes in corner
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(self.axes_actor)
        self.orientation_widget.SetInteractor(self.vtk_widget)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.3, 0.3)  # Bottom left corner
        self.orientation_widget.EnabledOn()
        self.orientation_widget.InteractiveOff()  # Don't allow user to move it
        
        # Add snap-to-axis buttons next to the orientation widget
        self.setup_snap_to_axis_buttons()
        
        print("Dynamic coordinate axes with snap buttons added")

    def create_circle(self, radius, height, circle_type):
        """Create a circle for latitude lines - with proper default opacity"""
        try:
            # Create a circle in the XY plane
            circle_source = vtk.vtkRegularPolygonSource()
            circle_source.SetNumberOfSides(60)  # Smooth circle
            circle_source.SetRadius(radius)
            circle_source.SetCenter(0, 0, height)
            circle_source.SetNormal(0, 0, 1)  # XY plane
            circle_source.Update()
            
            # Create mapper
            circle_mapper = vtk.vtkPolyDataMapper()
            circle_mapper.SetInputConnection(circle_source.GetOutputPort())
            
            # Create actor
            circle_actor = vtk.vtkActor()
            circle_actor.SetMapper(circle_mapper)
            
            # CRITICAL: Set to wireframe representation (lines only)
            circle_actor.GetProperty().SetRepresentationToWireframe()
            
            # Set properties based on type
            if circle_type == 'equator':
                circle_actor.GetProperty().SetColor(1.0, 0.6, 0.0)  # Orange
                circle_actor.GetProperty().SetLineWidth(2.5)
                circle_actor.GetProperty().SetOpacity(0.72)  # 90% of default 80% Earth
            else:  # Regular latitude line
                circle_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
                circle_actor.GetProperty().SetLineWidth(1.5)
                circle_actor.GetProperty().SetOpacity(0.64)  # 80% of default 80% Earth
            
            return circle_actor
            
        except Exception as e:
            print(f"Error creating circle: {e}")
            return None
        
    def setup_snap_to_axis_buttons(self):
        """Add snap-to-axis buttons overlaid on the VTK widget"""
        # Create a widget to hold the snap buttons
        self.snap_buttons_widget = QWidget(self.vtk_widget)
        self.snap_buttons_widget.setGeometry(10, 10, 200, 30)  # Top-left corner
        
        # Create horizontal layout for buttons
        snap_layout = QHBoxLayout(self.snap_buttons_widget)
        snap_layout.setContentsMargins(2, 2, 2, 2)
        snap_layout.setSpacing(3)
        
        # Create snap buttons with tooltips
        self.snap_x_button = QPushButton("X")
        self.snap_x_button.setMaximumSize(25, 25)
        self.snap_x_button.setToolTip("Snap to X-axis view (X→right, Y→up, Z→out)")
        self.snap_x_button.clicked.connect(self.snap_to_x_axis)
        self.snap_x_button.setStyleSheet("""
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
        
        self.snap_y_button = QPushButton("Y")
        self.snap_y_button.setMaximumSize(25, 25)
        self.snap_y_button.setToolTip("Snap to Y-axis view (Y→right, Z→up, X→out)")
        self.snap_y_button.clicked.connect(self.snap_to_y_axis)
        self.snap_y_button.setStyleSheet("""
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
        
        self.snap_z_button = QPushButton("Z")
        self.snap_z_button.setMaximumSize(25, 25)
        self.snap_z_button.setToolTip("Snap to Z-axis view (X→right, Z→up, Y→out)")
        self.snap_z_button.clicked.connect(self.snap_to_z_axis)
        self.snap_z_button.setStyleSheet("""
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
        
        # Add ISO button for isometric view
        self.snap_iso_button = QPushButton("ISO")
        self.snap_iso_button.setMaximumSize(35, 25)
        self.snap_iso_button.setToolTip("Snap to isometric 3D view")
        self.snap_iso_button.clicked.connect(self.snap_to_isometric)
        self.snap_iso_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(100, 100, 100, 180);
                color: white;
                border: 1px solid white;
                border-radius: 3px;
                font-weight: bold;
                font-size: 9px;
            }
            QPushButton:hover {
                background-color: rgba(150, 150, 150, 200);
            }
        """)
        
        # Add buttons to layout
        snap_layout.addWidget(QLabel("View:"))
        snap_layout.addWidget(self.snap_x_button)
        snap_layout.addWidget(self.snap_y_button) 
        snap_layout.addWidget(self.snap_z_button)
        snap_layout.addWidget(self.snap_iso_button)
        snap_layout.addStretch()
        
        # Make the widget visible
        self.snap_buttons_widget.show()
        self.snap_buttons_widget.raise_()

    def snap_to_x_axis(self):
        """Snap camera to X-axis aligned view"""
        print("Snapping to X-axis view...")
        
        if not hasattr(self, 'vtk_data') or not self.vtk_data:
            # Use default center
            focal_point = [0, 0, 0]
            distance = 30000
        else:
            # Use data center
            bounds = self.vtk_data.GetBounds()
            focal_point = [
                (bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2
            ]
            max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            distance = max_range * 1.5
        
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_point)
        
        # X→right, Y→up, Z→out of page
        # Camera looks from negative Z direction
        camera.SetPosition(focal_point[0], focal_point[1], focal_point[2] - distance)
        camera.SetViewUp(0, 1, 0)  # Y axis points up
        
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def snap_to_y_axis(self):
        """Snap camera to Y-axis aligned view"""
        print("Snapping to Y-axis view...")
        
        if not hasattr(self, 'vtk_data') or not self.vtk_data:
            focal_point = [0, 0, 0]
            distance = 30000
        else:
            bounds = self.vtk_data.GetBounds()
            focal_point = [
                (bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2
            ]
            max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            distance = max_range * 1.5
        
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_point)
        
        # Y→right, Z→up, X→out of page
        # Camera looks from negative X direction
        camera.SetPosition(focal_point[0] - distance, focal_point[1], focal_point[2])
        camera.SetViewUp(0, 0, 1)  # Z axis points up
        
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def snap_to_z_axis(self):
        """Snap camera to Z-axis aligned view"""
        print("Snapping to Z-axis view...")
        
        if not hasattr(self, 'vtk_data') or not self.vtk_data:
            focal_point = [0, 0, 0]
            distance = 30000
        else:
            bounds = self.vtk_data.GetBounds()
            focal_point = [
                (bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2
            ]
            max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            distance = max_range * 1.5
        
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_point)
        
        # X→right, Z→up, Y→out of page
        # Camera looks from negative Y direction
        camera.SetPosition(focal_point[0], focal_point[1] - distance, focal_point[2])
        camera.SetViewUp(0, 0, 1)  # Z axis points up
        
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def snap_to_isometric(self):
        """Snap camera to nice isometric 3D view"""
        print("Snapping to isometric view...")
        
        if not hasattr(self, 'vtk_data') or not self.vtk_data:
            focal_point = [0, 0, 0]
            distance = 30000
        else:
            bounds = self.vtk_data.GetBounds()
            focal_point = [
                (bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2
            ]
            max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            distance = max_range * 1.5
        
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_point)
        
        # Nice isometric view: roughly 45° angles
        camera.SetPosition(
            focal_point[0] + distance * 0.7,
            focal_point[1] + distance * 0.7,
            focal_point[2] + distance * 0.5
        )
        camera.SetViewUp(0, 0, 1)  # Z axis points up
        
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_starfield(self, state):
        """Toggle starfield background visibility - REVERTED TO ORIGINAL"""
        try:
            show_starfield = state == 2  # Qt.Checked = 2
            
            print(f"Toggling starfield: {'ON' if show_starfield else 'OFF'}")
            
            if show_starfield:
                # Create or show the starfield
                if not hasattr(self, 'starfield_actor') or not self.starfield_actor:
                    self.create_starfield_background()
                else:
                    # Show existing starfield
                    self.starfield_actor.SetVisibility(True)
            else:
                # Hide the starfield
                if hasattr(self, 'starfield_actor') and self.starfield_actor:
                    self.starfield_actor.SetVisibility(False)
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                
            status = "visible" if show_starfield else "hidden"
            print(f"Starfield is now {status}")
                
        except Exception as e:
            print(f"Error toggling starfield: {e}")
            import traceback
            traceback.print_exc()
        
    def create_starfield_background(self):
        """Create a realistic starfield background sphere"""
        try:
            print("Creating starfield background...")
            
            # Remove existing starfield if it exists
            if hasattr(self, 'starfield_actor'):
                self.renderer.RemoveActor(self.starfield_actor)
            
            # Create a very large sphere to contain the scene
            starfield_radius = 500000.0  # 500,000 km - much larger than data bounds
            
            starfield_sphere = vtk.vtkSphereSource()
            starfield_sphere.SetRadius(starfield_radius)
            starfield_sphere.SetThetaResolution(120)  # High resolution for smooth stars
            starfield_sphere.SetPhiResolution(120)
            starfield_sphere.SetCenter(0.0, 0.0, 0.0)
            starfield_sphere.Update()
            
            # Get the sphere data
            sphere_data = starfield_sphere.GetOutput()
            
            # Add texture coordinates for star map
            self.add_starfield_texture_coordinates(sphere_data)
            
            # Create mapper and actor
            starfield_mapper = vtk.vtkPolyDataMapper()
            starfield_mapper.SetInputData(sphere_data)
            
            self.starfield_actor = vtk.vtkActor()
            self.starfield_actor.SetMapper(starfield_mapper)
            
            # Try to load a star map texture
            star_texture = self.load_starfield_texture()
            if star_texture:
                self.starfield_actor.SetTexture(star_texture)
                print("Applied star map texture")
            else:
                # Fallback to procedural starfield
                self.apply_procedural_starfield_material()
                print("Applied procedural starfield")
            
            # Set starfield properties
            starfield_property = self.starfield_actor.GetProperty()
            starfield_property.SetRepresentationToSurface()
            starfield_property.SetAmbient(1.0)  # Fully ambient (self-illuminated)
            starfield_property.SetDiffuse(0.0)  # No diffuse lighting
            starfield_property.SetSpecular(0.0)  # No specular highlights
            
            # Render on the inside of the sphere
            starfield_property.BackfaceCullingOff()
            starfield_property.FrontfaceCullingOn()
            
            # Add to renderer with lowest priority (renders first, behind everything)
            self.starfield_actor.SetVisibility(True)
            self.renderer.AddActor(self.starfield_actor)
            
            print(f"Starfield background created successfully (radius: {starfield_radius/1000:.0f}k km)")
            return True
            
        except Exception as e:
            print(f"Error creating starfield background: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_starfield_texture_coordinates(self, sphere_data):
        """Add texture coordinates for starfield mapping"""
        try:
            points = sphere_data.GetPoints()
            num_points = points.GetNumberOfPoints()
            
            # Create texture coordinate array
            tex_coords = vtk.vtkFloatArray()
            tex_coords.SetNumberOfComponents(2)
            tex_coords.SetNumberOfTuples(num_points)
            tex_coords.SetName("StarfieldTextureCoordinates")
            
            for i in range(num_points):
                point = points.GetPoint(i)
                x, y, z = point
                
                # Convert to spherical coordinates
                r = np.sqrt(x*x + y*y + z*z)
                
                if r > 0:
                    # For star maps, we want standard equirectangular projection
                    longitude = np.arctan2(y, x)  # -π to π
                    latitude = np.arcsin(np.clip(z / r, -1.0, 1.0))  # -π/2 to π/2
                    
                    # Map to texture coordinates (0 to 1)
                    u = (longitude + np.pi) / (2 * np.pi)  # 0 to 1
                    v = (latitude + np.pi/2) / np.pi  # 0 to 1
                    
                    # Ensure coordinates are in valid range
                    u = np.clip(u, 0.001, 0.999)
                    v = np.clip(v, 0.001, 0.999)
                else:
                    u, v = 0.5, 0.5
                
                tex_coords.SetTuple2(i, u, v)
            
            sphere_data.GetPointData().SetTCoords(tex_coords)
            print(f"Added starfield texture coordinates for {num_points} points")
            
        except Exception as e:
            print(f"Error adding starfield texture coordinates: {e}")
    
    def load_starfield_texture(self):
        """Load starfield texture from common star map files"""
        try:
            # List of possible starfield texture files
            starfield_files = [
                "starfield.jpg", "starfield.png", "star_map.jpg", "star_map.png",
                "stars.jpg", "stars.png", "milky_way.jpg", "milky_way.png",
                "celestial_sphere.jpg", "celestial_sphere.png", "night_sky.jpg",
                "starmap_4k.jpg", "starmap_8k.jpg", "starmap.jpg",
                "eso_milky_way.jpg", "hubble_starfield.jpg"
            ]
            
            # Try to find and load a starfield texture
            for filename in starfield_files:
                texture = self.try_load_starfield_file(filename)
                if texture:
                    print(f"Successfully loaded starfield texture: {filename}")
                    return texture
            
            print("No starfield texture file found, creating procedural starfield...")
            return self.create_procedural_starfield_texture()
            
        except Exception as e:
            print(f"Error loading starfield texture: {e}")
            return None
    
    def try_load_starfield_file(self, filename):
        """Try to load a specific starfield texture file"""
        try:
            import os
            
            if not os.path.exists(filename):
                return None
            
            # Create appropriate reader
            if filename.lower().endswith(('.jpg', '.jpeg')):
                reader = vtk.vtkJPEGReader()
            elif filename.lower().endswith('.png'):
                reader = vtk.vtkPNGReader()
            else:
                return None
            
            reader.SetFileName(filename)
            reader.Update()
            
            # Check if image loaded
            if reader.GetOutput().GetNumberOfPoints() == 0:
                return None
            
            # Create texture
            texture = vtk.vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
            texture.InterpolateOn()
            texture.RepeatOff()
            texture.EdgeClampOn()
            texture.SetWrap(vtk.vtkTexture.ClampToEdge)
            
            return texture
            
        except Exception as e:
            print(f"Error loading starfield file {filename}: {e}")
            return None
    
    def create_procedural_starfield_texture(self):
        """Create a procedural starfield texture"""
        try:
            print("Creating procedural starfield texture...")
            
            # Create high-resolution starfield
            width, height = 2048, 1024  # Equirectangular format
            
            # Create image data
            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)  # RGB
            
            # Generate realistic starfield
            import random
            random.seed(42)  # Reproducible starfield
            
            # Background space color (very dark blue-black)
            bg_color = [5, 5, 15]
            
            # Fill background
            for y in range(height):
                for x in range(width):
                    image.SetScalarComponentFromFloat(x, y, 0, 0, bg_color[0])
                    image.SetScalarComponentFromFloat(x, y, 0, 1, bg_color[1])
                    image.SetScalarComponentFromFloat(x, y, 0, 2, bg_color[2])
            
            # Add stars with realistic distribution
            num_stars = 8000  # Dense starfield
            
            for _ in range(num_stars):
                # Random position
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                
                # Star magnitude (brightness) - weighted toward dimmer stars
                magnitude = random.random() ** 3  # Cube for realistic distribution
                
                # Star color - most stars are white/blue-white, some yellow/red
                color_type = random.random()
                if color_type < 0.7:  # Blue-white stars (most common)
                    base_color = [200, 210, 255]
                elif color_type < 0.9:  # Yellow stars (like our Sun)
                    base_color = [255, 245, 200]
                else:  # Red stars
                    base_color = [255, 200, 150]
                
                # Apply magnitude
                star_color = [int(c * magnitude) for c in base_color]
                star_color = [max(0, min(255, c)) for c in star_color]
                
                # Set star pixel
                image.SetScalarComponentFromFloat(x, y, 0, 0, star_color[0])
                image.SetScalarComponentFromFloat(x, y, 0, 1, star_color[1])
                image.SetScalarComponentFromFloat(x, y, 0, 2, star_color[2])
                
                # Add some bright stars with glow
                if magnitude > 0.95:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            gx, gy = x + dx, y + dy
                            if 0 <= gx < width and 0 <= gy < height:
                                glow_intensity = 0.3 * magnitude
                                current_r = image.GetScalarComponentAsFloat(gx, gy, 0, 0)
                                current_g = image.GetScalarComponentAsFloat(gx, gy, 0, 1)
                                current_b = image.GetScalarComponentAsFloat(gx, gy, 0, 2)
                                
                                new_r = min(255, current_r + star_color[0] * glow_intensity)
                                new_g = min(255, current_g + star_color[1] * glow_intensity)
                                new_b = min(255, current_b + star_color[2] * glow_intensity)
                                
                                image.SetScalarComponentFromFloat(gx, gy, 0, 0, new_r)
                                image.SetScalarComponentFromFloat(gx, gy, 0, 1, new_g)
                                image.SetScalarComponentFromFloat(gx, gy, 0, 2, new_b)
            
            # Add some nebulosity (faint background glow)
            for _ in range(50):
                center_x = random.randint(0, width - 1)
                center_y = random.randint(0, height - 1)
                nebula_radius = random.randint(20, 100)
                nebula_intensity = random.random() * 0.1
                
                # Nebula color (faint blue/purple)
                nebula_color = [20, 30, 60]
                
                for dx in range(-nebula_radius, nebula_radius):
                    for dy in range(-nebula_radius, nebula_radius):
                        nx, ny = center_x + dx, center_y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < nebula_radius:
                                fade = (1.0 - distance / nebula_radius) * nebula_intensity
                                
                                current_r = image.GetScalarComponentAsFloat(nx, ny, 0, 0)
                                current_g = image.GetScalarComponentAsFloat(nx, ny, 0, 1)
                                current_b = image.GetScalarComponentAsFloat(nx, ny, 0, 2)
                                
                                new_r = min(255, current_r + nebula_color[0] * fade)
                                new_g = min(255, current_g + nebula_color[1] * fade)
                                new_b = min(255, current_b + nebula_color[2] * fade)
                                
                                image.SetScalarComponentFromFloat(nx, ny, 0, 0, new_r)
                                image.SetScalarComponentFromFloat(nx, ny, 0, 1, new_g)
                                image.SetScalarComponentFromFloat(nx, ny, 0, 2, new_b)
            
            # Create texture
            texture = vtk.vtkTexture()
            texture.SetInputData(image)
            texture.InterpolateOn()
            texture.RepeatOff()
            texture.EdgeClampOn()
            
            print("Procedural starfield texture created successfully")
            return texture
            
        except Exception as e:
            print(f"Error creating procedural starfield: {e}")
            return None
    
    def apply_procedural_starfield_material(self):
        """Apply a simple dark material if no texture is available"""
        try:
            # Very dark space color with slight blue tint
            self.starfield_actor.GetProperty().SetColor(0.02, 0.02, 0.08)
            print("Applied simple dark space material")
            
        except Exception as e:
            print(f"Error applying starfield material: {e}")
    
    def download_starfield_instructions(self):
        """Print instructions for downloading high-quality star maps"""
        print("""
To get high-quality star map textures, you can download from:

1. ESA/Gaia Star Map:
   https://www.cosmos.esa.int/web/gaia/data-release-3
   
2. NASA Hubble Legacy Archive:
   https://hla.stsci.edu/
   
3. Free star map textures:
   - https://www.solarsystemscope.com/textures/ (8K star maps)
   - http://planetpixelemporium.com/planets.html (Free celestial textures)
   - https://svs.gsfc.nasa.gov/cgi-bin/search.cgi (NASA visualizations)

4. Hipparcos Star Catalog visualizations:
   - https://www.cosmos.esa.int/web/hipparcos

Save the image as 'starfield.jpg' or 'star_map.jpg' in the same directory.
For best results, use equirectangular projection (2:1 aspect ratio).
Recommended resolution: 4K (4096x2048) or higher.
        """)
        
    def create_earth_representation(self):
        """Create Earth - MODIFIED to not auto-create grid"""
        print("Creating Earth (without automatic grid)...")
        
        try:
            # Remove existing Earth actors if they exist
            self.cleanup_existing_earth_actors()
            
            # Initialize lists (but don't populate them yet)
            self.lat_long_actors = []
            self.grid_labels = []
            
            # 1. Create Earth sphere (solid)
            print("Creating Earth sphere...")
            earth_sphere = vtk.vtkSphereSource()
            earth_sphere.SetRadius(6371.0)  # Earth radius in km
            earth_sphere.SetThetaResolution(120)
            earth_sphere.SetPhiResolution(120)
            earth_sphere.SetCenter(0.0, 0.0, 0.0)
            earth_sphere.Update()
            
            # 2. Add texture coordinates
            sphere_data = earth_sphere.GetOutput()
            self.add_correct_texture_coordinates(sphere_data)
            
            # 3. Create Earth mapper and actor (solid surface)
            earth_mapper = vtk.vtkPolyDataMapper()
            earth_mapper.SetInputData(sphere_data)
            
            self.earth_actor = vtk.vtkActor()
            self.earth_actor.SetMapper(earth_mapper)
            
            # 4. Apply Earth texture
            earth_texture = self.load_earth_texture()
            if earth_texture:
                earth_texture.SetRepeat(0)
                earth_texture.SetInterpolate(1)
                earth_texture.SetWrap(vtk.vtkTexture.ClampToEdge)
                self.earth_actor.SetTexture(earth_texture)
                print("Earth texture applied")
            else:
                self.earth_actor.GetProperty().SetColor(0.2, 0.5, 0.8)
            
            # 5. Set Earth properties
            earth_property = self.earth_actor.GetProperty()
            earth_property.SetRepresentationToSurface()  # Solid surface for texture
            earth_property.SetOpacity(0.8)
            earth_property.SetAmbient(0.4)
            earth_property.SetDiffuse(0.8)
            earth_property.SetSpecular(0.05)
            earth_property.SetSpecularPower(10)
            
            self.earth_actor.SetVisibility(True)
            self.renderer.AddActor(self.earth_actor)
            
            # NOTE: Grid is now created only when checkbox is checked
            print("Earth created successfully (grid will be created when checkbox is checked)")
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
            
            return True
            
        except Exception as e:
            print(f"Error creating Earth: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_earth_opacity_control(self, parent_layout):
        """Create Earth opacity control with lat/long grid, hide orbital paths, and starfield checkboxes"""
        try:
            # Create opacity control widget
            opacity_widget = QWidget()
            opacity_layout = QHBoxLayout(opacity_widget)  # Horizontal layout for single row
            opacity_layout.setContentsMargins(10, 5, 10, 5)
            opacity_layout.setSpacing(10)
            
            # Add Earth opacity label
            opacity_label = QLabel("Earth Opacity:")
            opacity_label.setStyleSheet("color: white; font-weight: bold;")
            opacity_layout.addWidget(opacity_label)
            
            # Add opacity slider
            self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self.earth_opacity_slider.setRange(0, 100)
            self.earth_opacity_slider.setValue(80)  # Default 80%
            self.earth_opacity_slider.setFixedWidth(120)  # Slightly narrower to make room
            self.earth_opacity_slider.valueChanged.connect(self.update_earth_opacity)
            opacity_layout.addWidget(self.earth_opacity_slider)
            
            # Add opacity value label
            self.earth_opacity_value_label = QLabel("80%")
            self.earth_opacity_value_label.setStyleSheet("color: white; font-weight: bold; min-width: 35px;")
            opacity_layout.addWidget(self.earth_opacity_value_label)
            
            # Add some space between opacity and checkboxes
            opacity_layout.addSpacing(15)
            
            # Add checkbox for lat/long grid
            self.show_latlong_grid = QCheckBox("Show Lat/Long Grid")
            self.show_latlong_grid.setChecked(False)  # Default unchecked
            self.show_latlong_grid.setStyleSheet("color: white; font-weight: bold;")
            self.show_latlong_grid.stateChanged.connect(self.toggle_latlong_grid)
            opacity_layout.addWidget(self.show_latlong_grid)
            
            # Add some space between checkboxes
            opacity_layout.addSpacing(10)
            
            # Add checkbox for hiding orbital paths
            self.hide_orbital_paths = QCheckBox("Hide Orbital Paths")
            self.hide_orbital_paths.setChecked(False)  # Default unchecked (paths visible)
            self.hide_orbital_paths.setStyleSheet("color: white; font-weight: bold;")
            self.hide_orbital_paths.stateChanged.connect(self.toggle_orbital_paths)
            opacity_layout.addWidget(self.hide_orbital_paths)
            
            # Add some space between checkboxes
            opacity_layout.addSpacing(10)
            
            # Add checkbox for starfield background (NEW)
            self.show_starfield = QCheckBox("Show Starfield")
            self.show_starfield.setChecked(True)  # Default checked (starfield visible)
            self.show_starfield.setStyleSheet("color: white; font-weight: bold;")
            self.show_starfield.stateChanged.connect(self.toggle_starfield)
            opacity_layout.addWidget(self.show_starfield)
            
            # Add final spacer to push everything left
            opacity_layout.addStretch()
            
            # Style the widget
            opacity_widget.setStyleSheet("""
                QWidget {
                    background-color: rgba(40, 40, 40, 180);
                    border: 1px solid #555;
                    border-radius: 5px;
                    margin: 2px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #666;
                    height: 6px;
                    background: #333;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #888;
                    border: 1px solid #555;
                    width: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #5C6A72;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #333;
                    border: 1px solid #666;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked {
                    background-color: #4CAF50;
                    border: 1px solid #45a049;
                    border-radius: 3px;
                }
            """)
            
            opacity_widget.setFixedHeight(35)  # Single row height
            
            # Add to parent layout
            parent_layout.addWidget(opacity_widget)
            
            print("Earth opacity control with lat/long grid, hide orbital paths, and starfield checkboxes created successfully")
            
        except Exception as e:
            print(f"Error creating Earth opacity control: {e}")
    
    def toggle_starfield(self, state):
        """Toggle starfield background visibility"""
        try:
            show_starfield = state == 2  # Qt.Checked = 2
            
            print(f"Toggling starfield: {'ON' if show_starfield else 'OFF'}")
            
            if hasattr(self, 'starfield_actor') and self.starfield_actor:
                self.starfield_actor.SetVisibility(show_starfield)
                print(f"Starfield visibility: {show_starfield}")
            elif show_starfield:
                # Create starfield if it doesn't exist
                print("Creating starfield on demand...")
                self.create_starfield_background()
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                
            status = "visible" if show_starfield else "hidden"
            print(f"Starfield is now {status}")
                
        except Exception as e:
            print(f"Error toggling starfield: {e}")
            import traceback
            traceback.print_exc()

    def toggle_orbital_paths(self, state):
        """Toggle orbital path visibility"""
        try:
            hide_paths = state == 2  # Qt.Checked = 2
            show_paths = not hide_paths
            
            print(f"Toggling orbital paths: {'HIDDEN' if hide_paths else 'VISIBLE'}")
            
            # Toggle main orbital path
            if hasattr(self, 'path_actor') and self.path_actor:
                self.path_actor.SetVisibility(show_paths)
                print(f"Main orbital path visibility: {show_paths}")
            
            # Toggle satellite trail (keep visible even when hiding main path)
            # The trail shows recent movement which is still useful
            if hasattr(self, 'trail_actor') and self.trail_actor:
                # Trail stays visible - it's different from the full orbital path
                self.trail_actor.SetVisibility(True)
                print("Satellite trail remains visible")
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                
            status = "hidden" if hide_paths else "visible"
            print(f"Orbital paths are now {status}")
                
        except Exception as e:
            print(f"Error toggling orbital paths: {e}")
            import traceback
            traceback.print_exc()


    def toggle_latlong_grid(self, state):
        """Toggle lat/long grid and labels visibility"""
        try:
            show_grid = state == 2  # Qt.Checked = 2
            print(f"Toggling lat/long grid: {'ON' if show_grid else 'OFF'}")
            
            if show_grid:
                # Create or show the grid
                if not hasattr(self, 'lat_long_actors') or not self.lat_long_actors:
                    self.create_latlong_grid_and_labels()
                else:
                    # Show existing grid
                    for actor in self.lat_long_actors:
                        if actor:
                            actor.SetVisibility(True)
                    
                    if hasattr(self, 'equator_actor') and self.equator_actor:
                        self.equator_actor.SetVisibility(True)
                    
                    if hasattr(self, 'prime_meridian_actor') and self.prime_meridian_actor:
                        self.prime_meridian_actor.SetVisibility(True)
                        
                    # Show labels
                    if hasattr(self, 'grid_labels'):
                        for label in self.grid_labels:
                            if label:
                                label.SetVisibility(True)
            else:
                # Hide the grid
                if hasattr(self, 'lat_long_actors'):
                    for actor in self.lat_long_actors:
                        if actor:
                            actor.SetVisibility(False)
                
                if hasattr(self, 'equator_actor') and self.equator_actor:
                    self.equator_actor.SetVisibility(False)
                
                if hasattr(self, 'prime_meridian_actor') and self.prime_meridian_actor:
                    self.prime_meridian_actor.SetVisibility(False)
                    
                # Hide labels
                if hasattr(self, 'grid_labels'):
                    for label in self.grid_labels:
                        if label:
                            label.SetVisibility(False)
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                
        except Exception as e:
            print(f"Error toggling lat/long grid: {e}")
            import traceback
            traceback.print_exc()

    def create_latlong_grid_and_labels(self):
        """Create lat/long grid with coordinate labels"""
        try:
            print("Creating lat/long grid with coordinate labels...")
            
            # Initialize lists
            self.lat_long_actors = []
            self.grid_labels = []
            
            # Get current Earth opacity for proper initial opacity
            earth_opacity = self.earth_opacity_slider.value() / 100.0 if hasattr(self, 'earth_opacity_slider') else 0.8
            
            # Create latitude lines with labels
            print("Creating latitude lines with labels...")
            for lat_deg in range(-75, 90, 15):  # Every 15 degrees
                lat_rad = np.radians(lat_deg)
                radius = 6372.0 * np.cos(lat_rad)
                height = 6372.0 * np.sin(lat_rad)
                
                if radius > 100:  # Skip very small circles near poles
                    # Create latitude circle
                    circle = self.create_circle(radius, height, 'lat')
                    if circle:
                        circle.GetProperty().SetOpacity(earth_opacity * 0.8)
                        self.lat_long_actors.append(circle)
                        self.renderer.AddActor(circle)
                        
                        # Create latitude label
                        label = self.create_coordinate_label(f"{lat_deg}°", radius + 500, 0, height)
                        if label:
                            self.grid_labels.append(label)
                            self.renderer.AddActor2D(label)
            
            # Create longitude lines with labels
            print("Creating longitude lines with labels...")
            for lon_deg in range(0, 360, 30):  # Every 30 degrees for longitude labels (less crowded)
                # Create meridian
                meridian = self.create_meridian(lon_deg)
                if meridian:
                    meridian.GetProperty().SetOpacity(earth_opacity * 0.8)
                    self.lat_long_actors.append(meridian)
                    self.renderer.AddActor(meridian)
                    
                    # Create longitude label at equator
                    lon_rad = np.radians(lon_deg)
                    x = 6872.0 * np.cos(lon_rad)  # 500km above Earth surface
                    y = 6872.0 * np.sin(lon_rad)
                    
                    # Format longitude label
                    if lon_deg == 0:
                        lon_label = "0°"
                    elif lon_deg <= 180:
                        lon_label = f"{lon_deg}°E"
                    else:
                        lon_label = f"{360-lon_deg}°W"
                        
                    label = self.create_coordinate_label(lon_label, x, y, 0)
                    if label:
                        self.grid_labels.append(label)
                        self.renderer.AddActor2D(label)
            
            # Create highlighted equator and prime meridian
            print("Creating equator and prime meridian...")
            self.equator_actor = self.create_circle(6372.5, 0.0, 'equator')
            if self.equator_actor:
                self.equator_actor.GetProperty().SetOpacity(earth_opacity * 0.9)
                self.renderer.AddActor(self.equator_actor)
            
            self.prime_meridian_actor = self.create_meridian(0, highlight=True)
            if self.prime_meridian_actor:
                self.prime_meridian_actor.GetProperty().SetOpacity(earth_opacity * 0.9)
                self.renderer.AddActor(self.prime_meridian_actor)
            
            print(f"Created {len(self.lat_long_actors)} grid actors and {len(self.grid_labels)} labels")
            
        except Exception as e:
            print(f"Error creating lat/long grid and labels: {e}")
            import traceback
            traceback.print_exc()

    def create_coordinate_label(self, text, x, y, z):
        """Create a 3D coordinate label"""
        try:
            # Create text source
            text_source = vtk.vtkVectorText()
            text_source.SetText(text)
            text_source.Update()
            
            # Create follower (billboard text that always faces camera)
            follower = vtk.vtkFollower()
            
            # Create mapper
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())
            follower.SetMapper(text_mapper)
            
            # Position the label
            follower.SetPosition(x, y, z)
            follower.SetScale(300, 300, 300)  # Scale to make visible
            
            # Set camera for billboard effect
            if hasattr(self, 'renderer'):
                camera = self.renderer.GetActiveCamera()
                follower.SetCamera(camera)
            
            # Style the label
            follower.GetProperty().SetColor(1.0, 1.0, 0.8)  # Light yellow
            follower.GetProperty().SetOpacity(0.8)
            
            return follower
            
        except Exception as e:
            print(f"Error creating coordinate label: {e}")
            return None

    def update_earth_opacity(self, value):
        """Update Earth opacity - ALSO UPDATE GRID AND LABELS IF VISIBLE"""
        try:
            opacity = value / 100.0
            self.earth_opacity_value_label.setText(f"{value}%")
            
            print(f"Updating Earth opacity to {value}% (opacity={opacity})")
            
            # Update main Earth actor
            if hasattr(self, 'earth_actor') and self.earth_actor:
                self.earth_actor.GetProperty().SetOpacity(opacity)
            
            # Update grid lines if they're visible
            if (hasattr(self, 'show_latlong_grid') and self.show_latlong_grid.isChecked() and
                hasattr(self, 'lat_long_actors') and self.lat_long_actors):
                
                grid_opacity = opacity * 0.8  # Grid is 80% of Earth opacity
                
                for actor in self.lat_long_actors:
                    if actor and actor.GetVisibility():
                        actor.GetProperty().SetOpacity(grid_opacity)
                
                # Update special lines
                if hasattr(self, 'equator_actor') and self.equator_actor and self.equator_actor.GetVisibility():
                    self.equator_actor.GetProperty().SetOpacity(opacity * 0.9)
                    
                if hasattr(self, 'prime_meridian_actor') and self.prime_meridian_actor and self.prime_meridian_actor.GetVisibility():
                    self.prime_meridian_actor.GetProperty().SetOpacity(opacity * 0.9)
                
                # Update labels
                if hasattr(self, 'grid_labels'):
                    label_opacity = opacity * 0.8
                    for label in self.grid_labels:
                        if label and label.GetVisibility():
                            label.GetProperty().SetOpacity(label_opacity)
            
            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                    
        except Exception as e:
            print(f"Error updating Earth opacity: {e}")
            import traceback
            traceback.print_exc()
        
    def create_meridian(self, longitude_deg, highlight=False):
        """Create a meridian with proper default opacity"""
        try:
            lon_rad = np.radians(longitude_deg)
            
            # Create points from south pole to north pole
            n_points = 120  # Higher resolution for smoother lines
            latitudes = np.linspace(-np.pi/2, np.pi/2, n_points)
            
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            for i, lat in enumerate(latitudes):
                x = 6372.0 * np.cos(lat) * np.cos(lon_rad)  # 1km above Earth
                y = 6372.0 * np.cos(lat) * np.sin(lon_rad)
                z = 6372.0 * np.sin(lat)
                points.InsertNextPoint(x, y, z)
                
                if i > 0:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i-1)
                    line.GetPointIds().SetId(1, i)
                    lines.InsertNextCell(line)
            
            # Create polydata
            meridian_polydata = vtk.vtkPolyData()
            meridian_polydata.SetPoints(points)
            meridian_polydata.SetLines(lines)
            
            # Create mapper
            meridian_mapper = vtk.vtkPolyDataMapper()
            meridian_mapper.SetInputData(meridian_polydata)
            
            # Create actor
            meridian_actor = vtk.vtkActor()
            meridian_actor.SetMapper(meridian_mapper)
            
            # Set properties based on whether it's highlighted
            if highlight:  # Prime meridian or special meridian
                meridian_actor.GetProperty().SetColor(1.0, 0.6, 0.0)  # Orange
                meridian_actor.GetProperty().SetLineWidth(2.5)
                meridian_actor.GetProperty().SetOpacity(0.72)  # 90% of default 80% Earth
            else:  # Regular meridian
                meridian_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
                meridian_actor.GetProperty().SetLineWidth(1.5)
                meridian_actor.GetProperty().SetOpacity(0.64)  # 80% of default 80% Earth
            
            return meridian_actor
            
        except Exception as e:
            print(f"Error creating meridian at {longitude_deg}°: {e}")
            return None

    def add_correct_texture_coordinates(self, sphere_data):
        """Add texture coordinates with Pacific-specific seam fix"""
        try:
            print("Computing texture coordinates with Pacific-specific seam fix...")
            
            points = sphere_data.GetPoints()
            num_points = points.GetNumberOfPoints()
            
            # Create texture coordinate array
            tex_coords = vtk.vtkFloatArray()
            tex_coords.SetNumberOfComponents(2)
            tex_coords.SetNumberOfTuples(num_points)
            tex_coords.SetName("TextureCoordinates")
            
            for i in range(num_points):
                # Get 3D point
                point = points.GetPoint(i)
                x, y, z = point
                
                # Convert to spherical coordinates
                r = np.sqrt(x*x + y*y + z*z)
                
                # PACIFIC SEAM FIX: Use a different longitude calculation
                # Instead of atan2(y, x), shift the coordinate system
                longitude = np.arctan2(y, x)  # -π to π
                
                # Shift longitude to avoid the Pacific seam
                # Move the "seam" from the Pacific to somewhere over Africa/Atlantic
                longitude_shifted = longitude + np.pi  # 0 to 2π, seam now at Greenwich
                if longitude_shifted >= 2 * np.pi:
                    longitude_shifted -= 2 * np.pi
                
                # Map to texture coordinates
                u = longitude_shifted / (2 * np.pi)  # 0 to 1
                
                # Ensure we don't hit exact 0 or 1
                epsilon = 0.0005  # Small safety margin
                u = np.clip(u, epsilon, 1.0 - epsilon)
                
                # Calculate latitude (v coordinate) - standard calculation
                if r > 0:
                    latitude = np.arcsin(np.clip(z / r, -1.0, 1.0))  # -π/2 to π/2
                else:
                    latitude = 0
                
                # Map latitude to texture coordinates (0 to 1)
                v = (latitude + np.pi/2) / np.pi  # 0 to 1
                v = np.clip(v, epsilon, 1.0 - epsilon)
                
                # Set texture coordinates
                tex_coords.SetTuple2(i, u, v)
            
            # Add texture coordinates to the sphere data
            sphere_data.GetPointData().SetTCoords(tex_coords)
            print(f"Added Pacific-specific seam-fixed texture coordinates for {num_points} points")
            print("Longitude seam moved from Pacific to Greenwich meridian")
            
        except Exception as e:
            print(f"Error computing texture coordinates: {e}")
            import traceback
            traceback.print_exc()

    def load_earth_texture(self):
        """Load Earth texture - REVERTED to original working version"""
        try:
            # List of possible texture files
            texture_files = [
                "earth_texture.jpg", "earth_map.jpg", "world_map.jpg",
                "earth_texture.png", "earth_map.png", "world_map.png",
                "blue_marble.jpg", "blue_marble.png",
                "natural_earth.jpg", "natural_earth.png",
                "earth.jpg", "earth.png", "world.jpg", "world.png"
            ]
            
            # Try to find and load texture file
            for filename in texture_files:
                texture = self.try_load_texture_file(filename)
                if texture:
                    print(f"Successfully loaded texture: {filename}")
                    return texture
            
            # If no local file found, create procedural texture
            print("No texture file found, creating procedural Earth texture...")
            return self.create_procedural_earth_texture()
            
        except Exception as e:
            print(f"Error loading Earth texture: {e}")
            return None

    def try_load_texture_file(self, filename):
        """Try to load a specific texture file - REVERTED to original working version"""
        try:
            import os
            
            if not os.path.exists(filename):
                return None
            
            # Create appropriate reader
            if filename.lower().endswith(('.jpg', '.jpeg')):
                reader = vtk.vtkJPEGReader()
            elif filename.lower().endswith('.png'):
                reader = vtk.vtkPNGReader()
            else:
                return None
            
            reader.SetFileName(filename)
            reader.Update()
            
            # Check if image loaded
            if reader.GetOutput().GetNumberOfPoints() == 0:
                return None
            
            # Create texture with ORIGINAL settings that were working
            texture = vtk.vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
            texture.InterpolateOn()  # Smooth interpolation
            texture.RepeatOff()      # Don't repeat - this is crucial
            texture.EdgeClampOn()    # Clamp edges
            
            # ONLY the essential seam prevention setting
            texture.SetWrap(vtk.vtkTexture.ClampToEdge)
            
            print(f"Loaded texture {filename} with basic seam prevention")
            return texture
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def create_procedural_earth_texture(self):
        """Create a simple procedural Earth texture - CLEAN VERSION"""
        try:
            print("Creating procedural Earth texture...")
            
            # Standard equirectangular dimensions
            width, height = 1024, 512
            
            # Create image data
            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)  # RGB
            
            # Create a simple but recognizable Earth pattern
            for y in range(height):
                for x in range(width):
                    # Convert pixel to lat/lon
                    lon = (x / width) * 360 - 180  # -180 to 180
                    lat = 90 - (y / height) * 180   # 90 to -90
                    
                    # Simple continent approximation
                    is_land = self.simple_land_check(lon, lat)
                    
                    if is_land:
                        # Land colors (green/brown with variation)
                        variation = (x + y) % 50 / 50.0
                        r = int(80 + variation * 60)   # Brown-green
                        g = int(100 + variation * 80)  # Green
                        b = int(40 + variation * 40)   # Minimal blue
                    else:
                        # Ocean colors (blue with variation)
                        variation = (x * 3 + y * 2) % 40 / 40.0
                        r = int(20 + variation * 30)   # Minimal red
                        g = int(60 + variation * 60)   # Medium green
                        b = int(120 + variation * 80)  # Strong blue
                    
                    # Clamp values
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))
                    
                    # Set pixel
                    image.SetScalarComponentFromFloat(x, y, 0, 0, r)
                    image.SetScalarComponentFromFloat(x, y, 0, 1, g)
                    image.SetScalarComponentFromFloat(x, y, 0, 2, b)
            
            # Create texture with BASIC settings only
            texture = vtk.vtkTexture()
            texture.SetInputData(image)
            texture.InterpolateOn()
            texture.RepeatOff()
            texture.EdgeClampOn()
            
            print("Procedural Earth texture created")
            return texture
            
        except Exception as e:
            print(f"Error creating procedural texture: {e}")
            return None

    def simple_land_check(self, lon, lat):
        """Simple land/ocean check for procedural texture"""
        # Very simplified continent shapes
        # North America
        if (-140 < lon < -60 and 15 < lat < 70):
            return True
        # South America  
        if (-80 < lon < -35 and -55 < lat < 15):
            return True
        # Africa
        if (-20 < lon < 50 and -35 < lat < 35):
            return True
        # Europe
        if (-10 < lon < 40 and 35 < lat < 70):
            return True
        # Asia
        if (40 < lon < 180 and 10 < lat < 75):
            return True
        # Australia
        if (110 < lon < 155 and -45 < lat < -10):
            return True
        
        return False

    # def add_earth_opacity_control(self):
    #     """Add Earth opacity control below the visualization"""
    #     try:
    #         # Find the main layout that contains the VTK widget
    #         vtk_frame = self.vtk_widget.parent()
    #         vtk_layout = vtk_frame.layout()
            
    #         # Create opacity control widget
    #         opacity_control = QWidget()
    #         opacity_layout = QHBoxLayout(opacity_control)
    #         opacity_layout.setContentsMargins(10, 5, 10, 5)
            
    #         # Add label
    #         opacity_label = QLabel("Earth Opacity:")
    #         opacity_layout.addWidget(opacity_label)
            
    #         # Add slider
    #         self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
    #         self.earth_opacity_slider.setRange(0, 100)
    #         self.earth_opacity_slider.setValue(80)  # Default 80%
    #         self.earth_opacity_slider.setMaximumWidth(200)
    #         self.earth_opacity_slider.valueChanged.connect(self.update_earth_opacity)
    #         opacity_layout.addWidget(self.earth_opacity_slider)
            
    #         # Add value label
    #         self.earth_opacity_label = QLabel("80%")
    #         self.earth_opacity_label.setMinimumWidth(40)
    #         opacity_layout.addWidget(self.earth_opacity_label)
            
    #         # Add some space
    #         opacity_layout.addStretch()
            
    #         # Style the control
    #         opacity_control.setStyleSheet("""
    #             QWidget {
    #                 background-color: rgba(50, 50, 50, 200);
    #                 border: 1px solid #666;
    #                 border-radius: 5px;
    #             }
    #             QLabel {
    #                 color: white;
    #                 font-weight: bold;
    #             }
    #             QSlider::groove:horizontal {
    #                 border: 1px solid #999;
    #                 height: 8px;
    #                 background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B0B0B0, stop:1 #A0A0A0);
    #                 margin: 2px 0;
    #                 border-radius: 4px;
    #             }
    #             QSlider::handle:horizontal {
    #                 background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
    #                 border: 1px solid #5c5c5c;
    #                 width: 18px;
    #                 margin: -2px 0;
    #                 border-radius: 3px;
    #             }
    #         """)
            
    #         # Add to the VTK layout at the bottom
    #         vtk_layout.addWidget(opacity_control)
            
    #         print("Earth opacity control added below visualization")
            
    #     except Exception as e:
    #         print(f"Error adding opacity control: {e}")
    #         # Fallback: add to main control panel if VTK layout fails
    #         try:
    #             self.add_opacity_control_to_sidebar()
    #         except:
    #             print("Could not add opacity control anywhere")

    # def add_opacity_control_to_sidebar(self):
    #     """Fallback: add opacity control to sidebar"""
    #     # Find analysis parameters group
    #     for i in range(self.control_layout.count()):
    #         item = self.control_layout.itemAt(i)
    #         if item and isinstance(item.widget(), QGroupBox) and item.widget().title() == "Analysis Parameters":
    #             analysis_group = item.widget()
    #             analysis_layout = analysis_group.layout()
                
    #             # Add Earth opacity control
    #             self.earth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
    #             self.earth_opacity_slider.setRange(0, 100)
    #             self.earth_opacity_slider.setValue(80)
    #             self.earth_opacity_slider.valueChanged.connect(self.update_earth_opacity)
                
    #             self.earth_opacity_label = QLabel("80%")
    #             self.earth_opacity_label.setMinimumWidth(50)
                
    #             analysis_layout.addRow("Earth Opacity:", self.earth_opacity_slider)
    #             analysis_layout.addRow("", self.earth_opacity_label)
    #            break

    # def update_earth_opacity(self, value):
    #     """Update Earth opacity from slider"""
    #     try:
    #         opacity = value / 100.0
    #         self.earth_opacity_label.setText(f"{value}%")
            
    #         if hasattr(self, 'earth_actor') and self.earth_actor:
    #             self.earth_actor.GetProperty().SetOpacity(opacity)
                
    #             # Force render
    #             if hasattr(self, 'vtk_widget'):
    #                 self.vtk_widget.GetRenderWindow().Render()
                    
    #             print(f"Earth opacity updated to {value}%")
                
    #     except Exception as e:
    #         print(f"Error updating Earth opacity: {e}")
    
    def cleanup_existing_earth_actors(self):
        """Clean up all existing Earth-related actors - INCLUDING LABELS"""
        print("Cleaning up existing Earth actors...")
        
        actors_to_cleanup = [
            'earth_actor', 'equator_actor', 'prime_meridian_actor'
        ]
        
        for actor_name in actors_to_cleanup:
            if hasattr(self, actor_name):
                actor = getattr(self, actor_name)
                if actor:
                    self.renderer.RemoveActor(actor)
                    print(f"Removed {actor_name}")
                setattr(self, actor_name, None)
        
        # Clean up grid actor lists
        if hasattr(self, 'lat_long_actors') and self.lat_long_actors:
            for i, actor in enumerate(self.lat_long_actors):
                if actor:
                    self.renderer.RemoveActor(actor)
            print(f"Removed {len(self.lat_long_actors)} lat/long grid actors")
            self.lat_long_actors = []
            
        # Clean up label lists
        if hasattr(self, 'grid_labels') and self.grid_labels:
            for i, label in enumerate(self.grid_labels):
                if label:
                    self.renderer.RemoveActor2D(label)
            print(f"Removed {len(self.grid_labels)} grid labels")
            self.grid_labels = []

    def download_earth_texture(self):
        """Download a free Earth texture (helper function)"""
        print("""
To get a high-quality Earth texture, you can download one from:

1. NASA Blue Marble: 
   https://visibleearth.nasa.gov/images/57752/blue-marble-land-surface-shallow-water-and-shaded-topography
   
2. Natural Earth: 
   https://www.naturalearthdata.com/downloads/10m-raster-data/
   
3. Free texture sites:
   - https://www.solarsystemscope.com/textures/ (Planet textures)
   - https://planetpixelemporium.com/planets.html (Free planet textures)

Save the image as 'earth_texture.jpg' in the same directory as your script.
For best results, use an equirectangular projection (2:1 aspect ratio).
        """)

    def cleanup_existing_earth_actors(self):
        """Clean up all existing Earth-related actors"""
        actors_to_cleanup = [
            'earth_actor', 'ocean_actor', 'earth_wireframe_actor', 
            'equator_actor'
        ]
        
        for actor_name in actors_to_cleanup:
            if hasattr(self, actor_name):
                actor = getattr(self, actor_name)
                if actor:
                    self.renderer.RemoveActor(actor)
                    setattr(self, actor_name, None)
        
        # Clean up actor lists
        if hasattr(self, 'lat_long_actors') and self.lat_long_actors:
            for actor in self.lat_long_actors:
                if actor:
                    self.renderer.RemoveActor(actor)
            self.lat_long_actors = []
        
    def clear_field_visualization(self):
        """Clear field visualization while preserving Earth texture and lat/long grid"""
        print("=== CLEARING FIELD VISUALIZATION (PRESERVING EARTH) ===")
        
        actors_removed = 0
        
        # Store Earth-related actors that should be preserved
        earth_actors_to_preserve = {
            'earth_actor': getattr(self, 'earth_actor', None),
            'equator_actor': getattr(self, 'equator_actor', None),
            'lat_long_actors': getattr(self, 'lat_long_actors', [])
        }
        
        # Flatten the list of actors to preserve for easy checking
        preserve_list = []
        if earth_actors_to_preserve['earth_actor']:
            preserve_list.append(earth_actors_to_preserve['earth_actor'])
        if earth_actors_to_preserve['equator_actor']:
            preserve_list.append(earth_actors_to_preserve['equator_actor'])
        preserve_list.extend(earth_actors_to_preserve['lat_long_actors'])
        
        # Remove field visualization actors (but preserve Earth)
        if hasattr(self, 'field_actor') and self.field_actor:
            if self.field_actor not in preserve_list:
                self.renderer.RemoveActor(self.field_actor)
                self.field_actor = None
                actors_removed += 1
                print("Removed field_actor")
        
        # Remove volume actor
        if hasattr(self, 'volume_actor') and self.volume_actor:
            self.renderer.RemoveVolume(self.volume_actor)
            self.volume_actor = None
            actors_removed += 1
            print("Removed volume_actor")
        
        # Remove other visualization actors while preserving Earth
        visualization_actor_lists = ['wireframe_actors', 'slice_actors', 'isosurface_actors']
        
        for actor_list_name in visualization_actor_lists:
            if hasattr(self, actor_list_name):
                actor_list = getattr(self, actor_list_name)
                for actor in actor_list:
                    if actor and actor not in preserve_list:
                        self.renderer.RemoveActor(actor)
                        actors_removed += 1
                setattr(self, actor_list_name, [])
                if actor_list:
                    print(f"Removed {actor_list_name}")
        
        # Remove scalar bar
        if hasattr(self, 'scalar_bar') and self.scalar_bar:
            self.renderer.RemoveViewProp(self.scalar_bar)
            self.scalar_bar = None
            print("Removed scalar_bar")
        
        # Verify Earth actors are still in renderer (re-add if missing)
        self.verify_earth_actors_in_renderer(earth_actors_to_preserve)
        
        print(f"Total field actors removed: {actors_removed} (Earth preserved)")
        print("=== CLEAR COMPLETE ===")
    
    def verify_earth_actors_in_renderer(self, earth_actors_to_preserve):
        """Ensure all Earth actors are still in the renderer"""
        # Get current actors in renderer
        current_actors = []
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        while True:
            actor = actors.GetNextActor()
            if not actor:
                break
            current_actors.append(actor)
        
        # Check and re-add Earth actor if missing
        if earth_actors_to_preserve['earth_actor']:
            if earth_actors_to_preserve['earth_actor'] not in current_actors:
                self.renderer.AddActor(earth_actors_to_preserve['earth_actor'])
                print("Re-added Earth texture actor")
        
        # Check and re-add equator actor if missing
        if earth_actors_to_preserve['equator_actor']:
            if earth_actors_to_preserve['equator_actor'] not in current_actors:
                self.renderer.AddActor(earth_actors_to_preserve['equator_actor'])
                print("Re-added equator actor")
        
        # Check and re-add lat/long grid actors if missing
        readded_grid_count = 0
        for grid_actor in earth_actors_to_preserve['lat_long_actors']:
            if grid_actor and grid_actor not in current_actors:
                self.renderer.AddActor(grid_actor)
                readded_grid_count += 1
        
        if readded_grid_count > 0:
            print(f"Re-added {readded_grid_count} lat/long grid actors")

    def _get_multi_iso_levels(self):
        levels = []
        if hasattr(self, 'multi_iso_controls'):
            for cb, slider, _ in self.multi_iso_controls:
                if cb.isChecked():
                    levels.append(slider.value())
        return levels

    def update_multi_isosurface_controls(self, *args):
        # Only rebuild when in Wireframe & Multiple Isosurface
        if (hasattr(self, 'viz_mode_combo') and self.viz_mode_combo.currentText() == "Wireframe" and
            hasattr(self, 'wireframe_style_combo') and self.wireframe_style_combo.currentText() == "Multiple Isosurface" and
            hasattr(self, 'vtk_data') and self.vtk_data):
            self.clear_field_visualization()
            self.setup_wireframe_rendering()
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
            
    def setup_visualization_controls(self):
        """Enhanced controls with volume threshold control integrated below mode selector"""

        # Find the control layout (existing code)
        control_layout = None
        for i in range(self.control_layout.count()):
            item = self.control_layout.itemAt(i)
            if item and isinstance(item.widget(), QGroupBox) and item.widget().title() == "Analysis Parameters":
                control_layout = self.control_layout
                insert_index = i + 1
                break

        if not control_layout:
            control_layout = self.control_layout
            insert_index = -1

        # Create visualization controls group
        viz_group = QGroupBox("Field Visualization")
        viz_layout = QVBoxLayout(viz_group)

        # Visualization mode selection
        mode_layout = QFormLayout()
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems([
            "Volume Rendering",
            "Isosurfaces", 
            "Point Cloud",
            "Wireframe",
            "Surface with Edges",
            "Slice Planes"
        ])
        self.viz_mode_combo.setCurrentText("Point Cloud")
        self.viz_mode_combo.currentTextChanged.connect(self.change_visualization_mode)
        mode_layout.addRow("Visualization Mode:", self.viz_mode_combo)

        viz_layout.addLayout(mode_layout)

        # Volume Rendering Controls with LONGER slider
        self.volume_controls = QWidget()
        volume_layout = QVBoxLayout(self.volume_controls)  # Changed to VBoxLayout for stacking

        # Volume Flux Threshold Control with label below for longer slider

        # Title for the control
        threshold_title = QLabel("Flux Threshold:")
        threshold_title.setStyleSheet("font-weight: bold;")
        volume_layout.addWidget(threshold_title)

        # Slider gets full width (much longer)
        self.volume_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_threshold_slider.setRange(0, 100)
        self.volume_threshold_slider.setValue(50)  # Start at 50% = exactly 90th percentile
        self.volume_threshold_slider.valueChanged.connect(self.update_volume_threshold)
        self.volume_threshold_slider.setMinimumWidth(300)  # Ensure it's nice and long
        volume_layout.addWidget(self.volume_threshold_slider)

        # Label below the slider (centered)
        self.volume_threshold_label = QLabel("90th percentile")
        self.volume_threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the text
        self.volume_threshold_label.setStyleSheet("color: #5C6A72; font-size: 11px; margin-top: 2px;")
        volume_layout.addWidget(self.volume_threshold_label)

        # Add explanation text (more compact since we have more space)
        explanation_label = QLabel(
            "Dual-scale: 0-50% = 1st-90th percentile | 50-100% = 90th-99.9th percentile"
        )
        explanation_label.setStyleSheet("color: #5C6A72; font-size: 9px; margin-top: 5px;")
        explanation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation_label.setWordWrap(True)
        volume_layout.addWidget(explanation_label)

        # Add some spacing
        volume_layout.addSpacing(5)

        # Initially hide volume controls
        self.volume_controls.setVisible(False)
        viz_layout.addWidget(self.volume_controls)

        # Point Cloud Controls (existing pattern - keep horizontal layout for comparison)
        self.point_cloud_controls = QWidget()
        pc_layout = QFormLayout(self.point_cloud_controls)

        # Point Density Control (keep existing horizontal layout)
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Point Density:"))
        self.point_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_density_slider.setRange(500, 10000)
        self.point_density_slider.setValue(5000)
        self.point_density_slider.valueChanged.connect(self.update_point_density)
        self.point_density_label = QLabel("5,000 points")
        self.point_density_label.setMinimumWidth(100)
        self.point_density_label.setStyleSheet("color: #5C6A72;")
        density_row.addWidget(self.point_density_slider)
        density_row.addWidget(self.point_density_label)
        pc_layout.addRow(density_row)

        # Point Size Control (keep existing horizontal layout)
        point_size_row = QHBoxLayout()
        point_size_row.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(100, 800)
        self.point_size_slider.setValue(400)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        self.point_size_label = QLabel("400m radius")
        self.point_size_label.setMinimumWidth(100)
        self.point_size_label.setStyleSheet("color: #5C6A72;")
        point_size_row.addWidget(self.point_size_slider)
        point_size_row.addWidget(self.point_size_label)
        pc_layout.addRow(point_size_row)

        # Show point cloud controls initially (since Point Cloud is default)
        self.point_cloud_controls.setVisible(True)
        viz_layout.addWidget(self.point_cloud_controls)

        # Wireframe Style Controls (enhanced with color mapping)
        self.wireframe_controls = QWidget()
        wireframe_layout = QFormLayout(self.wireframe_controls)

        # Wireframe style selector (keep existing)
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Isosurface Style:"))
        self.wireframe_style_combo = QComboBox()
        self.wireframe_style_combo.addItems([
            "Single Isosurface",
            "Multiple Isosurface", 
            "Boundary Box"
        ])
        self.wireframe_style_combo.currentTextChanged.connect(self.change_wireframe_style)
        style_row.addWidget(self.wireframe_style_combo)
        wireframe_layout.addRow(style_row)

        # Multiple Isosurface Controls (replaces Color Mode)
        multi_row = QVBoxLayout()
        self.multi_iso_controls = []
        default_percents = [20, 40, 60, 80]
        for idx, pct in enumerate(default_percents, start=1):
            row = QHBoxLayout()
            cb = QCheckBox(f"Isosurface {idx}")
            cb.setChecked(True)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(10, 90)
            slider.setValue(pct)
            value_label = QLabel(f"{pct}th %")
            value_label.setMinimumWidth(60)
            value_label.setStyleSheet("color: #5C6A72;")
            # Store and wire signals
            self.multi_iso_controls.append((cb, slider, value_label))
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}th %"))
            cb.stateChanged.connect(self.update_multi_isosurface_controls)
            slider.valueChanged.connect(self.update_multi_isosurface_controls)
            row.addWidget(cb)
            row.addWidget(slider)
            row.addWidget(value_label)
            multi_row.addLayout(row)
        wireframe_layout.addRow(multi_row)
            
        # Isosurface Level Slider (updated to match volume controls style)
        self.isosurface_controls = QWidget()
        iso_layout = QVBoxLayout(self.isosurface_controls)  # Changed from QFormLayout to QVBoxLayout

        # Title for the control
        iso_title = QLabel("Contour Level:")
        iso_title.setStyleSheet("font-weight: bold;")
        iso_layout.addWidget(iso_title)

        # Slider gets full width (longer like volume threshold)
        self.isosurface_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.isosurface_level_slider.setRange(10, 90)
        self.isosurface_level_slider.setValue(50)
        self.isosurface_level_slider.valueChanged.connect(self.update_isosurface_level)
        self.isosurface_level_slider.setMinimumWidth(300)  # Same width as volume threshold
        self.isosurface_level_slider.setToolTip("Adjust the contour level for isosurface wireframe")  # Added tooltip
        iso_layout.addWidget(self.isosurface_level_slider)

        # Label below the slider (centered like volume threshold) - CLARIFIED PERCENTILE
        self.isosurface_level_label = QLabel("50th flux percentile")
        self.isosurface_level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the text
        self.isosurface_level_label.setStyleSheet("color: #5C6A72; font-size: 11px; margin-top: 2px;")
        iso_layout.addWidget(self.isosurface_level_label)

        # Add some spacing
        iso_layout.addSpacing(5)

        wireframe_layout.addRow(self.isosurface_controls)
        self.isosurface_controls.setVisible(False)  # Initially hidden

        # Initially hide wireframe controls
        self.wireframe_controls.setVisible(False)
        viz_layout.addWidget(self.wireframe_controls)

        # Slice Plane Controls (existing pattern)
        self.slice_controls = QWidget()
        slice_layout = QFormLayout(self.slice_controls)

        # Slice axis selector (keep existing)
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Slice Orientation:"))
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems([
            "X-Axis (YZ Plane)", 
            "Y-Axis (XZ Plane)", 
            "Z-Axis (XY Plane)"
        ])
        self.slice_axis_combo.setCurrentText("Z-Axis (XY Plane)")
        self.slice_axis_combo.currentTextChanged.connect(self.change_slice_axis)
        axis_row.addWidget(self.slice_axis_combo)
        slice_layout.addRow(axis_row)

        # Slice Position Slider (keep existing horizontal layout)
        slice_pos_row = QHBoxLayout()
        slice_pos_row.addWidget(QLabel("Position:"))
        self.slice_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_position_slider.setRange(0, 100)
        self.slice_position_slider.setValue(50)
        self.slice_position_slider.valueChanged.connect(self.update_slice_position)
        self.slice_position_label = QLabel("Center (0 km)")
        self.slice_position_label.setMinimumWidth(120)
        self.slice_position_label.setStyleSheet("color: #5C6A72;")
        slice_pos_row.addWidget(self.slice_position_slider)
        slice_pos_row.addWidget(self.slice_position_label)
        slice_layout.addRow(slice_pos_row)

        # Initially hide slice controls
        self.slice_controls.setVisible(False)
        viz_layout.addWidget(self.slice_controls)

        # === PERMANENT CONTROLS FOR ALL MODES ===
        permanent_layout = QFormLayout()

        # Transparency control (permanent) - keep horizontal layout
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Transparency:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.opacity_label = QLabel("70%")
        self.opacity_label.setMinimumWidth(50)
        self.opacity_label.setStyleSheet("color: #5C6A72;")
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)
        permanent_layout.addRow(opacity_row)

        # Min Flux Cutoff (permanent) - keep horizontal layout
        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(QLabel("Min Flux Cutoff:"))

        self.flux_cutoff_edit = QLineEdit()
        self.flux_cutoff_edit.setText("1e-8")
        self.flux_cutoff_edit.setMaximumWidth(100)
        self.flux_cutoff_edit.editingFinished.connect(self.update_flux_cutoff_from_text)
        cutoff_row.addWidget(self.flux_cutoff_edit)

        units_label = QLabel("particles/cm²/s")
        units_label.setStyleSheet("color: #5C6A72; font-size: 10px;")
        cutoff_row.addWidget(units_label)
        cutoff_row.addStretch()
        permanent_layout.addRow(cutoff_row)

        self.current_flux_cutoff = 1e-8

        # Color Scale toggle (permanent) - keep horizontal layout
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Color Scale:"))
        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["Linear", "Logarithmic"])
        self.scale_mode_combo.setCurrentText("Linear")
        self.scale_mode_combo.currentTextChanged.connect(self.change_scale_mode)
        scale_row.addWidget(self.scale_mode_combo)
        scale_row.addStretch()
        permanent_layout.addRow(scale_row)

        # Color map selection (permanent) - keep horizontal layout
        colormap_row = QHBoxLayout()
        colormap_row.addWidget(QLabel("Color Map:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Blue to Red",
            "Viridis", 
            "Plasma",
            "Cool to Warm",
            "Rainbow",
            "Grayscale"
        ])
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        colormap_row.addWidget(self.colormap_combo)
        colormap_row.addStretch()
        permanent_layout.addRow(colormap_row)

        viz_layout.addLayout(permanent_layout)

        # Add debug button
        debug_button = QPushButton("Debug Scalar Ranges")
        debug_button.clicked.connect(self.debug_scalar_ranges)
        viz_layout.addWidget(debug_button)

        # Insert into main control layout
        if insert_index >= 0:
            control_layout.insertWidget(insert_index, viz_group)
        else:
            control_layout.addWidget(viz_group)

    def change_wireframe_color_mode(self, color_mode):
        """Change wireframe color mode between solid color and flux-based coloring"""
        print(f"Changing wireframe color mode to: {color_mode}")
        
        # Only update if we're in wireframe mode
        if self.viz_mode_combo.currentText() == "Wireframe" and self.vtk_data:
            # Clear and recreate wireframe with new color mode
            self.clear_field_visualization()
            self.setup_wireframe_rendering()
            self.vtk_widget.GetRenderWindow().Render()
            
    # def test_dual_scale_mapping(self):
    #     """Test method to verify dual-scale mapping is correct"""
    #     print("=== TESTING DUAL-SCALE MAPPING ===")

    #     test_values = [0, 25, 50, 75, 100]

    #     for value in test_values:
    #         if value <= 50:
    #             percentile = 1 + (value / 50.0) * 89
    #             expected_desc = f"broad range: 1st-90th"
    #         else:
    #             normalized = (value - 50) / 50.0
    #             percentile = 90 + normalized * 9.9
    #             expected_desc = f"high-res range: 90th-99.9th"

    #         print(f"Slider {value}% -> {percentile:.1f}th percentile ({expected_desc})")

    #     print("Expected behavior:")
    #     print("  0% -> 1st percentile")
    #     print(" 25% -> 45th percentile") 
    #     print(" 50% -> 90th percentile (TRANSITION POINT)")
    #     print(" 75% -> 95th percentile")
    #     print("100% -> 99.9th percentile")
    #     print("=== TEST COMPLETE ===")
            
    def update_flux_cutoff_from_text(self):
        """Update flux cutoff - with better default"""
        try:
            text = self.flux_cutoff_edit.text().strip()
            
            # Handle scientific notation
            if 'e' in text.lower():
                value = float(text)
            else:
                value = float(text)
            
            # Clamp to reasonable range
            value = max(1e-15, min(1e15, value))
            
            self.current_flux_cutoff = value
            print(f"Updated flux cutoff to: {value:.2e} particles/cm²/s")
            
            # Update the display to show clean scientific notation
            if value >= 1e-3 and value < 1e3:
                self.flux_cutoff_edit.setText(f"{value:.6f}".rstrip('0').rstrip('.'))
            else:
                self.flux_cutoff_edit.setText(f"{value:.2e}")
            
            # Update visualization immediately
            if hasattr(self, 'vtk_data') and self.vtk_data:
                self.update_current_visualization_scale()
                
        except ValueError:
            print(f"Invalid flux cutoff value: {self.flux_cutoff_edit.text()}")
            # Reset to previous valid value
            self.flux_cutoff_edit.setText(f"{self.current_flux_cutoff:.2e}")

    # def get_effective_scalar_range(self):
    #     """Get scalar range with flux cutoff applied"""
    #     if not hasattr(self, 'vtk_data') or not self.vtk_data:
    #         return (0, 1)

    #     # Get raw scalar range
    #     raw_range = self.vtk_data.GetScalarRange()

    #     # Apply cutoff to minimum
    #     cutoff = getattr(self, 'current_flux_cutoff', 1e-5)
    #     effective_min = max(raw_range[0], cutoff)
    #     effective_max = raw_range[1]

    #     # Ensure we have a valid range
    #     if effective_min >= effective_max:
    #         effective_min = effective_max * 0.001  # Use 0.1% of max as minimum

    #     return (effective_min, effective_max)

    def change_scale_mode(self, scale_mode):
        """Change scale mode with debugging and volume rendering support"""
        print(f"\n=== CHANGING SCALE MODE TO: {scale_mode} ===")

        # Add debugging
        self.debug_scalar_ranges()

        try:
            # Update regular field visualizations (existing logic)
            if hasattr(self, 'vtk_data') and self.vtk_data:
                self.update_current_visualization_scale()

            # NEW: Update volume rendering if active
            if (hasattr(self, 'volume_actor') and self.volume_actor and 
                hasattr(self, 'viz_mode_combo') and 
                self.viz_mode_combo.currentText() == "Volume Rendering"):

                print(f"Updating volume rendering scale to: {scale_mode}")

                # Get current threshold and reapply with new scale
                if hasattr(self, 'volume_threshold_slider'):
                    current_slider_value = self.volume_threshold_slider.value()
                    print(f"Reapplying volume threshold with new scale: slider={current_slider_value}%")
                    self.update_volume_threshold(current_slider_value)
                else:
                    print("Warning: volume_threshold_slider not found")

            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()

            print(f"Scale mode successfully changed to: {scale_mode}")
            print(f"=== SCALE MODE CHANGE COMPLETE ===\n")

        except Exception as e:
            print(f"Error changing scale mode: {e}")
            import traceback
            traceback.print_exc()

            # Try to recover by updating just the basic visualization
            try:
                if hasattr(self, 'vtk_data') and self.vtk_data:
                    self.update_current_visualization_scale()
                    if hasattr(self, 'vtk_widget'):
                        self.vtk_widget.GetRenderWindow().Render()
                print("Recovered with basic scale update")
            except Exception as recovery_error:
                print(f"Recovery also failed: {recovery_error}")


    def update_current_visualization_scale(self):
        """Update the color scale - FIXED to handle point cloud log scale properly"""
        try:
            print(f"\n=== UPDATING VISUALIZATION SCALE ===")
            
            # Get current mapper
            mapper = None
            if hasattr(self, 'field_actor') and self.field_actor:
                mapper = self.field_actor.GetMapper()
            elif hasattr(self, 'volume_actor') and self.volume_actor:
                self.update_volume_transfer_functions()
                return
            
            if not mapper:
                print("ERROR: No mapper found")
                return
            
            # Get STORED original range (not current filtered range)
            if hasattr(self, 'current_scalar_range') and self.current_scalar_range:
                original_range = self.current_scalar_range
                print(f"Using stored original range: {original_range}")
            else:
                original_range = self.vtk_data.GetScalarRange() if self.vtk_data else (1e-5, 1e7)
                self.current_scalar_range = original_range
                print(f"Fallback original range: {original_range}")
            
            # Get current settings
            cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
            scale_mode = self.scale_mode_combo.currentText()
            
            print(f"Flux cutoff: {cutoff:.2e}")
            print(f"Scale mode: {scale_mode}")
            
            # Calculate effective range for color mapping
            effective_min = max(original_range[0], cutoff)
            effective_max = original_range[1]
            
            # CRITICAL: For log scale, ensure minimum is positive and reasonable
            if scale_mode == "Logarithmic":
                if effective_min <= 0:
                    effective_min = cutoff if cutoff > 0 else 1e-8
                    print(f"Fixed zero minimum for log scale: {effective_min:.2e}")
                
                if effective_max <= effective_min:
                    effective_max = effective_min * 1000
                    print(f"Fixed max value for log scale: {effective_max:.2e}")
            
            effective_range = (effective_min, effective_max)
            print(f"Effective range for {scale_mode}: {effective_range}")
            
            # Create new lookup table with EFFECTIVE range (not original)
            lut = self.create_lookup_table_with_scale(
                self.colormap_combo.currentText(), 
                scale_mode,
                effective_range  # Use effective range with cutoff!
            )
            
            # IMPORTANT: Set mapper range to EFFECTIVE range, not original
            print(f"Setting mapper range to effective range: {effective_range}")
            mapper.SetScalarRange(effective_range[0], effective_range[1])
            mapper.SetLookupTable(lut)

            # ALSO update multiple isosurface (wireframe) actors with the effective range/LUT
            if hasattr(self, 'wireframe_actors'):
                for actor in self.wireframe_actors:
                    if actor:
                        wm = actor.GetMapper()
                        if wm:
                            wm.SetScalarRange(effective_range[0], effective_range[1])
                            wm.SetLookupTable(lut)

            # ALSO update multiple isosurface (wireframe) actors with the effective range/LUT
            if hasattr(self, 'wireframe_actors'):
                for actor in self.wireframe_actors:
                    if actor:
                        wm = actor.GetMapper()
                        if wm:
                            wm.SetScalarRange(effective_range[0], effective_range[1])
                            wm.SetLookupTable(lut)

            
            # Update scalar bar with effective range
            if hasattr(self, 'scalar_bar') and self.scalar_bar:
                scalar_array = self.vtk_data.GetPointData().GetScalars()
                scalar_name = scalar_array.GetName() if scalar_array else "Field Value"
                self.setup_scalar_bar(lut, scalar_name)
            
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            
            print(f"=== SCALE UPDATE COMPLETE ===\n")
            
        except Exception as e:
            print(f"ERROR updating scale: {e}")
            import traceback
            traceback.print_exc()

    # Add a manual debug button to your debug controls section
    def add_debug_button_to_controls(self):
        """Add debug button to existing debug controls"""
        # Find debug group and add the button
        for i in range(self.control_layout.count()):
            item = self.control_layout.itemAt(i)
            if item and isinstance(item.widget(), QGroupBox) and item.widget().title() == "Debug Controls":
                debug_group = item.widget()
                debug_layout = debug_group.layout()
                
                self.debug_ranges_button = QPushButton("Debug Scalar Ranges")
                self.debug_ranges_button.clicked.connect(self.debug_scalar_ranges)
                debug_layout.addWidget(self.debug_ranges_button)
                break
            
    def create_lookup_table_with_scale(self, colormap_name, scale_mode, scalar_range):
        """Create lookup table - FIXED to properly implement distinct colormaps"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        try:
            min_val, max_val = scalar_range
            
            print(f"\n=== CREATING {scale_mode} LUT for {colormap_name} ===")
            print(f"Input range: {min_val:.2e} to {max_val:.2e}")
            
            # CRITICAL FIX: Handle zero minimum for logarithmic scale
            if scale_mode == "Logarithmic":
                if min_val <= 0:
                    # Use flux cutoff or a fraction of max value
                    cutoff = getattr(self, 'current_flux_cutoff', 1e-5)
                    min_val = max(cutoff, max_val * 1e-6)
                    print(f"FIXED zero minimum for log scale: {min_val:.2e}")
                
                if max_val <= min_val:
                    max_val = min_val * 1000
                    print(f"FIXED invalid max for log scale: {max_val:.2e}")
            
            # Validate range is finite
            if not (np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val):
                print("FIXING invalid range with fallback")
                min_val, max_val = 1e-5, 1e7
            
            print(f"Final range: {min_val:.2e} to {max_val:.2e}")
            
            # Set range FIRST
            lut.SetRange(min_val, max_val)
            
            # FIXED: Build colormap with proper color assignment (no HSV override)
            for i in range(256):
                t = i / 255.0  # 0 to 1
                rgb_color = self.get_colormap_color(colormap_name, t)
                lut.SetTableValue(i, rgb_color[0], rgb_color[1], rgb_color[2], 1.0)
            
            # Set scale mode AFTER building colors
            if scale_mode == "Logarithmic":
                lut.SetScaleToLog10()
                print("Applied logarithmic scaling")
            else:
                lut.SetScaleToLinear()
                print("Applied linear scaling")
            
            # Test the LUT
            test_val = (min_val + max_val) / 2
            test_color = [0, 0, 0]
            lut.GetColor(test_val, test_color)
            print(f"Test color at {test_val:.2e}: {test_color}")
            
            if any(np.isnan(test_color)):
                print("WARNING: NaN in colors, forcing linear")
                lut.SetScaleToLinear()
            
            print(f"=== {colormap_name} LUT CREATION COMPLETE ===\n")
            return lut
            
        except Exception as e:
            print(f"ERROR in LUT creation: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback
            lut.SetRange(1e-5, 1e7)
            for i in range(256):
                t = i / 255.0
                lut.SetTableValue(i, t, 0.0, 1.0-t, 1.0)  # Simple blue to red
            lut.SetScaleToLinear()
            return lut

    def get_colormap_color(self, colormap_name, t):
        """Get RGB color for given colormap at position t (0-1) - FIXED distinct colormaps"""
        t = np.clip(t, 0, 1)

        if colormap_name == "Viridis":
            return self.get_viridis_color(t)
        elif colormap_name == "Plasma":
            return self.get_plasma_color(t)
        elif colormap_name == "Cool to Warm":
            # FIXED: Distinct cool to warm colormap (blue -> white -> red)
            if t < 0.5:
                # Blue to white
                factor = t * 2
                return [factor, factor, 1.0]
            else:
                # White to red
                factor = (t - 0.5) * 2
                return [1.0, 1.0 - factor, 1.0 - factor]
        elif colormap_name == "Grayscale":
            # FIXED: True grayscale
            return [t, t, t]
        elif colormap_name == "Rainbow":
            # FIXED: True rainbow using HSV conversion
            return self.get_rainbow_color(t)
        else:  # "Blue to Red" or default
            # FIXED: Simple blue to red transition
            return [t, 0.0, 1.0 - t]

    def get_rainbow_color(self, t):
        """Get rainbow color using proper HSV to RGB conversion"""
        # Map t to hue (0 = red, 0.17 = yellow, 0.33 = green, 0.5 = cyan, 0.67 = blue, 0.83 = magenta, 1 = red)
        hue = (1.0 - t) * 0.83  # Reverse so red is high values
        saturation = 1.0
        value = 1.0
        
        # HSV to RGB conversion
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return list(rgb)
        
    def get_viridis_color(self, t):
        """Improved viridis colormap approximation"""
        # High-quality viridis approximation using key points
        viridis_points = [
            (0.0, [0.267004, 0.004874, 0.329415]),
            (0.25, [0.229739, 0.322361, 0.545706]),
            (0.5, [0.127568, 0.566949, 0.550556]),
            (0.75, [0.369214, 0.788675, 0.382914]),
            (1.0, [0.993248, 0.909560, 0.143936])
        ]
        
        # Find the two points to interpolate between
        for i in range(len(viridis_points) - 1):
            t1, color1 = viridis_points[i]
            t2, color2 = viridis_points[i + 1]
            
            if t1 <= t <= t2:
                # Linear interpolation
                factor = (t - t1) / (t2 - t1)
                return [
                    color1[0] + factor * (color2[0] - color1[0]),
                    color1[1] + factor * (color2[1] - color1[1]),
                    color1[2] + factor * (color2[2] - color1[2])
                ]
        
        # Fallback (shouldn't happen)
        return [t, t, t]

    def get_plasma_color(self, t):
        """Improved plasma colormap approximation"""
        # High-quality plasma approximation using key points
        plasma_points = [
            (0.0, [0.050383, 0.029803, 0.527975]),
            (0.25, [0.513094, 0.038756, 0.627828]),
            (0.5, [0.796386, 0.278894, 0.469397]),
            (0.75, [0.940015, 0.644680, 0.222675]),
            (1.0, [0.940015, 0.975158, 0.131326])
        ]
        
        # Find the two points to interpolate between
        for i in range(len(plasma_points) - 1):
            t1, color1 = plasma_points[i]
            t2, color2 = plasma_points[i + 1]
            
            if t1 <= t <= t2:
                # Linear interpolation
                factor = (t - t1) / (t2 - t1)
                return [
                    color1[0] + factor * (color2[0] - color1[0]),
                    color1[1] + factor * (color2[1] - color1[1]),
                    color1[2] + factor * (color2[2] - color1[2])
                ]
        
        # Fallback
        return [t, 0, 1-t]

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import colorsys
        return list(colorsys.hsv_to_rgb(h, s, v))

    def change_colormap(self, colormap_name):
        """Change the color mapping with volume rendering support"""
        print(f"\n=== CHANGING COLORMAP TO: {colormap_name} ===")

        try:
            # Get current scale mode
            scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"

            # Get scalar range
            scalar_range = getattr(self, 'current_scalar_range', 
                                  self.vtk_data.GetScalarRange() if hasattr(self, 'vtk_data') and self.vtk_data else (0, 1))

            # Create new lookup table with current scale mode for regular visualizations
            new_lut = self.create_lookup_table_with_scale(colormap_name, scale_mode, scalar_range)

            # Update regular field actors
            if hasattr(self, 'field_actor') and self.field_actor:
                mapper = self.field_actor.GetMapper()
                if mapper:
                    mapper.SetLookupTable(new_lut)
                    print("Updated field_actor colormap")

            # Update slice actors
            if hasattr(self, 'slice_actors'):
                for i, actor in enumerate(self.slice_actors):
                    if actor:
                        mapper = actor.GetMapper()
                        if mapper:
                            mapper.SetLookupTable(new_lut)
                if hasattr(self, 'slice_actors') and self.slice_actors:
                    print(f"Updated {len(self.slice_actors)} slice actors colormap")

            # Update wireframe (multiple isosurface) mappers to the new LUT
            if hasattr(self, 'wireframe_actors'):
                # Determine EFFECTIVE range for consistency with current scale
                cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
                original_range = getattr(self, 'current_scalar_range',
                                             self.vtk_data.GetScalarRange() if self.vtk_data else (1e-8, 1))
                effective_min = max(original_range[0], cutoff)
                effective_max = max(original_range[1], effective_min * 1.0001)
                effective_range = (effective_min, effective_max)

                for actor in self.wireframe_actors:
                    if actor:
                        wm = actor.GetMapper()
                        if wm:
                            wm.SetLookupTable(new_lut)
                            wm.SetScalarRange(effective_range[0], effective_range[1])

            # Update wireframe actors (if they use colormaps)
            if hasattr(self, 'wireframe_actors'):
                for actor in self.wireframe_actors:
                    if actor:
                        mapper = actor.GetMapper()
                        if mapper and hasattr(mapper, 'GetLookupTable') and mapper.GetLookupTable():
                            mapper.SetLookupTable(new_lut)

            # NEW: Update volume rendering if active
            if (hasattr(self, 'volume_actor') and self.volume_actor and 
                hasattr(self, 'viz_mode_combo') and 
                self.viz_mode_combo.currentText() == "Volume Rendering"):

                print(f"Updating volume rendering colormap to: {colormap_name}")

                # Get current threshold and reapply with new colormap
                if hasattr(self, 'volume_threshold_slider'):
                    current_slider_value = self.volume_threshold_slider.value()
                    print(f"Reapplying volume threshold with new colormap: slider={current_slider_value}%")
                    self.update_volume_threshold(current_slider_value)
                else:
                    print("Warning: volume_threshold_slider not found")

            # Update volume rendering transfer functions for other modes too
            if hasattr(self, 'volume_actor') and self.volume_actor:
                self.update_volume_transfer_functions()

            # Update scalar bar
            if hasattr(self, 'scalar_bar') and self.scalar_bar:
                self.scalar_bar.SetLookupTable(new_lut)
                print("Updated scalar bar colormap")

            # Force render
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()

            print(f"Colormap successfully changed to: {colormap_name} ({scale_mode} scale)")
            print(f"=== COLORMAP CHANGE COMPLETE ===\n")

        except Exception as e:
            print(f"Error changing colormap: {e}")
            import traceback
            traceback.print_exc()

            # Try basic recovery
            try:
                if hasattr(self, 'vtk_widget'):
                    self.vtk_widget.GetRenderWindow().Render()
                print("Attempted basic recovery render")
            except:
                print("Recovery render also failed")

    def update_volume_transfer_functions(self):
        """Update volume rendering transfer functions for log scale"""
        if not hasattr(self, 'volume_actor') or not self.volume_actor:
            return

        try:
            volume_property = self.volume_actor.GetProperty()
            scalar_range = self.vtk_data.GetScalarRange()
            scale_mode = self.scale_mode_combo.currentText()

            # Create new transfer functions
            color_func = vtk.vtkColorTransferFunction()
            opacity_func = vtk.vtkPiecewiseFunction()

            if scale_mode == "Logarithmic" and scalar_range[1] > scalar_range[0] and scalar_range[0] > 0:
                # Log scale for volume rendering
                log_min = np.log10(scalar_range[0])
                log_max = np.log10(scalar_range[1])

                # Create points in log space
                for i in range(5):
                    log_val = log_min + i * (log_max - log_min) / 4
                    linear_val = 10**log_val
                    t = i / 4.0

                    # Color points
                    if t < 0.25:
                        color = [0.0, 0.0, 0.2 + t*0.8]
                    elif t < 0.5:
                        color = [0.0, (t-0.25)*4, 1.0]
                    elif t < 0.75:
                        color = [(t-0.5)*4, 1.0, 1.0-(t-0.5)*4]
                    else:
                        color = [1.0, 1.0-(t-0.75)*4, 0.0]

                    color_func.AddRGBPoint(linear_val, color[0], color[1], color[2])

                    # Opacity points (more transparent for low values)
                    opacity = min(0.3, 0.05 + t * 0.25)
                    opacity_func.AddPoint(linear_val, opacity)
            else:
                # Linear scale (original)
                color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.2)
                color_func.AddRGBPoint(scalar_range[1] * 0.25, 0.0, 0.0, 1.0)
                color_func.AddRGBPoint(scalar_range[1] * 0.5, 0.0, 1.0, 0.0)
                color_func.AddRGBPoint(scalar_range[1] * 0.75, 1.0, 1.0, 0.0)
                color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)

                opacity_func.AddPoint(scalar_range[0], 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.1, 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.3, 0.05)
                opacity_func.AddPoint(scalar_range[1] * 0.6, 0.1)
                opacity_func.AddPoint(scalar_range[1], 0.2)

            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)

            self.vtk_widget.GetRenderWindow().Render()
            print(f"Updated volume rendering to {scale_mode} scale")

        except Exception as e:
            print(f"Error updating volume transfer functions: {e}")

        def change_colormap(self, colormap_name):
            """Change the color mapping - UPDATED for scale awareness"""
            try:
                # Get current scale mode
                scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"

                # Get scalar range
                scalar_range = getattr(self, 'current_scalar_range', self.vtk_data.GetScalarRange() if self.vtk_data else (0, 1))

                # Create new lookup table with current scale mode
                new_lut = self.create_lookup_table_with_scale(colormap_name, scale_mode, scalar_range)

                # Update all relevant mappers
                if hasattr(self, 'field_actor') and self.field_actor:
                    mapper = self.field_actor.GetMapper()
                    if mapper:
                        mapper.SetLookupTable(new_lut)

                if hasattr(self, 'slice_actors'):
                    for actor in self.slice_actors:
                        if actor:
                            mapper = actor.GetMapper()
                            if mapper:
                                mapper.SetLookupTable(new_lut)

                # Update volume rendering if needed
                if hasattr(self, 'volume_actor') and self.volume_actor:
                    self.update_volume_transfer_functions()

                # Update scalar bar
                if hasattr(self, 'scalar_bar') and self.scalar_bar:
                    self.scalar_bar.SetLookupTable(new_lut)

                self.vtk_widget.GetRenderWindow().Render()
                print(f"Changed colormap to: {colormap_name} ({scale_mode} scale)")

            except Exception as e:
                print(f"Error changing colormap: {e}")
        
    def update_point_density(self, density):
        """COMPACT: Update point density with simple labeling"""
        self.point_density_label.setText(f"{density:,} points")
        
        # Only regenerate if we're in point cloud mode
        if (hasattr(self, 'viz_mode_combo') and 
            self.viz_mode_combo.currentText() == "Point Cloud" and
            hasattr(self, 'vtk_data') and self.vtk_data):
            
            if hasattr(self, '_density_update_timer'):
                self._density_update_timer.stop()
                
            self._density_update_timer = QTimer()
            self._density_update_timer.setSingleShot(True)
            self._density_update_timer.timeout.connect(lambda: self._regenerate_point_cloud(density))
            self._density_update_timer.start(200)

    def change_wireframe_style(self, style_name):
        """Clear and rebuild wireframe/isoviz when the style changes."""
        print(f"[WF] Style changed to: {style_name}")

        # Show/hide isosurface controls (only for Single Isosurface)
        if hasattr(self, 'isosurface_controls'):
            self.isosurface_controls.setVisible(style_name == "Single Isosurface")

        if not self.vtk_data:
            print("[WF] No vtk_data; ignoring style change.")
            return

        mode = self.viz_mode_combo.currentText() if hasattr(self, 'viz_mode_combo') else ""
        print(f"[WF] Current viz mode: {mode}")

        # Rebuild when we are in Wireframe mode.
        if mode == "Wireframe":
            print("[WF] Rebuilding wireframe…")
            self.clear_field_visualization()
            self.setup_wireframe_rendering()
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
            return

        # Keep existing behavior for isosurface mode
        if mode in ("Isosurface", "Isosurfaces"):
            print("[WF] Rebuilding isosurfaces…")
            self.clear_field_visualization()
            self.setup_wireframe_rendering()
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
            return

        print(f"[WF] Style change ignored for mode: {mode}")

    def update_isosurface_level(self, level_percent):
        """COMPACT: Update isosurface level with percentile clarification"""
        # Update label to clearly indicate it's a flux percentile
        if hasattr(self, 'vtk_data') and self.vtk_data:
            scalar_range = self.vtk_data.GetScalarRange()
            actual_value = scalar_range[1] * (level_percent / 100.0)
            self.isosurface_level_label.setText(f"{level_percent}th flux percentile ({actual_value:.1e})")
        else:
            self.isosurface_level_label.setText(f"{level_percent}th flux percentile")
        
        # Only update if we're in the right mode
        if not (self.viz_mode_combo.currentText() == "Wireframe" and 
                self.wireframe_style_combo.currentText() == "Single Isosurface" and
                self.vtk_data):
            return
        
        # FAST UPDATE: Direct contour level change
        if hasattr(self, 'field_actor') and self.field_actor:
            try:
                mapper = self.field_actor.GetMapper()
                if mapper:
                    input_conn = mapper.GetInputConnection(0, 0)
                    if input_conn:
                        edges_filter = input_conn.GetProducer()
                        contour_conn = edges_filter.GetInputConnection(0, 0)
                        if contour_conn:
                            contour_filter = contour_conn.GetProducer()
                            
                            scalar_range = self.vtk_data.GetScalarRange()
                            new_level = scalar_range[1] * (level_percent / 100.0)
                            
                            contour_filter.SetValue(0, new_level)
                            contour_filter.Modified()
                            
                            self.vtk_widget.GetRenderWindow().Render()
                            return
            except:
                pass
        
        self._schedule_isosurface_update()

    def _schedule_isosurface_update(self):
        """Schedule isosurface update with longer debounce to avoid spam"""
        if hasattr(self, '_iso_update_timer'):
            self._iso_update_timer.stop()
            
        self._iso_update_timer = QTimer()
        self._iso_update_timer.setSingleShot(True)
        self._iso_update_timer.timeout.connect(self._rebuild_isosurface)
        self._iso_update_timer.start(300)  # 300ms delay (longer to avoid spam)

    def _rebuild_isosurface(self):
        """Full rebuild of isosurface (slow method)"""
        print("Rebuilding isosurface (slow method)")
        self.clear_field_visualization()
        self.setup_wireframe_rendering()
        self.vtk_widget.GetRenderWindow().Render()
            
    # def _update_isosurface(self):
    #     """Actually update the isosurface"""
    #     self.clear_field_visualization()
    #     self.setup_wireframe_rendering()
    #     self.vtk_widget.GetRenderWindow().Render()

    def change_slice_axis(self, axis_text):
        """FIXED: Clear previous slice when axis changes"""
        print(f"Changing slice axis to: {axis_text}")
        
        if self.viz_mode_combo.currentText() == "Slice Planes" and self.vtk_data:
            # IMPORTANT: Clear previous slice first
            self.clear_field_visualization()
            
            # Create new slice with new axis
            self.setup_slice_planes()
            self.vtk_widget.GetRenderWindow().Render()

    def update_slice_position(self, position_percent):
        """COMPACT: Update slice position with concise coordinate info"""
        # More compact labeling
        if hasattr(self, 'vtk_data') and self.vtk_data:
            bounds = self.vtk_data.GetBounds()
            axis_text = self.slice_axis_combo.currentText()
            
            if "X-Axis" in axis_text:
                coord_range = bounds[1] - bounds[0]
                coord_pos = bounds[0] + (position_percent/100.0) * coord_range
                axis_name = "X"
            elif "Y-Axis" in axis_text:
                coord_range = bounds[3] - bounds[2]
                coord_pos = bounds[2] + (position_percent/100.0) * coord_range
                axis_name = "Y"
            else:  # Z-Axis
                coord_range = bounds[5] - bounds[4]
                coord_pos = bounds[4] + (position_percent/100.0) * coord_range
                axis_name = "Z"
            
            # Compact label format
            if position_percent <= 25:
                position_name = "Near"
            elif position_percent >= 75:
                position_name = "Far"
            else:
                position_name = "Center"
                
            self.slice_position_label.setText(f"{position_name} ({axis_name}={coord_pos:.0f}km)")
        else:
            if position_percent <= 25:
                label = "Near"
            elif position_percent >= 75:
                label = "Far" 
            else:
                label = "Center"
            self.slice_position_label.setText(f"{label} ({position_percent}%)")
        
        # Only update if we're in slice mode
        if not (self.viz_mode_combo.currentText() == "Slice Planes" and self.vtk_data):
            return
        
        # FAST UPDATE: Direct plane origin change
        if hasattr(self, 'field_actor') and self.field_actor:
            try:
                mapper = self.field_actor.GetMapper()
                if mapper:
                    input_conn = mapper.GetInputConnection(0, 0)
                    if input_conn:
                        cutter = input_conn.GetProducer()
                        if hasattr(cutter, 'GetCutFunction'):
                            plane = cutter.GetCutFunction()
                            
                            bounds = self.vtk_data.GetBounds()
                            axis_text = self.slice_axis_combo.currentText()
                            
                            if "X-Axis" in axis_text:
                                origin_coord = bounds[0] + (position_percent/100.0) * (bounds[1] - bounds[0])
                                new_origin = [origin_coord, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                            elif "Y-Axis" in axis_text:
                                origin_coord = bounds[2] + (position_percent/100.0) * (bounds[3] - bounds[2])
                                new_origin = [(bounds[0]+bounds[1])/2, origin_coord, (bounds[4]+bounds[5])/2]
                            else:  # Z-Axis
                                origin_coord = bounds[4] + (position_percent/100.0) * (bounds[5] - bounds[4])
                                new_origin = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, origin_coord]
                            
                            plane.SetOrigin(new_origin)
                            plane.Modified()
                            cutter.Modified()
                            
                            self.vtk_widget.GetRenderWindow().Render()
                            return
            except:
                pass
        
        self._schedule_slice_update()

    def _schedule_slice_update(self):
        """Schedule slice update with debounce"""
        if hasattr(self, '_slice_update_timer'):
            self._slice_update_timer.stop()
            
        self._slice_update_timer = QTimer()
        self._slice_update_timer.setSingleShot(True)
        self._slice_update_timer.timeout.connect(self._rebuild_slice)
        self._slice_update_timer.start(200)  # 200ms delay

    def _rebuild_slice(self):
        """Full rebuild of slice (slow method)"""
        print("Rebuilding slice (slow method)")
        self.clear_field_visualization()
        self.setup_slice_planes()
        self.vtk_widget.GetRenderWindow().Render()

    # def _update_slice(self):
    #     """Actually update the slice"""
    #     self.clear_field_visualization()
    #     self.setup_slice_planes()
    #     self.vtk_widget.GetRenderWindow().Render()
            
    def update_point_size(self, value):
        """COMPACT: Update point size with simple labeling"""
        radius_m = value
        self.point_size_label.setText(f"{radius_m}m radius")
        
        # Only update if we're in point cloud mode and have a field actor
        if (self.viz_mode_combo.currentText() == "Point Cloud" and 
            hasattr(self, 'field_actor') and self.field_actor):
            
            self.update_point_cloud_size(radius_m)

    def _regenerate_point_cloud(self, density):
        """Regenerate point cloud with new density"""
        try:
            print(f"Regenerating point cloud with {density} points...")
            
            # Store the new target density
            self.target_point_count = density
            
            # Clear existing field visualization
            self.clear_field_visualization()
            
            # Regenerate point cloud
            self.setup_point_cloud_rendering()
            
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error regenerating point cloud: {e}")

    def setup_point_cloud_rendering(self):
        """Point cloud rendering - FIXED to preserve proper scalar range for log scale"""
        if not self.vtk_data:
            return
            
        print("Setting up point cloud rendering...")
        
        try:
            # Get scalar data
            scalar_array = self.vtk_data.GetPointData().GetScalars()
            if not scalar_array:
                print("No scalar data for point cloud")
                return
                
            scalar_range = scalar_array.GetRange()
            num_points = self.vtk_data.GetNumberOfPoints()
            print(f"Point cloud: {num_points} points, range: {scalar_range}")
            
            # CRITICAL: Store ORIGINAL range BEFORE any filtering for proper color scaling
            self.current_scalar_range = scalar_range
            print(f"Stored original scalar range: {scalar_range}")
            
            # Get target density from slider
            target_density = getattr(self, 'target_point_count', 
                                   self.point_density_slider.value() if hasattr(self, 'point_density_slider') else 5000)
            
            print(f"Target density: {target_density} points")
            
            # Use flux cutoff for thresholding, but with fallback
            cutoff = getattr(self, 'current_flux_cutoff', 1e-5)
            
            # Determine threshold value
            if cutoff > 0 and cutoff < scalar_range[1] * 0.1:  # Cutoff is reasonable
                threshold_value = cutoff
                print(f"Using flux cutoff: {threshold_value:.2e}")
            else:  # Cutoff is too high or invalid, use percentage
                threshold_value = scalar_range[1] * 0.001  # 0.1% of max
                print(f"Cutoff too high, using 0.1% of max: {threshold_value:.2e}")
            
            # Apply threshold
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(self.vtk_data)
            threshold.SetLowerThreshold(threshold_value)
            threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
            threshold.Update()
            
            significant_data = threshold.GetOutput()
            significant_points = significant_data.GetNumberOfPoints()
            print(f"Significant points (>{threshold_value:.2e}): {significant_points}")
            
            # Fallback if no significant points
            if significant_points == 0:
                print("No points found, using lower threshold...")
                for lower_fraction in [0.0001, 0.00001, 0.0]:
                    threshold_value = scalar_range[1] * lower_fraction if lower_fraction > 0 else scalar_range[0]
                    threshold.SetLowerThreshold(threshold_value)
                    threshold.Update()
                    significant_data = threshold.GetOutput()
                    significant_points = significant_data.GetNumberOfPoints()
                    print(f"  Trying threshold {threshold_value:.2e}: {significant_points} points")
                    if significant_points > 0 or lower_fraction == 0:
                        break
                        
            if significant_points == 0:
                print("Still no points, using all data")
                significant_data = self.vtk_data
                significant_points = num_points
            
            # Subsample to target density
            if significant_points > target_density:
                print(f"Subsampling {significant_points} -> {target_density} points")
                mask = vtk.vtkMaskPoints()
                mask.SetInputData(significant_data)
                mask.SetMaximumNumberOfPoints(target_density)
                mask.SetRandomMode(True)
                mask.Update()
                final_data = mask.GetOutput()
            else:
                final_data = significant_data
                
            final_count = final_data.GetNumberOfPoints()
            print(f"Final point count: {final_count}")
            
            if final_count == 0:
                print("ERROR: No points to visualize!")
                return
            
            # Add jitter and create visualization
            jittered_data = self.add_spatial_jitter(final_data)
            self.current_final_data = jittered_data
            
            # Create glyphs
            initial_radius = self.point_size_slider.value() if hasattr(self, 'point_size_slider') else 400
            self.create_point_cloud_glyphs(jittered_data, initial_radius)
            
            print(f"Point cloud complete: {final_count} spheres")
            
        except Exception as e:
            print(f"Point cloud rendering failed: {e}")
            import traceback
            traceback.print_exc()

    def update_point_size(self, value):
        """Update point size for point cloud visualization"""
        radius_m = value  # Slider value is in meters
        self.point_size_label.setText(f"{radius_m}m radius")
        
        # Only update if we're in point cloud mode and have a field actor
        if (self.viz_mode_combo.currentText() == "Point Cloud" and 
            hasattr(self, 'field_actor') and self.field_actor):
            
            print(f"Updating point size to {radius_m}m radius")
            
            # Re-create the point cloud with new size
            self.update_point_cloud_size(radius_m)

    def update_point_cloud_size(self, radius_m):
        """Fast point cloud size update - OPTIMIZED"""
        if not hasattr(self, 'current_final_data') or not self.current_final_data:
            return
            
        # Use a timer to debounce rapid slider changes
        if hasattr(self, '_size_update_timer'):
            self._size_update_timer.stop()
            
        self._size_update_timer = QTimer()
        self._size_update_timer.setSingleShot(True)
        self._size_update_timer.timeout.connect(lambda: self._do_size_update(radius_m))
        self._size_update_timer.start(50)  # 50ms delay to debounce

    def _do_size_update(self, radius_m):
        """Actually perform the size update"""
        try:
            print(f"Updating to radius {radius_m}m...")
            
            # Quick method: just update the sphere source if possible
            if (hasattr(self, 'field_actor') and self.field_actor and 
                hasattr(self, 'current_sphere_source')):
                
                # Try to update existing sphere source
                self.current_sphere_source.SetRadius(radius_m)
                self.current_sphere_source.Modified()
                
            else:
                # Fallback: recreate glyphs (slower but more reliable)
                self.create_point_cloud_glyphs(self.current_final_data, radius_m)
            
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error in size update: {e}")
            # Fallback to full recreation
            self.create_point_cloud_glyphs(self.current_final_data, radius_m)
            self.vtk_widget.GetRenderWindow().Render()

    # def update_threshold_display(self):
    #     """Update threshold labels without applying"""
    #     min_val = self.threshold_min_slider.value()
    #     max_val = self.threshold_max_slider.value()
    #     self.threshold_min_label.setText(f"{min_val}%")
    #     self.threshold_max_label.setText(f"{max_val}%")

    # def apply_threshold_filter(self):
    #     """Apply threshold filter to current visualization"""
    #     if self.viz_mode_combo.currentText() == "Point Cloud":
    #         # For point cloud, just update with current threshold
    #         current_size = self.point_size_slider.value()
    #         self.update_point_cloud_size(current_size)
    #     else:
    #         # For other modes, use existing threshold logic
    #         self.update_threshold()


    def apply_volume_threshold(self, flux_threshold):
        """Apply flux threshold with user-selected color mapping (clean version)"""
        try:
            if not hasattr(self, 'volume_actor') or not self.volume_actor:
                return

            volume_property = self.volume_actor.GetProperty()
            scalar_range = self.volume_data.GetScalarRange()

            # Get user preferences (no debug output)
            colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Blue to Red'
            scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else 'Linear'

            # Create transfer functions
            color_func = self.create_volume_color_function(flux_threshold, scalar_range, colormap_name, scale_mode)
            opacity_func = self.create_volume_opacity_function(flux_threshold, scalar_range)

            # Apply the transfer functions
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)

            # Only log significant threshold changes
            if not hasattr(self, '_last_logged_threshold') or abs(flux_threshold - self._last_logged_threshold) > flux_threshold * 0.1:
                print(f"Volume threshold updated: {flux_threshold:.1e} ({colormap_name}, {scale_mode})")
                self._last_logged_threshold = flux_threshold

        except Exception as e:
            print(f"Error applying volume threshold: {e}")

    def create_volume_opacity_function(self, flux_threshold, scalar_range):
        """Create opacity function with fewer points to reduce texture size"""
        try:
            opacity_func = vtk.vtkPiecewiseFunction()

            # KEY FIX: Use minimal opacity points to reduce texture requirements
            opacity_func.AddPoint(scalar_range[0], 0.0)                      # Background invisible
            opacity_func.AddPoint(flux_threshold * 0.999, 0.0)              # Just below threshold invisible
            opacity_func.AddPoint(flux_threshold, 0.6)                      # At threshold suddenly visible
            opacity_func.AddPoint(scalar_range[1], 0.9)                     # Maximum flux most opaque
            # Removed intermediate points to reduce texture size

            return opacity_func

        except Exception as e:
            print(f"Error creating volume opacity function: {e}")
            # Minimal fallback
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(scalar_range[0], 0.0)
            opacity_func.AddPoint(flux_threshold, 0.7)
            opacity_func.AddPoint(scalar_range[1], 0.9)
            return opacity_func

    def create_volume_color_function(self, flux_threshold, scalar_range, colormap_name, scale_mode):
        """Create color transfer function with reasonable number of points"""
        try:
            color_func = vtk.vtkColorTransferFunction()

            # Everything below threshold = black (invisible)
            color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(flux_threshold * 0.999, 0.0, 0.0, 0.0)

            # Color range for visible data
            color_range_min = flux_threshold
            color_range_max = scalar_range[1]

            # KEY FIX: Use fewer color points to reduce texture requirements
            if scale_mode == "Logarithmic" and color_range_min > 0:
                # Logarithmic color mapping with fewer points
                log_min = np.log10(color_range_min)
                log_max = np.log10(color_range_max)

                num_points = 4  # Reduced from 8 to 4
                for i in range(num_points):
                    log_pos = log_min + i * (log_max - log_min) / (num_points - 1)
                    flux_value = 10**log_pos
                    color_position = i / (num_points - 1)
                    rgb_color = self.get_colormap_color(colormap_name, color_position)
                    color_func.AddRGBPoint(flux_value, rgb_color[0], rgb_color[1], rgb_color[2])
            else:
                # Linear color mapping with fewer points
                num_points = 3  # Reduced from 6 to 3
                for i in range(num_points):
                    flux_value = color_range_min + i * (color_range_max - color_range_min) / (num_points - 1)
                    color_position = i / (num_points - 1)
                    rgb_color = self.get_colormap_color(colormap_name, color_position)
                    color_func.AddRGBPoint(flux_value, rgb_color[0], rgb_color[1], rgb_color[2])

            return color_func

        except Exception as e:
            print(f"Error creating volume color function: {e}")
            # Simple fallback
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(flux_threshold, 0.0, 0.0, 1.0)
            color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)
            return color_func

    def update_volume_threshold(self, value):
        """Update volume threshold with dual-scale mapping (clean version)"""
        if not (hasattr(self, 'volume_actor') and self.volume_actor and 
                self.viz_mode_combo.currentText() == "Volume Rendering"):
            return

        try:
            # Dual-scale mapping
            if value <= 50:
                # First half: 0-50% slider -> 1st-90th percentile (linear)
                percentile = 1 + (value / 50.0) * 89
                scale_region = "broad"
            else:
                # Second half: 50-100% slider -> 90th-99.9th percentile (linear)
                normalized = (value - 50) / 50.0
                percentile = 90 + normalized * 9.9
                scale_region = "high-res"

            # Calculate actual flux threshold from percentile
            flux_threshold = self.calculate_flux_from_percentile(percentile)

            # Update label with appropriate precision
            if scale_region == "broad":
                self.volume_threshold_label.setText(f"{percentile:.0f}th %ile ({flux_threshold:.1e})")
            else:
                self.volume_threshold_label.setText(f"{percentile:.1f}th %ile ({flux_threshold:.1e})")

            # Apply the threshold to volume rendering
            self.apply_volume_threshold(flux_threshold)

            # Force render
            self.vtk_widget.GetRenderWindow().Render()

            # Minimal debug output (only for major changes)
            if value % 25 == 0:  # Only at 0%, 25%, 50%, 75%, 100%
                print(f"Volume threshold: {percentile:.1f}th percentile ({flux_threshold:.1e})")

        except Exception as e:
            print(f"Error updating volume threshold: {e}")

    def calculate_flux_from_percentile(self, target_percentile):
        """Calculate flux value for a given percentile (1-99th range)"""
        try:
            if not hasattr(self, 'volume_data') or not self.volume_data:
                return 1e-6  # Fallback

            scalar_array = self.volume_data.GetPointData().GetScalars()
            if not scalar_array:
                return 1e-6

            # Convert to numpy for percentile calculation
            import vtk.util.numpy_support as vtk_np
            flux_values = vtk_np.vtk_to_numpy(scalar_array)

            # Remove zeros/background for meaningful percentiles
            non_zero_flux = flux_values[flux_values > 0]

            if len(non_zero_flux) == 0:
                print("No non-zero flux values found")
                return flux_values.max() * 0.1

            # Calculate the percentile directly
            flux_threshold = np.percentile(non_zero_flux, target_percentile)

            return flux_threshold

        except Exception as e:
            print(f"Error calculating flux from percentile: {e}")
            return 1e-6

    # def calculate_extended_volume_percentiles(self):
    #     """Calculate extended percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th) for smooth interpolation"""
    #     try:
    #         scalar_array = self.volume_data.GetPointData().GetScalars()
    #         if not scalar_array:
    #             return

    #         import vtk.util.numpy_support as vtk_np
    #         flux_values = vtk_np.vtk_to_numpy(scalar_array)

    #         # Remove zeros/background
    #         non_zero_flux = flux_values[flux_values > 0]

    #         if len(non_zero_flux) == 0:
    #             return

    #         # Calculate key percentiles for smooth interpolation
    #         percentiles_to_calc = [10, 25, 50, 75, 90, 95, 99]
    #         flux_percentiles = np.percentile(non_zero_flux, percentiles_to_calc)

    #         # Store as dictionary for easy lookup
    #         self.flux_percentiles = dict(zip(percentiles_to_calc, flux_percentiles))

    #         print(f"Extended flux percentiles calculated: {self.flux_percentiles}")

    #     except Exception as e:
    #         print(f"Error calculating extended percentiles: {e}")

    # def interpolate_flux_from_percentile(self, target_percentile):
    #     """Interpolate flux value from percentile using stored percentile data"""
    #     try:
    #         if not hasattr(self, 'flux_percentiles'):
    #             return self.flux_10th_percentile  # Fallback

    #         percentile_keys = sorted(self.flux_percentiles.keys())

    #         # Find the two percentiles that bracket our target
    #         if target_percentile <= percentile_keys[0]:
    #             return self.flux_percentiles[percentile_keys[0]]
    #         elif target_percentile >= percentile_keys[-1]:
    #             return self.flux_percentiles[percentile_keys[-1]]
    #         else:
    #             # Interpolate between two bracketing percentiles
    #             for i in range(len(percentile_keys) - 1):
    #                 lower_p = percentile_keys[i]
    #                 upper_p = percentile_keys[i + 1]

    #                 if lower_p <= target_percentile <= upper_p:
    #                     # Linear interpolation
    #                     lower_flux = self.flux_percentiles[lower_p]
    #                     upper_flux = self.flux_percentiles[upper_p]

    #                     fraction = (target_percentile - lower_p) / (upper_p - lower_p)
    #                     interpolated_flux = lower_flux + fraction * (upper_flux - lower_flux)

    #                     return interpolated_flux

    #         # Fallback
    #         return self.flux_10th_percentile

    #     except Exception as e:
    #         print(f"Error interpolating flux from percentile: {e}")
    #         return self.flux_10th_percentile
            
    def change_visualization_mode(self, mode):
        """Enhanced mode change with volume threshold control visibility"""
        print(f"=== CHANGING VISUALIZATION MODE TO: {mode} ===")

        # Show/hide relevant controls based on mode
        self.point_cloud_controls.setVisible(mode == "Point Cloud")
        self.wireframe_controls.setVisible(mode == "Wireframe")
        self.slice_controls.setVisible(mode == "Slice Planes")
        self.volume_controls.setVisible(mode == "Volume Rendering")  # NEW: Show volume controls

        # Show/hide sub-controls for wireframe
        if mode == "Wireframe":
            style = self.wireframe_style_combo.currentText()
            self.isosurface_controls.setVisible(style == "Single Isosurface")

        # Handle transparency slider behavior
        if mode == "Volume Rendering":
            # Disable transparency slider for volume rendering (we have dedicated threshold control)
            self.opacity_slider.setEnabled(False)
            self.opacity_label.setText("N/A (use threshold)")
            print("Volume rendering: using dedicated threshold control")
        else:
            # Enable transparency slider for other modes
            self.opacity_slider.setEnabled(True)
            self.opacity_label.setText(f"{self.opacity_slider.value()}%")
            print("Using normal transparency control")

        # Don't proceed if no data loaded
        if not self.vtk_data:
            return

        try:
            # Clear existing visualization
            self.clear_field_visualization()

            # Apply the selected visualization mode
            if mode == "Volume Rendering":
                self.setup_volume_rendering()
            elif mode == "Isosurfaces":
                self.setup_isosurface_rendering()
            elif mode == "Point Cloud":
                self.setup_point_cloud_rendering()
            elif mode == "Wireframe":
                self.setup_wireframe_rendering()
            elif mode == "Surface with Edges":
                self.setup_surface_with_edges()
            elif mode == "Slice Planes":
                self.setup_slice_planes()
            else:
                self.setup_field_visualization()

            self.vtk_widget.GetRenderWindow().Render()
            print(f"=== VISUALIZATION MODE CHANGED TO: {mode} ===")

        except Exception as e:
            print(f"Error changing visualization mode: {e}")
            self.setup_point_cloud_rendering()
            self.vtk_widget.GetRenderWindow().Render()

    # def update_volume_threshold_label(self):
    #     """Update the transparency label to show threshold info for volume rendering"""
    #     try:
    #         if (hasattr(self, 'volume_actor') and self.volume_actor and 
    #             self.viz_mode_combo.currentText() == "Volume Rendering"):

    #             # Change label to show it's a threshold, not transparency
    #             slider_value = self.opacity_slider.value()

    #             if hasattr(self, 'flux_10th_percentile') and hasattr(self, 'flux_90th_percentile'):
    #                 current_threshold = self.flux_10th_percentile + (slider_value / 100.0) * (
    #                     self.flux_90th_percentile - self.flux_10th_percentile
    #                 )

    #                 # Update the label to show threshold info
    #                 threshold_percentile = 10 + (slider_value / 100.0) * 80  # 10th to 90th percentile
    #                 self.opacity_label.setText(f"{threshold_percentile:.0f}th %ile ({current_threshold:.1e})")
    #             else:
    #                 self.opacity_label.setText(f"{slider_value}% threshold")
    #         else:
    #             # Regular transparency label for other modes
    #             self.opacity_label.setText(f"{self.opacity_slider.value()}%")

    #     except Exception as e:
    #         print(f"Error updating volume threshold label: {e}")

    def clear_field_visualization(self):
        """ENHANCED clearing to handle ALL visualization actors properly"""
        print("=== CLEARING ALL FIELD VISUALIZATION ===")
        
        actors_removed = 0
        
        # Remove field actor (used by most modes)
        if hasattr(self, 'field_actor') and self.field_actor:
            self.renderer.RemoveActor(self.field_actor)
            self.field_actor = None
            actors_removed += 1
            print("Removed field_actor")
            
        # Remove volume actor
        if hasattr(self, 'volume_actor') and self.volume_actor:
            self.renderer.RemoveVolume(self.volume_actor)
            self.volume_actor = None
            actors_removed += 1
            print("Removed volume_actor")
            
        # Remove wireframe actors (for multiple wireframe mode)
        if hasattr(self, 'wireframe_actors'):
            for i, actor in enumerate(self.wireframe_actors):
                if actor:
                    self.renderer.RemoveActor(actor)
                    actors_removed += 1
            self.wireframe_actors = []
            print(f"Removed wireframe_actors")
            
        # Remove slice actors (for multiple slice mode)
        if hasattr(self, 'slice_actors'):
            for i, actor in enumerate(self.slice_actors):
                if actor:
                    self.renderer.RemoveActor(actor)
                    actors_removed += 1
            self.slice_actors = []
            print(f"Removed slice_actors")
            
        # Remove isosurface actors
        if hasattr(self, 'isosurface_actors'):
            for i, actor in enumerate(self.isosurface_actors):
                if actor:
                    self.renderer.RemoveActor(actor)
                    actors_removed += 1
            self.isosurface_actors = []
            print(f"Removed isosurface_actors")
            
        # IMPORTANT: Remove scalar bar (it persists across modes)
        if hasattr(self, 'scalar_bar') and self.scalar_bar:
            self.renderer.RemoveViewProp(self.scalar_bar)
            self.scalar_bar = None
            print("Removed scalar_bar")
            
        print(f"Total actors removed: {actors_removed}")
        print("=== CLEAR COMPLETE ===")

    def setup_volume_rendering(self):
        """Volume rendering setup with GPU-friendly texture sizes"""
        if not self.vtk_data:
            return

        print("Setting up volume rendering...")

        try:
            # Safety check
            num_points = self.vtk_data.GetNumberOfPoints()
            if num_points > 500000:
                reply = QMessageBox.question(
                    self, "Large Dataset Warning", 
                    f"Dataset has {num_points:,} points. Volume rendering may be slow.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.viz_mode_combo.setCurrentText("Point Cloud")
                    return

            # Get or create volume data
            if isinstance(self.vtk_data, vtk.vtkImageData):
                volume_data = self.vtk_data
            else:
                volume_data = self.create_downsampled_volume_data()

            if not volume_data or volume_data.GetNumberOfPoints() == 0:
                print("Volume data creation failed, switching to point cloud")
                self.setup_point_cloud_rendering()
                return

            # Store for threshold calculations
            self.volume_data = volume_data

            # Create volume mapper with GPU-friendly settings
            try:
                volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
                # KEY FIX: Set maximum texture size to something reasonable
                volume_mapper.SetMaxMemoryInBytes(256 * 1024 * 1024)  # 256MB limit
                volume_mapper.SetMaxMemoryFraction(0.5)  # Use max 50% of GPU memory
            except:
                volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()

            volume_mapper.SetInputData(volume_data)

            # Volume properties
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOff()

            # Create and add volume
            self.volume_actor = vtk.vtkVolume()
            self.volume_actor.SetMapper(volume_mapper)
            self.volume_actor.SetProperty(volume_property)
            self.renderer.AddVolume(self.volume_actor)

            # Apply initial threshold
            initial_value = self.volume_threshold_slider.value() if hasattr(self, 'volume_threshold_slider') else 50
            self.update_volume_threshold(initial_value)

            print("Volume rendering setup complete")

        except Exception as e:
            print(f"Volume rendering setup failed: {e}")
            self.setup_point_cloud_rendering()
            
    def create_downsampled_volume_data(self):
        """Create heavily downsampled volume data for performance"""
        try:
            # Get bounds
            bounds = self.vtk_data.GetBounds()

            # AGGRESSIVE downsampling - max 30x30x30 = 27k points
            max_dimension = 30

            spacing = [
                (bounds[1]-bounds[0]) / max_dimension,
                (bounds[3]-bounds[2]) / max_dimension, 
                (bounds[5]-bounds[4]) / max_dimension
            ]

            print(f"Downsampling to {max_dimension}³ = {max_dimension**3:,} points")
            print(f"Spacing: {spacing[0]:.0f} x {spacing[1]:.0f} x {spacing[2]:.0f} km")

            # Resample to structured grid
            resample = vtk.vtkResampleToImage()
            resample.SetInputData(self.vtk_data)
            resample.SetSamplingDimensions(max_dimension, max_dimension, max_dimension)
            resample.SetSamplingBounds(bounds)
            resample.Update()

            volume_data = resample.GetOutput()

            if volume_data.GetNumberOfPoints() == 0:
                print("Resampling failed")
                return None

            print(f"Volume data created: {volume_data.GetNumberOfPoints():,} points")
            return volume_data

        except Exception as e:
            print(f"Error creating downsampled volume: {e}")
            return None

    def update_volume_transfer_functions(self):
        """Update volume rendering transfer functions - OPTIMIZED VERSION"""
        if not hasattr(self, 'volume_actor') or not self.volume_actor:
            return

        try:
            volume_property = self.volume_actor.GetProperty()

            # Get the volume data (might be downsampled)
            volume_mapper = self.volume_actor.GetMapper()
            volume_data = volume_mapper.GetInput()
            scalar_range = volume_data.GetScalarRange()

            scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"

            # Create MINIMAL transfer functions for speed
            color_func = vtk.vtkColorTransferFunction()
            opacity_func = vtk.vtkPiecewiseFunction()

            if scale_mode == "Logarithmic" and scalar_range[1] > scalar_range[0] and scalar_range[0] > 0:
                # Log scale - but keep it SIMPLE
                log_min = np.log10(max(scalar_range[0], scalar_range[1] * 1e-6))
                log_max = np.log10(scalar_range[1])

                # Only 3 points for speed
                for i in range(3):
                    log_val = log_min + i * (log_max - log_min) / 2
                    linear_val = 10**log_val
                    t = i / 2.0

                    # Simple color progression
                    if t < 0.5:
                        color = [0.0, t*2, 1.0]
                    else:
                        color = [(t-0.5)*2, 1.0, 1.0-(t-0.5)*2]

                    color_func.AddRGBPoint(linear_val, color[0], color[1], color[2])

                    # VERY low opacity for performance
                    opacity = min(0.1, 0.01 + t * 0.09)
                    opacity_func.AddPoint(linear_val, opacity)
            else:
                # Linear scale - MINIMAL points
                color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.2)
                color_func.AddRGBPoint(scalar_range[1] * 0.5, 0.0, 1.0, 0.0)
                color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)

                # VERY sparse and low opacity
                opacity_func.AddPoint(scalar_range[0], 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.3, 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.7, 0.05)
                opacity_func.AddPoint(scalar_range[1], 0.1)

            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)

            # Force a render
            self.vtk_widget.GetRenderWindow().Render()
            print(f"Updated volume rendering to {scale_mode} scale (optimized)")

        except Exception as e:
            print(f"Error updating volume transfer functions: {e}")

    def setup_isosurface_rendering(self):
        """Fixed isosurface rendering"""
        if not self.vtk_data:
            return
            
        print("Setting up isosurface rendering...")
        
        try:
            # Get scalar range
            scalar_range = self.vtk_data.GetScalarRange()
            print(f"Scalar range for isosurfaces: {scalar_range}")
            
            if scalar_range[1] <= scalar_range[0]:
                print("Invalid scalar range for isosurfaces")
                return
                
            # Create contour filter
            contour = vtk.vtkContourFilter()
            contour.SetInputData(self.vtk_data)
            
            # Generate isosurface values - only in meaningful range
            num_contours = self.num_isosurfaces_spin.value()
            min_val = scalar_range[1] * 0.3  # Start at 30% of max
            max_val = scalar_range[1] * 0.9  # End at 90% of max
            
            if max_val > min_val:
                contour.GenerateValues(num_contours, min_val, max_val)
                contour.Update()
                
                output = contour.GetOutput()
                print(f"Contour generated {output.GetNumberOfPoints()} points")
                
                if output.GetNumberOfPoints() > 0:
                    # Create mapper and actor
                    contour_mapper = vtk.vtkPolyDataMapper()
                    contour_mapper.SetInputConnection(contour.GetOutputPort())
                    contour_mapper.SetScalarRange(scalar_range)
                    
                    # Setup lookup table
                    lut = self.create_lookup_table(self.colormap_combo.currentText())
                    contour_mapper.SetLookupTable(lut)
                    
                    self.field_actor = vtk.vtkActor()
                    self.field_actor.SetMapper(contour_mapper)
                    self.field_actor.GetProperty().SetOpacity(self.opacity_slider.value() / 100.0)
                    
                    self.renderer.AddActor(self.field_actor)
                    print(f"Created {num_contours} isosurfaces")
                else:
                    print("No isosurfaces generated - trying point cloud fallback")
                    self.setup_point_cloud_rendering()
            else:
                print("Invalid contour range")
                
        except Exception as e:
            print(f"Isosurface rendering failed: {e}")
            self.setup_point_cloud_rendering()

    def add_spatial_jitter(self, input_data):
        """Add random spatial jitter to break up grid artifacts"""
        if input_data.GetNumberOfPoints() == 0:
            print("No points to jitter")
            return input_data
            
        print("Adding spatial jitter to reduce grid artifacts...")
        
        try:
            # Get the spacing of the original structured grid
            jitter_radius = 300.0  # km
            
            # Create a new points array with jitter
            original_points = input_data.GetPoints()
            n_points = original_points.GetNumberOfPoints()
            
            jittered_points = vtk.vtkPoints()
            
            import random
            random.seed(42)  # Reproducible jitter
            
            for i in range(n_points):
                # Get original point
                orig_point = original_points.GetPoint(i)
                
                # Add random jitter in all three dimensions
                jitter_x = (random.random() - 0.5) * 2 * jitter_radius
                jitter_y = (random.random() - 0.5) * 2 * jitter_radius  
                jitter_z = (random.random() - 0.5) * 2 * jitter_radius
                
                # Create jittered point
                new_point = [
                    orig_point[0] + jitter_x,
                    orig_point[1] + jitter_y,
                    orig_point[2] + jitter_z
                ]
                
                jittered_points.InsertNextPoint(new_point)
            
            # Create new polydata with jittered points
            jittered_polydata = vtk.vtkPolyData()
            jittered_polydata.SetPoints(jittered_points)
            
            # Copy scalar data
            jittered_polydata.GetPointData().SetScalars(input_data.GetPointData().GetScalars())
            
            # Create vertices
            vertices = vtk.vtkCellArray()
            for i in range(n_points):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                vertices.InsertNextCell(vertex)
            jittered_polydata.SetVerts(vertices)
            
            print(f"Applied jitter to {n_points} points (±{jitter_radius}km)")
            return jittered_polydata
            
        except Exception as e:
            print(f"Error adding jitter: {e}")
            return input_data  # Return original if jitter fails

    def create_point_cloud_glyphs(self, point_data, radius):
        """Create the actual glyph visualization - FIXED to prevent log scale NaN issues"""
        if point_data.GetNumberOfPoints() == 0:
            print("No points to create glyphs for")
            return
            
        try:
            print(f"Creating glyphs for {point_data.GetNumberOfPoints()} points...")
            
            # Use simpler spheres for better performance
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(8)  # Reduced for performance
            sphere.SetPhiResolution(8)    # Reduced for performance
            
            # Store sphere source for potential fast updates
            self.current_sphere_source = sphere
            
            # Create glyph
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(point_data)
            glyph.SetSourceConnection(sphere.GetOutputPort())
            glyph.SetScaleModeToDataScalingOff()
            glyph.SetColorModeToColorByScalar()
            glyph.Update()
            
            print(f"Glyph filter created {glyph.GetOutput().GetNumberOfPoints()} glyph points")
            
            # Create mapper
            glyph_mapper = vtk.vtkPolyDataMapper()
            glyph_mapper.SetInputConnection(glyph.GetOutputPort())
            
            # CRITICAL FIX: Always use the stored original range, never the filtered range
            if hasattr(self, 'current_scalar_range') and self.current_scalar_range:
                original_range = self.current_scalar_range
                print(f"Using stored original range: {original_range}")
            else:
                # Fallback: get from original data
                original_range = self.vtk_data.GetScalarRange()
                self.current_scalar_range = original_range
                print(f"Fallback to VTK data range: {original_range}")
            
            # ADDITIONAL FIX: For log scale, ensure range doesn't include zeros
            scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"
            if scale_mode == "Logarithmic":
                min_val, max_val = original_range
                if min_val <= 0:
                    cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
                    safe_min = max(cutoff, max_val * 1e-6)
                    original_range = (safe_min, max_val)
                    print(f"FIXED range for log scale: {original_range}")
            
            glyph_mapper.SetScalarRange(original_range)
            glyph_mapper.ScalarVisibilityOn()
            
            print(f"Mapper scalar range set to: {original_range}")
            
            # Setup lookup table with current settings and SAFE range
            colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Blue to Red'
            lut = self.create_lookup_table_with_scale(colormap_name, scale_mode, original_range)
            glyph_mapper.SetLookupTable(lut)
            
            # Create or update actor
            if hasattr(self, 'field_actor') and self.field_actor:
                # Update existing actor
                self.field_actor.SetMapper(glyph_mapper)
            else:
                # Create new actor
                self.field_actor = vtk.vtkActor()
                self.field_actor.SetMapper(glyph_mapper)
                self.renderer.AddActor(self.field_actor)
            
            # Set opacity
            opacity = self.opacity_slider.value() / 100.0 if hasattr(self, 'opacity_slider') else 0.7
            self.field_actor.GetProperty().SetOpacity(opacity)
            
            # Setup scalar bar with SAFE original range
            scalar_name = self.vtk_data.GetPointData().GetScalars().GetName()
            self.setup_scalar_bar(lut, scalar_name)
            
            print(f"Glyphs created successfully: {point_data.GetNumberOfPoints()} spheres, radius {radius}m")
            print(f"Color scale: {scale_mode} with range {original_range}")
            
        except Exception as e:
            print(f"Error creating glyphs: {e}")
            import traceback
            traceback.print_exc()

    def setup_wireframe_rendering(self):
        """Enhanced wireframe with user-controllable options and color mapping"""
        if not self.vtk_data:
            return

        print("Setting up enhanced wireframe rendering with color mapping...")

        try:
            scalar_array = self.vtk_data.GetPointData().GetScalars()
            if not scalar_array:
                print("No scalar data for wireframe")
                return

            scalar_range = scalar_array.GetRange()

            # Get current wireframe style
            style = getattr(self, 'wireframe_style_combo', None)
            if style:
                current_style = style.currentText()
            else:
                current_style = "Single Isosurface"

            # Always use LUT-based coloring for wireframes
            current_color_mode = "Color by Flux Value"

            if current_style == "Single Isosurface":
                self.create_controllable_single_isosurface_with_color(scalar_range, current_color_mode)
            elif current_style == "Multiple Isosurface":
                self.create_isosurface_wireframes_with_color(scalar_range, current_color_mode)
            else:
                self.create_boundary_wireframe()

            print("Enhanced wireframe rendering with color mapping complete")

        except Exception as e:
            print(f"Wireframe rendering failed: {e}")

    def create_controllable_single_isosurface_with_color(self, scalar_range, color_mode):
        """Create isosurface with color mapping support"""
        if scalar_range[1] <= scalar_range[0]:
            return
            
        level_percent = getattr(self, 'isosurface_level_slider', None)
        if level_percent:
            percent = level_percent.value() / 100.0
        else:
            percent = 0.5
            
        contour_level = scalar_range[1] * percent
        print(f"Creating isosurface wireframe at {percent*100:.0f}% ({contour_level:.2e}) with {color_mode}")
        
        # Create contour filter
        self.current_contour_filter = vtk.vtkContourFilter()
        self.current_contour_filter.SetInputData(self.vtk_data)
        self.current_contour_filter.SetValue(0, contour_level)
        self.current_contour_filter.Update()
        
        contour_output = self.current_contour_filter.GetOutput()
        
        if contour_output.GetNumberOfPoints() > 0:
            # Create edges filter
            self.current_edges_filter = vtk.vtkExtractEdges()
            self.current_edges_filter.SetInputConnection(self.current_contour_filter.GetOutputPort())
            self.current_edges_filter.Update()
            
            wireframe_mapper = vtk.vtkPolyDataMapper()
            wireframe_mapper.SetInputConnection(self.current_edges_filter.GetOutputPort())
            
            # Apply LUT coloring (always) with safe log range
            cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
            effective_min = max(scalar_range[0], cutoff)
            effective_max = max(scalar_range[1], effective_min * 1.0001)
            effective_range = (effective_min, effective_max)

            wireframe_mapper.ScalarVisibilityOn()
            if hasattr(wireframe_mapper, 'SetScalarModeToUsePointData'):
                wireframe_mapper.SetScalarModeToUsePointData()
                scalar_array = self.vtk_data.GetPointData().GetScalars()
            if scalar_array and scalar_array.GetName():
                wireframe_mapper.SelectColorArray(scalar_array.GetName())

            lut = self.create_lookup_table_with_scale(self.colormap_combo.currentText(),
                                                          self.scale_mode_combo.currentText(),
                                                          effective_range)
            wireframe_mapper.SetScalarRange(effective_range)
            wireframe_mapper.SetLookupTable(lut)

            # Scalar bar
            self.setup_scalar_bar(lut, scalar_array.GetName() if scalar_array else "Field Value")
            
            self.field_actor = vtk.vtkActor()
            self.field_actor.SetMapper(wireframe_mapper)
            
            if color_mode == "Solid Color":
                self.field_actor.GetProperty().SetColor(0.9, 0.9, 0.2)  # Yellow
                
            self.field_actor.GetProperty().SetLineWidth(2.0)
            self.field_actor.GetProperty().SetOpacity(self.opacity_slider.value() / 100.0)
            
            self.renderer.AddActor(self.field_actor)
            print(f"Isosurface wireframe created: {contour_output.GetNumberOfPoints()} points with {color_mode}")
        else:
            print("No contour generated at this level")

    def create_isosurface_wireframes_with_color(self, scalar_range, color_mode):
        """Create multiple isosurface wireframes from UI levels with LUT coloring (log-safe)."""
        import vtk
        import numpy as np

        # Basic guards
        if not hasattr(self, 'vtk_data') or not self.vtk_data:
            return
        if not scalar_range or scalar_range[1] <= scalar_range[0]:
            return

        # Build levels (percent → absolute) from the UI; fall back to defaults
        ui_percents = self._get_multi_iso_levels() if hasattr(self, '_get_multi_iso_levels') else None
        if not ui_percents:
            ui_percents = [20, 40, 60, 80]
        levels = [scalar_range[1] * (p / 100.0) for p in ui_percents]

        # Clear any previous wireframe actors for a clean rebuild
        if hasattr(self, 'wireframe_actors') and self.wireframe_actors:
            for actor in self.wireframe_actors:
                try:
                    self.renderer.RemoveActor(actor)
                except Exception:
                    pass
        self.wireframe_actors = []

        # Compute a log-safe effective range (avoid log(0) → NaN)
        cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
        effective_min = max(scalar_range[0], cutoff)
        effective_max = max(scalar_range[1], effective_min * 1.0001)
        effective_range = (effective_min, effective_max)

        # Colormap / scale mode
        colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Blue to Red'
        scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else 'Linear'
        lut = self.create_lookup_table_with_scale(colormap_name, scale_mode, effective_range)

        # Scalar array name (for mapper selection)
        scalar_array = self.vtk_data.GetPointData().GetScalars()
        scalar_name = scalar_array.GetName() if scalar_array else "Field Value"

        # Opacity from UI if present
        try:
            ui_opacity = (self.opacity_slider.value() / 100.0) if hasattr(self, 'opacity_slider') else 0.9
        except Exception:
            ui_opacity = 0.9

        # Build each level
        for i, level in enumerate(levels):
            contour = vtk.vtkContourFilter()
            contour.SetInputData(self.vtk_data)
            contour.SetValue(0, float(level))
            contour.Update()

            contour_output = contour.GetOutput()
            if not contour_output or contour_output.GetNumberOfPoints() == 0:
                continue

            # Extract edges to render as wireframe
            edges = vtk.vtkExtractEdges()
            edges.SetInputData(contour_output)
            edges.Update()

            # Mapper with LUT coloring (always on)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(edges.GetOutputPort())
            mapper.ScalarVisibilityOn()
            if hasattr(mapper, 'SetScalarModeToUsePointData'):
                mapper.SetScalarModeToUsePointData()
            if scalar_array and scalar_name:
                mapper.SelectColorArray(scalar_name)
            mapper.SetScalarRange(effective_range[0], effective_range[1])
            mapper.SetLookupTable(lut)

            # Actor properties
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetLineWidth(2.5)
            actor.GetProperty().SetOpacity(ui_opacity)

            # Add to scene and keep a handle
            self.renderer.AddActor(actor)
            self.wireframe_actors.append(actor)

        # One scalar bar for all
        if hasattr(self, 'setup_scalar_bar'):
            self.setup_scalar_bar(lut, scalar_name)

        # Trigger a render
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()

    def create_boundary_wireframe(self):
        import vtk

        if not getattr(self, "renderer", None) or not getattr(self, "vtk_data", None):
            return False

        # Remove prior actor
        if getattr(self, "field_actor", None):
            try: self.renderer.RemoveActor(self.field_actor)
            except Exception: pass
            self.field_actor = None

        # Get bounds (fallback via geometry filter if needed)
        data = self.vtk_data
        bounds = data.GetBounds() if hasattr(data, "GetBounds") else None
        if not bounds or len(bounds) != 6 or any(b is None for b in bounds):
            gf = vtk.vtkGeometryFilter()
            if hasattr(data, "GetOutputPort"): gf.SetInputConnection(data.GetOutputPort())
            else:                               gf.SetInputData(data)
            gf.Update()
            bounds = gf.GetOutput().GetBounds()
            if not bounds or len(bounds) != 6: return False

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        eps = 1e-6
        if xmax - xmin < eps: xmax = xmin + eps
        if ymax - ymin < eps: ymax = ymin + eps
        if zmax - zmin < eps: zmax = zmin + eps

        # Try outline filter first
        use_outline = True
        try:
            outline = vtk.vtkOutlineFilter()
            if hasattr(data, "GetOutputPort"): outline.SetInputConnection(data.GetOutputPort())
            else:                              outline.SetInputData(data)
            outline.Update()
            if outline.GetOutput().GetNumberOfCells() <= 0:
                use_outline = False
        except Exception:
            use_outline = False

        if use_outline:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())
            mapper.ScalarVisibilityOff()
        else:
            cube = vtk.vtkCubeSource()
            if hasattr(cube, "SetBounds"): cube.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)
            else:
                cube.SetCenter((xmin + xmax)/2.0, (ymin + ymax)/2.0, (zmin + zmax)/2.0)
                cube.SetXLength(max(xmax - xmin, eps))
                cube.SetYLength(max(ymax - ymin, eps))
                cube.SetZLength(max(zmax - zmin, eps))
            cube.Update()
            edges = vtk.vtkExtractEdges()
            edges.SetInputConnection(cube.GetOutputPort())
            edges.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(edges.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetRepresentationToWireframe()
        prop.SetLighting(False)
        # Color with basic contrast heuristic
        bg = self.renderer.GetBackground() if hasattr(self.renderer, "GetBackground") else (0.0, 0.0, 0.0)
        line = (1.0, 1.0, 0.0) if (0.2126*bg[0] + 0.7152*bg[1] + 0.0722*bg[2]) < 0.5 else (0.0, 0.0, 0.0)
        prop.SetColor(*line)
        prop.SetLineWidth(2.5)
        try:
            opacity = self.opacity_slider.value()/100.0
        except Exception:
            opacity = 1.0
        prop.SetOpacity(max(0.05, min(1.0, float(opacity))))

        self.field_actor = actor
        self.renderer.AddActor(actor)
        self.renderer.ResetCameraClippingRange()
        if hasattr(self, "vtk_widget"):
            self.vtk_widget.GetRenderWindow().Render()
        return True

    # Optional: backward-compat alias if anything else calls the old helper.
    def create_bounding_box_wireframe(self):
        return self.create_boundary_wireframe()

    def setup_surface_with_edges(self):
        """Fixed surface with edges rendering"""
        if not self.vtk_data:
            return
            
        print("Setting up surface with edges...")
        
        try:
            # Extract outer surface
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(self.vtk_data)
            surface.Update()
            
            surface_data = surface.GetOutput()
            print(f"Extracted surface with {surface_data.GetNumberOfPoints()} points")
            
            if surface_data.GetNumberOfPoints() == 0:
                print("No surface extracted - falling back to point cloud")
                self.setup_point_cloud_rendering()
                return
            
            # Create mapper
            surface_mapper = vtk.vtkPolyDataMapper()
            surface_mapper.SetInputData(surface_data)
            
            scalar_range = self.vtk_data.GetScalarRange()
            surface_mapper.SetScalarRange(scalar_range)
            
            lut = self.create_lookup_table(self.colormap_combo.currentText())
            surface_mapper.SetLookupTable(lut)
            
            self.field_actor = vtk.vtkActor()
            self.field_actor.SetMapper(surface_mapper)
            self.field_actor.GetProperty().SetRepresentationToSurface()
            self.field_actor.GetProperty().EdgeVisibilityOn()
            self.field_actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
            self.field_actor.GetProperty().SetLineWidth(1.0)
            self.field_actor.GetProperty().SetOpacity(self.opacity_slider.value() / 100.0)
            
            self.renderer.AddActor(self.field_actor)
            print("Surface with edges setup complete")
            
        except Exception as e:
            print(f"Surface with edges failed: {e}")

    def setup_slice_planes(self):
        """Create slice with stored references for fast updates"""
        if not self.vtk_data:
            return
            
        print("Setting up enhanced slice planes...")
        
        try:
            bounds = self.vtk_data.GetBounds()
            
            # Get current settings
            axis_combo = getattr(self, 'slice_axis_combo', None)
            axis_text = axis_combo.currentText() if axis_combo else "Z-Axis (XY Plane)"
            
            pos_slider = getattr(self, 'slice_position_slider', None)
            position_percent = pos_slider.value() / 100.0 if pos_slider else 0.5
            
            # Determine axis and position
            if "X-Axis" in axis_text:
                normal = [1, 0, 0]
                origin_coord = bounds[0] + position_percent * (bounds[1] - bounds[0])
                origin = [origin_coord, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                axis_name = "X"
            elif "Y-Axis" in axis_text:
                normal = [0, 1, 0]
                origin_coord = bounds[2] + position_percent * (bounds[3] - bounds[2])
                origin = [(bounds[0]+bounds[1])/2, origin_coord, (bounds[4]+bounds[5])/2]
                axis_name = "Y"
            else:  # Z-Axis
                normal = [0, 0, 1]
                origin_coord = bounds[4] + position_percent * (bounds[5] - bounds[4])
                origin = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, origin_coord]
                axis_name = "Z"
                
            print(f"Creating {axis_name}-axis slice at {position_percent*100:.0f}% ({origin_coord:.0f} km)")
            
            # Create cutting plane (store reference for fast updates)
            self.current_slice_plane = vtk.vtkPlane()
            self.current_slice_plane.SetOrigin(origin)
            self.current_slice_plane.SetNormal(normal)
            
            # Create cutter (store reference for fast updates)
            self.current_slice_cutter = vtk.vtkCutter()
            self.current_slice_cutter.SetInputData(self.vtk_data)
            self.current_slice_cutter.SetCutFunction(self.current_slice_plane)
            self.current_slice_cutter.Update()
            
            slice_data = self.current_slice_cutter.GetOutput()
            
            if slice_data.GetNumberOfPoints() > 0:
                slice_mapper = vtk.vtkPolyDataMapper()
                slice_mapper.SetInputConnection(self.current_slice_cutter.GetOutputPort())
                
                scalar_range = self.vtk_data.GetScalarRange()
                slice_mapper.SetScalarRange(scalar_range)
                
                lut = self.create_lookup_table(self.colormap_combo.currentText())
                slice_mapper.SetLookupTable(lut)
                
                self.field_actor = vtk.vtkActor()
                self.field_actor.SetMapper(slice_mapper)
                self.field_actor.GetProperty().SetOpacity(self.opacity_slider.value() / 100.0)
                
                self.renderer.AddActor(self.field_actor)
                
                # Setup scalar bar
                self.setup_scalar_bar(lut, "electron_flux")
                
                print(f"Solid slice plane created: {slice_data.GetNumberOfPoints()} points")
            else:
                print("No slice data generated")
                
        except Exception as e:
            print(f"Slice planes failed: {e}")

    def create_lookup_table(self, colormap_name):
        """Create lookup table - wrapper that uses current settings"""
        # Get current scale mode
        scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"
        
        # Get scalar range from stored current range (not modified by cutoff)
        if hasattr(self, 'current_scalar_range'):
            original_range = self.current_scalar_range
        else:
            original_range = self.vtk_data.GetScalarRange() if self.vtk_data else (1e-5, 1e7)
        
        # Apply flux cutoff to create effective range for display
        cutoff = getattr(self, 'current_flux_cutoff', 1e-5)
        effective_min = max(original_range[0], cutoff)
        effective_max = original_range[1]
        
        # Ensure valid log range
        if scale_mode == "Logarithmic":
            if effective_min <= 0:
                effective_min = effective_max * 1e-6 if effective_max > 0 else 1e-10
            if effective_max <= effective_min:
                effective_max = effective_min * 1000
        
        effective_range = (effective_min, effective_max)
        
        return self.create_lookup_table_with_scale(colormap_name, scale_mode, effective_range)

    def update_opacity(self, value):
        """Update opacity - EXCLUDE all trail actors from transparency changes"""
        self.opacity_label.setText(f"{value}%")

        # Skip volume rendering
        if (hasattr(self, 'volume_actor') and self.volume_actor and 
            self.viz_mode_combo.currentText() == "Volume Rendering"):
            return

        try:
            opacity = value / 100.0

            # Handle regular field actors
            if hasattr(self, 'field_actor') and self.field_actor:
                self.field_actor.GetProperty().SetOpacity(opacity)

            # Handle slice actors
            if hasattr(self, 'slice_actors'):
                for actor in self.slice_actors:
                    if actor:
                        actor.GetProperty().SetOpacity(opacity)

            # Handle wireframe actors (multiple isosurfaces)
            if hasattr(self, 'wireframe_actors'):
                for actor in self.wireframe_actors:
                    if actor:
                        actor.GetProperty().SetOpacity(opacity)

            # Handle wireframe actors (multiple isosurfaces)
            if hasattr(self, 'wireframe_actors'):
                for actor in self.wireframe_actors:
                    if actor:
                        actor.GetProperty().SetOpacity(opacity)
            
            # KEEP ALL TRAIL ACTORS at their designed opacity
            if hasattr(self, 'trail_actors'):
                for i, actor in enumerate(self.trail_actors):
                    if actor:
                        # Preserve the fading effect opacity
                        trail_position = i / (len(self.trail_actors) - 1) if len(self.trail_actors) > 1 else 1.0
                        designed_opacity = 0.1 + 0.8 * trail_position
                        actor.GetProperty().SetOpacity(designed_opacity)
                print(f"Trail actors kept at designed opacity despite transparency setting of {value}%")
            elif hasattr(self, 'trail_actor') and self.trail_actor:
                # Handle old single actor case
                self.trail_actor.GetProperty().SetOpacity(0.8)

            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            print(f"Error updating opacity: {e}")

    # def scale_volume_opacity_smoothly(self, slider_opacity):
    #     """Scale volume opacity smoothly without making blocks disappear"""
    #     try:
    #         if not hasattr(self, 'base_opacity_func') or not hasattr(self, 'volume_scalar_range'):
    #             print("No base opacity function stored, recreating...")
    #             # Fallback: recreate the smooth opacity function
    #             self.recreate_smooth_volume_opacity(slider_opacity)
    #             return

    #         volume_property = self.volume_actor.GetProperty()
    #         scalar_range = self.volume_scalar_range

    #         # Create new opacity function by scaling the base function
    #         new_opacity_func = vtk.vtkPiecewiseFunction()

    #         # Always keep a minimum opacity to prevent blocks from disappearing
    #         min_opacity = 0.02 * slider_opacity  # Minimum visibility scales with slider
    #         max_opacity = 0.8 * slider_opacity   # Maximum visibility scales with slider

    #         # Apply the scaled opacity curve
    #         new_opacity_func.AddPoint(scalar_range[0], 0.0)                           # Background stays invisible
    #         new_opacity_func.AddPoint(scalar_range[1] * 0.1, max(min_opacity, 0.01)) # Minimum for low flux
    #         new_opacity_func.AddPoint(scalar_range[1] * 0.3, min_opacity * 2)        # Scale medium flux
    #         new_opacity_func.AddPoint(scalar_range[1] * 0.6, min_opacity * 4)        # Scale higher flux  
    #         new_opacity_func.AddPoint(scalar_range[1], max_opacity)                  # Scale max flux

    #         # Apply the new opacity function
    #         volume_property.SetScalarOpacity(new_opacity_func)

    #         print(f"Volume opacity scaled smoothly: min={min_opacity:.3f}, max={max_opacity:.3f}")

    #     except Exception as e:
    #         print(f"Error scaling volume opacity: {e}")
    #         # Fallback
    #         self.recreate_smooth_volume_opacity(slider_opacity)

    def recreate_smooth_volume_opacity(self, slider_opacity):
        """Recreate smooth volume opacity from scratch if scaling fails"""
        try:
            if not hasattr(self, 'volume_actor') or not self.volume_actor:
                return

            volume_property = self.volume_actor.GetProperty()
            volume_mapper = self.volume_actor.GetMapper()
            volume_data = volume_mapper.GetInput()
            scalar_range = volume_data.GetScalarRange()

            # Create smooth opacity function
            opacity_func = vtk.vtkPiecewiseFunction()

            min_opacity = 0.02 * slider_opacity
            max_opacity = 0.8 * slider_opacity

            opacity_func.AddPoint(scalar_range[0], 0.0)
            opacity_func.AddPoint(scalar_range[1] * 0.1, max(min_opacity, 0.01))
            opacity_func.AddPoint(scalar_range[1] * 0.3, min_opacity * 2)
            opacity_func.AddPoint(scalar_range[1] * 0.6, min_opacity * 4)
            opacity_func.AddPoint(scalar_range[1], max_opacity)

            volume_property.SetScalarOpacity(opacity_func)

            # Store for future scaling
            self.base_opacity_func = opacity_func
            self.volume_scalar_range = scalar_range

            print(f"Recreated smooth volume opacity: min={min_opacity:.3f}, max={max_opacity:.3f}")

        except Exception as e:
            print(f"Error recreating smooth volume opacity: {e}")

    # def toggle_threshold(self, enabled):
    #     """Toggle threshold filtering - FIXED"""
    #     try:
    #         if enabled and self.vtk_data:
    #             self.update_threshold()
    #         else:
    #             # Restore original data visualization
    #             current_mode = self.viz_mode_combo.currentText()
    #             self.change_visualization_mode(current_mode)
    #     except Exception as e:
    #         print(f"Error toggling threshold: {e}")

    def update_threshold(self):
        """Update threshold values - FIXED"""
        if not self.threshold_enabled.isChecked() or not self.vtk_data:
            return
            
        try:
            scalar_range = self.vtk_data.GetScalarRange()
            if scalar_range[1] <= scalar_range[0]:
                print("Invalid scalar range for thresholding")
                return
                
            min_val = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * (self.threshold_min_slider.value() / 100.0)
            max_val = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * (self.threshold_max_slider.value() / 100.0)
            
            if min_val >= max_val:
                print("Invalid threshold range")
                return
                
            print(f"Applying threshold: {min_val:.2e} to {max_val:.2e}")
            
            # Apply threshold filter
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(self.vtk_data)
            threshold.SetLowerThreshold(min_val)
            threshold.SetUpperThreshold(max_val)
            threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            threshold.Update()
            
            thresholded_output = threshold.GetOutput()
            print(f"Threshold result: {thresholded_output.GetNumberOfPoints()} points")
            
            if thresholded_output.GetNumberOfPoints() > 0:
                # Apply current visualization mode to thresholded data
                self.apply_visualization_to_data(thresholded_output)
            else:
                print("No points passed threshold")
                
        except Exception as e:
            print(f"Error updating threshold: {e}")

    def apply_visualization_to_data(self, data):
        """Apply current visualization mode to given data - FIXED"""
        if not data:
            return
            
        try:
            # Clear existing visualization
            self.clear_field_visualization()
            
            # Temporarily replace vtk_data
            original_data = self.vtk_data
            self.vtk_data = data
            
            # Apply current visualization mode
            current_mode = self.viz_mode_combo.currentText()
            
            if current_mode == "Volume Rendering":
                self.setup_volume_rendering()
            elif current_mode == "Isosurfaces":
                self.setup_isosurface_rendering()
            elif current_mode == "Point Cloud":
                self.setup_point_cloud_rendering()
            elif current_mode == "Wireframe":
                self.setup_wireframe_rendering()
            elif current_mode == "Surface with Edges":
                self.setup_surface_with_edges()
            elif current_mode == "Slice Planes":
                self.setup_slice_planes()
            else:
                self.setup_field_visualization()
                
            # Restore original data reference
            self.vtk_data = original_data
            
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error applying visualization to data: {e}")
            # Restore original data and try basic visualization
            self.vtk_data = original_data
            self.setup_field_visualization()

    # def toggle_isosurfaces(self, enabled):
    #     """Toggle isosurface display - FIXED"""
    #     try:
    #         if enabled and self.vtk_data:
    #             if self.viz_mode_combo.currentText() != "Isosurfaces":
    #                 self.viz_mode_combo.setCurrentText("Isosurfaces")
    #             else:
    #                 self.setup_isosurface_rendering()
    #         elif not enabled:
    #             # Switch to a different mode or clear isosurfaces
    #             if self.viz_mode_combo.currentText() == "Isosurfaces":
    #                 self.viz_mode_combo.setCurrentText("Point Cloud")
    #     except Exception as e:
    #         print(f"Error toggling isosurfaces: {e}")

    # def update_isosurfaces(self):
    #     """Update number of isosurfaces - FIXED"""
    #     try:
    #         if (self.viz_mode_combo.currentText() == "Isosurfaces" and 
    #             self.isosurface_enabled.isChecked() and 
    #             self.vtk_data):
    #             self.setup_isosurface_rendering()
    #     except Exception as e:
    #         print(f"Error updating isosurfaces: {e}")

    def change_colormap(self, colormap_name):
        """Change the color mapping - FIXED"""
        try:
            new_lut = self.create_lookup_table(colormap_name)
            
            if hasattr(self, 'field_actor') and self.field_actor:
                mapper = self.field_actor.GetMapper()
                if mapper:
                    mapper.SetLookupTable(new_lut)
                    
            if hasattr(self, 'slice_actors'):
                for actor in self.slice_actors:
                    if actor:
                        mapper = actor.GetMapper()
                        if mapper:
                            mapper.SetLookupTable(new_lut)

            # Update multiple isosurface (wireframe) mappers with new LUT and effective range
            if hasattr(self, 'wireframe_actors'):
                cutoff = getattr(self, 'current_flux_cutoff', 1e-8)
                original_range = getattr(self, 'current_scalar_range',
                                         self.vtk_data.GetScalarRange() if self.vtk_data else (1e-8, 1))
                effective_min = max(original_range[0], cutoff)
                effective_max = max(original_range[1], effective_min * 1.0001)
                effective_range = (effective_min, effective_max)
                for actor in self.wireframe_actors:
                    if actor:
                        wm = actor.GetMapper()
                        if wm:
                            wm.SetLookupTable(new_lut)
                            wm.SetScalarRange(effective_range[0], effective_range[1])
                            
            # Update scalar bar
            if hasattr(self, 'scalar_bar') and self.scalar_bar:
                self.scalar_bar.SetLookupTable(new_lut)
                    
            self.vtk_widget.GetRenderWindow().Render()
            print(f"Changed colormap to: {colormap_name}")
            
        except Exception as e:
            print(f"Error changing colormap: {e}")
        
    # def create_earth_representation(self):
    #     """Create a simple Earth sphere for reference"""
    #     earth_sphere = vtk.vtkSphereSource()
    #     earth_sphere.SetRadius(6371.0)  # Earth radius in km
    #     earth_sphere.SetThetaResolution(50)
    #     earth_sphere.SetPhiResolution(50)
        
    #     earth_mapper = vtk.vtkPolyDataMapper()
    #     earth_mapper.SetInputConnection(earth_sphere.GetOutputPort())
        
    #     self.earth_actor = vtk.vtkActor()
    #     self.earth_actor.SetMapper(earth_mapper)
    #     self.earth_actor.GetProperty().SetColor(0.3, 0.3, 0.8)  # Blue-ish
    #     self.earth_actor.GetProperty().SetOpacity(0.3)
        
    #     self.renderer.AddActor(self.earth_actor)
        
    def connect_signals(self):
        """Connect UI signals - REMOVED duplicate timer connection"""
        print("DEBUG: Connecting UI signals...")
        
        self.load_vtk_button.clicked.connect(self.load_vtk_data)
        self.load_orbit_button.clicked.connect(self.load_orbital_data)
        
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        
        # NOTE: Timer connection is now done in __init__
        print("DEBUG: Timer connection handled in __init__")
        
        self.time_slider.valueChanged.connect(self.jump_to_time)
        self.speed_spinbox.valueChanged.connect(self.set_animation_speed)
        self.cross_section_spinbox.valueChanged.connect(self.update_cross_section)
        
        self.show_slice_button.clicked.connect(self.show_slice_window)
        self.show_spectrum_button.clicked.connect(self.show_spectrum_window)
        self.show_flux_time_button.clicked.connect(self.show_flux_time_window)
        
        print("DEBUG: All UI signals connected")


    def load_vtk_data(self):
        """Load VTK data file with comprehensive format support - CLEAN VERSION"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load VTK Data", "", 
            "VTK Files (*.vtk *.vtu *.vtp *.vts *.vti);;XML VTK (*.vtu *.vtp *.vts *.vti);;Legacy VTK (*.vtk);;All Files (*)"
        )
        
        if not file_path:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Determine file type and create appropriate reader
            file_ext = Path(file_path).suffix.lower()
            
            print(f"Loading VTK file: {file_path}")
            print(f"File extension: {file_ext}")
            
            if file_ext == '.vts':
                print("Using XML Structured Grid Reader")
                reader = vtk.vtkXMLStructuredGridReader()
            elif file_ext == '.vtu':
                print("Using XML Unstructured Grid Reader")
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_ext == '.vtp':
                print("Using XML PolyData Reader")
                reader = vtk.vtkXMLPolyDataReader()
            elif file_ext == '.vti':
                print("Using XML Image Data Reader")
                reader = vtk.vtkXMLImageDataReader()
            elif file_ext == '.vtk':
                # Legacy format - auto-detect type
                reader = self.create_legacy_vtk_reader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            self.progress_bar.setValue(25)
            
            # Set filename and read
            reader.SetFileName(file_path)
            print("Reading VTK file...")
            reader.Update()
            
            self.progress_bar.setValue(50)
            
            # Get the output
            output = reader.GetOutput()
            print(f"VTK object type: {type(output).__name__}")
            print(f"Number of points: {output.GetNumberOfPoints()}")
            print(f"Number of cells: {output.GetNumberOfCells()}")
            
            if output.GetNumberOfPoints() == 0:
                raise ValueError("VTK file contains no data points")
            
            self.progress_bar.setValue(75)
            
            # Use original data directly without conversion for structured grids
            print(f"Using original {type(output).__name__} directly")
            self.vtk_data = output
            
            # Check and setup scalar data
            scalar_array = output.GetPointData().GetScalars()
            if scalar_array:
                print(f"Scalar data: {scalar_array.GetName()}, range: {scalar_array.GetRange()}")
            else:
                print("No primary scalars found, checking available arrays...")
                point_data = output.GetPointData()
                for i in range(point_data.GetNumberOfArrays()):
                    array = point_data.GetArray(i)
                    print(f"  Array {i}: {array.GetName()}")
                
                # Set first array as scalars if available
                if point_data.GetNumberOfArrays() > 0:
                    first_array = point_data.GetArray(0)
                    point_data.SetScalars(first_array)
                    print(f"Set '{first_array.GetName()}' as primary scalars")
                    scalar_array = first_array
            
            # Verify scalar data exists
            if not scalar_array:
                raise ValueError("No scalar data available for visualization")
            
            # Clear any existing visualization
            self.clear_field_visualization()
            
            # Setup field visualization
            print("Setting up field visualization...")
            self.setup_field_visualization()
            
            # Update analyzer
            self.flux_analyzer.set_vtk_data(self.vtk_data)
            
            self.progress_bar.setValue(100)
            
            # Update status
            scalar_name = scalar_array.GetName() if scalar_array else "None"
            scalar_range = scalar_array.GetRange() if scalar_array else (0, 0)
            
            self.status_label.setText(
                f"✓ Loaded VTK data successfully!\n"
                f"Points: {self.vtk_data.GetNumberOfPoints():,} | "
                f"Cells: {self.vtk_data.GetNumberOfCells():,}\n"
                f"Scalar: {scalar_name} | "
                f"Range: {scalar_range[0]:.2e} to {scalar_range[1]:.2e}"
            )
            
            # Force render
            print("Forcing render...")
            self.vtk_widget.GetRenderWindow().Render()
            
            print("VTK file loaded successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Error loading VTK file: {str(e)}")
            QMessageBox.critical(self, "VTK Loading Error", 
                               f"Failed to load VTK data:\n\n{str(e)}\n\n"
                               f"Supported formats:\n"
                               f"• .vts (XML Structured Grid)\n"
                               f"• .vtu (XML Unstructured Grid)\n"
                               f"• .vtp (XML PolyData)\n"
                               f"• .vti (XML Image Data)\n"
                               f"• .vtk (Legacy VTK)")
            import traceback
            print(f"Error details:\n{traceback.format_exc()}")
            
        finally:
            self.progress_bar.setVisible(False)

    def create_legacy_vtk_reader(self, file_path):
        """Create appropriate reader for legacy VTK files with better detection"""
        try:
            # Read header to determine type
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i in range(15):  # Read more lines for better detection
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.strip().upper())
            
            content = ' '.join(lines)
            print(f"VTK file header content: {content[:200]}...")
            
            # Detect format
            if 'STRUCTURED_GRID' in content:
                print("Detected: Structured Grid")
                return vtk.vtkStructuredGridReader()
            elif 'UNSTRUCTURED_GRID' in content:
                print("Detected: Unstructured Grid")
                return vtk.vtkUnstructuredGridReader()
            elif 'POLYDATA' in content:
                print("Detected: PolyData")
                return vtk.vtkPolyDataReader()
            elif 'STRUCTURED_POINTS' in content or 'DATASET STRUCTURED_POINTS' in content:
                print("Detected: Structured Points")
                return vtk.vtkStructuredPointsReader()
            elif 'RECTILINEAR_GRID' in content:
                print("Detected: Rectilinear Grid")
                return vtk.vtkRectilinearGridReader()
            else:
                print("Unknown format, trying generic data reader")
                # Try the generic reader first
                return vtk.vtkDataSetReader()
                
        except Exception as e:
            print(f"Error detecting VTK format: {e}")
            print("Defaulting to UnstructuredGridReader")
            return vtk.vtkUnstructuredGridReader()

    # def convert_to_unstructured_grid(self, structured_data):
    #     """Convert structured data (ImageData, StructuredGrid) to UnstructuredGrid"""
    #     print("Converting structured data to unstructured grid...")
        
    #     if isinstance(structured_data, vtk.vtkImageData):
    #         # Convert ImageData to UnstructuredGrid
    #         converter = vtk.vtkImageDataGeometryFilter()
    #         converter.SetInputData(structured_data)
    #         converter.Update()
            
    #         # Convert resulting polydata to unstructured grid
    #         return self.convert_polydata_to_unstructured_grid(converter.GetOutput())
            
    #     elif isinstance(structured_data, vtk.vtkStructuredGrid):
    #         # Convert StructuredGrid to UnstructuredGrid
    #         converter = vtk.vtkStructuredGridGeometryFilter()
    #         converter.SetInputData(structured_data)
    #         converter.Update()
            
    #         # Convert resulting polydata to unstructured grid
    #         return self.convert_polydata_to_unstructured_grid(converter.GetOutput())
            
    #     elif isinstance(structured_data, vtk.vtkRectilinearGrid):
    #         # Convert RectilinearGrid to UnstructuredGrid
    #         converter = vtk.vtkRectilinearGridGeometryFilter()
    #         converter.SetInputData(structured_data)
    #         converter.Update()
            
    #         return self.convert_polydata_to_unstructured_grid(converter.GetOutput())
        
    #     else:
    #         # If it's already unstructured, return as-is
    #         return structured_data

    def convert_polydata_to_unstructured_grid(self, polydata):
        """Convert PolyData to UnstructuredGrid"""
        print("Converting PolyData to UnstructuredGrid...")
        
        ugrid = vtk.vtkUnstructuredGrid()
        
        # Copy points
        ugrid.SetPoints(polydata.GetPoints())
        
        # Copy cells
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        
        # Copy point data
        ugrid.GetPointData().ShallowCopy(polydata.GetPointData())
        
        # Copy cell data
        ugrid.GetCellData().ShallowCopy(polydata.GetCellData())
        
        return ugrid

    # def setup_scalar_data(self):
    #     """Setup and verify scalar data for visualization"""
    #     if not self.vtk_data:
    #         return
            
    #     point_data = self.vtk_data.GetPointData()
        
    #     # Check if we have scalar data
    #     scalar_array = point_data.GetScalars()
        
    #     if not scalar_array:
    #         print("No scalar data found, checking available arrays...")
            
    #         # List all available arrays
    #         n_arrays = point_data.GetNumberOfArrays()
    #         print(f"Found {n_arrays} point data arrays:")
            
    #         for i in range(n_arrays):
    #             array = point_data.GetArray(i)
    #             print(f"  {i}: {array.GetName()} ({array.GetNumberOfTuples()} tuples, {array.GetNumberOfComponents()} components)")
            
    #         if n_arrays > 0:
    #             # Use the first array as scalars
    #             first_array = point_data.GetArray(0)
    #             point_data.SetScalars(first_array)
    #             scalar_array = first_array
    #             print(f"Using '{first_array.GetName()}' as scalar field")
    #         else:
    #             # Create a default scalar field based on distance from origin
    #             print("Creating default scalar field based on distance from Earth center...")
    #             self.create_default_scalar_field()
    #             scalar_array = self.vtk_data.GetPointData().GetScalars()
        
    #     # Ensure scalar array has a proper name
    #     if scalar_array and not scalar_array.GetName():
    #         scalar_array.SetName("flux_field")
            
    #     print(f"Scalar field setup complete: {scalar_array.GetName() if scalar_array else 'None'}")

    def create_default_scalar_field(self):
        """Create a default scalar field when none exists"""
        if not self.vtk_data:
            return
            
        n_points = self.vtk_data.GetNumberOfPoints()
        scalar_values = []
        
        earth_radius = 6371.0  # km
        
        for i in range(n_points):
            point = self.vtk_data.GetPoint(i)
            x, y, z = point
            
            # Distance from Earth center
            r = np.sqrt(x*x + y*y + z*z)
            
            # Create a simple Van Allen belt-like field
            flux = 0.0
            if r > earth_radius:
                if 1.2 * earth_radius <= r <= 6 * earth_radius:
                    flux = 1e6 * np.exp(-((r - 3*earth_radius)**2) / (earth_radius)**2)
            
            scalar_values.append(flux)
        
        # Add to VTK data
        from vtk.util.numpy_support import numpy_to_vtk
        scalar_array = numpy_to_vtk(np.array(scalar_values), deep=True)
        scalar_array.SetName("generated_flux")
        self.vtk_data.GetPointData().SetScalars(scalar_array)
        
        print(f"Created default scalar field with {len(scalar_values)} values")

    def setup_field_visualization(self):
        """Setup field visualization with automatic zoom to fit data"""
        if not self.vtk_data:
            print("ERROR: No VTK data for field visualization")
            return
            
        print("Setting up field visualization...")
        
        # Determine if we should use point cloud as default
        current_mode = self.viz_mode_combo.currentText() if hasattr(self, 'viz_mode_combo') else "Point Cloud"
        
        if current_mode == "Point Cloud":
            self.setup_point_cloud_rendering()
        else:
            # Use the existing field setup for other modes
            try:
                scalar_array = self.vtk_data.GetPointData().GetScalars()
                if not scalar_array:
                    print("ERROR: No scalar data for visualization")
                    return
                    
                scalar_range = scalar_array.GetRange()
                print(f"Scalar range: {scalar_range[0]:.2e} to {scalar_range[1]:.2e}")
                
                num_points = self.vtk_data.GetNumberOfPoints()
                print(f"Total points: {num_points}")
                
                # Create a threshold filter to only show points with significant flux
                threshold = vtk.vtkThreshold()
                threshold.SetInputData(self.vtk_data)
                threshold.SetLowerThreshold(scalar_range[1] * 0.01)
                threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
                threshold.Update()
                
                thresholded_data = threshold.GetOutput()
                print(f"Points with significant flux: {thresholded_data.GetNumberOfPoints()}")
                
                if thresholded_data.GetNumberOfPoints() == 0:
                    print("No points above threshold, using all points")
                    thresholded_data = self.vtk_data
                
                # Create mapper
                field_mapper = vtk.vtkDataSetMapper()
                field_mapper.SetInputData(thresholded_data)
                field_mapper.SetScalarRange(scalar_range)
                field_mapper.ScalarVisibilityOn()
                
                # Setup lookup table
                lut = self.create_lookup_table(self.colormap_combo.currentText())
                field_mapper.SetLookupTable(lut)
                
                # Create field actor
                self.field_actor = vtk.vtkActor()
                self.field_actor.SetMapper(field_mapper)
                self.field_actor.GetProperty().SetOpacity(0.7)
                
                self.renderer.AddActor(self.field_actor)
                
                # Setup scalar bar
                self.setup_scalar_bar(lut, scalar_array.GetName())
                
                print("Field visualization setup complete")
                
            except Exception as e:
                print(f"ERROR in setup_field_visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # AUTOMATIC ZOOM TO FIT DATA
        self.zoom_to_fit_data()

    def zoom_to_fit_data(self):
        """Automatically zoom to encompass the entire VTK field"""
        if not self.vtk_data:
            return
            
        print("Auto-zooming to fit entire VTK field...")
        
        try:
            # Get data bounds
            bounds = self.vtk_data.GetBounds()
            print(f"Data bounds: X({bounds[0]:.0f}, {bounds[1]:.0f}) Y({bounds[2]:.0f}, {bounds[3]:.0f}) Z({bounds[4]:.0f}, {bounds[5]:.0f})")
            
            # Calculate the center and size of the data
            center = [
                (bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2
            ]
            
            # Calculate the maximum extent
            x_range = bounds[1] - bounds[0]
            y_range = bounds[3] - bounds[2]
            z_range = bounds[5] - bounds[4]
            max_range = max(x_range, y_range, z_range)
            
            print(f"Data center: ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")
            print(f"Max range: {max_range:.0f} km")
            
            # Position camera to see entire data with some margin
            camera = self.renderer.GetActiveCamera()
            
            # Set focal point to data center
            camera.SetFocalPoint(center[0], center[1], center[2])
            
            # Position camera at a distance that shows all data with 20% margin
            distance = max_range * 1.5  # 1.5x for good margin
            
            # Position camera at 45-degree angle for good 3D view
            camera.SetPosition(
                center[0] + distance * 0.7,  # X offset
                center[1] + distance * 0.7,  # Y offset  
                center[2] + distance * 0.5   # Z offset (slightly above)
            )
            
            # Set up vector (Z-axis up)
            camera.SetViewUp(0, 0, 1)
            
            # Reset camera to ensure proper clipping planes
            self.renderer.ResetCamera()
            
            # Fine-tune the zoom to show everything with margin
            camera.Zoom(0.8)  # Zoom out a bit more for safety margin
            
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            
            print(f"Camera positioned at: {camera.GetPosition()}")
            print("Auto-zoom complete - entire VTK field should be visible")
            
        except Exception as e:
            print(f"Error in auto-zoom: {e}")
            # Fallback to default camera position
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(20000, 20000, 10000)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
        
    # def debug_field_actor(self):
    #     """Specific debug for field actor"""
    #     print("\n=== FIELD ACTOR DEBUG ===")
        
    #     if not hasattr(self, 'field_actor') or not self.field_actor:
    #         print("No field_actor exists!")
    #         return
            
    #     print("Field actor exists")
        
    #     # Check mapper
    #     mapper = self.field_actor.GetMapper()
    #     if mapper:
    #         print("Mapper exists")
    #         input_data = mapper.GetInput()
    #         if input_data:
    #             print(f"Mapper input: {input_data.GetNumberOfPoints()} points")
    #             scalar_range = mapper.GetScalarRange()
    #             print(f"Mapper scalar range: {scalar_range}")
    #         else:
    #             print("No input data on mapper!")
    #     else:
    #         print("No mapper on field actor!")
            
    #     # Check properties
    #     prop = self.field_actor.GetProperty()
    #     print(f"Opacity: {prop.GetOpacity()}")
    #     print(f"Point size: {prop.GetPointSize()}")
    #     print(f"Color: {prop.GetColor()}")
    #     print(f"Representation: {prop.GetRepresentation()}")
        
    #     # Check visibility
    #     print(f"Visibility: {self.field_actor.GetVisibility()}")
        
    #     print("=========================\n")
            
    def setup_scalar_bar(self, lut, scalar_name):
        """Setup scalar bar - FIXED"""
        try:
            print(f"\n=== SETTING UP SCALAR BAR ===")
            
            # Remove existing scalar bar
            if hasattr(self, 'scalar_bar') and self.scalar_bar:
                self.renderer.RemoveViewProp(self.scalar_bar)
                
            self.scalar_bar = vtk.vtkScalarBarActor()
            self.scalar_bar.SetLookupTable(lut)
            
            # Get scale mode for title
            scale_mode = self.scale_mode_combo.currentText() if hasattr(self, 'scale_mode_combo') else "Linear"
            scale_text = "LOG" if scale_mode == "Logarithmic" else "LIN"
            
            if scalar_name and "flux" in scalar_name.lower():
                title = f"Electron Flux ({scale_text})\n(particles/cm²/s)"
            else:
                title = f"{scalar_name} ({scale_text})\n(data units)"
                    
            self.scalar_bar.SetTitle(title)
            self.scalar_bar.SetPosition(0.85, 0.1)
            self.scalar_bar.SetWidth(0.12)
            self.scalar_bar.SetHeight(0.8)
            
            # Configure appearance
            self.scalar_bar.SetNumberOfLabels(6)
            self.scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
            self.scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
            self.scalar_bar.GetTitleTextProperty().SetFontSize(10)
            self.scalar_bar.GetLabelTextProperty().SetFontSize(8)
            
            # Debug the final range
            if lut:
                final_range = lut.GetRange()
                print(f"Scalar bar LUT range: {final_range}")
            
            self.renderer.AddViewProp(self.scalar_bar)
            print(f"=== SCALAR BAR SETUP COMPLETE ===\n")
            
        except Exception as e:
            print(f"Error setting up scalar bar: {e}")
            import traceback
            traceback.print_exc()

    # def update_status_with_units(self):
    #     """ENHANCED: Update status display with proper units"""
    #     if not self.orbital_path or self.current_time_index >= len(self.orbital_path):
    #         return
            
    #     current_point = self.orbital_path[self.current_time_index]
        
    #     # Calculate additional useful metrics
    #     distance_from_earth = np.sqrt(current_point.x**2 + current_point.y**2 + current_point.z**2)
    #     altitude_km = distance_from_earth - 6371  # Earth radius
        
    #     # Calculate flux with units
    #     flux = self.flux_analyzer.analyze_flux_at_point(current_point)
        
    #     # Enhanced status with better units and formatting
    #     self.status_label.setText(
    #         f"Time: {current_point.time:.2f} h | Altitude: {altitude_km:.1f} km\n"
    #         f"Position: ({current_point.x:.0f}, {current_point.y:.0f}, {current_point.z:.0f}) km\n"
    #         f"Flux: {flux:.2e} particles/s through {self.cross_section_spinbox.value():.1f} m² cross-section"
    #     )
        
    def load_orbital_data(self):
        """Load orbital CSV data - UPDATED with legend creation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Orbital Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Store the filename for the legend
            self.satellite_file_name = self.extract_filename_from_path(file_path)
            print(f"Loading orbital data from: {self.satellite_file_name}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_cols = ['time', 'x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Convert to orbital points
            self.orbital_path = []
            for _, row in df.iterrows():
                vx = row.get('vx', 0)
                vy = row.get('vy', 0) 
                vz = row.get('vz', 0)
                
                point = OrbitalPoint(
                    time=row['time'], x=row['x'], y=row['y'], z=row['z'],
                    vx=vx, vy=vy, vz=vz
                )
                self.orbital_path.append(point)
                
            # Update time slider
            self.time_slider.setMaximum(len(self.orbital_path) - 1)
            
            # Update analyzer
            self.flux_analyzer.set_orbital_data(self.orbital_path)
            
            # Create path visualization (will respect hide checkbox if it exists)
            self.create_path_visualization()
            
            # Create satellite legend
            self.create_satellite_legend()
            
            # Reset animation
            self.reset_animation()
            
            # Update status message
            status_msg = (
                f"✓ Loaded orbital data: {len(self.orbital_path)} points "
                f"over {self.orbital_path[-1].time:.2f} hours\n"
                f"Satellite: {self.satellite_file_name}\n"
                f"Use 'Hide Orbital Paths' checkbox to toggle path visibility"
            )
            self.status_label.setText(status_msg)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load orbital data:\n{str(e)}")

    def create_satellite_legend(self):
        """Create a legend showing the satellite name with colored indicator"""
        try:
            if not hasattr(self, 'satellite_file_name') or not self.satellite_file_name:
                return
                
            # Remove existing legend if it exists
            self.cleanup_satellite_legend()
                
            # Create legend as a QWidget overlay
            self.legend_widget = QWidget(self.vtk_widget)
            self.legend_widget.setGeometry(10, 80, 300, 30)  # Position below View buttons
            
            # Create horizontal layout for the legend
            legend_layout = QHBoxLayout(self.legend_widget)
            legend_layout.setContentsMargins(5, 5, 5, 5)
            legend_layout.setSpacing(8)
            
            # Create colored indicator (small colored square)
            self.legend_color_indicator = QLabel()
            self.legend_color_indicator.setFixedSize(16, 16)
            
            # Set background color to match satellite
            if hasattr(self, 'chosen_color'):
                r, g, b = self.chosen_color
                color_style = f"background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)}); border: 1px solid white; border-radius: 2px;"
            else:
                color_style = "background-color: rgb(255, 255, 255); border: 1px solid white; border-radius: 2px;"
            
            self.legend_color_indicator.setStyleSheet(color_style)
            
            # Create text label
            self.legend_text_label = QLabel(f"Satellite: {self.satellite_file_name}")
            self.legend_text_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    background-color: rgba(0, 0, 0, 120);
                    padding: 2px 6px;
                    border-radius: 3px;
                }
            """)
            
            # Add to layout
            legend_layout.addWidget(self.legend_color_indicator)
            legend_layout.addWidget(self.legend_text_label)
            legend_layout.addStretch()  # Push everything to the left
            
            # Style the container widget
            self.legend_widget.setStyleSheet("""
                QWidget {
                    background-color: rgba(40, 40, 40, 150);
                    border: 1px solid #666;
                    border-radius: 5px;
                }
            """)
            
            # Show the legend
            self.legend_widget.show()
            self.legend_widget.raise_()
            
            print(f"Created satellite legend: {self.satellite_file_name}")
            
        except Exception as e:
            print(f"Error creating satellite legend: {e}")
            import traceback
            traceback.print_exc()
            
    def update_satellite_legend(self):
        """Update the legend when satellite changes"""
        if hasattr(self, 'satellite_file_name') and self.satellite_file_name:
            self.create_satellite_legend()
            
    def cleanup_satellite_legend(self):
        """Clean up satellite legend components"""
        try:
            # Remove the widget-based legend
            if hasattr(self, 'legend_widget'):
                self.legend_widget.deleteLater()
                delattr(self, 'legend_widget')
                
            if hasattr(self, 'legend_color_indicator'):
                delattr(self, 'legend_color_indicator')
                
            if hasattr(self, 'legend_text_label'):
                delattr(self, 'legend_text_label')
                
            # Remove any old VTK-based legend actors (cleanup from previous approach)
            if hasattr(self, 'satellite_legend_actor'):
                self.renderer.RemoveActor2D(self.satellite_legend_actor)
                delattr(self, 'satellite_legend_actor')
                
            if hasattr(self, 'legend_indicator_actor'):
                self.renderer.RemoveActor(self.legend_indicator_actor)
                delattr(self, 'legend_indicator_actor')
                
            print("Cleaned up satellite legend")
            
        except Exception as e:
            print(f"Error cleaning up satellite legend: {e}")
            
    def extract_filename_from_path(self, file_path):
        """Extract just the filename from a full path"""
        try:
            from pathlib import Path
            return Path(file_path).stem  # Gets filename without extension
        except:
            # Fallback for simple extraction
            return file_path.split('/')[-1].split('\\')[-1].split('.')[0]

    def cleanup_satellite_legend(self):
        """Clean up satellite legend actors"""
        try:
            # Remove text legend
            if hasattr(self, 'satellite_legend_actor'):
                self.renderer.RemoveActor2D(self.satellite_legend_actor)
                delattr(self, 'satellite_legend_actor')
                
            # Remove color indicator
            if hasattr(self, 'legend_indicator_actor'):
                self.renderer.RemoveActor(self.legend_indicator_actor)
                delattr(self, 'legend_indicator_actor')
                
            print("Cleaned up satellite legend")
            
        except Exception as e:
            print(f"Error cleaning up satellite legend: {e}")
            
    def create_path_visualization(self):
        """Create 3D orbital path visualization with improved styling - UPDATED for visibility control"""
        if not self.orbital_path:
            return
            
        print(f"Creating path visualization with {len(self.orbital_path)} points")
            
        # Create path polyline
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for i, point in enumerate(self.orbital_path):
            points.InsertNextPoint(point.x, point.y, point.z)
            
        # Create line segments
        for i in range(len(self.orbital_path) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
            
        path_polydata = vtk.vtkPolyData()
        path_polydata.SetPoints(points)
        path_polydata.SetLines(lines)
        
        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(path_polydata)
        
        if hasattr(self, 'path_actor'):
            self.renderer.RemoveActor(self.path_actor)
            
        self.path_actor = vtk.vtkActor()
        self.path_actor.SetMapper(path_mapper)
        self.path_actor.GetProperty().SetColor(0.9, 0.9, 0.2)  # Bright yellow
        self.path_actor.GetProperty().SetLineWidth(3.0)  # Thicker line
        self.path_actor.GetProperty().SetOpacity(0.8)
        
        # Check if orbital paths should be hidden on creation
        if hasattr(self, 'hide_orbital_paths') and self.hide_orbital_paths.isChecked():
            self.path_actor.SetVisibility(False)
            print("Orbital path created but hidden due to checkbox setting")
        else:
            self.path_actor.SetVisibility(True)
            print("Orbital path created and visible")
        
        self.renderer.AddActor(self.path_actor)
        
        # Create object representation
        self.create_object_representation()
        
        # Set initial position
        if self.orbital_path:
            first_point = self.orbital_path[0]
            if hasattr(self, 'object_actor'):
                self.object_actor.SetPosition(first_point.x, first_point.y, first_point.z)
            if hasattr(self, 'satellite_border_actor'):
                self.satellite_border_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        self.vtk_widget.GetRenderWindow().Render()
        print("Path visualization created successfully")
        
    def create_object_representation(self):
        """Create satellite with DETAILED color debugging"""
        import random
        
        print("\n=== SATELLITE INFO ===")
        
        # Calculate size
        base_radius = max(self.cross_section_spinbox.value() * 100.0, 500.0)
        print(f"Satellite radius: {base_radius}")
        
        # Remove existing satellite actors
        if hasattr(self, 'object_actor'):
            self.renderer.RemoveActor(self.object_actor)
        if hasattr(self, 'satellite_border_actor'):
            self.renderer.RemoveActor(self.satellite_border_actor)
        
        # Generate color ONLY if we don't already have one (MINIMAL FIX)
        if not hasattr(self, 'chosen_color'):
            colors = [
                [1.0, 0.2, 0.2],  # Bright red
                [0.2, 1.0, 0.2],  # Bright green  
                [1.0, 0.2, 1.0],  # Bright magenta
                [0.2, 0.8, 1.0],  # Bright cyan
                [1.0, 0.8, 0.2],  # Bright orange
                [0.8, 0.2, 1.0],  # Bright purple
                [1.0, 0.5, 0.8],  # Bright pink
                [0.2, 1.0, 0.8],  # Bright teal
            ]
            self.chosen_color = random.choice(colors)
        
        print(f"Chosen color: {self.chosen_color}")
        
        # Create sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(base_radius)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.Update()
        print(f"Sphere created with {sphere.GetOutput().GetNumberOfPoints()} points")
        
        # Create mapper
        object_mapper = vtk.vtkPolyDataMapper()
        object_mapper.SetInputConnection(sphere.GetOutputPort())
        print(f"Mapper scalar visibility BEFORE: {object_mapper.GetScalarVisibility()}")
        object_mapper.ScalarVisibilityOff()
        print(f"Mapper scalar visibility AFTER: {object_mapper.GetScalarVisibility()}")
        
        # Create actor
        self.object_actor = vtk.vtkActor()
        self.object_actor.SetMapper(object_mapper)
        
        # Apply color
        prop = self.object_actor.GetProperty()
        print(f"Property color BEFORE: {prop.GetColor()}")
        prop.SetColor(self.chosen_color[0], self.chosen_color[1], self.chosen_color[2])
        print(f"Property color AFTER: {prop.GetColor()}")
        
        prop.SetAmbient(0.9)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.1)
        prop.SetOpacity(1.0)
        
        print(f"Ambient: {prop.GetAmbient()}")
        print(f"Diffuse: {prop.GetDiffuse()}")
        print(f"Opacity: {prop.GetOpacity()}")
        
        # Add to renderer
        self.renderer.AddActor(self.object_actor)
        print("Actor added to renderer")
        
        # Check if actor is in renderer
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor_count = 0
        found_our_actor = False
        while True:
            actor = actors.GetNextActor()
            if not actor:
                break
            actor_count += 1
            if actor == self.object_actor:
                found_our_actor = True
                print(f"Found our actor in renderer at position {actor_count}")
                actor_color = actor.GetProperty().GetColor()
                print(f"Actor color in renderer: {actor_color}")
        
        print(f"Total actors in renderer: {actor_count}")
        print(f"Our actor found: {found_our_actor}")
        
        # Check visibility and lighting
        print(f"Actor visibility: {self.object_actor.GetVisibility()}")
        print(f"Actor pickable: {self.object_actor.GetPickable()}")
        
        # Check position
        pos = self.object_actor.GetPosition()
        print(f"Actor position: {pos}")
        
        # Check bounds
        bounds = self.object_actor.GetBounds()
        print(f"Actor bounds: {bounds}")
        
        # Check renderer lights
        lights = self.renderer.GetLights()
        print(f"Renderer has {lights.GetNumberOfItems()} lights")
        
        # Check camera position
        camera = self.renderer.GetActiveCamera()
        cam_pos = camera.GetPosition()
        cam_focal = camera.GetFocalPoint()
        print(f"Camera position: {cam_pos}")
        print(f"Camera focal point: {cam_focal}")
        
        # Check if satellite is at origin (might be hidden by Earth)
        if abs(pos[0]) < 10 and abs(pos[1]) < 10 and abs(pos[2]) < 10:
            print("WARNING: Satellite is at origin - might be hidden inside Earth!")
        
        # Update satellite legend with new color
        self.update_satellite_legend()
        
        print("=== SATELLITE INFO COMPLETE ===\n")
        
    def create_satellite_trail(self):
        """Create fading and tapering trail with multiple segments"""
        if len(self.trail_points) < 2:
            return
            
        # Remove existing trail actors
        if hasattr(self, 'trail_actors'):
            for actor in self.trail_actors:
                self.renderer.RemoveActor(actor)
        self.trail_actors = []
        
        try:
            # Create individual segments with fading and tapering
            for i in range(len(self.trail_points) - 1):
                # Create single segment
                points = vtk.vtkPoints()
                points.InsertNextPoint(self.trail_points[i])
                points.InsertNextPoint(self.trail_points[i + 1])
                
                lines = vtk.vtkCellArray()
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, 0)
                line.GetPointIds().SetId(1, 1)
                lines.InsertNextCell(line)
                
                segment_polydata = vtk.vtkPolyData()
                segment_polydata.SetPoints(points)
                segment_polydata.SetLines(lines)
                
                # Create tube for this segment
                tube_filter = vtk.vtkTubeFilter()
                tube_filter.SetInputData(segment_polydata)
                
                # Calculate position in trail (0 = oldest, 1 = newest)
                trail_position = i / (len(self.trail_points) - 1)
                
                # Tapering: thicker at newest end, thinner at oldest end
                min_radius = 50.0
                max_radius = 300.0
                radius = min_radius + (max_radius - min_radius) * trail_position
                
                tube_filter.SetRadius(radius)
                tube_filter.SetNumberOfSides(8)
                tube_filter.Update()
                
                # Create mapper and actor
                segment_mapper = vtk.vtkPolyDataMapper()
                segment_mapper.SetInputConnection(tube_filter.GetOutputPort())
                
                segment_actor = vtk.vtkActor()
                segment_actor.SetMapper(segment_mapper)
                
                # Fading: more transparent at older end, more opaque at newer end
                min_opacity = 0.1
                max_opacity = 0.9
                opacity = min_opacity + (max_opacity - min_opacity) * trail_position
                
                # Set properties
                prop = segment_actor.GetProperty()
                prop.SetColor(1.0, 1.0, 1.0)  # White
                prop.SetOpacity(opacity)
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
                
                self.renderer.AddActor(segment_actor)
                self.trail_actors.append(segment_actor)
            
        except Exception as e:
            # Fallback to simple line trail
            self.create_simple_trail_fallback()

    def create_simple_trail_fallback(self):
        """Fallback simple trail if tube filter fails"""
        try:
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            for i, point in enumerate(self.trail_points):
                points.InsertNextPoint(point[0], point[1], point[2])
            
            for i in range(len(self.trail_points) - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)
            
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            trail_actor = vtk.vtkActor()
            trail_actor.SetMapper(mapper)
            trail_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            trail_actor.GetProperty().SetLineWidth(20.0)
            trail_actor.GetProperty().SetOpacity(0.8)
            
            self.renderer.AddActor(trail_actor)
            self.trail_actors = [trail_actor]
            
        except Exception as e:
            print(f"Trail creation failed: {e}")

    def update_visualization(self):
        """Update visualization - no debug output"""
        if not self.orbital_path or self.current_time_index >= len(self.orbital_path):
            return
            
        current_point = self.orbital_path[self.current_time_index]
        satellite_position = [current_point.x, current_point.y, current_point.z]
        
        # Update satellite position
        if hasattr(self, 'object_actor'):
            self.object_actor.SetPosition(*satellite_position)
            
        if hasattr(self, 'satellite_border_actor'):
            self.satellite_border_actor.SetPosition(*satellite_position)
        
        # Initialize trail system if needed
        if not hasattr(self, 'trail_points'):
            self.trail_points = []
            self.max_trail_length = 15
        
        # Add current position to trail
        self.trail_points.append([current_point.x, current_point.y, current_point.z])
        
        # Limit trail length
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        # Create trail if we have enough points
        if len(self.trail_points) >= 2:
            self.create_satellite_trail()
        
        # Update UI elements
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_time_index)
        self.time_slider.blockSignals(False)
        
        hours = current_point.time
        h = int(hours)
        m = int((hours - h) * 60)
        s = int(((hours - h) * 60 - m) * 60)
        self.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        
        flux = self.flux_analyzer.analyze_flux_at_point(current_point)
        distance_from_earth = np.sqrt(current_point.x**2 + current_point.y**2 + current_point.z**2)
        altitude = distance_from_earth - 6371
        
        self.status_label.setText(
            f"Time: {current_point.time:.2f}h | Alt: {altitude:.1f}km\n"
            f"Pos: ({current_point.x:.0f}, {current_point.y:.0f}, {current_point.z:.0f}) km\n"
            f"Flux: {flux:.2e} particles/s"
        )
        
        # Force render
        self.vtk_widget.GetRenderWindow().Render()

    def animation_step(self):
        """Perform one animation step with wraparound - no debug output"""
        if not self.orbital_path:
            return
            
        if self.current_time_index >= len(self.orbital_path) - 1:
            # Loop back to beginning for continuous animation
            self.current_time_index = 0
            # Clear trail for new orbit
            if hasattr(self, 'trail_points'):
                self.trail_points.clear()
        else:
            self.current_time_index += 1
            
        self.update_visualization()
        self.update_plots()

    def start_animation(self):
        """Start animation - clean version"""
        if not self.orbital_path or not self.vtk_data:
            QMessageBox.warning(self, "Warning", "Please load both VTK data and orbital path first.")
            return
        
        # Clear trail system
        if hasattr(self, 'trail_points'):
            self.trail_points.clear()
        else:
            self.trail_points = []
            
        if hasattr(self, 'trail_actors'):
            for actor in self.trail_actors:
                self.renderer.RemoveActor(actor)
            self.trail_actors = []
        elif hasattr(self, 'trail_actor'):
            self.renderer.RemoveActor(self.trail_actor)
            delattr(self, 'trail_actor')
        
        # Start animation
        self.is_playing = True
        self.animation_timer.start(self.speed_spinbox.value())
        
        # Update UI
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def reset_animation(self):
        """Reset animation - ENHANCED for multiple trail actors and legend cleanup"""
        print("\n=== RESETTING ANIMATION ===")
        
        self.current_time_index = 0
        
        # Clear trail points
        if hasattr(self, 'trail_points'):
            self.trail_points.clear()
            print("Cleared trail points")
            
        # Remove all trail actors
        if hasattr(self, 'trail_actors'):
            for actor in self.trail_actors:
                self.renderer.RemoveActor(actor)
            self.trail_actors = []
            print("Removed all trail actors")
        elif hasattr(self, 'trail_actor'):
            # Handle old single actor case
            self.renderer.RemoveActor(self.trail_actor)
            delattr(self, 'trail_actor')
            print("Removed single trail actor")
            
        # Clear flux data
        if self.flux_time_window:
            self.flux_time_window.clear_data()
            
        # Force new satellite color and recreate
        if hasattr(self, 'chosen_color'):
            delattr(self, 'chosen_color')
            
        if hasattr(self, 'orbital_path') and self.orbital_path:
            self.create_object_representation()
            first_point = self.orbital_path[0]
            if hasattr(self, 'object_actor'):
                self.object_actor.SetPosition(first_point.x, first_point.y, first_point.z)
            if hasattr(self, 'satellite_border_actor'):
                self.satellite_border_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        self.update_visualization()
        print("=== RESET COMPLETE with new satellite color ===\n")

    def jump_to_time(self, time_index):
        """Jump to specific time index with trail reset for large jumps"""
        if 0 <= time_index < len(self.orbital_path):
            old_index = self.current_time_index
            self.current_time_index = time_index
            
            # If jumping backwards or far forward, reset trail
            if time_index < old_index or abs(time_index - old_index) > 10:
                if hasattr(self, 'trail_points'):
                    self.trail_points.clear()
                if hasattr(self, 'trail_actors'):
                    for actor in self.trail_actors:
                        self.renderer.RemoveActor(actor)
                    self.trail_actors = []
            
            self.update_visualization()
            self.update_plots()

    def update_cross_section(self, radius):
        """Update object cross section and visual representation - UPDATED for new satellite design"""
        self.flux_analyzer.set_cross_section(radius)
        
        # Recreate satellite with new size
        if hasattr(self, 'object_actor') or hasattr(self, 'satellite_border_actor'):
            # Store current position if we have one
            current_position = None
            if hasattr(self, 'object_actor') and self.object_actor:
                current_position = self.object_actor.GetPosition()
            
            # Recreate the satellite representation
            self.create_object_representation()
            
            # Restore position if we had one
            if current_position and self.orbital_path and 0 <= self.current_time_index < len(self.orbital_path):
                current_point = self.orbital_path[self.current_time_index]
                satellite_position = [current_point.x, current_point.y, current_point.z]
                
                if hasattr(self, 'object_actor'):
                    self.object_actor.SetPosition(*satellite_position)
                if hasattr(self, 'satellite_border_actor'):
                    self.satellite_border_actor.SetPosition(*satellite_position)
                
            self.vtk_widget.GetRenderWindow().Render()
            
        print(f"Updated satellite cross-section radius to {radius:.1f} m")

    def create_path_visualization(self):
        """Create 3D orbital path visualization with improved styling"""
        if not self.orbital_path:
            return
            
        print(f"Creating path visualization with {len(self.orbital_path)} points")
            
        # Create path polyline
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for i, point in enumerate(self.orbital_path):
            points.InsertNextPoint(point.x, point.y, point.z)
            
        # Create line segments
        for i in range(len(self.orbital_path) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
            
        path_polydata = vtk.vtkPolyData()
        path_polydata.SetPoints(points)
        path_polydata.SetLines(lines)
        
        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(path_polydata)
        
        if hasattr(self, 'path_actor'):
            self.renderer.RemoveActor(self.path_actor)
            
        self.path_actor = vtk.vtkActor()
        self.path_actor.SetMapper(path_mapper)
        self.path_actor.GetProperty().SetColor(0.9, 0.9, 0.2)  # Bright yellow
        self.path_actor.GetProperty().SetLineWidth(3.0)  # Thicker line
        self.path_actor.GetProperty().SetOpacity(0.8)
        
        self.renderer.AddActor(self.path_actor)
        
        # Create object representation
        self.create_object_representation()
        
        # Set initial position
        if self.orbital_path:
            first_point = self.orbital_path[0]
            if hasattr(self, 'object_actor'):
                self.object_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        self.vtk_widget.GetRenderWindow().Render()
        print("Path visualization created successfully")
        
    def start_animation(self):
        """Start the orbital animation with trail debugging"""
        if not self.orbital_path:
            QMessageBox.warning(self, "Warning", "Please load orbital path data first.")
            return
            
        if not self.vtk_data:
            QMessageBox.warning(self, "Warning", "Please load VTK flux data first.")
            return
            
        print(f"Starting animation with {len(self.orbital_path)} orbital points")
        print(f"Current time index: {self.current_time_index}")
        print(f"Animation speed: {self.speed_spinbox.value()} ms")
        
        # Clear trail on start
        if hasattr(self, 'trail_points'):
            self.trail_points.clear()
            print("Cleared trail on animation start")
        
        self.is_playing = True
        self.animation_timer.start(self.speed_spinbox.value())
        
        # Show plot windows if they exist
        if self.slice_window:
            self.slice_window.show()
        if self.spectrum_window:
            self.spectrum_window.show()
        if self.flux_time_window:
            self.flux_time_window.show()
            
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        # Force an immediate update
        self.update_visualization()
        
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
        self.reset_animation()
        
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
    def animation_step(self):
        """Perform one animation step with wraparound"""
        if not self.orbital_path:
            return
            
        if self.current_time_index >= len(self.orbital_path) - 1:
            self.current_time_index = 0
            if hasattr(self, 'trail_points'):
                self.trail_points.clear()
        else:
            self.current_time_index += 1
            
        self.update_visualization()
        self.update_plots()
        
    def update_visualization(self):
        """Update visualization"""
        if not self.orbital_path or self.current_time_index >= len(self.orbital_path):
            return
            
        current_point = self.orbital_path[self.current_time_index]
        satellite_position = [current_point.x, current_point.y, current_point.z]
        
        # Update satellite position
        if hasattr(self, 'object_actor'):
            self.object_actor.SetPosition(*satellite_position)
            
        if hasattr(self, 'satellite_border_actor'):
            self.satellite_border_actor.SetPosition(*satellite_position)
        
        # Initialize trail system if needed
        if not hasattr(self, 'trail_points'):
            self.trail_points = []
            self.max_trail_length = 15
        
        # Add current position to trail
        self.trail_points.append([current_point.x, current_point.y, current_point.z])
        
        # Limit trail length
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        # Create trail if we have enough points
        if len(self.trail_points) >= 2:
            self.create_satellite_trail()
        
        # Update UI elements
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_time_index)
        self.time_slider.blockSignals(False)
        
        hours = current_point.time
        h = int(hours)
        m = int((hours - h) * 60)
        s = int(((hours - h) * 60 - m) * 60)
        self.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        
        flux = self.flux_analyzer.analyze_flux_at_point(current_point)
        distance_from_earth = np.sqrt(current_point.x**2 + current_point.y**2 + current_point.z**2)
        altitude = distance_from_earth - 6371
        
        self.status_label.setText(
            f"Time: {current_point.time:.2f}h | Alt: {altitude:.1f}km\n"
            f"Pos: ({current_point.x:.0f}, {current_point.y:.0f}, {current_point.z:.0f}) km\n"
            f"Flux: {flux:.2e} particles/s"
        )
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def update_plots(self):
        """Update all plot windows with real-time satellite data"""
        if not self.orbital_path or self.current_time_index >= len(self.orbital_path):
            return
            
        current_point = self.orbital_path[self.current_time_index]
        
        # Update slice plot
        if self.slice_window and self.vtk_data:
            self.slice_window.update_slice(current_point.phi, self.vtk_data)
            self.slice_window.set_object_position(current_point.x, current_point.y, current_point.z)
            
        # Update spectrum plot with real satellite data
        if self.spectrum_window:
            self.spectrum_window.update_spectrum_from_satellite_position()
            
        # Update flux time plot
        if self.flux_time_window:
            flux = self.flux_analyzer.analyze_flux_at_point(current_point)
            self.flux_time_window.add_flux_point(current_point.time, flux)
            
    def jump_to_time(self, time_index):
        """Jump to specific time index"""
        if 0 <= time_index < len(self.orbital_path):
            self.current_time_index = time_index
            self.update_visualization()
            self.update_plots()
            
    def set_animation_speed(self, speed_ms):
        """Set animation speed"""
        if self.is_playing:
            self.animation_timer.setInterval(speed_ms)
            
    def reset_animation(self):
        """Reset animation"""
        self.current_time_index = 0
        
        # Clear trail points
        if hasattr(self, 'trail_points'):
            self.trail_points.clear()
            
        # Remove all trail actors
        if hasattr(self, 'trail_actors'):
            for actor in self.trail_actors:
                self.renderer.RemoveActor(actor)
            self.trail_actors = []
        elif hasattr(self, 'trail_actor'):
            self.renderer.RemoveActor(self.trail_actor)
            delattr(self, 'trail_actor')
            
        # Clear flux data
        if self.flux_time_window:
            self.flux_time_window.clear_data()
            
        # Force new satellite color and recreate
        if hasattr(self, 'satellite_color'):
            delattr(self, 'satellite_color')
            
        if hasattr(self, 'orbital_path') and self.orbital_path:
            self.create_object_representation()
            first_point = self.orbital_path[0]
            if hasattr(self, 'object_actor'):
                self.object_actor.SetPosition(first_point.x, first_point.y, first_point.z)
            if hasattr(self, 'satellite_border_actor'):
                self.satellite_border_actor.SetPosition(first_point.x, first_point.y, first_point.z)
        
        self.update_visualization()
        
    def show_slice_window(self):
        """Show the slice plot window"""
        if not self.slice_window:
            self.slice_window = SlicePlotWindow(self)
        self.slice_window.show()
        self.slice_window.raise_()
        
    def show_spectrum_window(self):
        """Show the spectrum plot window"""
        if not self.spectrum_window:
            self.spectrum_window = SpectrumPlotWindow(self)
        self.spectrum_window.show()
        self.spectrum_window.raise_()
        
    def show_flux_time_window(self):
        """Show the flux vs time window"""
        if not self.flux_time_window:
            self.flux_time_window = FluxTimePlotWindow(self)
        self.flux_time_window.show()
        self.flux_time_window.raise_()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Electron Flux Visualizer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Scientific Visualization Lab")
    
    # Create and show main window
    window = ElectronFluxVisualizer()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
