"""
Spectrum analysis plot windows for STRATOS
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

# Import matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import Config
from core import OrbitalPoint


class BaseSpectrumWindow(QMainWindow):
    """Base class for spectrum analysis windows"""
    
    window_closed = pyqtSignal(str)  # Emit window ID when closed
    
    def __init__(self, window_id, title, satellite_file_path, satellite_name, config):
        super().__init__()
        
        self.window_id = window_id
        self.satellite_file_path = satellite_file_path
        self.satellite_name = satellite_name
        self.config = config
        
        # Data
        self.orbital_data = []
        self.flux_data = None
        self.current_time_index = 0
        
        # No individual animation - controlled by main window
        
        self._setup_ui(title)
        
    def _setup_ui(self, title):
        """Setup the window UI"""
        self.setWindowTitle(f"{title} - {self.satellite_name}")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Plot area (no control panel)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Bottom status bar only
        self.status_label = QLabel(f"Tracking: {self.satellite_name} - Ready")
        self.status_label.setStyleSheet("""
            padding: 5px;
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #555;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        """)
        layout.addWidget(self.status_label)
        
    
    def set_orbital_data(self, orbital_data):
        """Set the orbital data"""
        self.orbital_data = orbital_data
        self.current_time_index = 0
        # Initialize axis limits if the window has this method
        if hasattr(self, '_initialize_axis_limits'):
            self._initialize_axis_limits()
        self._update_plot()
        
    def set_flux_data(self, flux_data):
        """Set the flux data"""
        self.flux_data = flux_data
        self._update_plot()
        
    def set_current_time_index(self, index):
        """Set the current time index"""
        if 0 <= index < len(self.orbital_data):
            self.current_time_index = index
            self._update_plot()
    
    def _update_plot(self):
        """Update the plot - override in subclasses"""
        if not self.orbital_data:
            return
            
        # Update status
        if self.current_time_index < len(self.orbital_data):
            current_time = self.orbital_data[self.current_time_index].time
            self.status_label.setText(f"Time: {current_time:.2f} hours")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.window_closed.emit(self.window_id)
        event.accept()


class EnergySpectrumWindow(BaseSpectrumWindow):
    """Energy spectrum analysis window"""
    
    def __init__(self, satellite_file_path, satellite_name, config):
        # Energy spectrum parameters - initialize BEFORE calling super()
        self.energy_range = (10, 10000)  # keV
        self.energy_bins = 50
        
        window_id = f"energy_spectrum_{satellite_file_path}"
        super().__init__(window_id, "Energy Spectrum", satellite_file_path, satellite_name, config)
        
        # Initialize axis limits
        self._initialize_axis_limits()
        
    
    def set_energy_parameters(self, energy_range, energy_bins):
        """Set energy parameters from main analysis panel"""
        self.energy_range = energy_range
        self.energy_bins = energy_bins
        self._initialize_axis_limits()  # Reinitialize limits when parameters change
        self._update_plot()
    
    def _initialize_axis_limits(self):
        """Initialize fixed axis limits for energy spectrum"""
        self.x_limits = (self.energy_range[0], self.energy_range[1])
        self.y_limits = (1e-2, 1e10)  # Fixed y-axis range for flux
    
    def _update_plot(self):
        """Update the energy spectrum plot"""
        super()._update_plot()
        
        if not self.orbital_data:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Generate energy spectrum based on altitude (Van Allen belt characteristics)
        if self.current_time_index < len(self.orbital_data):
            current_point = self.orbital_data[self.current_time_index]
            altitude = current_point.altitude
            
            # Generate energy spectrum based on altitude and config parameters
            energies = np.logspace(np.log10(self.energy_range[0]), 
                                  np.log10(self.energy_range[1]), 
                                  self.energy_bins)
            
            # Spectral characteristics by altitude region (from config)
            if altitude < self.config.LEO_ALTITUDE_MAX:
                spectral_index = self.config.LEO_SPECTRAL_INDEX
                char_energy = self.config.LEO_CHARACTERISTIC_ENERGY
            elif altitude < self.config.INNER_BELT_ALTITUDE_MAX:
                spectral_index = self.config.INNER_BELT_SPECTRAL_INDEX
                char_energy = self.config.INNER_BELT_CHARACTERISTIC_ENERGY
            elif altitude < self.config.PEAK_BELT_ALTITUDE_MAX:
                spectral_index = self.config.PEAK_BELT_SPECTRAL_INDEX
                char_energy = self.config.PEAK_BELT_CHARACTERISTIC_ENERGY
            else:
                spectral_index = self.config.OUTER_BELT_SPECTRAL_INDEX
                char_energy = self.config.OUTER_BELT_CHARACTERISTIC_ENERGY
            
            # Power law spectrum with exponential cutoff
            normalization = 1e6 * (altitude / 1000.0)  # Scale with altitude
            spectrum = normalization * np.power(energies / char_energy, spectral_index) * \
                      np.exp(-energies / (char_energy * 10))
            
            # Add some flux data influence if available
            if hasattr(self.flux_data, 'GetPointData') and self.flux_data.GetPointData().GetScalars():
                # Sample flux at current position
                from vtk import vtkProbeFilter
                import vtk
                
                probe = vtkProbeFilter()
                points = vtk.vtkPoints()
                points.InsertNextPoint(current_point.x, current_point.y, current_point.z)
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                
                probe.SetInputData(polydata)
                probe.SetSourceData(self.flux_data)
                probe.Update()
                
                result = probe.GetOutput()
                if result.GetNumberOfPoints() > 0:
                    scalar_array = result.GetPointData().GetScalars()
                    if scalar_array and scalar_array.GetNumberOfTuples() > 0:
                        flux_value = scalar_array.GetValue(0)
                        spectrum *= max(0.01, flux_value / 1e6)  # Scale spectrum by local flux
            
            # Plot spectrum
            ax.loglog(energies, spectrum, 'b-', linewidth=2, label=f'Alt: {altitude:.0f} km')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel('Differential Flux (particles/cm²/s/keV)')
            ax.set_title(f'Energy Spectrum - Time: {current_point.time:.2f} h, Alt: {altitude:.0f} km')
            
            # Apply fixed axis limits
            if self.x_limits and self.y_limits:
                ax.set_xlim(self.x_limits)
                ax.set_ylim(self.y_limits)
            
            ax.grid(True, alpha=0.3)
            
            # Add vertical line for characteristic energy
            ax.axvline(char_energy, color='r', linestyle='--', alpha=0.7, 
                      label=f'Char. Energy: {char_energy} keV')
            ax.legend()
        
        self.canvas.draw()


class PitchAngleWindow(BaseSpectrumWindow):
    """Pitch angle distribution window"""
    
    def __init__(self, satellite_file_path, satellite_name, config):
        window_id = f"pitch_angle_{satellite_file_path}"
        super().__init__(window_id, "Pitch Angle Distribution", satellite_file_path, satellite_name, config)
        # Initialize default axis limits
        self.x_limits = (0, 180)
        self.y_limits = (0, 3.0)
    
    def _initialize_axis_limits(self):
        """Initialize fixed axis limits for pitch angle distribution"""
        self.x_limits = (0, 180)
        self.y_limits = (0, 3.0)  # Fixed y-axis range for relative flux
    
    def _update_plot(self):
        """Update the pitch angle distribution plot"""
        super()._update_plot()
        
        if not self.orbital_data:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.current_time_index < len(self.orbital_data):
            current_point = self.orbital_data[self.current_time_index]
            
            # Generate pitch angle distribution (simplified model)
            pitch_angles = np.linspace(0, 180, 37)  # 5-degree bins
            
            # Model based on magnetic field geometry and particle trapping
            # Peaked distribution at 90 degrees (equatorially trapped particles)
            distribution = np.exp(-0.5 * ((pitch_angles - 90) / 30)**2) + \
                         0.3 * np.exp(-0.5 * ((pitch_angles - 0) / 15)**2) + \
                         0.3 * np.exp(-0.5 * ((pitch_angles - 180) / 15)**2)
            
            # Modulate by altitude - more trapped particles at higher altitudes
            altitude = current_point.altitude
            trapped_factor = 1.0 + 0.5 * min(1.0, altitude / 10000.0)  # More trapped at high alt
            distribution *= trapped_factor
            
            # Plot distribution
            ax.plot(pitch_angles, distribution, 'g-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Pitch Angle (degrees)')
            ax.set_ylabel('Relative Flux')
            ax.set_title(f'Pitch Angle Distribution - Time: {current_point.time:.2f} h, Alt: {altitude:.0f} km')
            
            # Apply fixed axis limits
            if self.x_limits and self.y_limits:
                ax.set_xlim(self.x_limits)
                ax.set_ylim(self.y_limits)
            
            ax.grid(True, alpha=0.3)
            
            # Add markers for special angles
            ax.axvline(90, color='r', linestyle='--', alpha=0.7, label='90° (Equatorial)')
            ax.axvline(0, color='orange', linestyle='--', alpha=0.7, label='0° (Field-aligned)')
            ax.axvline(180, color='orange', linestyle='--', alpha=0.7, label='180° (Field-aligned)')
            ax.legend()
        
        self.canvas.draw()


class PhaseSpaceWindow(BaseSpectrumWindow):
    """Phase space plot window"""
    
    def __init__(self, satellite_file_path, satellite_name, config):
        window_id = f"phase_space_{satellite_file_path}"
        super().__init__(window_id, "Phase Space Plot", satellite_file_path, satellite_name, config)
        # Initialize default axis limits
        self.x_limits = (6000, 50000)
        self.y_limits = (0, 12)
    
    def _initialize_axis_limits(self):
        """Initialize fixed axis limits for phase space plot"""
        if self.orbital_data:
            # Calculate position and velocity ranges for fixed limits
            positions = []
            velocities = []
            
            for point in self.orbital_data:
                r = np.sqrt(point.x**2 + point.y**2 + point.z**2)
                positions.append(r)
                
                if hasattr(point, 'vx') and hasattr(point, 'vy') and hasattr(point, 'vz'):
                    v = np.sqrt(point.vx**2 + point.vy**2 + point.vz**2)
                else:
                    v = 7500  # Typical orbital velocity (m/s)
                velocities.append(v)
            
            self.x_limits = (6000, max(positions) / 1000 * 1.1)  # Earth surface to max + 10%
            self.y_limits = (0, max(velocities) / 1000 * 1.2)  # 0 to max velocity + 20%
        else:
            # Default limits
            self.x_limits = (6000, 50000)
            self.y_limits = (0, 12)
    
    def _update_plot(self):
        """Update the phase space plot"""
        super()._update_plot()
        
        if not self.orbital_data or len(self.orbital_data) < 2:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract position and velocity data for phase space plot
        positions = []
        velocities = []
        
        for point in self.orbital_data:
            # Radial distance from Earth center
            r = np.sqrt(point.x**2 + point.y**2 + point.z**2)
            positions.append(r)
            
            # Calculate velocity magnitude if velocity data available
            if hasattr(point, 'vx') and hasattr(point, 'vy') and hasattr(point, 'vz'):
                v = np.sqrt(point.vx**2 + point.vy**2 + point.vz**2)
            else:
                # Estimate velocity from position changes (numerical derivative)
                if len(positions) > 1:
                    dt = 3600  # Assume 1-hour timesteps (convert to seconds)
                    dr = positions[-1] - positions[-2]
                    v = abs(dr) / dt * 1000  # Convert km/hr to m/s
                else:
                    v = 7500  # Typical orbital velocity (m/s)
            velocities.append(v)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Create phase space plot (position vs velocity) - only plot up to current time
        current_positions = positions[:self.current_time_index + 1] / 1000
        current_velocities = velocities[:self.current_time_index + 1] / 1000
        
        # Plot trail up to current position
        if len(current_positions) > 1:
            ax.plot(current_positions, current_velocities, 'b-', linewidth=1, alpha=0.7)
            ax.scatter(current_positions[:-1], current_velocities[:-1], 
                      c=np.arange(len(current_positions)-1), cmap='viridis', s=15, alpha=0.6)
        
        # Highlight current position
        if self.current_time_index < len(positions):
            current_r = positions[self.current_time_index] / 1000
            current_v = velocities[self.current_time_index] / 1000
            ax.scatter(current_r, current_v, c='red', s=100, marker='*', 
                      label=f'Current (t={self.orbital_data[self.current_time_index].time:.1f}h)')
        
        ax.set_xlabel('Radial Distance (km)')
        ax.set_ylabel('Speed (km/s)')
        ax.set_title('Phase Space Plot (Position vs Velocity)')
        
        # Apply fixed axis limits
        if self.x_limits and self.y_limits:
            ax.set_xlim(self.x_limits)
            ax.set_ylim(self.y_limits)
        
        ax.grid(True, alpha=0.3)
        
        # Add Earth radius reference
        earth_radius = 6371  # km
        ax.axvline(earth_radius, color='brown', linestyle='--', alpha=0.5, label='Earth Surface')
        ax.legend()
        
        self.canvas.draw()