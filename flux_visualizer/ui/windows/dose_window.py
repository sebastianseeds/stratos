"""
Dose calculator window for STRATOS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt


class DoseWindow(QDialog):
    """Window for displaying dynamic dose calculations"""
    
    # Signals
    window_closed = pyqtSignal(str)  # Emit window ID when closed
    
    def __init__(self, satellite_file_path, satellite_name, config, parent=None):
        """
        Initialize dose window
        
        Args:
            satellite_file_path: Path to satellite file
            satellite_name: Display name for satellite
            config: Application configuration
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.satellite_file_path = satellite_file_path
        self.satellite_name = satellite_name
        self.config = config
        self.current_time_index = 0
        self.dose_data = None
        self.flux_types = []
        self.flux_contributions = []
        
        # Time indicator elements (will be created when data is set)
        self.time_line1 = None
        self.time_line2 = None
        self.time_marker1 = None
        self.time_marker2 = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the window UI"""
        self.setWindowTitle(f"Dose Calculator - {self.satellite_name}")
        self.setMinimumSize(800, 600)
        
        self.layout = QVBoxLayout(self)
        
        # Matplotlib figure will be added when data is set
        self.canvas = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
    
    def set_dose_data(self, dose_data, flux_types=None, flux_contributions=None):
        """Set dose data and create the plot"""
        self.dose_data = dose_data
        self.flux_types = flux_types or []
        self.flux_contributions = flux_contributions or []
        self._create_plot()
    
    def _create_plot(self):
        """Create the matplotlib plot"""
        if self.dose_data is None:
            return
        
        # Clear existing plot if any
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Create title based on flux types
        if len(self.flux_types) > 1:
            flux_type_text = f" ({', '.join(self.flux_types)})"
            title = f'Combined Radiation Dose Along Orbital Path{flux_type_text}'
        elif len(self.flux_types) == 1:
            title = f'{self.flux_types[0]} Radiation Dose Along Orbital Path'
        else:
            title = 'Radiation Dose Along Orbital Path'
        
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        times = self.dose_data['times']
        dose_rates = self.dose_data['dose_rates_mGy_per_s']
        cumulative_dose = self.dose_data['cumulative_dose_mGy']
        
        # Plot dose rate
        self.ax1.plot(times, dose_rates, 'b-', linewidth=2, label='Dose Rate')
        self.ax1.set_ylabel('Dose Rate (mGy/s)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Instantaneous Dose Rate')
        
        # Plot cumulative dose
        self.ax2.plot(times, cumulative_dose, 'r-', linewidth=2, label='Cumulative Dose')
        self.ax2.set_xlabel('Time (hours)', fontsize=12)
        self.ax2.set_ylabel('Cumulative Dose (mGy)', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Cumulative Radiation Dose')
        
        # Set x-axis limits to known orbital time range
        time_min, time_max = self.dose_data['time_range_hours']
        self.ax1.set_xlim(time_min, time_max)
        self.ax2.set_xlim(time_min, time_max)
        
        # Initial time indicators (will be updated dynamically)
        self._update_time_indicators()
        
        plt.tight_layout()
        
        # Add to dialog
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        # Info label - at bottom with better styling
        if len(self.flux_types) > 1:
            flux_info = f"Combined flux from {len(self.flux_types)} sources: {', '.join(self.flux_types)}"
        elif len(self.flux_types) == 1:
            flux_info = f"Flux source: {self.flux_types[0]}"
        else:
            flux_info = "Single flux source"
        
        info_text = (f"{flux_info}  |  "
                    f"Cross Section: {self.dose_data['cross_section_m2']:.3f} m²  |  "
                    f"Particle Energy: {self.dose_data['particle_energy_MeV']:.1f} MeV  |  "
                    f"Time Range: {self.dose_data['time_range_hours'][0]:.1f}h to {self.dose_data['time_range_hours'][1]:.1f}h")
        self.info_label = QLabel(info_text)
        self.info_label.setStyleSheet("""
            padding: 8px; 
            background-color: #2b2b2b; 
            color: #ffffff; 
            border: 1px solid #555; 
            border-radius: 4px; 
            font-weight: bold;
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.info_label)
    
    def set_current_time_index(self, time_index):
        """Update the current time index and refresh indicators"""
        self.current_time_index = time_index
        self._update_time_indicators()
    
    def _update_time_indicators(self):
        """Update the time indicator lines and markers"""
        if self.dose_data is None or self.ax1 is None or self.ax2 is None:
            return
        
        times = self.dose_data['times']
        dose_rates = self.dose_data['dose_rates_mGy_per_s']
        cumulative_dose = self.dose_data['cumulative_dose_mGy']
        
        # Remove existing indicators
        if self.time_line1:
            self.time_line1.remove()
        if self.time_line2:
            self.time_line2.remove()
        if self.time_marker1:
            self.time_marker1.remove()
        if self.time_marker2:
            self.time_marker2.remove()
        
        # Get current time
        if self.current_time_index < len(times):
            current_time = times[self.current_time_index]
            current_dose_rate = dose_rates[self.current_time_index]
            current_cum_dose = cumulative_dose[self.current_time_index]
            
            # Add new vertical lines
            self.time_line1 = self.ax1.axvline(x=current_time, color='red', linestyle='--', 
                                             linewidth=2, alpha=0.7, label=f'Current Time: {current_time:.1f}h')
            self.time_line2 = self.ax2.axvline(x=current_time, color='red', linestyle='--', 
                                             linewidth=2, alpha=0.7, label=f'Current Time: {current_time:.1f}h')
            
            # Add new markers
            self.time_marker1 = self.ax1.plot(current_time, current_dose_rate, 'ro', markersize=8, 
                                            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
                                            label=f'{current_dose_rate:.2e} mGy/s')[0]
            self.time_marker2 = self.ax2.plot(current_time, current_cum_dose, 'ro', markersize=8,
                                            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
                                            label=f'{current_cum_dose:.2f} mGy')[0]
            
            # Update legends
            self.ax1.legend()
            self.ax2.legend()
            
            # Refresh canvas if it exists
            if self.canvas:
                self.canvas.draw()
        
    def update_dose_data(self, dose_data, flux_types=None, flux_contributions=None):
        """Update with new dose data"""
        self.dose_data = dose_data
        self.flux_types = flux_types or []
        self.flux_contributions = flux_contributions or []
        self._create_plot()
    
    def closeEvent(self, event):
        """Handle window close event"""
        window_id = f"dose_{self.satellite_file_path}"
        self.window_closed.emit(window_id)
        super().closeEvent(event)