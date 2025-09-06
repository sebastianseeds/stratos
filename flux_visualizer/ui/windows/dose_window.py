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
        
        # Use actual flux uncertainties if available, otherwise synthetic estimates
        if self.dose_data.get('has_uncertainty', False):
            dose_rate_errors = self.dose_data['dose_rate_uncertainties_mGy_per_s']
            cumulative_dose_errors = self.dose_data['cumulative_dose_uncertainties_mGy']
            print(f"Using actual flux uncertainties from data (mean relative: {self.dose_data.get('mean_relative_uncertainty', 0)*100:.1f}%)")
        else:
            # Fall back to synthetic uncertainty estimates
            dose_rate_errors = self._calculate_dose_rate_errors(dose_rates)
            cumulative_dose_errors = self._calculate_cumulative_dose_errors(cumulative_dose, dose_rate_errors, times)
            print("Using synthetic uncertainty estimates (no flux uncertainty data)")
        
        # Plot dose rate with error band (fill_between)
        self.ax1.plot(times, dose_rates, 'b-', linewidth=2, label='Dose Rate')
        self.ax1.fill_between(times, 
                             dose_rates - dose_rate_errors,
                             dose_rates + dose_rate_errors,
                             color='lightblue', alpha=0.3, label='±1σ uncertainty')
        self.ax1.set_ylabel('Dose Rate (mGy/s)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Instantaneous Dose Rate')
        
        # Plot cumulative dose with error band
        self.ax2.plot(times, cumulative_dose, 'r-', linewidth=2, label='Cumulative Dose')
        self.ax2.fill_between(times,
                             cumulative_dose - cumulative_dose_errors,
                             cumulative_dose + cumulative_dose_errors,
                             color='lightcoral', alpha=0.3, label='±1σ uncertainty')
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
    
    def _calculate_dose_rate_errors(self, dose_rates):
        """
        Calculate uncertainty estimates for instantaneous dose rate measurements.
        
        Dose rate uncertainties are INSTANTANEOUS and do not accumulate with time.
        Each measurement is independent with the same systematic uncertainties.
        
        Components:
        1. Poisson statistical uncertainty from particle counting
        2. Flux field interpolation uncertainty  
        3. Cross-sectional area measurement uncertainty
        4. Energy conversion factor uncertainty
        5. Baseline measurement uncertainty
        
        Args:
            dose_rates: Array of dose rate values (mGy/s)
            
        Returns:
            Array of uncertainty estimates (1-sigma) - constant in time
        """
        dose_rates = np.array(dose_rates)
        
        # Component 1: Poisson statistical uncertainty
        # Relative uncertainty ∝ 1/√N for each independent measurement
        effective_count_time = 1.0  # seconds (sampling resolution)
        particle_counts = dose_rates * effective_count_time * 1e6  # scaled estimate
        poisson_relative_error = 1.0 / np.sqrt(np.maximum(particle_counts, 1))
        poisson_error = dose_rates * poisson_relative_error
        
        # Component 2: Flux field interpolation uncertainty
        # Constant relative uncertainty for spatial interpolation
        flux_field_relative_error = 0.10  # 10% interpolation uncertainty
        flux_field_error = dose_rates * flux_field_relative_error
        
        # Component 3: Cross-sectional area measurement uncertainty
        # Constant systematic uncertainty in geometric measurement
        area_relative_error = 0.02  # 2% measurement uncertainty
        area_error = dose_rates * area_relative_error
        
        # Component 4: Energy conversion factor uncertainty
        # Constant systematic uncertainty in dose conversion
        energy_conversion_relative_error = 0.05  # 5% energy spectrum uncertainty
        energy_error = dose_rates * energy_conversion_relative_error
        
        # Component 5: Baseline measurement uncertainty (minimum floor)
        # Instrument noise floor - constant absolute uncertainty
        baseline_error = np.full_like(dose_rates, np.max(dose_rates) * 0.01)
        
        # Combine uncertainties in quadrature (assuming independence)
        # NOTE: These are INSTANTANEOUS uncertainties - they do NOT grow with time
        total_error = np.sqrt(poisson_error**2 + 
                             flux_field_error**2 + 
                             area_error**2 + 
                             energy_error**2 + 
                             baseline_error**2)
        
        return total_error
    
    def _calculate_cumulative_dose_errors(self, cumulative_dose, dose_rate_errors, times):
        """
        Calculate uncertainty propagation for cumulative dose.
        
        Cumulative dose uncertainties GROW WITH TIME due to:
        1. Statistical error accumulation through integration
        2. Systematic drift and calibration uncertainties
        3. Correlated measurement errors over time
        
        Unlike instantaneous dose rates, cumulative uncertainties increase
        because we are integrating (summing) uncertain measurements over time.
        
        Args:
            cumulative_dose: Array of cumulative dose values (mGy)
            dose_rate_errors: Array of dose rate uncertainties (mGy/s)
            times: Array of time values (hours)
            
        Returns:
            Array of cumulative dose uncertainties (1-sigma) - grows with time
        """
        if len(times) < 2:
            return np.zeros_like(cumulative_dose)
        
        # Calculate time intervals
        dt = np.diff(times) * 3600  # Convert hours to seconds
        
        # Initialize cumulative error array
        cumulative_errors = np.zeros_like(cumulative_dose)
        cumulative_errors[0] = 0  # No error at t=0
        
        # Statistical error accumulation through integration
        # For integration of uncertain measurements: σ²_cum = Σ(σ²_rate_i * Δt²_i)
        statistical_variance_sum = 0
        
        for i in range(1, len(cumulative_dose)):
            # Add this interval's statistical contribution to total variance
            interval_statistical_variance = (dose_rate_errors[i] * dt[i-1])**2
            statistical_variance_sum += interval_statistical_variance
            
            # Calculate the statistical component
            statistical_error = np.sqrt(statistical_variance_sum)
            
            # Systematic drift grows with time (calibration drift, model uncertainties)
            # This represents the reality that longer measurements accumulate systematic errors
            drift_relative_error = 0.002 * times[i]  # 0.2% per hour drift
            systematic_drift_error = cumulative_dose[i] * drift_relative_error
            
            # Long-term correlation effects (measurements aren't perfectly independent)
            # Some systematic biases persist and accumulate
            correlation_factor = 1.0 + 0.001 * np.sqrt(times[i])  # Grows as √t
            
            # Total cumulative uncertainty combines statistical and systematic components
            # The systematic component dominates for long integrations
            cumulative_errors[i] = np.sqrt(
                (statistical_error * correlation_factor)**2 + 
                systematic_drift_error**2
            )
        
        return cumulative_errors
        
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