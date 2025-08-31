"""
Analysis control panel for STRATOS
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLabel, QWidget, QPushButton,
    QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox
)
from PyQt6.QtCore import pyqtSignal


class AnalysisPanel(QGroupBox):
    """Panel for analysis controls"""
    
    # Signals
    analysis_mode_changed = pyqtSignal(str)
    flux_time_clicked = pyqtSignal()
    peak_analysis_clicked = pyqtSignal()
    dose_calc_clicked = pyqtSignal()
    energy_spectrum_clicked = pyqtSignal(str)  # Pass selected satellite
    pitch_angle_clicked = pyqtSignal(str)  # Pass selected satellite
    phase_space_clicked = pyqtSignal(str)  # Pass selected satellite
    orbit_stats_clicked = pyqtSignal()
    ground_track_clicked = pyqtSignal()
    altitude_profile_clicked = pyqtSignal()
    settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the analysis panel"""
        super().__init__("Analysis", parent)
        self.loaded_satellites = {}  # Track loaded satellite files
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        self.main_layout = QVBoxLayout(self)
        
        # Analysis mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems([
            "Flux Analysis", 
            "Spectrum Analysis",
            "Orbital Analysis"
        ])
        self.analysis_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.analysis_mode_combo)
        self.main_layout.addLayout(mode_layout)
        
        # Dynamic content area
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_widget)
        self.main_layout.addWidget(self.dynamic_widget)
        
        # Initialize with default mode
        self._update_dynamic_controls()
    
    def _on_mode_changed(self, mode):
        """Handle analysis mode change"""
        self._update_dynamic_controls()
        self.analysis_mode_changed.emit(mode)
    
    def _update_dynamic_controls(self):
        """Update controls based on selected mode"""
        # Clear existing widgets
        self._clear_layout(self.dynamic_layout)
        
        analysis_type = self.analysis_mode_combo.currentText()
        
        if analysis_type == "Flux Analysis":
            self._add_flux_analysis_controls()
        elif analysis_type == "Spectrum Analysis":
            self._add_spectrum_analysis_controls()
        elif analysis_type == "Orbital Analysis":
            self._add_orbital_analysis_controls()
    
    def _add_flux_analysis_controls(self):
        """Add flux analysis specific controls"""
        # Analysis windows
        self.dynamic_layout.addWidget(QLabel("Analysis Windows:"))
        
        self.flux_time_button = QPushButton("Flux vs Time")
        self.flux_time_button.clicked.connect(self.flux_time_clicked.emit)
        self.dynamic_layout.addWidget(self.flux_time_button)
    
    def _add_spectrum_analysis_controls(self):
        """Add spectrum analysis specific controls"""
        spectrum_layout = QFormLayout()
        
        # Satellite selector (only show if multiple satellites loaded)
        if len(self.loaded_satellites) > 1:
            self.satellite_combo = QComboBox()
            self.satellite_combo.addItem("Select Satellite...")
            for file_path, display_name in self.loaded_satellites.items():
                self.satellite_combo.addItem(display_name, file_path)
            spectrum_layout.addRow("Satellite:", self.satellite_combo)
        elif len(self.loaded_satellites) == 1:
            # Single satellite - show as label
            satellite_name = list(self.loaded_satellites.values())[0]
            satellite_label = QLabel(f"Tracking: {satellite_name}")
            satellite_label.setStyleSheet("font-weight: bold; color: #666;")
            spectrum_layout.addRow("Satellite:", satellite_label)
        else:
            # No satellites loaded
            no_sat_label = QLabel("No satellite trajectories loaded")
            no_sat_label.setStyleSheet("color: #888; font-style: italic;")
            spectrum_layout.addRow("Satellite:", no_sat_label)
        
        # Energy bins
        self.energy_bins_spin = QSpinBox()
        self.energy_bins_spin.setRange(10, 200)
        self.energy_bins_spin.setValue(50)
        self.energy_bins_spin.valueChanged.connect(self.settings_changed.emit)
        spectrum_layout.addRow("Energy Bins:", self.energy_bins_spin)
        
        # Energy range
        energy_range_layout = QHBoxLayout()
        self.min_energy_input = QLineEdit("10")
        self.max_energy_input = QLineEdit("10000")
        self.min_energy_input.textChanged.connect(self.settings_changed.emit)
        self.max_energy_input.textChanged.connect(self.settings_changed.emit)
        energy_range_layout.addWidget(QLabel("Min:"))
        energy_range_layout.addWidget(self.min_energy_input)
        energy_range_layout.addWidget(QLabel("keV Max:"))
        energy_range_layout.addWidget(self.max_energy_input)
        energy_range_layout.addWidget(QLabel("keV"))
        spectrum_layout.addRow("Energy Range:", energy_range_layout)
        
        self.dynamic_layout.addLayout(spectrum_layout)
        
        # Analysis windows
        self.dynamic_layout.addWidget(QLabel("Spectrum Windows:"))
        
        self.energy_spectrum_button = QPushButton("Energy Spectrum")
        self.phase_space_button = QPushButton("Phase Space Plot")
        
        # Enable buttons only if satellites are loaded
        has_satellites = len(self.loaded_satellites) > 0
        self.energy_spectrum_button.setEnabled(has_satellites)
        self.phase_space_button.setEnabled(has_satellites)
        
        self.energy_spectrum_button.clicked.connect(self._on_energy_spectrum_clicked)
        self.phase_space_button.clicked.connect(self._on_phase_space_clicked)
        
        self.dynamic_layout.addWidget(self.energy_spectrum_button)
        self.dynamic_layout.addWidget(self.phase_space_button)
    
    def _on_energy_spectrum_clicked(self):
        """Handle energy spectrum button click"""
        selected_satellite = self._get_selected_satellite()
        if selected_satellite:
            self.energy_spectrum_clicked.emit(selected_satellite)
    
    
    def _on_phase_space_clicked(self):
        """Handle phase space button click"""
        selected_satellite = self._get_selected_satellite()
        if selected_satellite:
            self.phase_space_clicked.emit(selected_satellite)
    
    def _get_selected_satellite(self):
        """Get the currently selected satellite file path
        
        Returns:
            str: Path to selected satellite file, or None if no selection/satellites
        """
        if len(self.loaded_satellites) == 0:
            return None
        elif len(self.loaded_satellites) == 1:
            return list(self.loaded_satellites.keys())[0]
        else:
            if hasattr(self, 'satellite_combo'):
                current_index = self.satellite_combo.currentIndex()
                if current_index > 0:  # Index 0 is "Select Satellite..."
                    return self.satellite_combo.itemData(current_index)
            return None
    
    def _get_orbital_selected_satellite(self):
        """Get the currently selected satellite file path for orbital analysis
        
        Returns:
            str: Path to selected satellite file, or None if no selection/satellites
        """
        if len(self.loaded_satellites) == 0:
            return None
        elif len(self.loaded_satellites) == 1:
            return list(self.loaded_satellites.keys())[0]
        else:
            if hasattr(self, 'orbital_satellite_combo'):
                current_index = self.orbital_satellite_combo.currentIndex()
                if current_index > 0:  # Index 0 is "Select Satellite..."
                    return self.orbital_satellite_combo.itemData(current_index)
            return None
    
    def _add_orbital_analysis_controls(self):
        """Add orbital analysis specific controls"""
        orbital_layout = QFormLayout()
        
        # Satellite selector (only show if multiple satellites loaded)  
        if len(self.loaded_satellites) > 1:
            self.orbital_satellite_combo = QComboBox()
            self.orbital_satellite_combo.addItem("Select Satellite...")
            for file_path, display_name in self.loaded_satellites.items():
                self.orbital_satellite_combo.addItem(display_name, file_path)
            orbital_layout.addRow("Satellite:", self.orbital_satellite_combo)
        elif len(self.loaded_satellites) == 1:
            # Single satellite - show as label
            satellite_name = list(self.loaded_satellites.values())[0]
            satellite_label = QLabel(f"Tracking: {satellite_name}")
            satellite_label.setStyleSheet("font-weight: bold; color: #666;")
            orbital_layout.addRow("Satellite:", satellite_label)
        else:
            # No satellites loaded
            no_sat_label = QLabel("No satellite trajectories loaded")
            no_sat_label.setStyleSheet("color: #888; font-style: italic;")
            orbital_layout.addRow("Satellite:", no_sat_label)
        
        # Cross section inputs for each loaded satellite
        if self.loaded_satellites:
            orbital_layout.addRow("", QLabel(""))  # Spacer
            self.orbital_cross_sections = {}  # Store cross section spinboxes
            
            for file_path, display_name in self.loaded_satellites.items():
                cross_section_spin = QDoubleSpinBox()
                cross_section_spin.setRange(0.01, 1000.0)
                cross_section_spin.setValue(1.0)
                cross_section_spin.setSuffix(" m²")
                cross_section_spin.setDecimals(3)
                cross_section_spin.setMinimumWidth(80)
                cross_section_spin.setMaximumWidth(120)
                cross_section_spin.valueChanged.connect(self.settings_changed.emit)
                
                # Store the spinbox for retrieval
                self.orbital_cross_sections[file_path] = cross_section_spin
                
                # Add to layout with satellite name - make it more compact
                short_name = display_name.split('/')[-1] if '/' in display_name else display_name
                # Limit label length to prevent overflow
                if len(short_name) > 15:
                    short_name = short_name[:12] + "..."
                orbital_layout.addRow(f"CS ({short_name}):", cross_section_spin)
        
        self.dynamic_layout.addLayout(orbital_layout)
        
        # Analysis windows
        self.dynamic_layout.addWidget(QLabel("Orbital Analysis Windows:"))
        
        # Enable buttons only if satellites are loaded
        has_satellites = len(self.loaded_satellites) > 0
        
        # Trajectory/orbit tools
        self.orbit_stats_button = QPushButton("Orbital Statistics")
        self.ground_track_button = QPushButton("Ground Track")
        self.altitude_profile_button = QPushButton("Altitude Profile")
        
        # Moved from spectrum analysis
        self.pitch_angle_button = QPushButton("Pitch Angle Distribution")
        
        # Moved from flux analysis
        self.peak_analysis_button = QPushButton("Peak Exposure Analysis")
        self.dose_calc_button = QPushButton("Dose Calculator")
        
        # Set enabled state
        self.orbit_stats_button.setEnabled(has_satellites)
        self.ground_track_button.setEnabled(has_satellites)
        self.altitude_profile_button.setEnabled(has_satellites)
        self.pitch_angle_button.setEnabled(has_satellites)
        self.peak_analysis_button.setEnabled(has_satellites)
        self.dose_calc_button.setEnabled(has_satellites)
        
        # Connect signals - use orbital-specific handlers
        self.orbit_stats_button.clicked.connect(self._on_orbital_stats_clicked)
        self.ground_track_button.clicked.connect(self._on_ground_track_clicked)
        self.altitude_profile_button.clicked.connect(self._on_altitude_profile_clicked)
        self.pitch_angle_button.clicked.connect(self._on_orbital_pitch_angle_clicked)
        self.peak_analysis_button.clicked.connect(self._on_orbital_peak_analysis_clicked)
        self.dose_calc_button.clicked.connect(self._on_orbital_dose_calc_clicked)
        
        # Add buttons to layout
        self.dynamic_layout.addWidget(self.orbit_stats_button)
        self.dynamic_layout.addWidget(self.ground_track_button)
        self.dynamic_layout.addWidget(self.altitude_profile_button)
        self.dynamic_layout.addWidget(self.pitch_angle_button)
        self.dynamic_layout.addWidget(self.peak_analysis_button)
        self.dynamic_layout.addWidget(self.dose_calc_button)
    
    def _clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
    
    def get_current_mode(self):
        """Get current analysis mode"""
        return self.analysis_mode_combo.currentText()
    
    def get_cross_section(self):
        """Get cross section value (flux analysis)"""
        if hasattr(self, 'cross_section_spin'):
            return self.cross_section_spin.value()
        return 1.0
    
    def get_integration_time(self):
        """Get integration time (flux analysis)"""
        if hasattr(self, 'integration_spin'):
            return self.integration_spin.value()
        return 1.0
    
    def add_satellite_file(self, file_path, display_name):
        """Add a satellite file to the analysis panel
        
        Args:
            file_path: Path to the satellite file
            display_name: Display name for the file
        """
        self.loaded_satellites[file_path] = display_name
        # Refresh spectrum or orbital controls if currently shown
        current_mode = self.analysis_mode_combo.currentText()
        if current_mode in ["Spectrum Analysis", "Orbital Analysis"]:
            self._update_dynamic_controls()
    
    def remove_satellite_file(self, file_path):
        """Remove a satellite file from the analysis panel
        
        Args:
            file_path: Path to the satellite file
        """
        if file_path in self.loaded_satellites:
            del self.loaded_satellites[file_path]
            # Refresh spectrum or orbital controls if currently shown
            current_mode = self.analysis_mode_combo.currentText()
            if current_mode in ["Spectrum Analysis", "Orbital Analysis"]:
                self._update_dynamic_controls()
    
    def get_energy_range(self):
        """Get the energy range for spectrum analysis
        
        Returns:
            tuple: (min_energy, max_energy) in keV
        """
        if hasattr(self, 'min_energy_input') and hasattr(self, 'max_energy_input'):
            try:
                min_energy = float(self.min_energy_input.text())
                max_energy = float(self.max_energy_input.text())
                return (min_energy, max_energy)
            except ValueError:
                pass
        return (10, 10000)  # Default range
    
    def get_energy_bins(self):
        """Get the number of energy bins for spectrum analysis
        
        Returns:
            int: Number of energy bins
        """
        if hasattr(self, 'energy_bins_spin'):
            return self.energy_bins_spin.value()
        return 50  # Default
    
    # Orbital analysis handlers
    def _on_orbital_stats_clicked(self):
        """Handle orbital statistics button click"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.orbit_stats_clicked.emit()
    
    def _on_ground_track_clicked(self):
        """Handle ground track button click"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.ground_track_clicked.emit()
    
    def _on_altitude_profile_clicked(self):
        """Handle altitude profile button click"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.altitude_profile_clicked.emit()
    
    def _on_orbital_pitch_angle_clicked(self):
        """Handle pitch angle button click in orbital analysis"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.pitch_angle_clicked.emit(selected_satellite)
    
    def _on_orbital_peak_analysis_clicked(self):
        """Handle peak analysis button click in orbital analysis"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.peak_analysis_clicked.emit()
    
    def _on_orbital_dose_calc_clicked(self):
        """Handle dose calculator button click in orbital analysis"""
        selected_satellite = self._get_orbital_selected_satellite()
        if selected_satellite:
            self.dose_calc_clicked.emit()
    
    def get_orbital_cross_section(self, file_path):
        """Get the cross section value for a specific orbital file
        
        Args:
            file_path: Path to the orbital file
            
        Returns:
            float: Cross section value in m²
        """
        if hasattr(self, 'orbital_cross_sections') and file_path in self.orbital_cross_sections:
            return self.orbital_cross_sections[file_path].value()
        return 1.0  # Default value
    
    def get_all_orbital_cross_sections(self):
        """Get cross section values for all loaded satellites
        
        Returns:
            dict: Dictionary mapping file_path to cross section value
        """
        cross_sections = {}
        if hasattr(self, 'orbital_cross_sections'):
            for file_path, spinbox in self.orbital_cross_sections.items():
                cross_sections[file_path] = spinbox.value()
        return cross_sections
