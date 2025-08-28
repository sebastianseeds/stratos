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
    energy_spectrum_clicked = pyqtSignal()
    pitch_angle_clicked = pyqtSignal()
    phase_space_clicked = pyqtSignal()
    orbit_stats_clicked = pyqtSignal()
    ground_track_clicked = pyqtSignal()
    altitude_profile_clicked = pyqtSignal()
    settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the analysis panel"""
        super().__init__("Analysis", parent)
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
            "Trajectory Statistics"
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
        elif analysis_type == "Trajectory Statistics":
            self._add_trajectory_controls()
    
    def _add_flux_analysis_controls(self):
        """Add flux analysis specific controls"""
        flux_layout = QFormLayout()
        
        # Cross section
        self.cross_section_spin = QDoubleSpinBox()
        self.cross_section_spin.setRange(0.1, 100.0)
        self.cross_section_spin.setValue(1.0)
        self.cross_section_spin.setSuffix(" m")
        self.cross_section_spin.valueChanged.connect(self.settings_changed.emit)
        flux_layout.addRow("Cross Section:", self.cross_section_spin)
        
        # Integration time
        self.integration_spin = QDoubleSpinBox()
        self.integration_spin.setRange(0.1, 1000.0)
        self.integration_spin.setValue(1.0)
        self.integration_spin.setSuffix(" s")
        self.integration_spin.valueChanged.connect(self.settings_changed.emit)
        flux_layout.addRow("Integration:", self.integration_spin)
        
        self.dynamic_layout.addLayout(flux_layout)
        
        # Analysis windows
        self.dynamic_layout.addWidget(QLabel("Analysis Windows:"))
        
        self.flux_time_button = QPushButton("Flux vs Time")
        self.peak_analysis_button = QPushButton("Peak Exposure Analysis")
        self.dose_calc_button = QPushButton("Dose Calculator")
        
        self.flux_time_button.clicked.connect(self.flux_time_clicked.emit)
        self.peak_analysis_button.clicked.connect(self.peak_analysis_clicked.emit)
        self.dose_calc_button.clicked.connect(self.dose_calc_clicked.emit)
        
        self.dynamic_layout.addWidget(self.flux_time_button)
        self.dynamic_layout.addWidget(self.peak_analysis_button)
        self.dynamic_layout.addWidget(self.dose_calc_button)
    
    def _add_spectrum_analysis_controls(self):
        """Add spectrum analysis specific controls"""
        spectrum_layout = QFormLayout()
        
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
        self.pitch_angle_button = QPushButton("Pitch Angle Distribution")
        self.phase_space_button = QPushButton("Phase Space Plot")
        
        self.energy_spectrum_button.clicked.connect(self.energy_spectrum_clicked.emit)
        self.pitch_angle_button.clicked.connect(self.pitch_angle_clicked.emit)
        self.phase_space_button.clicked.connect(self.phase_space_clicked.emit)
        
        self.dynamic_layout.addWidget(self.energy_spectrum_button)
        self.dynamic_layout.addWidget(self.pitch_angle_button)
        self.dynamic_layout.addWidget(self.phase_space_button)
    
    def _add_trajectory_controls(self):
        """Add trajectory statistics specific controls"""
        traj_layout = QFormLayout()
        
        # Statistics options
        self.include_velocity_check = QCheckBox("Include Velocity Analysis")
        self.include_altitude_check = QCheckBox("Include Altitude Profile")
        self.include_velocity_check.stateChanged.connect(self.settings_changed.emit)
        self.include_altitude_check.stateChanged.connect(self.settings_changed.emit)
        traj_layout.addRow("Options:", self.include_velocity_check)
        traj_layout.addRow("", self.include_altitude_check)
        
        self.dynamic_layout.addLayout(traj_layout)
        
        # Analysis windows
        self.dynamic_layout.addWidget(QLabel("Trajectory Windows:"))
        
        self.orbit_stats_button = QPushButton("Orbital Statistics")
        self.ground_track_button = QPushButton("Ground Track")
        self.altitude_profile_button = QPushButton("Altitude Profile")
        
        self.orbit_stats_button.clicked.connect(self.orbit_stats_clicked.emit)
        self.ground_track_button.clicked.connect(self.ground_track_clicked.emit)
        self.altitude_profile_button.clicked.connect(self.altitude_profile_clicked.emit)
        
        self.dynamic_layout.addWidget(self.orbit_stats_button)
        self.dynamic_layout.addWidget(self.ground_track_button)
        self.dynamic_layout.addWidget(self.altitude_profile_button)
    
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
