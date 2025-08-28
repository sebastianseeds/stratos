"""
Animation control panel for STRATOS
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal


class AnimationControlPanel(QGroupBox):
    """Panel for animation controls"""
    
    # Signals
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    time_changed = pyqtSignal(int)
    speed_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        """Initialize the animation control panel"""
        super().__init__("Animation Controls", parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        layout = QVBoxLayout(self)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        self.play_button.clicked.connect(self.play_clicked.emit)
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.time_changed.emit)
        layout.addWidget(QLabel("Time:"))
        layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("No data loaded")
        layout.addWidget(self.time_label)
        
        # Animation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("100 ms")
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
    
    def _on_speed_changed(self, value):
        """Handle speed slider change"""
        self.speed_label.setText(f"{value} ms")
        self.speed_changed.emit(value)
    
    def set_animation_playing(self, playing):
        """Update button states for playing/stopped"""
        self.play_button.setEnabled(not playing)
        self.pause_button.setEnabled(playing)
        self.stop_button.setEnabled(playing)
    
    def set_time_range(self, max_time):
        """Set the maximum value for time slider"""
        self.time_slider.setMaximum(max_time)
        self.time_slider.setEnabled(True)
    
    def set_time_value(self, value):
        """Set current time slider value"""
        self.time_slider.setValue(value)
    
    def set_time_label(self, text):
        """Update the time label"""
        self.time_label.setText(text)
    
    def get_speed(self):
        """Get current animation speed in milliseconds"""
        return self.speed_slider.value()
