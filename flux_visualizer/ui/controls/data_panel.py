"""
Data loading panel for STRATOS
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QPushButton, QWidget,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal


class DataLoadingPanel(QGroupBox):
    """Panel for data loading controls"""
    
    # Signals
    load_flux_clicked = pyqtSignal()
    load_orbital_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the data loading panel"""
        super().__init__("Data Loading", parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel UI"""
        self.main_layout = QVBoxLayout(self)
        
        # Load Flux File button
        self.load_flux_button = QPushButton("Load Flux File")
        self.load_flux_button.clicked.connect(self.load_flux_clicked.emit)
        self.main_layout.addWidget(self.load_flux_button)
        
        # Container for loaded flux files
        self.flux_files_container = QWidget()
        self.flux_files_layout = QVBoxLayout(self.flux_files_container)
        self.flux_files_layout.setContentsMargins(0, 5, 0, 10)
        self.flux_files_layout.setSpacing(2)
        self.main_layout.addWidget(self.flux_files_container)
        
        # Load Orbital File button
        self.load_orbit_button = QPushButton("Load Orbital File")
        self.load_orbit_button.clicked.connect(self.load_orbital_clicked.emit)
        self.main_layout.addWidget(self.load_orbit_button)
        
        # Scrollable container for loaded orbital files
        self.orbital_scroll = QScrollArea()
        self.orbital_scroll.setWidgetResizable(True)
        self.orbital_scroll.setMaximumHeight(120)  # ~5 items
        self.orbital_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.orbital_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: #f0f0f0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #888;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
        """)
        
        # Widget inside scroll area
        self.orbital_files_widget = QWidget()
        self.orbital_files_layout = QVBoxLayout(self.orbital_files_widget)
        self.orbital_files_layout.setContentsMargins(0, 5, 0, 0)
        self.orbital_files_layout.setSpacing(2)
        self.orbital_scroll.setWidget(self.orbital_files_widget)
        
        # Initially hide scroll area
        self.orbital_scroll.setVisible(False)
        self.main_layout.addWidget(self.orbital_scroll)
        
        # Add stretch to push everything to the top
        self.main_layout.addStretch()
    
    def add_flux_file_widget(self, widget):
        """Add a flux file widget to the panel"""
        self.flux_files_layout.addWidget(widget)
    
    def add_orbital_file_widget(self, widget):
        """Add an orbital file widget to the panel"""
        self.orbital_files_layout.addWidget(widget)
        self.orbital_scroll.setVisible(True)
    
    def remove_flux_file_widget(self, widget):
        """Remove a flux file widget from the panel"""
        self.flux_files_layout.removeWidget(widget)
    
    def remove_orbital_file_widget(self, widget):
        """Remove an orbital file widget from the panel"""
        self.orbital_files_layout.removeWidget(widget)
        # Hide scroll if no more files
        if self.orbital_files_layout.count() == 0:
            self.orbital_scroll.setVisible(False)
