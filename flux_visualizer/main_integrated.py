"""
Electron Flux Orbital Visualizer - Integrated Version
This is a modified version that uses the new modular components
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# NEW: Import refactored components
from config import Config
from core import OrbitalPoint, OrbitalPath
from data_io import VTKDataLoader, OrbitalDataLoader  
from visualization import ColorManager

# Original imports (keep all of these)
import numpy as np
import vtk
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QDoubleSpinBox, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QFormLayout, QSpinBox,
    QSplitter, QFrame, QComboBox, QCheckBox, QLineEdit
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont

# Keep all other original imports...

class ElectronFluxVisualizer(QMainWindow):
    """Main application window - INTEGRATED VERSION"""
    
    def __init__(self):
        super().__init__()
        
        # NEW: Initialize refactored components
        self.config = Config()
        self.color_manager = ColorManager()
        
        # Original initialization code...
        self.setWindowTitle("Electron Flux Orbital Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Rest of your original __init__ code...
        # ... (keep everything else the same)
        
    def load_vtk_data(self):
        """Load VTK data file - UPDATED TO USE VTKDataLoader"""
        file_filter = VTKDataLoader.get_file_filter()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load VTK Data", "", file_filter
        )
        
        if not file_path:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # NEW: Use VTKDataLoader
            self.vtk_data = VTKDataLoader.load(file_path)
            
            # Get data info
            info = VTKDataLoader.get_data_info(self.vtk_data)
            
            self.progress_bar.setValue(75)
            
            # Clear any existing visualization
            self.clear_field_visualization()
            
            # Setup field visualization
            self.setup_field_visualization()
            
            # Update analyzer
            self.flux_analyzer.set_vtk_data(self.vtk_data)
            
            self.progress_bar.setValue(100)
            
            # Update status
            self.status_label.setText(
                f"✓ Loaded VTK data successfully!\n"
                f"Points: {info['num_points']:,} | "
                f"Cells: {info['num_cells']:,}\n"
                f"Scalar: {info['scalar_name']} | "
                f"Range: {info['scalar_range'][0]:.2e} to {info['scalar_range'][1]:.2e}"
            )
            
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            self.status_label.setText(f"Error loading VTK file: {str(e)}")
            QMessageBox.critical(self, "VTK Loading Error", str(e))
            
        finally:
            self.progress_bar.setVisible(False)
    
    def load_orbital_data(self):
        """Load orbital CSV data - UPDATED TO USE OrbitalDataLoader"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Orbital Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # NEW: Use OrbitalDataLoader
            self.orbital_path = OrbitalDataLoader.load_csv(file_path)
            
            # Store filename for legend
            self.satellite_file_name = Path(file_path).stem
            
            # Update time slider
            self.time_slider.setMaximum(len(self.orbital_path) - 1)
            
            # Update analyzer
            self.flux_analyzer.set_orbital_data(self.orbital_path)
            
            # Create path visualization
            self.create_path_visualization()
            
            # Create satellite legend
            self.create_satellite_legend()
            
            # Reset animation
            self.reset_animation()
            
            # Get trajectory info
            info = OrbitalDataLoader.get_trajectory_info(self.orbital_path)
            
            # Update status
            self.status_label.setText(
                f"✓ Loaded orbital data: {info['num_points']} points "
                f"over {info['time_span']:.2f} hours\n"
                f"Altitude: {info['altitude_min']:.1f} - {info['altitude_max']:.1f} km\n"
                f"Satellite: {self.satellite_file_name}"
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load orbital data:\n{str(e)}")
    
    def create_lookup_table_with_scale(self, colormap_name, scale_mode, scalar_range):
        """DELEGATE to ColorManager"""
        return self.color_manager.create_lookup_table_with_scale(
            colormap_name, scale_mode, scalar_range
        )
    
    def get_colormap_color(self, colormap_name, t):
        """DELEGATE to ColorManager"""
        return self.color_manager.get_colormap_color(colormap_name, t)
    
    # Keep ALL other methods exactly as they are...
    # ... (rest of your original code)

# Keep your main function
def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName(Config.ORGANIZATION)
    
    # Create and show main window
    window = ElectronFluxVisualizer()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
