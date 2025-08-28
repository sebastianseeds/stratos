# app.py
"""
STRATOS - Space Trajectory Radiation Analysis Toolkit for Orbital Simulation
Main application entry point
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from config import Config
from ui import ElectronFluxVisualizerApp


def main():
    """Main entry point for STRATOS application"""
    app = QApplication(sys.argv)
    
    # Set application properties from Config
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName(Config.ORGANIZATION)
    
    # Create and show main window
    window = ElectronFluxVisualizerApp()
    window.show()
    
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
