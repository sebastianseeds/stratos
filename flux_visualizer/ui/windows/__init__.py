"""
Analysis and plot windows for STRATOS
"""

from .spectrum_window import EnergySpectrumWindow, PitchAngleWindow, PhaseSpaceWindow
from .dose_window import DoseWindow

__all__ = ['EnergySpectrumWindow', 'PitchAngleWindow', 'PhaseSpaceWindow', 'DoseWindow']