"""
Control panels for STRATOS UI
"""

from .data_panel import DataLoadingPanel
from .animation_panel import AnimationControlPanel
from .visualization_panel import VisualizationPanel
from .analysis_panel import AnalysisPanel
from .earth_controls import EarthControlsWidget

__all__ = [
    'DataLoadingPanel',
    'AnimationControlPanel', 
    'VisualizationPanel',
    'AnalysisPanel',
    'EarthControlsWidget',
]
