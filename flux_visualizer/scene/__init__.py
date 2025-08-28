# scene/__init__.py
"""
Scene components for STRATOS
"""

from .earth_renderer import EarthRenderer
from .orbital_renderer import OrbitalRenderer, SatelliteRenderer

__all__ = [
    'EarthRenderer',
    'OrbitalRenderer', 
    'SatelliteRenderer',
]
