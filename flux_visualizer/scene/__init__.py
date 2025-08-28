# scene/__init__.py
"""
Scene module for Flux Orbital Visualizer
Handles 3D scene components including Earth, satellites, and orbital paths
"""

from .earth_renderer import EarthRenderer

__all__ = ['EarthRenderer']
