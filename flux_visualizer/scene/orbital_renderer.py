# flux_visualizer/scene/orbital_renderer.py
"""
Orbital path and satellite rendering for STRATOS
"""

import vtk
import numpy as np
from typing import List, Tuple, Optional
from core import OrbitalPoint
from config import Config


class OrbitalRenderer:
    """Handles rendering of orbital paths"""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize orbital renderer.
        
        Args:
            renderer: VTK renderer to add actors to
        """
        self.renderer = renderer
        self.path_actors = {}  # Dict of file_path -> path_actor
        self.is_visible = True
    
    def create_orbital_path(self, orbital_data: List[OrbitalPoint], 
                           color: Tuple[float, float, float] = (0.9, 0.9, 0.2),
                           file_path: str = None) -> vtk.vtkActor:
        """
        Create 3D orbital path visualization.
        
        Args:
            orbital_data: List of OrbitalPoint objects
            color: RGB color tuple (0-1 range)
            file_path: Optional file path for tracking
            
        Returns:
            VTK actor for the orbital path
        """
        if not orbital_data:
            return None
        
        print(f"Creating orbital path with {len(orbital_data)} points")
        
        # Create path polyline
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for i, point in enumerate(orbital_data):
            points.InsertNextPoint(point.x, point.y, point.z)
        
        # Create line segments
        for i in range(len(orbital_data) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
        
        path_polydata = vtk.vtkPolyData()
        path_polydata.SetPoints(points)
        path_polydata.SetLines(lines)
        
        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(path_polydata)
        
        path_actor = vtk.vtkActor()
        path_actor.SetMapper(path_mapper)
        path_actor.GetProperty().SetColor(color)
        path_actor.GetProperty().SetLineWidth(3.0)
        path_actor.GetProperty().SetOpacity(0.8)
        path_actor.SetVisibility(self.is_visible)
        
        # Store and add to renderer
        if file_path:
            # Remove old path if exists
            if file_path in self.path_actors:
                self.renderer.RemoveActor(self.path_actors[file_path])
            self.path_actors[file_path] = path_actor
        
        self.renderer.AddActor(path_actor)
        print(f"Orbital path created {'and hidden' if not self.is_visible else 'and visible'}")
        return path_actor
    
    def toggle_visibility(self, visible: bool):
        """
        Toggle visibility of all orbital paths.
        
        Args:
            visible: True to show paths, False to hide
        """
        self.is_visible = visible
        for actor in self.path_actors.values():
            actor.SetVisibility(visible)
        print(f"Orbital paths {'visible' if visible else 'hidden'}")
    
    def remove_path(self, file_path: str):
        """Remove a specific orbital path"""
        if file_path in self.path_actors:
            self.renderer.RemoveActor(self.path_actors[file_path])
            del self.path_actors[file_path]
            print(f"Removed orbital path for {file_path}")
    
    def clear_all_paths(self):
        """Remove all orbital paths"""
        for actor in self.path_actors.values():
            self.renderer.RemoveActor(actor)
        self.path_actors.clear()
        print("Cleared all orbital paths")
    
    def get_info(self) -> dict:
        """Get information about current orbital paths"""
        return {
            'num_paths': len(self.path_actors),
            'paths_visible': self.is_visible,
            'file_paths': list(self.path_actors.keys())
        }


class SatelliteRenderer:
    """Handles rendering of satellites and their trails"""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize satellite renderer.
        
        Args:
            renderer: VTK renderer to add actors to
        """
        self.renderer = renderer
        self.satellite_actors = {}  # Dict of file_path -> satellite_actor
        self.trail_actors = {}  # Dict of file_path -> list of trail actors
        self.trail_points = {}  # Dict of file_path -> list of trail positions
        self.max_trail_length = Config.SATELLITE_TRAIL_LENGTH if hasattr(Config, 'SATELLITE_TRAIL_LENGTH') else 15
        self.trail_visible = True
    
    def create_satellite(self, position: Tuple[float, float, float],
                        color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
                        radius: float = None,
                        file_path: str = None) -> vtk.vtkActor:
        """
        Create a satellite sphere.
        
        Args:
            position: Initial (x, y, z) position in km
            color: RGB color tuple (0-1 range)
            radius: Satellite sphere radius in km (defaults to Config value)
            file_path: Optional file path for tracking
            
        Returns:
            VTK actor for the satellite
        """
        if radius is None:
            radius = Config.SATELLITE_SPHERE_RADIUS if hasattr(Config, 'SATELLITE_SPHERE_RADIUS') else 500.0
        
        print(f"Creating satellite with radius {radius} km at position {position}")
        
        # Create sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.Update()
        
        # Create mapper
        sat_mapper = vtk.vtkPolyDataMapper()
        sat_mapper.SetInputConnection(sphere.GetOutputPort())
        sat_mapper.ScalarVisibilityOff()
        
        # Create actor
        sat_actor = vtk.vtkActor()
        sat_actor.SetMapper(sat_mapper)
        sat_actor.GetProperty().SetColor(color)
        sat_actor.GetProperty().SetAmbient(0.9)
        sat_actor.GetProperty().SetDiffuse(0.8)
        sat_actor.GetProperty().SetSpecular(0.1)
        sat_actor.GetProperty().SetOpacity(1.0)
        sat_actor.SetPosition(position)
        
        # Store and add to renderer
        if file_path:
            # Remove old satellite if exists
            if file_path in self.satellite_actors:
                self.renderer.RemoveActor(self.satellite_actors[file_path])
            self.satellite_actors[file_path] = sat_actor
            
            # Initialize trail tracking
            self.trail_points[file_path] = []
            self.trail_actors[file_path] = []
        
        self.renderer.AddActor(sat_actor)
        print(f"Satellite created")
        return sat_actor
    
    def update_satellite_position(self, position: Tuple[float, float, float],
                                 file_path: str = None):
        """
        Update satellite position and trail.
        
        Args:
            position: New (x, y, z) position in km
            file_path: File path to identify which satellite
        """
        if file_path and file_path in self.satellite_actors:
            # Update satellite position
            self.satellite_actors[file_path].SetPosition(position)
            
            # Update trail if visible
            if self.trail_visible:
                self._update_trail(position, file_path)

    def update_all_satellite_sizes(self, new_radius: float):
        """Update the size of all satellites

        Args:
            new_radius: New radius in km for all satellites
        """
        # Check if there are any satellites to update
        if not self.satellite_actors:
            print(f"No satellites loaded to resize")
            return

        for file_path, actor in self.satellite_actors.items():
            # Create new sphere with updated radius
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(new_radius)
            sphere.SetThetaResolution(24)
            sphere.SetPhiResolution(24)
            sphere.Update()

            # Update the actor's mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor.SetMapper(mapper)

        print(f"Updated {len(self.satellite_actors)} satellite(s) to radius {new_radius} km")
    
    def _update_trail(self, position: Tuple[float, float, float], file_path: str):
        """
        Update the satellite trail with fading effect.
        
        Args:
            position: Current position to add to trail
            file_path: File path to identify which satellite
        """
        if file_path not in self.trail_points:
            self.trail_points[file_path] = []
            self.trail_actors[file_path] = []
        
        # Add current position to trail
        self.trail_points[file_path].append(list(position))
        
        # Limit trail length
        if len(self.trail_points[file_path]) > self.max_trail_length:
            self.trail_points[file_path].pop(0)
        
        # Remove old trail actors
        for actor in self.trail_actors[file_path]:
            self.renderer.RemoveActor(actor)
        self.trail_actors[file_path] = []
        
        # Create new trail with fading and tapering
        trail_points = self.trail_points[file_path]
        if len(trail_points) >= 2:
            self._create_fading_trail(trail_points, file_path)
    
    def _create_fading_trail(self, trail_points: List, file_path: str):
        """
        Create trail segments with fading and tapering effect.
        
        Args:
            trail_points: List of trail positions
            file_path: File path for tracking
        """
        try:
            # Create individual segments with fading
            for i in range(len(trail_points) - 1):
                # Create single segment
                points = vtk.vtkPoints()
                points.InsertNextPoint(trail_points[i])
                points.InsertNextPoint(trail_points[i + 1])
                
                lines = vtk.vtkCellArray()
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, 0)
                line.GetPointIds().SetId(1, 1)
                lines.InsertNextCell(line)
                
                segment_polydata = vtk.vtkPolyData()
                segment_polydata.SetPoints(points)
                segment_polydata.SetLines(lines)
                
                # Create tube for this segment
                tube_filter = vtk.vtkTubeFilter()
                tube_filter.SetInputData(segment_polydata)
                
                # Calculate position in trail (0 = oldest, 1 = newest)
                trail_position = i / (len(trail_points) - 1) if len(trail_points) > 1 else 1.0
                
                # Tapering: thicker at newest end
                min_radius = 50.0
                max_radius = 300.0
                radius = min_radius + (max_radius - min_radius) * trail_position
                
                tube_filter.SetRadius(radius)
                tube_filter.SetNumberOfSides(8)
                tube_filter.Update()
                
                # Create mapper and actor
                segment_mapper = vtk.vtkPolyDataMapper()
                segment_mapper.SetInputConnection(tube_filter.GetOutputPort())
                
                segment_actor = vtk.vtkActor()
                segment_actor.SetMapper(segment_mapper)
                
                # Fading: more transparent at older end
                min_opacity = 0.1
                max_opacity = 0.9
                opacity = min_opacity + (max_opacity - min_opacity) * trail_position
                
                # Set properties (white trail)
                prop = segment_actor.GetProperty()
                prop.SetColor(1.0, 1.0, 1.0)  # White
                prop.SetOpacity(opacity)
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
                
                self.renderer.AddActor(segment_actor)
                self.trail_actors[file_path].append(segment_actor)
                
        except Exception:
            # Fallback to simple line trail
            self._create_simple_trail(trail_points, file_path)
    
    def _create_simple_trail(self, trail_points: List, file_path: str):
        """Fallback simple trail if tube filter fails"""
        try:
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            for i, point in enumerate(trail_points):
                points.InsertNextPoint(point[0], point[1], point[2])
            
            for i in range(len(trail_points) - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)
            
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            trail_actor = vtk.vtkActor()
            trail_actor.SetMapper(mapper)
            trail_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            trail_actor.GetProperty().SetLineWidth(20.0)
            trail_actor.GetProperty().SetOpacity(0.8)
            
            self.renderer.AddActor(trail_actor)
            self.trail_actors[file_path] = [trail_actor]
            
        except Exception as e:
            print(f"Trail creation failed: {e}")
    
    def toggle_trail_visibility(self, visible: bool):
        """Toggle visibility of all trails"""
        self.trail_visible = visible
        if not visible:
            # Hide all trails
            for actors in self.trail_actors.values():
                for actor in actors:
                    self.renderer.RemoveActor(actor)
            # Clear trail actors but keep points
            for file_path in self.trail_actors:
                self.trail_actors[file_path] = []
        print(f"Satellite trails {'visible' if visible else 'hidden'}")
    
    def clear_trail(self, file_path: str):
        """Clear trail for a specific satellite"""
        if file_path in self.trail_actors:
            for actor in self.trail_actors[file_path]:
                self.renderer.RemoveActor(actor)
            self.trail_actors[file_path] = []
        
        if file_path in self.trail_points:
            self.trail_points[file_path] = []
    
    def clear_all_trails(self):
        """Clear all satellite trails"""
        for actors in self.trail_actors.values():
            for actor in actors:
                self.renderer.RemoveActor(actor)
        self.trail_actors.clear()
        self.trail_points.clear()
        print("Cleared all satellite trails")
    
    def remove_satellite(self, file_path: str):
        """Remove a specific satellite and its trail"""
        if file_path in self.satellite_actors:
            self.renderer.RemoveActor(self.satellite_actors[file_path])
            del self.satellite_actors[file_path]
        
        self.clear_trail(file_path)
        
        if file_path in self.trail_points:
            del self.trail_points[file_path]
        
        print(f"Removed satellite for {file_path}")
    
    def get_info(self) -> dict:
        """Get information about current satellites"""
        return {
            'num_satellites': len(self.satellite_actors),
            'trails_visible': self.trail_visible,
            'max_trail_length': self.max_trail_length,
            'file_paths': list(self.satellite_actors.keys())
        }
