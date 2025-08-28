# scene/earth_renderer.py
"""
Earth Visualization Component for Flux Orbital Visualizer
Handles Earth sphere, texture, lat/long grid, and coordinate labels
"""

import vtk
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import os

from config import Config


class EarthRenderer:
    """Handles all Earth visualization including sphere, grid, and labels"""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize Earth renderer.
        
        Args:
            renderer: VTK renderer to add Earth components to
        """
        self.renderer = renderer
        
        # Earth sphere components
        self.earth_actor: Optional[vtk.vtkActor] = None
        
        # Grid components
        self.lat_long_actors: List[vtk.vtkActor] = []
        self.grid_labels: List[vtk.vtkFollower] = []
        self.equator_actor: Optional[vtk.vtkActor] = None
        self.prime_meridian_actor: Optional[vtk.vtkActor] = None
        
        # State
        self.grid_visible = False
        self.current_opacity = Config.EARTH_DEFAULT_OPACITY
        
    def create_earth(self) -> bool:
        """
        Create Earth representation with optional texture.
        
        Returns:
            True if Earth was created successfully
        """
        print("Creating Earth representation...")
        
        try:
            # Clean up any existing Earth actors
            self.cleanup()
            
            # Create Earth sphere
            earth_sphere = vtk.vtkSphereSource()
            earth_sphere.SetRadius(Config.EARTH_RADIUS_KM)
            earth_sphere.SetThetaResolution(Config.EARTH_SPHERE_RESOLUTION_THETA)
            earth_sphere.SetPhiResolution(Config.EARTH_SPHERE_RESOLUTION_PHI)
            earth_sphere.SetCenter(0.0, 0.0, 0.0)
            earth_sphere.Update()
            
            # Get sphere data and add texture coordinates
            sphere_data = earth_sphere.GetOutput()
            self._add_texture_coordinates(sphere_data)
            
            # Create mapper and actor
            earth_mapper = vtk.vtkPolyDataMapper()
            earth_mapper.SetInputData(sphere_data)
            
            self.earth_actor = vtk.vtkActor()
            self.earth_actor.SetMapper(earth_mapper)
            
            # Try to apply Earth texture
            earth_texture = self._load_earth_texture()
            if earth_texture:
                earth_texture.SetRepeat(0)
                earth_texture.SetInterpolate(1)
                earth_texture.SetWrap(vtk.vtkTexture.ClampToEdge)
                self.earth_actor.SetTexture(earth_texture)
                print("Earth texture applied successfully")
            else:
                # Fallback to solid color
                self.earth_actor.GetProperty().SetColor(*Config.EARTH_DEFAULT_COLOR)
                print("Using default Earth color (no texture found)")
            
            # Set Earth material properties
            earth_property = self.earth_actor.GetProperty()
            earth_property.SetRepresentationToSurface()
            earth_property.SetOpacity(self.current_opacity)
            earth_property.SetAmbient(Config.EARTH_AMBIENT)
            earth_property.SetDiffuse(Config.EARTH_DIFFUSE)
            earth_property.SetSpecular(Config.EARTH_SPECULAR)
            earth_property.SetSpecularPower(Config.EARTH_SPECULAR_POWER)
            
            # Add to renderer
            self.earth_actor.SetVisibility(True)
            self.renderer.AddActor(self.earth_actor)
            
            print("Earth created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating Earth: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_lat_long_grid(self) -> None:
        """Create latitude/longitude grid with coordinate labels."""
        if self.grid_visible:
            return  # Grid already exists
            
        try:
            print("Creating lat/long grid with coordinate labels...")
            
            # Clear any existing grid
            self._clear_grid()
            
            # Create latitude lines
            for lat_deg in range(-75, 90, Config.LATITUDE_GRID_SPACING):
                lat_rad = np.radians(lat_deg)
                radius = Config.EARTH_RADIUS_KM * np.cos(lat_rad)
                height = Config.EARTH_RADIUS_KM * np.sin(lat_rad)
                
                if radius > 100:  # Skip very small circles near poles
                    circle = self._create_circle(radius, height, 'latitude')
                    if circle:
                        circle.GetProperty().SetOpacity(self.current_opacity * Config.LATITUDE_LINE_OPACITY_FACTOR)
                        self.lat_long_actors.append(circle)
                        self.renderer.AddActor(circle)
                        
                        # Create latitude label
                        label = self._create_coordinate_label(
                            f"{lat_deg}°",
                            radius + Config.GRID_LABEL_OFFSET,
                            0,
                            height
                        )
                        if label:
                            self.grid_labels.append(label)
                            self.renderer.AddActor(label)
            
            # Create longitude lines (meridians)
            for lon_deg in range(0, 360, Config.LONGITUDE_GRID_SPACING):
                meridian = self._create_meridian(lon_deg)
                if meridian:
                    meridian.GetProperty().SetOpacity(self.current_opacity * Config.LATITUDE_LINE_OPACITY_FACTOR)
                    self.lat_long_actors.append(meridian)
                    self.renderer.AddActor(meridian)
                    
                    # Create longitude label at equator
                    lon_rad = np.radians(lon_deg)
                    x = (Config.EARTH_RADIUS_KM + Config.GRID_LABEL_OFFSET) * np.cos(lon_rad)
                    y = (Config.EARTH_RADIUS_KM + Config.GRID_LABEL_OFFSET) * np.sin(lon_rad)
                    
                    # Format longitude label
                    if lon_deg == 0:
                        lon_label = "0°"
                    elif lon_deg <= 180:
                        lon_label = f"{lon_deg}°E"
                    else:
                        lon_label = f"{360-lon_deg}°W"
                        
                    label = self._create_coordinate_label(lon_label, x, y, 0)
                    if label:
                        self.grid_labels.append(label)
                        self.renderer.AddActor(label)
            
            # Create highlighted equator
            self.equator_actor = self._create_circle(
                Config.EARTH_RADIUS_KM + 1,  # Slightly above surface
                0.0,
                'equator'
            )
            if self.equator_actor:
                self.equator_actor.GetProperty().SetOpacity(self.current_opacity * Config.EQUATOR_OPACITY_FACTOR)
                self.renderer.AddActor(self.equator_actor)
            
            # Create highlighted prime meridian
            self.prime_meridian_actor = self._create_meridian(0, highlight=True)
            if self.prime_meridian_actor:
                self.prime_meridian_actor.GetProperty().SetOpacity(self.current_opacity * Config.EQUATOR_OPACITY_FACTOR)
                self.renderer.AddActor(self.prime_meridian_actor)
            
            self.grid_visible = True
            print(f"Created {len(self.lat_long_actors)} grid lines and {len(self.grid_labels)} labels")
            
        except Exception as e:
            print(f"Error creating lat/long grid: {e}")
            import traceback
            traceback.print_exc()
    
    def toggle_grid(self, visible: bool) -> None:
        """
        Toggle visibility of lat/long grid and labels.
        
        Args:
            visible: Whether to show the grid
        """
        try:
            if visible and not self.grid_visible:
                # Create grid if it doesn't exist
                self.create_lat_long_grid()
            elif visible and self.grid_visible:
                # Show existing grid
                for actor in self.lat_long_actors:
                    if actor:
                        actor.SetVisibility(True)
                        
                if self.equator_actor:
                    self.equator_actor.SetVisibility(True)
                    
                if self.prime_meridian_actor:
                    self.prime_meridian_actor.SetVisibility(True)
                    
                for label in self.grid_labels:
                    if label:
                        label.SetVisibility(True)
            else:
                # Hide grid
                for actor in self.lat_long_actors:
                    if actor:
                        actor.SetVisibility(False)
                        
                if self.equator_actor:
                    self.equator_actor.SetVisibility(False)
                    
                if self.prime_meridian_actor:
                    self.prime_meridian_actor.SetVisibility(False)
                    
                for label in self.grid_labels:
                    if label:
                        label.SetVisibility(False)
            
            self.grid_visible = visible
            print(f"Lat/long grid {'visible' if visible else 'hidden'}")
            
        except Exception as e:
            print(f"Error toggling grid: {e}")
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set Earth opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        try:
            self.current_opacity = np.clip(opacity, 0.0, 1.0)
            
            # Update Earth sphere
            if self.earth_actor:
                self.earth_actor.GetProperty().SetOpacity(self.current_opacity)
            
            # Update grid if visible
            if self.grid_visible:
                grid_opacity = self.current_opacity * Config.LATITUDE_LINE_OPACITY_FACTOR
                
                for actor in self.lat_long_actors:
                    if actor:
                        actor.GetProperty().SetOpacity(grid_opacity)
                
                if self.equator_actor:
                    self.equator_actor.GetProperty().SetOpacity(
                        self.current_opacity * Config.EQUATOR_OPACITY_FACTOR
                    )
                    
                if self.prime_meridian_actor:
                    self.prime_meridian_actor.GetProperty().SetOpacity(
                        self.current_opacity * Config.EQUATOR_OPACITY_FACTOR
                    )
                
                # Update label opacity
                label_opacity = self.current_opacity * Config.GRID_LABEL_OPACITY
                for label in self.grid_labels:
                    if label:
                        label.GetProperty().SetOpacity(label_opacity)
            
            print(f"Earth opacity set to {self.current_opacity:.2f}")
            
        except Exception as e:
            print(f"Error setting Earth opacity: {e}")
    
    def cleanup(self) -> None:
        """Clean up all Earth-related actors."""
        print("Cleaning up Earth actors...")
        
        # Remove Earth sphere
        if self.earth_actor:
            self.renderer.RemoveActor(self.earth_actor)
            self.earth_actor = None
        
        # Remove grid
        self._clear_grid()
        
        print("Earth cleanup complete")
    
    # ============================================
    # Private Helper Methods
    # ============================================
    
    def _clear_grid(self) -> None:
        """Clear all grid actors and labels."""
        # Remove grid lines
        for actor in self.lat_long_actors:
            if actor:
                self.renderer.RemoveActor(actor)
        self.lat_long_actors = []
        
        # Remove special lines
        if self.equator_actor:
            self.renderer.RemoveActor(self.equator_actor)
            self.equator_actor = None
            
        if self.prime_meridian_actor:
            self.renderer.RemoveActor(self.prime_meridian_actor)
            self.prime_meridian_actor = None
        
        # Remove labels
        for label in self.grid_labels:
            if label:
                self.renderer.RemoveActor(label)
        self.grid_labels = []
        
        self.grid_visible = False
    
    def _create_circle(self, radius: float, height: float, circle_type: str) -> Optional[vtk.vtkActor]:
        """
        Create a circle for latitude lines.
        
        Args:
            radius: Circle radius in km
            height: Height above/below equatorial plane in km
            circle_type: 'latitude', 'equator', etc.
            
        Returns:
            Circle actor or None if creation failed
        """
        try:
            circle_source = vtk.vtkRegularPolygonSource()
            circle_source.SetNumberOfSides(Config.LATITUDE_CIRCLE_SEGMENTS)
            circle_source.SetRadius(radius)
            circle_source.SetCenter(0, 0, height)
            circle_source.SetNormal(0, 0, 1)  # XY plane
            circle_source.Update()
            
            circle_mapper = vtk.vtkPolyDataMapper()
            circle_mapper.SetInputConnection(circle_source.GetOutputPort())
            
            circle_actor = vtk.vtkActor()
            circle_actor.SetMapper(circle_mapper)
            circle_actor.GetProperty().SetRepresentationToWireframe()
            
            # Set properties based on type
            if circle_type == 'equator':
                circle_actor.GetProperty().SetColor(*Config.EQUATOR_COLOR)
                circle_actor.GetProperty().SetLineWidth(Config.EQUATOR_LINE_WIDTH)
            else:  # Regular latitude line
                circle_actor.GetProperty().SetColor(*Config.LATITUDE_LINE_COLOR)
                circle_actor.GetProperty().SetLineWidth(Config.LATITUDE_LINE_WIDTH)
            
            return circle_actor
            
        except Exception as e:
            print(f"Error creating circle: {e}")
            return None
    
    def _create_meridian(self, longitude_deg: float, highlight: bool = False) -> Optional[vtk.vtkActor]:
        """
        Create a meridian (longitude line).
        
        Args:
            longitude_deg: Longitude in degrees
            highlight: Whether this is a special meridian (e.g., prime meridian)
            
        Returns:
            Meridian actor or None if creation failed
        """
        try:
            lon_rad = np.radians(longitude_deg)
            
            # Create points from south pole to north pole
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            latitudes = np.linspace(-np.pi/2, np.pi/2, Config.LONGITUDE_MERIDIAN_POINTS)
            
            for i, lat in enumerate(latitudes):
                x = (Config.EARTH_RADIUS_KM + 1) * np.cos(lat) * np.cos(lon_rad)
                y = (Config.EARTH_RADIUS_KM + 1) * np.cos(lat) * np.sin(lon_rad)
                z = (Config.EARTH_RADIUS_KM + 1) * np.sin(lat)
                points.InsertNextPoint(x, y, z)
                
                if i > 0:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i-1)
                    line.GetPointIds().SetId(1, i)
                    lines.InsertNextCell(line)
            
            # Create polydata
            meridian_polydata = vtk.vtkPolyData()
            meridian_polydata.SetPoints(points)
            meridian_polydata.SetLines(lines)
            
            # Create mapper and actor
            meridian_mapper = vtk.vtkPolyDataMapper()
            meridian_mapper.SetInputData(meridian_polydata)
            
            meridian_actor = vtk.vtkActor()
            meridian_actor.SetMapper(meridian_mapper)
            
            # Set properties
            if highlight:
                meridian_actor.GetProperty().SetColor(*Config.PRIME_MERIDIAN_COLOR)
                meridian_actor.GetProperty().SetLineWidth(Config.PRIME_MERIDIAN_LINE_WIDTH)
            else:
                meridian_actor.GetProperty().SetColor(*Config.LATITUDE_LINE_COLOR)
                meridian_actor.GetProperty().SetLineWidth(Config.LATITUDE_LINE_WIDTH)
            
            return meridian_actor
            
        except Exception as e:
            print(f"Error creating meridian: {e}")
            return None
    
    def _create_coordinate_label(self, text: str, x: float, y: float, z: float) -> Optional[vtk.vtkFollower]:
        """
        Create a 3D coordinate label that faces the camera.
        
        Args:
            text: Label text
            x, y, z: Position in km
            
        Returns:
            Label actor or None if creation failed
        """
        try:
            # Create text source
            text_source = vtk.vtkVectorText()
            text_source.SetText(text)
            text_source.Update()
            
            # Create follower (billboard text)
            follower = vtk.vtkFollower()
            
            # Create mapper
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())
            follower.SetMapper(text_mapper)
            
            # Position and scale
            follower.SetPosition(x, y, z)
            follower.SetScale(*Config.GRID_LABEL_SCALE)
            
            # Set camera for billboard effect
            camera = self.renderer.GetActiveCamera()
            follower.SetCamera(camera)
            
            # Style
            follower.GetProperty().SetColor(*Config.GRID_LABEL_COLOR)
            follower.GetProperty().SetOpacity(Config.GRID_LABEL_OPACITY)
            
            return follower
            
        except Exception as e:
            print(f"Error creating label: {e}")
            return None
    
    def _add_texture_coordinates(self, sphere_data: vtk.vtkPolyData) -> None:
        """
        Add texture coordinates to sphere for texture mapping.
        
        Args:
            sphere_data: VTK sphere polydata
        """
        try:
            points = sphere_data.GetPoints()
            num_points = points.GetNumberOfPoints()
            
            tex_coords = vtk.vtkFloatArray()
            tex_coords.SetNumberOfComponents(2)
            tex_coords.SetNumberOfTuples(num_points)
            tex_coords.SetName("TextureCoordinates")
            
            for i in range(num_points):
                point = points.GetPoint(i)
                x, y, z = point
                
                # Convert to spherical coordinates
                r = np.sqrt(x*x + y*y + z*z)
                
                # Handle texture seam
                longitude = np.arctan2(y, x)
                longitude_shifted = longitude + np.pi
                if longitude_shifted >= 2 * np.pi:
                    longitude_shifted -= 2 * np.pi
                
                u = longitude_shifted / (2 * np.pi)
                epsilon = 0.0005
                u = np.clip(u, epsilon, 1.0 - epsilon)
                
                # Calculate latitude
                if r > 0:
                    latitude = np.arcsin(np.clip(z / r, -1.0, 1.0))
                else:
                    latitude = 0
                
                v = (latitude + np.pi/2) / np.pi
                v = np.clip(v, epsilon, 1.0 - epsilon)
                
                tex_coords.SetTuple2(i, u, v)
            
            sphere_data.GetPointData().SetTCoords(tex_coords)
            print(f"Added texture coordinates for {num_points} points")
            
        except Exception as e:
            print(f"Error adding texture coordinates: {e}")
    
    def _load_earth_texture(self) -> Optional[vtk.vtkTexture]:
        """
        Load Earth texture from file or create procedural texture.
        
        Returns:
            VTK texture or None if loading failed
        """
        try:
            # Try to find and load texture file
            for filename in Config.EARTH_TEXTURE_FILES:
                texture = self._try_load_texture_file(filename)
                if texture:
                    print(f"Successfully loaded Earth texture: {filename}")
                    return texture
            
            # No texture file found, create procedural texture
            print("No Earth texture file found, creating procedural texture...")
            return self._create_procedural_earth_texture()
            
        except Exception as e:
            print(f"Error loading Earth texture: {e}")
            return None
    
    def _try_load_texture_file(self, filename: str) -> Optional[vtk.vtkTexture]:
        """
        Try to load a specific texture file.
        
        Args:
            filename: Texture filename to try
            
        Returns:
            VTK texture or None if file doesn't exist
        """
        try:
            # Check in resources/textures folder first
            texture_paths = [
                Path("resources/textures") / filename,
                Path(filename),
                Path(".") / filename
            ]
            
            for path in texture_paths:
                if path.exists():
                    # Create appropriate reader
                    if path.suffix.lower() in ['.jpg', '.jpeg']:
                        reader = vtk.vtkJPEGReader()
                    elif path.suffix.lower() == '.png':
                        reader = vtk.vtkPNGReader()
                    else:
                        continue
                    
                    reader.SetFileName(str(path))
                    reader.Update()
                    
                    # Check if image loaded
                    if reader.GetOutput().GetNumberOfPoints() == 0:
                        continue
                    
                    # Create texture
                    texture = vtk.vtkTexture()
                    texture.SetInputConnection(reader.GetOutputPort())
                    texture.InterpolateOn()
                    
                    print(f"Loaded texture from: {path}")
                    return texture
            
            return None
            
        except Exception as e:
            return None
    
    def _create_procedural_earth_texture(self) -> vtk.vtkTexture:
        """
        Create a simple procedural Earth texture.
        
        Returns:
            Procedurally generated Earth texture
        """
        try:
            width = Config.EARTH_TEXTURE_WIDTH
            height = Config.EARTH_TEXTURE_HEIGHT
            
            # Create image data
            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
            
            # Create simple Earth pattern
            for y in range(height):
                for x in range(width):
                    # Convert to lat/lon
                    lon = (x / width) * 360 - 180
                    lat = 90 - (y / height) * 180
                    
                    # Simple continent check
                    is_land = self._simple_land_check(lon, lat)
                    
                    if is_land:
                        # Land colors (green/brown)
                        variation = (x + y) % 50 / 50.0
                        r = int(80 + variation * 60)
                        g = int(100 + variation * 80)
                        b = int(40 + variation * 40)
                    else:
                        # Ocean colors (blue)
                        variation = (x * 3 + y * 2) % 40 / 40.0
                        r = int(20 + variation * 30)
                        g = int(60 + variation * 60)
                        b = int(120 + variation * 80)
                    
                    # Set pixel
                    image.SetScalarComponentFromFloat(x, y, 0, 0, r)
                    image.SetScalarComponentFromFloat(x, y, 0, 1, g)
                    image.SetScalarComponentFromFloat(x, y, 0, 2, b)
            
            # Create texture
            texture = vtk.vtkTexture()
            texture.SetInputData(image)
            texture.InterpolateOn()
            
            print("Created procedural Earth texture")
            return texture
            
        except Exception as e:
            print(f"Error creating procedural texture: {e}")
            # Return a simple solid color texture as fallback
            return self._create_solid_color_texture(*Config.EARTH_DEFAULT_COLOR)
    
    def _simple_land_check(self, lon: float, lat: float) -> bool:
        """
        Simple land/ocean check for procedural texture.
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            
        Returns:
            True if position is land, False if ocean
        """
        # Very simplified continent shapes
        # North America
        if (-140 < lon < -60 and 15 < lat < 70):
            return True
        # South America  
        if (-80 < lon < -35 and -55 < lat < 15):
            return True
        # Africa
        if (-20 < lon < 50 and -35 < lat < 35):
            return True
        # Europe
        if (-10 < lon < 40 and 35 < lat < 70):
            return True
        # Asia
        if (40 < lon < 180 and 10 < lat < 75):
            return True
        # Australia
        if (110 < lon < 155 and -45 < lat < -10):
            return True
        
        return False
    
    def _create_solid_color_texture(self, r: float, g: float, b: float) -> vtk.vtkTexture:
        """
        Create a simple solid color texture as ultimate fallback.
        
        Args:
            r, g, b: Color components (0-1)
            
        Returns:
            Solid color texture
        """
        image = vtk.vtkImageData()
        image.SetDimensions(2, 2, 1)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        
        color = [int(r * 255), int(g * 255), int(b * 255)]
        
        for y in range(2):
            for x in range(2):
                for c in range(3):
                    image.SetScalarComponentFromFloat(x, y, 0, c, color[c])
        
        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        
        return texture
    
    def get_info(self) -> Dict[str, any]:
        """
        Get information about current Earth state.
        
        Returns:
            Dictionary with Earth renderer information
        """
        return {
            'earth_exists': self.earth_actor is not None,
            'grid_visible': self.grid_visible,
            'current_opacity': self.current_opacity,
            'num_grid_lines': len(self.lat_long_actors),
            'num_labels': len(self.grid_labels)
        }
