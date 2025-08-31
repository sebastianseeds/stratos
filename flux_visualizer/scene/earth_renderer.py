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
            
            # Create custom sphere geometry to avoid texture seams
            sphere_data = self._create_seamless_sphere()
            
            # Create mapper and actor
            earth_mapper = vtk.vtkPolyDataMapper()
            earth_mapper.SetInputData(sphere_data)
            
            self.earth_actor = vtk.vtkActor()
            self.earth_actor.SetMapper(earth_mapper)
            
            # Try to apply Earth texture
            earth_texture = self._load_earth_texture()
            if earth_texture:
                # Configure texture for seamless wrapping
                earth_texture.SetRepeat(0)       # Don't tile/repeat
                earth_texture.SetInterpolate(1)  # Enable smooth filtering
                earth_texture.EdgeClampOn()      # Clamp to edge to prevent seams
                
                # Ensure the texture uses proper mirroring behavior at boundaries
                earth_texture.SetWrap(vtk.vtkTexture.ClampToEdge)
                
                self.earth_actor.SetTexture(earth_texture)
                print("Earth texture applied with seamless settings")
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
        Add texture coordinates to sphere for texture mapping with seamless wrapping.
        
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
                
                if r > 0:
                    # Calculate longitude (phi): -π to π, then map to 0-1
                    longitude = np.arctan2(y, x)
                    u = (longitude + np.pi) / (2 * np.pi)
                    
                    # Calculate latitude (theta): -π/2 to π/2, then map to 0-1  
                    latitude = np.arcsin(np.clip(z / r, -1.0, 1.0))
                    v = (latitude + np.pi/2) / np.pi
                    
                    # For seamless wrapping, don't clip coordinates too aggressively
                    # Only prevent exact 0 and 1 which can cause issues at poles
                    u = np.clip(u, 0.001, 0.999)
                    v = np.clip(v, 0.001, 0.999)
                else:
                    u, v = 0.5, 0.5  # Center for degenerate points
                
                tex_coords.SetTuple2(i, u, v)
            
            sphere_data.GetPointData().SetTCoords(tex_coords)
            print(f"Added texture coordinates for {num_points} points")
            
        except Exception as e:
            print(f"Error adding texture coordinates: {e}")
    
    def _create_seamless_sphere(self) -> vtk.vtkPolyData:
        """
        Create a sphere with explicit geometry that avoids texture seams.
        
        Returns:
            VTK polydata containing seamless sphere geometry
        """
        try:
            print("Creating seamless sphere geometry...")
            
            # Parameters
            radius = Config.EARTH_RADIUS_KM
            theta_res = Config.EARTH_SPHERE_RESOLUTION_THETA  # Longitude divisions
            phi_res = Config.EARTH_SPHERE_RESOLUTION_PHI      # Latitude divisions
            
            # Create points and texture coordinates
            points = vtk.vtkPoints()
            tex_coords = vtk.vtkFloatArray()
            tex_coords.SetNumberOfComponents(2)
            tex_coords.SetName("TextureCoordinates")
            
            # Create cells for triangulation
            polys = vtk.vtkCellArray()
            
            # Generate vertices
            # Note: We add an extra column of vertices for longitude to avoid sharing
            # vertices at 0° and 360°, which causes texture seams
            for j in range(phi_res + 1):  # Latitude from north pole to south pole
                phi = np.pi * j / phi_res  # 0 to π
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                
                for i in range(theta_res + 1):  # Longitude including wrap-around
                    theta = 2.0 * np.pi * i / theta_res  # 0 to 2π
                    sin_theta = np.sin(theta)
                    cos_theta = np.cos(theta)
                    
                    # Spherical to Cartesian coordinates
                    x = radius * sin_phi * cos_theta
                    y = radius * sin_phi * sin_theta
                    z = radius * cos_phi
                    
                    points.InsertNextPoint(x, y, z)
                    
                    # Texture coordinates (flip only vertically for proper pole orientation)
                    u = float(i) / theta_res         # 0 to 1 (normal horizontal)
                    v = 1.0 - float(j) / phi_res    # 1 to 0 (flipped vertically)
                    tex_coords.InsertNextTuple2(u, v)
            
            # Create triangular faces
            for j in range(phi_res):
                for i in range(theta_res):
                    # Current vertex indices
                    p1 = j * (theta_res + 1) + i
                    p2 = p1 + 1
                    p3 = (j + 1) * (theta_res + 1) + i
                    p4 = p3 + 1
                    
                    # Skip degenerate triangles at poles
                    if j > 0:  # Not north pole
                        triangle1 = vtk.vtkTriangle()
                        triangle1.GetPointIds().SetId(0, p1)
                        triangle1.GetPointIds().SetId(1, p2)
                        triangle1.GetPointIds().SetId(2, p3)
                        polys.InsertNextCell(triangle1)
                    
                    if j < phi_res - 1:  # Not south pole
                        triangle2 = vtk.vtkTriangle()
                        triangle2.GetPointIds().SetId(0, p2)
                        triangle2.GetPointIds().SetId(1, p4)
                        triangle2.GetPointIds().SetId(2, p3)
                        polys.InsertNextCell(triangle2)
            
            # Create polydata
            sphere_data = vtk.vtkPolyData()
            sphere_data.SetPoints(points)
            sphere_data.SetPolys(polys)
            sphere_data.GetPointData().SetTCoords(tex_coords)
            
            # Compute normals for proper lighting
            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(sphere_data)
            normals_filter.ComputePointNormalsOn()
            normals_filter.ComputeCellNormalsOff()
            normals_filter.SplittingOff()  # Don't split vertices
            normals_filter.Update()
            
            final_sphere = normals_filter.GetOutput()
            
            print(f"Created seamless sphere: {points.GetNumberOfPoints()} vertices, "
                  f"{polys.GetNumberOfCells()} faces")
            
            return final_sphere
            
        except Exception as e:
            print(f"Error creating seamless sphere: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to VTK sphere
            print("Falling back to VTK sphere source...")
            fallback_sphere = vtk.vtkSphereSource()
            fallback_sphere.SetRadius(Config.EARTH_RADIUS_KM)
            fallback_sphere.SetThetaResolution(Config.EARTH_SPHERE_RESOLUTION_THETA)
            fallback_sphere.SetPhiResolution(Config.EARTH_SPHERE_RESOLUTION_PHI)
            fallback_sphere.SetCenter(0.0, 0.0, 0.0)
            fallback_sphere.Update()
            
            sphere_data = fallback_sphere.GetOutput()
            self._add_proper_texture_coordinates(sphere_data)
            return sphere_data

    def _add_proper_texture_coordinates(self, sphere_data: vtk.vtkPolyData) -> None:
        """
        Add texture coordinates with a clean, simple approach to avoid seams.
        
        Args:
            sphere_data: VTK sphere polydata
        """
        try:
            points = sphere_data.GetPoints()
            num_points = points.GetNumberOfPoints()
            
            # Create texture coordinates
            tex_coords = vtk.vtkFloatArray()
            tex_coords.SetNumberOfComponents(2)
            tex_coords.SetNumberOfTuples(num_points)
            tex_coords.SetName("TextureCoordinates")
            
            for i in range(num_points):
                point = points.GetPoint(i)
                x, y, z = point
                
                # Convert to spherical coordinates
                r = np.sqrt(x*x + y*y + z*z)
                
                if r > 0:
                    # Longitude: atan2(y, x) gives -π to π
                    # Map to 0-1 for texture coordinate
                    lon = np.arctan2(y, x)
                    u = (lon + np.pi) / (2.0 * np.pi)
                    
                    # Latitude: asin(z/r) gives -π/2 to π/2  
                    # Map to 0-1 for texture coordinate
                    lat = np.asin(np.clip(z / r, -1.0, 1.0))
                    v = (lat + np.pi/2.0) / np.pi
                    
                    # No aggressive clipping - let the texture wrap naturally
                    tex_coords.SetTuple2(i, u, v)
                else:
                    # Degenerate point at origin
                    tex_coords.SetTuple2(i, 0.5, 0.5)
            
            # Set the texture coordinates
            sphere_data.GetPointData().SetTCoords(tex_coords)
            print(f"Added proper texture coordinates for {num_points} points")
            
        except Exception as e:
            print(f"Error adding proper texture coordinates: {e}")
    
    def _use_vtk_texture_coordinates(self, sphere_data: vtk.vtkPolyData) -> None:
        """
        Use VTK's coordinate generation but fix the mirroring issue.
        
        Args:
            sphere_data: VTK sphere polydata
        """
        try:
            # Use VTK's texture map to sphere filter
            texture_map = vtk.vtkTextureMapToSphere()
            texture_map.SetInputData(sphere_data)
            texture_map.PreventSeamOn()  # Prevent seams
            texture_map.Update()
            
            # Get the generated coordinates
            mapped_data = texture_map.GetOutput()
            vtk_coords = mapped_data.GetPointData().GetTCoords()
            
            if vtk_coords:
                # Create our own corrected coordinates
                points = sphere_data.GetPoints()
                num_points = points.GetNumberOfPoints()
                
                corrected_coords = vtk.vtkFloatArray()
                corrected_coords.SetNumberOfComponents(2)
                corrected_coords.SetNumberOfTuples(num_points)
                corrected_coords.SetName("TextureCoordinates")
                
                for i in range(num_points):
                    # Get VTK's coordinates
                    u, v = vtk_coords.GetTuple2(i)
                    
                    # Get the actual 3D point to determine correct longitude
                    point = points.GetPoint(i)
                    x, y, z = point
                    
                    # Calculate the correct longitude ourselves
                    longitude = np.arctan2(y, x)
                    corrected_u = (longitude + np.pi) / (2 * np.pi)
                    
                    # Use VTK's latitude (v) which is usually correct
                    # But clamp to avoid seam issues
                    corrected_u = np.clip(corrected_u, 0.001, 0.999)
                    corrected_v = np.clip(v, 0.001, 0.999)
                    
                    corrected_coords.SetTuple2(i, corrected_u, corrected_v)
                
                sphere_data.GetPointData().SetTCoords(corrected_coords)
                print(f"Applied corrected VTK texture mapping to {num_points} points")
            else:
                raise Exception("VTK failed to generate texture coordinates")
            
        except Exception as e:
            print(f"Error with VTK texture mapping, falling back to manual method: {e}")
            # Fallback to our manual method
            self._add_texture_coordinates_fallback(sphere_data)
    
    def _add_texture_coordinates_fallback(self, sphere_data: vtk.vtkPolyData) -> None:
        """
        Fallback manual texture coordinate generation.
        
        Args:
            sphere_data: VTK sphere polydata
        """
        # This is our previous manual method as fallback
        self._add_texture_coordinates(sphere_data)
    
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
            # Get flux_visualizer directory
            flux_viz_dir = Path(__file__).parent.parent     # Go up from scene/ to flux_visualizer/
            
            # Textures are always in flux_visualizer/textures/
            texture_path = flux_viz_dir / "textures" / filename
            
            if texture_path.exists():
                # Create appropriate reader
                if texture_path.suffix.lower() in ['.jpg', '.jpeg']:
                    reader = vtk.vtkJPEGReader()
                elif texture_path.suffix.lower() == '.png':
                    reader = vtk.vtkPNGReader()
                else:
                    return None
                
                reader.SetFileName(str(texture_path))
                reader.Update()
                
                # Check if image loaded
                if reader.GetOutput().GetNumberOfPoints() == 0:
                    return None
                
                # Create texture
                texture = vtk.vtkTexture()
                texture.SetInputConnection(reader.GetOutputPort())
                texture.InterpolateOn()
                
                print(f"Loaded texture from: {texture_path}")
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
