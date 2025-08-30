"""
Starfield background renderer for STRATOS
Creates a procedural starfield background sphere
"""

import vtk
import numpy as np
import random
from pathlib import Path


class StarfieldRenderer:
    """Handles procedural starfield background rendering"""
    
    def __init__(self, renderer):
        """Initialize the starfield renderer
        
        Args:
            renderer: VTK renderer to add starfield to
        """
        self.renderer = renderer
        self.starfield_actor = None
        
    def create_starfield_background(self):
        """Create a realistic starfield background sphere"""
        try:
            print("Creating starfield background...")
            
            # Remove existing starfield if it exists
            if self.starfield_actor:
                self.renderer.RemoveActor(self.starfield_actor)
            
            # Create a very large sphere to contain the scene
            starfield_radius = 500000.0  # 500,000 km - much larger than data bounds
            
            starfield_sphere = vtk.vtkSphereSource()
            starfield_sphere.SetRadius(starfield_radius)
            starfield_sphere.SetThetaResolution(120)  # High resolution for smooth stars
            starfield_sphere.SetPhiResolution(120)
            starfield_sphere.SetCenter(0.0, 0.0, 0.0)
            starfield_sphere.Update()
            
            # Get the sphere data
            sphere_data = starfield_sphere.GetOutput()
            
            # Add texture coordinates for star map
            self._add_texture_coordinates(sphere_data)
            
            # Create mapper and actor
            starfield_mapper = vtk.vtkPolyDataMapper()
            starfield_mapper.SetInputData(sphere_data)
            
            self.starfield_actor = vtk.vtkActor()
            self.starfield_actor.SetMapper(starfield_mapper)
            
            # Try to load starfield texture, fallback to procedural
            star_texture = self._load_starfield_texture()
            if star_texture:
                self.starfield_actor.SetTexture(star_texture)
                print("Applied starfield texture")
            
            # Set starfield properties
            starfield_property = self.starfield_actor.GetProperty()
            starfield_property.SetRepresentationToSurface()
            starfield_property.SetAmbient(1.0)  # Fully ambient (self-illuminated)
            starfield_property.SetDiffuse(0.0)  # No diffuse lighting
            starfield_property.SetSpecular(0.0)  # No specular highlights
            
            # Render on the inside of the sphere
            starfield_property.BackfaceCullingOff()
            starfield_property.FrontfaceCullingOn()
            
            # Add to renderer with lowest priority (renders first, behind everything)
            self.starfield_actor.SetVisibility(True)
            self.renderer.AddActor(self.starfield_actor)
            
            print(f"Starfield background created successfully (radius: {starfield_radius/1000:.0f}k km)")
            return True
            
        except Exception as e:
            print(f"Error creating starfield background: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_texture_coordinates(self, sphere_data):
        """Add texture coordinates for starfield mapping"""
        try:
            points = sphere_data.GetPoints()
            num_points = points.GetNumberOfPoints()
            
            # Create texture coordinates array
            tcoords = vtk.vtkFloatArray()
            tcoords.SetNumberOfComponents(2)
            tcoords.SetNumberOfTuples(num_points)
            tcoords.SetName("TextureCoordinates")
            
            print(f"Adding texture coordinates for {num_points} points")
            
            for i in range(num_points):
                point = points.GetPoint(i)
                x, y, z = point
                
                # Convert Cartesian to spherical coordinates
                radius = np.sqrt(x*x + y*y + z*z)
                if radius > 0:
                    # Longitude (phi): -π to π mapped to 0 to 1
                    phi = np.arctan2(y, x)
                    u = (phi + np.pi) / (2 * np.pi)
                    
                    # Latitude (theta): π to 0 mapped to 0 to 1
                    theta = np.arccos(np.clip(z / radius, -1, 1))
                    v = theta / np.pi
                    
                    tcoords.SetTuple2(i, u, v)
                else:
                    tcoords.SetTuple2(i, 0, 0)
            
            sphere_data.GetPointData().SetTCoords(tcoords)
            print(f"Added texture coordinates for {num_points} points")
            
        except Exception as e:
            print(f"Error adding texture coordinates: {e}")
    
    def _load_starfield_texture(self):
        """Load starfield texture from file or create procedural texture"""
        try:
            # Try to find starfield.jpg in data/textures
            texture_paths = [
                Path("data/textures/starfield.jpg"),
                Path("flux_visualizer/data/textures/starfield.jpg"),
                Path("resources/textures/starfield.jpg"),
                Path("starfield.jpg")
            ]
            
            for path in texture_paths:
                if path.exists():
                    print(f"Loading starfield texture from: {path}")
                    
                    # Create JPEG reader
                    reader = vtk.vtkJPEGReader()
                    reader.SetFileName(str(path))
                    reader.Update()
                    
                    # Check if image loaded
                    if reader.GetOutput().GetNumberOfPoints() > 0:
                        # Create texture
                        texture = vtk.vtkTexture()
                        texture.SetInputConnection(reader.GetOutputPort())
                        texture.InterpolateOn()
                        texture.RepeatOff()
                        texture.EdgeClampOn()
                        
                        print(f"Successfully loaded starfield texture: {path}")
                        return texture
            
            # No texture file found, create procedural
            print("No starfield.jpg found, creating procedural starfield...")
            return self._create_procedural_starfield_texture()
            
        except Exception as e:
            print(f"Error loading starfield texture: {e}")
            return self._create_procedural_starfield_texture()

    def _create_procedural_starfield_texture(self):
        """Create a procedural starfield texture"""
        try:
            print("Creating procedural starfield texture...")
            
            # Create high-resolution starfield
            width, height = 2048, 1024  # Equirectangular format
            
            # Create image data
            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)  # RGB
            
            # Generate realistic starfield
            random.seed(42)  # Reproducible starfield
            
            # Background space color (very dark blue-black)
            bg_color = [5, 5, 15]
            
            # Fill background
            for y in range(height):
                for x in range(width):
                    image.SetScalarComponentFromFloat(x, y, 0, 0, bg_color[0])
                    image.SetScalarComponentFromFloat(x, y, 0, 1, bg_color[1])
                    image.SetScalarComponentFromFloat(x, y, 0, 2, bg_color[2])
            
            # Add stars with realistic distribution
            num_stars = 8000  # Dense starfield
            
            for _ in range(num_stars):
                # Random position
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                
                # Star magnitude (brightness) - weighted toward dimmer stars
                magnitude = random.random() ** 3  # Cube for realistic distribution
                
                # Star color - most stars are white/blue-white, some yellow/red
                color_type = random.random()
                if color_type < 0.7:  # Blue-white stars (most common)
                    base_color = [200, 210, 255]
                elif color_type < 0.9:  # Yellow stars (like our Sun)
                    base_color = [255, 245, 200]
                else:  # Red stars
                    base_color = [255, 200, 150]
                
                # Apply magnitude
                star_color = [int(c * magnitude) for c in base_color]
                star_color = [max(0, min(255, c)) for c in star_color]
                
                # Set star pixel
                image.SetScalarComponentFromFloat(x, y, 0, 0, star_color[0])
                image.SetScalarComponentFromFloat(x, y, 0, 1, star_color[1])
                image.SetScalarComponentFromFloat(x, y, 0, 2, star_color[2])
                
                # Add some bright stars with glow
                if magnitude > 0.95:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            gx, gy = x + dx, y + dy
                            if 0 <= gx < width and 0 <= gy < height:
                                glow_intensity = 0.3 * magnitude
                                current_r = image.GetScalarComponentAsFloat(gx, gy, 0, 0)
                                current_g = image.GetScalarComponentAsFloat(gx, gy, 0, 1)
                                current_b = image.GetScalarComponentAsFloat(gx, gy, 0, 2)
                                
                                new_r = min(255, current_r + star_color[0] * glow_intensity)
                                new_g = min(255, current_g + star_color[1] * glow_intensity)
                                new_b = min(255, current_b + star_color[2] * glow_intensity)
                                
                                image.SetScalarComponentFromFloat(gx, gy, 0, 0, new_r)
                                image.SetScalarComponentFromFloat(gx, gy, 0, 1, new_g)
                                image.SetScalarComponentFromFloat(gx, gy, 0, 2, new_b)
            
            # Add some nebulosity (faint background glow)
            for _ in range(50):
                center_x = random.randint(0, width - 1)
                center_y = random.randint(0, height - 1)
                nebula_radius = random.randint(20, 100)
                nebula_intensity = random.random() * 0.1
                
                # Nebula color (faint blue/purple)
                nebula_color = [20, 30, 60]
                
                for dx in range(-nebula_radius, nebula_radius):
                    for dy in range(-nebula_radius, nebula_radius):
                        nx, ny = center_x + dx, center_y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < nebula_radius:
                                fade = (1.0 - distance / nebula_radius) * nebula_intensity
                                
                                current_r = image.GetScalarComponentAsFloat(nx, ny, 0, 0)
                                current_g = image.GetScalarComponentAsFloat(nx, ny, 0, 1)
                                current_b = image.GetScalarComponentAsFloat(nx, ny, 0, 2)
                                
                                new_r = min(255, current_r + nebula_color[0] * fade)
                                new_g = min(255, current_g + nebula_color[1] * fade)
                                new_b = min(255, current_b + nebula_color[2] * fade)
                                
                                image.SetScalarComponentFromFloat(nx, ny, 0, 0, new_r)
                                image.SetScalarComponentFromFloat(nx, ny, 0, 1, new_g)
                                image.SetScalarComponentFromFloat(nx, ny, 0, 2, new_b)
            
            # Create texture
            texture = vtk.vtkTexture()
            texture.SetInputData(image)
            texture.InterpolateOn()
            texture.RepeatOff()
            texture.EdgeClampOn()
            
            print("Procedural starfield texture created successfully")
            return texture
            
        except Exception as e:
            print(f"Error creating procedural starfield: {e}")
            return None
    
    def set_visibility(self, visible):
        """Toggle starfield visibility"""
        if self.starfield_actor:
            self.starfield_actor.SetVisibility(visible)
    
    def cleanup(self):
        """Remove starfield from renderer"""
        if self.starfield_actor:
            self.renderer.RemoveActor(self.starfield_actor)
            self.starfield_actor = None