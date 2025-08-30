"""
Flux field renderer for various visualization modes
"""

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


class FluxFieldRenderer:
    """Handles rendering of flux field data with different visualization modes"""
    
    def __init__(self, renderer):
        """Initialize the flux field renderer
        
        Args:
            renderer: VTK renderer to add actors to
        """
        self.renderer = renderer
        self.field_actors = {}  # Dict of file_path -> actor
        self.field_data = {}    # Dict of file_path -> processed data
        self.visualization_mode = "Point Cloud"
        
        # Point cloud settings
        self.point_density = 10000
        self.point_size = 400  # meters radius
        
        # Isosurface settings
        self.isosurface_style = "Single Isosurface"
        self.isosurface_level = 50  # percentile
        self.multiple_levels = [20, 40, 60, 80]  # percentiles
        
        # Slice plane settings
        self.slice_axis = "Z-Axis (XY Plane)"
        self.slice_position = 50  # percentage
        
        # Scalar bar
        self.scalar_bar = None
        self.current_lut = None
        
    def add_flux_field(self, file_path, vtk_data, color_lut=None, opacity=0.8, min_flux=None, max_flux=None):
        """Add a flux field to the renderer
        
        Args:
            file_path: Path to the flux file (used as identifier)
            vtk_data: VTK dataset containing the flux field
            color_lut: Optional color lookup table
            opacity: Opacity value (0-1)
            min_flux: Minimum flux threshold (replaces zero values)
            max_flux: Maximum flux threshold (clips high values)
        """
        # Remove existing actor if present
        self.remove_flux_field(file_path)
        
        # Apply flux thresholds to data if specified
        processed_data = self._apply_flux_thresholds(vtk_data, min_flux, max_flux)
        
        # Store the data
        self.field_data[file_path] = processed_data
        
        # Create visualization based on current mode
        if self.visualization_mode == "Point Cloud":
            actor = self._create_point_cloud(processed_data, color_lut, opacity)
        elif self.visualization_mode == "Isosurfaces":
            actor = self._create_isosurfaces(processed_data, color_lut, opacity)
        elif self.visualization_mode == "Slice Planes":
            actor = self._create_slice_plane(processed_data, color_lut, opacity)
        else:
            # Fallback to simple visualization
            actor = self._create_simple_visualization(processed_data, color_lut, opacity)
        
        if actor:
            self.field_actors[file_path] = actor
            self.renderer.AddActor(actor)
            
            # Update scalar bar with the LUT from this field
            if color_lut:
                self._update_scalar_bar(color_lut, processed_data)
    
    def remove_flux_field(self, file_path):
        """Remove a flux field from the renderer
        
        Args:
            file_path: Path to the flux file (identifier)
        """
        if file_path in self.field_actors:
            actor = self.field_actors[file_path]
            self.renderer.RemoveActor(actor)
            del self.field_actors[file_path]
        
        if file_path in self.field_data:
            del self.field_data[file_path]
    
    def _create_point_cloud(self, vtk_data, color_lut, opacity):
        """Create point cloud visualization of flux field
        
        Args:
            vtk_data: VTK dataset containing the flux field
            color_lut: Color lookup table
            opacity: Opacity value
            
        Returns:
            VTK actor for the point cloud
        """
        # Get scalar data
        scalar_array = vtk_data.GetPointData().GetScalars()
        if not scalar_array:
            return None
        
        scalar_range = scalar_array.GetRange()
        num_points = vtk_data.GetNumberOfPoints()
        
        # Store original range for color mapping
        original_range = scalar_range
        
        # Filter based on density
        target_points = min(self.point_density, num_points)
        
        # Smart sampling based on flux values
        scalar_data = vtk_to_numpy(scalar_array)
        
        # Calculate sampling probability based on flux (higher flux = higher probability)
        min_flux = scalar_range[0]
        max_flux = scalar_range[1]
        
        if max_flux > min_flux:
            # Normalize to 0-1 range, but add small offset to avoid zero probabilities
            normalized = (scalar_data - min_flux) / (max_flux - min_flux)
            # Add base probability to ensure all points can be selected
            probabilities = 0.1 + 0.9 * np.sqrt(normalized)
            probabilities = probabilities / probabilities.sum()
        else:
            # Uniform sampling if we can't use flux-based sampling
            probabilities = np.ones(num_points) / num_points
        
        # Sample points
        if target_points < num_points:
            sample_indices = np.random.choice(
                num_points, 
                size=target_points, 
                replace=False,
                p=probabilities
            )
        else:
            sample_indices = np.arange(num_points)
        
        
        # Create new points and scalars for sampled data
        sampled_points = vtk.vtkPoints()
        sampled_scalars = vtk.vtkFloatArray()
        sampled_scalars.SetNumberOfComponents(1)
        sampled_scalars.SetName("flux")
        
        for idx in sample_indices:
            point = vtk_data.GetPoint(idx)
            scalar = scalar_array.GetValue(idx)
            
            # Only add points with positive flux values (skip zeros)
            if scalar > 0:
                # Add jitter to break up grid line-up artifacts
                jitter = np.random.normal(0, 200, 3)  # 200km jitter (increased from 50km)
                jittered_point = [point[i] + jitter[i] for i in range(3)]
                
                sampled_points.InsertNextPoint(jittered_point)
                sampled_scalars.InsertNextValue(scalar)
        
        # Fallback: if no points were added (all zeros), add some random points
        if sampled_points.GetNumberOfPoints() == 0:
            # Add first 100 points regardless of flux value
            num_fallback = min(100, num_points)
            for i in range(num_fallback):
                point = vtk_data.GetPoint(i)
                scalar = scalar_array.GetValue(i)
                sampled_points.InsertNextPoint(point)
                sampled_scalars.InsertNextValue(max(scalar, 1e-10))  # Ensure non-zero
        
        # Create polydata from sampled points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(sampled_points)
        polydata.GetPointData().SetScalars(sampled_scalars)
        
        # Create spheres for each point
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(self.point_size)
        sphere_source.SetThetaResolution(6)
        sphere_source.SetPhiResolution(6)
        
        # Use glyph3D to create spheres at each point
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetInputData(polydata)
        glyph3D.SetSourceConnection(sphere_source.GetOutputPort())
        glyph3D.SetScaleModeToDataScalingOff()
        glyph3D.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph3D.GetOutputPort())
        mapper.SetScalarRange(original_range)
        
        # Set color lookup table
        if color_lut:
            mapper.SetLookupTable(color_lut)
        else:
            # Create default lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue to red
            lut.SetRange(original_range)
            lut.Build()
            mapper.SetLookupTable(lut)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        
        actual_points = sampled_points.GetNumberOfPoints()
        
        if actual_points == 0:
            return None
        
        return actor
    
    def _create_simple_visualization(self, vtk_data, color_lut, opacity):
        """Create simple visualization as fallback
        
        Args:
            vtk_data: VTK dataset
            color_lut: Color lookup table
            opacity: Opacity value
            
        Returns:
            VTK actor
        """
        # Simple threshold filter
        scalar_range = vtk_data.GetScalarRange()
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(vtk_data)
        threshold.SetLowerThreshold(scalar_range[1] * 0.01)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
        threshold.Update()
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(threshold.GetOutputPort())
        mapper.SetScalarRange(scalar_range)
        
        if color_lut:
            mapper.SetLookupTable(color_lut)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(5)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor
    
    def set_visualization_mode(self, mode):
        """Change the visualization mode
        
        Args:
            mode: Visualization mode ("Point Cloud", etc.)
        """
        if mode != self.visualization_mode:
            self.visualization_mode = mode
            # Recreate all visualizations with new mode
            self.refresh_all()
    
    def set_point_density(self, density):
        """Set the point cloud density
        
        Args:
            density: Number of points to display
        """
        if density != self.point_density:
            self.point_density = density
            if self.visualization_mode == "Point Cloud":
                self.refresh_all()
    
    def set_point_size(self, size_meters):
        """Set the point size in meters
        
        Args:
            size_meters: Radius of points in meters
        """
        if size_meters != self.point_size:
            self.point_size = size_meters
            # Fast update for point size (just update glyph scale)
            for file_path, actor in self.field_actors.items():
                if self.visualization_mode == "Point Cloud":
                    # Recreate with new size
                    vtk_data = self.field_data[file_path]
                    mapper = actor.GetMapper()
                    lut = mapper.GetLookupTable() if mapper else None
                    opacity = actor.GetProperty().GetOpacity()
                    
                    new_actor = self._create_point_cloud(vtk_data, lut, opacity)
                    if new_actor:
                        self.renderer.RemoveActor(actor)
                        self.renderer.AddActor(new_actor)
                        self.field_actors[file_path] = new_actor
    
    def refresh_all(self):
        """Refresh all flux field visualizations"""
        # Store current actors and settings
        actors_to_refresh = list(self.field_actors.items())
        
        for file_path, actor in actors_to_refresh:
            if file_path in self.field_data:
                vtk_data = self.field_data[file_path]
                mapper = actor.GetMapper()
                lut = mapper.GetLookupTable() if mapper else None
                opacity = actor.GetProperty().GetOpacity()
                
                # Recreate the visualization
                self.add_flux_field(file_path, vtk_data, lut, opacity)
    
    def update_opacity(self, opacity):
        """Update opacity for all flux field actors
        
        Args:
            opacity: Opacity value (0-1)
        """
        for actor in self.field_actors.values():
            actor.GetProperty().SetOpacity(opacity)
    
    def toggle_visibility(self, visible):
        """Toggle visibility of all flux fields
        
        Args:
            visible: True to show, False to hide
        """
        for actor in self.field_actors.values():
            actor.SetVisibility(visible)
    
    def clear_all(self):
        """Remove all flux field visualizations"""
        for file_path in list(self.field_actors.keys()):
            self.remove_flux_field(file_path)
        
        # Remove scalar bar
        self._remove_scalar_bar()
    
    def _apply_flux_thresholds(self, vtk_data, min_flux=None, max_flux=None):
        """Apply flux thresholds to the data
        
        Args:
            vtk_data: Original VTK dataset
            min_flux: Minimum flux threshold (replaces values <= 0)
            max_flux: Maximum flux threshold (clips high values)
            
        Returns:
            VTK dataset with modified scalar values
        """
        if min_flux is None and max_flux is None:
            return vtk_data
        
        # Create a copy of the data
        new_data = vtk_data.NewInstance()
        new_data.DeepCopy(vtk_data)
        
        # Get scalar array
        scalar_array = new_data.GetPointData().GetScalars()
        if not scalar_array:
            return new_data
        
        # Convert to numpy for processing
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        scalar_np = vtk_to_numpy(scalar_array)
        modified_scalars = scalar_np.copy()
        
        # Apply minimum flux threshold (replace zero/negative values)
        if min_flux is not None:
            # Replace values <= 0 with min_flux (realistic minimum for space radiation)
            zero_mask = modified_scalars <= 0
            modified_scalars[zero_mask] = min_flux
            print(f"Applied min flux threshold: {zero_mask.sum()} zero values set to {min_flux}")
        
        # Apply maximum flux threshold (clip high values)
        if max_flux is not None:
            # Clip values above max_flux
            clipped_mask = modified_scalars > max_flux
            modified_scalars[clipped_mask] = max_flux
            if clipped_mask.sum() > 0:
                print(f"Applied max flux threshold: {clipped_mask.sum()} values clipped to {max_flux}")
        
        # Convert back to VTK and update the array
        new_scalar_array = numpy_to_vtk(modified_scalars)
        new_scalar_array.SetName(scalar_array.GetName())
        new_data.GetPointData().SetScalars(new_scalar_array)
        
        return new_data
    
    def _update_scalar_bar(self, lut, vtk_data):
        """Update or create the scalar bar
        
        Args:
            lut: VTK lookup table
            vtk_data: VTK dataset for getting scalar range and name
        """
        # Remove existing scalar bar if present
        self._remove_scalar_bar()
        
        # Get scalar array info
        scalar_array = vtk_data.GetPointData().GetScalars()
        if not scalar_array:
            return
        
        scalar_name = scalar_array.GetName() if scalar_array.GetName() else "Flux"
        scalar_range = scalar_array.GetRange()
        
        # With flux thresholding, we should no longer have zero values,
        # but keep this as a safety check
        if scalar_range[0] <= 0:
            if scalar_range[1] > 0:
                # This shouldn't happen with proper thresholding, but handle it
                print(f"Warning: Found zero/negative values in processed data: {scalar_range}")
                min_val = max(1e2, scalar_range[1] * 1e-6)  # Use realistic minimum
                lut.SetRange(min_val, scalar_range[1])
                scalar_range = (min_val, scalar_range[1])
            else:
                return
        
        # Create scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(lut)
        
        # Set title and format
        self.scalar_bar.SetTitle(f"{scalar_name}")
        self.scalar_bar.GetTitleTextProperty().SetFontSize(14)
        self.scalar_bar.GetTitleTextProperty().SetBold(True)
        
        # Position on right side of screen
        self.scalar_bar.SetPosition(0.85, 0.1)  # Right side, bottom margin
        self.scalar_bar.SetWidth(0.12)          # 12% of screen width
        self.scalar_bar.SetHeight(0.8)          # 80% of screen height
        
        # Format the labels
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.GetLabelTextProperty().SetFontSize(10)
        
        # Use scientific notation for large numbers
        if scalar_range[1] > 1e6 or scalar_range[1] < 1e-3:
            self.scalar_bar.SetLabelFormat("%.2e")
        else:
            self.scalar_bar.SetLabelFormat("%.3g")
        
        # Add to renderer
        self.renderer.AddActor2D(self.scalar_bar)
        self.current_lut = lut
    
    def _remove_scalar_bar(self):
        """Remove the scalar bar from the renderer"""
        if self.scalar_bar:
            self.renderer.RemoveActor2D(self.scalar_bar)
            self.scalar_bar = None
            self.current_lut = None
    
    def _create_isosurfaces(self, vtk_data, color_lut, opacity):
        """Create isosurface visualization of flux field
        
        Args:
            vtk_data: VTK dataset containing the flux field
            color_lut: Color lookup table  
            opacity: Opacity value
            
        Returns:
            VTK actor for the isosurfaces
        """
        # Get scalar data
        scalar_array = vtk_data.GetPointData().GetScalars()
        if not scalar_array:
            return None
        
        scalar_range = scalar_array.GetRange()
        if scalar_range[1] <= scalar_range[0]:
            return None
        
        # Create contour filter for isosurfaces
        contour = vtk.vtkContourFilter()
        contour.SetInputData(vtk_data)
        
        # Generate isosurface levels based on mode
        if self.isosurface_style == "Single Isosurface":
            # Single isosurface at specified level
            contour.SetNumberOfContours(1)
            iso_value = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * (self.isosurface_level / 100.0)
            contour.SetValue(0, iso_value)
        else:
            # Multiple isosurfaces at specified levels
            levels = self.multiple_levels
            contour.SetNumberOfContours(len(levels))
            
            for i, level_pct in enumerate(levels):
                iso_value = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * (level_pct / 100.0)
                contour.SetValue(i, iso_value)
        
        contour.Update()
        
        contour_output = contour.GetOutput()
        if contour_output.GetNumberOfPoints() == 0:
            return None
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(contour_output)
        
        # Apply color mapping
        if color_lut:
            mapper.SetLookupTable(color_lut)
            mapper.SetScalarRange(scalar_range)
            mapper.SetScalarModeToUsePointData()
            mapper.ColorByArrayComponent("flux", 0)  # Color by flux values
        else:
            # Fallback to solid color
            mapper.ScalarVisibilityOff()
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set properties for isosurface appearance
        prop = actor.GetProperty()
        prop.SetOpacity(opacity)
        prop.SetRepresentationToSurface()
        
        if not color_lut:
            # Default color for isosurfaces (blue-green)
            prop.SetColor(0.2, 0.8, 0.8)
        
        # Add some ambient lighting for better visibility
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        
        return actor
    
    def set_isosurface_style(self, style):
        """Set the isosurface style
        
        Args:
            style: "Single Isosurface" or "Multiple Isosurface"
        """
        if style != self.isosurface_style:
            self.isosurface_style = style
            if self.visualization_mode == "Isosurfaces":
                self.refresh_all()
    
    def set_isosurface_level(self, level_percent):
        """Set the single isosurface level
        
        Args:
            level_percent: Percentile level (10-90)
        """
        if level_percent != self.isosurface_level:
            self.isosurface_level = level_percent
            if self.visualization_mode == "Isosurfaces" and self.isosurface_style == "Single Isosurface":
                self.refresh_all()
    
    def set_multiple_isosurface_levels(self, levels):
        """Set the multiple isosurface levels
        
        Args:
            levels: List of percentile levels
        """
        if levels != self.multiple_levels:
            self.multiple_levels = levels
            if self.visualization_mode == "Isosurfaces" and self.isosurface_style == "Multiple Isosurface":
                self.refresh_all()
    
    def _create_slice_plane(self, vtk_data, color_lut, opacity):
        """Create slice plane visualization of flux field
        
        Args:
            vtk_data: VTK dataset containing the flux field
            color_lut: Color lookup table
            opacity: Opacity value
            
        Returns:
            VTK actor for the slice plane
        """
        # Get data bounds
        bounds = vtk_data.GetBounds()
        
        # Determine slice orientation and position
        axis_text = self.slice_axis
        position_percent = self.slice_position / 100.0
        
        # Set up plane normal and origin based on axis
        if "X-Axis" in axis_text:
            normal = [1, 0, 0]
            origin_coord = bounds[0] + position_percent * (bounds[1] - bounds[0])
            origin = [origin_coord, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
        elif "Y-Axis" in axis_text:
            normal = [0, 1, 0]
            origin_coord = bounds[2] + position_percent * (bounds[3] - bounds[2])
            origin = [(bounds[0]+bounds[1])/2, origin_coord, (bounds[4]+bounds[5])/2]
        else:  # Z-Axis (default)
            normal = [0, 0, 1]
            origin_coord = bounds[4] + position_percent * (bounds[5] - bounds[4])
            origin = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, origin_coord]
        
        # Create a high-resolution plane for interpolation
        plane_source = vtk.vtkPlaneSource()
        
        # Calculate plane dimensions based on bounds and orientation
        if "X-Axis" in axis_text:
            # YZ plane
            width = bounds[3] - bounds[2]   # Y extent
            height = bounds[5] - bounds[4]  # Z extent
            plane_source.SetOrigin(origin[0], bounds[2], bounds[4])
            plane_source.SetPoint1(origin[0], bounds[3], bounds[4])
            plane_source.SetPoint2(origin[0], bounds[2], bounds[5])
        elif "Y-Axis" in axis_text:
            # XZ plane
            width = bounds[1] - bounds[0]   # X extent
            height = bounds[5] - bounds[4]  # Z extent
            plane_source.SetOrigin(bounds[0], origin[1], bounds[4])
            plane_source.SetPoint1(bounds[1], origin[1], bounds[4])
            plane_source.SetPoint2(bounds[0], origin[1], bounds[5])
        else:  # Z-Axis
            # XY plane
            width = bounds[1] - bounds[0]   # X extent
            height = bounds[3] - bounds[2]  # Y extent
            plane_source.SetOrigin(bounds[0], bounds[2], origin[2])
            plane_source.SetPoint1(bounds[1], bounds[2], origin[2])
            plane_source.SetPoint2(bounds[0], bounds[3], origin[2])
        
        # Set resolution for smooth interpolation (higher = smoother)
        resolution = 200  # Adjust this for quality vs performance
        plane_source.SetXResolution(resolution)
        plane_source.SetYResolution(resolution)
        plane_source.Update()
        
        # Use probe filter to interpolate data onto the plane
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(plane_source.GetOutput())
        probe.SetSourceData(vtk_data)
        probe.Update()
        
        slice_data = probe.GetOutput()
        
        # Check if slice has data
        if slice_data.GetNumberOfPoints() == 0:
            return None
        
        # Create mapper for the slice
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(slice_data)
        
        # Apply color mapping
        if color_lut:
            scalar_array = vtk_data.GetPointData().GetScalars()
            if scalar_array:
                scalar_range = scalar_array.GetRange()
                mapper.SetLookupTable(color_lut)
                mapper.SetScalarRange(scalar_range)
                mapper.SetScalarModeToUsePointData()
                mapper.ColorByArrayComponent(scalar_array.GetName(), 0)
        else:
            mapper.ScalarVisibilityOff()
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set properties
        prop = actor.GetProperty()
        prop.SetOpacity(opacity)
        prop.SetRepresentationToSurface()
        
        if not color_lut:
            # Default color for slices (cyan)
            prop.SetColor(0.0, 0.8, 0.8)
        
        # Add some lighting
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.8)
        
        return actor
    
    def set_slice_axis(self, axis):
        """Set the slice plane axis/orientation
        
        Args:
            axis: Axis text like "Z-Axis (XY Plane)"
        """
        if axis != self.slice_axis:
            self.slice_axis = axis
            if self.visualization_mode == "Slice Planes":
                self.refresh_all()
    
    def set_slice_position(self, position_percent):
        """Set the slice plane position
        
        Args:
            position_percent: Position percentage (0-100)
        """
        if position_percent != self.slice_position:
            self.slice_position = position_percent
            if self.visualization_mode == "Slice Planes":
                self.refresh_all()