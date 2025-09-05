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
        self.field_particle_types = {}  # Dict of file_path -> particle type
        self.time_dependent_data = {}  # Dict of file_path -> multiblock data
        self.time_info = {}     # Dict of file_path -> time information
        self.current_time_index = {}  # Dict of file_path -> current time index
        self.visualization_mode = "Point Cloud"
        
        # Point cloud settings
        self.point_density = 10000
        self.point_size = 400  # meters radius
        
        # Isosurface settings
        self.isosurface_style = "Single Isosurface"
        self.isosurface_level = 50  # percentile
        self.multiple_levels = [20, 40, 60, 80]  # percentiles
        
        # Slice plane settings
        self.slice_style = "Single Slice"
        self.slice_axis = "Z-Axis (XY Plane)"
        self.slice_position = 50  # percentage
        self.three_plane_positions = (50, 50, 50)  # x, y, z percentages
        
        # Scalar bar
        self.scalar_bar = None
        self.current_lut = None
        
    def add_flux_field(self, file_path, vtk_data, color_lut=None, opacity=0.8, min_flux=None, max_flux=None, particle_type=None):
        """Add a flux field to the renderer
        
        Args:
            file_path: Path to the flux file (used as identifier)
            vtk_data: VTK dataset containing the flux field
            color_lut: Optional color lookup table
            opacity: Opacity value (0-1)
            min_flux: Minimum flux threshold (replaces zero values)
            max_flux: Maximum flux threshold (clips high values)
            particle_type: Type of particle (e.g., 'electron', 'proton') for scale bar title
        """
        from data_io import VTKDataLoader
        
        # Remove existing actor if present
        self.remove_flux_field(file_path)
        
        # Check if this is time-dependent data
        if VTKDataLoader.is_time_dependent(vtk_data):
            print(f"Detected time-dependent flux field: {file_path}")
            
            # Store the multiblock data and time information
            self.time_dependent_data[file_path] = vtk_data
            self.time_info[file_path] = VTKDataLoader.get_time_info(vtk_data)
            self.current_time_index[file_path] = 0  # Start with first time step
            
            print(f"Time steps: {self.time_info[file_path]['num_time_steps']}")
            print(f"Time range: {self.time_info[file_path]['time_range']} hours")
            
            # Extract data for the first time step
            current_data = VTKDataLoader.get_time_step_data(vtk_data, 0)
            if current_data is None:
                print(f"ERROR: Could not extract data for time step 0 from {file_path}")
                return
            processed_data = self._apply_flux_thresholds(current_data, min_flux, max_flux)
        else:
            # Static (non-time-dependent) data
            processed_data = self._apply_flux_thresholds(vtk_data, min_flux, max_flux)
        
        # Store the processed data for the current time step
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
            # Handle both single actors and lists of actors
            if isinstance(actor, list):
                # Store list of actors for three-plane mode
                self.field_actors[file_path] = actor
                for a in actor:
                    self.renderer.AddActor(a)
            else:
                # Single actor for single slice mode
                self.field_actors[file_path] = actor
                self.renderer.AddActor(actor)
            
            # Store particle type
            self.field_particle_types[file_path] = particle_type
            
            # Update scalar bar with the LUT from this field
            if color_lut:
                self._update_scalar_bar(color_lut, processed_data, particle_type)
    
    def remove_flux_field(self, file_path):
        """Remove a flux field from the renderer
        
        Args:
            file_path: Path to the flux file (identifier)
        """
        if file_path in self.field_actors:
            actor_or_list = self.field_actors[file_path]
            # Handle both single actors and lists of actors
            if isinstance(actor_or_list, list):
                for actor in actor_or_list:
                    self.renderer.RemoveActor(actor)
            else:
                self.renderer.RemoveActor(actor_or_list)
            del self.field_actors[file_path]
        
        if file_path in self.field_data:
            del self.field_data[file_path]
            
        if file_path in self.field_particle_types:
            del self.field_particle_types[file_path]
        
        # Clean up time-dependent data
        if file_path in self.time_dependent_data:
            del self.time_dependent_data[file_path]
        if file_path in self.time_info:
            del self.time_info[file_path]
        if file_path in self.current_time_index:
            del self.current_time_index[file_path]
    
    def is_time_dependent(self, file_path):
        """Check if a loaded flux field is time-dependent
        
        Args:
            file_path: Path to the flux file (identifier)
            
        Returns:
            True if the field has time-dependent data
        """
        return file_path in self.time_dependent_data
    
    def get_time_info(self, file_path):
        """Get time information for a flux field
        
        Args:
            file_path: Path to the flux file (identifier)
            
        Returns:
            Dictionary with time information or None if not time-dependent
        """
        return self.time_info.get(file_path)
    
    def set_time_step(self, file_path, time_index):
        """Set the current time step for a time-dependent flux field
        
        Args:
            file_path: Path to the flux file (identifier)
            time_index: Time step index (0-based)
        """
        if file_path not in self.time_dependent_data:
            print(f"Warning: {file_path} is not time-dependent")
            return
        
        from data_io import VTKDataLoader
        
        vtk_data = self.time_dependent_data[file_path]
        time_info = self.time_info[file_path]
        
        # Validate time index
        if time_index < 0 or time_index >= time_info['num_time_steps']:
            print(f"Warning: Time index {time_index} out of range (0-{time_info['num_time_steps']-1})")
            return
        
        # Update current time index
        self.current_time_index[file_path] = time_index
        
        # Get data for the specified time step (works for both multiblock and single datasets)
        time_step_data = VTKDataLoader.get_time_step_data(vtk_data, time_index)
        if time_step_data is None:
            print(f"Warning: Could not get data for time step {time_index}")
            return
        
        # Apply thresholds (reuse settings from original load)
        processed_data = self._apply_flux_thresholds(time_step_data)
        
        # Store the updated data
        self.field_data[file_path] = processed_data
        
        # Remove existing actors
        if file_path in self.field_actors:
            actor_or_list = self.field_actors[file_path]
            if isinstance(actor_or_list, list):
                for actor in actor_or_list:
                    self.renderer.RemoveActor(actor)
            else:
                self.renderer.RemoveActor(actor_or_list)
        
        # Create new visualization with updated data
        color_lut = self.current_lut  # Use current color lookup table
        opacity = 0.8  # Default opacity
        
        if self.visualization_mode == "Point Cloud":
            actor = self._create_point_cloud(processed_data, color_lut, opacity)
        elif self.visualization_mode == "Isosurfaces":
            actor = self._create_isosurfaces(processed_data, color_lut, opacity)
        elif self.visualization_mode == "Slice Planes":
            actor = self._create_slice_plane(processed_data, color_lut, opacity)
        else:
            actor = self._create_simple_visualization(processed_data, color_lut, opacity)
        
        if actor:
            # Add new actors to renderer
            if isinstance(actor, list):
                self.field_actors[file_path] = actor
                for a in actor:
                    self.renderer.AddActor(a)
            else:
                self.field_actors[file_path] = actor
                self.renderer.AddActor(actor)
            
            # Update scalar bar
            if color_lut:
                particle_type = self.field_particle_types.get(file_path)
                self._update_scalar_bar(color_lut, processed_data, particle_type)
            
            # Print time information
            current_time = time_info['time_values'][time_index]
            print(f"Updated to time step {time_index}: t = {current_time:.1f} hours")
    
    def get_current_time_step(self, file_path):
        """Get the current time step index for a flux field
        
        Args:
            file_path: Path to the flux file (identifier)
            
        Returns:
            Current time step index or None if not time-dependent
        """
        return self.current_time_index.get(file_path)
    
    def get_current_time_value(self, file_path):
        """Get the current time value for a flux field
        
        Args:
            file_path: Path to the flux file (identifier)
            
        Returns:
            Current time value in hours or None if not time-dependent
        """
        if file_path not in self.time_info:
            return None
        
        time_index = self.current_time_index.get(file_path, 0)
        time_values = self.time_info[file_path]['time_values']
        
        if time_index < len(time_values):
            return time_values[time_index]
        return None
    
    def update_time_dependent_field(self, file_path, target_time_hours):
        """Update time-dependent flux field to the closest time step
        
        Args:
            file_path: Path to the flux file (identifier)
            target_time_hours: Target time in hours
        """
        if file_path not in self.time_dependent_data:
            return  # Not a time-dependent field
        
        time_info = self.time_info[file_path]
        time_values = time_info['time_values']
        
        # Find the closest time step
        closest_index = 0
        min_diff = abs(time_values[0] - target_time_hours)
        
        for i, time_val in enumerate(time_values):
            diff = abs(time_val - target_time_hours)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        
        # Only update if we need to change time steps
        if self.current_time_index.get(file_path, 0) != closest_index:
            self.set_time_step(file_path, closest_index)
    
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
        for actor_or_list in self.field_actors.values():
            # Handle both single actors and lists of actors
            if isinstance(actor_or_list, list):
                for actor in actor_or_list:
                    actor.GetProperty().SetOpacity(opacity)
            else:
                actor_or_list.GetProperty().SetOpacity(opacity)
    
    def toggle_visibility(self, visible):
        """Toggle visibility of all flux fields
        
        Args:
            visible: True to show, False to hide
        """
        for actor_or_list in self.field_actors.values():
            # Handle both single actors and lists of actors
            if isinstance(actor_or_list, list):
                for actor in actor_or_list:
                    actor.SetVisibility(visible)
            else:
                actor_or_list.SetVisibility(visible)
    
    def clear_all(self):
        """Remove all flux field visualizations"""
        for file_path in list(self.field_actors.keys()):
            self.remove_flux_field(file_path)
        
        # Clear particle types tracking
        self.field_particle_types.clear()
        
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
    
    def _update_scalar_bar(self, lut, vtk_data, particle_type=None):
        """Update or create the scalar bar
        
        Args:
            lut: VTK lookup table
            vtk_data: VTK dataset for getting scalar range and name
            particle_type: Type of particle for title
        """
        # Don't always remove existing scalar bar - we might want to update it
        
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
        
        # If scalar bar doesn't exist, create it
        if not self.scalar_bar:
            # Create scalar bar
            self.scalar_bar = vtk.vtkScalarBarActor()
            self.scalar_bar.SetLookupTable(lut)
            # Add to renderer only once
            self.renderer.AddActor2D(self.scalar_bar)
        else:
            # Update the lookup table
            self.scalar_bar.SetLookupTable(lut)
        
        # Build title based on all active particle types
        active_types = []
        for file_path, p_type in self.field_particle_types.items():
            if file_path in self.field_actors and p_type:
                if p_type not in active_types:
                    active_types.append(p_type)
        
        # Set title based on active particle types
        if len(active_types) > 1:
            # Multiple types - show combined title
            title = f"Mixed flux ({', '.join(active_types)})"
        elif len(active_types) == 1:
            # Single type
            title = f"{active_types[0]} flux"
        elif particle_type:
            # Use provided particle type
            title = f"{particle_type} flux"
        else:
            title = scalar_name if scalar_name != "electron_flux" else "electron flux"
        self.scalar_bar.SetTitle(title)
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
    
    def update_scalar_bar_title(self):
        """Update the scalar bar title based on current active particle types"""
        if not self.scalar_bar:
            return
        
        # Build title based on all active particle types
        active_types = []
        for file_path, p_type in self.field_particle_types.items():
            if file_path in self.field_actors and p_type:
                if p_type not in active_types:
                    active_types.append(p_type)
        
        # Set title based on active particle types
        if len(active_types) > 1:
            # Multiple types - show combined title
            title = f"Mixed flux ({', '.join(active_types)})"
        elif len(active_types) == 1:
            # Single type
            title = f"{active_types[0]} flux"
        else:
            title = "Flux"
        
        self.scalar_bar.SetTitle(title)
    
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
            VTK actor or list of actors for the slice planes
        """
        if self.slice_style == "Single Slice":
            return self._create_single_slice(vtk_data, color_lut, opacity)
        else:  # Three Plane Slice
            return self._create_three_plane_slice(vtk_data, color_lut, opacity)
    
    def _create_single_slice(self, vtk_data, color_lut, opacity):
        """Create single slice plane visualization
        
        Args:
            vtk_data: VTK dataset containing the flux field
            color_lut: Color lookup table
            opacity: Opacity value
            
        Returns:
            VTK actor for the single slice plane
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
        
        # Calculate adaptive resolution based on data size and performance target
        resolution = self._calculate_adaptive_resolution(vtk_data, width, height)
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
    
    def _create_three_plane_slice(self, vtk_data, color_lut, opacity):
        """Create three orthogonal slice planes
        
        Args:
            vtk_data: VTK dataset containing the flux field
            color_lut: Color lookup table
            opacity: Opacity value
            
        Returns:
            List of VTK actors for the three slice planes
        """
        bounds = vtk_data.GetBounds()
        x_pos, y_pos, z_pos = self.three_plane_positions
        
        print(f"DEBUG: Creating three planes at positions: X={x_pos}%, Y={y_pos}%, Z={z_pos}%")
        
        actors = []
        
        # Create XY plane (normal along Z axis)
        xy_actor = self._create_plane_at_position(
            vtk_data, color_lut, opacity, bounds,
            normal=[0, 0, 1],
            position_percent=z_pos / 100.0,
            axis_name="Z"
        )
        if xy_actor:
            actors.append(xy_actor)
            print(f"DEBUG: Created XY plane actor")
        else:
            print(f"DEBUG: Failed to create XY plane actor")
        
        # Create XZ plane (normal along Y axis)  
        xz_actor = self._create_plane_at_position(
            vtk_data, color_lut, opacity, bounds,
            normal=[0, 1, 0],
            position_percent=y_pos / 100.0,
            axis_name="Y"
        )
        if xz_actor:
            actors.append(xz_actor)
            print(f"DEBUG: Created XZ plane actor")
        else:
            print(f"DEBUG: Failed to create XZ plane actor")
        
        # Create YZ plane (normal along X axis)
        yz_actor = self._create_plane_at_position(
            vtk_data, color_lut, opacity, bounds,
            normal=[1, 0, 0],
            position_percent=x_pos / 100.0,
            axis_name="X"
        )
        if yz_actor:
            actors.append(yz_actor)
            print(f"DEBUG: Created YZ plane actor")
        else:
            print(f"DEBUG: Failed to create YZ plane actor")
        
        print(f"DEBUG: Total actors created: {len(actors)}")
        return actors
    
    def _create_plane_at_position(self, vtk_data, color_lut, opacity, bounds, normal, position_percent, axis_name):
        """Helper method to create a single slice plane at specified position
        
        Args:
            vtk_data: VTK dataset
            color_lut: Color lookup table
            opacity: Opacity value
            bounds: Data bounds
            normal: Plane normal vector [x, y, z]
            position_percent: Position along axis (0.0-1.0)
            axis_name: Axis name for plane dimensions
            
        Returns:
            VTK actor for the slice plane
        """
        # Calculate plane origin based on normal direction
        if normal[0] == 1:  # YZ plane
            width = bounds[3] - bounds[2]   # Y extent
            height = bounds[5] - bounds[4]  # Z extent
            origin_coord = bounds[0] + position_percent * (bounds[1] - bounds[0])
            origin = [origin_coord, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
            
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(origin[0], bounds[2], bounds[4])
            plane_source.SetPoint1(origin[0], bounds[3], bounds[4])
            plane_source.SetPoint2(origin[0], bounds[2], bounds[5])
            
        elif normal[1] == 1:  # XZ plane
            width = bounds[1] - bounds[0]   # X extent
            height = bounds[5] - bounds[4]  # Z extent
            origin_coord = bounds[2] + position_percent * (bounds[3] - bounds[2])
            origin = [(bounds[0]+bounds[1])/2, origin_coord, (bounds[4]+bounds[5])/2]
            
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(bounds[0], origin[1], bounds[4])
            plane_source.SetPoint1(bounds[1], origin[1], bounds[4])
            plane_source.SetPoint2(bounds[0], origin[1], bounds[5])
            
        else:  # XY plane (normal[2] == 1)
            width = bounds[1] - bounds[0]   # X extent
            height = bounds[3] - bounds[2]  # Y extent
            origin_coord = bounds[4] + position_percent * (bounds[5] - bounds[4])
            origin = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, origin_coord]
            
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(bounds[0], bounds[2], origin[2])
            plane_source.SetPoint1(bounds[1], bounds[2], origin[2])
            plane_source.SetPoint2(bounds[0], bounds[3], origin[2])
        
        # Calculate adaptive resolution
        resolution = self._calculate_adaptive_resolution(vtk_data, width, height)
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
        
        # Create mapper
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
        
        # Set properties with slightly different opacity for multiple planes
        prop = actor.GetProperty()
        prop.SetOpacity(opacity * 0.7)  # Reduce opacity for multiple planes
        prop.SetRepresentationToSurface()
        
        if not color_lut:
            # Different default colors for different planes
            if axis_name == "X":
                prop.SetColor(0.8, 0.2, 0.2)  # Red for YZ plane
            elif axis_name == "Y":
                prop.SetColor(0.2, 0.8, 0.2)  # Green for XZ plane  
            else:  # Z
                prop.SetColor(0.2, 0.2, 0.8)  # Blue for XY plane
        
        # Add some lighting
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.8)
        
        return actor
    
    def _calculate_adaptive_resolution(self, vtk_data, plane_width, plane_height):
        """Calculate optimal slice resolution based on data size and performance target
        
        Args:
            vtk_data: The source VTK dataset
            plane_width: Physical width of the slice plane
            plane_height: Physical height of the slice plane
            
        Returns:
            int: Resolution (number of points per dimension)
        """
        # Get data characteristics
        num_points = vtk_data.GetNumberOfPoints()
        num_cells = vtk_data.GetNumberOfCells()
        
        # Performance targets (adjust these based on your hardware)
        target_slice_points = 50000  # Target ~50k points for good performance
        min_resolution = 50          # Minimum quality threshold
        max_resolution = 400         # Maximum for very fast systems
        
        # Calculate base resolution from data density
        # More points in source = can afford higher slice resolution
        if num_points < 50000:          # Small datasets
            base_resolution = 100
        elif num_points < 500000:       # Medium datasets  
            base_resolution = 150
        elif num_points < 2000000:      # Large datasets
            base_resolution = 200
        else:                           # Very large datasets
            base_resolution = 250
        
        # Adjust for slice plane size (larger planes need more points for same density)
        bounds = vtk_data.GetBounds()
        max_data_extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        plane_diagonal = (plane_width**2 + plane_height**2)**0.5
        size_factor = plane_diagonal / max_data_extent
        
        # Apply size adjustment
        adjusted_resolution = int(base_resolution * (0.7 + 0.6 * size_factor))
        
        # Ensure we hit our target point count
        current_points = adjusted_resolution * adjusted_resolution
        if current_points > target_slice_points:
            # Scale down to meet performance target
            scale_factor = (target_slice_points / current_points)**0.5
            adjusted_resolution = int(adjusted_resolution * scale_factor)
        
        # Apply bounds
        final_resolution = max(min_resolution, min(adjusted_resolution, max_resolution))
        
        # Log resolution decision for debugging
        total_slice_points = final_resolution * final_resolution
        print(f"Slice resolution: {final_resolution}x{final_resolution} = {total_slice_points:,} points "
              f"(data: {num_points:,} points)")
        
        return final_resolution
    
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
    
    def set_slice_style(self, style):
        """Set the slice plane style
        
        Args:
            style: "Single Slice" or "Three Plane Slice"
        """
        if style != self.slice_style:
            self.slice_style = style
            if self.visualization_mode == "Slice Planes":
                self.refresh_all()
    
    def set_three_plane_positions(self, positions):
        """Set the three-plane intersection positions
        
        Args:
            positions: Tuple of (x_percent, y_percent, z_percent)
        """
        if positions != self.three_plane_positions:
            self.three_plane_positions = positions
            if self.visualization_mode == "Slice Planes" and self.slice_style == "Three Plane Slice":
                self.refresh_all()