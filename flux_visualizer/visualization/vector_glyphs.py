"""
Vector glyph utilities for flux field visualization
Provides arrow and vector visualization components for different flux representation modes.
"""

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

class VectorGlyphRenderer:
    """Handles creation of vector glyphs (arrows) for flux field visualization"""
    
    def __init__(self):
        """Initialize vector glyph renderer"""
        pass
    
    def create_arrow_source(self, arrow_length=1000.0, arrow_radius=200.0, 
                          tip_length=0.3, tip_radius=0.1):
        """
        Create an arrow source for vector glyphs.
        
        Args:
            arrow_length: Length of arrow in km
            arrow_radius: Radius of arrow shaft in km
            tip_length: Tip length as fraction of total length
            tip_radius: Tip radius as fraction of shaft radius
            
        Returns:
            VTK arrow source
        """
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(tip_length)
        arrow_source.SetTipRadius(tip_radius)
        arrow_source.SetShaftRadius(0.03)  # Thin shaft for visibility
        arrow_source.SetTipResolution(8)
        arrow_source.SetShaftResolution(8)
        return arrow_source
    
    def create_vector_point_cloud(self, vtk_data, target_points=5000, 
                                min_magnitude=None, arrow_scale=1.0):
        """
        Create vector arrow point cloud from VTK data with vector field.
        
        Args:
            vtk_data: VTK dataset with vector data
            target_points: Target number of arrows to display
            min_magnitude: Minimum vector magnitude to display
            arrow_scale: Scaling factor for arrow size
            
        Returns:
            Tuple of (polydata_with_vectors, has_vectors)
        """
        print("=== CREATE VECTOR POINT CLOUD DEBUG ===")
        print(f"Target points: {target_points}, arrow_scale: {arrow_scale}")
        
        # Check for vector data
        vector_array = vtk_data.GetPointData().GetVectors()
        if not vector_array:
            print("No active vector array, searching for vector arrays...")
            # Look for vector arrays in the point data
            point_data = vtk_data.GetPointData()
            for i in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(i)
                array_name = array.GetName() if array.GetName() else f"Array_{i}"
                print(f"  Checking array '{array_name}': {array.GetNumberOfComponents()} components")
                if (array.GetNumberOfComponents() == 3 and 
                    'vector' in array_name.lower()):
                    vector_array = array
                    print(f"  Using vector array: '{array_name}'")
                    break
        else:
            print(f"Using active vector array: '{vector_array.GetName()}'")
        
        if not vector_array:
            print("No suitable vector array found")
            return None, False
        
        # Get scalar data for magnitude-based sampling
        scalar_array = vtk_data.GetPointData().GetScalars()
        num_points = vtk_data.GetNumberOfPoints()
        
        # Convert vector data to numpy for analysis
        vector_data = vtk_to_numpy(vector_array).reshape(-1, 3)
        magnitudes = np.linalg.norm(vector_data, axis=1)
        
        print(f"Vector data shape: {vector_data.shape}")
        print(f"Magnitude range: [{magnitudes.min():.2e}, {magnitudes.max():.2e}]")
        print(f"Non-zero vectors: {np.sum(magnitudes > 1e-10)}/{len(magnitudes)}")
        
        # Filter by minimum magnitude if specified
        if min_magnitude is not None:
            valid_indices = magnitudes >= min_magnitude
            print(f"Applied min magnitude filter ({min_magnitude}): {np.sum(valid_indices)} valid vectors")
        else:
            valid_indices = magnitudes > 1e-10  # Avoid zero vectors
            print(f"Applied default magnitude filter (> 1e-10): {np.sum(valid_indices)} valid vectors")
        
        valid_count = np.sum(valid_indices)
        if valid_count == 0:
            print("No valid vectors found after filtering")
            return None, False
        
        # Smart sampling based on vector magnitude
        target_points = min(target_points, valid_count)
        
        if target_points < valid_count:
            # Probability sampling based on vector magnitude
            valid_magnitudes = magnitudes[valid_indices]
            probabilities = valid_magnitudes / valid_magnitudes.sum()
            
            valid_idx_array = np.where(valid_indices)[0]
            sample_indices = np.random.choice(
                valid_idx_array,
                size=target_points,
                replace=False,
                p=probabilities
            )
        else:
            sample_indices = np.where(valid_indices)[0]
        
        # Create new polydata with sampled points and vectors
        sampled_points = vtk.vtkPoints()
        sampled_vectors = vtk.vtkFloatArray()
        sampled_vectors.SetNumberOfComponents(3)
        sampled_vectors.SetName("vectors")
        
        sampled_scalars = None
        if scalar_array:
            sampled_scalars = vtk.vtkFloatArray()
            sampled_scalars.SetNumberOfComponents(1)
            sampled_scalars.SetName("flux_magnitude")
        
        for idx in sample_indices:
            point = vtk_data.GetPoint(idx)
            vector = vector_data[idx]
            
            # Add small jitter to avoid overlap
            jitter = np.random.normal(0, 100, 3)  # 100km jitter
            jittered_point = [point[i] + jitter[i] for i in range(3)]
            
            sampled_points.InsertNextPoint(jittered_point)
            sampled_vectors.InsertNextTuple(vector)
            
            if sampled_scalars and scalar_array:
                scalar_val = scalar_array.GetValue(idx)
                sampled_scalars.InsertNextValue(scalar_val)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(sampled_points)
        polydata.GetPointData().SetVectors(sampled_vectors)
        
        if sampled_scalars:
            polydata.GetPointData().SetScalars(sampled_scalars)
        
        final_point_count = sampled_points.GetNumberOfPoints()
        print(f"Created vector point cloud with {final_point_count} arrows")
        print("=== END CREATE VECTOR POINT CLOUD DEBUG ===")
        
        return polydata, True
    
    def create_vector_glyphs(self, polydata, color_lut=None, arrow_scale=1.0, 
                           opacity=0.8, scale_by_magnitude=True):
        """
        Create vector arrow glyphs from polydata with vectors.
        
        Args:
            polydata: VTK polydata with vector field
            color_lut: Color lookup table
            arrow_scale: Global scaling for arrows
            opacity: Arrow opacity
            scale_by_magnitude: Scale arrow size by vector magnitude
            
        Returns:
            VTK actor with arrow glyphs
        """
        print("=== CREATE VECTOR GLYPHS DEBUG ===")
        print(f"Input polydata points: {polydata.GetNumberOfPoints() if polydata else 'None'}")
        print(f"Arrow scale: {arrow_scale}, Opacity: {opacity}, Scale by magnitude: {scale_by_magnitude}")
        
        if not polydata:
            print("No polydata provided")
            return None
        
        # Create arrow source
        arrow_source = self.create_arrow_source()
        
        # Create glyph filter
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetInputData(polydata)
        glyph3D.SetSourceConnection(arrow_source.GetOutputPort())
        glyph3D.OrientOn()
        glyph3D.SetVectorModeToUseVector()
        
        # Always use fixed-size arrows, let color indicate magnitude
        glyph3D.SetScaleModeToDataScalingOff()
        glyph3D.SetScaleFactor(arrow_scale)
        
        glyph3D.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph3D.GetOutputPort())
        
        # Set color mapping - color by scalar flux magnitude, not vector components
        scalar_array = polydata.GetPointData().GetScalars()
        if scalar_array and color_lut:
            # Use scalar flux values for coloring (not vector components)
            mapper.SetScalarModeToUsePointData()
            mapper.SetLookupTable(color_lut)
            scalar_range = scalar_array.GetRange()
            mapper.SetScalarRange(scalar_range)
            print(f"Using scalar flux range for arrow coloring: [{scalar_range[0]:.2e}, {scalar_range[1]:.2e}]")
        elif color_lut:
            # Fallback: color by vector magnitude
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("vectors")
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(color_lut)
            # Calculate magnitude range for proper color mapping
            vector_array = polydata.GetPointData().GetVectors()
            if vector_array:
                vector_data = vtk_to_numpy(vector_array)
                magnitudes = np.linalg.norm(vector_data, axis=1)
                mag_range = (magnitudes.min(), magnitudes.max())
                mapper.SetScalarRange(mag_range)
                print(f"Using vector magnitude range for coloring: [{mag_range[0]:.2e}, {mag_range[1]:.2e}]")
        else:
            # No color mapping - arrows will be white
            mapper.ScalarVisibilityOff()
            print("No color lookup table provided - arrows will be white")
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        
        print(f"Created vector glyph actor with {polydata.GetNumberOfPoints()} arrows")
        print("=== END CREATE VECTOR GLYPHS DEBUG ===")
        
        return actor
    
    def add_vectors_to_slice_plane(self, slice_polydata, original_vtk_data, 
                                 vector_density=50, arrow_scale=0.5):
        """
        Add vector arrows to a slice plane by sampling from actual data points near the plane.
        
        Args:
            slice_polydata: The slice plane polydata (used to get plane equation)
            original_vtk_data: Original VTK data with vector field
            vector_density: Target number of vectors on the slice
            arrow_scale: Scaling for arrows on slice
            
        Returns:
            List of actors [vector_actors]
        """
        print("=== ADD VECTORS TO SLICE PLANE DEBUG ===")
        print(f"Target vector density: {vector_density}")
        
        # Check if original data has vectors
        vector_array = original_vtk_data.GetPointData().GetVectors()
        if not vector_array:
            # Look for vector arrays
            point_data = original_vtk_data.GetPointData()
            for i in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(i)
                if (array.GetNumberOfComponents() == 3 and 
                    'vector' in array.GetName().lower()):
                    vector_array = array
                    break
        
        if not vector_array:
            print("No vector data found")
            return []
        
        # Get scalar data for coloring
        scalar_array = original_vtk_data.GetPointData().GetScalars()
        
        # Instead of using sparse slice grid, sample directly from data points near the plane
        # Get the plane's center and bounds to define "near"
        slice_bounds = slice_polydata.GetBounds()
        slice_center = [
            (slice_bounds[0] + slice_bounds[1]) / 2,
            (slice_bounds[2] + slice_bounds[3]) / 2, 
            (slice_bounds[4] + slice_bounds[5]) / 2
        ]
        
        # Calculate slice plane normal from first 3 points
        if slice_polydata.GetNumberOfPoints() >= 3:
            p0 = slice_polydata.GetPoint(0)
            p1 = slice_polydata.GetPoint(1) 
            p2 = slice_polydata.GetPoint(2)
            
            # Two vectors in the plane
            v1 = [p1[i] - p0[i] for i in range(3)]
            v2 = [p2[i] - p0[i] for i in range(3)]
            
            # Cross product gives normal
            normal = [
                v1[1]*v2[2] - v1[2]*v2[1],
                v1[2]*v2[0] - v1[0]*v2[2], 
                v1[0]*v2[1] - v1[1]*v2[0]
            ]
            
            # Normalize
            normal_mag = np.sqrt(sum(n*n for n in normal))
            if normal_mag > 0:
                normal = [n/normal_mag for n in normal]
            else:
                normal = [0, 0, 1]  # Default to Z-axis
        else:
            normal = [0, 0, 1]
            
        print(f"Slice plane normal: {normal}")
        
        # Find data points near the slice plane
        num_data_points = original_vtk_data.GetNumberOfPoints()
        thickness = 2000.0  # km - thickness of slice for sampling
        
        near_plane_points = []
        near_plane_vectors = []
        near_plane_scalars = []
        
        for i in range(num_data_points):
            point = original_vtk_data.GetPoint(i)
            vector = vector_array.GetTuple(i)
            
            # Calculate distance from point to plane
            to_point = [point[j] - slice_center[j] for j in range(3)]
            distance = abs(sum(to_point[j] * normal[j] for j in range(3)))
            
            # Check if point is near the slice plane and has significant vector
            vector_mag = np.linalg.norm(vector)
            if distance <= thickness and vector_mag > 1e-10:
                near_plane_points.append(point)
                near_plane_vectors.append(vector)
                if scalar_array:
                    near_plane_scalars.append(scalar_array.GetValue(i))
        
        print(f"Found {len(near_plane_points)} data points near slice plane")
        
        if len(near_plane_points) == 0:
            return []
        
        # Sample from the near-plane points 
        target_points = min(vector_density, len(near_plane_points))
        
        if target_points < len(near_plane_points):
            # Smart sampling based on vector magnitude
            magnitudes = [np.linalg.norm(v) for v in near_plane_vectors]
            total_mag = sum(magnitudes)
            
            if total_mag > 0:
                # Probability sampling based on vector magnitude
                probabilities = [m/total_mag for m in magnitudes]
                sample_indices = np.random.choice(
                    len(near_plane_points), 
                    size=target_points,
                    replace=False,
                    p=probabilities
                )
            else:
                # Uniform sampling fallback
                sample_indices = np.random.choice(
                    len(near_plane_points),
                    size=target_points,
                    replace=False
                )
        else:
            sample_indices = range(len(near_plane_points))
        
        # Create vector polydata
        vector_points = vtk.vtkPoints()
        vector_vectors = vtk.vtkFloatArray()
        vector_vectors.SetNumberOfComponents(3)
        vector_vectors.SetName("vectors")
        
        sampled_scalars = None
        if scalar_array:
            sampled_scalars = vtk.vtkFloatArray()
            sampled_scalars.SetNumberOfComponents(1)
            sampled_scalars.SetName("flux_magnitude")
        
        for idx in sample_indices:
            vector_points.InsertNextPoint(near_plane_points[idx])
            vector_vectors.InsertNextTuple(near_plane_vectors[idx])
            
            if sampled_scalars and idx < len(near_plane_scalars):
                sampled_scalars.InsertNextValue(near_plane_scalars[idx])
        
        # Create polydata for vectors
        vector_polydata = vtk.vtkPolyData()
        vector_polydata.SetPoints(vector_points)
        vector_polydata.GetPointData().SetVectors(vector_vectors)
        
        if sampled_scalars:
            vector_polydata.GetPointData().SetScalars(sampled_scalars)
        
        print(f"Created {vector_points.GetNumberOfPoints()} arrows for slice")
        print("=== END ADD VECTORS TO SLICE PLANE DEBUG ===")
        
        # Create vector glyphs
        vector_actor = self.create_vector_glyphs(
            vector_polydata, 
            arrow_scale=arrow_scale,
            scale_by_magnitude=False  # Fixed size arrows
        )
        
        return [vector_actor] if vector_actor else []
    
    def has_vector_data(self, vtk_data):
        """
        Check if VTK data contains vector arrays.
        
        Args:
            vtk_data: VTK dataset to check
            
        Returns:
            True if vector data is available
        """
        print("=== VECTOR DATA DEBUG ===")
        print(f"VTK data type: {type(vtk_data)}")
        print(f"Number of points: {vtk_data.GetNumberOfPoints()}")
        
        point_data = vtk_data.GetPointData()
        print(f"Number of point data arrays: {point_data.GetNumberOfArrays()}")
        
        # List all point data arrays
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName() if array.GetName() else f"Array_{i}"
            num_components = array.GetNumberOfComponents()
            print(f"  Array {i}: '{array_name}' - {num_components} components")
        
        # Check for vector data
        vector_array = vtk_data.GetPointData().GetVectors()
        if vector_array:
            print(f"Found active vector array: '{vector_array.GetName()}' with {vector_array.GetNumberOfComponents()} components")
            return True
        else:
            print("No active vector array found")
        
        # Look for arrays with 3 components containing "vector" in name
        found_vector_arrays = []
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName() if array.GetName() else f"Array_{i}"
            if (array.GetNumberOfComponents() == 3 and 
                'vector' in array_name.lower()):
                found_vector_arrays.append(array_name)
        
        if found_vector_arrays:
            print(f"Found potential vector arrays: {found_vector_arrays}")
            return True
        else:
            print("No 3-component arrays with 'vector' in name found")
        
        print("=== END VECTOR DATA DEBUG ===")
        return False