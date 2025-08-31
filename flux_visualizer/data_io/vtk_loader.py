"""
VTK Data Loading Module for STRATOS Flux Visualizer
Handles loading and validation of various VTK file formats
"""

import vtk
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class VTKDataLoader:
    """Handles loading of VTK data files with comprehensive format support"""
    
    # Supported VTK file extensions and their corresponding readers
    SUPPORTED_EXTENSIONS = {
        '.vts': vtk.vtkXMLStructuredGridReader,     # XML Structured Grid
        '.vtu': vtk.vtkXMLUnstructuredGridReader,   # XML Unstructured Grid
        '.vtp': vtk.vtkXMLPolyDataReader,          # XML PolyData
        '.vti': vtk.vtkXMLImageDataReader,         # XML Image Data
        '.vtm': vtk.vtkXMLMultiBlockDataReader,    # XML MultiBlock Data (time series)
        '.vtk': None  # Legacy format - requires detection
    }
    
    # File type descriptions for UI
    FILE_TYPE_DESCRIPTIONS = {
        '.vts': 'XML Structured Grid',
        '.vtu': 'XML Unstructured Grid',
        '.vtp': 'XML PolyData',
        '.vti': 'XML Image Data',
        '.vtm': 'XML MultiBlock Data (Time Series)',
        '.vtk': 'Legacy VTK'
    }
    
    @classmethod
    def load(cls, file_path: str) -> vtk.vtkDataObject:
        """
        Load VTK data from file with automatic format detection.
        
        Args:
            file_path: Path to the VTK file
            
        Returns:
            VTK data object
            
        Raises:
            ValueError: If file format is unsupported or file is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        file_ext = file_path.suffix.lower()
        
        print(f"Loading VTK file: {file_path}")
        print(f"File extension: {file_ext}")
        
        # Check if format is supported
        if file_ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {file_ext}\n"
                f"Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Create appropriate reader
        if file_ext == '.vtk':
            # Legacy format requires content inspection
            reader = cls._create_legacy_vtk_reader(str(file_path))
        else:
            # XML format with known reader
            reader_class = cls.SUPPORTED_EXTENSIONS[file_ext]
            reader = reader_class()
            print(f"Using {reader_class.__name__}")
        
        # Set filename and read
        reader.SetFileName(str(file_path))
        print("Reading VTK file...")
        reader.Update()
        
        # Get the output
        output = reader.GetOutput()
        
        # Handle multiblock data differently
        if isinstance(output, vtk.vtkMultiBlockDataSet):
            print(f"Multiblock dataset loaded with {output.GetNumberOfBlocks()} blocks")
            # Don't validate individual blocks here - they will be extracted later
        else:
            # Validate the data for single datasets
            cls._validate_data(output)
            
            # Setup scalar data if needed
            cls._setup_scalar_data(output)
        
        print(f"Successfully loaded VTK data:")
        print(f"  Type: {type(output).__name__}")
        
        if isinstance(output, vtk.vtkMultiBlockDataSet):
            print(f"  Blocks: {output.GetNumberOfBlocks()}")
            # Show info for first block if available
            if output.GetNumberOfBlocks() > 0:
                first_block = output.GetBlock(0)
                if first_block:
                    print(f"  First block type: {type(first_block).__name__}")
                    print(f"  First block points: {first_block.GetNumberOfPoints():,}")
                    print(f"  First block cells: {first_block.GetNumberOfCells():,}")
        else:
            print(f"  Points: {output.GetNumberOfPoints():,}")
            print(f"  Cells: {output.GetNumberOfCells():,}")
        
        return output
    
    @classmethod
    def _create_legacy_vtk_reader(cls, file_path: str) -> vtk.vtkDataReader:
        """
        Create appropriate reader for legacy VTK files by inspecting content.
        
        Args:
            file_path: Path to the legacy VTK file
            
        Returns:
            Appropriate VTK reader for the detected format
        """
        try:
            # Read header to determine type
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i in range(15):  # Read first 15 lines for detection
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.strip().upper())
            
            content = ' '.join(lines)
            print(f"VTK file header content: {content[:200]}...")
            
            # Detect specific dataset type
            if 'STRUCTURED_GRID' in content:
                print("Detected: Structured Grid")
                return vtk.vtkStructuredGridReader()
            elif 'UNSTRUCTURED_GRID' in content:
                print("Detected: Unstructured Grid")
                return vtk.vtkUnstructuredGridReader()
            elif 'POLYDATA' in content:
                print("Detected: PolyData")
                return vtk.vtkPolyDataReader()
            elif 'STRUCTURED_POINTS' in content or 'DATASET STRUCTURED_POINTS' in content:
                print("Detected: Structured Points")
                return vtk.vtkStructuredPointsReader()
            elif 'RECTILINEAR_GRID' in content:
                print("Detected: Rectilinear Grid")
                return vtk.vtkRectilinearGridReader()
            else:
                print("Unknown format, trying generic data reader")
                return vtk.vtkDataSetReader()
                
        except Exception as e:
            print(f"Error detecting VTK format: {e}")
            print("Defaulting to UnstructuredGridReader")
            return vtk.vtkUnstructuredGridReader()
    
    @classmethod
    def _validate_data(cls, vtk_data: vtk.vtkDataObject) -> None:
        """
        Validate the loaded VTK data.
        
        Args:
            vtk_data: VTK data object to validate
            
        Raises:
            ValueError: If data is invalid or empty
        """
        if vtk_data is None:
            raise ValueError("Failed to load VTK data - reader returned None")
        
        if vtk_data.GetNumberOfPoints() == 0:
            raise ValueError("VTK file contains no data points")
        
        # Check for scalar data
        point_data = vtk_data.GetPointData()
        if point_data.GetNumberOfArrays() == 0:
            print("Warning: No point data arrays found")
    
    @classmethod
    def _setup_scalar_data(cls, vtk_data: vtk.vtkDataObject) -> None:
        """
        Setup and verify scalar data for visualization.
        
        Args:
            vtk_data: VTK data object
        """
        point_data = vtk_data.GetPointData()
        
        # Check if we already have scalar data
        scalar_array = point_data.GetScalars()
        
        if scalar_array:
            print(f"Primary scalar data: {scalar_array.GetName()}")
            print(f"Scalar range: {scalar_array.GetRange()}")
        else:
            print("No primary scalars found, checking available arrays...")
            
            # List all available arrays
            for i in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(i)
                print(f"  Array {i}: {array.GetName()} "
                      f"({array.GetNumberOfTuples()} tuples, "
                      f"{array.GetNumberOfComponents()} components)")
            
            # Set first array as scalars if available
            if point_data.GetNumberOfArrays() > 0:
                first_array = point_data.GetArray(0)
                point_data.SetScalars(first_array)
                print(f"Set '{first_array.GetName()}' as primary scalars")
                print(f"Scalar range: {first_array.GetRange()}")
    
    @classmethod
    def get_file_filter(cls) -> str:
        """
        Get file filter string for file dialog.
        
        Returns:
            File filter string for QFileDialog
        """
        filters = []
        
        # Add individual format filters
        for ext, desc in cls.FILE_TYPE_DESCRIPTIONS.items():
            filters.append(f"{desc} (*{ext})")
        
        # Add combined VTK filter
        all_extensions = ' '.join(f'*{ext}' for ext in cls.SUPPORTED_EXTENSIONS.keys())
        filters.insert(0, f"VTK Files ({all_extensions})")
        
        # Add all files filter
        filters.append("All Files (*)")
        
        return ";;".join(filters)
    
    @classmethod
    def get_data_info(cls, vtk_data: vtk.vtkDataObject) -> Dict[str, Any]:
        """
        Get detailed information about VTK data.
        
        Args:
            vtk_data: VTK data object
            
        Returns:
            Dictionary containing data information
        """
        info = {
            'type': type(vtk_data).__name__,
            'scalar_range': None,
            'scalar_name': None,
            'arrays': []
        }
        
        if isinstance(vtk_data, vtk.vtkMultiBlockDataSet):
            # Handle multiblock data
            info['num_blocks'] = vtk_data.GetNumberOfBlocks()
            info['num_points'] = 0
            info['num_cells'] = 0
            info['bounds'] = None
            
            # Get info from first block
            if vtk_data.GetNumberOfBlocks() > 0:
                first_block = vtk_data.GetBlock(0)
                if first_block:
                    info['num_points'] = first_block.GetNumberOfPoints()
                    info['num_cells'] = first_block.GetNumberOfCells()
                    info['bounds'] = first_block.GetBounds()
                    
                    # Get scalar info from first block
                    point_data = first_block.GetPointData()
                    scalar_array = point_data.GetScalars()
                    
                    if scalar_array:
                        info['scalar_name'] = scalar_array.GetName()
                        info['scalar_range'] = scalar_array.GetRange()
                    
                    # Get all arrays from first block
                    for i in range(point_data.GetNumberOfArrays()):
                        array = point_data.GetArray(i)
                        info['arrays'].append({
                            'name': array.GetName(),
                            'num_tuples': array.GetNumberOfTuples(),
                            'num_components': array.GetNumberOfComponents(),
                            'range': array.GetRange() if array.GetNumberOfComponents() == 1 else None
                        })
        else:
            # Handle single dataset
            info['num_points'] = vtk_data.GetNumberOfPoints()
            info['num_cells'] = vtk_data.GetNumberOfCells()
            info['bounds'] = vtk_data.GetBounds()
            
            # Get scalar information for single dataset
            point_data = vtk_data.GetPointData()
            scalar_array = point_data.GetScalars()
            
            if scalar_array:
                info['scalar_name'] = scalar_array.GetName()
                info['scalar_range'] = scalar_array.GetRange()
            
            # Get all arrays for single dataset
            for i in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(i)
                info['arrays'].append({
                    'name': array.GetName(),
                    'num_tuples': array.GetNumberOfTuples(),
                    'num_components': array.GetNumberOfComponents(),
                    'range': array.GetRange() if array.GetNumberOfComponents() == 1 else None
                })
        
        return info
    
    @classmethod
    def create_default_scalar_field(cls, vtk_data: vtk.vtkDataObject) -> None:
        """
        Create a default scalar field when none exists.
        Based on Van Allen belt-like field distribution.
        
        Args:
            vtk_data: VTK data object to add scalar field to
        """
        from vtk.util.numpy_support import numpy_to_vtk
        
        n_points = vtk_data.GetNumberOfPoints()
        scalar_values = []
        
        earth_radius = 6371.0  # km
        
        print("Creating default scalar field based on distance from Earth center...")
        
        for i in range(n_points):
            point = vtk_data.GetPoint(i)
            x, y, z = point
            
            # Distance from Earth center
            r = np.sqrt(x*x + y*y + z*z)
            
            # Create a simple Van Allen belt-like field
            flux = 0.0
            if r > earth_radius:
                # Peak flux in Van Allen belts (1.2-6 Earth radii)
                if 1.2 * earth_radius <= r <= 6 * earth_radius:
                    # Gaussian-like distribution centered at 3 Earth radii
                    flux = 1e6 * np.exp(-((r - 3*earth_radius)**2) / (earth_radius)**2)
            
            scalar_values.append(flux)
        
        # Add to VTK data
        scalar_array = numpy_to_vtk(np.array(scalar_values), deep=True)
        scalar_array.SetName("generated_flux")
        vtk_data.GetPointData().SetScalars(scalar_array)
        
        print(f"Created default scalar field with {len(scalar_values)} values")
        print(f"Scalar range: {min(scalar_values):.2e} to {max(scalar_values):.2e}")
    
    @classmethod
    def convert_to_unstructured_grid(cls, data: vtk.vtkDataObject) -> vtk.vtkUnstructuredGrid:
        """
        Convert any VTK data type to UnstructuredGrid if needed.
        
        Args:
            data: VTK data object
            
        Returns:
            UnstructuredGrid representation of the data
        """
        # If already unstructured grid, return as-is
        if isinstance(data, vtk.vtkUnstructuredGrid):
            return data
        
        print(f"Converting {type(data).__name__} to UnstructuredGrid...")
        
        if isinstance(data, vtk.vtkImageData):
            # Convert ImageData to UnstructuredGrid
            converter = vtk.vtkImageDataGeometryFilter()
            converter.SetInputData(data)
            converter.Update()
            return cls._polydata_to_unstructured_grid(converter.GetOutput())
            
        elif isinstance(data, vtk.vtkStructuredGrid):
            # Convert StructuredGrid to UnstructuredGrid
            converter = vtk.vtkStructuredGridGeometryFilter()
            converter.SetInputData(data)
            converter.Update()
            return cls._polydata_to_unstructured_grid(converter.GetOutput())
            
        elif isinstance(data, vtk.vtkRectilinearGrid):
            # Convert RectilinearGrid to UnstructuredGrid
            converter = vtk.vtkRectilinearGridGeometryFilter()
            converter.SetInputData(data)
            converter.Update()
            return cls._polydata_to_unstructured_grid(converter.GetOutput())
            
        elif isinstance(data, vtk.vtkPolyData):
            # Convert PolyData to UnstructuredGrid
            return cls._polydata_to_unstructured_grid(data)
        
        else:
            # Unknown type, try to return as-is
            print(f"Warning: Unknown data type {type(data).__name__}, returning as-is")
            return data
    
    @classmethod
    def _polydata_to_unstructured_grid(cls, polydata: vtk.vtkPolyData) -> vtk.vtkUnstructuredGrid:
        """
        Convert PolyData to UnstructuredGrid.
        
        Args:
            polydata: PolyData object
            
        Returns:
            UnstructuredGrid representation
        """
        ugrid = vtk.vtkUnstructuredGrid()
        
        # Copy points
        ugrid.SetPoints(polydata.GetPoints())
        
        # Copy cells
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            ugrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        
        # Copy point data
        ugrid.GetPointData().ShallowCopy(polydata.GetPointData())
        
        # Copy cell data
        ugrid.GetCellData().ShallowCopy(polydata.GetCellData())
        
        print(f"Converted PolyData to UnstructuredGrid: {ugrid.GetNumberOfPoints()} points, "
              f"{ugrid.GetNumberOfCells()} cells")
        
        return ugrid
    
    @classmethod
    def estimate_memory_usage(cls, vtk_data: vtk.vtkDataObject) -> float:
        """
        Estimate memory usage of VTK data in MB.
        
        Args:
            vtk_data: VTK data object
            
        Returns:
            Estimated memory usage in megabytes
        """
        # Basic estimation: 
        # Each point: 3 floats (position) + 1 float (scalar) = 16 bytes
        # Each cell: varies, estimate 4 integers = 16 bytes
        
        num_points = vtk_data.GetNumberOfPoints()
        num_cells = vtk_data.GetNumberOfCells()
        
        point_memory = num_points * 16  # bytes
        cell_memory = num_cells * 16    # bytes
        
        # Add overhead for data structures (rough estimate)
        overhead = (point_memory + cell_memory) * 0.2
        
        total_bytes = point_memory + cell_memory + overhead
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb
    
    @classmethod
    def is_time_dependent(cls, vtk_data: vtk.vtkDataObject) -> bool:
        """
        Check if VTK data contains time-dependent information.
        
        Args:
            vtk_data: VTK data object
            
        Returns:
            True if data is time-dependent (multiblock with time info or single dataset with multiple time arrays)
        """
        if isinstance(vtk_data, vtk.vtkMultiBlockDataSet):
            # Check if field data contains time information
            field_data = vtk_data.GetFieldData()
            if field_data and field_data.GetArray("TimeValue"):
                return True
        else:
            # Check single dataset for time-dependent arrays
            # Look for field data with TimeValues array (our new format)
            field_data = vtk_data.GetFieldData()
            if field_data and (field_data.GetArray("TimeValues") or field_data.GetArray("time_values")):
                return True
            
            # Check for multiple scalar arrays with time patterns
            point_data = vtk_data.GetPointData()
            time_arrays = []
            for i in range(point_data.GetNumberOfArrays()):
                array_name = point_data.GetArrayName(i)
                if array_name and (
                    "_t+" in array_name or 
                    "_t-" in array_name or 
                    "_t00" in array_name  # Our new format: electron_flux_t001, etc.
                ):
                    time_arrays.append(array_name)
            
            if len(time_arrays) >= 2:  # Need at least 2 time steps for animation
                return True
                
        return False
    
    @classmethod
    def _get_single_dataset_time_values(cls, vtk_data: vtk.vtkDataObject) -> Optional[np.ndarray]:
        """
        Extract time values from single dataset with multiple time arrays.
        
        Args:
            vtk_data: VTK single dataset
            
        Returns:
            Array of time values or None if not time-dependent
        """
        import re
        
        # First check if there's a TimeValues or time_values array in field data
        field_data = vtk_data.GetFieldData()
        if field_data:
            # Try our new format first
            time_array = field_data.GetArray("TimeValues")
            if not time_array:
                # Fall back to old format
                time_array = field_data.GetArray("time_values")
                
            if time_array:
                # Extract time values from the array
                time_values = []
                for i in range(time_array.GetNumberOfTuples()):
                    time_values.append(time_array.GetValue(i))
                
                # Return as sorted array
                return np.array(sorted(time_values))
        
        # Fall back to extracting time values from array names
        point_data = vtk_data.GetPointData()
        time_values = []
        
        # Pattern to match our new format: electron_flux_t000, electron_flux_t001, etc.
        time_pattern_new = r"_t(\d{3})$"  # Matches _t000, _t001, etc.
        
        # Pattern to match old format: t+6h, t-6h, t+0h
        time_pattern_old = r"_t([+-]?)(\d+)h"
        
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            if array_name:
                # Try new format first (electron_flux_t000, electron_flux_t001, etc.)
                match_new = re.search(time_pattern_new, array_name)
                if match_new:
                    # For new format, we need to get the actual time values from TimeValues field data
                    # For now, just store the index - will be resolved above by TimeValues array
                    continue
                else:
                    # Fall back to old format (t+6h, t-6h, t+0h)
                    match_old = re.search(time_pattern_old, array_name)
                    if match_old:
                        sign = match_old.group(1) or "+"
                        hours = int(match_old.group(2))
                        time_value = hours if sign == "+" else -hours
                        time_values.append(time_value)
        
        if len(time_values) >= 2:
            unique_times = sorted(list(set(time_values)))
            return np.array(unique_times, dtype=float)
        
        return None
    
    @classmethod
    def get_time_values(cls, multiblock_data: vtk.vtkMultiBlockDataSet) -> Optional[np.ndarray]:
        """
        Extract time values from multiblock dataset.
        
        Args:
            multiblock_data: VTK multiblock dataset
            
        Returns:
            Array of time values or None if not time-dependent
        """
        if not isinstance(multiblock_data, vtk.vtkMultiBlockDataSet):
            return None
        
        field_data = multiblock_data.GetFieldData()
        if not field_data:
            return None
        
        time_array = field_data.GetArray("TimeValue")
        if not time_array:
            return None
        
        # Convert VTK array to numpy
        n_times = time_array.GetNumberOfTuples()
        time_values = np.zeros(n_times)
        for i in range(n_times):
            time_values[i] = time_array.GetValue(i)
        
        return time_values
    
    @classmethod
    def get_time_step_data(cls, multiblock_data: vtk.vtkMultiBlockDataSet, time_index: int) -> Optional[vtk.vtkDataObject]:
        """
        Extract data for a specific time step from multiblock dataset.
        
        Args:
            multiblock_data: VTK multiblock dataset
            time_index: Time step index (0-based)
            
        Returns:
            VTK data object for the specified time step or None
        """
        if not isinstance(multiblock_data, vtk.vtkMultiBlockDataSet):
            return None
        
        if time_index < 0 or time_index >= multiblock_data.GetNumberOfBlocks():
            return None
        
        return multiblock_data.GetBlock(time_index)
    
    @classmethod
    def get_time_info(cls, vtk_data: vtk.vtkDataObject) -> Dict[str, Any]:
        """
        Get comprehensive time information from VTK dataset.
        
        Args:
            vtk_data: VTK data object (multiblock or single dataset)
            
        Returns:
            Dictionary containing time information
        """
        if isinstance(vtk_data, vtk.vtkMultiBlockDataSet):
            # Handle multiblock datasets
            time_values = cls.get_time_values(vtk_data)
            
            if time_values is None:
                return {'is_time_dependent': False}
            
            return {
                'is_time_dependent': True,
                'num_time_steps': len(time_values),
                'time_values': time_values,
                'time_range': (time_values.min(), time_values.max()),
                'time_step': time_values[1] - time_values[0] if len(time_values) > 1 else 0.0
            }
        
        else:
            # Handle single datasets with multiple time arrays
            time_values = cls._get_single_dataset_time_values(vtk_data)
            
            if time_values is None:
                return {'is_time_dependent': False}
            
            return {
                'is_time_dependent': True,
                'num_time_steps': len(time_values),
                'time_values': time_values,
                'time_range': (time_values.min(), time_values.max()),
                'time_step': time_values[1] - time_values[0] if len(time_values) > 1 else 0.0
            }
    
    @classmethod
    def get_time_step_data(cls, vtk_data: vtk.vtkDataObject, time_index: int) -> Optional[vtk.vtkDataObject]:
        """
        Get data for a specific time step from time-dependent VTK data.
        
        Args:
            vtk_data: VTK data object (multiblock or single dataset)
            time_index: Time step index (0-based)
            
        Returns:
            VTK data object for the specific time step
        """
        if isinstance(vtk_data, vtk.vtkMultiBlockDataSet):
            return cls.get_time_step_multiblock(vtk_data, time_index)
        else:
            # For single datasets, create a copy and set the appropriate scalar array
            return cls._get_single_dataset_time_step(vtk_data, time_index)
    
    @classmethod
    def _get_single_dataset_time_step(cls, vtk_data: vtk.vtkDataObject, time_index: int) -> Optional[vtk.vtkDataObject]:
        """
        Get data for specific time step from single dataset with multiple time arrays.
        
        Args:
            vtk_data: VTK single dataset
            time_index: Time step index
            
        Returns:
            VTK dataset with appropriate scalar array set for visualization
        """
        import re
        
        # Get time values to map index to actual time
        time_values = cls._get_single_dataset_time_values(vtk_data)
        if time_values is None or time_index >= len(time_values):
            return None
        
        target_time = time_values[time_index]
        
        # Find the array name corresponding to this time
        point_data = vtk_data.GetPointData()
        target_array_name = None
        
        # Pattern to match our new format: electron_flux_t000, electron_flux_t001, etc.
        time_pattern_new = r"_t(\d{3})$"  # Matches _t000, _t001, etc.
        
        # Pattern to match old format: t+6h, t-6h, t+0h
        time_pattern_old = r"_t([+-]?)(\d+)h"
        
        print(f"Looking for time index {time_index} (time {target_time:.1f}h) among arrays:")
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            print(f"  Array {i}: {array_name}")
            if array_name:
                # Try new format first (electron_flux_t000, electron_flux_t001, etc.)
                match_new = re.search(time_pattern_new, array_name)
                if match_new:
                    array_time_index = int(match_new.group(1))
                    print(f"    Parsed index (new format): {array_time_index}")
                    
                    if array_time_index == time_index:
                        target_array_name = array_name
                        print(f"    MATCH! Using {array_name}")
                        break
                else:
                    # Fall back to old format (t+6h, t-6h, t+0h)
                    match_old = re.search(time_pattern_old, array_name)
                    if match_old:
                        sign = match_old.group(1) or "+"
                        hours = int(match_old.group(2))
                        time_value = hours if sign == "+" else -hours
                        print(f"    Parsed time (old format): {time_value:+.0f}h")
                        
                        if abs(time_value - target_time) < 0.1:  # Allow small floating point tolerance
                            target_array_name = array_name
                            print(f"    MATCH! Using {array_name}")
                            break
        
        if target_array_name is None:
            return None
        
        # Create a shallow copy and set the target array as active scalars
        output = vtk_data.NewInstance()
        output.ShallowCopy(vtk_data)
        
        # Set the time-specific array as the active scalars
        target_array = point_data.GetArray(target_array_name)
        if target_array:
            output.GetPointData().SetScalars(target_array)
            print(f"Set active scalars to {target_array_name} for time step {time_index} (t={target_time:+.0f}h)")
        
        return output
