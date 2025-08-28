"""
VTK Data Loading Module for Flux Orbital Visualizer
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
        '.vtk': None  # Legacy format - requires detection
    }
    
    # File type descriptions for UI
    FILE_TYPE_DESCRIPTIONS = {
        '.vts': 'XML Structured Grid',
        '.vtu': 'XML Unstructured Grid',
        '.vtp': 'XML PolyData',
        '.vti': 'XML Image Data',
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
        
        # Validate the data
        cls._validate_data(output)
        
        # Setup scalar data if needed
        cls._setup_scalar_data(output)
        
        print(f"Successfully loaded VTK data:")
        print(f"  Type: {type(output).__name__}")
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
            'num_points': vtk_data.GetNumberOfPoints(),
            'num_cells': vtk_data.GetNumberOfCells(),
            'bounds': vtk_data.GetBounds(),
            'scalar_range': None,
            'scalar_name': None,
            'arrays': []
        }
        
        # Get scalar information
        point_data = vtk_data.GetPointData()
        scalar_array = point_data.GetScalars()
        
        if scalar_array:
            info['scalar_name'] = scalar_array.GetName()
            info['scalar_range'] = scalar_array.GetRange()
        
        # Get all arrays
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
