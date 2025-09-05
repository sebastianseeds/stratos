#!/usr/bin/env python3
"""
Test time-dependent detection logic directly without full imports
"""

import vtk


def is_time_dependent_simple(vtk_data):
    """Simple time-dependent detection logic"""
    
    # Check single dataset for time-dependent arrays
    # Look for field data with TimeValues array
    field_data = vtk_data.GetFieldData()
    if field_data and (field_data.GetArray("TimeValues") or field_data.GetArray("time_values")):
        print("✓ Found TimeValues field data")
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
    
    print(f"Found {len(time_arrays)} time arrays: {time_arrays[:3]}{'...' if len(time_arrays) > 3 else ''}")
    
    if len(time_arrays) >= 2:  # Need at least 2 time steps for animation
        print("✓ Found sufficient time arrays")
        return True
        
    return False


def test_file(filename):
    """Test a VTS file"""
    print(f"=== Testing {filename} ===")
    
    # Load the VTS file
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    
    if not vtk_data or vtk_data.GetNumberOfPoints() == 0:
        print("❌ Failed to load file or file is empty")
        return False
    
    print(f"Loaded {vtk_data.GetNumberOfPoints()} points")
    
    # Test time-dependent detection
    is_time_dep = is_time_dependent_simple(vtk_data)
    
    if is_time_dep:
        print("✅ DETECTED AS TIME-DEPENDENT")
    else:
        print("❌ NOT DETECTED AS TIME-DEPENDENT")
    
    print()
    return is_time_dep


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific file
        test_file(sys.argv[1])
    else:
        # Test both files
        test_file("data/flux/gaussian_wave_test.vts")
        test_file("data/flux/simple_gaussian_wave.vts")