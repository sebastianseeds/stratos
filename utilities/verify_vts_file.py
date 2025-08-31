#!/usr/bin/env python3
"""
Verify VTS file structure and check for time-dependent data.
"""

import vtk
import sys
import os


def verify_vts_file(filename):
    """Verify a VTS file and print detailed statistics"""
    
    if not os.path.exists(filename):
        print(f"ERROR: File {filename} does not exist")
        return False
    
    print(f"=== VTS File Verification: {filename} ===")
    print()
    
    try:
        # Load the VTS file
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_data = reader.GetOutput()
        
        if not vtk_data:
            print("ERROR: Failed to load VTS data")
            return False
        
        # Basic grid information
        dims = [0, 0, 0]
        vtk_data.GetDimensions(dims)
        num_points = vtk_data.GetNumberOfPoints()
        num_cells = vtk_data.GetNumberOfCells()
        
        print("GRID STRUCTURE:")
        print(f"  Dimensions: {dims[0]} × {dims[1]} × {dims[2]}")
        print(f"  Points: {num_points:,}")
        print(f"  Cells: {num_cells:,}")
        
        if num_points == 0:
            print("ERROR: Grid has no points - file may be corrupted")
            return False
        
        # Check bounds
        bounds = vtk_data.GetBounds()
        print(f"  Bounds: X=[{bounds[0]:.1f}, {bounds[1]:.1f}], "
              f"Y=[{bounds[2]:.1f}, {bounds[3]:.1f}], "
              f"Z=[{bounds[4]:.1f}, {bounds[5]:.1f}]")
        print()
        
        # Field data analysis
        field_data = vtk_data.GetFieldData()
        num_field_arrays = field_data.GetNumberOfArrays()
        
        print("FIELD DATA:")
        print(f"  Number of field arrays: {num_field_arrays}")
        
        time_values = None
        num_time_steps = None
        
        for i in range(num_field_arrays):
            array = field_data.GetArray(i)
            array_name = array.GetName()
            num_tuples = array.GetNumberOfTuples()
            
            print(f"  {i}: '{array_name}' ({num_tuples} values)")
            
            if array_name == "TimeValues":
                time_values = [array.GetValue(j) for j in range(num_tuples)]
                print(f"      Time values: {time_values}")
            elif array_name == "NumberOfTimeSteps":
                num_time_steps = array.GetValue(0)
                print(f"      Number of time steps: {num_time_steps}")
        
        print()
        
        # Point data analysis
        point_data = vtk_data.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()
        
        print("POINT DATA:")
        print(f"  Number of point data arrays: {num_point_arrays}")
        
        active_scalars = point_data.GetScalars()
        if active_scalars:
            print(f"  Active scalars: '{active_scalars.GetName()}'")
        else:
            print("  Active scalars: None")
        
        # Analyze each point data array
        time_step_arrays = []
        regular_arrays = []
        
        for i in range(num_point_arrays):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            num_tuples = array.GetNumberOfTuples()
            num_components = array.GetNumberOfComponents()
            array_range = array.GetRange()
            
            print(f"  {i}: '{array_name}' ({num_tuples:,} tuples, {num_components} components)")
            print(f"      Range: [{array_range[0]:.2e}, {array_range[1]:.2e}]")
            
            # Check if this looks like a time step array
            if "_t" in array_name and array_name.count("_t") == 1:
                time_step_arrays.append(array_name)
            else:
                regular_arrays.append(array_name)
        
        print()
        
        # Time dependency analysis
        print("TIME DEPENDENCY ANALYSIS:")
        
        is_time_dependent = False
        
        # Method 1: Check for TimeValues field data
        if time_values is not None and len(time_values) > 1:
            print(f"  ✓ Found TimeValues field data with {len(time_values)} time steps")
            print(f"    Time range: {time_values[0]:.1f} to {time_values[-1]:.1f}")
            is_time_dependent = True
        else:
            print("  ✗ No TimeValues field data found")
        
        # Method 2: Check for time step arrays
        if len(time_step_arrays) > 1:
            print(f"  ✓ Found {len(time_step_arrays)} time step arrays:")
            for array_name in sorted(time_step_arrays)[:5]:  # Show first 5
                print(f"    - {array_name}")
            if len(time_step_arrays) > 5:
                print(f"    ... and {len(time_step_arrays) - 5} more")
            is_time_dependent = True
        else:
            print(f"  ✗ Found only {len(time_step_arrays)} time step arrays (need ≥2)")
        
        # Method 3: Check NumberOfTimeSteps
        if num_time_steps and num_time_steps > 1:
            print(f"  ✓ NumberOfTimeSteps field indicates {num_time_steps} time steps")
            is_time_dependent = True
        else:
            print("  ✗ No NumberOfTimeSteps field or value ≤ 1")
        
        print()
        print("SUMMARY:")
        if is_time_dependent:
            print("  ✅ FILE IS TIME-DEPENDENT")
            print(f"     - Contains {len(time_step_arrays)} time step arrays")
            if time_values:
                print(f"     - Time range: {time_values[0]:.1f} to {time_values[-1]:.1f} hours")
                print(f"     - Time step count: {len(time_values)}")
        else:
            print("  ❌ FILE IS NOT TIME-DEPENDENT")
            print("     - Missing required time dependency markers")
        
        print(f"  Grid size: {dims[0]}×{dims[1]}×{dims[2]} = {num_points:,} points")
        print(f"  Data arrays: {num_point_arrays} point arrays, {num_field_arrays} field arrays")
        
        return is_time_dependent
        
    except Exception as e:
        print(f"ERROR: Failed to analyze VTS file: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 verify_vts_file.py <filename.vts>")
        sys.exit(1)
    
    filename = sys.argv[1]
    is_time_dependent = verify_vts_file(filename)
    
    # Exit code indicates time dependency status
    sys.exit(0 if is_time_dependent else 1)


if __name__ == "__main__":
    main()