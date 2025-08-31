"""
Create a working VTM multiblock file from individual VTS files for STRATOS
"""

import numpy as np
import vtk
from pathlib import Path
from vtk.util import numpy_support


def create_multiblock_from_vts_files(
    vts_directory="data/flux/storm_evolution",
    output_file="data/flux/storm_evolution/time_series.vtm",
    time_start=-6,
    time_step=3
):
    """
    Create a VTM multiblock file from individual VTS files.
    
    Args:
        vts_directory: Directory containing VTS files
        output_file: Output VTM file path
        time_start: Start time in hours
        time_step: Time step in hours
    """
    
    vts_dir = Path(vts_directory)
    if not vts_dir.exists():
        raise FileNotFoundError(f"Directory not found: {vts_directory}")
    
    # Find all VTS files in chronological order
    vts_files = sorted(vts_dir.glob("flux_t*.vts"))
    if not vts_files:
        raise FileNotFoundError(f"No VTS files found in {vts_directory}")
    
    print(f"Found {len(vts_files)} VTS files")
    
    # Create multiblock dataset
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(len(vts_files))
    
    # Time values array
    time_values = []
    
    # Load each VTS file as a block
    for i, vts_file in enumerate(vts_files):
        print(f"Loading block {i}: {vts_file.name}")
        
        # Load VTS file
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(vts_file))
        reader.Update()
        
        sgrid = reader.GetOutput()
        if sgrid.GetNumberOfPoints() == 0:
            print(f"Warning: {vts_file.name} has no points")
            continue
        
        # Add to multiblock
        multiblock.SetBlock(i, sgrid)
        multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), f"t={time_start + i * time_step:+.0f}h")
        
        # Calculate time value
        time_value = time_start + i * time_step
        time_values.append(time_value)
        
        print(f"  Block {i}: {sgrid.GetNumberOfPoints()} points, t={time_value:+.0f}h")
    
    # Add time information to field data
    time_array = vtk.vtkDoubleArray()
    time_array.SetName("TimeValue")
    time_array.SetNumberOfTuples(len(time_values))
    for i, t in enumerate(time_values):
        time_array.SetValue(i, t)
    
    multiblock.GetFieldData().AddArray(time_array)
    
    # Add metadata
    info_array = vtk.vtkStringArray()
    info_array.SetName("Description")
    info_array.SetNumberOfTuples(1)
    info_array.SetValue(0, "Van Allen belt flux evolution during geomagnetic storm")
    multiblock.GetFieldData().AddArray(info_array)
    
    # Write VTM file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(multiblock)
    writer.Write()
    
    print(f"\nMultiblock VTM file created: {output_path}")
    print(f"Blocks: {len(vts_files)}")
    print(f"Time range: {time_values[0]:+.0f}h to {time_values[-1]:+.0f}h")
    print(f"Time step: {time_step}h")
    
    return str(output_path)


def verify_vtm_file(vtm_file):
    """Verify the created VTM file"""
    print(f"\nVerifying VTM file: {vtm_file}")
    
    # Load VTM file
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_file)
    reader.Update()
    
    multiblock = reader.GetOutput()
    
    print(f"Number of blocks: {multiblock.GetNumberOfBlocks()}")
    
    # Check field data
    field_data = multiblock.GetFieldData()
    if field_data:
        time_array = field_data.GetArray("TimeValue")
        if time_array:
            print("Time values found:")
            for i in range(time_array.GetNumberOfTuples()):
                print(f"  Block {i}: {time_array.GetValue(i):+.1f}h")
        else:
            print("No TimeValue array found in field data")
    
    # Check blocks
    for i in range(multiblock.GetNumberOfBlocks()):
        block = multiblock.GetBlock(i)
        if block:
            name = multiblock.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) or f"Block {i}"
            print(f"Block {i} ({name}): {block.GetNumberOfPoints()} points")
            
            # Check scalar data
            point_data = block.GetPointData()
            scalars = point_data.GetScalars()
            if scalars:
                scalar_range = scalars.GetRange()
                print(f"  Scalars: {scalars.GetName()}, range: {scalar_range[0]:.2e} to {scalar_range[1]:.2e}")
        else:
            print(f"Block {i}: Empty")
    
    print("Verification complete")


if __name__ == "__main__":
    # Create VTM from existing VTS files
    try:
        vtm_file = create_multiblock_from_vts_files()
        verify_vtm_file(vtm_file)
        print("\nVTM file ready for STRATOS!")
        print("Load this file to see storm evolution animation")
        
    except Exception as e:
        print(f"Error: {e}")