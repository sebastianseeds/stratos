"""
VTM file verification utility for STRATOS

This script reads a VTM (VTK MultiBlock) file and verifies that it contains
time-dependent data with proper structure and metadata.
"""

import vtk
import numpy as np
from pathlib import Path
import sys
import argparse


def verify_vtm_file(file_path):
    """
    Verify a VTM file contains valid time-dependent data.
    
    Args:
        file_path: Path to the VTM file
    """
    file_path = Path(file_path)
    
    print("=" * 60)
    print(f"VTM File Verification: {file_path.name}")
    print("=" * 60)
    
    # Check if file exists
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False
    
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        # Load the multiblock data
        print("\nLoading multiblock dataset...")
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(str(file_path))
        reader.Update()
        
        multiblock = reader.GetOutput()
        
        if not isinstance(multiblock, vtk.vtkMultiBlockDataSet):
            print(f"ERROR: Expected vtkMultiBlockDataSet, got {type(multiblock).__name__}")
            return False
        
        print(f"Successfully loaded multiblock dataset")
        
        # Basic multiblock info
        num_blocks = multiblock.GetNumberOfBlocks()
        print(f"\nMultiblock Structure:")
        print(f"   Number of blocks: {num_blocks}")
        
        if num_blocks == 0:
            print("ERROR: No blocks found in multiblock dataset")
            return False
        
        # Check for time information
        print(f"\nTime Information:")
        field_data = multiblock.GetFieldData()
        
        if not field_data:
            print("ERROR: No field data found")
            return False
        
        time_array = field_data.GetArray("TimeValue")
        
        if not time_array:
            print("ERROR: No 'TimeValue' array found in field data")
            return False
        
        # Extract time values
        n_times = time_array.GetNumberOfTuples()
        time_values = np.zeros(n_times)
        for i in range(n_times):
            time_values[i] = time_array.GetValue(i)
        
        print(f"Time-dependent data detected:")
        print(f"   Time steps: {len(time_values)}")
        print(f"   Time range: {time_values.min():.1f}h to {time_values.max():.1f}h")
        print(f"   Time step: {time_values[1] - time_values[0]:.1f}h (assuming uniform)")
        print(f"   Duration: {time_values.max() - time_values.min():.1f} hours")
        
        # Verify block count matches time steps
        if num_blocks != len(time_values):
            print(f"WARNING: Block count ({num_blocks}) != time steps ({len(time_values)})")
        
        # Analyze individual blocks
        print(f"\nBlock Analysis:")
        
        block_info = []
        for i in range(min(num_blocks, 5)):  # Check first 5 blocks
            block = multiblock.GetBlock(i)
            
            if block is None:
                print(f"   Block {i}: NULL")
                continue
            
            block_type = type(block).__name__
            num_points = block.GetNumberOfPoints()
            num_cells = block.GetNumberOfCells()
            
            # Get scalar info
            point_data = block.GetPointData()
            scalar_array = point_data.GetScalars()
            
            scalar_name = "None"
            scalar_range = (0, 0)
            
            if scalar_array:
                scalar_name = scalar_array.GetName()
                scalar_range = scalar_array.GetRange()
            
            block_info.append({
                'index': i,
                'type': block_type,
                'points': num_points,
                'cells': num_cells,
                'scalar_name': scalar_name,
                'scalar_range': scalar_range,
                'time': time_values[i] if i < len(time_values) else 'N/A'
            })
            
            print(f"   Block {i} (t={time_values[i] if i < len(time_values) else 'N/A'}h):")
            print(f"     Type: {block_type}")
            print(f"     Points: {num_points:,}")
            print(f"     Cells: {num_cells:,}")
            print(f"     Scalar: {scalar_name} [{scalar_range[0]:.2e}, {scalar_range[1]:.2e}]")
        
        if num_blocks > 5:
            print(f"   ... and {num_blocks - 5} more blocks")
        
        # Check data consistency across blocks
        print(f"\nConsistency Check:")
        
        first_block = multiblock.GetBlock(0)
        if first_block:
            reference_points = first_block.GetNumberOfPoints()
            reference_cells = first_block.GetNumberOfCells()
            reference_type = type(first_block).__name__
            
            consistent = True
            for i in range(1, min(num_blocks, 10)):  # Check first 10 blocks
                block = multiblock.GetBlock(i)
                if block:
                    if (block.GetNumberOfPoints() != reference_points or
                        block.GetNumberOfCells() != reference_cells or
                        type(block).__name__ != reference_type):
                        consistent = False
                        break
            
            if consistent:
                print(f"All blocks have consistent structure:")
                print(f"   Type: {reference_type}")
                print(f"   Points: {reference_points:,}")
                print(f"   Cells: {reference_cells:,}")
            else:
                print(f"WARNING: Blocks have inconsistent structure")
        
        # Memory estimation
        if first_block:
            # Estimate memory per time step
            points_size = reference_points * 3 * 4  # 3 coordinates, 4 bytes per float
            scalar_size = reference_points * 4      # 4 bytes per scalar
            cells_size = reference_cells * 8        # Rough estimate
            
            block_size_mb = (points_size + scalar_size + cells_size) / (1024 * 1024)
            total_size_mb = block_size_mb * num_blocks
            
            print(f"\nMemory Estimation:")
            print(f"   Per time step: ~{block_size_mb:.1f} MB")
            print(f"   Total dataset: ~{total_size_mb:.1f} MB")
        
        # Storm timeline analysis (if time values suggest storm data)
        print(f"\nStorm Timeline Analysis:")
        
        if time_values.min() < 0 and time_values.max() > 0:
            print("Detected storm timeline structure:")
            
            pre_storm = time_values[time_values < 0]
            main_phase = time_values[(time_values >= 0) & (time_values <= 8)]
            recovery = time_values[time_values > 8]
            
            print(f"   Pre-storm: {len(pre_storm)} steps ({pre_storm.min():.1f}h to {pre_storm.max():.1f}h)")
            print(f"   Main phase: {len(main_phase)} steps (0h to 8h)")
            print(f"   Recovery: {len(recovery)} steps (8h to {recovery.max():.1f}h)")
        else:
            print(f"Generic time series: {time_values.min():.1f}h to {time_values.max():.1f}h")
        
        print(f"\n" + "=" * 60)
        print("VTM file verification PASSED")
        print("Ready for STRATOS time-dependent visualization")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Verify VTM files for time-dependent flux visualization in STRATOS"
    )
    parser.add_argument(
        "file_path", 
        help="Path to the VTM file to verify"
    )
    
    args = parser.parse_args()
    
    success = verify_vtm_file(args.file_path)
    
    if not success:
        sys.exit(1)
    
    print(f"\nTo use in STRATOS:")
    print(f"   1. Launch STRATOS: python flux_visualizer.py")
    print(f"   2. Click 'Load Flux File(s)'")
    print(f"   3. Select: {args.file_path}")
    print(f"   4. Use animation controls to step through time")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("VTM File Verification Utility")
        print("============================")
        
        file_path = input("Enter path to VTM file: ").strip().strip('"\'')
        
        if file_path:
            verify_vtm_file(file_path)
        else:
            print("No file path provided")
    else:
        main()