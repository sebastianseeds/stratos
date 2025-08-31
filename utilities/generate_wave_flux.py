#!/usr/bin/env python3
"""
Generate a time-dependent VTS file with a Gaussian XY plane wave
propagating from -Z to +Z for testing time-dependent flux visualization.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import os

def create_gaussian_wave_flux():
    """
    Create a time-dependent flux field with a Gaussian wave propagating in Z direction.
    The wave will be a Gaussian blob in XY plane that moves from -Z to +Z over time.
    """
    
    # Grid parameters
    nx, ny, nz = 50, 50, 100  # Resolution
    x_range = (-10000, 10000)  # km
    y_range = (-10000, 10000)  # km  
    z_range = (-20000, 20000)  # km
    
    # Time parameters
    num_time_steps = 50
    time_start = 0.0  # hours
    time_end = 24.0   # hours
    time_values = np.linspace(time_start, time_end, num_time_steps)
    
    # Wave parameters
    gaussian_sigma_xy = 3000.0  # km, width of Gaussian in XY
    gaussian_sigma_z = 2000.0   # km, width of Gaussian in Z
    wave_amplitude = 1e8        # particles/cm²/s/sr/MeV
    wave_speed_km_per_hour = 3000.0  # Speed of wave propagation
    
    print(f"Creating {num_time_steps} time steps from {time_start} to {time_end} hours")
    print(f"Grid size: {nx}×{ny}×{nz}")
    print(f"Spatial domain: X=[{x_range[0]}, {x_range[1]}], Y=[{y_range[0]}, {y_range[1]}], Z=[{z_range[0]}, {z_range[1]}] km")
    print(f"Wave speed: {wave_speed_km_per_hour} km/h")
    
    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create multiblock dataset for time series
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(num_time_steps)
    
    for t_idx, time_hours in enumerate(time_values):
        print(f"  Generating time step {t_idx+1}/{num_time_steps} (t = {time_hours:.1f} h)")
        
        # Calculate wave center Z position (moves from -Z to +Z)
        wave_center_z = z_range[0] + (time_hours - time_start) / (time_end - time_start) * (z_range[1] - z_range[0])
        
        # Create structured grid for this time step
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        flux_values = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Add point
                    points.InsertNextPoint(X[i, j, k], Y[i, j, k], Z[i, j, k])
                    
                    # Calculate Gaussian wave value
                    # Distance from wave center in XY plane
                    r_xy = np.sqrt(X[i, j, k]**2 + Y[i, j, k]**2)
                    
                    # Distance from wave center in Z direction
                    r_z = abs(Z[i, j, k] - wave_center_z)
                    
                    # Gaussian in XY plane and Z direction
                    flux_val = wave_amplitude * np.exp(-(r_xy**2) / (2 * gaussian_sigma_xy**2)) * \
                                              np.exp(-(r_z**2) / (2 * gaussian_sigma_z**2))
                    
                    flux_values.append(flux_val)
        
        # Set points
        structured_grid.SetPoints(points)
        
        # Add flux data as scalar field
        flux_array = numpy_to_vtk(np.array(flux_values))
        flux_array.SetName("electron_flux")
        structured_grid.GetPointData().SetScalars(flux_array)
        
        # Add time value as field data
        time_array = vtk.vtkFloatArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, time_hours)
        structured_grid.GetFieldData().AddArray(time_array)
        
        # Add to multiblock
        multiblock.SetBlock(t_idx, structured_grid)
        multiblock.GetMetaData(t_idx).Set(vtk.vtkCompositeDataSet.NAME(), f"TimeStep_{t_idx}")
    
    return multiblock, time_values

def save_time_dependent_vts(multiblock, time_values, filename):
    """Save the multiblock dataset as a time-dependent VTS file with uniform XY plane wave"""
    
    # Grid parameters - extend to ±80000 km as requested
    nx, ny, nz = 80, 80, 80  # Higher resolution for large domain
    x_range = (-80000, 80000)  # km - full domain
    y_range = (-80000, 80000)  # km - full domain  
    z_range = (-80000, 80000)  # km - full domain
    
    # Time parameters - more time steps for smoother animation
    num_time_steps = 20
    time_vals = np.linspace(time_values[0], time_values[-1], num_time_steps)
    
    print(f"Creating uniform XY plane wave VTS with {num_time_steps} time steps")
    print(f"Grid size: {nx}×{ny}×{nz} = {nx*ny*nz} points")
    print(f"Domain: {x_range[0]} to {x_range[1]} km in all axes")
    
    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create structured grid
    output_grid = vtk.vtkStructuredGrid()
    output_grid.SetDimensions(nx, ny, nz)
    
    # Create points
    points = vtk.vtkPoints()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.InsertNextPoint(X[i, j, k], Y[i, j, k], Z[i, j, k])
    output_grid.SetPoints(points)
    
    # Uniform plane wave parameters
    plane_amplitude = 1e8        # particles/cm²/s/sr/MeV - uniform across XY plane
    plane_thickness = 5000.0     # km - thickness of the moving plane
    background_flux = 1e3        # particles/cm²/s/sr/MeV - very low background
    
    # Add flux data for each time step as separate arrays
    for t_idx, time_hours in enumerate(time_vals):
        print(f"  Adding time step {t_idx}: t={time_hours:.1f}h")
        
        # Calculate plane center Z position (moves from -80000 to +80000)
        progress = (time_hours - time_vals[0]) / (time_vals[-1] - time_vals[0])
        plane_center_z = z_range[0] + progress * (z_range[1] - z_range[0])
        
        print(f"    Plane at Z = {plane_center_z:.0f} km")
        
        flux_values = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Distance from current Z position to the moving plane
                    z_dist = abs(Z[i, j, k] - plane_center_z)
                    
                    # Create uniform plane: high flux near plane center, background elsewhere
                    if z_dist <= plane_thickness:
                        # Gaussian falloff within the plane thickness
                        flux_val = plane_amplitude * np.exp(-(z_dist**2) / (2 * (plane_thickness/3)**2))
                    else:
                        # Low background flux
                        flux_val = background_flux
                    
                    flux_values.append(flux_val)
        
        # Create array for this time step
        time_flux_array = numpy_to_vtk(np.array(flux_values))
        time_flux_array.SetName(f"electron_flux_t{t_idx:03d}")
        output_grid.GetPointData().AddArray(time_flux_array)
        
        # Set first time step as active scalars
        if t_idx == 0:
            output_grid.GetPointData().SetScalars(time_flux_array)
    
    # Add time values as field data
    time_val_array = numpy_to_vtk(time_vals)
    time_val_array.SetName("TimeValues")
    output_grid.GetFieldData().AddArray(time_val_array)
    
    # Add number of time steps
    num_steps_array = vtk.vtkIntArray()
    num_steps_array.SetName("NumberOfTimeSteps")
    num_steps_array.SetNumberOfTuples(1)
    num_steps_array.SetValue(0, num_time_steps)
    output_grid.GetFieldData().AddArray(num_steps_array)
    
    # Write VTS file
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(output_grid)
    writer.SetDataModeToAscii()
    writer.Write()
    
    print(f"Saved time-dependent VTS file: {filename}")
    print(f"Contains {num_time_steps} time steps as separate scalar arrays")
    print(f"Time range: {time_vals[0]:.1f} to {time_vals[-1]:.1f} hours")
    
    return filename

def main():
    """Main function to generate the test flux file"""
    print("Generating Gaussian wave flux field for testing...")
    
    # Create output directory if it doesn't exist
    output_dir = "data/flux"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the flux field
    multiblock, time_values = create_gaussian_wave_flux()
    
    # Save as time-dependent VTS file
    output_file = os.path.join(output_dir, "gaussian_wave_test.vts")
    vts_file = save_time_dependent_vts(multiblock, time_values, output_file)
    
    print(f"\nGeneration complete!")
    print(f"Output file: {vts_file}")
    print(f"\nTo test:")
    print(f"  1. Load {vts_file} in STRATOS")
    print(f"  2. Start animation to see the Gaussian wave propagate from -Z to +Z")
    print(f"  3. The wave should take 24 hours to traverse the domain")
    print(f"  4. The VTS file contains all {len(time_values)} time steps as separate scalar arrays")

if __name__ == "__main__":
    main()