#!/usr/bin/env python3
"""
Generate a simple single VTS file with embedded time information
for testing time-dependent flux visualization.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import os

def create_simple_wave_vts():
    """Create a single VTS file with a moving Gaussian wave"""
    
    # Grid parameters - smaller for single file
    nx, ny, nz = 30, 30, 60
    x_range = (-5000, 5000)   # km
    y_range = (-5000, 5000)   # km  
    z_range = (-10000, 10000) # km
    
    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create structured grid
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(nx, ny, nz)
    
    # Create points and initial flux values (t=0)
    points = vtk.vtkPoints()
    flux_values = []
    
    # Wave parameters for t=0
    wave_center_z = -8000  # Start near -Z end
    gaussian_sigma_xy = 1500.0  # km
    gaussian_sigma_z = 1000.0   # km
    wave_amplitude = 5e7        # particles/cm²/s/sr/MeV
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Add point
                points.InsertNextPoint(X[i, j, k], Y[i, j, k], Z[i, j, k])
                
                # Calculate Gaussian wave value at t=0
                r_xy = np.sqrt(X[i, j, k]**2 + Y[i, j, k]**2)
                r_z = abs(Z[i, j, k] - wave_center_z)
                
                flux_val = wave_amplitude * np.exp(-(r_xy**2) / (2 * gaussian_sigma_xy**2)) * \
                                          np.exp(-(r_z**2) / (2 * gaussian_sigma_z**2))
                
                flux_values.append(flux_val)
    
    # Set points
    structured_grid.SetPoints(points)
    
    # Add flux data
    flux_array = numpy_to_vtk(np.array(flux_values))
    flux_array.SetName("electron_flux")
    structured_grid.GetPointData().SetScalars(flux_array)
    
    return structured_grid

def main():
    """Generate simple test VTS file"""
    print("Generating simple Gaussian wave VTS file...")
    
    # Create output directory
    output_dir = "data/flux"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the grid
    grid = create_simple_wave_vts()
    
    # Save as VTS file
    output_file = os.path.join(output_dir, "simple_gaussian_wave.vts")
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()
    writer.Write()
    
    print(f"Generated: {output_file}")
    print("\nThis is a static VTS file showing a Gaussian blob.")
    print("For time-dependent testing, use: data/flux/gaussian_wave_test.vtm")

if __name__ == "__main__":
    main()