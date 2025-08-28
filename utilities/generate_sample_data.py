#!/usr/bin/env python3
"""
Fixed VTK Data Generator for Electron Flux Visualizer
Creates proper structured grid data with corrected Van Allen belt orientation and extended data range.
"""

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import time
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """Create terminal progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def van_allen_belt_model(x, y, z):
    """
    Accurate Van Allen radiation belt model with correct magnetic field orientation.
    
    The Van Allen belts are aligned with Earth's magnetic dipole field, which is:
    - Tilted ~11° from the rotation axis
    - The magnetic poles are roughly at geographic coordinates:
      North: 86.5°N, 164.4°W (in northern Canada)
      South: 64.4°S, 137.0°E (near Antarctica)
    
    For simplicity, we'll model the tilt as primarily in the X-Z plane.
    """
    earth_radius = 6371.0  # km
    
    # Magnetic dipole tilt (11 degrees from Z-axis toward positive X)
    magnetic_tilt = np.radians(11.0)
    
    # Rotate coordinates to magnetic coordinate system
    # Rotation matrix around Y-axis by magnetic_tilt
    cos_tilt = np.cos(magnetic_tilt)
    sin_tilt = np.sin(magnetic_tilt)
    
    # Transform to magnetic coordinates
    x_mag = x * cos_tilt + z * sin_tilt
    y_mag = y  # No change in Y
    z_mag = -x * sin_tilt + z * cos_tilt
    
    # Distance from Earth center in magnetic coordinates
    r_mag = np.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
    
    # Magnetic latitude (angle from magnetic equator)
    if r_mag > earth_radius:
        magnetic_latitude = np.arcsin(np.abs(z_mag) / r_mag)
    else:
        return 0.0  # Inside Earth
    
    # L-shell parameter (distance to equatorial crossing)
    L_shell = r_mag / (earth_radius * np.cos(magnetic_latitude)**2)
    
    flux = 0.0
    
    # Inner Van Allen Belt (L = 1.2 to 2.5, peak at L ≈ 1.5)
    if 1.2 <= L_shell <= 2.8:
        # Electron flux peaks around L = 1.6
        inner_peak = 1.6
        inner_width = 0.4
        inner_amplitude = 2e7  # particles/(cm²·s·sr·MeV)
        
        # Gaussian profile in L-shell
        inner_flux = inner_amplitude * np.exp(-((L_shell - inner_peak)**2) / (2 * inner_width**2))
        
        # Latitudinal dependence (stronger at higher latitudes)
        lat_factor = 1.0 + 2.0 * np.sin(magnetic_latitude)**2
        
        flux += inner_flux * lat_factor
    
    # Slot Region (L = 2.5 to 3.0) - reduced flux
    if 2.5 <= L_shell <= 3.2:
        slot_flux = 1e5 * np.exp(-((L_shell - 2.8)**2) / (2 * 0.2**2))
        flux += slot_flux
    
    # Outer Van Allen Belt (L = 3.0 to 7.0, peak at L ≈ 4.5)
    if 3.0 <= L_shell <= 8.0:
        # Outer belt is more variable and extends further
        outer_peak = 4.5
        outer_width = 1.2
        outer_amplitude = 1e7  # particles/(cm²·s·sr·MeV)
        
        # Broader Gaussian profile
        outer_flux = outer_amplitude * np.exp(-((L_shell - outer_peak)**2) / (2 * outer_width**2))
        
        # Latitudinal dependence
        lat_factor = 1.0 + 1.5 * np.sin(magnetic_latitude)**2
        
        # Local time asymmetry (day/night effect)
        local_time_angle = np.arctan2(y_mag, x_mag)  # Approximate local time
        day_night_factor = 1.0 + 0.3 * np.cos(local_time_angle)
        
        flux += outer_flux * lat_factor * day_night_factor
    
    # Add some plasma sheet contribution at high L-shells
    if L_shell > 6.0:
        plasma_sheet_flux = 5e5 * np.exp(-(L_shell - 8.0)**2 / (2 * 2.0**2))
        flux += plasma_sheet_flux
    
    # Add small amount of background/noise
    if flux > 0:
        noise_factor = 1.0 + 0.1 * np.random.normal(0, 1)
        flux *= max(0.1, noise_factor)  # Ensure flux stays positive
        
        # Add small uniform background
        flux += np.random.uniform(0, 1e4)
    
    return max(0.0, flux)

def create_structured_grid_vtk(filename="electron_flux_structured.vts"):
    """Create a proper structured grid VTK file with extended range and correct Van Allen belts"""
    
    print(f"Creating structured grid VTK data: {filename}")
    start_time = time.time()
    
    # EXTENDED grid to capture full Van Allen belt system
    nx, ny, nz = 60, 60, 60  # Increased resolution
    total_points = nx * ny * nz
    
    print(f"  Generating {nx}×{ny}×{nz} = {total_points:,} grid points...")
    
    # Create structured grid
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nx, ny, nz)
    
    # Create points
    points = vtk.vtkPoints()
    
    # EXTENDED grid bounds to capture L-shells up to L=10
    # This corresponds to about 10 * 6371 = 63,710 km from Earth center
    earth_radius = 6371  # km
    max_distance = 12 * earth_radius  # ~76,000 km (well beyond GEO at 42,164 km)
    
    x_min, x_max = -max_distance, max_distance
    y_min, y_max = -max_distance, max_distance  
    z_min, z_max = -max_distance, max_distance
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    flux_values = []
    
    print("  Computing Van Allen belt flux field with correct orientation...")
    print(f"  Grid extends from {x_min/1000:.0f} to {x_max/1000:.0f} Mm")
    print(f"  This covers L-shells from 1 to ~{max_distance/earth_radius:.1f}")
    
    point_count = 0
    significant_flux_points = 0
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                xi, yj, zk = x[i], y[j], z[k]
                points.InsertNextPoint(xi, yj, zk)
                
                # Use the corrected Van Allen belt model
                flux = van_allen_belt_model(xi, yj, zk)
                
                flux_values.append(flux)
                point_count += 1
                
                if flux > 1000:  # Count significant flux points
                    significant_flux_points += 1
                
                if point_count % 5000 == 0:
                    print_progress_bar(point_count, total_points, prefix='    Computing', 
                                     suffix=f'({significant_flux_points:,} significant)')
    
    print_progress_bar(total_points, total_points, prefix='    Computing', 
                      suffix=f'({significant_flux_points:,} significant)')
    
    sgrid.SetPoints(points)
    
    # Add flux data
    print("  Adding flux data to grid...")
    flux_array = numpy_to_vtk(np.array(flux_values), deep=True)
    flux_array.SetName("electron_flux")
    sgrid.GetPointData().SetScalars(flux_array)
    
    # Write XML structured grid file
    print(f"  Writing XML file: {filename}")
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(sgrid)
    writer.Write()
    
    elapsed_time = time.time() - start_time
    flux_array_np = np.array(flux_values)
    
    print(f"✓ Structured grid VTK file created!")
    print(f"  Processing time: {elapsed_time:.1f} seconds")
    print(f"  Grid dimensions: {nx} × {ny} × {nz}")
    print(f"  Grid extent: ±{max_distance/1000:.0f} Mm (±{max_distance/earth_radius:.1f} Earth radii)")
    print(f"  Total points: {total_points:,}")
    print(f"  Significant flux points: {significant_flux_points:,} ({significant_flux_points/total_points*100:.1f}%)")
    print(f"  Flux range: {flux_array_np.min():.2e} to {flux_array_np.max():.2e}")
    print(f"  Non-zero flux points: {np.sum(flux_array_np > 0):,}")
    print(f"  File: {filename}")
    
    return sgrid

def create_unstructured_grid_vtk(filename="electron_flux_unstructured.vtu"):
    """Create an unstructured grid VTK file with extended range and corrected Van Allen belts"""
    
    print(f"Creating unstructured grid VTK data: {filename}")
    start_time = time.time()
    
    earth_radius = 6371  # km
    
    # EXTENDED radial sampling to capture full magnetosphere
    r_shells = np.concatenate([
        np.linspace(1.1 * earth_radius, 3 * earth_radius, 15),    # Inner belt region
        np.linspace(3.1 * earth_radius, 8 * earth_radius, 20),   # Outer belt region  
        np.linspace(8.5 * earth_radius, 15 * earth_radius, 10)   # Plasma sheet region
    ])
    
    # Higher angular resolution for better belt structure
    n_theta = 36  # polar angle (every 5 degrees)
    n_phi = 72    # azimuthal angle (every 5 degrees)
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    
    total_points = len(r_shells) * n_theta * n_phi
    print(f"  Generating {len(r_shells)} × {n_theta} × {n_phi} = {total_points:,} points...")
    print(f"  Radial range: {r_shells[0]/earth_radius:.2f} to {r_shells[-1]/earth_radius:.1f} Earth radii")
    
    points = vtk.vtkPoints()
    flux_values = []
    
    point_count = 0
    significant_points = 0
    
    print("  Computing Van Allen belt flux in spherical shells...")
    
    for i, r in enumerate(r_shells):
        for j, t in enumerate(theta):
            for k, p in enumerate(phi):
                # Convert to Cartesian
                x = r * np.sin(t) * np.cos(p)
                y = r * np.sin(t) * np.sin(p)
                z = r * np.cos(t)
                
                # Use corrected Van Allen belt model
                flux = van_allen_belt_model(x, y, z)
                
                # Lower threshold to capture more belt structure
                if flux > 100:  # Much lower threshold
                    points.InsertNextPoint(x, y, z)
                    flux_values.append(flux)
                    significant_points += 1
                
                point_count += 1
                if point_count % 10000 == 0:
                    print_progress_bar(point_count, total_points, prefix='    Sampling',
                                     suffix=f'({significant_points:,} significant)')
    
    print_progress_bar(total_points, total_points, prefix='    Sampling',
                      suffix=f'({significant_points:,} significant)')
    
    print(f"  Creating unstructured grid with {significant_points:,} points...")
    
    # Create unstructured grid
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    
    # Create vertex cells
    for i in range(points.GetNumberOfPoints()):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        ugrid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())
        
        if (i + 1) % 5000 == 0:
            print_progress_bar(i + 1, points.GetNumberOfPoints(), prefix='    Cells')
    
    print_progress_bar(points.GetNumberOfPoints(), points.GetNumberOfPoints(), prefix='    Cells')
    
    # Add flux data
    print("  Adding flux data...")
    flux_array = numpy_to_vtk(np.array(flux_values), deep=True)
    flux_array.SetName("electron_flux")
    ugrid.GetPointData().SetScalars(flux_array)
    
    # Write XML unstructured grid file
    print(f"  Writing XML file: {filename}")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(ugrid)
    writer.Write()
    
    elapsed_time = time.time() - start_time
    flux_array_np = np.array(flux_values)
    
    print(f"✓ Unstructured grid VTK file created!")
    print(f"  Processing time: {elapsed_time:.1f} seconds")
    print(f"  Significant points: {significant_points:,}")
    print(f"  Radial extent: {r_shells[0]/1000:.0f} to {r_shells[-1]/1000:.0f} Mm")
    print(f"  Flux range: {flux_array_np.min():.2e} to {flux_array_np.max():.2e}")
    print(f"  File: {filename}")
    
    return ugrid

def create_legacy_vtk(filename="electron_flux_legacy.vtk"):
    """Create a legacy format VTK file with extended range"""
    
    print(f"Creating legacy VTK file: {filename}")
    start_time = time.time()
    
    # Extended structured points
    nx, ny, nz = 80, 80, 80
    
    # Create image data with extended range
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(nx, ny, nz)
    
    # Spacing to cover ±80,000 km (about 12.5 Earth radii)
    total_extent = 160000  # km (±80,000)
    spacing = total_extent / (nx - 1)
    imageData.SetSpacing(spacing, spacing, spacing)
    imageData.SetOrigin(-total_extent/2, -total_extent/2, -total_extent/2)
    
    print(f"  Grid spacing: {spacing:.0f} km")
    print(f"  Grid extent: ±{total_extent/2/1000:.0f} Mm")
    
    # Generate flux data
    print("  Generating extended Van Allen belt field...")
    flux_values = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Get world coordinates
                x = -total_extent/2 + i * spacing
                y = -total_extent/2 + j * spacing  
                z = -total_extent/2 + k * spacing
                
                # Use corrected Van Allen belt model
                flux = van_allen_belt_model(x, y, z)
                flux_values.append(flux)
        
        # Progress update
        if (k + 1) % 10 == 0:
            print_progress_bar(k + 1, nz, prefix='    Computing')
    
    print_progress_bar(nz, nz, prefix='    Computing')
    
    # Add scalar data
    flux_array = numpy_to_vtk(np.array(flux_values), deep=True)
    flux_array.SetName("electron_flux")
    imageData.GetPointData().SetScalars(flux_array)
    
    # Write legacy VTK file
    print(f"  Writing legacy VTK file: {filename}")
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()
    
    elapsed_time = time.time() - start_time
    flux_array_np = np.array(flux_values)
    
    print(f"✓ Legacy VTK file created!")
    print(f"  Processing time: {elapsed_time:.1f} seconds")
    print(f"  Dimensions: {nx} × {ny} × {nz}")
    print(f"  Spacing: {spacing:.0f} km")
    print(f"  Extent: ±{total_extent/2/1000:.0f} Mm")
    print(f"  Flux range: {flux_array_np.min():.2e} to {flux_array_np.max():.2e}")
    print(f"  Non-zero points: {np.sum(flux_array_np > 0):,}")
    print(f"  File: {filename}")

def create_sample_orbital_data(filename="sample_orbit.csv", orbit_type="LEO"):
    """Create sample orbital CSV data"""
    
    print(f"Creating orbital data: {filename} ({orbit_type})")
    start_time = time.time()
    
    if orbit_type == "LEO":
        altitude = 400  # km
        period = 1.5    # hours
        inclination = 51.6  # degrees
    elif orbit_type == "GEO":
        altitude = 35786  # km
        period = 24.0   # hours  
        inclination = 0.0  # degrees
    elif orbit_type == "HEO":  # Highly Elliptical Orbit
        altitude = 39000  # km (apogee)
        period = 12.0   # hours
        inclination = 63.4  # degrees (Molniya orbit)
    else:  # MEO
        altitude = 20200  # km
        period = 12.0   # hours
        inclination = 55.0  # degrees
    
    earth_radius = 6371  # km
    orbital_radius = earth_radius + altitude
    
    # More points for smoother animation through Van Allen belts
    n_points = 500
    times = np.linspace(0, period, n_points)
    
    print(f"  Computing {n_points} orbital positions...")
    
    inclination_rad = np.radians(inclination)
    
    positions = []
    velocities = []
    
    for idx, t in enumerate(times):
        # Mean anomaly
        M = 2 * np.pi * t / period
        
        # Position in orbital plane
        x_orb = orbital_radius * np.cos(M)
        y_orb = orbital_radius * np.sin(M)
        z_orb = 0
        
        # Rotate by inclination
        x = x_orb
        y = y_orb * np.cos(inclination_rad) - z_orb * np.sin(inclination_rad)
        z = y_orb * np.sin(inclination_rad) + z_orb * np.cos(inclination_rad)
        
        positions.append([x, y, z])
        
        # Velocity
        orbital_speed = 2 * np.pi * orbital_radius / (period * 3600)  # km/s
        
        vx_orb = -orbital_speed * np.sin(M)
        vy_orb = orbital_speed * np.cos(M)
        vz_orb = 0
        
        vx = vx_orb
        vy = vy_orb * np.cos(inclination_rad) - vz_orb * np.sin(inclination_rad)
        vz = vy_orb * np.sin(inclination_rad) + vz_orb * np.cos(inclination_rad)
        
        velocities.append([vx, vy, vz])
        
        if (idx + 1) % 100 == 0:
            print_progress_bar(idx + 1, n_points, prefix='    Computing')
    
    print_progress_bar(n_points, n_points, prefix='    Computing')
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'x': positions[:, 0],
        'y': positions[:, 1], 
        'z': positions[:, 2],
        'vx': velocities[:, 0],
        'vy': velocities[:, 1],
        'vz': velocities[:, 2]
    })
    
    df.to_csv(filename, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"  ✓ Orbital data created! ({elapsed_time:.2f}s)")
    print(f"    Points: {n_points}, Period: {period}h, Alt: {altitude:,}km")

def main():
    """Generate multiple VTK formats for testing with corrected Van Allen belts"""
    print("Electron Flux Visualizer - Fixed Data Generator")
    print("Van Allen Belt Corrections Applied:")
    print("  • Extended data range to L-shell = 15")
    print("  • Corrected magnetic dipole orientation (11° tilt)")
    print("  • Proper inner/outer belt separation")
    print("  • Day/night asymmetry")
    print("=" * 60)
    
    overall_start = time.time()
    
    print("\nGenerating VTK files with corrected Van Allen belts...")
    print("-" * 50)
    
    # Create XML structured grid (recommended)
    print("\n1. XML Structured Grid (.vts) - EXTENDED RANGE")
    create_structured_grid_vtk("flux_field_structured_fixed.vts")
    
    # Create XML unstructured grid  
    print("\n2. XML Unstructured Grid (.vtu) - EXTENDED RANGE")
    create_unstructured_grid_vtk("flux_field_unstructured_fixed.vtu")
    
    # Create legacy VTK
    print("\n3. Legacy VTK (.vtk) - EXTENDED RANGE")
    create_legacy_vtk("flux_field_legacy_fixed.vtk")
    
    # Create orbital data including HEO that passes through belts
    print("\n" + "-" * 50)
    print("Generating orbital data...")
    create_sample_orbital_data("orbit_leo_detailed.csv", "LEO")
    print()
    create_sample_orbital_data("orbit_meo_detailed.csv", "MEO")
    print()
    create_sample_orbital_data("orbit_geo_detailed.csv", "GEO")
    print()
    create_sample_orbital_data("orbit_heo_detailed.csv", "HEO")  # New: passes through both belts
    
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 60)
    print("✓ All corrected sample data generated successfully!")
    print(f"  Total time: {total_time:.1f} seconds")
    print("\nGenerated files with Van Allen belt fixes:")
    print("  • flux_field_structured_fixed.vts   - XML structured grid (RECOMMENDED)")
    print("  • flux_field_unstructured_fixed.vtu - XML unstructured grid")  
    print("  • flux_field_legacy_fixed.vtk       - Legacy VTK format")
    print("  • orbit_leo_detailed.csv            - LEO satellite trajectory")
    print("  • orbit_meo_detailed.csv            - MEO satellite trajectory")
    print("  • orbit_geo_detailed.csv            - GEO satellite trajectory")
    print("  • orbit_heo_detailed.csv            - HEO trajectory (through both belts)")
    
    print("\nKey fixes applied:")
    print("  ✓ Extended data range to ±76,000 km (L-shell up to 12)")
    print("  ✓ Corrected Van Allen belt magnetic orientation")
    print("  ✓ Proper inner belt (L=1.2-2.8) and outer belt (L=3-8)")
    print("  ✓ Magnetic dipole tilt (11° from rotation axis)")
    print("  ✓ Day/night flux asymmetry")
    print("  ✓ Latitudinal flux variations")
    
    print("\nTo test:")
    print("  1. python flux_visualizer.py")
    print("  2. Load VTK Data → flux_field_structured_fixed.vts")
    print("  3. Load Orbital CSV → orbit_heo_detailed.csv  (passes through both belts)")
    print("  4. Click Play to animate and see correct Van Allen belt structure")

if __name__ == '__main__':
    main()
