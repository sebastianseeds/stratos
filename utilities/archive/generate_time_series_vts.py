"""
Time-dependent flux field generator for STRATOS with individual VTS files

This script generates realistic Van Allen radiation belt flux fields that evolve
over time during a coronal mass ejection (CME) event. Creates individual VTS files
that can be loaded sequentially in STRATOS.
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path
import time
import json

# Add the flux_visualizer module to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from flux_visualizer.core import OrbitalPoint


def numpy_to_vtk(array):
    """Convert numpy array to VTK array"""
    vtk_array = numpy_support.numpy_to_vtk(array, deep=True)
    return vtk_array


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


def cme_flux_model(x, y, z, time_hours):
    """
    Time-dependent flux model during CME event with Van Allen belt warping.
    
    Args:
        x, y, z: Position in km (GSM coordinates)
        time_hours: Time in hours relative to CME arrival
        
    Returns:
        Particle flux in particles/(cm²·s·sr·MeV)
    """
    
    earth_radius = 6371  # km
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Avoid singularity at Earth's center
    if r < earth_radius:
        return 0.0
    
    # Convert to L-shell (simplified dipole approximation)
    L_shell = r / earth_radius
    
    # Dst index model for storm effects
    if time_hours < 0:
        dst = -15 - 5 * np.exp(time_hours / 6)  # Pre-storm
    elif 0 <= time_hours <= 2:
        dst = -15 - 50 * time_hours  # Storm onset
    elif 2 < time_hours <= 8:
        dst = -115 - 30 * np.sin((time_hours - 2) * np.pi / 6)  # Main phase
    elif 8 < time_hours <= 24:
        dst = -115 * np.exp(-(time_hours - 8) / 8)  # Recovery
    else:
        dst = -15 + 5 * np.sin(time_hours * 0.1)  # Long-term recovery
    
    # === Van Allen Belt Structure ===
    
    # Inner belt (L = 1.2-2.5): relatively stable
    if 1.2 <= L_shell <= 2.5:
        inner_belt = 8e6 * np.exp(-((L_shell - 1.6) / 0.4)**2)
    else:
        inner_belt = 0
    
    # Slot region depletion
    slot_factor = 1 - 0.95 * np.exp(-((L_shell - 2.8) / 0.2)**2)
    
    # Outer belt (L = 3-7): highly variable
    if 3 <= L_shell <= 7:
        outer_belt = 3e5 * np.exp(-((L_shell - 4.2) / 1.0)**2)
    else:
        outer_belt = 0
    
    # Plasma sheet (L > 6)
    if L_shell > 6:
        plasma_sheet = 5e3 * np.exp(-(L_shell - 6) / 2)
    else:
        plasma_sheet = 0
    
    # Base flux
    base_flux = inner_belt + outer_belt * slot_factor + plasma_sheet
    
    # === Storm Effects ===
    
    storm_factor = 1.0
    
    if time_hours >= 0:  # During storm
        compression_strength = abs(dst) / 100.0
        
        # Belt compression - moves flux inward
        compression_factor = 1.0 - 0.2 * compression_strength
        L_compressed = L_shell * compression_factor
        
        # Enhanced flux during storm
        if 0 <= time_hours <= 6:
            enhancement = 1.5 + 1.0 * compression_strength * np.exp(-time_hours / 3)
        elif 6 < time_hours <= 18:
            enhancement = 1.2 + 0.8 * compression_strength * np.sin((time_hours - 6) * np.pi / 12)
        else:
            enhancement = 1.1 + 0.2 * compression_strength * np.exp(-(time_hours - 18) / 12)
        
        storm_factor = enhancement
        
        # Ring current effects - depression at L=3-5
        if 3 <= L_shell <= 5:
            ring_depression = 1.0 - 0.4 * compression_strength * np.exp(-((L_shell - 4) / 0.8)**2)
            storm_factor *= ring_depression
    
    # Apply storm effects
    final_flux = base_flux * storm_factor
    
    # Magnetopause boundary
    if time_hours >= 0:
        mp_compression = 1.0 + abs(dst) / 200.0
        mp_distance = 10 * earth_radius / np.power(mp_compression, 1/6)
    else:
        mp_distance = 11 * earth_radius
    
    if r > mp_distance:
        final_flux *= 0.01  # Solar wind
    
    # Add variability
    noise = 1.0 + 0.05 * np.sin(time_hours * 3.7) * np.cos(L_shell * 2.1)
    final_flux *= noise
    
    return max(final_flux, 1.0)


def create_time_series_vts_files(
    output_dir="data/flux/storm_evolution",
    time_start=-6,     # Start 6 hours before CME
    time_end=24,       # End 24 hours after CME arrival  
    time_step=3,       # 3-hour time steps
    nx=20, ny=20, nz=20  # Grid resolution
):
    """
    Create individual VTS files for each time step.
    
    Args:
        output_dir: Directory to store VTS files
        time_start: Start time in hours (relative to CME arrival)
        time_end: End time in hours
        time_step: Time step in hours
        nx, ny, nz: Grid resolution
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating time-dependent Van Allen belt flux series")
    print(f"Time range: {time_start} to {time_end} hours (relative to CME arrival)")
    print(f"Time step: {time_step} hours")
    print(f"Grid resolution: {nx} × {ny} × {nz}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Time array
    time_points = np.arange(time_start, time_end + time_step, time_step)
    n_timesteps = len(time_points)
    
    print(f"Generating {n_timesteps} VTS files...")
    
    # Grid setup
    earth_radius = 6371  # km
    max_distance = 8 * earth_radius  # ~51,000 km
    
    x = np.linspace(-max_distance, max_distance, nx)
    y = np.linspace(-max_distance, max_distance, ny)
    z = np.linspace(-max_distance, max_distance, nz)
    
    total_points = nx * ny * nz
    overall_start_time = time.time()
    
    file_list = []
    
    # Generate each time step as a separate VTS file
    for t_idx, current_time in enumerate(time_points):
        step_start_time = time.time()
        
        print(f"\nTime step {t_idx+1}/{n_timesteps}: t = {current_time:+.1f} hours")
        
        # Create structured grid for this time step
        sgrid = vtk.vtkStructuredGrid()
        sgrid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        flux_values = []
        significant_flux_points = 0
        
        print("  Computing flux field...")
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    xi, yj, zk = x[i], y[j], z[k]
                    points.InsertNextPoint(xi, yj, zk)
                    
                    # Time-dependent flux calculation
                    flux = cme_flux_model(xi, yj, zk, current_time)
                    flux_values.append(flux)
                    
                    if flux > 1000:
                        significant_flux_points += 1
                    
                    # Progress bar every 1000 points
                    if (k * ny * nx + j * nx + i + 1) % 1000 == 0:
                        progress = (k * ny * nx + j * nx + i + 1)
                        print_progress_bar(progress, total_points, prefix='    ',
                                         suffix=f'({significant_flux_points:,} significant)')
        
        print_progress_bar(total_points, total_points, prefix='    ',
                          suffix=f'({significant_flux_points:,} significant)')
        
        sgrid.SetPoints(points)
        
        # Add flux data
        flux_array = numpy_to_vtk(np.array(flux_values))
        flux_array.SetName("electron_flux")
        sgrid.GetPointData().SetScalars(flux_array)
        
        # Write VTS file
        filename = f"flux_t{current_time:+03.0f}h.vts"
        filepath = Path(output_dir) / filename
        
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(filepath))
        writer.SetInputData(sgrid)
        writer.Write()
        
        file_list.append(filename)
        
        step_time = time.time() - step_start_time
        flux_stats = np.array(flux_values)
        
        print(f"  Complete ({step_time:.1f}s)")
        print(f"    File: {filename}")
        print(f"    Flux range: {flux_stats.min():.2e} to {flux_stats.max():.2e}")
        print(f"    Significant points: {significant_flux_points:,} ({significant_flux_points/total_points*100:.1f}%)")
    
    # Create metadata file
    metadata = {
        'description': 'Time-dependent Van Allen belt flux during CME',
        'file_type': 'vtk_structured_grid_series',
        'time_start': float(time_points[0]),
        'time_end': float(time_points[-1]),
        'time_step': float(time_step),
        'n_timesteps': len(time_points),
        'files': file_list,
        'time_values': time_points.tolist(),
        'storm_phases': {
            'pre_storm': 't < 0h',
            'storm_onset': 't = 0-3h',
            'main_phase': 't = 3-9h', 
            'recovery': 't > 9h'
        }
    }
    
    metadata_file = Path(output_dir) / "time_series_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create simple orbital data
    create_sample_orbit(output_dir, time_start, time_end)
    
    total_time = time.time() - overall_start_time
    
    print("\n" + "=" * 60)
    print("Time-dependent flux VTS series generated successfully!")
    print(f"  Total processing time: {total_time:.1f} seconds")
    print(f"  Files created: {n_timesteps}")
    print(f"  Output directory: {output_dir}")
    
    print("\nTo visualize storm evolution in STRATOS:")
    print("  1. Load files in chronological order:")
    for i, filename in enumerate(file_list):
        time_val = time_points[i]
        print(f"     {filename} (t = {time_val:+.0f}h)")
    print("  2. Use animation controls to step through time")
    print("  3. Try different visualization modes during the storm")


def create_sample_orbit(output_dir, time_start, time_end):
    """Create a sample orbital trajectory"""
    
    print("  Creating sample orbital data...")
    
    # MEO orbit that passes through both belts
    altitude = 15000  # km
    inclination = 45  # degrees
    period = 8.0      # hours
    
    # Time points every 30 minutes
    time_hours = np.arange(time_start, time_end + 0.5, 0.5)
    
    positions = []
    for t in time_hours:
        # Circular orbit parameters
        angle = 2 * np.pi * t / period
        r = 6371 + altitude  # km from Earth center
        
        # Position in orbit
        x = r * np.cos(angle)
        y = r * np.sin(angle) * np.cos(np.radians(inclination))
        z = r * np.sin(angle) * np.sin(np.radians(inclination))
        
        positions.append([t, x, y, z, altitude])
    
    # Save orbital data
    orbit_file = Path(output_dir) / "sample_orbit.csv"
    
    with open(orbit_file, 'w') as f:
        f.write("# Sample MEO orbit during storm\n")
        f.write("time,x,y,z,altitude\n")
        for pos in positions:
            f.write(f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f},{pos[3]:.1f},{pos[4]:.1f}\n")
    
    print(f"  Orbital data: {orbit_file} ({len(positions)} points)")


if __name__ == "__main__":
    # Generate time-dependent flux as individual VTS files
    create_time_series_vts_files(
        output_dir="data/flux/storm_evolution",
        time_start=-6,   # 6 hours before storm
        time_end=18,     # 18 hours after storm arrival
        time_step=3,     # 3-hour time steps (9 files)
        nx=25, ny=25, nz=25  # Good resolution for detailed visualization
    )
    
    print("\n" + "=" * 60)
    print("Storm evolution flux files ready for STRATOS!")
    print("Load individual files to see Van Allen belt changes over time")
    print("=" * 60)