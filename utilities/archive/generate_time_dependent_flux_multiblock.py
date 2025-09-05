"""
Time-dependent flux field generator for STRATOS with multiblock VTK output

This script generates realistic Van Allen radiation belt flux fields that evolve
over time during a coronal mass ejection (CME) event. Uses a single multiblock
VTK file to store all time steps.
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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


def dst_index_model(time_hours):
    """
    Model the Dst index evolution during a geomagnetic storm.
    
    Dst index represents ring current intensity:
    - Quiet conditions: ~0 nT
    - Storm sudden commencement: brief positive spike
    - Main phase: strong negative (down to -200 nT for severe storms)
    - Recovery phase: gradual return to quiet levels
    
    Args:
        time_hours: Time in hours relative to CME arrival (t=0)
        
    Returns:
        Dst index in nT
    """
    
    if time_hours < -6:
        # Pre-storm quiet conditions
        return -10 + 5 * np.sin(time_hours * 0.1)  # Quiet with small variations
    
    elif -6 <= time_hours < 0:
        # Pre-arrival enhancement
        return -10 - 15 * np.exp((time_hours + 6) / 3)
    
    elif 0 <= time_hours <= 1:
        # Storm sudden commencement (SSC) - brief positive spike
        return 20 * np.exp(-((time_hours - 0.5) / 0.3)**2) - 30
    
    elif 1 < time_hours <= 8:
        # Main phase - rapid intensification to minimum Dst
        t_normalized = (time_hours - 1) / 7
        return -30 - 120 * t_normalized * np.exp(-t_normalized / 0.7)
    
    elif 8 < time_hours <= 48:
        # Recovery phase - exponential recovery with multiple time constants
        t_recovery = time_hours - 8
        fast_recovery = -150 * np.exp(-t_recovery / 8)    # Fast initial recovery
        slow_recovery = -50 * np.exp(-t_recovery / 20)    # Slower tail recovery
        return fast_recovery + slow_recovery - 10
    
    else:
        # Long-term quiet
        return -10 + 3 * np.sin(time_hours * 0.05)


def cme_flux_model(x, y, z, time_hours):
    """
    Time-dependent flux model during CME event with Van Allen belt warping.
    
    Models the evolution of trapped particle populations in the magnetosphere
    during a geomagnetic storm, including belt compression, enhancement, and
    particle injection.
    
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
    
    # Get storm effects
    dst = dst_index_model(time_hours)
    
    # === Van Allen Belt Structure (Quiet Time Baseline) ===
    
    # Inner belt (L = 1.2-2.5): protons dominant, stable
    inner_belt = 5e6 * np.exp(-((L_shell - 1.5) / 0.5)**2) if 1.2 <= L_shell <= 2.5 else 0
    
    # Slot region (L = 2.5-3): depleted region
    slot_depletion = 1 - 0.9 * np.exp(-((L_shell - 2.8) / 0.3)**2)
    
    # Outer belt (L = 3-7): electrons dominant, highly variable
    outer_belt = 2e5 * np.exp(-((L_shell - 4.5) / 1.2)**2) if 3 <= L_shell <= 7 else 0
    
    # Plasma sheet (L > 7): tenuous, variable population
    if L_shell > 7:
        plasma_sheet = 1e3 * np.exp(-(L_shell - 7) / 3) * (1 + 0.5 * np.sin(time_hours * 0.2))
    else:
        plasma_sheet = 0
    
    # Base flux (quiet conditions)
    quiet_flux = inner_belt + outer_belt * slot_depletion + plasma_sheet
    
    # === Storm-Time Modifications ===
    
    # Belt compression and enhancement factors
    if 0 <= time_hours <= 6:
        # Main phase: compression and enhancement
        compression_strength = abs(dst) / 150.0  # Normalize by typical storm Dst
        
        # Magnetopause compression moves belts inward
        compression_factor = 1.0 - 0.3 * compression_strength
        
        # Enhanced convection creates new particle populations
        enhancement_factor = 1.0 + 2.0 * compression_strength * np.exp(-time_hours / 3)
        
        # Ring current creates local depressions
        if 2 <= L_shell <= 6:
            ring_current_depression = 1.0 - 0.6 * compression_strength * np.exp(-((L_shell - 4) / 1)**2)
        else:
            ring_current_depression = 1.0
            
    elif 6 < time_hours <= 24:
        # Recovery phase: particle acceleration and radial diffusion
        recovery_strength = abs(dst) / 150.0
        
        compression_factor = 1.0 - 0.1 * recovery_strength
        
        # Substorm-driven particle acceleration
        if 3 <= L_shell <= 6:
            substorm_acceleration = 1.0 + 1.5 * recovery_strength * np.sin(time_hours * 0.5)**2
            enhancement_factor = substorm_acceleration
        else:
            enhancement_factor = 1.0 + 0.3 * recovery_strength
            
        ring_current_depression = 1.0 - 0.2 * recovery_strength * np.exp(-((L_shell - 4) / 1.5)**2)
        
    else:
        # Quiet or late recovery
        compression_factor = 1.0
        enhancement_factor = 1.0
        ring_current_depression = 1.0
    
    # === Magnetopause Boundary Effects ===
    
    # Dynamic pressure effects on magnetopause position
    if time_hours >= 0:
        pressure_enhancement = 1.0 + 2.0 * np.exp(-time_hours / 4)
    else:
        pressure_enhancement = 1.0
    
    # Magnetopause standoff distance (Chapman-Ferraro, simplified)
    mp_distance = 10 * earth_radius / np.power(pressure_enhancement, 1/6)
    
    # Sharp cutoff at magnetopause
    if r > mp_distance:
        magnetopause_factor = 0.01  # Solar wind environment
    else:
        magnetopause_factor = 1.0
    
    # === Apply Storm Effects ===
    
    # Apply compression to effective L-shell
    L_effective = L_shell * compression_factor
    
    # Recompute flux with compressed coordinates
    if L_effective != L_shell and L_effective > 1:
        # Recalculate belt populations with compressed L-shell
        inner_belt_compressed = 5e6 * np.exp(-((L_effective - 1.5) / 0.5)**2) if 1.2 <= L_effective <= 2.5 else 0
        outer_belt_compressed = 2e5 * np.exp(-((L_effective - 4.5) / 1.2)**2) if 3 <= L_effective <= 7 else 0
        compressed_flux = inner_belt_compressed + outer_belt_compressed * slot_depletion + plasma_sheet
    else:
        compressed_flux = quiet_flux
    
    # Apply all storm effects
    storm_flux = compressed_flux * enhancement_factor * ring_current_depression * magnetopause_factor
    
    # === Additional Physical Effects ===
    
    # Energy-dependent loss processes
    if 1.5 <= L_shell <= 2.5:
        # Inner belt: stable, minimal losses
        loss_factor = 0.98
    elif 2.5 <= L_shell <= 6:
        # Outer belt: strong losses during storms
        loss_rate = 0.3 * abs(dst) / 100.0
        loss_factor = np.exp(-loss_rate * max(0, time_hours) / 24)
    else:
        # Plasma sheet: moderate losses
        loss_factor = 0.9
    
    storm_flux *= loss_factor
    
    # Time-dependent noise and variability
    noise_amplitude = 0.1 * (1 + abs(dst) / 100.0)
    time_noise = 1.0 + noise_amplitude * np.sin(time_hours * 2.3) * np.cos(time_hours * 1.7)
    storm_flux *= time_noise
    
    # Minimum flux floor
    return max(storm_flux, 1.0)


def create_time_dependent_vtk_multiblock(
    output_dir="data/flux/time_dependent",
    time_start=-12,    # Start 12 hours before CME
    time_end=72,       # End 72 hours after CME
    time_step=2,       # 2-hour time steps
    nx=50, ny=50, nz=50  # Grid resolution
):
    """
    Create a single multiblock VTK file with multiple time steps representing time-dependent flux evolution.
    
    Args:
        output_dir: Directory to store VTK file
        time_start: Start time in hours (relative to CME arrival)
        time_end: End time in hours
        time_step: Time step in hours
        nx, ny, nz: Grid resolution
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating time-dependent Van Allen belt flux multiblock dataset")
    print(f"Time range: {time_start} to {time_end} hours (relative to CME arrival)")
    print(f"Time step: {time_step} hours")
    print(f"Grid resolution: {nx} × {ny} × {nz}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Time array
    time_points = np.arange(time_start, time_end + time_step, time_step)
    n_timesteps = len(time_points)
    
    print(f"Generating {n_timesteps} time steps...")
    
    # Grid setup
    earth_radius = 6371  # km
    max_distance = 12 * earth_radius  # ~76,000 km
    
    x = np.linspace(-max_distance, max_distance, nx)
    y = np.linspace(-max_distance, max_distance, ny)
    z = np.linspace(-max_distance, max_distance, nz)
    
    total_points = nx * ny * nz
    overall_start_time = time.time()
    
    # Create single multiblock dataset with time steps
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(n_timesteps)
    
    # Time values array for the multiblock
    time_values = vtk.vtkDoubleArray()
    time_values.SetName("TimeValue")
    time_values.SetNumberOfTuples(n_timesteps)
    
    # Create points once (same grid for all time steps)
    points = vtk.vtkPoints()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                points.InsertNextPoint(x[i], y[j], z[k])
    
    for t_idx, current_time in enumerate(time_points):
        step_start_time = time.time()
        
        print(f"\nTime step {t_idx+1}/{n_timesteps}: t = {current_time:+.1f} hours")
        dst_value = dst_index_model(current_time)
        print(f"  Dst index: {dst_value:.1f} nT")
        
        # Create structured grid for this time step
        sgrid = vtk.vtkStructuredGrid()
        sgrid.SetDimensions(nx, ny, nz)
        sgrid.SetPoints(points)  # Reuse same points
        
        # Compute flux values for this time step
        print("  Computing flux field...")
        flux_values = []
        significant_flux_points = 0
        point_count = 0
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    xi, yj, zk = x[i], y[j], z[k]
                    
                    # Time-dependent flux calculation
                    flux = cme_flux_model(xi, yj, zk, current_time)
                    flux_values.append(flux)
                    point_count += 1
                    
                    if flux > 1000:
                        significant_flux_points += 1
                    
                    if point_count % 10000 == 0:
                        print_progress_bar(point_count, total_points, prefix='    Computing',
                                         suffix=f'({significant_flux_points:,} significant)')
        
        print_progress_bar(total_points, total_points, prefix='    Computing',
                          suffix=f'({significant_flux_points:,} significant)')
        
        # Add flux data to grid
        flux_array = numpy_to_vtk(np.array(flux_values))
        flux_array.SetName("electron_flux")
        sgrid.GetPointData().SetScalars(flux_array)
        
        # Add this grid to the multiblock dataset
        multiblock.SetBlock(t_idx, sgrid)
        multiblock.GetMetaData(t_idx).Set(vtk.vtkCompositeDataSet.NAME(), f"t={current_time:+.1f}h")
        
        # Store time value
        time_values.SetValue(t_idx, current_time)
        
        step_time = time.time() - step_start_time
        flux_stats = np.array(flux_values)
        
        print(f"  Complete ({step_time:.1f}s)")
        print(f"    Flux range: {flux_stats.min():.2e} to {flux_stats.max():.2e}")
        print(f"    Significant points: {significant_flux_points:,} ({significant_flux_points/total_points*100:.1f}%)")
    
    # Add time information to the multiblock dataset
    multiblock.GetFieldData().AddArray(time_values)
    
    # Write single VTK multiblock file
    filename = f"{output_dir}/time_dependent_flux.vtm"
    print(f"\nWriting multiblock file: {filename}")
    
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(multiblock)
    writer.Write()
    
    # Create time series metadata file
    create_time_series_metadata(output_dir, time_points)
    
    # Create sample orbital data that spans the time period
    create_storm_time_orbital_data(output_dir, time_start, time_end)
    
    total_time = time.time() - overall_start_time
    
    print("\n" + "=" * 60)
    print("Time-dependent flux multiblock dataset generated successfully!")
    print(f"  Total processing time: {total_time:.1f} seconds")
    print(f"  Average time per step: {total_time/n_timesteps:.1f} seconds")
    print(f"  Time steps: {n_timesteps}")
    print(f"  Output file: {filename}")
    
    print("\nStorm timeline modeled:")
    print("  t < 0h    : Pre-storm quiet conditions")
    print("  t = 0h    : CME arrival / Storm sudden commencement")
    print("  t = 0-2h  : Initial phase / Magnetosphere compression")
    print("  t = 2-8h  : Main phase / Maximum belt distortion")
    print("  t = 8-24h : Early recovery / Particle acceleration")
    print("  t > 24h   : Late recovery / Return to quiet levels")


def create_time_series_metadata(output_dir, time_points):
    """Create metadata file describing the time series"""
    
    metadata = {
        'description': 'Time-dependent Van Allen belt flux with CME effects (multiblock)',
        'file_type': 'vtk_multiblock',
        'time_start': float(time_points[0]),
        'time_end': float(time_points[-1]),
        'time_step': float(time_points[1] - time_points[0]),
        'n_timesteps': len(time_points),
        'storm_arrival': 0.0,
        'units': {
            'time': 'hours',
            'flux': 'particles/(cm²·s·sr·MeV)',
            'position': 'km'
        },
        'storm_phases': {
            'quiet': 't < 0h',
            'sudden_commencement': 't = 0h',
            'initial_phase': 't = 0-2h',
            'main_phase': 't = 2-8h',
            'early_recovery': 't = 8-24h',
            'late_recovery': 't > 24h'
        },
        'physical_effects': [
            'Van Allen belt compression',
            'Magnetopause boundary dynamics',
            'Ring current field modifications',
            'Substorm particle acceleration',
            'Plasma sheet particle injection',
            'Energy-dependent loss processes'
        ]
    }
    
    metadata_file = Path(output_dir) / "time_series_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved: {metadata_file}")


def create_storm_time_orbital_data(output_dir, time_start, time_end):
    """Create sample orbital data for a highly elliptical orbit during the storm period"""
    
    print("  Creating storm-time orbital data...")
    
    # HEO orbit parameters (similar to Molniya orbit)
    semi_major_axis = 26560  # km (12-hour orbit)
    eccentricity = 0.74
    inclination = 63.4  # degrees (critical inclination for frozen orbit)
    
    # Time points every hour
    time_hours = np.arange(time_start, time_end + 1, 1.0)
    n_points = len(time_hours)
    
    orbital_data = []
    
    for i, t in enumerate(time_hours):
        # Mean anomaly (orbit fraction)
        orbital_period = 12.0  # hours
        mean_anomaly = 2 * np.pi * (t % orbital_period) / orbital_period
        
        # Solve Kepler's equation (simplified)
        eccentric_anomaly = mean_anomaly + eccentricity * np.sin(mean_anomaly)
        
        # True anomaly
        true_anomaly = 2 * np.arctan2(
            np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
            np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2)
        )
        
        # Distance from Earth center
        r = semi_major_axis * (1 - eccentricity * np.cos(eccentric_anomaly))
        
        # Position in orbital plane
        x_orbit = r * np.cos(true_anomaly)
        y_orbit = r * np.sin(true_anomaly)
        z_orbit = 0
        
        # Rotate to inclined orbit
        inc_rad = np.radians(inclination)
        x = x_orbit
        y = y_orbit * np.cos(inc_rad) - z_orbit * np.sin(inc_rad)
        z = y_orbit * np.sin(inc_rad) + z_orbit * np.cos(inc_rad)
        
        # Create orbital point
        altitude = r - 6371  # km above Earth surface
        orbital_point = OrbitalPoint(x, y, z, t, altitude)
        orbital_data.append(orbital_point)
    
    # Write orbital data file
    orbital_file = Path(output_dir) / "storm_time_heo_orbit.orb"
    
    with open(orbital_file, 'w') as f:
        f.write("# Storm-time HEO orbital data\n")
        f.write("# Time (hours), X (km), Y (km), Z (km), Altitude (km)\n")
        
        for point in orbital_data:
            f.write(f"{point.time:.3f}, {point.x:.1f}, {point.y:.1f}, {point.z:.1f}, {point.altitude:.1f}\n")
    
    print(f"  Orbital data saved: {orbital_file} ({len(orbital_data)} points)")


if __name__ == "__main__":
    # Generate time-dependent flux with multiblock VTK format
    create_time_dependent_vtk_multiblock(
        output_dir="data/flux/time_dependent",
        time_start=-12,  # 12 hours before storm
        time_end=60,     # 60 hours after storm arrival
        time_step=3,     # 3-hour time steps (24 total steps)
        nx=40, ny=40, nz=40  # Moderate resolution for reasonable file size
    )
    
    print("\n" + "=" * 60)
    print("Time-dependent Van Allen belt simulation complete!")
    print("Load the multiblock file in STRATOS to visualize storm evolution")
    print("=" * 60)