#!/usr/bin/env python3
"""
Time-Dependent VTK Flux Field Generator for STRATOS
Creates time-varying Van Allen belt flux fields with coronal mass ejection (CME) effects.
Includes realistic belt compression, enhancement, particle injection, and recovery dynamics.
"""

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import time
import sys
import os
from pathlib import Path

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """Create terminal progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def dst_index_model(time_hours):
    """
    Model Dst index (geomagnetic storm intensity) with CME arrival and recovery.
    
    Args:
        time_hours: Time in hours (0 = CME arrival)
        
    Returns:
        Dst index in nT (negative values indicate storm intensity)
    """
    if time_hours < 0:
        # Pre-storm quiet conditions
        return np.random.normal(-10, 5)  # Quiet time Dst ~ -10 ± 5 nT
    
    elif 0 <= time_hours < 2:
        # Storm sudden commencement and initial phase
        # Rapid drop from -10 nT to storm minimum
        storm_minimum = -150  # nT (moderate to strong storm)
        return -10 + (storm_minimum + 10) * (time_hours / 2)**0.5
    
    elif 2 <= time_hours < 8:
        # Main phase - Dst at minimum with fluctuations
        storm_minimum = -150
        fluctuation = 20 * np.sin(2 * np.pi * time_hours / 3) * np.exp(-(time_hours-2)/4)
        return storm_minimum + fluctuation
    
    else:
        # Recovery phase - exponential recovery to quiet levels
        recovery_time_constant = 12  # hours
        initial_recovery = -150
        quiet_level = -10
        recovery_factor = np.exp(-(time_hours - 8) / recovery_time_constant)
        return quiet_level + (initial_recovery - quiet_level) * recovery_factor

def cme_flux_model(x, y, z, time_hours):
    """
    Time-dependent Van Allen belt model with CME effects.
    
    Args:
        x, y, z: Position in km (Earth-centered coordinates)
        time_hours: Time in hours (0 = CME arrival)
        
    Returns:
        Electron flux in particles/(cm²·s·sr·MeV)
    """
    earth_radius = 6371.0  # km
    
    # Magnetic dipole tilt (11 degrees from Z-axis toward positive X)
    magnetic_tilt = np.radians(11.0)
    
    # Rotate coordinates to magnetic coordinate system
    cos_tilt = np.cos(magnetic_tilt)
    sin_tilt = np.sin(magnetic_tilt)
    
    # Transform to magnetic coordinates
    x_mag = x * cos_tilt + z * sin_tilt
    y_mag = y  # No change in Y
    z_mag = -x * sin_tilt + z * cos_tilt
    
    # Distance from Earth center in magnetic coordinates
    r_mag = np.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
    
    # Magnetic latitude (angle from magnetic equator)
    if r_mag <= earth_radius:
        return 0.0  # Inside Earth
    
    magnetic_latitude = np.arcsin(np.abs(z_mag) / r_mag)
    
    # L-shell parameter (distance to equatorial crossing)
    L_shell = r_mag / (earth_radius * np.cos(magnetic_latitude)**2)
    
    # Get storm effects
    dst = dst_index_model(time_hours)
    
    # CME-induced modifications
    # 1. Magnetopause compression (Shue et al. model)
    solar_wind_pressure_enhancement = 1.0
    if 0 <= time_hours <= 12:
        # Enhanced solar wind pressure during storm
        solar_wind_pressure_enhancement = 2.0 + 1.5 * np.exp(-time_hours / 4)
    
    # Magnetopause compression affects outer boundary
    magnetopause_distance = 10.0 / (solar_wind_pressure_enhancement**0.17)  # In Earth radii
    
    # 2. Ring current enhancement (affects L-shell structure)
    ring_current_effect = 1.0
    if time_hours >= 0:
        # Ring current builds up during storm, affects field configuration
        ring_current_intensity = abs(dst) / 150.0  # Normalize to our storm strength
        ring_current_effect = 1.0 + 0.3 * ring_current_intensity * np.exp(-time_hours / 10)
    
    # 3. Belt compression and enhancement factors
    compression_factor = 1.0
    enhancement_factor = 1.0
    
    if 0 <= time_hours <= 6:
        # Compression phase - belts move inward
        compression_strength = abs(dst) / 150.0
        compression_factor = 1.0 - 0.3 * compression_strength
        # Enhancement due to particle acceleration
        enhancement_factor = 1.0 + 2.0 * compression_strength * np.exp(-time_hours / 3)
    
    elif 6 < time_hours <= 24:
        # Recovery phase - gradual outward expansion
        recovery_progress = (time_hours - 6) / 18.0
        compression_factor = 0.7 + 0.3 * recovery_progress
        enhancement_factor = 1.0 + 1.5 * np.exp(-(time_hours - 6) / 8)
    
    # Apply compression to effective L-shell
    L_effective = L_shell * compression_factor
    
    flux = 0.0
    
    # Inner Van Allen Belt (compressed and enhanced during storm)
    if 1.2 <= L_effective <= 2.8:
        inner_peak = 1.6 * compression_factor
        inner_width = 0.4 * compression_factor
        inner_amplitude = 2e7 * enhancement_factor
        
        # Additional storm-time enhancement for inner belt
        if 0 <= time_hours <= 8:
            storm_enhancement = 1.0 + 0.5 * abs(dst) / 150.0
            inner_amplitude *= storm_enhancement
        
        inner_flux = inner_amplitude * np.exp(-((L_effective - inner_peak)**2) / (2 * inner_width**2))
        lat_factor = 1.0 + 2.0 * np.sin(magnetic_latitude)**2
        flux += inner_flux * lat_factor
    
    # Slot Region (fills in during storms)
    slot_flux = 1e5
    if 2.5 <= L_effective <= 3.2:
        # Storm-time slot filling
        if 0 <= time_hours <= 12:
            slot_enhancement = 1.0 + 5.0 * abs(dst) / 150.0 * np.exp(-time_hours / 6)
            slot_flux *= slot_enhancement
        
        slot_flux *= np.exp(-((L_effective - 2.8)**2) / (2 * 0.2**2))
        flux += slot_flux
    
    # Outer Van Allen Belt (most affected by storms)
    if 3.0 <= L_effective <= 8.0 and L_shell < magnetopause_distance:
        outer_peak = 4.5 * compression_factor
        outer_width = 1.2 * compression_factor
        outer_amplitude = 1e7
        
        # Major storm-time modifications for outer belt
        if time_hours >= 0:
            # Immediate loss due to magnetopause compression
            if 0 <= time_hours <= 2:
                loss_factor = 0.1 + 0.9 * np.exp(-3 * time_hours)  # Rapid initial loss
                outer_amplitude *= loss_factor
            
            # Subsequent acceleration and rebuilding
            elif 2 < time_hours <= 24:
                # Gradual acceleration and flux enhancement
                acceleration_factor = 1.0 + 3.0 * (abs(dst) / 150.0) * np.exp(-(time_hours - 2) / 8)
                outer_amplitude *= acceleration_factor
            
            # Recovery phase
            elif time_hours > 24:
                # Slow decay back to pre-storm levels
                decay_factor = 1.0 + 2.0 * np.exp(-(time_hours - 24) / 48)
                outer_amplitude *= decay_factor
        
        outer_flux = outer_amplitude * np.exp(-((L_effective - outer_peak)**2) / (2 * outer_width**2))
        
        # Enhanced latitudinal dependence during storms
        lat_factor = 1.0 + 1.5 * np.sin(magnetic_latitude)**2
        if time_hours >= 0:
            lat_factor *= (1.0 + 0.5 * abs(dst) / 150.0)
        
        # Local time asymmetry (enhanced during storms)
        local_time_angle = np.arctan2(y_mag, x_mag)
        day_night_factor = 1.0 + 0.3 * np.cos(local_time_angle)
        if 0 <= time_hours <= 12:
            # Enhanced day-night asymmetry during active period
            day_night_factor *= (1.0 + 0.5 * abs(dst) / 150.0)
        
        flux += outer_flux * lat_factor * day_night_factor * ring_current_effect
    
    # Plasma sheet injection (temporary enhancement at high L-shells)
    if L_shell > 6.0 and time_hours >= 0:
        if 0 <= time_hours <= 8:
            # Strong plasma sheet injection during main phase
            injection_strength = 2e6 * abs(dst) / 150.0 * np.exp(-time_hours / 4)
            plasma_sheet_flux = injection_strength * np.exp(-(L_shell - 8.0)**2 / (2 * 2.0**2))
            flux += plasma_sheet_flux
    
    # Add realistic noise
    if flux > 0:
        # Time-dependent noise level (higher during storms)
        noise_level = 0.1
        if time_hours >= 0:
            noise_level += 0.2 * abs(dst) / 150.0
        
        noise_factor = 1.0 + noise_level * np.random.normal(0, 1)
        flux *= max(0.1, noise_factor)
        
        # Background level (varies with storm intensity)
        background = np.random.uniform(0, 1e4)
        if time_hours >= 0:
            background *= (1.0 + abs(dst) / 150.0)
        flux += background
    
    return max(0.0, flux)

def create_time_dependent_vtk_series(
    output_dir="time_dependent_flux",
    time_start=-12,    # Start 12 hours before CME
    time_end=72,       # End 72 hours after CME
    time_step=2,       # 2-hour time steps
    nx=50, ny=50, nz=50  # Grid resolution
):
    """
    Create a series of VTK files representing time-dependent flux evolution.
    
    Args:
        output_dir: Directory to store VTK files
        time_start: Start time in hours (relative to CME arrival)
        time_end: End time in hours
        time_step: Time step in hours
        nx, ny, nz: Grid resolution
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Creating time-dependent Van Allen belt flux series")
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
    
    # Create VTK files for each time step
    overall_start_time = time.time()
    
    for t_idx, current_time in enumerate(time_points):
        step_start_time = time.time()
        
        print(f"\nTime step {t_idx+1}/{n_timesteps}: t = {current_time:+.1f} hours")
        dst_value = dst_index_model(current_time)
        print(f"  Dst index: {dst_value:.1f} nT")
        
        # Create structured grid
        sgrid = vtk.vtkStructuredGrid()
        sgrid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        flux_values = []
        
        print("  Computing flux field...")
        point_count = 0
        significant_flux_points = 0
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    xi, yj, zk = x[i], y[j], z[k]
                    points.InsertNextPoint(xi, yj, zk)
                    
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
        
        sgrid.SetPoints(points)
        
        # Add flux data
        flux_array = numpy_to_vtk(np.array(flux_values), deep=True)
        flux_array.SetName("electron_flux")
        sgrid.GetPointData().SetScalars(flux_array)
        
        # Add time as field data
        time_array = vtk.vtkDoubleArray()
        time_array.SetName("TIME")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, current_time)
        sgrid.GetFieldData().AddArray(time_array)
        
        # Write VTK file
        filename = f"{output_dir}/flux_t{current_time:+06.1f}h.vts"
        print(f"  Writing: {filename}")
        
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(sgrid)
        writer.Write()
        
        step_time = time.time() - step_start_time
        flux_stats = np.array(flux_values)
        
        print(f"  Complete ({step_time:.1f}s)")
        print(f"    Flux range: {flux_stats.min():.2e} to {flux_stats.max():.2e}")
        print(f"    Significant points: {significant_flux_points:,} ({significant_flux_points/total_points*100:.1f}%)")
    
    # Create time series metadata file
    create_time_series_metadata(output_dir, time_points)
    
    # Create sample orbital data that spans the time period
    create_storm_time_orbital_data(output_dir, time_start, time_end)
    
    total_time = time.time() - overall_start_time
    
    print("\n" + "=" * 60)
    print("Time-dependent flux series generated successfully!")
    print(f"  Total processing time: {total_time:.1f} seconds")
    print(f"  Average time per step: {total_time/n_timesteps:.1f} seconds")
    print(f"  Files created: {n_timesteps}")
    print(f"  Output directory: {output_dir}")
    
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
        'description': 'Time-dependent Van Allen belt flux with CME effects',
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
            'initial_phase': '0h < t < 2h',
            'main_phase': '2h < t < 8h',
            'recovery_phase': 't > 8h'
        }
    }
    
    import json
    with open(f"{output_dir}/time_series_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create file list for easy loading
    file_list = []
    for t in time_points:
        filename = f"flux_t{t:+06.1f}h.vts"
        file_list.append({
            'time': float(t),
            'filename': filename,
            'dst_index': float(dst_index_model(t))
        })
    
    with open(f"{output_dir}/file_list.json", 'w') as f:
        json.dump(file_list, f, indent=2)

def create_storm_time_orbital_data(output_dir, time_start, time_end):
    """Create orbital data that spans the storm period"""
    
    print(f"\nGenerating storm-time orbital data...")
    
    # HEO orbit that passes through both belts during storm
    earth_radius = 6371  # km
    perigee = 1000  # km altitude
    apogee = 40000  # km altitude
    
    perigee_radius = earth_radius + perigee
    apogee_radius = earth_radius + apogee
    
    # Semi-major axis
    a = (perigee_radius + apogee_radius) / 2
    
    # Eccentricity
    e = (apogee_radius - perigee_radius) / (apogee_radius + perigee_radius)
    
    # Orbital period (hours)
    period = 2 * np.pi * np.sqrt(a**3 / (3.986e5)) / 3600  # hours
    
    print(f"  Orbit: {perigee} × {apogee} km")
    print(f"  Period: {period:.1f} hours")
    print(f"  Eccentricity: {e:.3f}")
    
    # Time points (higher resolution for storm period)
    time_hours = np.arange(time_start, time_end + 0.5, 0.5)  # 30-minute steps
    n_points = len(time_hours)
    
    positions = []
    velocities = []
    
    inclination = np.radians(63.4)  # Molniya-type orbit
    
    for t in time_hours:
        # Mean anomaly
        M = 2 * np.pi * (t - time_start) / period
        
        # Solve Kepler's equation (simplified)
        E = M  # Starting guess
        for _ in range(10):  # Newton-Raphson iterations
            E = M + e * np.sin(E)
        
        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # Distance from Earth center
        r = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        # Rotate by inclination
        x = x_orb
        y = y_orb * np.cos(inclination) - z_orb * np.sin(inclination)
        z = y_orb * np.sin(inclination) + z_orb * np.cos(inclination)
        
        positions.append([x, y, z])
        
        # Velocity (simplified)
        v_mag = np.sqrt(3.986e5 * (2/r - 1/a))  # km/s
        
        # Velocity direction (tangent to orbit)
        vx_orb = -v_mag * np.sin(nu)
        vy_orb = v_mag * (np.cos(nu) + e)
        vz_orb = 0
        
        # Normalize
        v_norm = np.sqrt(vx_orb**2 + vy_orb**2)
        vx_orb = vx_orb * v_mag / v_norm
        vy_orb = vy_orb * v_mag / v_norm
        
        vx = vx_orb
        vy = vy_orb * np.cos(inclination) - vz_orb * np.sin(inclination)
        vz = vy_orb * np.sin(inclination) + vz_orb * np.cos(inclination)
        
        velocities.append([vx, vy, vz])
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_hours,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'vx': velocities[:, 0],
        'vy': velocities[:, 1],
        'vz': velocities[:, 2]
    })
    
    filename = f"{output_dir}/orbit_storm_heo.csv"
    df.to_csv(filename, index=False)
    
    print(f"  Storm-time orbital data: {filename}")
    print(f"    Points: {n_points}, Duration: {time_end - time_start:.1f}h")
    
    # Calculate belt passage times
    belt_passages = []
    for i, (t, pos) in enumerate(zip(time_hours, positions)):
        r = np.linalg.norm(pos)
        L_shell = r / 6371.0  # Approximate L-shell
        
        if 1.5 < L_shell < 2.5:  # Inner belt
            belt_passages.append(f"t={t:+.1f}h: Inner belt (L≈{L_shell:.1f})")
        elif 3.0 < L_shell < 6.0:  # Outer belt
            belt_passages.append(f"t={t:+.1f}h: Outer belt (L≈{L_shell:.1f})")
    
    if belt_passages:
        print("  Belt passages during storm:")
        for passage in belt_passages[:10]:  # Show first 10
            print(f"    {passage}")
        if len(belt_passages) > 10:
            print(f"    ... and {len(belt_passages)-10} more")

def main():
    """Generate time-dependent Van Allen belt flux series with CME effects"""
    
    print("STRATOS Time-Dependent Flux Generator")
    print("Coronal Mass Ejection (CME) Van Allen Belt Simulation")
    print("=" * 60)
    print("Features:")
    print("  • Realistic storm timeline (quiet → CME → recovery)")
    print("  • Belt compression and enhancement")
    print("  • Magnetopause boundary effects")
    print("  • Ring current field modifications") 
    print("  • Plasma sheet particle injection")
    print("  • Time-dependent noise and variability")
    print("=" * 60)
    
    # Create time-dependent flux series
    create_time_dependent_vtk_series(
        output_dir="time_dependent_flux",
        time_start=-12,    # 12 hours before storm
        time_end=72,       # 72 hours after storm arrival
        time_step=2,       # 2-hour time steps  
        nx=45, ny=45, nz=45  # Moderate resolution for manageable file sizes
    )
    
    print("\n" + "=" * 60)
    print("Time-dependent flux field generation complete!")
    print("\nTo use in STRATOS:")
    print("  1. Load any VTK file from time_dependent_flux/ directory")
    print("  2. Load orbit_storm_heo.csv for HEO satellite trajectory")
    print("  3. Use animation controls to see storm evolution")
    print("  4. Try different time steps to observe:")
    print("     - Belt compression during storm main phase")
    print("     - Flux enhancement and particle acceleration")  
    print("     - Gradual recovery to quiet conditions")
    print("\nRecommended viewing sequence:")
    print("  • flux_t-012.0h.vts  (quiet conditions)")
    print("  • flux_t+000.0h.vts  (CME arrival)")
    print("  • flux_t+004.0h.vts  (peak compression)")
    print("  • flux_t+012.0h.vts  (recovery begins)")
    print("  • flux_t+048.0h.vts  (late recovery)")

if __name__ == '__main__':
    main()