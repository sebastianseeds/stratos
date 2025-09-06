#!/usr/bin/env python3
"""
Orbital Data Generator for STRATOS
Creates orbital trajectory CSV files with configurable number of orbits and orbital parameters.
"""

import numpy as np
import pandas as pd
import argparse
import time
import os

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """Create terminal progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def create_orbital_data(filename, orbit_type="LEO", num_orbits=1, points_per_orbit=500):
    """
    Create orbital CSV data with configurable number of orbits
    
    Parameters:
    - filename: Output CSV filename
    - orbit_type: Type of orbit (LEO, MEO, GEO, HEO)
    - num_orbits: Number of complete orbits to generate
    - points_per_orbit: Number of points per orbit (resolution)
    """
    
    print(f"Creating orbital data: {filename}")
    print(f"  Orbit type: {orbit_type}")
    print(f"  Number of orbits: {num_orbits}")
    print(f"  Points per orbit: {points_per_orbit}")
    
    start_time = time.time()
    
    # Orbital parameters based on type
    if orbit_type == "LEO":
        altitude = 400  # km
        period = 1.5    # hours
        inclination = 51.6  # degrees (ISS-like)
        description = "Low Earth Orbit (ISS-like)"
    elif orbit_type == "GEO":
        altitude = 35786  # km
        period = 24.0   # hours  
        inclination = 0.0  # degrees (equatorial)
        description = "Geostationary Earth Orbit"
    elif orbit_type == "HEO":
        altitude = 39000  # km (apogee approximation)
        period = 12.0   # hours
        inclination = 63.4  # degrees (Molniya orbit)
        description = "Highly Elliptical Orbit (Molniya-like)"
    else:  # MEO
        altitude = 20200  # km (GPS-like)
        period = 12.0   # hours
        inclination = 55.0  # degrees (GPS constellation)
        description = "Medium Earth Orbit (GPS-like)"
    
    print(f"  Description: {description}")
    print(f"  Altitude: {altitude:,} km")
    print(f"  Period: {period} hours")
    print(f"  Inclination: {inclination}°")
    
    earth_radius = 6371  # km
    orbital_radius = earth_radius + altitude
    
    # Total simulation parameters
    total_points = num_orbits * points_per_orbit
    total_duration = num_orbits * period  # hours
    times = np.linspace(0, total_duration, total_points)
    
    print(f"  Total duration: {total_duration} hours")
    print(f"  Total points: {total_points:,}")
    print(f"  Time resolution: {total_duration*60/total_points:.2f} minutes per point")
    
    inclination_rad = np.radians(inclination)
    
    positions = []
    velocities = []
    
    print("  Computing orbital positions...")
    
    for idx, t in enumerate(times):
        # Mean anomaly (cycles through 2π every period)
        M = 2 * np.pi * t / period
        
        # Position in orbital plane (circular orbit approximation)
        x_orb = orbital_radius * np.cos(M)
        y_orb = orbital_radius * np.sin(M)
        z_orb = 0
        
        # Rotate by inclination (simple rotation about x-axis)
        x = x_orb
        y = y_orb * np.cos(inclination_rad) - z_orb * np.sin(inclination_rad)
        z = y_orb * np.sin(inclination_rad) + z_orb * np.cos(inclination_rad)
        
        positions.append([x, y, z])
        
        # Velocity computation
        orbital_speed = 2 * np.pi * orbital_radius / (period * 3600)  # km/s
        
        # Velocity in orbital plane
        vx_orb = -orbital_speed * np.sin(M)
        vy_orb = orbital_speed * np.cos(M)
        vz_orb = 0
        
        # Rotate velocity by inclination
        vx = vx_orb
        vy = vy_orb * np.cos(inclination_rad) - vz_orb * np.sin(inclination_rad)
        vz = vy_orb * np.sin(inclination_rad) + vz_orb * np.cos(inclination_rad)
        
        velocities.append([vx, vy, vz])
        
        # Progress indicator
        if (idx + 1) % (total_points // 20) == 0 or idx + 1 == total_points:
            print_progress_bar(idx + 1, total_points, prefix='    Progress')
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Create DataFrame with required columns
    df = pd.DataFrame({
        'time': times,
        'x': positions[:, 0],
        'y': positions[:, 1], 
        'z': positions[:, 2],
        'vx': velocities[:, 0],
        'vy': velocities[:, 1],
        'vz': velocities[:, 2]
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    elapsed_time = time.time() - start_time
    
    # Calculate orbital statistics
    altitudes = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2) - earth_radius
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2 + velocities[:, 2]**2)
    
    print(f"\n  ✓ Orbital data created! ({elapsed_time:.2f}s)")
    print(f"    File: {filename}")
    print(f"    Data points: {len(df):,}")
    print(f"    Time span: {times[0]:.1f} to {times[-1]:.1f} hours")
    print(f"    Altitude range: {altitudes.min():.0f} to {altitudes.max():.0f} km")
    print(f"    Speed range: {speeds.min():.2f} to {speeds.max():.2f} km/s")
    print(f"    Avg orbital speed: {speeds.mean():.2f} km/s")

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Generate orbital trajectory data for STRATOS visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --type LEO --orbits 3 --output leo_3orbits.csv
  %(prog)s --type GEO --orbits 1 --points 1000 --output geo_highres.csv
  %(prog)s --type HEO --orbits 2 --output heo_molniya.csv

Orbit Types:
  LEO  - Low Earth Orbit (400 km, 1.5h period, 51.6° inclination)
  MEO  - Medium Earth Orbit (20,200 km, 12h period, 55° inclination) 
  GEO  - Geostationary Earth Orbit (35,786 km, 24h period, 0° inclination)
  HEO  - Highly Elliptical Orbit (39,000 km, 12h period, 63.4° inclination)

Output format: CSV with columns [time, x, y, z, vx, vy, vz]
""")
    
    parser.add_argument('--type', '-t', 
                       choices=['LEO', 'MEO', 'GEO', 'HEO'], 
                       default='LEO',
                       help='Orbit type (default: LEO)')
    
    parser.add_argument('--orbits', '-n',
                       type=int, 
                       default=1,
                       help='Number of complete orbits (default: 1)')
    
    parser.add_argument('--points', '-p',
                       type=int, 
                       default=500,
                       help='Points per orbit for resolution (default: 500)')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       help='Output CSV filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Auto-generate filename if not provided
    if not args.output:
        args.output = f"../data/orbits/orbit_{args.type.lower()}_{args.orbits}orbits.csv"
    
    # Ensure the filename ends with .csv
    if not args.output.endswith('.csv'):
        args.output += '.csv'
    
    print("=" * 60)
    print("STRATOS Orbital Data Generator")
    print("=" * 60)
    
    # Generate the orbital data
    create_orbital_data(
        filename=args.output,
        orbit_type=args.type,
        num_orbits=args.orbits,
        points_per_orbit=args.points
    )
    
    print(f"\n✓ Generation complete!")
    print(f"Load '{args.output}' in STRATOS to visualize the orbital trajectory.")

if __name__ == "__main__":
    main()