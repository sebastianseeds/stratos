#!/usr/bin/env python3
"""
Animation Debug Tool
Quick test to verify orbital data and check animation timing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_orbital_file(filename):
    """Test an orbital CSV file and show key statistics"""
    print(f"\nTesting orbital file: {filename}")
    print("-" * 50)
    
    if not Path(filename).exists():
        print(f"❌ File not found: {filename}")
        return False
        
    try:
        # Load the data
        df = pd.read_csv(filename)
        print(f"File loaded successfully")
        
        # Check columns
        required_cols = ['time', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return False
            
        print(f"Required columns present: {required_cols}")
        
        # Check optional columns
        optional_cols = ['vx', 'vy', 'vz']
        present_optional = [col for col in optional_cols if col in df.columns]
        if present_optional:
            print(f"Optional columns present: {present_optional}")
            
        # Basic statistics
        n_points = len(df)
        time_span = df['time'].max() - df['time'].min()
        dt_mean = df['time'].diff().mean()
        
        print(f"Data points: {n_points}")
        print(f"Time span: {time_span:.2f} hours")
        print(f"Average time step: {dt_mean:.4f} hours ({dt_mean*3600:.1f} seconds)")
        
        # Position statistics
        positions = df[['x', 'y', 'z']].values
        distances = np.sqrt(np.sum(positions**2, axis=1))
        
        print(f"Distance from origin:")
        print(f"   Min: {distances.min():.1f} km")
        print(f"   Max: {distances.max():.1f} km") 
        print(f"   Mean: {distances.mean():.1f} km")
        
        # Earth radius check
        earth_radius = 6371  # km
        altitude_min = distances.min() - earth_radius
        altitude_max = distances.max() - earth_radius
        
        print(f"Altitude above Earth:")
        print(f"   Min: {altitude_min:.1f} km")
        print(f"   Max: {altitude_max:.1f} km")
        
        if altitude_min < 0:
            print("Warning: Orbit goes below Earth surface!")
            
        # Orbital period estimation
        if n_points > 10:
            # Find when satellite returns close to starting position
            start_pos = positions[0]
            distances_from_start = np.sqrt(np.sum((positions - start_pos)**2, axis=1))
            
            # Look for minimum after at least 10% of the orbit
            min_idx = np.argmin(distances_from_start[n_points//10:]) + n_points//10
            estimated_period = df['time'].iloc[min_idx] - df['time'].iloc[0]
            
            print(f"Estimated orbital period: {estimated_period:.2f} hours")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

def create_animation_timing_plot():
    """Create a plot showing animation timing for different orbit types"""
    
    files = [
        ("orbit_leo.csv", "LEO", "blue"),
        ("orbit_meo.csv", "MEO", "green"), 
        ("orbit_geo.csv", "GEO", "red")
    ]
    
    plt.figure(figsize=(12, 8))
    
    valid_files = []
    
    for filename, label, color in files:
        if Path(filename).exists():
            try:
                df = pd.read_csv(filename)
                
                # Calculate distance from Earth center
                positions = df[['x', 'y', 'z']].values
                distances = np.sqrt(np.sum(positions**2, axis=1))
                altitudes = distances - 6371  # Earth radius
                
                # Plot altitude vs time
                plt.subplot(2, 2, 1)
                plt.plot(df['time'], altitudes, label=f"{label} ({len(df)} pts)", color=color, linewidth=2)
                plt.xlabel('Time (hours)')
                plt.ylabel('Altitude (km)')
                plt.title('Altitude vs Time')
                plt.legend()
                plt.grid(True)
                
                # Plot 3D trajectory
                plt.subplot(2, 2, 2)
                plt.plot(df['x'], df['y'], label=label, color=color, alpha=0.7)
                plt.xlabel('X (km)')
                plt.ylabel('Y (km)')
                plt.title('Orbital Paths (X-Y plane)')
                plt.axis('equal')
                plt.legend()
                plt.grid(True)
                
                # Add Earth circle
                earth_circle = plt.Circle((0, 0), 6371, fill=False, color='black', linestyle='--')
                plt.gca().add_patch(earth_circle)
                
                # Animation timing analysis
                plt.subplot(2, 2, 3)
                time_diffs = df['time'].diff()[1:]  # Skip first NaN
                plt.plot(df['time'][1:], time_diffs * 3600, label=f"{label} Δt", color=color)
                plt.xlabel('Time (hours)')
                plt.ylabel('Time Step (seconds)')
                plt.title('Animation Time Steps')
                plt.legend()
                plt.grid(True)
                
                valid_files.append((filename, label, len(df), df['time'].max()))
                
            except Exception as e:
                print(f"Error plotting {filename}: {e}")
    
    # Summary table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    if valid_files:
        table_data = []
        for filename, label, n_points, duration in valid_files:
            table_data.append([label, f"{n_points}", f"{duration:.1f}h"])
            
        table = plt.table(cellText=table_data,
                         colLabels=['Orbit', 'Points', 'Duration'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Orbit Summary')
    
    plt.tight_layout()
    plt.savefig('orbital_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nOrbital analysis plot saved as 'orbital_analysis.png'")
    plt.show()

def test_animation_speed():
    """Test what animation speeds work well for different orbits"""
    
    print("\nAnimation Speed Recommendations:")
    print("-" * 40)
    
    files = [
        ("orbit_leo.csv", "LEO"),
        ("orbit_meo.csv", "MEO"),
        ("orbit_geo.csv", "GEO")
    ]
    
    for filename, orbit_type in files:
        if Path(filename).exists():
            try:
                df = pd.read_csv(filename)
                n_points = len(df)
                duration_hours = df['time'].max() - df['time'].min()
                
                # Calculate recommended speeds for different playback rates
                print(f"\n{orbit_type} Orbit ({filename}):")
                print(f"  Real duration: {duration_hours:.1f} hours")
                print(f"  Data points: {n_points}")
                
                # Recommended animation speeds for different playback times
                target_times = [10, 30, 60, 120]  # seconds
                
                for target_sec in target_times:
                    speed_ms = (target_sec * 1000) // n_points
                    real_time_factor = (duration_hours * 3600) / target_sec
                    
                    print(f"  {target_sec:3d}s playback: {speed_ms:3d}ms/step ({real_time_factor:.0f}× speed)")
                    
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")

def main():
    """Run all animation debugging tests"""
    print("Electron Flux Visualizer - Animation Debug Tool")
    print("=" * 60)
    
    # Test individual files
    test_files = [
        "orbit_leo.csv",
        "orbit_meo.csv", 
        "orbit_geo.csv"
    ]
    
    all_good = True
    for filename in test_files:
        if not test_orbital_file(filename):
            all_good = False
    
    if all_good:
        print(f"\nAll orbital files are valid!")
        
        # Create analysis plots
        try:
            create_animation_timing_plot()
        except Exception as e:
            print(f"Plot creation failed: {e}")
            
        # Animation speed recommendations
        test_animation_speed()
        
        print(f"\nAnimation Troubleshooting Checklist:")
        print(f"   1. Orbital data files are valid")
        print(f"   2. Check if satellite object is visible (try larger cross-section)")
        print(f"   3. Check animation speed (try 100-500ms for LEO)")
        print(f"   4. Check if VTK data is loaded properly")
        print(f"   5. Try resetting camera view (reset view in VTK window)")
        
    else:
        print(f"\nSome orbital files have issues. Run the data generator first:")
        print(f"   python generate_sample_data_fixed.py")

if __name__ == "__main__":
    main()
