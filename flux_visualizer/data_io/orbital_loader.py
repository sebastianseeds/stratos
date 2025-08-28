"""
Orbital Data Loading Module for Flux Orbital Visualizer
Handles loading spacecraft orbital data from CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from core.orbital_data import OrbitalPoint, OrbitalPath

class OrbitalDataLoader:
    """Handles loading and validation of orbital trajectory data from CSV files."""
    
    # Required columns for orbital data
    REQUIRED_COLUMNS = ['time', 'x', 'y', 'z']
    OPTIONAL_COLUMNS = ['vx', 'vy', 'vz']
    
    @classmethod
    def load_csv(cls, 
                 file_path: str, 
                 validate: bool = True) -> List[OrbitalPoint]:
        """
        Load orbital data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            validate: Whether to validate the data
            
        Returns:
            List of OrbitalPoint objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading orbital data from: {file_path.name}")
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate columns
        cls._validate_columns(df)
        
        # Create orbital points
        orbital_points = cls._create_orbital_points(df)
        
        # Validate data if requested
        if validate:
            cls._validate_orbital_data(orbital_points)
        
        print(f"Successfully loaded {len(orbital_points)} orbital points")
        print(f"Time span: {orbital_points[0].time:.2f} to {orbital_points[-1].time:.2f} hours")
        print(f"Altitude range: {min(p.altitude for p in orbital_points):.1f} to "
              f"{max(p.altitude for p in orbital_points):.1f} km")
        
        return orbital_points
    
    @classmethod
    def _validate_columns(cls, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in the dataframe.
        
        Args:
            df: Pandas dataframe to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {available_cols}\n"
                f"Required columns: {cls.REQUIRED_COLUMNS}"
            )
        
        print(f"Found required columns: {cls.REQUIRED_COLUMNS}")
        
        # Check for optional columns
        optional_present = [col for col in cls.OPTIONAL_COLUMNS if col in df.columns]
        if optional_present:
            print(f"Found optional columns: {optional_present}")
    
    @classmethod
    def _create_orbital_points(cls, df: pd.DataFrame) -> List[OrbitalPoint]:
        """
        Convert dataframe rows to OrbitalPoint objects.
        
        Args:
            df: Pandas dataframe with orbital data
            
        Returns:
            List of OrbitalPoint objects
        """
        orbital_points = []
        
        for _, row in df.iterrows():
            # Get required values
            time = row['time']
            x = row['x']
            y = row['y']
            z = row['z']
            
            # Get optional velocity values
            vx = row.get('vx', 0)
            vy = row.get('vy', 0)
            vz = row.get('vz', 0)
            
            point = OrbitalPoint(
                time=time, x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz
            )
            orbital_points.append(point)
        
        return orbital_points
    
    @classmethod
    def _validate_orbital_data(cls, orbital_points: List[OrbitalPoint]) -> None:
        """
        Validate the orbital data for physical reasonableness.
        
        Args:
            orbital_points: List of orbital points to validate
            
        Raises:
            ValueError: If data appears invalid
        """
        if not orbital_points:
            raise ValueError("No orbital points created")
        
        # Check for reasonable altitudes (not inside Earth, not too far)
        min_altitude = min(p.altitude for p in orbital_points)
        max_altitude = max(p.altitude for p in orbital_points)
        
        if min_altitude < -100:  # Allow small negative for numerical errors
            raise ValueError(f"Orbital path goes inside Earth (min altitude: {min_altitude:.1f} km)")
        
        if max_altitude > 100000:  # 100,000 km is very far
            print(f"Warning: Very high altitude detected ({max_altitude:.1f} km)")
        
        # Check for time monotonicity
        times = [p.time for p in orbital_points]
        if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
            raise ValueError("Time values are not monotonically increasing")
        
        # Check for duplicate times
        if len(set(times)) != len(times):
            print("Warning: Duplicate time values detected")
        
        # Check for reasonable velocities (if provided)
        if orbital_points[0].vx != 0 or orbital_points[0].vy != 0 or orbital_points[0].vz != 0:
            max_velocity = max(
                np.sqrt(p.vx**2 + p.vy**2 + p.vz**2) 
                for p in orbital_points
            )
            if max_velocity > 20:  # km/s - escape velocity is ~11 km/s
                print(f"Warning: High velocity detected ({max_velocity:.1f} km/s)")
    
    @classmethod
    def get_trajectory_info(cls, orbital_points: List[OrbitalPoint]) -> Dict[str, Any]:
        """
        Get summary information about the trajectory.
        
        Args:
            orbital_points: List of orbital points
            
        Returns:
            Dictionary containing trajectory information
        """
        if not orbital_points:
            return {}
        
        altitudes = [p.altitude for p in orbital_points]
        times = [p.time for p in orbital_points]
        
        # Calculate orbital period (if complete orbit)
        orbital_period = None
        if len(orbital_points) > 10:
            # Simple check: see if we return close to starting position
            start_pos = np.array([orbital_points[0].x, orbital_points[0].y, orbital_points[0].z])
            for i, p in enumerate(orbital_points[1:], 1):
                pos = np.array([p.x, p.y, p.z])
                distance = np.linalg.norm(pos - start_pos)
                if distance < 100 and i > len(orbital_points) // 2:  # Within 100km after halfway
                    orbital_period = p.time - orbital_points[0].time
                    break
        
        return {
            'num_points': len(orbital_points),
            'time_span': times[-1] - times[0],
            'time_start': times[0],
            'time_end': times[-1],
            'altitude_min': min(altitudes),
            'altitude_max': max(altitudes),
            'altitude_mean': np.mean(altitudes),
            'orbital_period': orbital_period,
            'has_velocity': orbital_points[0].vx != 0 or orbital_points[0].vy != 0 or orbital_points[0].vz != 0
        }
    
    @classmethod
    def interpolate_trajectory(cls, 
                              orbital_points: List[OrbitalPoint], 
                              num_points: int) -> List[OrbitalPoint]:
        """
        Interpolate trajectory to a different number of points.
        
        Args:
            orbital_points: Original orbital points
            num_points: Desired number of points
            
        Returns:
            Interpolated list of orbital points
        """
        if len(orbital_points) == num_points:
            return orbital_points
        
        # Extract arrays for interpolation
        times = np.array([p.time for p in orbital_points])
        xs = np.array([p.x for p in orbital_points])
        ys = np.array([p.y for p in orbital_points])
        zs = np.array([p.z for p in orbital_points])
        
        # Create new time array
        new_times = np.linspace(times[0], times[-1], num_points)
        
        # Interpolate positions
        new_xs = np.interp(new_times, times, xs)
        new_ys = np.interp(new_times, times, ys)
        new_zs = np.interp(new_times, times, zs)
        
        # Interpolate velocities if available
        if orbital_points[0].vx != 0 or orbital_points[0].vy != 0 or orbital_points[0].vz != 0:
            vxs = np.array([p.vx for p in orbital_points])
            vys = np.array([p.vy for p in orbital_points])
            vzs = np.array([p.vz for p in orbital_points])
            
            new_vxs = np.interp(new_times, times, vxs)
            new_vys = np.interp(new_times, times, vys)
            new_vzs = np.interp(new_times, times, vzs)
        else:
            new_vxs = new_vys = new_vzs = np.zeros(num_points)
        
        # Create new orbital points
        interpolated_points = []
        for i in range(num_points):
            point = OrbitalPoint(
                time=new_times[i],
                x=new_xs[i], y=new_ys[i], z=new_zs[i],
                vx=new_vxs[i], vy=new_vys[i], vz=new_vzs[i]
            )
            interpolated_points.append(point)
        
        return interpolated_points
    
    @classmethod
    def export_to_csv(cls, 
                     orbital_points: List[OrbitalPoint], 
                     output_path: str) -> None:
        """
        Export orbital points back to CSV format.
        
        Args:
            orbital_points: List of orbital points to export
            output_path: Path for output CSV file
        """
        data = {
            'time': [p.time for p in orbital_points],
            'x': [p.x for p in orbital_points],
            'y': [p.y for p in orbital_points],
            'z': [p.z for p in orbital_points],
        }
        
        # Add velocities if available
        if orbital_points[0].vx != 0 or orbital_points[0].vy != 0 or orbital_points[0].vz != 0:
            data['vx'] = [p.vx for p in orbital_points]
            data['vy'] = [p.vy for p in orbital_points]
            data['vz'] = [p.vz for p in orbital_points]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(orbital_points)} points to {output_path}")
