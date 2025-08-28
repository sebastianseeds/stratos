"""
Orbital Data Structures for Flux Orbital Visualizer
Defines data classes for orbital trajectories and paths
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OrbitalPoint:
    """
    Data structure to hold a single orbital position and velocity.
    
    Attributes:
        time: Time in hours
        x, y, z: Position in km (Earth-centered coordinates)
        vx, vy, vz: Velocity in km/s (optional)
    """
    
    # Required fields
    time: float  # hours
    x: float     # km
    y: float     # km
    z: float     # km
    
    # Optional fields with defaults
    vx: float = 0.0  # km/s
    vy: float = 0.0  # km/s
    vz: float = 0.0  # km/s
    
    # Derived properties (computed post-init)
    phi: float = field(init=False)      # azimuthal angle (radians)
    r: float = field(init=False)        # distance from Earth center (km)
    altitude: float = field(init=False)  # altitude above Earth surface (km)
    
    # Constants
    EARTH_RADIUS_KM: float = field(default=6371.0, init=False, repr=False)
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        # Calculate azimuthal angle
        self.phi = np.arctan2(self.y, self.x)
        if self.phi < 0:
            self.phi += 2 * np.pi
        
        # Calculate radial distance from Earth center
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        # Calculate altitude above Earth surface
        self.altitude = self.r - self.EARTH_RADIUS_KM
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get position as tuple.
        
        Returns:
            (x, y, z) position in km
        """
        return (self.x, self.y, self.z)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get velocity as tuple.
        
        Returns:
            (vx, vy, vz) velocity in km/s
        """
        return (self.vx, self.vy, self.vz)
    
    def get_velocity_magnitude(self) -> float:
        """
        Get magnitude of velocity vector.
        
        Returns:
            Speed in km/s
        """
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def get_position_vector(self) -> np.ndarray:
        """
        Get position as numpy array.
        
        Returns:
            Position vector [x, y, z] in km
        """
        return np.array([self.x, self.y, self.z])
    
    def get_velocity_vector(self) -> np.ndarray:
        """
        Get velocity as numpy array.
        
        Returns:
            Velocity vector [vx, vy, vz] in km/s
        """
        return np.array([self.vx, self.vy, self.vz])
    
    def distance_to(self, other: 'OrbitalPoint') -> float:
        """
        Calculate distance to another orbital point.
        
        Args:
            other: Another OrbitalPoint
            
        Returns:
            Distance in km
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def is_in_eclipse(self) -> bool:
        """
        Simple check if satellite is in Earth's shadow (eclipse).
        This is a simplified check - assumes sun is at +X direction.
        
        Returns:
            True if point is likely in eclipse
        """
        # If x < 0 and within Earth's shadow cone
        if self.x < 0:
            shadow_radius = self.EARTH_RADIUS_KM * abs(self.x) / abs(self.x - 150e6)  # 150M km to sun
            distance_from_x_axis = np.sqrt(self.y**2 + self.z**2)
            return distance_from_x_axis < shadow_radius
        return False
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"OrbitalPoint(t={self.time:.2f}h, "
                f"pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f})km, "
                f"alt={self.altitude:.1f}km)")


class OrbitalPath:
    """
    Container for a complete orbital trajectory.
    Provides methods for trajectory analysis and manipulation.
    """
    
    def __init__(self, points: Optional[List[OrbitalPoint]] = None):
        """
        Initialize an orbital path.
        
        Args:
            points: List of OrbitalPoint objects (optional)
        """
        self.points: List[OrbitalPoint] = points if points is not None else []
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that points are in time order."""
        if len(self.points) > 1:
            times = [p.time for p in self.points]
            if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
                raise ValueError("Orbital points must be in chronological order")
    
    def add_point(self, point: OrbitalPoint) -> None:
        """
        Add a point to the trajectory.
        
        Args:
            point: OrbitalPoint to add
            
        Raises:
            ValueError: If point time is before the last point
        """
        if self.points and point.time < self.points[-1].time:
            raise ValueError(f"Point time {point.time} is before last point time {self.points[-1].time}")
        self.points.append(point)
    
    def get_point_at_time(self, time: float) -> Optional[OrbitalPoint]:
        """
        Get the orbital point at a specific time (interpolated if necessary).
        
        Args:
            time: Time in hours
            
        Returns:
            Interpolated OrbitalPoint at the specified time, or None if out of range
        """
        if not self.points:
            return None
        
        # Check bounds
        if time < self.points[0].time or time > self.points[-1].time:
            return None
        
        # Find surrounding points
        for i in range(len(self.points) - 1):
            if self.points[i].time <= time <= self.points[i+1].time:
                return self._interpolate_between_points(
                    self.points[i], 
                    self.points[i+1], 
                    time
                )
        
        return self.points[-1]  # Return last point if time matches exactly
    
    def get_point_at_index(self, index: int) -> Optional[OrbitalPoint]:
        """
        Get orbital point at specific index.
        
        Args:
            index: Index in the points list
            
        Returns:
            OrbitalPoint at index or None if out of range
        """
        if 0 <= index < len(self.points):
            return self.points[index]
        return None
    
    def _interpolate_between_points(self, 
                                   p1: OrbitalPoint, 
                                   p2: OrbitalPoint, 
                                   time: float) -> OrbitalPoint:
        """
        Linearly interpolate between two orbital points.
        
        Args:
            p1: First orbital point
            p2: Second orbital point
            time: Time to interpolate at
            
        Returns:
            Interpolated OrbitalPoint
        """
        # Calculate interpolation factor
        if p2.time == p1.time:
            t = 0.0
        else:
            t = (time - p1.time) / (p2.time - p1.time)
        
        # Interpolate position
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)
        z = p1.z + t * (p2.z - p1.z)
        
        # Interpolate velocity
        vx = p1.vx + t * (p2.vx - p1.vx)
        vy = p1.vy + t * (p2.vy - p1.vy)
        vz = p1.vz + t * (p2.vz - p1.vz)
        
        return OrbitalPoint(time=time, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    
    def get_total_duration(self) -> float:
        """
        Get total time duration of the trajectory.
        
        Returns:
            Duration in hours
        """
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].time - self.points[0].time
    
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get spatial bounds of the trajectory.
        
        Returns:
            (xmin, xmax, ymin, ymax, zmin, zmax) in km
        """
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    
    def get_altitude_range(self) -> Tuple[float, float]:
        """
        Get altitude range of the trajectory.
        
        Returns:
            (min_altitude, max_altitude) in km
        """
        if not self.points:
            return (0.0, 0.0)
        
        altitudes = [p.altitude for p in self.points]
        return (min(altitudes), max(altitudes))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate trajectory statistics.
        
        Returns:
            Dictionary containing trajectory statistics
        """
        if not self.points:
            return {}
        
        altitudes = [p.altitude for p in self.points]
        speeds = [p.get_velocity_magnitude() for p in self.points]
        
        # Check for complete orbit
        orbital_period = self._estimate_orbital_period()
        
        return {
            'num_points': len(self.points),
            'duration_hours': self.get_total_duration(),
            'start_time': self.points[0].time,
            'end_time': self.points[-1].time,
            'altitude_min_km': min(altitudes),
            'altitude_max_km': max(altitudes),
            'altitude_mean_km': np.mean(altitudes),
            'altitude_std_km': np.std(altitudes),
            'speed_min_km_s': min(speeds) if speeds[0] > 0 else None,
            'speed_max_km_s': max(speeds) if speeds[0] > 0 else None,
            'speed_mean_km_s': np.mean(speeds) if speeds[0] > 0 else None,
            'orbital_period_hours': orbital_period,
            'num_orbits': self.get_total_duration() / orbital_period if orbital_period else None
        }
    
    def _estimate_orbital_period(self) -> Optional[float]:
        """
        Estimate orbital period by looking for return to similar position.
        
        Returns:
            Estimated period in hours, or None if not detected
        """
        if len(self.points) < 10:
            return None
        
        start_pos = self.points[0].get_position_vector()
        threshold_km = 100.0  # Consider "same position" if within 100 km
        
        # Look for return to start position after at least half the trajectory
        min_index = len(self.points) // 2
        for i in range(min_index, len(self.points)):
            pos = self.points[i].get_position_vector()
            distance = np.linalg.norm(pos - start_pos)
            if distance < threshold_km:
                return self.points[i].time - self.points[0].time
        
        return None
    
    def find_apogee_perigee(self) -> Dict[str, Any]:
        """
        Find apogee (highest) and perigee (lowest) points.
        
        Returns:
            Dictionary with apogee and perigee information
        """
        if not self.points:
            return {}
        
        # Find highest and lowest altitude points
        apogee_point = max(self.points, key=lambda p: p.altitude)
        perigee_point = min(self.points, key=lambda p: p.altitude)
        
        return {
            'apogee': {
                'altitude_km': apogee_point.altitude,
                'time_hours': apogee_point.time,
                'position': apogee_point.get_position()
            },
            'perigee': {
                'altitude_km': perigee_point.altitude,
                'time_hours': perigee_point.time,
                'position': perigee_point.get_position()
            },
            'eccentricity_estimate': (apogee_point.r - perigee_point.r) / 
                                    (apogee_point.r + perigee_point.r)
        }
    
    def __len__(self) -> int:
        """Get number of points in the path."""
        return len(self.points)
    
    def __getitem__(self, index: int) -> OrbitalPoint:
        """Get point by index."""
        return self.points[index]
    
    def __iter__(self):
        """Iterate over points."""
        return iter(self.points)
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.points:
            return "OrbitalPath(empty)"
        return (f"OrbitalPath({len(self.points)} points, "
                f"duration={self.get_total_duration():.2f}h, "
                f"alt={self.get_altitude_range()[0]:.1f}-{self.get_altitude_range()[1]:.1f}km)")
