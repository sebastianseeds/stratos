# analysis/flux_analyzer.py
"""
Flux Analysis Module
Handles flux calculations at orbital points and cross-sections
"""

import vtk
import numpy as np
from typing import Optional, List, Dict, Any
from core import OrbitalPoint


class FluxAnalyzer:
    """Handles flux analysis calculations"""
    
    def __init__(self):
        self.vtk_data: Optional[vtk.vtkDataObject] = None
        self.orbital_path: List[OrbitalPoint] = []
        self.cross_section_radius: float = 1.0  # meters
        
        # Cache for performance
        self._flux_cache: Dict[int, float] = {}
        
    def set_vtk_data(self, vtk_data: vtk.vtkDataObject) -> None:
        """Set the VTK dataset containing flux field"""
        self.vtk_data = vtk_data
        self._flux_cache.clear()
        
    def set_orbital_data(self, orbital_points: List[OrbitalPoint]) -> None:
        """Set the orbital path data"""
        self.orbital_path = orbital_points
        self._flux_cache.clear()
        
    def set_cross_section(self, radius_meters: float) -> None:
        """Set object cross-sectional radius in meters"""
        self.cross_section_radius = radius_meters
        self._flux_cache.clear()
        
    def analyze_flux_at_point(self, orbital_point: OrbitalPoint) -> float:
        """
        Analyze flux at a specific orbital point.
        
        Args:
            orbital_point: Point to analyze
            
        Returns:
            Integrated flux through cross-sectional area (particles/s)
        """
        if not self.vtk_data:
            return 0.0
        
        # Check cache first
        cache_key = hash((orbital_point.time, orbital_point.x, orbital_point.y, orbital_point.z))
        if cache_key in self._flux_cache:
            return self._flux_cache[cache_key]
            
        # Create a probe to sample the field at this point
        probe = vtk.vtkProbeFilter()
        
        # Create a point to probe
        points = vtk.vtkPoints()
        points.InsertNextPoint(orbital_point.x, orbital_point.y, orbital_point.z)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        probe.SetInputData(polydata)
        probe.SetSourceData(self.vtk_data)
        probe.Update()
        
        # Get the flux value
        result = probe.GetOutput()
        flux_value = 0.0
        
        if result.GetNumberOfPoints() > 0:
            scalar_array = result.GetPointData().GetScalars()
            if scalar_array and scalar_array.GetNumberOfTuples() > 0:
                flux_density = scalar_array.GetValue(0)  # particles/cm²/s
                
                # Calculate integrated flux through cross-sectional area
                # Convert radius from meters to cm for consistency
                area_cm2 = np.pi * (self.cross_section_radius * 100) ** 2
                flux_value = flux_density * area_cm2  # particles/s
        
        # Cache the result
        self._flux_cache[cache_key] = flux_value
        
        return flux_value
    
    def analyze_flux_along_path(self) -> Dict[str, Any]:
        """
        Analyze flux along entire orbital path.
        
        Returns:
            Dictionary with flux statistics
        """
        if not self.orbital_path or not self.vtk_data:
            return {}
        
        flux_values = []
        for point in self.orbital_path:
            flux = self.analyze_flux_at_point(point)
            flux_values.append(flux)
        
        flux_array = np.array(flux_values)
        
        return {
            'mean': np.mean(flux_array),
            'std': np.std(flux_array),
            'min': np.min(flux_array),
            'max': np.max(flux_array),
            'median': np.median(flux_array),
            'total_integrated': np.sum(flux_array) * (self.orbital_path[1].time - self.orbital_path[0].time) * 3600 if len(self.orbital_path) > 1 else 0,
            'values': flux_array
        }
    
    def find_peak_exposure_regions(self, threshold_percentile: float = 90) -> List[Dict[str, Any]]:
        """
        Find regions of peak radiation exposure.
        
        Args:
            threshold_percentile: Percentile threshold for peak detection
            
        Returns:
            List of peak exposure regions
        """
        if not self.orbital_path or not self.vtk_data:
            return []
        
        flux_stats = self.analyze_flux_along_path()
        if 'values' not in flux_stats:
            return []
        
        flux_values = flux_stats['values']
        threshold = np.percentile(flux_values, threshold_percentile)
        
        regions = []
        in_region = False
        start_idx = 0
        
        for i, flux in enumerate(flux_values):
            if flux >= threshold and not in_region:
                # Start of new region
                in_region = True
                start_idx = i
            elif flux < threshold and in_region:
                # End of region
                in_region = False
                regions.append({
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'start_time': self.orbital_path[start_idx].time,
                    'end_time': self.orbital_path[i - 1].time,
                    'duration': self.orbital_path[i - 1].time - self.orbital_path[start_idx].time,
                    'max_flux': np.max(flux_values[start_idx:i]),
                    'mean_flux': np.mean(flux_values[start_idx:i]),
                    'start_altitude': self.orbital_path[start_idx].altitude,
                    'end_altitude': self.orbital_path[i - 1].altitude
                })
        
        # Handle case where we're still in a region at the end
        if in_region:
            regions.append({
                'start_index': start_idx,
                'end_index': len(flux_values) - 1,
                'start_time': self.orbital_path[start_idx].time,
                'end_time': self.orbital_path[-1].time,
                'duration': self.orbital_path[-1].time - self.orbital_path[start_idx].time,
                'max_flux': np.max(flux_values[start_idx:]),
                'mean_flux': np.mean(flux_values[start_idx:]),
                'start_altitude': self.orbital_path[start_idx].altitude,
                'end_altitude': self.orbital_path[-1].altitude
            })
        
        return regions
    
    def calculate_total_dose(self, particle_energy_MeV: float = 1.0) -> Dict[str, float]:
        """
        Calculate total radiation dose along path.
        
        Args:
            particle_energy_MeV: Average particle energy in MeV
            
        Returns:
            Dictionary with dose calculations
        """
        flux_stats = self.analyze_flux_along_path()
        if not flux_stats:
            return {}
        
        # Simple dose calculation (would need more sophisticated model in practice)
        # Dose = Flux * Energy * Time
        total_particles = flux_stats.get('total_integrated', 0)
        
        # Convert to various dose units
        # These are simplified calculations - real dose would need proper energy spectrum
        joules_per_MeV = 1.60218e-13
        total_energy_J = total_particles * particle_energy_MeV * joules_per_MeV
        
        # Assume 1 kg mass for specific dose
        mass_kg = 1.0
        dose_Gy = total_energy_J / mass_kg  # Gray (J/kg)
        dose_rad = dose_Gy * 100  # rad
        
        return {
            'total_particles': total_particles,
            'total_energy_J': total_energy_J,
            'dose_Gy': dose_Gy,
            'dose_rad': dose_rad,
            'dose_mGy': dose_Gy * 1000,
            'assumed_energy_MeV': particle_energy_MeV
        }
    
    def calculate_dose_time_series(self, cross_section_m2: float = 1.0, particle_energy_MeV: float = 1.0) -> Dict[str, any]:
        """
        Calculate dose rate time series along orbital path.
        
        Args:
            cross_section_m2: Cross section in m²
            particle_energy_MeV: Average particle energy in MeV
            
        Returns:
            Dictionary with time series data for plotting
        """
        if not self.orbital_path or not self.vtk_data:
            return {}
        
        times = []
        dose_rates = []
        cumulative_dose = []
        altitudes = []
        
        cumulative_dose_value = 0.0
        joules_per_MeV = 1.60218e-13
        
        # Sample flux at each orbital point
        for i, point in enumerate(self.orbital_path):
            times.append(point.time)
            altitudes.append(point.altitude)
            
            # Get flux at this position
            flux = self._sample_flux_at_position(point.x, point.y, point.z)
            
            # Convert flux to dose rate
            # flux is in particles/(cm²·s·sr·MeV)  
            # Convert to particles/(m²·s) by multiplying by cross_section_m2 * 10000 (cm²/m²)
            particle_rate = flux * cross_section_m2 * 10000  # particles/s hitting cross section
            
            # Convert to energy rate (J/s)
            energy_rate_J_per_s = particle_rate * particle_energy_MeV * joules_per_MeV
            
            # Convert to dose rate (Gy/s) assuming 1 kg mass
            mass_kg = 1.0
            dose_rate_Gy_per_s = energy_rate_J_per_s / mass_kg
            dose_rate_mGy_per_s = dose_rate_Gy_per_s * 1000
            
            dose_rates.append(dose_rate_mGy_per_s)
            
            # Calculate cumulative dose (integrate over time)
            if i > 0:
                dt = times[i] - times[i-1]  # Time step in hours
                dt_seconds = dt * 3600  # Convert to seconds
                cumulative_dose_value += dose_rate_Gy_per_s * dt_seconds
            
            cumulative_dose.append(cumulative_dose_value * 1000)  # Convert to mGy
        
        return {
            'times': np.array(times),
            'dose_rates_mGy_per_s': np.array(dose_rates),
            'cumulative_dose_mGy': np.array(cumulative_dose),
            'altitudes': np.array(altitudes),
            'cross_section_m2': cross_section_m2,
            'particle_energy_MeV': particle_energy_MeV,
            'time_range_hours': (times[0], times[-1]) if times else (0, 0)
        }
    
    def _sample_flux_at_position(self, x: float, y: float, z: float) -> float:
        """
        Sample flux at a specific position.
        
        Args:
            x, y, z: Position coordinates in km
            
        Returns:
            Flux value at the position
        """
        # Create a temporary OrbitalPoint for the existing analyze_flux_at_point method
        temp_point = OrbitalPoint(time=0.0, x=x, y=y, z=z)
        
        # Use existing method but return raw flux density instead of integrated flux
        if not self.vtk_data:
            return 0.0
        
        # Create a probe to sample the field at this point
        probe = vtk.vtkProbeFilter()
        
        # Create a point to probe
        points = vtk.vtkPoints()
        points.InsertNextPoint(x, y, z)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        probe.SetInputData(polydata)
        probe.SetSourceData(self.vtk_data)
        probe.Update()
        
        # Get the flux value
        result = probe.GetOutput()
        
        if result.GetNumberOfPoints() > 0:
            scalar_array = result.GetPointData().GetScalars()
            if scalar_array and scalar_array.GetNumberOfTuples() > 0:
                return scalar_array.GetValue(0)  # Return raw flux density
        
        return 0.0
