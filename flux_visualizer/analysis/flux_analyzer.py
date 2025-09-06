# analysis/flux_analyzer.py
"""
Flux Analysis Module
Handles flux calculations at orbital points and cross-sections
"""

import vtk
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
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
        
    def analyze_flux_at_point(self, orbital_point: OrbitalPoint) -> Tuple[float, Optional[float]]:
        """
        Analyze flux and uncertainty at a specific orbital point.
        
        Args:
            orbital_point: Point to analyze
            
        Returns:
            Tuple of (flux_value, uncertainty_value) in particles/s
            uncertainty_value is None if no uncertainty data available
        """
        if not self.vtk_data:
            return 0.0, None
        
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
        
        # Get the flux value and uncertainty
        result = probe.GetOutput()
        flux_value = 0.0
        uncertainty_value = None
        
        if result.GetNumberOfPoints() > 0:
            scalar_array = result.GetPointData().GetScalars()
            if scalar_array and scalar_array.GetNumberOfTuples() > 0:
                flux_density = scalar_array.GetValue(0)  # particles/cm²/s
                
                # Calculate integrated flux through cross-sectional area
                # Convert radius from meters to cm for consistency
                area_cm2 = np.pi * (self.cross_section_radius * 100) ** 2
                flux_value = flux_density * area_cm2  # particles/s
                
                # Get uncertainty if available
                flux_array_name = scalar_array.GetName()
                uncertainty_value = self._get_uncertainty_at_point(result, flux_array_name, area_cm2)
        
        # Cache the result
        result_tuple = (flux_value, uncertainty_value)
        self._flux_cache[cache_key] = result_tuple
        
        return result_tuple
    
    def _get_uncertainty_at_point(self, probe_result: vtk.vtkPolyData, flux_array_name: str, area_cm2: float) -> Optional[float]:
        """
        Get uncertainty value at the probed point.
        
        Args:
            probe_result: VTK probe result containing sampled data
            flux_array_name: Name of the flux array
            area_cm2: Cross-sectional area in cm²
            
        Returns:
            Uncertainty value in particles/s or None if not available
        """
        from data_io import VTKDataLoader
        
        # Get uncertainty arrays for this flux
        uncertainty_arrays = VTKDataLoader.get_uncertainty_arrays(self.vtk_data, flux_array_name)
        
        if uncertainty_arrays['absolute']:
            # Sample absolute uncertainty
            unc_array_name = uncertainty_arrays['absolute'].GetName()
            unc_array = probe_result.GetPointData().GetArray(unc_array_name)
            if unc_array and unc_array.GetNumberOfTuples() > 0:
                uncertainty_density = unc_array.GetValue(0)  # particles/cm²/s
                return uncertainty_density * area_cm2  # particles/s
                
        elif uncertainty_arrays['relative']:
            # Sample relative uncertainty and convert to absolute
            rel_unc_array_name = uncertainty_arrays['relative'].GetName()
            rel_unc_array = probe_result.GetPointData().GetArray(rel_unc_array_name)
            if rel_unc_array and rel_unc_array.GetNumberOfTuples() > 0:
                relative_uncertainty = rel_unc_array.GetValue(0)
                # Need flux density to convert relative to absolute
                scalar_array = probe_result.GetPointData().GetScalars()
                if scalar_array and scalar_array.GetNumberOfTuples() > 0:
                    flux_density = scalar_array.GetValue(0)
                    uncertainty_density = flux_density * relative_uncertainty
                    return uncertainty_density * area_cm2  # particles/s
        
        return None
    
    def analyze_flux_along_path(self) -> Dict[str, Any]:
        """
        Analyze flux and uncertainties along entire orbital path.
        
        Returns:
            Dictionary with flux statistics and uncertainties
        """
        if not self.orbital_path or not self.vtk_data:
            return {}
        
        flux_values = []
        uncertainty_values = []
        has_uncertainty = False
        
        for point in self.orbital_path:
            flux, uncertainty = self.analyze_flux_at_point(point)
            flux_values.append(flux)
            if uncertainty is not None:
                uncertainty_values.append(uncertainty)
                has_uncertainty = True
            else:
                uncertainty_values.append(0.0)
        
        flux_array = np.array(flux_values)
        uncertainty_array = np.array(uncertainty_values) if has_uncertainty else None
        
        result = {
            'mean': np.mean(flux_array),
            'std': np.std(flux_array),
            'min': np.min(flux_array),
            'max': np.max(flux_array),
            'median': np.median(flux_array),
            'total_integrated': np.sum(flux_array) * (self.orbital_path[1].time - self.orbital_path[0].time) * 3600 if len(self.orbital_path) > 1 else 0,
            'values': flux_array,
            'has_uncertainty': has_uncertainty
        }
        
        if has_uncertainty:
            result.update({
                'uncertainties': uncertainty_array,
                'mean_uncertainty': np.mean(uncertainty_array),
                'uncertainty_range': (np.min(uncertainty_array), np.max(uncertainty_array)),
                'mean_relative_uncertainty': np.mean(uncertainty_array / np.maximum(flux_array, 1e-10))
            })
        
        return result
    
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
    
    def calculate_dose_time_series(self, cross_section_m2: float, particle_energy_MeV: float = 1.0) -> Dict[str, Any]:
        """
        Calculate dose time series with uncertainties along orbital path.
        
        Args:
            cross_section_m2: Cross-sectional area in m²
            particle_energy_MeV: Average particle energy in MeV
            
        Returns:
            Dictionary with dose time series data including uncertainties
        """
        if not self.orbital_path or not self.vtk_data:
            return {}
        
        # Set cross section for flux calculations
        original_cross_section = self.cross_section_radius
        self.cross_section_radius = np.sqrt(cross_section_m2 / np.pi)  # Convert area to radius
        
        try:
            # Analyze flux along path with uncertainties
            flux_stats = self.analyze_flux_along_path()
            if not flux_stats or 'values' not in flux_stats:
                return {}
            
            flux_values = flux_stats['values']  # particles/s
            times = np.array([point.time for point in self.orbital_path])  # hours
            has_uncertainty = flux_stats.get('has_uncertainty', False)
            flux_uncertainties = flux_stats.get('uncertainties') if has_uncertainty else None
            
            # Convert flux to dose rate
            # Simple dose conversion: dose_rate = flux * energy * conversion_factor
            joules_per_MeV = 1.60218e-13  # J/MeV
            dose_conversion = particle_energy_MeV * joules_per_MeV  # J per particle
            
            # Assume 1 kg mass for dose calculation (dose = energy / mass)
            mass_kg = 1.0
            dose_rates_Gy_per_s = flux_values * dose_conversion / mass_kg  # Gy/s
            dose_rates_mGy_per_s = dose_rates_Gy_per_s * 1000  # mGy/s
            
            # Calculate dose rate uncertainties
            dose_rate_uncertainties = None
            if has_uncertainty and flux_uncertainties is not None:
                dose_rate_uncertainties = flux_uncertainties * dose_conversion / mass_kg * 1000  # mGy/s
            
            # Calculate cumulative dose using trapezoidal integration
            cumulative_dose_mGy = np.zeros_like(dose_rates_mGy_per_s)
            cumulative_dose_uncertainties = None
            if has_uncertainty:
                cumulative_dose_uncertainties = np.zeros_like(dose_rates_mGy_per_s)
            
            for i in range(1, len(times)):
                dt_hours = times[i] - times[i-1]
                dt_seconds = dt_hours * 3600
                
                # Trapezoidal integration for dose
                avg_dose_rate = (dose_rates_mGy_per_s[i] + dose_rates_mGy_per_s[i-1]) / 2
                dose_increment = avg_dose_rate * dt_seconds
                cumulative_dose_mGy[i] = cumulative_dose_mGy[i-1] + dose_increment
                
                # Uncertainty propagation for cumulative dose
                if has_uncertainty and dose_rate_uncertainties is not None:
                    # For integration: σ²_cum = σ²_cum_prev + (σ_rate * dt)²
                    avg_dose_rate_uncertainty = (dose_rate_uncertainties[i] + dose_rate_uncertainties[i-1]) / 2
                    dose_uncertainty_increment = avg_dose_rate_uncertainty * dt_seconds
                    cumulative_uncertainty_variance = (cumulative_dose_uncertainties[i-1]**2 + 
                                                     dose_uncertainty_increment**2)
                    cumulative_dose_uncertainties[i] = np.sqrt(cumulative_uncertainty_variance)
            
            result = {
                'times': times,
                'dose_rates_mGy_per_s': dose_rates_mGy_per_s,
                'cumulative_dose_mGy': cumulative_dose_mGy,
                'particle_energy_MeV': particle_energy_MeV,
                'cross_section_m2': cross_section_m2,
                'time_range_hours': (times[0], times[-1]),
                'has_uncertainty': has_uncertainty
            }
            
            if has_uncertainty:
                result.update({
                    'dose_rate_uncertainties_mGy_per_s': dose_rate_uncertainties,
                    'cumulative_dose_uncertainties_mGy': cumulative_dose_uncertainties,
                    'mean_relative_uncertainty': flux_stats.get('mean_relative_uncertainty', 0.0)
                })
            
            return result
            
        finally:
            # Restore original cross section
            self.cross_section_radius = original_cross_section
    
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
