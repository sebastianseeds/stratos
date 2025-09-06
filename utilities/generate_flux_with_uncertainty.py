#!/usr/bin/env python3
"""
Advanced Flux Field Generator with Uncertainty for STRATOS
Generates static VTK flux fields with configurable physics and uncertainty data.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import argparse
import time
from typing import Dict, Tuple, Optional

class FluxFieldGenerator:
    """Generate radiation flux fields with physical models and uncertainties"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration parameters"""
        self.config = config
        self.earth_radius = 6371.0  # km
        self.magnetic_tilt = np.radians(11.0)  # Magnetic dipole tilt
        
    def calculate_magnetic_coordinates(self, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
        """Transform geographic to magnetic coordinates and calculate L-shell"""
        # Rotation matrix for magnetic dipole tilt
        cos_tilt = np.cos(self.magnetic_tilt)
        sin_tilt = np.sin(self.magnetic_tilt)
        
        # Transform to magnetic coordinates
        x_mag = x * cos_tilt + z * sin_tilt
        y_mag = y
        z_mag = -x * sin_tilt + z * cos_tilt
        
        # Distance and L-shell
        r_mag = np.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
        
        if r_mag <= self.earth_radius:
            return x_mag, y_mag, z_mag, 0.0  # Inside Earth
        
        magnetic_latitude = np.arcsin(np.abs(z_mag) / r_mag)
        L_shell = r_mag / (self.earth_radius * np.cos(magnetic_latitude)**2)
        
        return x_mag, y_mag, z_mag, L_shell
    
    def calculate_flux_scalar(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """
        Calculate scalar flux and uncertainty at a point.
        
        Returns:
            (flux, uncertainty) in particles/(cm²·s·sr·MeV)
        """
        x_mag, y_mag, z_mag, L_shell = self.calculate_magnetic_coordinates(x, y, z)
        
        if L_shell == 0:
            return 0.0, 0.0
        
        r_mag = np.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
        magnetic_latitude = np.arcsin(np.abs(z_mag) / r_mag)
        
        flux = 0.0
        uncertainty_components = []
        
        # Inner Belt
        if self.config['inner_belt']['enabled'] and \
           self.config['inner_belt']['L_min'] <= L_shell <= self.config['inner_belt']['L_max']:
            
            inner_config = self.config['inner_belt']
            L_peak = inner_config['L_peak']
            L_width = inner_config['L_width']
            amplitude = inner_config['amplitude']
            
            # Gaussian profile in L-shell
            inner_flux = amplitude * np.exp(-((L_shell - L_peak)**2) / (2 * L_width**2))
            
            # Latitudinal dependence
            lat_factor = 1.0 + inner_config['lat_dependence'] * np.sin(magnetic_latitude)**2
            
            inner_contribution = inner_flux * lat_factor
            flux += inner_contribution
            
            # Uncertainty for inner belt (higher due to variability)
            inner_uncertainty = inner_contribution * inner_config['relative_uncertainty']
            uncertainty_components.append(inner_uncertainty)
        
        # Slot Region
        if self.config['slot_region']['enabled'] and \
           self.config['slot_region']['L_min'] <= L_shell <= self.config['slot_region']['L_max']:
            
            slot_config = self.config['slot_region']
            slot_flux = slot_config['amplitude'] * \
                       np.exp(-((L_shell - slot_config['L_center'])**2) / (2 * slot_config['L_width']**2))
            flux += slot_flux
            
            # High uncertainty in slot region
            slot_uncertainty = slot_flux * slot_config['relative_uncertainty']
            uncertainty_components.append(slot_uncertainty)
        
        # Outer Belt
        if self.config['outer_belt']['enabled'] and \
           self.config['outer_belt']['L_min'] <= L_shell <= self.config['outer_belt']['L_max']:
            
            outer_config = self.config['outer_belt']
            L_peak = outer_config['L_peak']
            L_width = outer_config['L_width']
            amplitude = outer_config['amplitude']
            
            # Gaussian profile
            outer_flux = amplitude * np.exp(-((L_shell - L_peak)**2) / (2 * L_width**2))
            
            # Latitudinal dependence
            lat_factor = 1.0 + outer_config['lat_dependence'] * np.sin(magnetic_latitude)**2
            
            # Local time asymmetry (day/night)
            if outer_config['day_night_asymmetry'] > 0:
                local_time_angle = np.arctan2(y_mag, x_mag)
                day_night_factor = 1.0 + outer_config['day_night_asymmetry'] * np.cos(local_time_angle)
            else:
                day_night_factor = 1.0
            
            outer_contribution = outer_flux * lat_factor * day_night_factor
            flux += outer_contribution
            
            # Outer belt has highest variability
            outer_uncertainty = outer_contribution * outer_config['relative_uncertainty']
            uncertainty_components.append(outer_uncertainty)
        
        # Plasma Sheet
        if self.config['plasma_sheet']['enabled'] and L_shell > self.config['plasma_sheet']['L_min']:
            
            plasma_config = self.config['plasma_sheet']
            plasma_flux = plasma_config['amplitude'] * \
                         np.exp(-(L_shell - plasma_config['L_center'])**2 / (2 * plasma_config['L_width']**2))
            flux += plasma_flux
            
            # Very high uncertainty in plasma sheet
            plasma_uncertainty = plasma_flux * plasma_config['relative_uncertainty']
            uncertainty_components.append(plasma_uncertainty)
        
        # Background radiation
        if self.config['background']['enabled']:
            background = self.config['background']['amplitude']
            flux += background
            background_uncertainty = background * self.config['background']['relative_uncertainty']
            uncertainty_components.append(background_uncertainty)
        
        # Combine uncertainties in quadrature
        if uncertainty_components:
            total_uncertainty = np.sqrt(sum(u**2 for u in uncertainty_components))
        else:
            total_uncertainty = 0.0
        
        # Add measurement uncertainty floor
        measurement_uncertainty = flux * self.config['measurement_uncertainty']
        total_uncertainty = np.sqrt(total_uncertainty**2 + measurement_uncertainty**2)
        
        return flux, total_uncertainty
    
    def calculate_flux_vector(self, x: float, y: float, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate vector flux and uncertainty at a point.
        
        Returns:
            (flux_vector, uncertainty_vector) where each is [vx, vy, vz] components
        """
        # Get scalar flux first
        flux_scalar, uncertainty_scalar = self.calculate_flux_scalar(x, y, z)
        
        if flux_scalar == 0:
            return np.zeros(3), np.zeros(3)
        
        x_mag, y_mag, z_mag, L_shell = self.calculate_magnetic_coordinates(x, y, z)
        
        if L_shell == 0:
            return np.zeros(3), np.zeros(3)
        
        # Calculate pitch angle distribution
        r_mag = np.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
        
        # Magnetic field direction (dipole approximation)
        B_r = 2 * z_mag / r_mag  # Radial component
        B_theta = np.sqrt(x_mag**2 + y_mag**2) / r_mag  # Tangential component
        
        # Normalize magnetic field
        B_mag = np.sqrt(B_r**2 + B_theta**2)
        if B_mag > 0:
            B_unit = np.array([x_mag * B_theta, y_mag * B_theta, z_mag * B_r]) / (r_mag * B_mag)
        else:
            B_unit = np.array([0, 0, 1])
        
        # Anisotropy factor (particles follow field lines)
        anisotropy = self.config['anisotropy_factor']
        
        if anisotropy == 0:
            # Isotropic flux
            flux_vector = flux_scalar * np.ones(3) / 3
            uncertainty_vector = uncertainty_scalar * np.ones(3) / 3
        else:
            # Anisotropic flux aligned with magnetic field
            parallel_fraction = 0.5 * (1 + anisotropy)
            perpendicular_fraction = 0.5 * (1 - anisotropy)
            
            # Create perpendicular components
            if abs(B_unit[2]) < 0.99:
                perp1 = np.cross(B_unit, [0, 0, 1])
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(B_unit, perp1)
            else:
                perp1 = np.array([1, 0, 0])
                perp2 = np.array([0, 1, 0])
            
            # Combine parallel and perpendicular components
            flux_vector = flux_scalar * (
                parallel_fraction * B_unit +
                perpendicular_fraction * 0.5 * (perp1 + perp2)
            )
            
            # Uncertainty propagates similarly
            uncertainty_vector = uncertainty_scalar * (
                parallel_fraction * B_unit +
                perpendicular_fraction * 0.5 * (perp1 + perp2)
            )
        
        return flux_vector, uncertainty_vector
    
    def generate_field(self, nx: int, ny: int, nz: int, 
                      x_range: Tuple[float, float],
                      y_range: Tuple[float, float], 
                      z_range: Tuple[float, float]) -> vtk.vtkStructuredGrid:
        """Generate the complete flux field with uncertainties"""
        
        print(f"Generating {nx}×{ny}×{nz} flux field...")
        print(f"  X range: {x_range[0]:.0f} to {x_range[1]:.0f} km")
        print(f"  Y range: {y_range[0]:.0f} to {y_range[1]:.0f} km")
        print(f"  Z range: {z_range[0]:.0f} to {z_range[1]:.0f} km")
        
        # Create coordinate arrays
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)
        
        # Create structured grid
        sgrid = vtk.vtkStructuredGrid()
        sgrid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(nx * ny * nz)
        
        # Arrays for data
        flux_values = []
        uncertainty_values = []
        flux_vectors = [] if self.config['include_vector'] else None
        uncertainty_vectors = [] if self.config['include_vector'] else None
        
        # Progress tracking
        total_points = nx * ny * nz
        point_idx = 0
        start_time = time.time()
        
        print("Computing flux field...")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Set point coordinates
                    points.SetPoint(point_idx, x[i], y[j], z[k])
                    
                    # Calculate flux
                    if self.config['include_vector']:
                        flux_vec, unc_vec = self.calculate_flux_vector(x[i], y[j], z[k])
                        flux_vectors.extend(flux_vec)
                        uncertainty_vectors.extend(unc_vec)
                        # Also store magnitude for scalar visualization
                        flux_values.append(np.linalg.norm(flux_vec))
                        uncertainty_values.append(np.linalg.norm(unc_vec))
                    else:
                        flux, uncertainty = self.calculate_flux_scalar(x[i], y[j], z[k])
                        flux_values.append(flux)
                        uncertainty_values.append(uncertainty)
                    
                    point_idx += 1
                    
                    # Progress bar
                    if point_idx % 1000 == 0 or point_idx == total_points:
                        progress = point_idx / total_points
                        bar_length = 50
                        filled = int(bar_length * progress)
                        bar = '█' * filled + '-' * (bar_length - filled)
                        elapsed = time.time() - start_time
                        eta = elapsed / progress - elapsed if progress > 0 else 0
                        print(f'\r  |{bar}| {progress*100:.1f}% - ETA: {eta:.0f}s', end='')
        
        print()  # New line after progress bar
        
        # Set points to grid
        sgrid.SetPoints(points)
        
        # Add scalar flux data
        flux_array = numpy_to_vtk(np.array(flux_values))
        flux_array.SetName(self.config['flux_name'])
        sgrid.GetPointData().SetScalars(flux_array)
        
        # Add uncertainty data
        uncertainty_array = numpy_to_vtk(np.array(uncertainty_values))
        uncertainty_array.SetName(f"{self.config['flux_name']}_uncertainty")
        sgrid.GetPointData().AddArray(uncertainty_array)
        
        # Add relative uncertainty
        relative_uncertainty = np.array([
            u/f if f > 0 else 0 for f, u in zip(flux_values, uncertainty_values)
        ])
        rel_unc_array = numpy_to_vtk(relative_uncertainty)
        rel_unc_array.SetName(f"{self.config['flux_name']}_relative_uncertainty")
        sgrid.GetPointData().AddArray(rel_unc_array)
        
        # Add vector data if requested
        if self.config['include_vector']:
            # Flux vector
            flux_vec_array = numpy_to_vtk(np.array(flux_vectors).reshape(-1, 3))
            flux_vec_array.SetName(f"{self.config['flux_name']}_vector")
            sgrid.GetPointData().SetVectors(flux_vec_array)
            
            # Uncertainty vector
            unc_vec_array = numpy_to_vtk(np.array(uncertainty_vectors).reshape(-1, 3))
            unc_vec_array.SetName(f"{self.config['flux_name']}_uncertainty_vector")
            sgrid.GetPointData().AddArray(unc_vec_array)
        
        # Print statistics
        flux_np = np.array(flux_values)
        unc_np = np.array(uncertainty_values)
        
        print(f"\nFlux field statistics:")
        print(f"  Flux range: {flux_np.min():.2e} to {flux_np.max():.2e}")
        print(f"  Mean flux: {flux_np.mean():.2e}")
        print(f"  Uncertainty range: {unc_np.min():.2e} to {unc_np.max():.2e}")
        print(f"  Mean relative uncertainty: {(unc_np/np.maximum(flux_np, 1e-10)).mean()*100:.1f}%")
        print(f"  Non-zero points: {np.sum(flux_np > 0):,} / {len(flux_np):,}")
        
        return sgrid


def get_default_config():
    """Get default configuration for flux field generation"""
    return {
        # Output settings
        'flux_name': 'electron_flux',
        'include_vector': False,
        'anisotropy_factor': 0.0,  # 0 = isotropic, 1 = field-aligned
        
        # Measurement uncertainty (applies to all measurements)
        'measurement_uncertainty': 0.05,  # 5% baseline measurement uncertainty
        
        # Inner Van Allen Belt
        'inner_belt': {
            'enabled': True,
            'L_min': 1.2,
            'L_max': 2.8,
            'L_peak': 1.6,
            'L_width': 0.4,
            'amplitude': 2e7,  # particles/(cm²·s·sr·MeV)
            'lat_dependence': 2.0,
            'relative_uncertainty': 0.15  # 15% uncertainty
        },
        
        # Slot Region
        'slot_region': {
            'enabled': True,
            'L_min': 2.5,
            'L_max': 3.2,
            'L_center': 2.8,
            'L_width': 0.2,
            'amplitude': 1e5,
            'relative_uncertainty': 0.30  # 30% uncertainty (highly variable)
        },
        
        # Outer Van Allen Belt
        'outer_belt': {
            'enabled': True,
            'L_min': 3.0,
            'L_max': 8.0,
            'L_peak': 4.5,
            'L_width': 1.2,
            'amplitude': 1e7,
            'lat_dependence': 1.5,
            'day_night_asymmetry': 0.3,
            'relative_uncertainty': 0.20  # 20% uncertainty
        },
        
        # Plasma Sheet
        'plasma_sheet': {
            'enabled': True,
            'L_min': 6.0,
            'L_center': 8.0,
            'L_width': 2.0,
            'amplitude': 5e5,
            'relative_uncertainty': 0.40  # 40% uncertainty (highly dynamic)
        },
        
        # Background
        'background': {
            'enabled': True,
            'amplitude': 1e4,
            'relative_uncertainty': 0.10
        }
    }


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Generate static flux fields with uncertainty for STRATOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output flux_with_errors.vts
  %(prog)s --no-inner --no-slot --vector --output outer_belt_only.vts
  %(prog)s --resolution 100 --extent 100000 --output high_res_flux.vts
  %(prog)s --particle proton --inner-amp 5e7 --output proton_flux.vts

Belt Configuration:
  Use --no-inner, --no-slot, --no-outer, --no-plasma to disable regions
  Use --inner-amp, --outer-amp, etc. to adjust flux amplitudes
  
Output Arrays:
  The VTS file will contain the following arrays:
  - [particle]_flux: Primary scalar flux values
  - [particle]_flux_uncertainty: Absolute uncertainty values
  - [particle]_flux_relative_uncertainty: Relative uncertainty (0-1)
  - [particle]_flux_vector: Flux vector (if --vector is used)
  - [particle]_flux_uncertainty_vector: Uncertainty vector (if --vector is used)
""")
    
    # Output options
    parser.add_argument('--output', '-o', default='../data/flux/static/flux_with_uncertainty.vts',
                       help='Output VTS filename')
    parser.add_argument('--particle', '-p', default='electron',
                       choices=['electron', 'proton', 'alpha', 'ion'],
                       help='Particle type (default: electron)')
    
    # Grid resolution
    parser.add_argument('--resolution', '-r', type=int, default=80,
                       help='Grid resolution (NxNxN) (default: 80)')
    parser.add_argument('--extent', '-e', type=float, default=76000,
                       help='Spatial extent in km from origin (default: 76000)')
    
    # Physics options
    parser.add_argument('--vector', action='store_true',
                       help='Include vector flux components')
    parser.add_argument('--anisotropy', type=float, default=0.0,
                       help='Anisotropy factor 0=isotropic, 1=field-aligned (default: 0)')
    
    # Belt configuration
    parser.add_argument('--no-inner', action='store_true',
                       help='Disable inner Van Allen belt')
    parser.add_argument('--no-slot', action='store_true',
                       help='Disable slot region')
    parser.add_argument('--no-outer', action='store_true',
                       help='Disable outer Van Allen belt')
    parser.add_argument('--no-plasma', action='store_true',
                       help='Disable plasma sheet')
    parser.add_argument('--no-background', action='store_true',
                       help='Disable background radiation')
    
    # Amplitude adjustments
    parser.add_argument('--inner-amp', type=float,
                       help='Inner belt amplitude (default: 2e7)')
    parser.add_argument('--outer-amp', type=float,
                       help='Outer belt amplitude (default: 1e7)')
    parser.add_argument('--plasma-amp', type=float,
                       help='Plasma sheet amplitude (default: 5e5)')
    
    # Uncertainty adjustments
    parser.add_argument('--measurement-error', type=float, default=0.05,
                       help='Baseline measurement uncertainty (default: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = get_default_config()
    
    # Update particle type
    config['flux_name'] = f"{args.particle}_flux"
    
    # Update physics options
    config['include_vector'] = args.vector
    config['anisotropy_factor'] = args.anisotropy
    config['measurement_uncertainty'] = args.measurement_error
    
    # Update belt configuration
    if args.no_inner:
        config['inner_belt']['enabled'] = False
    if args.no_slot:
        config['slot_region']['enabled'] = False
    if args.no_outer:
        config['outer_belt']['enabled'] = False
    if args.no_plasma:
        config['plasma_sheet']['enabled'] = False
    if args.no_background:
        config['background']['enabled'] = False
    
    # Update amplitudes if specified
    if args.inner_amp:
        config['inner_belt']['amplitude'] = args.inner_amp
    if args.outer_amp:
        config['outer_belt']['amplitude'] = args.outer_amp
    if args.plasma_amp:
        config['plasma_sheet']['amplitude'] = args.plasma_amp
    
    # Adjust amplitudes for different particle types
    if args.particle == 'proton':
        # Protons have different distributions
        config['inner_belt']['amplitude'] *= 2.0  # More protons in inner belt
        config['outer_belt']['amplitude'] *= 0.3  # Fewer in outer belt
    elif args.particle == 'alpha':
        # Alpha particles are less abundant
        config['inner_belt']['amplitude'] *= 0.05
        config['outer_belt']['amplitude'] *= 0.02
    elif args.particle == 'ion':
        # Heavy ions are rare
        config['inner_belt']['amplitude'] *= 0.01
        config['outer_belt']['amplitude'] *= 0.005
    
    print("=" * 60)
    print("STRATOS Flux Field Generator with Uncertainty")
    print("=" * 60)
    print(f"Particle type: {args.particle}")
    print(f"Grid resolution: {args.resolution}×{args.resolution}×{args.resolution}")
    print(f"Spatial extent: ±{args.extent:.0f} km")
    print(f"Vector flux: {'Yes' if args.vector else 'No'}")
    if args.vector:
        print(f"Anisotropy: {args.anisotropy:.1f}")
    print(f"Measurement uncertainty: {args.measurement_error*100:.1f}%")
    print()
    print("Active regions:")
    if config['inner_belt']['enabled']:
        print(f"  ✓ Inner belt (L={config['inner_belt']['L_min']}-{config['inner_belt']['L_max']})")
    if config['slot_region']['enabled']:
        print(f"  ✓ Slot region (L={config['slot_region']['L_min']}-{config['slot_region']['L_max']})")
    if config['outer_belt']['enabled']:
        print(f"  ✓ Outer belt (L={config['outer_belt']['L_min']}-{config['outer_belt']['L_max']})")
    if config['plasma_sheet']['enabled']:
        print(f"  ✓ Plasma sheet (L>{config['plasma_sheet']['L_min']})")
    if config['background']['enabled']:
        print(f"  ✓ Background radiation")
    print()
    
    # Create generator
    generator = FluxFieldGenerator(config)
    
    # Generate field
    extent = args.extent
    grid = generator.generate_field(
        nx=args.resolution, ny=args.resolution, nz=args.resolution,
        x_range=(-extent, extent),
        y_range=(-extent, extent),
        z_range=(-extent, extent)
    )
    
    # Write to file
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()
    writer.Write()
    
    print(f"\n✓ Flux field saved to: {args.output}")
    print(f"\nLoad this file in STRATOS to visualize flux with uncertainty data.")
    print(f"Available arrays:")
    print(f"  - {config['flux_name']} (primary scalar)")
    print(f"  - {config['flux_name']}_uncertainty (absolute uncertainty)")
    print(f"  - {config['flux_name']}_relative_uncertainty (relative uncertainty)")
    if config['include_vector']:
        print(f"  - {config['flux_name']}_vector (flux vector field)")
        print(f"  - {config['flux_name']}_uncertainty_vector (uncertainty vector)")


if __name__ == "__main__":
    main()