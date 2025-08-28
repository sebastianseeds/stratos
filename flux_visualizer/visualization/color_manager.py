"""
Color Management Module for Electron Flux Orbital Visualizer
Handles all color mapping, lookup tables, and transfer functions
"""

import vtk
import numpy as np
import colorsys
from typing import Tuple, List, Optional


class ColorManager:
    """Manages color mapping and lookup tables for all visualization modes"""
    
    def __init__(self):
        """Initialize color manager with caching for performance"""
        self._lut_cache = {}
        self._color_cache = {}
        
    def create_lookup_table(self, colormap_name: str) -> vtk.vtkLookupTable:
        """
        Create a basic lookup table with the specified colormap.
        This is a simplified version that uses linear scale with default range.
        
        Args:
            colormap_name: Name of the colormap to use
            
        Returns:
            vtk.vtkLookupTable configured with the specified colormap
        """
        # Default to linear scale and a standard range
        return self.create_lookup_table_with_scale(
            colormap_name, 
            "Linear", 
            (1e-5, 1e7)  # Default range
        )
    
    def create_lookup_table_with_scale(self, 
                                      colormap_name: str, 
                                      scale_mode: str, 
                                      scalar_range: Tuple[float, float]) -> vtk.vtkLookupTable:
        """
        Create lookup table with proper scaling and colormap.
        
        Args:
            colormap_name: Name of the colormap ("Blue to Red", "Viridis", etc.)
            scale_mode: "Linear" or "Logarithmic"
            scalar_range: (min_val, max_val) tuple for the data range
            
        Returns:
            Configured vtkLookupTable
        """
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        try:
            min_val, max_val = scalar_range
            
            print(f"\n=== CREATING {scale_mode} LUT for {colormap_name} ===")
            print(f"Input range: {min_val:.2e} to {max_val:.2e}")
            
            # Handle zero minimum for logarithmic scale
            if scale_mode == "Logarithmic":
                if min_val <= 0:
                    # Use a fraction of max value as minimum
                    min_val = max(max_val * 1e-6, 1e-10)
                    print(f"FIXED zero minimum for log scale: {min_val:.2e}")
                
                if max_val <= min_val:
                    max_val = min_val * 1000
                    print(f"FIXED invalid max for log scale: {max_val:.2e}")
            
            # Validate range is finite
            if not (np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val):
                print("FIXING invalid range with fallback")
                min_val, max_val = 1e-5, 1e7
            
            print(f"Final range: {min_val:.2e} to {max_val:.2e}")
            
            # Set range FIRST
            lut.SetRange(min_val, max_val)
            
            # Build colormap with proper color assignment
            for i in range(256):
                t = i / 255.0  # 0 to 1
                rgb_color = self.get_colormap_color(colormap_name, t)
                lut.SetTableValue(i, rgb_color[0], rgb_color[1], rgb_color[2], 1.0)
            
            # Set scale mode AFTER building colors
            if scale_mode == "Logarithmic":
                lut.SetScaleToLog10()
                print("Applied logarithmic scaling")
            else:
                lut.SetScaleToLinear()
                print("Applied linear scaling")
            
            # Test the LUT
            test_val = (min_val + max_val) / 2
            test_color = [0, 0, 0]
            lut.GetColor(test_val, test_color)
            print(f"Test color at {test_val:.2e}: {test_color}")
            
            if any(np.isnan(test_color)):
                print("WARNING: NaN in colors, forcing linear")
                lut.SetScaleToLinear()
            
            print(f"=== {colormap_name} LUT CREATION COMPLETE ===\n")
            return lut
            
        except Exception as e:
            print(f"ERROR in LUT creation: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback
            lut.SetRange(1e-5, 1e7)
            for i in range(256):
                t = i / 255.0
                lut.SetTableValue(i, t, 0.0, 1.0-t, 1.0)  # Simple blue to red
            lut.SetScaleToLinear()
            return lut
    
    def get_colormap_color(self, colormap_name: str, t: float) -> List[float]:
        """
        Get RGB color for given colormap at position t (0-1).
        
        Args:
            colormap_name: Name of the colormap
            t: Position in colormap (0 to 1)
            
        Returns:
            [r, g, b] list with values 0-1
        """
        t = np.clip(t, 0, 1)
        
        # Check cache first
        cache_key = (colormap_name, round(t, 3))  # Round to 3 decimals for cache
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]
        
        # Calculate color based on colormap
        if colormap_name == "Viridis":
            color = self.get_viridis_color(t)
        elif colormap_name == "Plasma":
            color = self.get_plasma_color(t)
        elif colormap_name == "Cool to Warm":
            color = self.get_cool_warm_color(t)
        elif colormap_name == "Grayscale":
            color = [t, t, t]
        elif colormap_name == "Rainbow":
            color = self.get_rainbow_color(t)
        else:  # "Blue to Red" or default
            color = [t, 0.0, 1.0 - t]
        
        # Cache the result
        self._color_cache[cache_key] = color
        return color
    
    def get_viridis_color(self, t: float) -> List[float]:
        """
        Get Viridis colormap color using high-quality approximation.
        
        Args:
            t: Position in colormap (0 to 1)
            
        Returns:
            [r, g, b] color values
        """
        # High-quality viridis approximation using key points
        viridis_points = [
            (0.0, [0.267004, 0.004874, 0.329415]),
            (0.25, [0.229739, 0.322361, 0.545706]),
            (0.5, [0.127568, 0.566949, 0.550556]),
            (0.75, [0.369214, 0.788675, 0.382914]),
            (1.0, [0.993248, 0.909560, 0.143936])
        ]
        
        # Find the two points to interpolate between
        for i in range(len(viridis_points) - 1):
            t1, color1 = viridis_points[i]
            t2, color2 = viridis_points[i + 1]
            
            if t1 <= t <= t2:
                # Linear interpolation
                factor = (t - t1) / (t2 - t1)
                return [
                    color1[0] + factor * (color2[0] - color1[0]),
                    color1[1] + factor * (color2[1] - color1[1]),
                    color1[2] + factor * (color2[2] - color1[2])
                ]
        
        # Fallback (shouldn't happen)
        return [t, t, t]
    
    def get_plasma_color(self, t: float) -> List[float]:
        """
        Get Plasma colormap color using high-quality approximation.
        
        Args:
            t: Position in colormap (0 to 1)
            
        Returns:
            [r, g, b] color values
        """
        # High-quality plasma approximation using key points
        plasma_points = [
            (0.0, [0.050383, 0.029803, 0.527975]),
            (0.25, [0.513094, 0.038756, 0.627828]),
            (0.5, [0.796386, 0.278894, 0.469397]),
            (0.75, [0.940015, 0.644680, 0.222675]),
            (1.0, [0.940015, 0.975158, 0.131326])
        ]
        
        # Find the two points to interpolate between
        for i in range(len(plasma_points) - 1):
            t1, color1 = plasma_points[i]
            t2, color2 = plasma_points[i + 1]
            
            if t1 <= t <= t2:
                # Linear interpolation
                factor = (t - t1) / (t2 - t1)
                return [
                    color1[0] + factor * (color2[0] - color1[0]),
                    color1[1] + factor * (color2[1] - color1[1]),
                    color1[2] + factor * (color2[2] - color1[2])
                ]
        
        # Fallback
        return [t, 0, 1-t]
    
    def get_cool_warm_color(self, t: float) -> List[float]:
        """
        Get Cool to Warm colormap color (blue -> white -> red).
        
        Args:
            t: Position in colormap (0 to 1)
            
        Returns:
            [r, g, b] color values
        """
        if t < 0.5:
            # Blue to white
            factor = t * 2
            return [factor, factor, 1.0]
        else:
            # White to red
            factor = (t - 0.5) * 2
            return [1.0, 1.0 - factor, 1.0 - factor]
    
    def get_rainbow_color(self, t: float) -> List[float]:
        """
        Get rainbow color using proper HSV to RGB conversion.
        
        Args:
            t: Position in colormap (0 to 1)
            
        Returns:
            [r, g, b] color values
        """
        # Map t to hue (reverse so red is high values)
        hue = (1.0 - t) * 0.83  # 0.83 to avoid wrapping back to red
        saturation = 1.0
        value = 1.0
        
        # HSV to RGB conversion
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return list(rgb)
    
    def hsv_to_rgb(self, h: float, s: float, v: float) -> List[float]:
        """
        Convert HSV to RGB color space.
        
        Args:
            h: Hue (0-1)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            [r, g, b] color values
        """
        return list(colorsys.hsv_to_rgb(h, s, v))
    
    # ============================================
    # VOLUME RENDERING SPECIFIC FUNCTIONS
    # ============================================
    
    def create_volume_color_function(self, 
                                    flux_threshold: float, 
                                    scalar_range: Tuple[float, float], 
                                    colormap_name: str, 
                                    scale_mode: str) -> vtk.vtkColorTransferFunction:
        """
        Create color transfer function for volume rendering.
        
        Args:
            flux_threshold: Minimum flux value to show
            scalar_range: (min, max) range of the data
            colormap_name: Name of the colormap to use
            scale_mode: "Linear" or "Logarithmic"
            
        Returns:
            Configured vtkColorTransferFunction
        """
        try:
            color_func = vtk.vtkColorTransferFunction()
            
            # Everything below threshold = black (invisible)
            color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(flux_threshold * 0.999, 0.0, 0.0, 0.0)
            
            # Color range for visible data
            color_range_min = flux_threshold
            color_range_max = scalar_range[1]
            
            # Use fewer color points to reduce texture requirements
            if scale_mode == "Logarithmic" and color_range_min > 0:
                # Logarithmic color mapping with fewer points
                log_min = np.log10(color_range_min)
                log_max = np.log10(color_range_max)
                
                num_points = 4  # Reduced from 8 to 4
                for i in range(num_points):
                    log_pos = log_min + i * (log_max - log_min) / (num_points - 1)
                    flux_value = 10**log_pos
                    color_position = i / (num_points - 1)
                    rgb_color = self.get_colormap_color(colormap_name, color_position)
                    color_func.AddRGBPoint(flux_value, rgb_color[0], rgb_color[1], rgb_color[2])
            else:
                # Linear color mapping with fewer points
                num_points = 3  # Reduced from 6 to 3
                for i in range(num_points):
                    flux_value = color_range_min + i * (color_range_max - color_range_min) / (num_points - 1)
                    color_position = i / (num_points - 1)
                    rgb_color = self.get_colormap_color(colormap_name, color_position)
                    color_func.AddRGBPoint(flux_value, rgb_color[0], rgb_color[1], rgb_color[2])
            
            return color_func
            
        except Exception as e:
            print(f"Error creating volume color function: {e}")
            # Simple fallback
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(flux_threshold, 0.0, 0.0, 1.0)
            color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)
            return color_func
    
    def create_volume_opacity_function(self, 
                                      flux_threshold: float, 
                                      scalar_range: Tuple[float, float]) -> vtk.vtkPiecewiseFunction:
        """
        Create opacity function for volume rendering with fewer points to reduce texture size.
        
        Args:
            flux_threshold: Minimum flux value to show
            scalar_range: (min, max) range of the data
            
        Returns:
            Configured vtkPiecewiseFunction
        """
        try:
            opacity_func = vtk.vtkPiecewiseFunction()
            
            # Use minimal opacity points to reduce texture requirements
            opacity_func.AddPoint(scalar_range[0], 0.0)                      # Background invisible
            opacity_func.AddPoint(flux_threshold * 0.999, 0.0)              # Just below threshold invisible
            opacity_func.AddPoint(flux_threshold, 0.6)                      # At threshold suddenly visible
            opacity_func.AddPoint(scalar_range[1], 0.9)                     # Maximum flux most opaque
            
            return opacity_func
            
        except Exception as e:
            print(f"Error creating volume opacity function: {e}")
            # Minimal fallback
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(scalar_range[0], 0.0)
            opacity_func.AddPoint(flux_threshold, 0.7)
            opacity_func.AddPoint(scalar_range[1], 0.9)
            return opacity_func
    
    def update_volume_transfer_functions(self, 
                                        volume_actor: vtk.vtkVolume, 
                                        scalar_range: Tuple[float, float], 
                                        scale_mode: str) -> None:
        """
        Update volume rendering transfer functions for scale changes.
        
        Args:
            volume_actor: The volume actor to update
            scalar_range: (min, max) range of the data
            scale_mode: "Linear" or "Logarithmic"
        """
        if not volume_actor:
            return
        
        try:
            volume_property = volume_actor.GetProperty()
            
            # Create MINIMAL transfer functions for speed
            color_func = vtk.vtkColorTransferFunction()
            opacity_func = vtk.vtkPiecewiseFunction()
            
            if scale_mode == "Logarithmic" and scalar_range[1] > scalar_range[0] and scalar_range[0] > 0:
                # Log scale - but keep it SIMPLE
                log_min = np.log10(max(scalar_range[0], scalar_range[1] * 1e-6))
                log_max = np.log10(scalar_range[1])
                
                # Only 3 points for speed
                for i in range(3):
                    log_val = log_min + i * (log_max - log_min) / 2
                    linear_val = 10**log_val
                    t = i / 2.0
                    
                    # Simple color progression
                    if t < 0.5:
                        color = [0.0, t*2, 1.0]
                    else:
                        color = [(t-0.5)*2, 1.0, 1.0-(t-0.5)*2]
                    
                    color_func.AddRGBPoint(linear_val, color[0], color[1], color[2])
                    
                    # VERY low opacity for performance
                    opacity = min(0.1, 0.01 + t * 0.09)
                    opacity_func.AddPoint(linear_val, opacity)
            else:
                # Linear scale - MINIMAL points
                color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.2)
                color_func.AddRGBPoint(scalar_range[1] * 0.5, 0.0, 1.0, 0.0)
                color_func.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)
                
                # VERY sparse and low opacity
                opacity_func.AddPoint(scalar_range[0], 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.3, 0.0)
                opacity_func.AddPoint(scalar_range[1] * 0.7, 0.05)
                opacity_func.AddPoint(scalar_range[1], 0.1)
            
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            
            print(f"Updated volume rendering to {scale_mode} scale (optimized)")
            
        except Exception as e:
            print(f"Error updating volume transfer functions: {e}")
    
    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    def get_satellite_color(self) -> List[float]:
        """
        Get a random distinct color for satellite visualization.
        
        Returns:
            [r, g, b] color values
        """
        import random
        
        colors = [
            [1.0, 0.2, 0.2],  # Bright red
            [0.2, 1.0, 0.2],  # Bright green  
            [1.0, 0.2, 1.0],  # Bright magenta
            [0.2, 0.8, 1.0],  # Bright cyan
            [1.0, 0.8, 0.2],  # Bright orange
            [0.8, 0.2, 1.0],  # Bright purple
            [1.0, 0.5, 0.8],  # Bright pink
            [0.2, 1.0, 0.8],  # Bright teal
        ]
        return random.choice(colors)
    
    def interpolate_color(self, 
                         color1: List[float], 
                         color2: List[float], 
                         t: float) -> List[float]:
        """
        Linearly interpolate between two colors.
        
        Args:
            color1: Starting [r, g, b] color
            color2: Ending [r, g, b] color
            t: Interpolation factor (0 to 1)
            
        Returns:
            Interpolated [r, g, b] color
        """
        t = np.clip(t, 0, 1)
        return [
            color1[0] + t * (color2[0] - color1[0]),
            color1[1] + t * (color2[1] - color1[1]),
            color1[2] + t * (color2[2] - color1[2])
        ]
    
    def clear_cache(self):
        """Clear all cached lookup tables and colors for memory management."""
        self._lut_cache.clear()
        self._color_cache.clear()
        print("ColorManager cache cleared")
