# STRATOS - Space Trajectory Radiation Analysis Toolkit for Orbital Simulation

## Overview

STRATOS is a comprehensive 3D visualization and analysis application for space radiation environments and orbital trajectories. Developed for Los Alamos National Laboratory, it provides real-time visualization of particle flux fields in Earth's magnetosphere along with detailed orbital trajectory analysis capabilities.

## Features

### Core Capabilities

- **Multi-Format VTK Data Loading**: Support for various VTK formats including:
  - XML Structured Grid (.vts)
  - XML Unstructured Grid (.vtu)
  - XML PolyData (.vtp)
  - XML Image Data (.vti)
  - Legacy VTK (.vtk)

- **Orbital Trajectory Visualization**: 
  - CSV-based orbital data import
  - Real-time satellite position tracking
  - Animated orbital paths with customizable trails
  - Multi-satellite support with independent color coding

- **Earth Visualization**:
  - Realistic Earth sphere with texture mapping
  - Latitude/longitude grid with coordinate labels
  - Adjustable opacity for Earth representation
  - Support for both texture files and procedural textures

### Visualization Modes

1. **Point Cloud**: Density-adjustable point representation with size control
2. **Volume Rendering**: 3D volumetric visualization with threshold control
3. **Isosurfaces**: Multiple surface level visualization
4. **Wireframe**: Wire mesh representation
5. **Slice Planes**: Cross-sectional views along X, Y, or Z axes
6. **Surface with Edges**: Combined surface and edge visualization

### Analysis Tools

#### Flux Analysis
- Real-time flux calculation at orbital points
- Cross-sectional area integration
- Peak exposure region identification
- Total radiation dose calculations
- Time-integrated flux analysis

#### Spectrum Analysis
- Energy spectrum generation
- Pitch angle distribution analysis
- Phase space plotting
- Customizable energy bins (10 keV - 10 MeV range)

#### Trajectory Statistics
- Orbital period detection
- Apogee/perigee identification
- Altitude profiling
- Ground track visualization
- Velocity analysis

### Animation & Controls

- **Animation System**:
  - Play/Pause/Stop controls
  - Adjustable animation speed (10-1000ms intervals)
  - Time slider for manual navigation
  - Satellite trail visualization with fading effects

- **View Controls**:
  - Quick axis-aligned views (X, Y, Z buttons)
  - Interactive 3D navigation
  - Orientation widget for spatial reference
  - Automatic camera distance optimization

## Installation

### Prerequisites

- Python 3.8 or higher
- PyQt6
- VTK (Visualization Toolkit)
- NumPy
- Pandas

### Setup

```bash
# Clone the repository
git clone [repository-url]
cd flux_visualizer

# Install dependencies
pip install PyQt6 vtk numpy pandas

# Run the application
python app.py
```

## Usage

### Loading Data

1. **Flux Data**: Click "Load Flux File" and select VTK format files containing radiation field data
2. **Orbital Data**: Click "Load Orbital File" and select CSV files with trajectory data

### CSV Format for Orbital Data

Required columns:
- `time`: Time in hours
- `x`, `y`, `z`: Position in km (Earth-centered coordinates)

Optional columns:
- `vx`, `vy`, `vz`: Velocity components in km/s

### Interface Components

#### Main Visualization Window
- 3D rendering viewport with Earth and data visualization
- Overlay buttons for quick axis views
- Interactive camera controls (rotate, zoom, pan)

#### Control Panel Sections

1. **Data Loading Panel**
   - Load flux and orbital files
   - File management with checkboxes for visibility
   - Particle type selection for flux data
   - Color selection for orbital paths

2. **Animation Controls**
   - Play/Pause/Stop buttons
   - Time slider and display
   - Animation speed adjustment

3. **Field Visualization Panel**
   - Visualization mode selection
   - Colormap options (Viridis, Plasma, Cool to Warm, etc.)
   - Linear/Logarithmic scale toggle
   - Opacity control
   - Mode-specific settings

4. **Analysis Panel**
   - Analysis type selection
   - Cross-section and integration settings
   - Access to analysis windows

#### Earth Controls Bar
- Earth opacity slider (0-100%)
- Lat/Long grid toggle
- Orbital path visibility toggle
- Satellite trail toggle
- Satellite size adjustment (100-2000 km)

## Configuration

The application's behavior can be customized through the `config.py` file:

### Key Configuration Parameters

- **Earth Parameters**: Radius, resolution, default colors
- **Visualization Defaults**: Point density, size ranges, opacity settings
- **Animation Parameters**: Speed ranges, trail length
- **Performance Settings**: Debounce delays, memory limits
- **File I/O**: Supported formats, texture search patterns

## Architecture

### Module Structure

```
flux_visualizer/
├── app.py                 # Main application entry point
├── config.py             # Configuration constants
├── core/                 # Core data structures
│   ├── orbital_data.py   # OrbitalPoint and OrbitalPath classes
├── data_io/              # Data loading modules
│   ├── vtk_loader.py     # VTK file handling
│   └── orbital_loader.py # CSV trajectory loading
├── scene/                # 3D scene components
│   ├── earth_renderer.py # Earth visualization
│   └── orbital_renderer.py # Satellite and path rendering
├── ui/                   # User interface
│   ├── main_window.py    # Main application window
│   ├── controls/         # Control panels
│   └── widgets/          # Reusable UI widgets
├── visualization/        # Visual effects
│   └── color_manager.py  # Color mapping and LUTs
└── analysis/            # Analysis modules
    └── flux_analyzer.py  # Flux calculations
```

### Key Classes

- **ElectronFluxVisualizerApp**: Main application window and coordinator
- **OrbitalPoint/OrbitalPath**: Trajectory data structures
- **EarthRenderer**: Handles Earth sphere, grid, and labels
- **OrbitalRenderer/SatelliteRenderer**: Manages orbital paths and satellites
- **FluxAnalyzer**: Performs radiation flux calculations
- **ColorManager**: Handles all color mapping operations

## Performance Considerations

- **Point Cloud Mode**: Optimized for datasets up to 10,000 points
- **Volume Rendering**: Automatic downsampling for large datasets
- **Memory Management**: Configurable memory limits (default 256MB)
- **Caching**: Flux calculations cached for performance
- **Debouncing**: UI updates debounced to prevent lag

## Known Limitations

- Volume rendering limited by GPU memory
- Large VTK files (>1GB) may require downsampling
- Maximum of ~20 simultaneous orbital trajectories recommended
- Trail visualization performance depends on trail length setting

## File Format Support

### VTK Formats
- Structured and unstructured grids
- PolyData and ImageData
- Both XML and legacy VTK formats
- Automatic format detection

### Texture Files
Searches for Earth textures in order:
- resources/textures/ directory
- Application directory
- Falls back to procedural texture generation

Supported image formats: JPG, PNG

## Troubleshooting

### Common Issues

1. **"No scalar data found"**: Ensure VTK file contains point data arrays
2. **Black volume rendering**: Adjust threshold and opacity settings
3. **Performance issues**: Reduce point density or disable trails
4. **Memory errors**: Lower volume rendering quality or use slice planes

### Debug Mode

Enable debug output in `config.py`:
```python
DEBUG_ENABLED = True
VERBOSE_VTK_LOADING = True
```

## Future Enhancements

- [ ] Multi-threading for large dataset processing
- [ ] Export capabilities for analysis results
- [ ] Custom colormap editor
- [ ] Magnetic field line visualization
- [ ] Particle trajectory tracing
- [ ] Integration with space weather databases

## License

Developed for Los Alamos National Laboratory

## Authors

STRATOS Development Team

## Version

1.0 - Initial Release