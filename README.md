# STRATOS - Space Trajectory Radiation Analysis Toolkit for Orbital Simulation

## Overview

STRATOS is a comprehensive 3D visualization and analysis application for space radiation environments and orbital trajectories. Developed for Los Alamos National Laboratory, it provides real-time visualization of particle flux fields in Earth's magnetosphere along with detailed orbital trajectory analysis capabilities.

## Features

### Core Capabilities

- **Multi-Format VTK Data Loading**: Support for various VTK formats including:
  - XML Structured Grid (.vts) - including time-dependent datasets
  - XML Unstructured Grid (.vtu)
  - XML PolyData (.vtp)
  - XML Image Data (.vti)
  - XML MultiBlock (.vtm) - for time series data
  - Legacy VTK (.vtk)

- **Time-Dependent Data Support**:
  - Dynamic flux field animation with time slider controls
  - Automatic detection of time-dependent VTS and VTM files
  - Consistent flux scaling across all time steps
  - Time synchronization between flux fields and orbital trajectories

- **Orbital Trajectory Visualization**: 
  - CSV-based orbital data import (.orb files)
  - Real-time satellite position tracking
  - Animated orbital paths with customizable trails
  - Multi-satellite support with independent color coding
  - Satellite size and opacity controls

- **Earth Visualization**:
  - Realistic Earth sphere with Blue Marble texture mapping
  - Latitude/longitude grid with coordinate labels  
  - Adjustable opacity for Earth representation
  - Starfield background with realistic star textures

### Visualization Modes

1. **Point Cloud**: Intelligent density sampling with size control and opacity
2. **Isosurfaces**: Single or multiple surface level visualization with customizable levels
3. **Slice Planes**: 
   - Single slice mode (X, Y, or Z axis)
   - Three-plane mode showing all orthogonal cross-sections
   - Real-time position feedback in kilometers

### Analysis Tools

STRATOS provides three main analysis modes accessible through the Analysis Panel:

#### Flux Analysis
- Flux vs Time plotting along orbital trajectories
- Real-time flux calculation at satellite positions
- Dynamic updating with animation controls

#### Spectrum Analysis  
- **Energy Spectrum**: Interactive energy distribution plots with customizable energy bins (10 keV - 10 MeV range)
- **Phase Space Plots**: Position vs velocity visualization
- **Pitch Angle Distribution**: Particle angular distribution analysis
- **Multi-satellite support**: Analyze different satellites independently
- **Multiple flux sources**: Combined analysis of different particle types (electrons, protons, etc.)

#### Orbital Analysis
- **Dose Calculator**: Combined radiation dose from multiple flux sources with particle-type labeling
- **Orbital Statistics**: Period detection, apogee/perigee identification
- **Ground Track Visualization**: Satellite ground path plotting  
- **Altitude Profiles**: Altitude vs time analysis
- **Cross-section Configuration**: Adjustable cross-sectional areas for each satellite

### Animation & Controls

- **Animation System**:
  - Play/Pause/Stop controls with dynamic time display
  - Adjustable animation speed (10-1000ms intervals)
  - Time slider for manual navigation through datasets
  - Synchronized animation of flux fields and orbital trajectories
  - Satellite trail visualization with fading effects

- **View Controls**:
  - Quick axis-aligned views (X, Y, Z buttons) with smooth transitions
  - Interactive 3D navigation (rotate, zoom, pan)
  - Orbital controls for focused satellite tracking
  - Camera reset and automatic distance optimization

## Installation

### Prerequisites

- Python 3.8 or higher
- PyQt6
- VTK (Visualization Toolkit) 9.0+
- NumPy
- Pandas  
- Matplotlib (for analysis plots)

### Setup

```bash
# Clone the repository
git clone [repository-url]
cd stratos

# Install dependencies
pip install PyQt6 vtk numpy pandas matplotlib

# Run the application
cd flux_visualizer
python app.py
```

## Usage

### Loading Data

1. **Flux Data**: Click "Load Flux Files" to select VTK format files containing radiation field data
   - Supports both static and time-dependent datasets
   - Multiple flux files can be loaded simultaneously with particle type selection
   - Files are automatically detected as time-dependent (VTS/VTM) or static (VTK/VTS)

2. **Orbital Data**: Click "Load Orbital Files" to select .orb files with trajectory data  
   - Multiple orbital trajectories supported
   - Each trajectory gets independent color and visibility controls
   - Satellite size and trail settings configurable per trajectory

### Data Formats

#### Orbital Data (.orb files)
CSV format with required columns:
- `time`: Time in hours
- `x`, `y`, `z`: Position in km (Earth-centered coordinates)

Optional columns:
- `vx`, `vy`, `vz`: Velocity components in km/s

#### Flux Data
- **Static VTK/VTS**: Single-time flux field data
- **Time-dependent VTS**: Single file with embedded time series arrays (e.g., `electron_flux_t000`, `electron_flux_t001`)
- **MultiBlock VTM**: Multiple VTS files referenced in XML structure for time series

### Interface Components

#### Main Visualization Window
- 3D rendering viewport with Earth, flux fields, and orbital trajectories
- Overlay buttons for quick axis views (X, Y, Z) and camera controls
- Interactive VTK-based camera controls (rotate, zoom, pan)
- Dynamic scalar bar showing flux scales and particle types

#### Control Panel Sections

1. **Data Panel**
   - Load flux and orbital files with file browser dialogs
   - File management with visibility checkboxes for each loaded file
   - Particle type selection for flux data (electrons, protons, alpha, etc.)
   - Color and remove buttons for orbital trajectories
   - Status indicators showing loaded file information

2. **Animation Panel**
   - Play/Pause/Stop buttons with dynamic time display
   - Time slider with current time indicator
   - Animation speed adjustment (10-1000ms intervals)
   - Works with both orbital trajectories and time-dependent flux data

3. **Field Visualization Panel**
   - Visualization mode selection (Point Cloud, Isosurfaces, Slice Planes)
   - Colormap options (Viridis, Plasma, Inferno, Cool to Warm, etc.)  
   - Linear/Logarithmic scale toggle
   - Opacity control (0-100%)
   - Flux range controls (Min/Max thresholds with "Auto" detection)
   - Mode-specific settings (isosurface levels, slice positions)

4. **Analysis Panel**
   - Analysis type selection (Flux Analysis, Spectrum Analysis, Orbital Analysis)
   - Dynamic controls based on selected mode
   - Satellite selection for analysis
   - Cross-section configuration for dose calculations
   - Energy range and bin settings for spectrum analysis

#### Earth Controls Bar
- Earth opacity slider (0-100%)
- Latitude/Longitude grid toggle with coordinate labels
- Orbital path visibility toggle
- Satellite trail toggle with fading effects  
- Satellite size adjustment (100-2000 km radius)

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
stratos/
├── flux_visualizer/
│   ├── app.py                    # Main application entry point
│   ├── config.py                # Configuration constants and parameters
│   ├── __init__.py              # Package initialization
│   │
│   ├── core/                    # Core data structures
│   │   ├── __init__.py
│   │   └── orbital_data.py      # OrbitalPoint and OrbitalPath classes
│   │
│   ├── data_io/                 # Data loading modules
│   │   ├── __init__.py
│   │   ├── vtk_loader.py        # VTK file handling (VTS, VTU, VTK, VTM)
│   │   └── orbital_loader.py    # CSV trajectory loading (.orb files)
│   │
│   ├── scene/                   # 3D scene components
│   │   ├── __init__.py
│   │   ├── earth_renderer.py    # Earth sphere, textures, grids
│   │   ├── starfield_renderer.py # Background starfield  
│   │   ├── orbital_renderer.py  # Satellite paths and animations
│   │   └── flux_field_renderer.py # Flux field visualization
│   │
│   ├── ui/                      # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py       # Main application window coordinator
│   │   │
│   │   ├── controls/            # Control panels
│   │   │   ├── __init__.py
│   │   │   ├── data_panel.py    # File loading and management
│   │   │   ├── animation_panel.py # Time controls and animation  
│   │   │   ├── visualization_panel.py # Flux visualization settings
│   │   │   ├── analysis_panel.py # Analysis mode controls
│   │   │   └── earth_controls.py # Earth and camera controls
│   │   │
│   │   ├── widgets/             # Reusable UI components
│   │   │   ├── __init__.py
│   │   │   ├── file_widget.py   # Individual file management widgets
│   │   │   └── orbital_file_widget.py # Orbital trajectory widgets
│   │   │
│   │   └── windows/             # Analysis windows
│   │       ├── __init__.py
│   │       ├── spectrum_window.py # Energy/pitch angle/phase space plots
│   │       └── dose_window.py   # Dose calculation plots
│   │
│   ├── visualization/           # Visual effects and rendering
│   │   ├── __init__.py
│   │   └── color_manager.py     # Color mapping, LUTs, and scales
│   │
│   ├── analysis/               # Analysis modules
│   │   ├── __init__.py
│   │   └── flux_analyzer.py    # Flux calculations and dose analysis  
│   │
│   └── textures/               # Application textures
│       ├── .gitkeep
│       ├── blue_marble.jpg     # Earth texture
│       └── starfield.jpg       # Background stars
│
├── data/                       # Default data directories
│   ├── flux/                   # Flux field data
│   │   ├── static/             # Static VTK files
│   │   │   └── .gitkeep
│   │   └── time_dependent/     # Time-dependent VTS/VTM files
│   │       └── .gitkeep
│   └── orbits/                 # Orbital trajectory data
│       └── .gitkeep
│
├── utilities/                  # Development and testing tools
│   ├── generate_*.py           # Various data generation scripts
│   ├── verify_*.py            # File format verification tools
│   └── animation_debug_tool.py # Performance testing
│
└── README.md                   # This file
```

### Key Classes

- **ElectronFluxVisualizerApp**: Main application window and coordinator
- **OrbitalPoint/OrbitalPath**: Trajectory data structures with time indexing
- **VTKDataLoader**: Handles all VTK format loading with time-dependency detection
- **EarthRenderer**: Earth sphere, texture mapping, coordinate grids
- **FluxFieldRenderer**: Flux field visualization with multiple modes
- **OrbitalRenderer**: Satellite paths, animations, and trail effects
- **ColorManager**: Color mapping, lookup tables, and scale management
- **FluxAnalyzer**: Radiation flux and dose calculations

## Performance Considerations

- **Point Cloud Mode**: Intelligent density sampling for large datasets (default 10,000 points)
- **Time-Dependent Data**: Global scalar range calculation prevents scale flickering during animation
- **Memory Management**: VTK handles large datasets efficiently with streaming
- **Flux Range Optimization**: User-defined min/max limits prevent extreme scale values
- **UI Responsiveness**: Opacity and settings changes applied in real-time during animation

## Known Limitations

- **Time-dependent datasets**: Large VTS files with many time steps may require longer loading times
- **Multiple flux files**: Performance scales with number of simultaneously loaded flux files
- **Analysis windows**: Energy spectrum and dose calculation windows are independent and don't auto-update during animation
- **Orbital trajectory limits**: Recommended maximum of ~20 simultaneous trajectories for optimal performance

## File Format Support

### VTK Formats
- **Static formats**: VTK (legacy), VTS, VTU, VTP, VTI (XML formats)
- **Time-dependent formats**: 
  - Single VTS files with embedded time arrays (`particle_flux_t000`, `particle_flux_t001`, etc.)
  - VTM MultiBlock files referencing multiple VTS files
- **Automatic detection**: File format and time-dependency automatically detected
- **Particle types**: Supports multiple particle types (electrons, protons, alpha particles, etc.)

### Texture Files
Application searches for textures in:
- `flux_visualizer/textures/` directory (primary)
- Contains Blue Marble Earth texture and starfield background
- JPG and PNG formats supported

## Troubleshooting

### Common Issues

1. **"No scalar data found"**: Ensure VTK file contains point data arrays with numerical flux values
2. **Invisible flux visualization**: Check opacity settings and flux range controls (try "Auto" for max range)
3. **Animation not working**: Verify time-dependent data is loaded; check that orbital and flux data have compatible time ranges
4. **Analysis windows show no data**: Ensure satellite trajectory is selected and flux files are checked/visible
5. **Flickering flux scale**: This has been fixed with global range calculation across all time steps

### Debug Tools

- **VTS File Verification**: Use `utilities/verify_vts_file.py` to check time-dependency and data structure
- **VTM File Verification**: Use `utilities/verify_vtm_file.py` for MultiBlock file validation
- **Console Output**: Application prints detailed loading and time-step information to console

## Recent Enhancements

- ✅ **Fixed flux scale fluctuation**: Global range calculation prevents scale changes during time-dependent animation
- ✅ **Fixed opacity controls**: Dynamic flux now properly responds to opacity changes during animation  
- ✅ **Multi-flux analysis**: Dose calculator and energy spectrum now combine multiple selected flux sources
- ✅ **Improved slice visualization**: Three-plane mode shows all orthogonal cross-sections with position feedback
- ✅ **Enhanced scalar bar**: Dynamic titles showing particle types and mixed flux indicators

## Current Development Status

**STRATOS v2.0** - Enhanced time-dependent visualization with multiple flux source support

## License

Developed for Los Alamos National Laboratory

## Authors

STRATOS Development Team

---

*Note: The files `flux_visualizer.py`, `main_integrated.py`, and `test_integration.py` are legacy/outdated and not part of the current application architecture.*