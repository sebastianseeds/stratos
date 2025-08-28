# STRATOS
**Space Trajectory Radiation Analysis Toolkit for Orbital Simulation**

A comprehensive Python toolkit for visualizing and analyzing charged particle radiation exposure along spacecraft trajectories through Earth's magnetosphere.

## Project Overview

STRATOS is an advanced 3D visualization and analysis platform designed for:
- Real-time visualization of electron flux in Earth's radiation belts
- Spacecraft trajectory analysis through radiation fields
- Quantitative radiation exposure calculations
- Multi-scale particle flux visualization (electron, proton, heavy ion - planned)

### Current Capabilities
- VTK-based 3D visualization with multiple rendering modes
- CSV-based orbital trajectory loading and animation
- Real-time flux calculations at spacecraft positions
- Energy spectrum analysis
- Interactive Earth model with coordinate grid
- Multiple colormaps with linear/logarithmic scaling

### Planned Features
- Time-dependent radiation fields
- Directional flux analysis
- Multi-spacecraft mission planning
- Proton and heavy ion flux modeling
- Dynamic radiation sources (CME, solar events)

## Project Structure

```
stratos/
├── setup_environment.sh         # Environment setup script
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── LICENSE                     # License information
│
├── flux_visualizer/            # Main application (refactored modular structure)
│   ├── app.py                 # Main application entry point
│   ├── config.py              # Configuration and constants
│   │
│   ├── core/                  # Core data structures
│   │   ├── __init__.py
│   │   └── orbital_data.py   # OrbitalPoint and OrbitalPath classes
│   │
│   ├── data_io/               # Data loading modules
│   │   ├── __init__.py
│   │   ├── vtk_loader.py     # VTK file loading and validation
│   │   └── orbital_loader.py # CSV trajectory loading
│   │
│   ├── visualization/         # Visualization components
│   │   ├── __init__.py
│   │   ├── color_manager.py  # Colormap and LUT management
│   │   └── renderers/        # Visualization mode renderers (WIP)
│   │       ├── __init__.py
│   │       ├── point_cloud_renderer.py
│   │       ├── volume_renderer.py
│   │       └── ...
│   │
│   ├── scene/                 # 3D scene components
│   │   ├── __init__.py
│   │   ├── earth_renderer.py # Earth visualization with grid
│   │   ├── satellite_renderer.py (WIP)
│   │   ├── starfield_renderer.py (WIP)
│   │   └── orbital_path_renderer.py (WIP)
│   │
│   ├── analysis/              # Analysis modules (WIP)
│   │   ├── __init__.py
│   │   ├── flux_analyzer.py  # Flux calculations
│   │   ├── spectrum_analyzer.py
│   │   └── trajectory_analyzer.py
│   │
│   ├── ui/                    # User interface components (WIP)
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── controls/
│   │   └── windows/
│   │
│   ├── resources/             # Static resources
│   │   ├── textures/         # Earth and starfield textures
│   │   └── icons/            # UI icons
│   │
│   ├── data/                  # Data directory
│   │   ├── vtk/              # VTK field data files
│   │   ├── orbits/           # CSV orbital trajectories
│   │   └── sample/           # Sample data for testing
│   │
│   ├── output/                # Output directory
│   │   ├── screenshots/
│   │   ├── animations/
│   │   └── logs/
│   │
│   ├── test_integration.py   # Integration tests
│   └── flux_visualizer.py    # LEGACY: Original monolithic file (for reference)
│
└── utilities/                 # Development and debugging utilities
    ├── __init__.py
    ├── animation_debug_tool.py    # Standalone animation verifier
    └── generate_sample_data.py    # Synthetic Van Allen belt generator
```

## Getting Started

### Prerequisites
- Python 3.9+
- VTK 9.0+
- PyQt6
- NumPy, Pandas, Matplotlib

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stratos.git
cd stratos
```

2. **Set up the environment**
```bash
# Using the setup script
chmod +x setup_environment.sh
./setup_environment.sh

# Or manually
python -m venv stratos_env
source stratos_env/bin/activate  # On Windows: stratos_env\Scripts\activate
pip install -r requirements.txt
```

3. **Generate sample data (optional)**
```bash
python utilities/generate_sample_data.py
```

4. **Run STRATOS**
```bash
cd flux_visualizer
python app.py
```

## Data Formats

### VTK Field Data
STRATOS supports multiple VTK formats:
- `.vts` - Structured Grid (recommended)
- `.vtu` - Unstructured Grid
- `.vtp` - PolyData
- `.vti` - Image Data
- `.vtk` - Legacy VTK

Field data should contain scalar values representing particle flux (particles/cm²/s).

### Orbital Trajectories
CSV format with columns:
- `time` (hours) - Required
- `x`, `y`, `z` (km, Earth-centered) - Required
- `vx`, `vy`, `vz` (km/s) - Optional

Example:
```csv
time,x,y,z,vx,vy,vz
0.0,7000.0,0.0,0.0,0.0,7.5,0.0
0.1,6995.0,750.0,0.0,-0.75,7.49,0.0
...
```

## Development Status

### Completed Refactoring
- [x] Configuration management (`config.py`)
- [x] Color/LUT management (`visualization/color_manager.py`)
- [x] VTK data loading (`data_io/vtk_loader.py`)
- [x] Orbital data loading (`data_io/orbital_loader.py`)
- [x] Core data structures (`core/orbital_data.py`)
- [x] Earth rendering (`scene/earth_renderer.py`)
- [x] Basic GUI integration (`app.py`)

### In Progress
- [ ] Satellite and trail rendering
- [ ] Starfield background
- [ ] Visualization mode renderers (point cloud, volume, etc.)
- [ ] Flux analysis module
- [ ] Analysis windows (spectrum, flux vs time)

### TODO
- [ ] Complete UI component extraction
- [ ] Implement remaining visualization modes
- [ ] Add analysis result windows
- [ ] Performance optimizations
- [ ] Documentation and tutorials
- [ ] Unit tests for all modules

## Testing

### Run Integration Tests
```bash
cd flux_visualizer
python test_integration.py
```

### Test Animation System
```bash
python utilities/animation_debug_tool.py
```

## Utilities

### Generate Sample Data
Creates synthetic Van Allen belt data for testing:
```bash
python utilities/generate_sample_data.py --output flux_visualizer/data/sample/
```

### Animation Debug Tool
Verify Qt animation parity:
```bash
python utilities/animation_debug_tool.py --orbital-file flux_visualizer/data/orbits/test.csv
```

## Performance Considerations

- **Large VTK files**: Use structured grids (.vts) when possible
- **Point cloud density**: Default 5000 points, adjustable for performance
- **Volume rendering**: Automatically downsampled to 30³ grid
- **Animation**: Default 100ms update interval

## Contributing

STRATOS is actively being refactored from a monolithic structure to a modular architecture. Key principles:

1. **Separation of Concerns**: Visualization vs Analysis
2. **Testability**: Each module should be independently testable
3. **Configuration**: Use `config.py` for all constants
4. **Type Hints**: Add type hints to all new code
5. **Documentation**: Document all public methods

## Architecture Notes

### Refactoring Strategy
We're transitioning from a single 3000+ line file (`flux_visualizer.py`) to a modular structure:

1. **Phase 1**: Extract independent utilities (Complete)
   - Config, ColorManager, VTKLoader, OrbitalLoader

2. **Phase 2**: Scene components (In Progress)
   - EarthRenderer, SatelliteRenderer, StarfieldRenderer

3. **Phase 3**: Visualization renderers
   - One class per visualization mode

4. **Phase 4**: Analysis modules
   - Separate analysis from visualization

5. **Phase 5**: UI components
   - Modular panels and windows

### Design Patterns
- **Factory Pattern**: Renderer creation
- **Observer Pattern**: Animation updates
- **Strategy Pattern**: Visualization modes
- **Singleton**: Configuration

## Authors

- Dr. Sebastian Seeds - Initial development

## Acknowledgments

- NASA for Van Allen belt models
- VTK community for visualization framework
- PyQt team for GUI framework

---

**Note**: STRATOS is under active development. The modular refactoring is ongoing to improve maintainability and extensibility while preserving all original functionality from `flux_visualizer.py`.