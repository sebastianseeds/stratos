"""
Configuration file for Flux Orbital Visualizer
Contains all constants, default values, and configuration parameters
"""

class Config:
    """Central configuration for the application"""
    
    # ============================================
    # APPLICATION METADATA
    # ============================================
    APP_NAME = "Flux Orbital Visualizer"
    APP_VERSION = "1.0"
    ORGANIZATION = "Los Alamos National Laboratory"
    
    # ============================================
    # EARTH PARAMETERS
    # ============================================
    EARTH_RADIUS_KM = 6371.0
    EARTH_SPHERE_RESOLUTION_THETA = 120
    EARTH_SPHERE_RESOLUTION_PHI = 120
    EARTH_DEFAULT_OPACITY = 0.8
    EARTH_DEFAULT_COLOR = (0.3, 0.3, 0.8)  # Blue-ish
    
    # Earth material properties
    EARTH_AMBIENT = 0.4
    EARTH_DIFFUSE = 0.8
    EARTH_SPECULAR = 0.05
    EARTH_SPECULAR_POWER = 10
    
    # Earth grid properties
    LATITUDE_LINE_COLOR = (0.7, 0.7, 0.7)  # Light gray
    LATITUDE_LINE_WIDTH = 1.5
    LATITUDE_LINE_OPACITY_FACTOR = 0.8  # 80% of Earth opacity
    
    EQUATOR_COLOR = (1.0, 0.6, 0.0)  # Orange
    EQUATOR_LINE_WIDTH = 2.5
    EQUATOR_OPACITY_FACTOR = 0.9  # 90% of Earth opacity
    
    PRIME_MERIDIAN_COLOR = (1.0, 0.6, 0.0)  # Orange
    PRIME_MERIDIAN_LINE_WIDTH = 2.5
    
    # Grid parameters
    LATITUDE_GRID_SPACING = 15  # degrees
    LONGITUDE_GRID_SPACING = 30  # degrees
    GRID_LABEL_OFFSET = 500  # km above surface
    GRID_LABEL_COLOR = (1.0, 1.0, 0.8)  # Light yellow
    GRID_LABEL_OPACITY = 0.8
    GRID_LABEL_SCALE = (300, 300, 300)
    
    # Latitude/longitude circles
    LATITUDE_CIRCLE_SEGMENTS = 60  # Smoothness of circles
    LONGITUDE_MERIDIAN_POINTS = 120  # Points per meridian
    
    # ============================================
    # STARFIELD PARAMETERS
    # ============================================
    STARFIELD_RADIUS_KM = 500000.0  # 500,000 km sphere
    STARFIELD_RESOLUTION = 120
    STARFIELD_NUM_STARS = 8000
    STARFIELD_TEXTURE_WIDTH = 2048
    STARFIELD_TEXTURE_HEIGHT = 1024
    STARFIELD_BACKGROUND_COLOR = (5, 5, 15)  # Very dark blue-black
    
    # ============================================
    # VISUALIZATION DEFAULTS
    # ============================================
    
    # Point Cloud
    DEFAULT_POINT_DENSITY = 5000
    MIN_POINT_DENSITY = 500
    MAX_POINT_DENSITY = 10000
    DEFAULT_POINT_SIZE = 400  # km radius (VTK coordinates in km)
    MIN_POINT_SIZE = 100      # km radius  
    MAX_POINT_SIZE = 800      # km radius
    POINT_SPHERE_THETA_RESOLUTION = 8  # Reduced for performance
    POINT_SPHERE_PHI_RESOLUTION = 8
    JITTER_RADIUS = 300.0  # km - spatial jitter to break up grid artifacts
    
    # Volume Rendering
    VOLUME_MAX_POINTS_WARNING = 500000
    VOLUME_DOWNSAMPLE_DIMENSION = 30  # 30x30x30 grid
    VOLUME_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256MB
    VOLUME_MAX_MEMORY_FRACTION = 0.5  # Use max 50% of GPU memory
    
    # Isosurfaces
    DEFAULT_NUM_ISOSURFACES = 3
    ISOSURFACE_MIN_PERCENT = 30  # Start at 30% of max
    ISOSURFACE_MAX_PERCENT = 90  # End at 90% of max
    
    # Wireframe
    DEFAULT_WIREFRAME_LINE_WIDTH = 2.0
    DEFAULT_WIREFRAME_COLOR = (0.9, 0.9, 0.2)  # Yellow
    DEFAULT_ISOSURFACE_LEVEL = 50  # 50th percentile
    WIREFRAME_MULTI_ISO_DEFAULTS = [20, 40, 60, 80]  # Percentile levels
    
    # Slice Planes
    DEFAULT_SLICE_POSITION = 50  # Center (50%)
    DEFAULT_SLICE_AXIS = "Z-Axis (XY Plane)"
    
    # ============================================
    # FLUX PARAMETERS
    # ============================================
    DEFAULT_FLUX_CUTOFF = 1e-8  # particles/cm²/s
    MIN_FLUX_CUTOFF = 1e-15
    MAX_FLUX_CUTOFF = 1e15
    
    # Flux threshold percentiles for volume rendering
    VOLUME_PERCENTILE_MIN = 1  # 1st percentile
    VOLUME_PERCENTILE_TRANSITION = 90  # Transition point at 90th percentile
    VOLUME_PERCENTILE_MAX = 99.9  # 99.9th percentile
    
    # ============================================
    # ANIMATION PARAMETERS
    # ============================================
    DEFAULT_ANIMATION_SPEED_MS = 100
    MIN_ANIMATION_SPEED_MS = 10
    MAX_ANIMATION_SPEED_MS = 1000
    MAX_TRAIL_LENGTH = 15
    
    # Trail rendering
    TRAIL_MIN_RADIUS = 50.0  # km
    TRAIL_MAX_RADIUS = 300.0  # km
    TRAIL_MIN_OPACITY = 0.1
    TRAIL_MAX_OPACITY = 0.9
    TRAIL_TUBE_SIDES = 8
    TRAIL_COLOR = (1.0, 1.0, 1.0)  # White
    
    # ============================================
    # SATELLITE PARAMETERS
    # ============================================
    DEFAULT_CROSS_SECTION_RADIUS = 1.0  # meters
    MIN_CROSS_SECTION = 0.1
    MAX_CROSS_SECTION = 100.0
    
    SATELLITE_SPHERE_RESOLUTION = 24
    SATELLITE_BASE_RADIUS_MULTIPLIER = 100.0  # Multiply cross-section by this for visibility
    SATELLITE_MIN_RADIUS = 500.0  # Minimum radius in km for visibility
    
    # Satellite material properties
    SATELLITE_AMBIENT = 0.9
    SATELLITE_DIFFUSE = 0.8
    SATELLITE_SPECULAR = 0.1
    SATELLITE_OPACITY = 1.0
    
    # Satellite colors (for random selection)
    SATELLITE_COLORS = [
        [1.0, 0.2, 0.2],  # Bright red
        [0.2, 1.0, 0.2],  # Bright green
        [1.0, 0.2, 1.0],  # Bright magenta
        [0.2, 0.8, 1.0],  # Bright cyan
        [1.0, 0.8, 0.2],  # Bright orange
        [0.8, 0.2, 1.0],  # Bright purple
        [1.0, 0.5, 0.8],  # Bright pink
        [0.2, 1.0, 0.8],  # Bright teal
    ]
    
    # ============================================
    # ORBITAL PATH PARAMETERS
    # ============================================
    PATH_LINE_COLOR = (0.9, 0.9, 0.2)  # Bright yellow
    PATH_LINE_WIDTH = 3.0
    PATH_LINE_OPACITY = 0.8
    
    # ============================================
    # UI LAYOUT PARAMETERS
    # ============================================
    
    # Main window
    MAIN_WINDOW_WIDTH = 1600
    MAIN_WINDOW_HEIGHT = 900
    MAIN_WINDOW_START_X = 100
    MAIN_WINDOW_START_Y = 100
    
    # Control panel - more flexible for small screens
    CONTROL_PANEL_MIN_WIDTH = 280  # Reduced from 450 for small monitors
    CONTROL_PANEL_MAX_WIDTH = 450
    
    # Splitter sizes
    VTK_WIDGET_DEFAULT_WIDTH = 700
    CONTROL_PANEL_DEFAULT_WIDTH = 900
    
    # Plot windows
    SLICE_WINDOW_WIDTH = 800
    SLICE_WINDOW_HEIGHT = 600
    SPECTRUM_WINDOW_WIDTH = 700
    SPECTRUM_WINDOW_HEIGHT = 500
    FLUX_TIME_WINDOW_WIDTH = 600
    FLUX_TIME_WINDOW_HEIGHT = 400

    # Camera Configuration
    CAMERA_DEFAULT_DISTANCE = 30000  # Default distance in km when no data is loaded
    CAMERA_DEFAULT_POSITION = (20000, 20000, 10000)  # Will be scaled based on data
    CAMERA_DEFAULT_FOCAL_POINT = (0, 0, 0)
    CAMERA_DEFAULT_VIEW_UP = (0, 0, 1)
    
    # ============================================
    # VTK RENDERING PARAMETERS
    # ============================================
    
    # Renderer
    RENDERER_BACKGROUND_COLOR = (0.05, 0.05, 0.15)  # Dark blue background
    
    # Camera defaults
    CAMERA_DEFAULT_POSITION = (20000, 20000, 10000)  # km
    CAMERA_DEFAULT_FOCAL_POINT = (0, 0, 0)
    CAMERA_DEFAULT_VIEW_UP = (0, 0, 1)
    CAMERA_DEFAULT_ZOOM = 0.8  # Zoom out a bit for safety margin
    CAMERA_DISTANCE_MULTIPLIER = 1.5  # For auto-zoom
    
    # Scalar bar
    SCALAR_BAR_POSITION = (0.85, 0.1)
    SCALAR_BAR_WIDTH = 0.12
    SCALAR_BAR_HEIGHT = 0.8
    SCALAR_BAR_NUM_LABELS = 6
    SCALAR_BAR_LABEL_COLOR = (1, 1, 1)
    SCALAR_BAR_TITLE_FONT_SIZE = 10
    SCALAR_BAR_LABEL_FONT_SIZE = 8
    
    # Axes
    AXES_LENGTH = (5000, 5000, 5000)  # km
    AXES_LABEL_FONT_SIZE = 12
    AXES_LABEL_COLOR = (1, 1, 1)
    ORIENTATION_WIDGET_SIZE = 0.3  # 30% of viewport
    ORIENTATION_WIDGET_POSITION = (0.0, 0.0)  # Bottom-left corner
    
    # ============================================
    # COLOR MAPS
    # ============================================
    AVAILABLE_COLORMAPS = [
        "Blue to Red",
        "Viridis",
        "Plasma",
        "Cool to Warm",
        "Rainbow",
        "Grayscale"
    ]
    
    DEFAULT_COLORMAP = "Blue to Red"
    DEFAULT_SCALE_MODE = "Linear"
    
    # Lookup table
    LUT_NUM_TABLE_VALUES = 256
    
    # ============================================
    # FILE I/O
    # ============================================
    
    # Supported VTK formats
    VTK_FILE_EXTENSIONS = {
        '.vts': 'XML Structured Grid',
        '.vtu': 'XML Unstructured Grid',
        '.vtp': 'XML PolyData',
        '.vti': 'XML Image Data',
        '.vtk': 'Legacy VTK'
    }
    
    # Texture file search patterns
    EARTH_TEXTURE_FILES = [
        "earth_texture.jpg", "earth_map.jpg", "world_map.jpg",
        "earth_texture.png", "earth_map.png", "world_map.png",
        "blue_marble.jpg", "blue_marble.png",
        "natural_earth.jpg", "natural_earth.png",
        "earth.jpg", "earth.png", "world.jpg", "world.png"
    ]
    
    STARFIELD_TEXTURE_FILES = [
        "starfield.jpg", "starfield.png", "star_map.jpg", "star_map.png",
        "stars.jpg", "stars.png", "milky_way.jpg", "milky_way.png",
        "celestial_sphere.jpg", "celestial_sphere.png", "night_sky.jpg",
        "starmap_4k.jpg", "starmap_8k.jpg", "starmap.jpg",
        "eso_milky_way.jpg", "hubble_starfield.jpg"
    ]
    
    # CSV requirements
    REQUIRED_CSV_COLUMNS = ['time', 'x', 'y', 'z']
    OPTIONAL_CSV_COLUMNS = ['vx', 'vy', 'vz']
    
    # ============================================
    # ENERGY SPECTRUM PARAMETERS
    # ============================================
    ENERGY_BINS_MIN = 10  # keV
    ENERGY_BINS_MAX = 10000  # keV (10 MeV)
    ENERGY_BINS_COUNT = 50
    
    # Spectral characteristics by altitude
    LEO_ALTITUDE_MAX = 1000  # km
    LEO_SPECTRAL_INDEX = -2.5
    LEO_CHARACTERISTIC_ENERGY = 100  # keV
    
    INNER_BELT_ALTITUDE_MAX = 5000  # km
    INNER_BELT_SPECTRAL_INDEX = -1.8
    INNER_BELT_CHARACTERISTIC_ENERGY = 200  # keV
    
    PEAK_BELT_ALTITUDE_MAX = 15000  # km
    PEAK_BELT_SPECTRAL_INDEX = -1.2
    PEAK_BELT_CHARACTERISTIC_ENERGY = 500  # keV
    
    OUTER_BELT_SPECTRAL_INDEX = -3.0
    OUTER_BELT_CHARACTERISTIC_ENERGY = 50  # keV
    
    # ============================================
    # PERFORMANCE PARAMETERS
    # ============================================
    
    # Debounce delays (milliseconds)
    POINT_DENSITY_DEBOUNCE_MS = 200
    POINT_SIZE_DEBOUNCE_MS = 50
    SLICE_UPDATE_DEBOUNCE_MS = 200
    ISOSURFACE_UPDATE_DEBOUNCE_MS = 300
    
    # Thresholds
    SIGNIFICANT_FLUX_THRESHOLD_FACTOR = 0.01  # 1% of max
    FLUX_FALLBACK_THRESHOLDS = [0.0001, 0.00001, 0.0]
    
    # ============================================
    # DEBUG PARAMETERS
    # ============================================
    DEBUG_ENABLED = False
    DEBUG_LOG_LEVEL = "INFO"
    VERBOSE_VTK_LOADING = True
    SHOW_PERFORMANCE_METRICS = False
    
    # ============================================
    # PROCEDURAL TEXTURE PARAMETERS
    # ============================================
    
    # Earth texture
    EARTH_TEXTURE_WIDTH = 1024
    EARTH_TEXTURE_HEIGHT = 512
    
    # Starfield procedural generation
    STAR_MAGNITUDE_POWER = 3  # Cube for realistic distribution
    STAR_BLUE_WHITE_PROBABILITY = 0.7
    STAR_YELLOW_PROBABILITY = 0.2
    STAR_RED_PROBABILITY = 0.1
    
    STAR_BLUE_WHITE_COLOR = [200, 210, 255]
    STAR_YELLOW_COLOR = [255, 245, 200]
    STAR_RED_COLOR = [255, 200, 150]
    
    NEBULA_COUNT = 50
    NEBULA_MIN_RADIUS = 20
    NEBULA_MAX_RADIUS = 100
    NEBULA_COLOR = [20, 30, 60]  # Faint blue/purple
    NEBULA_MAX_INTENSITY = 0.1
