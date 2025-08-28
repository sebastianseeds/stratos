"""
Integration test for refactored components
Tests that the new modular structure works with the existing main file
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test config import
        from config import Config
        print("✓ Config imported successfully")
        print(f"  - Earth radius: {Config.EARTH_RADIUS_KM} km")
        
        # Test core imports
        from core import OrbitalPoint, OrbitalPath
        print("✓ Core module imported successfully")
        
        # Test IO imports
        from data_io import VTKDataLoader, OrbitalDataLoader
        print("✓ IO module imported successfully")
        
        # Test visualization imports
        from visualization import ColorManager
        print("✓ Visualization module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_orbital_point():
    """Test OrbitalPoint creation"""
    print("\nTesting OrbitalPoint...")
    
    from core import OrbitalPoint
    
    # Create a test point
    point = OrbitalPoint(
        time=1.5,
        x=7000.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=7.5,
        vz=0.0
    )
    
    print(f"  Created: {point}")
    print(f"  Altitude: {point.altitude:.2f} km")
    print(f"  Position: {point.get_position()}")
    assert point.altitude > 0, "Altitude should be positive"
    print("✓ OrbitalPoint works correctly")

def test_color_manager():
    """Test ColorManager"""
    print("\nTesting ColorManager...")
    
    from visualization import ColorManager
    import vtk
    
    manager = ColorManager()
    
    # Test colormap color
    color = manager.get_colormap_color("Viridis", 0.5)
    print(f"  Viridis at 0.5: RGB{color}")
    assert len(color) == 3, "Color should have 3 components"
    assert all(0 <= c <= 1 for c in color), "Color values should be 0-1"
    
    # Test LUT creation
    lut = manager.create_lookup_table_with_scale("Plasma", "Linear", (1e-6, 1e-2))
    assert isinstance(lut, vtk.vtkLookupTable), "Should return vtkLookupTable"
    print(f"  LUT created with range: {lut.GetRange()}")
    
    print("✓ ColorManager works correctly")

def test_config():
    """Test Config values"""
    print("\nTesting Config...")
    
    from config import Config
    
    print(f"  Earth radius: {Config.EARTH_RADIUS_KM} km")
    print(f"  Default point density: {Config.DEFAULT_POINT_DENSITY}")
    print(f"  Available colormaps: {Config.AVAILABLE_COLORMAPS}")
    
    assert Config.EARTH_RADIUS_KM == 6371.0, "Earth radius should be 6371 km"
    print("✓ Config works correctly")

def test_with_sample_data():
    """Test with sample CSV data if available"""
    print("\nTesting with sample data...")
    
    from data_io import OrbitalDataLoader
    import tempfile
    import pandas as pd
    
    # Create sample CSV data
    sample_data = pd.DataFrame({
        'time': [0.0, 0.5, 1.0, 1.5, 2.0],
        'x': [7000.0, 7000.0, 0.0, -7000.0, -7000.0],
        'y': [0.0, 3500.0, 7000.0, 3500.0, 0.0],
        'z': [0.0, 0.0, 0.0, 0.0, 0.0],
        'vx': [0.0, -3.75, -7.5, -3.75, 0.0],
        'vy': [7.5, 6.5, 0.0, -6.5, -7.5],
        'vz': [0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f, index=False)
        temp_path = f.name
    
    try:
        # Load the data
        orbital_points = OrbitalDataLoader.load_csv(temp_path)
        print(f"  Loaded {len(orbital_points)} points")
        
        # Check first point
        first = orbital_points[0]
        print(f"  First point: {first}")
        print(f"  Altitude range: {min(p.altitude for p in orbital_points):.1f} - "
              f"{max(p.altitude for p in orbital_points):.1f} km")
        
        print("✓ Orbital data loading works correctly")
        
    finally:
        # Cleanup
        os.unlink(temp_path)

def run_all_tests():
    """Run all integration tests"""
    print("=" * 50)
    print("INTEGRATION TESTS")
    print("=" * 50)
    
    all_passed = True
    
    # Run each test
    tests = [
        test_imports,
        test_orbital_point,
        test_color_manager,
        test_config,
        test_with_sample_data
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
