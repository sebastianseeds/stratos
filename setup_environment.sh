#!/bin/bash

echo "Setting up Python Electron Flux Visualizer Environment"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv flux_env

# Activate virtual environment
echo "Activating virtual environment..."
source flux_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements with error handling
echo "Installing core packages..."

# Install packages one by one with error checking
install_package() {
    echo "Installing $1..."
    if pip install "$1"; then
        echo "✓ $1 installed successfully"
    else
        echo "✗ Failed to install $1"
        echo "Trying alternative installation method for $1..."
        
        # Special handling for VTK
        if [ "$1" = "vtk" ]; then
            echo "Trying conda installation for VTK..."
            if command -v conda &> /dev/null; then
                conda install -c conda-forge vtk -y
            else
                echo "Conda not available. Please install VTK manually:"
                echo "  Option 1: conda install -c conda-forge vtk"
                echo "  Option 2: pip install --upgrade vtk"
                echo "  Option 3: Check VTK website for system-specific instructions"
            fi
        fi
    fi
}

# Install packages
install_package "numpy>=1.20.0"
install_package "PyQt6>=6.0.0"
install_package "pandas>=1.3.0"
install_package "matplotlib>=3.5.0"
install_package "vtk>=9.0.0"

echo ""
echo "Verifying installations..."

# Verify installation with better error handling
verify_import() {
    if python -c "import $1; print('✓ $1 imported successfully')" 2>/dev/null; then
        return 0
    else
        echo "✗ $1 import failed"
        return 1
    fi
}

# Test imports
verify_import "numpy"
verify_import "pandas"
verify_import "matplotlib"
verify_import "PyQt6"

# Special VTK verification with version info
if python -c "import vtk; print(f'✓ VTK version: {vtk.VTK_VERSION}')" 2>/dev/null; then
    echo "✓ VTK verified successfully"
else
    echo "✗ VTK verification failed"
    echo ""
    echo "VTK Installation Troubleshooting:"
    echo "1. Try: pip uninstall vtk && pip install vtk"
    echo "2. Try: conda install -c conda-forge vtk"
    echo "3. On Ubuntu/Debian: sudo apt-get install python3-vtk9"
    echo "4. On macOS: brew install vtk"
    echo "5. Check VTK documentation: https://vtk.org/download/"
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate environment in future sessions:"
echo "  source flux_env/bin/activate"
echo ""
echo "To run the application:"
echo "  python flux_visualizer.py"
echo ""
echo "Sample data format requirements:"
echo "VTK Files:"
echo "  - Any VTK format (.vtk, .vtu, .vts, .vtp, .vti)"
echo "  - Must contain scalar field data (electron flux)"
echo "  - Structured or unstructured grids supported"
echo ""
echo "CSV Files (orbital data):"
echo "  - Required columns: time, x, y, z"
echo "  - Optional columns: vx, vy, vz"
echo "  - Units: time (hours), positions (km), velocities (km/s)"
echo "  - Example: time,x,y,z,vx,vy,vz"
echo "           0.0,-6000,2000,1000,5.2,-1.1,0.3"
