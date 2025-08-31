"""
Setup script for STRATOS - Space-time Radiation Analysis and Trajectory Orbital Simulator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="stratos",
    version="0.1.0",
    description="Space-time Radiation Analysis and Trajectory Orbital Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="STRATOS Development Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "PyQt6>=6.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "vtk>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ]
    },
    entry_points={
        "console_scripts": [
            "stratos=flux_visualizer.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="space physics, radiation belts, orbital mechanics, visualization",
)