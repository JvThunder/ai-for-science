#!/usr/bin/env python3
"""
Setup script for Stereochemistry-Aware 3D GCN

This is a backwards-compatible setup.py for older pip versions.
Modern installations should use pyproject.toml with:
    pip install .
    pip install -e .  # editable mode
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="stereochem-gcn",
    version="1.0.0",
    description="Stereochemistry-Aware 3D Molecular Reconstruction using Graph Convolutional Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/stereochem-gcn",
    packages=find_packages(include=["utils", "utils.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "wandb>=0.15.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stereochem-train=local_geometry_gcn_stereo:main",
            "stereochem-reconstruct=reconstruction:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="molecular-geometry graph-neural-networks stereochemistry 3d-reconstruction lora fine-tuning",
    project_urls={
        "Documentation": "https://github.com/yourusername/stereochem-gcn/blob/main/README.md",
        "Source": "https://github.com/yourusername/stereochem-gcn",
        "Issues": "https://github.com/yourusername/stereochem-gcn/issues",
    },
    include_package_data=True,
    zip_safe=False,
)
