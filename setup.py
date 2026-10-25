from setuptools import setup, find_packages
import os

# Read README.md for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "HGPRGDel: Metabolic Hypergraph and GPR Rule Learning for Growth-Coupled Gene Deletion Prediction"

# Read requirements.txt for dependencies
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
except FileNotFoundError:
    requirements = [
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "transformers>=4.21.0",
        "fair-esm>=2.0.0",
        "rdkit-pypi>=2022.9.1",
        "scikit-learn>=1.1.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ]

setup(
    name="hgprgdel",
    version="1.0.0",
    author="Ziwei Yang",
    author_email="",
    license="MIT",
    description="HGPRGDel: Metabolic Hypergraph and GPR Rule Learning for Growth-Coupled Gene Deletion Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HGPRGDel/HGPRGDel",
    py_modules=['HGPRGDel_main'],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "hgprgdel=HGPRGDel_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.csv", "*.json", "*.yaml"],
    },
    zip_safe=False,
)
