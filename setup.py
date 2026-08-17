from __future__ import annotations
from pathlib import Path
import numpy
from setuptools import Extension, find_packages, setup
ROOT = Path(__file__).parent
native = Extension("weaveode._native", sources=["cpp/core.cpp", "cpp/python_module.cpp"], include_dirs=[str(ROOT / "include"), numpy.get_include()], language="c++", extra_compile_args=["-std=c++20", "-O3", "-Wall", "-Wextra", "-Wpedantic"])
setup(name="mosaic-ode", version="0.2.0a0", description="GPU-assisted topology-aware online-adaptive integrated ODE solver pre-product", package_dir={"": "src"}, packages=find_packages("src"), ext_modules=[native], install_requires=["numpy>=1.26", "scipy>=1.11", "torch>=2.2"], python_requires=">=3.11")
