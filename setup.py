import re
from pathlib import Path

from setuptools import find_namespace_packages, setup

_here = Path(__file__).resolve().parent

name = "mujoco_utils"

# Reference: https://github.com/patrick-kidger/equinox/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


with open(_here / "README.md", "r") as f:
    readme = f.read()

# Minimum requirements to import and run.
core_requirements = [
    "dm_control>=1.0.9",
    "mujoco>=2.3.1",
]

test_requirements = [
    "absl-py",
    "pytest-xdist",
]

# Requirements for development (use the Makefile for convenience).
dev_requirements = [
    "black",
    "mypy",
    "ruff",
] + test_requirements

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

author = "Kevin Zakka"

author_email = "kevinarmandzakka@gmail.com"

description = "Utilities for working with MuJoCo Python bindings and dm_control."


setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=f"https://github.com/kevinzakka/{name}",
    license="Apache License 2.0",
    license_files=("LICENSE",),
    packages=find_namespace_packages(exclude=["*_test.py"]),
    package_data={f"{name}": ["py.typed"]},
    python_requires=">=3.8",
    install_requires=core_requirements,
    classifiers=classifiers,
    extras_require={
        "test": test_requirements,
        "dev": dev_requirements,
    },
)
