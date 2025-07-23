"""
Setup script for Micro Language Model package.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="micro-lm",
    version="0.1.0",
    description="A simple educational language model implementation",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    author="Micro LM Project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)