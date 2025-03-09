"""
Setup script for the social-media-sentiment-analysis package.
"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Get package version
version = {}
with open("src/__init__.py") as f:
    exec(f.read(), version)

# Get requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="social-media-sentiment-analysis",
    version=version.get("__version__", "1.0.0"),
    description="A professional sentiment analysis tool for social media content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Latex999/social-media-sentiment-analysis",
    author="Latex999",
    author_email="M.Hassan17877@student.aast.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="sentiment-analysis, nlp, twitter, reddit, social-media, sentiment",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sentiment-analysis=src.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Latex999/social-media-sentiment-analysis/issues",
        "Source": "https://github.com/Latex999/social-media-sentiment-analysis",
    },
)