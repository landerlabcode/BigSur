# setup.py file for BigSur
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="bigsur",  # Required
    version="0.0.6",  # Required
    description="Basic Informatics and Gene Statistics from Unnormalized Reads, a feature selection tool for scRNAseq",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/landerlabcode/BigSur",  # Optional
    author="Emmanuel Dollinger",  # Optional
    author_email="edolling@uci.edu",  # Optional
    classifiers=[  # Optional
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: single-cell RNA-seq researchers",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="single-cell RNA-seq, feature selection, bioinformatics",  # Optional
    package_dir={"": "src"},  # Optional
    #
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.9, <4",
    install_requires=[
        'scanpy',
        'mpmath',
        'numexpr',
        'ipykernel',
        'python-igraph',
        'leidenalg',
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/landerlabcode/BigSur/issues",
        "Feature Selection Paper (biorxiv)": "https://doi.org/10.1101/2024.10.11.617709",
        "Correlations Paper": "https://doi.org/10.1186/s12859-024-05926-z",
    },
)
