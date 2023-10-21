#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.1.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=1.0",
    "torchdata",
    "pytorch-ignite",
    "seaborn",
    "altair",
    "rdkit-pypi",  # maybe switch to official rdkit?
    "pandas>1.0",
    "jupyter",
    "matplotlib",
    "numpy",
    "scipy",
    "scikit-learn",
    "boto3",
    "botocore",
    "tqdm",
    "due @ git+https://github.com/y0ast/DUE.git",
]

setup(
    authors="John Parkhill 🧙‍♂️, Edward Williams 🧟‍♂️, Carl Underkoffler 👨‍🎤, Ben Kaufman 🧑🏻‍💻, Ryan Pederson 🦦",
    author_email="jparkhill@terraytx.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="COATI: multi-modal contrastive pre-training for representing and traversing chemical space",
    install_requires=requirements,
    packages=find_packages(),
    long_description=readme,
    include_package_data=True,
    keywords="coati",
    name="coati",
    version=__version__,
    extras_require={"selfies": ["selfies"]},
    zip_safe=False,
)
