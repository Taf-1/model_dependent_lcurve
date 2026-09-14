from setuptools import setup, find_packages

setup(
    name="model_dependent_lcurve",
    version="0.1.0",
    description="Model-dependent MCMC light-curve fitting for ULTRACAM eclipsing binaries",
    author="Tafadzwa Zivave",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "astropy>=5.0",
        "emcee>=3.1",
        "h5py>=3.0",
        "ruamel.yaml>=0.18",
        "lcurve>=0.4",
        "astroquery>=0.4",
    ],
    scripts=["src/lcurve_mcmc.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
