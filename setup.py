from setuptools import setup, find_packages

setup(
    name="radjax",
    version="0.1.0",
    description="JAX-based line radiative transfer for protoplanetary disks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aviad Levis; Nhan (Len) Luong",
    url="https://github.com/aviadlevis/radjax",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax[cuda12]",
        "flax",
        "optax",
        "diffrax",
        "tensorboard",
        "tensorboardX",
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "jupyterlab",
        "tqdm",
        "ipympl",
        "ipyvolume",
        "mpmath",
        "ruamel.yaml",
        "corner",
        "gofish",
        "emcee",
        "h5py",
    ],
    extras_require={
        "dev": [pytest tests/test_grid_rotations.py

            "pytest",
            "black",
            "isort",
            "sphinx",
            "myst-parser",
            "nbsphinx",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)