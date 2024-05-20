# radjax
A `JAX` based line radiative transfer code for inference of protoplanetary disks. \
JAX based array operation produce significant speed ups enabling fast computations of line-rte. 

Installation
---
Start a conda virtual environment and add channels
```
conda config --add channels conda-forge
conda create -n jax python=3.12
conda activate jax
```

Install lastest CUDA if not already installed
```
conda install cuda -c nvidia
```

Install [`JAX`](https://jax.readthedocs.io/en/latest/installation.html) and related dependencies 
```
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"
pip install flax optax diffrax tensorboard tensorboardX
```

Install additional packages 
```
pip install numpy scipy scikit-learn scikit-image matplotlib jupyterlab nodejs tqdm ipympl ipyvolume mpmath ruamel.yaml corner
conda install emcee h5py ffmpeg
```
