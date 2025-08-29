# radjax
A `JAX` based line radiative transfer code for inference of protoplanetary disks.  
JAX-based array operations produce significant speedups, enabling fast computations of line-rte. 

---

## Installation

We recommend installing in a clean conda environment. Below are quick recipes for the three most common setups.


### Step-by-step

#### 1. Create environment
```
conda config --add channels conda-forge
conda create -n jax python=3.12
conda activate jax
```

#### 2. Install CUDA (Linux + NVIDIA GPU only)
```
conda install cuda -c nvidia

```
#### 3. Install JAX (platform-specific)
Follow the official [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Use the variant matching your system:
```
# Linux / Windows (CPU)
pip install "jax[cpu]"

# Linux + CUDA 12
pip install "jax[cuda12]"

# macOS (Apple Silicon, Metal)
pip install "jax[cpu]" "jax-metal"

```

#### 4. Install radjax (-e for developer mode) 
```
git clone https://github.com/aviadlevis/radjax.git
cd radjax
pip install -e .
```

#### 5. Optional: visualization + MCMC tools
```
conda install emcee h5py ffmpeg
```

---

## Getting started
To try out radjax, open the example notebooks in the `tutorials/` folder.
They demonstrate how to:
 - load disk + observation metadata from YAML files
 - render synthetic disk cubes
 - render paramtric disks and convolve with ALMA beams to compare against observational data.

Launch `JupyterLab` and explore:
```
jupyter lab tutorials/
```