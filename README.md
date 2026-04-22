# SpiralTorch
Library using PyTorch to implement Spiral-TAP, Sparsa and FISTA for total variation denoising.  This is a method for calculating gradients to solve Total Variation (TV) regularization problems.

The concept is that we wish to obtain a state variable $x$ from a physical model $f(x)$ relating the state to the noisy observations $y$.  Maximum likelihood estimation is employed to compare the parameterization of $f(x)$ to the noise observations so that estimate is

$\tilde{x} = argmin L[y,f(x)] + \lambda ||x||_{TV}$

where $L(y,\alpha)$ is the statistical model for the noisy observations given parameterization $\alpha$, $||x||_{TV}$ is the total variation of $x$ and $\lambda$ is a real scalar that sets the total variation regularization.

The optimizer implementation here is based on the following published works:

1.  Oh, A. K., Harmany, Z. T. & Willett, R. M. Logarithmic total variation regularization for cross-validation in photon-limited imaging, In 2013 IEEE International Conference on Image Processing, 484–488, DOI: 10.1109/ICIP.2013.6738100 (2013).

2. Harmany, Z. T., Marcia, R. F. & Willett, R. M. This is spiral-tap: Sparse Poisson intensity reconstruction algorithms-theory and practice. IEEE Trans. Image Process, 21, 1084–1096, DOI: 10.1109/TIP.2011.2168410 (2012).

3. Beck, A. & Teboulle, M. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems, IEEE Trans. Image Process. 18, 2419–2434, DOI: 10.1109/TIP.2009.2028250 (2009).

4. Wright, S. J. & Nowak, R. D. Sparse Reconstruction by Separable Approximation, IEEE Trans. Signal Processing, 57, 2479-2493, DOI: 10.1109/TSP.2009.2016892 (2009).

Work making use of software in this repository should cite one or more of the following works:

1. M. Hayman, R. A. Stillwell, A. Karboski, S. M. Spuler, "Signal processing to denoise and retrieve water vapor from multi-pulse-length lidar data", In Preparation.

2. M. Hayman, R. A. Stillwell, A. Karboski, W. J. Marais, S. M. Spuler, "Global Estimation of Range Resolved Thermodynamic Profiles from MicroPulse Differential Absorption Lidar," Opt Express, 32(8), 14442-14460 (2024).

3. M. Hayman, R. A. Stillwell, J. Carnes, G. J. Kirchhoff, J. P. Thayer, S. M. Spuler, "2D Signal Estimation for Sparse Distributed Target Photon Counting Data," Sci Rep, 14, 10325, DOI: 10.1038/s41598-024-60464-1 (2024).


# Examples using SpiralTorch

Examples of single variable and multi variable estimation can be found in
```
notebooks/casper/RunSpiral.ipynb
```
Also an example of backscatter and depolarization estimation is in
```
notebooks/casper/Spiral_MultiVariable_Estimation.ipynb
```

These notebooks are self contained within this repository (unlike some other examples that use actual MPD data).

# Python Environment
Installing packages using conda can be accomplished with an anaconda or [miniforge](https://github.com/conda-forge/miniforge) installation as follows

Create a new python environment
Create the python environment (in this exampled named `ptv-casper-cuda12`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cuda12 python=3.12
conda activate ptv-casper-cuda12
```
Install base packages
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4 jupyter
conda install scikit-learn -c conda-forge
conda install ninja  # only needed if using the custom cuda kernel for speedup
```
If you are only installing `PyTorch` for CPU, use the installation script
```
conda install pytorch -c pytorch  # this was tested with 2.3.0, but there is no reason to think other versions won't work
```

If you are planning to use a GPU, you will need to ensure that the CUDA compiler version is aligned with the install request.  E.g.
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
It is also important that you make sure your c++ compiler is compatable with the cuda compiler.  In the case of the available compilers on the Casper cluster, `nvcc 11` is not compatable with any `gcc > 11`.  Since Casper does not have any modules of `gcc <= 11` available, we have to use `nvcc 12` along with the `pytorch-cuda=12.1`.

To check your cuda compiler version:
```
nvcc --version
```

Note that if you intend to use the custom cuda kernel here, PyTorch changed how these kerenls are compiled after 2.3.  The current version will only work with PyTorch 2.3 and earlier.

# Install on Casper (NSF-NCAR Cluster)
Part of the challenge in building an environment that leverages the GPUs is ensuring that there is alignment between the CUDA compiler versions and the CUDA compiler used by PyTorch and the C++ compiler.  Here are the steps to install a GPU enabled environment through the example of the Casper cluster with a GPU (V100) node.

The NCAR conda module is way faster than the miniconda one in my home directory
```
module load conda
```

To list the available versions of cuda on the NCAR HPC systems
```
module spider cuda
```

Search versions of pytorch available for one that is compatible with the version of cuda and python available.
```
conda search pytorch -c pytorch
```
an example of the package we might want to install is
```
pytorch                        2.3.1 py3.12_cuda12.1_cudnn8.9.2_0  pytorch
```
which tells us we need cuda 12.1 (really any 12.x version) and we should install pytoch 2.3.0

Create the python environment (in this exampled named `ptv-casper-cuda12`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cuda12 python=3.12
conda activate ptv-casper-cuda12
```

Import the cuda module, make sure the version corresponds to the expected install version for pytorch (from the search above)
```
module swap intel gcc
module load gcc/12  # not generally necessary, but helps control the gcc compiler version
module load cuda/12
export CONDA_OVERRIDE_CUDA="12.2"
```
Install libraries
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4
conda install scikit-learn -c conda-forge
conda install ninja
conda install lmod  # this is for setting the environment on JupyterHub
```
Install pytorch libraries with cuda (for `cuda/12.2`)
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Fixing compiler checks in older versions of PyTorch
Some older versions of PyTorch use `-v` arguments to check compatability of compilers which will fail on the NCAR HPC systems (this argument will try to compile if it can, fail, and cause the program to fail).  If this failure occurs at the compiler check line, switching the `-v` flag to `--version` will likely solve the problem.  This is done by editing the PyTorch source python directly in
```
[EnvironmentPath]/conda-envs/ptv-casper-cuda12/lib/python3.12/site-packages/torch/utils/cpp_extension.py
```

Note that the `-v` flag does not always mean "version".  The `ninja` flag `-v` is equivalent to `--verbose`.  Don't change this to `--version` in the `cpp_extension.py` or it breaks the compile step!

Note that in PyTorch 2.3.0 the following `try/except` case has been added to `cpp_extension.py` so that this issue does not arise
```
try:
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
except Exception as e:
    try:
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception as e:
        return False
```

# Compiling CUDA for FISTA
There are two options for the FISTA function used in the gradient calculation.  `jit-fista` uses PyTorch's precompiler and is built on the PyTorch types and architecture.  

A faster version of FISTA (`st_fista_cuf` or `cuda-fista`) is a compiled cuda kernel.  While this approach is faster, it needs to be compiled on the system. The version of PyTorch needs to be 2.3 or earlier and algin with the cuda compiler version.  In the following command line installation the cuda version is 12.x.  The `--override-channels` flag is needed when other default repositories (e.g. conda-forge) may produce competing installation versions.
```
conda install pytorch=2.3 pytorch-cuda=12 -c pytorch -c nvidia --override-channels

```

### Note on the cuda definitions
The method for creating a custom cuda kernel in this repo follows this tutorial

https://pytorch.org/tutorials/advanced/cpp_extension.html

However this method is apparently being replaced in PyTorch.

Starting in PyTorch 2.4 the method for creating custom cuda operators changes

https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#testing-an-operator

This will require modifications to how the package is setup but the new tutorial is a bit hard to follow.  Anyone interested in updating to cuda kernel should contact me.  It would be great to have this work beyond Pytorch 2.3.

### Updates for Pytorch > 2.3
In the SpiralTorch directory type
```
pip install -e packages/STCuda
```