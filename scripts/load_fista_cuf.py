"""
script to import and compile fista_cuf and ensure that the correct modules are loaded
on the NSF HPC systems when this is done.

imports that are expected on Derecho:

# Derecho
module load cuda   # current cuda version is 11.7 on Casper
module unload intel

# Casper
module load cuda/11.8 ncarenv
module swap intel gcc

#some compiler checks:
gcc --version  # at creation this was gcc (SUSE Linux) 7.5.0 on Casper
echo $CXX  # this should be gcc, not icpx

# load the python environment
source $HOME/.bashrc
source activate ptv-torch-cuda



"""

import os,sys
import yaml

# cuda version of fista.
# gets compiled when you install into your env: pip install -e python/STCuda
# note that -e doesn't catch c++/cuda changes, need to (re)install
import STCuda

# load file path information from the home directory
file_path_yml = os.path.join(os.environ["HOME"], ".ncar_config_derecho.yaml")
path_data = {}
with open(file_path_yml, "r") as r:
    path_data = yaml.safe_load(r)


dirP_str = os.path.join(
    path_data["ptv_collection_path"], "SpiralTorch", "python"
)
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

# non-cuda fista library
from SpiralTorch import fista


#### Test ####

import torch
import numpy as np

is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
# device = torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')

dtype = torch.float64

# create a random test image
x_axis = np.linspace(-10,10,64)
y_axis = np.linspace(-10,10,128)
x_ax_mesh,y_ax_mesh = np.meshgrid(x_axis,y_axis)

rec_count = 80

rec_x_position_arr = 20*(np.random.rand(rec_count)-0.5)
rec_y_position_arr = 20*(np.random.rand(rec_count)-0.5)
rec_x_arr = np.random.randn(rec_count)*5
rec_y_arr = np.random.randn(rec_count)*5
alpha_rec_arr = 1000*np.random.rand(rec_count)

alpha_arr = np.zeros((y_axis.size,x_axis.size))

for idx in range(rec_count):
    rec_idx = np.where((y_ax_mesh >= rec_y_position_arr[idx]-rec_y_arr[idx]/2) & (y_ax_mesh <= rec_y_position_arr[idx]+rec_y_arr[idx]/2) & \
             (x_ax_mesh >= rec_x_position_arr[idx]-rec_x_arr[idx]/2) & (x_ax_mesh <= rec_x_position_arr[idx]+rec_x_arr[idx]/2))
    alpha_arr[rec_idx]+=alpha_rec_arr[idx]

alpha_arr+= 10


# setup FISTA run arguments

x0 = {'backscatter':torch.tensor(alpha_arr,dtype=dtype,device=device)}

alpha = 1e1
x_lb = torch.zeros_like(x0['backscatter'])-1e10
x_ub = torch.zeros_like(x0['backscatter'])+1e10
cu_fista = STCuda.ops.st_fista_subproblem

res_cu = cu_fista(x0['backscatter'],torch.tensor(1e-1/alpha,device=device,dtype=dtype),x_lb,x_ub)

# non-cuda version of fista
x0 = {'backscatter':torch.tensor(alpha_arr,dtype=dtype,device=device)}

jit_fista = torch.jit.trace(fista.solve_FISTA_subproblem_jit,(x0['backscatter'],torch.tensor(1e-1/alpha,device=device,dtype=dtype),
                                                    x_lb,x_ub))
res_jit = jit_fista(x0['backscatter'],torch.tensor(1e-1/alpha,device=device,dtype=dtype),x_lb,x_ub)

print("cuda-fista output")
print(torch.sum(res_cu))
print(res_cu.shape)

print("\njit-fista output")
print(torch.sum(res_jit))
print(res_jit.shape)