# Core libraries for various functionalities
import torch  # PyTorch library for deep learning
import os     # Operating system interfaces
import math   # Mathematical functions
import gzip   # Compression/decompression using gzip
import pickle # Object serialization
from jenkspy import JenksNaturalBreaks
import numpy as np
# Plotting library
import matplotlib.pyplot as plt

# For downloading files from URLs
from urllib.request import urlretrieve

# File and directory handling
from pathlib import Path

# Specific PyTorch imports
from torch import tensor  # Tensor data structure

# Computer vision libraries
import torchvision as tv  # PyTorch's computer vision library
import torchvision.transforms.functional as tvf  # Functional image transformations
from torchvision import io  # I/O operations for images and videos

# For loading custom CUDA extensions
from torch.utils.cpp_extension import load_inline, CUDA_HOME

# Verify the CUDA install path 
print(CUDA_HOME)


def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    """
    Load CUDA and C++ source code as a Python extension.

    This function compiles and loads CUDA and C++ source code as a Python extension,
    allowing for the use of custom CUDA kernels in Python.

    Args:
        cuda_src (str): CUDA source code as a string.
        cpp_src (str): C++ source code as a string.
        funcs (list): List of function names to be exposed from the extension.
        opt (bool, optional): Whether to enable optimization flags. Defaults to False.
        verbose (bool, optional): Whether to print verbose output during compilation. Defaults to False.

    Returns:
        module: Loaded Python extension module containing the compiled functions.
    """
    # Use load_inline to compile and load the CUDA and C++ source code
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")


# Define CUDA boilerplate code and utility macros
cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Macro to check if a tensor is a CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if a tensor is contiguous in memory
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguity requirements
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for ceiling division
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_bias = cuda_begin + r'''
template <typename T>
__global__ void Jenks_Optimization_Biases(T* d_B, T* d_var, int rows){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows)){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_B[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows; i++){
            mean_2 += d_B[i];
        }
        mean_2 = mean_2/(rows-idx);
    }
    __syncthreads();
    if(idx<rows){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_B[i] - mean_1)*(d_B[i] - mean_1);
        }
        for(int i = idx; i<rows; i++){
            var += (d_B[i] - mean_2)*(d_B[i] - mean_2);
        }
        d_var[idx] = var;
    }
}



torch::Tensor jenks_optimization_biases_cuda(torch::Tensor B){
    // Check input
    CHECK_INPUT(B);

    // Get dimensions
    int rows = B.size(0);

    // Allocate output tensor
    auto var = torch::empty({rows}, B.options());

    // Launch kernel
    const int threads = 256;
    const int blocks = cdiv(rows, threads);

    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "jenks_optimization_biases_cuda", ([&] {
        Jenks_Optimization_Biases<scalar_t><<<blocks, threads>>>(B.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), rows);
    }));

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return var;
}
'''

cuda_src = cuda_begin + r'''    

template <typename T>
__global__ void Jenks_Optimization(T* d_WB, T* d_var, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows*(cols))){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_WB[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows*(cols); i++){
            mean_2 += d_WB[i];
        }
        mean_2 = mean_2/(rows*(cols)-idx);
    }
    __syncthreads();
    if(idx<rows*(cols)){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_WB[i] - mean_1)*(d_WB[i] - mean_1);
        }
        for(int i = idx; i<rows*(cols); i++){
            var += (d_WB[i] - mean_2)*(d_WB[i] - mean_2);
        }
        d_var[idx] = var;
    }
}


torch::Tensor jenks_optimization_cuda(torch::Tensor WB){
    // Check input
    CHECK_INPUT(WB);

    // Get dimensions
    int rows = WB.size(0);
    int cols = WB.size(1);

    // Allocate output tensor
    auto var = torch::empty({rows*cols}, WB.options());

    // Launch kernel
    const int threads = 256;
    const int blocks = cdiv(rows*cols, threads);
    AT_DISPATCH_FLOATING_TYPES(WB.scalar_type(), "jenks_optimization_cuda", ([&] {
        Jenks_Optimization<scalar_t><<<blocks, threads>>>(WB.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), rows, cols);
    }));

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return var;
}


'''

cpp_src = "torch::Tensor jenks_optimization_cuda(torch::Tensor WB);"
cpp_bias_src = "torch::Tensor jenks_optimization_biases_cuda(torch::Tensor B);"

module_weights = load_cuda(cuda_src, cpp_src, ["jenks_optimization_cuda"], opt=True, verbose=True)
module_bias = load_cuda(cuda_bias, cpp_bias_src, ["jenks_optimization_biases_cuda"], opt=True, verbose=True)

# Test

# Create a random tensor

WB = torch.rand(4, 5)
WB_cuda = WB.cuda()
print(WB_cuda)
 
## Sort it

WB_cuda_flatten = WB_cuda.flatten()
WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
WB_cuda_sorted = WB_cuda_sorted.reshape(WB_cuda.shape)
print(WB_cuda_sorted)
# Call the custom CUDA function
print(WB_cuda_indices)

var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
print(var.shape)
print(WB_cuda_sorted.shape)
var_min = var.argmin().item()
# Print the output

print(var)
zeros = WB_cuda_indices[:var_min]
ones = WB_cuda_indices[var_min:]
arr = torch.zeros(WB_cuda.flatten().shape)
arr[zeros] = 0
arr[ones] = 1
arr = arr.reshape(WB_cuda.shape)
print(arr)
''' We now need to find the break point which '''

arr_score = WB.numpy()
arr_score_flat = arr_score.flatten()
jnb = JenksNaturalBreaks(2)
jnb.fit(arr_score_flat)
print(jnb.labels_)
labels = jnb.labels_
indices = np.where(labels == 1)[0]
indices_ = np.where(labels == 0)[0]

test_arr = np.zeros(arr_score_flat.shape)
test_arr[indices] = 1
test_arr[indices_] = 0
test_arr = test_arr.reshape(arr_score.shape)
print(test_arr)

''' Test they are equal'''

print(np.allclose(arr.numpy(), test_arr, atol=1e-4))

# Create a random tensor
B = torch.rand(4)
B_cuda = B.cuda()
# Call the custom CUDA function
B_cuda_sorted, B_cuda_indices = B_cuda.sort()
var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
print(var)
print(B_cuda_sorted)
var_min = var.argmin().item()
# Print the output
print(var_min)
print(B_cuda_sorted)
zeros = B_cuda_indices[:var_min]
ones = B_cuda_indices[var_min:]
arr = torch.zeros(B_cuda.shape)
arr[zeros] = 0
arr[ones] = 1
print(arr)

# Test
arr_score = B.numpy()
arr_score_flat = arr_score.flatten()
jnb = JenksNaturalBreaks(2)
jnb.fit(arr_score_flat)
print(jnb.labels_)
labels = jnb.labels_
indices = np.where(labels == 1)[0]
indices_ = np.where(labels == 0)[0]

test_arr = np.zeros(arr_score_flat.shape)
test_arr[indices] = 1
test_arr[indices_] = 0

print(test_arr)

''' Test they are equal'''

print(np.allclose(arr, test_arr, atol=1e-4))
