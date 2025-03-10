# Core libraries for various functionalities
import torch  # PyTorch library for deep learning
import os     # Operating system interfaces
import math   # Mathematical functions
import gzip   # Compression/decompression using gzip
import pickle # Object serialization

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