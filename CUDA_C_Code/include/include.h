#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/set_operations.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <cusolverDn.h>
#include "GPUErrors.h"


#define RANGE_MAX 0.5
#define RANGE_MIN -0.5


// functions.cpp
void InitializeMatrix(float *matrix, int ny, int nx);
void ZeroMatrix(float *temp, const int ny, const int nx);


//ml_time.cu
template <typename T>
__global__ void MatrixVectorMult(T* g_Matrix, T* g_V, T* g_P, const int num_row, const int num_col) {
	int row = threadIdx.x + (blockDim.x * blockIdx.x);//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (row < num_row) {
		//We are trying to ensure we are not using more threads than data we have
		for (int k{}; k < num_col;k++) {
			fSum += g_Matrix[row * num_col + k] * g_V[k];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[row] = fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}

// Activation functions
template <typename T>
__device__ T Sigmoid(T x)
{
    return 1.0f / (1.0f + exp(-x));
}
template <typename T>
__device__ T dSigmoid(T x)
{
    return x * (1.0f - x);
}
template <typename T>
__device__ T ReLU(T x)
{
    return x > 0.0f ? x : 0.0f;
}
template <typename T>
__device__ T dReLU(T x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}
template <typename T>
__device__ T LeakyReLU(T x, T alpha)
{
    return x > 0.0f ? x : -alpha * x;
}

template <typename T>
__device__ T dLeakyReLU(T x, T alpha)
{
    return x > 0.0f ? 1.0f : -alpha;
}

template <typename T>
__device__ T Tanh(T x)
{
    return tanh(x);
}

template <typename T>
__device__ T dTanh(T x)
{
    return 1.0f - x * x;
}
