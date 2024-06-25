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

template <typename T>
void InitializeMatrix(T *matrix, int ny, int nx)
{
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((T)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += nx;
	}
}

template <typename T>
void ZeroMatrix(T *temp, const int ny, const int nx)
{
	float *p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = 0.0f;
		}
		p += nx;
	}
}

template <typename T>
void InitializeVector(float* vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = ((T)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}


template <typename T>
void ZeroVector(T* vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = 0.0f;
	}
}



template <typename T>
class Matrix
{
public:
    T* input;
    Matrix(){
        this->rows = 0;
        this->cols = 0;
        this->weights = NULL;
        this->biases = NULL;
    }
    virtual Matrix(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
        this->weights = (T*)malloc(rows * cols * sizeof(T));
        this->biases = (T*)malloc(rows * sizeof(T));
        this->hidden_output = (T*)malloc(rows * sizeof(T));
        this->input = (T*)malloc(cols * sizeof(T));
        //Create random weights and biases
    }

    Matrix(int rows, int cols, T *weights, T *biases){
        this->rows = rows;
        this->cols = cols;
        this->weights = weights;
        this->biases = biases;
    }
    ~Matrix(){
        free(this->weights);
        free(this->biases);
    }
    void randomize(){
        for(int i=0; i<rows*cols; i++){
            weights[i] = (T)rand() / RAND_MAX;
            if(i<rows){
                biases[i] = (T)rand() / RAND_MAX;
            }
        }
    }
    int rows;
    int cols;
    T *weights;
    T *biases;
    T* d_weights;
    T* d_biases;
    T* hidden_output;
    void matrix_multiply(T *A, T *B, T *C);
    void matrix_add(T *A, T *B, T *C);
    void matrix_subtract(T *A, T *B, T *C);
    void matrix_transpose(T *A, T *C);
    void matrix_scalar_multiply(T *A, T *C, T scalar);
    void matrix_scalar_add(T *A, T *C, T scalar);
    void matrix_scalar_subtract(T *A, T *C, T scalar);
    void matrix_scalar_divide(T *A, T *C, T scalar);
    void matrix_elementwise_multiply(T *A, T *B, T *C);
    void matrix_elementwise_divide(T *A, T *B, T *C);
    void matrix_elementwise_add(T *A, T *B, T *C);
    void matrix_elementwise_subtract(T *A, T *B, T *C);
    void matrix_sum(T *A, T *C, int axis);
    void set_rows(int rows);
    void set_cols(int cols);
    void virtual forward(T *input, T *output);
    void virtual backward(T * loss){};
    void backward(T * loss, int size){};
    void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size){};
    void update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size){};
    void train(T *input, T *output, int epochs, T learning_rate){};
    int get_rows();
    int get_cols();
private:
    cudaError_t cudaStatus;
};



template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

template <typename T>
__global__ void matrix_multiply(T* A, T* B, T* C, int rows, int cols, int inter_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols){
        T sum = 0;
        for(int i = 0; i < inter_size; i++){
            sum += A[row * cols + i] * B[i * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_multiply_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        T sum = 0;
        for (int k = 0; k < cols; k++) {
            sum += A[row * cols + k] * B[k];
        }
        C[row] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_addition_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        C[row] = A[row] + B[row];
    }
}

template <typename T>
void Matrix<T>::forward(T *input, T *output){
        // Allocate device memory for input and output
        cout<<"Matrix forward"<<endl;
        memcpy(this->input, input, cols * sizeof(T));
        // this->input = input;
        T *d_input, *d_output;
        T *d_weights;
        T* d_biases;
        if(!HandleCUDAError(cudaMalloc((void**)&d_input, cols * sizeof(T)))){
            cout<<"Error in allocating memory for d_input"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_output, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_output"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_weights, rows * cols * sizeof(T)))){ 
            cout<<"Error in allocating memory for device_data"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_biases, rows * sizeof(T)))){
            cout<<"Error in allocating memory for d_biases"<<endl;
            return;
        }

        // Copy input from host to device
        if(!HandleCUDAError(cudaMemcpy(d_input, input, cols * sizeof(T), cudaMemcpyHostToDevice))){
            cout<<"Error in copying input from host to device"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
            cout<<"Error in copying d_data from host to device"<<endl;
            return;
        }

        // Define grid and block dimensions
        dim3 gridDim(1, 1, 1);
        dim3 blockDim(cols, rows, 1);

        // Launch the matrix multiplication kernel
        matrix_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_weights, d_input, d_output, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error in synchronizing device"<<endl;
            return;
        }
        matrix_vector_addition_kernel<T><<<gridDim, blockDim>>>(d_output, d_biases, d_output, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error in synchronizing device"<<endl;
            return;
        }
        // Copy the result output from device to host
        if(!HandleCUDAError(cudaMemcpy(output, d_output, rows * sizeof(T), cudaMemcpyDeviceToHost))){
            cout<<"Error in copying output from device to host"<<endl;
            return;
        }

        // Free device memory
        if(!HandleCUDAError(cudaFree(d_input))){
            cout<<"Error in freeing d_input"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaFree(d_output))){
            cout<<"Error in freeing d_output"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaFree(d_weights))){
            cout<<"Error in freeing device_data"<<endl;
            return;
        }
        if(!HandleCUDAError(cudaDeviceReset())){
            cout<<"Error in resetting device"<<endl;
            return;
        }
        memcpy(output, this->hidden_output, rows * sizeof(T));
}


template <typename T>
class Sigmoid: public Matrix<T>
{
public:
    Sigmoid(int rows, int cols);
    int rows;
    int cols;
    T* input;
    T* hidden_output;
    ~Sigmoid();
    void forward(T *input, T *output) override;
    void backward(T * loss) override;
};

template <typename T>
class RELU_layer: public Matrix<T>
{
public:
    RELU_layer(int rows, int cols) override;
    int rows;
    int cols;
    T* input;
    T* hidden_output;
    ~RELU_layer();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
__global__ void softmax_kernel(T *input, T *output, T reduce, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = exp(input[index]) / reduce;
    }
}

template <typename T>
class Softmax: public Matrix<T>
{
    public:
        int rows;
        int cols;
        Softmax(){
            this->rows = 0;
            this->cols = 0;
        }
        Softmax(int rows, int cols){
            this->rows = rows;
            this->cols = 1;
            this->input = (T*)malloc(rows * sizeof(T));
            this->hidden_output = (T*)malloc(rows * sizeof(T));
        }
        ~Softmax();
        T* input;
        T* hidden_output;
        void forward(T *input, T *output) override {
            // Allocate device memory for input and output
            cout<<"Softmax forward"<<endl;
            int size = rows;
            T *d_input, *d_output;
            if(input == NULL){
                cout<<"Input RELU is NULL"<<endl;
                input = (T*)malloc(size * sizeof(T));
                if(input == NULL){
                    cout<<"Input of RELU is NULL"<<endl;
                    exit(1);
                }
            }
            if(output == NULL){
                cout<<"Output of RELU is NULL"<<endl;
                output = (T*)malloc(size * sizeof(T));
                if(output == NULL){
                    cout<<"Output of RELU is NULL"<<endl;
                    exit(1);
                }
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_output"<<endl;
                return;
            }

            // Copy input from host to device
            if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);
            T reduce = 0;
            thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(d_input);
            reduce = thrust::reduce(dev_ptr, dev_ptr + size);
            // Launch the softmax kernel
            softmax_kernel<T><<<gridDim, blockDim>>>(d_input, d_output,reduce, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }

            // Copy the result output from device to host
            if(!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying output from device to host"<<endl;
                return;
            }

            // Free device memory
            if(!HandleCUDAError(cudaFree(d_input))){
                cout<<"Error in freeing d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_output))){
                cout<<"Error in freeing d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }
};



template <typename T>
class Linear: public Matrix<T>
{
    public:
        Linear(int rows, int cols) override { 
            this->rows = rows;
            this->cols = cols;
            this->weights = (T*)malloc(rows * cols * sizeof(T));
            this->biases = (T*)malloc(rows * sizeof(T));
            this->d_weights = (T*)malloc(rows * cols * sizeof(T));
            this->d_biases = (T*)malloc(rows * sizeof(T));
            this->hidden_output = (T*)malloc(rows * sizeof(T));
            this->input = (T*)malloc(cols * sizeof(T));
            this->dX = (T*)malloc(cols * sizeof(T));    
        }
        int rows;
        int cols;
        T* weights;
        T* biases;
        T* d_weights;
        T* d_biases;
        T* hidden_output;
        T* input;
        T* dX;
        ~Linear(){
            free(this->weights);
            free(this->biases);
        }
        void forward(T *input, T *output) override;
        void backward(T * loss) override;
        void update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size);
        void set_weights(T *weights, T *biases);
};



template <typename T>
class Conv2D: public Matrix<T>
{
    public:
        Conv2D(int rows, int cols){
            this->rows = rows;
            this->cols = cols;
            this->weights = (T*)malloc(rows * cols * sizeof(T));
            this->biases = (T*)malloc(rows * sizeof(T));
        }
        int rows;
        int cols;
        T* weights;
        T* biases;
        ~Conv2D(){
            free(this->weights);
            free(this->biases);
        }
        void forward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size);
        void set_weights(T *weights, T *biases);
        void set_kernel_size(int kernel_size);
        void set_weights(T *weights);
        void set_stride(int stride);
        void set_padding(int padding);
        int get_rows();
        int get_cols(){
            return cols;
        }
};

template <typename T>
class MaxPooling2D: public Matrix<T>
{
    public:
        MaxPooling2D(int rows, int cols){
            this->rows = rows;
            this->cols = cols;
        }
        int rows;
        int cols;
        ~MaxPooling2D();
        void forward(T *input, T *output, int size, int output_size);
        void backward(T *input, T *output, int size, int output_size);
};


template <typename T>
__global__ void Binary_Cross_Entropy_Kernel(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        loss[index] = -1 * (input[index] * log(output[index]) + (1 - input[index]) * log(1 - output[index]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        loss[index] = -1 * input[index] * log(output[index]);
    }
}


template <typename T>
__global__ void Mean_Squared_Error_Kernel(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        loss[index] = 0.5 * pow(input[index] - output[index], 2);
    }
}


template <typename T>
__global__ void Mean_Squared_Error_Derivative(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //The input is the output of the network and the output is the ground truth
    if (index < size) {
        loss[index] = input[index] - output[index];
    }
}

template <typename T>
__global__ void Binary_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //The input is the output of the network and the output is the ground truth
    if (index < size) {
        loss[index] = (output[index] - input[index]) / (output[index] * (1 - output[index]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //The input is the output of the network and the output is the ground truth
    if (index < size) {
        loss[index] = -1 * input[index] / output[index];
    }
}

template <typename T>
class Loss: public Matrix<T>
{
    public:
        Loss(){
            this->size = 0;
        }
        Loss(int size){
            this->size = size;
        }
        int size;
        T* loss;
        T* input;
        T* output;
        ~Loss(){};
        void forward(T *input, T *output) override {};
        void backward(T* loss) override {};
};

template <typename T>
class Mean_Squared_Error : public Loss<T>
{
    public:
        Mean_Squared_Error(){
            this->size = 0;
        }
        Mean_Squared_Error(int size):Loss<T>(size){
            this->size = size;
            this->loss = (T*)malloc(size * sizeof(T));
        }
        ~Mean_Squared_Error(){
            free(this->loss);
        }
        int size;
        T* loss;
        void forward(T *input, T *output) override {
            // Allocate device memory for input and output
            T *d_input, *d_output, *d_loss;
            if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_loss"<<endl;
                return;
            }

            // Copy input from host to device
            if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying output from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);

            // Launch the mean squared error kernel
            Mean_Squared_Error_Kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }

            // Copy the result loss from device to host
            if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }

            // Free device memory
            if(!HandleCUDAError(cudaFree(d_input))){
                cout<<"Error in freeing d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_output))){
                cout<<"Error in freeing d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_loss))){
                cout<<"Error in freeing d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }
        void backward(T* input, T* output) override { 
            /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
            cout<< "Mean Squared Error Backward"<<endl;
            T* d_loss;
            T* d_out, *d_gt;
            if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_out, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_out"<<endl;
                return;
            }


            if(!HandleCUDAError(cudaMemcpy(d_out, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMemcpy(d_gt, output, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying output from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);

            // Launch the mean squared error derivative kernel
            Mean_Squared_Error_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }

            // Copy the result loss from device to host
            if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_out))){
                cout<<"Error in freeing d_out"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_gt))){
                cout<<"Error in freeing d_gt"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_loss))){
                cout<<"Error in freeing d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }   
};


template <typename T>
class Binary_CrossEntropy : public Loss<T>
{
    public:
        Binary_CrossEntropy(){
            this->size = 0;
        }
        Binary_CrossEntropy(int size):Loss<T>(size){
            this->size = size;
            this->loss = (T*)malloc(size * sizeof(T));
        }
        ~Binary_CrossEntropy(){
            free(this->loss);
        }
        T* loss;
        int size;
        void forward(T *input, T *output){
            // Allocate device memory for input and output
            T *d_input, *d_output, *d_loss;
            if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_loss"<<endl;
                return;
            }

            // Copy input from host to device
            if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying output from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);

            // Launch the binary cross entropy kernel
            Binary_Cross_Entropy_Kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }
            T* temp = (T*)malloc(size * sizeof(T));
            if(!HandleCUDAError(cudaMemcpy(temp, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }
            T cost_val = 0;
            for(int i = 0; i < size; i++){
                cost_val += temp[i];
            }
            cout<<"Cost: "<<cost_val<<endl;
            // Copy the result loss from device to host
            if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_input))){
                cout<<"Error in freeing d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_output))){
                cout<<"Error in freeing d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_loss))){
                cout<<"Error in freeing d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }
        void backward(T* input, T* output) override {
            /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
            /*The derivative for Binary Cross Entropy is */
            cout<<"Binary Cross Entropy Backward"<<endl;
            T* d_loss;
            T* d_out, *d_gt;
            if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_out, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_out"<<endl;
                return;
            }

            if(!HandleCUDAError(cudaMemcpy(d_out, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMemcpy(d_gt, output, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying output from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);

            // Launch the binary cross entropy derivative kernel
            Binary_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }

            // Copy the result loss from device to host
            if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_out))){
                cout<<"Error in freeing d_out"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_gt))){
                cout<<"Error in freeing d_gt"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_loss))){
                cout<<"Error in freeing d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }
};

template <typename T>
class Categorical: public Loss<T>
{
    public:
        Categorical(){
            this->size = 0;
        }
        Categorical(int size){
            this->size = size;
            this->loss = (T*)malloc(size * sizeof(T));
        }
        ~Categorical(){
            free(this->loss);
        }
        T* loss;
        int size;
        void forward(T *input, T *output) override{
            // Allocate device memory for input and output
            cout<<"Categorical forward"<<endl;
            T *d_input, *d_output, *d_loss;
            if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
                cout<<"Error in allocating memory for d_loss"<<endl;
                return;
            }

            // Copy input from host to device
            if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying input from host to device"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice))){
                cout<<"Error in copying output from host to device"<<endl;
                return;
            }

            // Define grid and block dimensions
            dim3 gridDim(1, 1, 1);
            dim3 blockDim(size, 1, 1);

            // Launch the categorical cross entropy kernel
            Categorical_Cross_Entropy<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error in synchronizing device"<<endl;
                return;
            }

            // Copy the result loss from device to host
            if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
                cout<<"Error in copying loss from device to host"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_input))){
                cout<<"Error in freeing d_input"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_output))){
                cout<<"Error in freeing d_output"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaFree(d_loss))){
                cout<<"Error in freeing d_loss"<<endl;
                return;
            }
            if(!HandleCUDAError(cudaDeviceReset())){
                cout<<"Error in resetting device"<<endl;
                return;
            }
        }
};



template <typename T>
class Network
{
    public:
        Network(int input_size, int* hidden_size, int output_size, int num_layers);
        ~Network(){};
        int input_size;
        int *hidden_size;
        int output_size;
        int num_layers;
        int num_activation;
        float* input;
        float* output;
        thrust::host_vector<Matrix<T>*> layers;  
        thrust::host_vector<Matrix<T>*> activation;  
        thrust::host_vector<Matrix<T>*> d_layers;  
        thrust::host_vector<Matrix<T>*> d_activation;
        thrust::host_vector<float*> loss;
        thrust::host_vector<float*> hidden;
        void backward(T *input, T *output);
        void update_weights(T learning_rate);
        void addLayer(Linear<T> *layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;
        }
        void addLayer(Conv2D<T> *layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;
        }
        void addLayer(MaxPooling2D<T> *layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;
        }
        void addLayer(Sigmoid<T> *layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;

        }   
        void addLayer(RELU_layer<T>* layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;
        }
        void addLayer(Softmax<T>* layer){
            layers.push_back(layer);
            loss.push_back((T*)malloc(layer->rows * sizeof(T)));
            hidden.push_back((T*)malloc(layer->rows * sizeof(T)));
            num_layers++;
        }
        void addLoss(Binary_CrossEntropy<T>* loss){
            layers.push_back(loss);
            num_layers++;
        }
        void addLoss(Mean_Squared_Error<T>* loss){
            layers.push_back(loss);
            num_layers++;
        }
        void addLoss(Categorical<T>* loss){
            layers.push_back(loss);
            num_layers++;
        }
        void train(T *input, T *output, int epochs, T learning_rate);
        void predict(T *input, T *output);
        void set_input_size(int input_size);
        void set_hidden_size(int* hidden_size);
        void set_output_size(int output_size);
        void set_num_layers(int num_layers);
        int get_input_size();
        int* get_hidden_size();
        int get_output_size();
        void forward(T *input, T *output){
            layers[0]->forward(input, layers[0]->hidden_output);
            for(int i = 1; i < layers.size()-1; i++){
                // if(layers[i-1]->hidden_output == NULL || layers[i]->hidden_output == NULL){
                //     cout<<"gay"<<endl;
                //     return;
                // }
                layers[i]->forward(layers[i-1]->hidden_output, layers[i]->hidden_output);
            }
            //Should be the cost layer
            layers[layers.size()-1]->forward(layers[layers.size()-1]->hidden_output, output);
        }
};


template <typename T>
class Bernoulli_Network: public Network<T>
{
    public:
        Bernoulli_Network(int input_size, int* hidden_size, int output_size, int num_layers);
        ~Bernoulli_Network();
        void forward(T *input, T *output);
        void backward(T *input, T *output);
        void update_weights(T learning_rate);
        void train(T *input, T *output, int epochs, T learning_rate);
        void predict(T *input, T *output);
};







template <typename T>
__global__ void matrix_multiply_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        T sum = 0;
        for (int k = 0; k < cols; k++) {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}


template <typename T>
void Matrix<T>::matrix_multiply(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, cols * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, cols * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix multiplication kernel
    matrix_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_add_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_add(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix addition kernel
    matrix_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }
    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_subtract_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_subtract(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix subtraction kernel
    matrix_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_transpose_kernel(T *A, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[col * rows + row] = A[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_transpose(T *A, T *C){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, cols * rows * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix transpose kernel
    matrix_transpose_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, cols * rows * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_scalar_multiply_kernel(T *A, T scalar, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_multiply(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar multiplication kernel
    matrix_scalar_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
__global__ void matrix_scalar_add_kernel(T *A, T scalar, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_add(T *A, T *C,T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar addition kernel
    matrix_scalar_add_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
__global__ void matrix_scalar_subtract_kernel(T *A, T scalar, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] - scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_subtract(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar subtraction kernel
    matrix_scalar_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
void Matrix<T>::matrix_elementwise_multiply(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise multiplication kernel
    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_elementwise_divide_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] / B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_divide(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise division kernel
    matrix_elementwise_divide_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
__global__ void matrix_elementwise_add_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_add(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise addition kernel
    matrix_elementwise_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_elementwise_subtract_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_subtract(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_B, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrices A and B from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying B from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise subtraction kernel
    matrix_elementwise_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_B))){
        cout<<"Error in freeing d_B"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
__global__ void matrix_sum_axis0_kernel(T *A, T *C, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        T sum = 0;
        for (int row = 0; row < rows; row++) {
            sum += A[row * cols + col];
        }
        C[col] = sum;
    }
}

template <typename T>
__global__ void matrix_sum_axis1_kernel(T *A, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {
        T sum = 0;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col];
        }
        C[row] = sum;
    }
}


template <typename T>
void Matrix<T>::matrix_sum(T *A, T *C, int axis){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    if (axis == 0) {
        // Launch the matrix sum along axis 0 kernel
        matrix_sum_axis0_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error in synchronizing device"<<endl;
            return;
        }
    } else if (axis == 1) {
        // Launch the matrix sum along axis 1 kernel
        matrix_sum_axis1_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error in synchronizing device"<<endl;
            return;
        }
    }

    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void matrix_scalar_divide_kernel(T *A, T scalar, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] / scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_divide(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if(!HandleCUDAError(cudaMalloc((void**)&d_A, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_C, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_C"<<endl;
        return;
    }

    // Copy matrix A from host to device
    if(!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying A from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar division kernel
    matrix_scalar_divide_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }

    // Copy the result matrix C from device to host
    if(!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying C from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_A))){
        cout<<"Error in freeing d_A"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_C))){
        cout<<"Error in freeing d_C"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
void Matrix<T>:: set_cols(int cols){
    this->cols = cols;
}   

template <typename T>
void Matrix<T>:: set_rows(int rows){
    this->rows = rows;
}

template <typename T>
int Matrix<T>:: get_cols(){
    return cols;
}

template <typename T>
int Matrix<T>:: get_rows(){
    return rows;
}


template <typename T>
Sigmoid<T>::Sigmoid(int rows, int cols):Matrix<T>(rows, cols){
    this->rows = rows;
    this->cols = 1;
    this->input = (T*)malloc(rows * sizeof(T));
    this->hidden_output = (T*)malloc(rows * sizeof(T));
}


template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = 1 / (1 + exp(-input[index]));
    }
}

template <typename T>
void Sigmoid<T>::forward(T *input, T *output){
    // Allocate device memory for input and output
    int size = rows;
    T *d_input, *d_output;
    // this->input = input;
    if(input == NULL){
        cout<<"Input RELU is NULL"<<endl;
        input = (T*)malloc(size * sizeof(T));
        if(input == NULL){
            cout<<"Input of RELU is NULL"<<endl;
            exit(1);
        }
    }
    if(output == NULL){
        cout<<"Output of RELU is NULL"<<endl;
        output = (T*)malloc(size * sizeof(T));
        if(output == NULL){
            cout<<"Output of RELU is NULL"<<endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * sizeof(T));
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }

    // Copy input from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(size,1, 1);

    // Launch the sigmoid kernel
    sigmoid_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
    // this->hidden_output = output;
    memcpy(this->hidden_output, output, size * sizeof(T));
}



template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] * (1 - input[index]);
    }
}


template <typename T>
void Sigmoid<T>::backward(T * loss){
    // Allocate device memory for input and output
    cout<<"Sigmoid Layer"<<endl;
    T *d_input, *d_output;
    T* d_loss_mat;
    T *input = this->output;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, rows*sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, rows  * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_loss_mat, rows  * sizeof(T)))){
        cout<<"Error in allocating memory for d_loss_mat"<<endl;
        return;
    }

    // Copy input from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(rows, 1);

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, rows);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }

    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_output, d_loss_mat, d_loss_mat, rows, 1);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(loss, d_loss_mat, rows * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_loss_mat))){
        cout<<"Error in freeing d_loss_mat"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
RELU_layer<T>::RELU_layer(int rows, int cols):Matrix<T>(rows, cols){
    this->rows = rows;
    this->cols = 1;
    this->input = (T*)malloc(rows * sizeof(T));
    this->hidden_output = (T*)malloc(rows * sizeof(T));
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] > 0 ? input[index] : 0;
    }
}


template <typename T>
void RELU_layer<T>::forward(T *input, T *output){
    // Allocate device memory for input and output
    int size = rows;
    cout<<"ReLU Layer"<<endl;
    // this->input = input;
    if(input == NULL){
        cout<<"Input RELU is NULL"<<endl;
        input = (T*)malloc(size * sizeof(T));
        if(input == NULL){
            cout<<"Input of RELU is NULL"<<endl;
            exit(1);
        }
    }
    if(output == NULL){
        cout<<"Output of RELU is NULL"<<endl;
        output = (T*)malloc(size * sizeof(T));
        if(output == NULL){
            cout<<"Output of RELU is NULL"<<endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * sizeof(T));
    T *d_input, *d_output;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }

    // Copy input from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the RELU kernel
    RELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
    // this->hidden_output = output;
    if(this->hidden_output == NULL){
        cout<<"Hidden output is NULL for ReLU"<<endl;
        this->hidden_output = (T*)malloc(size * sizeof(T));
    }
    memcpy(this->hidden_output, output, size * sizeof(T));
}


template <typename T>
__global__ void RELU_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] > 0 ? 1 : 0;
    }
}

template <typename T>
void RELU_layer<T>::backward(T * loss){
        // Allocate device memory for input and output
    cout<<"RELU Layer"<<endl;
    T *d_input, *d_output;
    T* d_loss;
    T *input = this->hidden_output;
    int size = this->rows;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, size*sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_loss, size * sizeof(T)))){
        cout<<"Error in allocating memory for d_loss_mat"<<endl;
        return;
    }

    // Copy input from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the sigmoid derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }

    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_output, d_loss, d_loss, size, 1);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_loss))){
        cout<<"Error in freeing d_loss_mat"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
__global__ void linear_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < output_size) {
        T sum = 0;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + index];
        }
        output[index] = sum + biases[index];
    }
}

template <typename T>
void Linear<T>::forward(T *input, T *output){
    // Allocate device memory for input, output, weights, and biases
    int input_size = cols;
    int output_size = rows;

    // this->input = input;
    if(input == NULL){
        cout<<"Input Linear is NULL"<<endl;
        input = (T*)malloc(input_size * sizeof(T));
        if(input == NULL){
            cout<<"Input of RELU is NULL"<<endl;
            exit(1);
        }
    }
    if(output == NULL){
        cout<<"Output of Linear is NULL"<<endl;
        output = (T*)malloc(output_size * sizeof(T));
        if(output == NULL){
            cout<<"Output of RELU is NULL"<<endl;
            exit(1);
        }
    }
    memcpy(this->input, input, input_size * sizeof(T));
    cout<<"Linear Layer"<<endl;
    T *d_input, *d_output, *dev_weights, *dev_biases;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, input_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, output_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_biases, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }

    // Copy input, weights, and biases from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dev_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dev_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the linear kernel
    linear_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, dev_weights, dev_biases, input_size, output_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(dev_weights))){
        cout<<"Error in freeing d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(dev_biases))){
        cout<<"Error in freeing d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
    mempcpy(this->hidden_output, output, output_size * sizeof(T));
}

template <typename T>
__global__ void linear_derivative_kernel(T* loss, T* Weights, T* d_Weights, T* d_biases, T* d_F, T* output, int rows, int cols){

    /*Calculate the weight update for the backward step
    
    We need to update the loss being propogated and also save the change in weights
    This means we will need to pass in the activation from the prior layer, the weights (For W^T), and the loss from the last layer (delta)
    */

   /* Python code to describe:
        x = self.cache  # Retrieve the cached input

        # Compute gradient with respect to input
        dx = np.dot(dout, self.W.T)  # Gradient of the loss w.r.t. input x, delta* W^T

        # Compute gradient with respect to weights
        dW = np.dot(x.T, dout)  # Gradient of the loss w.r.t. weights W, delta* x^T(activation of previous layer)

        # Compute gradient with respect to biases
        db = np.sum(dout, axis=0, keepdims=True)  # Gradient of the loss w.r.t. biases b

        # Update weights and biases (if using a simple gradient descent step)
        learning_rate = 0.01  # Example learning rate
        self.W -= learning_rate * dW  # Update weights
        self.b -= learning_rate * db  # Update biases

        Rows represent the output shape, i.e. the shape of the delta function for the loss
        Columns represent the input shape, i.e. the shape of the activation from the previous layer
    */

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //Multiply the loss by the transpose of the weights (W^T)*delta
    //This is the derivative of the loss with respect to the input
    if (row < rows && col < cols) {
        T sum = 0;
        for (int i = 0; i < rows; i++) {
            sum += loss[i] * Weights[i*cols+col];
        }
        d_F[col]=sum;
    }

    //Multiply the loss by the transpose of the input (x^T)*delta
    //This is the derivative of the loss with respect to the weights
    //Should be an outer product between o_i and delta_j
    if (row < rows && col < cols) {
        d_Weights[row * cols + col] = output[row] * loss[col];
    }
    __syncthreads();
    //Sum the loss to get the derivative of the loss with respect to the biases
    if (row < rows && col < cols) {
        //Is this right?
        d_biases[col]=loss[col];
    }
    __syncthreads();
}


template <typename T>
void Linear<T>::backward(T * loss){
    // Allocate device memory for input, output, weights, and biases
    cout<<"Linear Layer"<<endl;
    T *d_loss, *d_output, *dev_weights, *dev_biases;
    T *dd_weights, *dd_biases;
    int rows = this->rows;
    int cols = this->cols;
    if(!HandleCUDAError(cudaMalloc((void**)&d_loss, cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, rows * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_biases, rows  * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dd_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dd_biases, rows  * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }

    // Copy input, weights, and biases from host to device
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dev_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dev_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dd_weights,d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(dd_biases,d_biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);
    //__global__ void linear_derivative_kernel(T* loss, T* Weights, T* d_Weights, T* d_biases, T* d_F, T* output, int input_size, int output_size, int size)
    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim>>>(d_loss, dev_weights, dd_weights, dd_biases, d_output, d_output, cols, rows);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(loss, d_output, rows * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_loss))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(dev_weights))){
        cout<<"Error in freeing d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(dev_biases))){
        cout<<"Error in freeing d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



template <typename T>
void Linear<T>::set_weights(T *weights, T *biases){
    this->weights = weights;
    this->biases = biases;
}

//Assemble network

template <typename T>
Network<T>::Network(int input_size, int* hidden_size, int output_size, int num_layers){
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->num_layers = num_layers;

}

template <typename T>
void Network<T>::backward(T * input, T * output){
    for(int i = num_layers-1; i > 0; i--){
        this->layers[i]->backward(output);
    }
}

template <typename T>
void Network<T>::update_weights(T learning_rate){
    this->layers[0]->update_weights(this->layers[0]->weights, this->layers[0]->biases, learning_rate, this->input_size, this->hidden_size[0]);
    for(int i = 1; i < num_layers-1; i++){
        this->layers[i]->update_weights(this->layers[i]->weights, this->layers[i]->biases, learning_rate, this->hidden_size[i-1], this->hidden_size[i]);
    }
    this->layers[num_layers-1]->update_weights(this->layers[num_layers-1]->weights, this->layers[num_layers-1]->biases, learning_rate, this->hidden_size[num_layers-1], this->output_size);
}


template <typename T>
void Network<T>::train(T *input, T *output, int epochs, T learning_rate){
    for(int i = 0; i < epochs; i++){
        cout<<"Epoch: "<<i<<endl;
        if(input == NULL || output == NULL){
            cout<<"Input or output is NULL"<<endl;
            return;
        }
        forward(input, output);

        backward(input, output);

        update_weights(learning_rate);
    }
}

template <typename T>
void Network<T>::predict(T *input, T *output){
    forward(input, output);
}

template <typename T>
void Network<T>::set_input_size(int input_size){
    this->input_size = input_size;
}

template <typename T>
void Network<T>::set_hidden_size(int* hidden_size){
    this->hidden_size = hidden_size;
}

template <typename T>
void Network<T>::set_output_size(int output_size){
    this->output_size = output_size;
}

template <typename T>
void Network<T>::set_num_layers(int num_layers){
    this->num_layers = num_layers;
}

template <typename T>
int Network<T>::get_input_size(){
    return input_size;
}

template <typename T>
int* Network<T>::get_hidden_size(){
    return hidden_size;
}

template <typename T>
int Network<T>::get_output_size(){
    return output_size;
}







template <typename T>
__global__ void update_weights_kernel(T *weights, T *biases, T *d_weights, T *d_biases, T learning_rate, int input_size, int output_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < output_size) {
        for (int i = 0; i < input_size; i++) {
            weights[i * output_size + index] -= learning_rate * d_weights[i * output_size + index];
        }
        biases[index] -= learning_rate * d_biases[index];
    }
}


template <typename T>
void Linear<T>::update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size){
    // Allocate device memory for weights, biases, d_weights, and d_biases
    T *d_weights, *d_biases, *d_d_weights, *d_d_biases;
    if(!HandleCUDAError(cudaMalloc((void**)&d_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_biases, rows * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_d_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_d_biases, rows * sizeof(T)))){
        cout<<"Error in allocating memory for d_d_biases"<<endl;
        return;
    }

    // Copy weights, biases, d_weights, and d_biases from host to device
    if(!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_d_weights, d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying d_weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_d_biases, d_biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying d_biases from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the update weights kernel
    update_weights_kernel<T><<<gridDim, blockDim>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, input_size, output_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result weights and biases from device to host
    if(!HandleCUDAError(cudaMemcpy(weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying weights from device to host"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying biases from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_weights))){
        cout<<"Error in freeing d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_biases))){
        cout<<"Error in freeing d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_d_weights))){
        cout<<"Error in freeing d_d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_d_biases))){
        cout<<"Error in freeing d_d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
void Conv2D<T>::set_weights(T* weights, T* biases){
    this->weights;
    this->biases;
}

template <typename T>
void Conv2D<T>::set_stride(int stride){
    this->stride = stride;
}

template <typename T>
void Conv2D<T>::set_padding(int padding){
    this->padding = padding;
}

template <typename T>
int Conv2D<T>::get_rows(){
    return rows;
}

__global__ void d_Gauss_Filter(unsigned char* in, unsigned char* out,int h, int w){
	// __shared__ unsigned char in_shared[16][16];
    int x=threadIdx.x+(blockIdx.x*blockDim.x);
    int y=threadIdx.y+(blockIdx.y*blockDim.y);
	int idx = y * (w-2)+x;
    if(x < (w-2) && y<(h-2)){
            out[idx]+=.0625*in[y*w+x];
            out[idx]+=.125*in[y*w+x+1];
            out[idx]+=.0625*in[y*w+x+2];
            out[idx]+=.125*in[(y+1)*w+x];
            out[idx]+=.25*in[(y+1)*w+x+1];
            out[idx]+=.125*in[(y+1)*w+x+2];
            out[idx]+=.0625*in[(y+2)*w+x];
            out[idx]+=.125*in[(y+2)*w+x+1];
            out[idx]+=.0625*in[(y+2)*w+x+2];
    }
}



#define TILE_WIDTH 16
__global__ void d_Gauss_Filter_v2(unsigned char* in, unsigned char* out,int h, int w){
	// __shared__ unsigned char in_shared[16][16];
    int x=threadIdx.x+(blockIdx.x*blockDim.x);
    int y=threadIdx.y+(blockIdx.y*blockDim.y);
	int idx = y * (w-2)+x;
	__shared__ unsigned char in_s[TILE_WIDTH+2][TILE_WIDTH+2];
	int tx=threadIdx.x;
	int ty =threadIdx.y;
	if(x<w && y<h){
		in_s[ty][tx]=in[y*w+x];
		if(ty>=TILE_WIDTH-2){
			in_s[ty+2][tx]=in[(y+2)*w+x];
		}
		if(tx>=TILE_WIDTH-2){
			in_s[ty][tx+2]=in[y*w+x+2];
		}
		if(tx==TILE_WIDTH-1 && ty ==TILE_WIDTH-1){
			in_s[ty+1][tx+1]=in[(y+1)*w+x+1];
			in_s[ty+1][tx+2]=in[(y+1)*w+x+2];
			in_s[ty+2][tx+1]=in[(y+2)*w+x+1];
			in_s[ty+2][tx+2]=in[(y+2)*w+x+2];
		}
	}
	__syncthreads();
	if(x < (w-2) && y<(h-2) ){
		out[idx]+=.0625*in_s[ty][tx];
		out[idx]+=.125*in_s[ty][tx+1];
		out[idx]+=.0625*in_s[ty][tx+2];
		out[idx]+=.125*in_s[ty+1][tx];
		out[idx]+=.25*in_s[ty+1][tx+1];
		out[idx]+=.125*in_s[ty+1][tx+2];
		out[idx]+=.0625*in_s[(ty+2)][tx];
		out[idx]+=.125*in_s[(ty+2)][tx+1];
		out[idx]+=.0625*in_s[(ty+2)][tx+2];
	}
}

template <typename T>
__global__ void conv2D_kernel(T *input, T *output, T *weights, T *biases, int radius,int width,int height, int out_width, int out_height){
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;
    int outRow = blockIdx.y*blockDim.y+threadIdx.y;
    T Pvalue=0;
    for(int i=0;i<2*radius+1;i++){
        for(int j=0;j<2*radius+1;j++){
            int inRow = outRow-radius+i;
            int inCol = outCol-radius+j;
            if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                Pvalue+=input[inRow*width+inCol]*weights[i*(2*radius+1)+j];
            }
        }
    }
    output[outRow*out_width+outCol]=Pvalue+biases[outRow];
}


template <typename T>
__global__ void conv2D_backward_kernel(T *input, T *output, T *weights, T *biases, T *d_weights, T *d_biases, int radius,int width,int height, int out_width, int out_height){
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;
    int outRow = blockIdx.y*blockDim.y+threadIdx.y;
    for(int i=0;i<2*radius+1;i++){
        for(int j=0;j<2*radius+1;j++){
            int inRow = outRow-radius+i;
            int inCol = outCol-radius+j;
            if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                d_weights[i*(2*radius+1)+j]+=input[inRow*width+inCol]*output[outRow*out_width+outCol];
            }
        }
    }
    d_biases[outRow]+=output[outRow*out_width+outCol];
}


template <typename T>
void Conv2D<T>::forward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, input_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, output_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_biases, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }

    // Copy input, weights, and biases from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);
    int radius = get_cols()/2;
    // Launch the linear kernel
    conv2D_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, radius, input_size, input_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_weights))){
        cout<<"Error in freeing d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_biases))){
        cout<<"Error in freeing d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}


template <typename T>
void Conv2D<T>::backward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    T* d_dweights, *d_dbiases, *d_dinput;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, input_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, output_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_weights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_biases, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_dweights, rows * cols * sizeof(T)))){
        cout<<"Error in allocating memory for d_dweights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_dbiases, rows * sizeof(T)))){
        cout<<"Error in allocating memory for d_dbiases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_dinput, input_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_dinput"<<endl;
        return;
    }


    // Copy input, weights, and biases from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying weights from host to device"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying biases from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the linear kernel
    // Compute the gradients of the weights, biases, and input
    conv2D_backward_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, d_dweights, d_dbiases, d_dinput, input_size, output_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the gradients from device to host
    if (!HandleCUDAError(cudaMemcpy(d_weights, d_dweights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))) {
        cout << "Error in copying dweights from device to host" << endl;
        return;
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, d_dbiases, rows * sizeof(T), cudaMemcpyDeviceToHost))) {
        cout << "Error in copying dbiases from device to host" << endl;
        return;
    }
    if (!HandleCUDAError(cudaMemcpy(d_input, d_dinput, input_size * sizeof(T), cudaMemcpyDeviceToHost))) {
        cout << "Error in copying dinput from device to host" << endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_weights))){
        cout<<"Error in freeing d_weights"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_biases))){
        cout<<"Error in freeing d_biases"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}



// template <typename T>
// void Conv2D<T>::update_weights(T learning_rate){
//     this->weights = this->weights - learning_rate * this->dweights;
//     this->biases = this->biases - learning_rate * this->dbiases;
// }

template <typename T>
__global__ void max_pooling_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index];
        for (int i = 0; i < size; i++) {
            if (input[i] > output[index]) {
                output[index] = input[i];
            }
        }
    }
}

template <typename T>
void MaxPooling2D<T>::forward(T *input, T *output, int input_size, int output_size){
    // Allocate device memory for input and output
    T *d_input, *d_output;
    if(!HandleCUDAError(cudaMalloc((void**)&d_input, input_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_output, output_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_output"<<endl;
        return;
    }

    // Copy input from host to device
    if(!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying input from host to device"<<endl;
        return;
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the max pooling kernel
    max_pooling_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, input_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in synchronizing device"<<endl;
        return;
    }
    // Copy the result output from device to host
    if(!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost))){
        cout<<"Error in copying output from device to host"<<endl;
        return;
    }

    // Free device memory
    if(!HandleCUDAError(cudaFree(d_input))){
        cout<<"Error in freeing d_input"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaFree(d_output))){
        cout<<"Error in freeing d_output"<<endl;
        return;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        cout<<"Error in resetting device"<<endl;
        return;
    }
}