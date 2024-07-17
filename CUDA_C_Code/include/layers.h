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
#include <thrust/host_vector.h>
#include <cusolverDn.h>
#include <cmath>
#include "GPUErrors.h"

#define WEATHER_DATA "../data/weather/weather_classification_data_cleaned.csv"
#define WEATHER_INPUT_SIZE 10
#define WEATHER_OUTPUT_SIZE 4
#define WEATHER_SIZE 13200
#define RANGE_MAX 0.5
#define RANGE_MIN -0.5
#define TRAIN .9
#define TEST .1

void Read_Weather_Data(float **data, float **output)
{
    std::ifstream file(WEATHER_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 10;
    int classes = 4;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}


void Read_Weather_Data_Norm(float **data, float **output)
{
    std::ifstream file(WEATHER_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 10;
    int classes = 4;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
    //Go through the data and normalize it
    //Find the max and min of each column
    float max[WEATHER_INPUT_SIZE];  
    float min[WEATHER_INPUT_SIZE];
    for(int i = 0; i < WEATHER_INPUT_SIZE; i++) {
        max[i] = -1000000;
        min[i] = 1000000;
    }
    for(int i = 0; i < WEATHER_SIZE; i++) {
        for(int j = 0; j < WEATHER_INPUT_SIZE; j++) {
            if(data[i][j] > max[j]) {
                max[j] = data[i][j];
            }
            if(data[i][j] < min[j]) {
                min[j] = data[i][j];
            }
        }
    }
    //Normalize the data
    for(int i = 0; i < WEATHER_SIZE; i++) {
        for(int j = 0; j < WEATHER_INPUT_SIZE; j++) {
            data[i][j] = (data[i][j] - min[j])/(max[j] - min[j]);
        }
    }
}

void Train_Split_Test(float **data, float **output, float **train_data, float **train_output, float **test_data, float **test_output, int size)
{
    int training_size = (int)WEATHER_SIZE * TRAIN;
    int test_size = WEATHER_SIZE - training_size;
    for (int i = 0; i < training_size; i++)
    {
        for (int j = 0; j < WEATHER_INPUT_SIZE; j++)
        {
            train_data[i][j] = data[i][j];
        }
        for (int j = 0; j < WEATHER_OUTPUT_SIZE; j++)
        {
            train_output[i][j] = output[i][j];
        }
    }
    for (int i = training_size; i < size; i++)
    {
        for (int j = 0; j < WEATHER_INPUT_SIZE; j++)
        {
            test_data[i - training_size][j] = data[i][j];
        }
        for (int j = 0; j < WEATHER_OUTPUT_SIZE; j++)
        {
            test_output[i - training_size][j] = output[i][j];
        }
    }
}

struct LayerMetadata
{
    int layerNumber;
    bool isUpdateable;

    LayerMetadata(int number, bool updateable) : layerNumber(number), isUpdateable(updateable) {}
};

template <typename T>
void InitializeMatrix(T *matrix, int ny, int nx)
{

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            matrix[j] = ((T)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
            matrix[j]/=nx;
        }
        matrix += nx;
    }
}

template <typename T>
void InitMatrix_Xavier(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            // srand(time(NULL));
            matrix[j] = ((T)rand() / (RAND_MAX + 1) * (upper - lower) + lower);
        }
        matrix += nx;
    }
}

template <typename T>
void InitMatrix_He(T* matrix, int ny, int nx){
    T upper = sqrt(2.0 / nx);
    T lower = -upper;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            matrix[j] = ((T)rand() / RAND_MAX * (upper - lower) + lower);
        }
        matrix += nx;
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
void InitializeVector(float *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = ((T)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
    }
}

template <typename T>
void ZeroVector(T *vec, int n)
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
    T *input;
    string name;
    Matrix()
    {
        this->rows = 0;
        this->cols = 0;
        this->weights = NULL;
        this->biases = NULL;
        cout << "Calling default constructor" << endl;
        // this->name = "default matrix";
    }
    Matrix(int cols, int rows)
    {
        this->rows = rows;
        this->cols = cols;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(cols * sizeof(T));
        // Create random weights and biases
        // InitializeMatrix<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->weights, rows, cols);
        InitializeVector<T>(this->biases, rows);
        this->name = "full matrix";
        this->next_loss = (T *)malloc(cols * sizeof(T));
        this->d_biases = (T *)malloc(rows * sizeof(T));
        this->d_weights = (T *)malloc(rows * cols * sizeof(T));
    }
    Matrix(int rows)
    {
        this->rows = rows;
        this->cols = 1;
        this->weights = (T *)malloc(rows * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(rows * sizeof(T));
        this->next_loss = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        InitializeVector<T>(this->weights, rows);
        InitializeVector<T>(this->biases, rows);
        ZeroVector<T>(this->input, rows);
        this->name = "vector";
    }

    Matrix(int rows, int cols, T *weights, T *biases)
    {
        this->rows = rows;
        this->cols = cols;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->weights = weights;
        this->biases = biases;
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(cols * sizeof(T));
        this->name = "matrix";
    }
    virtual ~Matrix()
    {
        free(this->weights);
        free(this->biases);
        cout << "Matrix Destructor" << endl;
    }
    void randomize()
    {
        for (int i = 0; i < rows * cols; i++)
        {
            weights[i] = (T)rand() / RAND_MAX;
            if (i < rows)
            {
                biases[i] = (T)rand() / RAND_MAX;
            }
        }
    }
    virtual void set_labels(float *labels) {};
    int rows;
    int cols;
    T *weights;
    T *biases;
    T *d_weights;
    T *d_biases;
    T *hidden_output;
    T *loss;
    T *next_loss;
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
    void virtual backward(T *loss){};
    void backward(T *loss, int size) {};
    void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size) {};
    virtual void update_weights_SGD(T learning_rate) { cout << "Hello" << endl; };
    virtual void update_weights_Momentum(T learning_rate, T momentum) {};
    virtual void update_weights_RMSProp(T learning_rate, T decay_rate) {};
    virtual void update_weights_Adam(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    void train(T *input, T *output, int epochs, T learning_rate) {};
    int get_rows();
    int get_cols();

private:
    cudaError_t cudaStatus;
};

template <typename T>
class Optimizer
{
public:
    Optimizer(T learning_rate, T momentum, T decay_rate, T beta1, T beta2, T epsilon)
    {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        this->decay_rate = decay_rate;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->name = "Optimizer";
    }
    Optimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {};
    Optimizer(T learning_rate, T decay_rate, T epsilon) : Optimizer(learning_rate, 0.0, decay_rate, 0.0, 0.0, epsilon) {};
    Optimizer(T learning_rate) : Optimizer(learning_rate, 0.0, 0.0, 0.0, 0.0, 0.0) {};
    T learning_rate;
    T momentum;
    T decay_rate;
    T beta1;
    T beta2;
    T epsilon;
    string name;
    virtual void update_weights(T *weights, T *d_weights, T *biases, T *d_biases, int rows, int cols){};
};

template <typename T>
class AdamOptimizer : public Optimizer<T>
{
public:
    AdamOptimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "Adam";};
};

template <typename T>
class SGD_Optimizer : public Optimizer<T>
{
public:
    SGD_Optimizer(T learning_rate) : Optimizer<T>(learning_rate, 0.0, 0.0, 0.0, 0.0, 0.0) {this->name = "SGD";};
};

template <typename T>
class MomentumOptimizer : public Optimizer<T>
{
public:
    MomentumOptimizer(T learning_rate, T momentum) : Optimizer<T>(learning_rate, momentum, 0.0, 0.0, 0.0, 0.0) {this->name = "Momentum";};
};

template <typename T>
class RMSPropOptimizer : public Optimizer<T>
{
public:
    RMSPropOptimizer(T learning_rate, T decay_rate, T epsilon) : Optimizer<T>(learning_rate, 0.0, decay_rate, 0.0, 0.0, epsilon) {this -> name = "RMSProp";};
};


template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

template <typename T>
__global__ void vector_elementwise_multiply_kernel(T *A, T *B, T *C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] * B[index];
        // printf("C[%d]: %f\n", index, C[index]);
        // printf("A[%d]: %f\n", index, A[index]);
        // printf("B[%d]: %f\n", index, B[index]);
    }
}

template <typename T>
__global__ void matrix_multiply(T *A, T *B, T *C, int rows, int cols, int inter_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        T sum = 0;
        for (int i = 0; i < inter_size; i++)
        {
            sum += A[row * cols + i] * B[i * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k] * B[k];
        }
        C[row] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_addition_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        C[row] = A[row] + B[row];
    }
}

template <typename T>
void Matrix<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    cout << "Matrix forward" << endl;
    memcpy(this->input, input, cols * sizeof(T));
    // this->input = input;
    T *d_input, *d_output;
    T *d_weights;
    T *d_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for device_data" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Matrix" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying d_data from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions

    int blocksize = 16;
    dim3 blockDim(blocksize, blocksize, 1);

    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the matrix multiplication kernel
    matrix_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_weights, d_input, d_output, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    matrix_vector_addition_kernel<T><<<gridDim, blockDim>>>(d_output, d_biases, d_output, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing device_data" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    memcpy(output, this->hidden_output, rows * sizeof(T));
}

template <typename T>
class Sigmoid : public Matrix<T>
{
public:
    Sigmoid(int rows) : Matrix<T>(rows)
    {
        // this->rows = rows;
        // this->cols = 1;
        // this->input = (T*)malloc(rows * sizeof(T));
        // this->hidden_output = (T*)malloc(rows * sizeof(T));
        // this->next_loss = (T*)malloc(rows * sizeof(T));
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "Sigmoid";
    }
    // int rows;
    // int cols;
    // T* input;
    // T* hidden_output;
    // T* loss;
    // T* next_loss;
    // string name;
    ~Sigmoid()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class RELU_layer : public Matrix<T>
{
public:
    RELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "RELU";
    }
    ~RELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
__global__ void LeakyRELU_kernel(T *input, T *output, T alpha, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] > 0 ? input[index] : alpha * input[index];
    }
}


template <typename T>
__global__ void LeakyRELU_derivative_kernel(T *input, T *output, T alpha, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] > 0 ? 1 : -alpha;
    }
}


template <typename T>
class LeakyRELU_layer : public Matrix<T>
{
public:
    LeakyRELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "LeakyRELU";
    }
    T alpha = 0.01;
    ~LeakyRELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
void LeakyRELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 256;
    dim3 blockDim(TPB, 1, 1);
    dim3 gridDim((size + TPB - 1) / TPB, 1, 1);
    // Launch the LeakyRELU kernel
    LeakyRELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, this->alpha, size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of LeakyRELU is NULL" << endl;
        exit(1);
    }   
    memcpy(output, this->hidden_output, size * sizeof(T));

}


template <typename T>
void LeakyRELU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    // Launch the LeakyRELU derivative kernel
    LeakyRELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss,this->alpha, rows);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    matrix_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_temp_loss, d_loss, d_loss, rows, rows);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_out)))
    {
        cout << "Error in freeing d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_temp_loss)))
    {
        cout << "Error in freeing d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }

}

template <typename T>
__global__ void softmax_kernel(T *input, T *output, T reduce, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] / reduce;
    }
}

template <typename T>
__global__ void softmax_derivative_kernel(T *input, T *output, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size)
    {
        if(row == col) {
            output[row * size + col] = input[row] * (1 - input[col]);
        } else {
            output[row * size + col] = -input[row] * input[col];
        }  

    }
}

template <typename T>
class Softmax : public Matrix<T>
{
public:
    //Need to edit for stability by doing e^(x-max(x))
    Softmax()
    {
        this->rows = 0;
        this->cols = 0;
    }
    Softmax(int cols, int rows) : Matrix<T>(cols, rows)
    {
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "softmax";
    }
    Softmax(int rows) : Matrix<T>(rows)
    {
        this->rows = rows;
        this->cols = 1;
        this->input = (T *)malloc(rows * sizeof(T));
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "softmax";
    }
    ~Softmax()
    {
        free(this->input);
        free(this->hidden_output);
    }
    // T* input;
    // T* hidden_output;
    // T* loss;
    // T* next_loss;
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T max = *thrust::max_element(thrust::host, input, input + this->rows);
        int size = this->rows;
        T *d_input, *d_output;
        if (input == NULL)
        {
            cout << "Input Softmax is NULL" << endl;
            input = (T *)malloc(size * sizeof(T));
            if (input == NULL)
            {
                cout << "Input of Softmax is NULL" << endl;
                exit(1);
            }
        } else {
            for(int i = 0; i<size; i++) {
                this->input[i] = input[i];
                // cout<<"Input["<<i<<"]"<<input[i]<<endl;
            }
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax" << endl;
            exit(1);
        }
        thrust::fill(thrust::device, d_output, d_output + size, (T)0);

        // Define grid and block dimensions
        // Launch the softmax kernel
        // Corrected transformation for applying exp

        //Find max of input

        thrust::transform(thrust::device, d_input, d_input + size, d_output, [max] __device__(T x)
                        { return x - max; });
        thrust::transform(thrust::device, d_input, d_input + size, d_output, [max] __device__(T x)
                        { return exp(x); });

        // Step 3: Sum the exponentials
        T sum = thrust::reduce(thrust::device, d_output, d_output + size, (T)0, thrust::plus<T>());

        // Step 4: Divide each exponential by the sum
        // Corrected transformation for dividing each element by the sum

        int TPB = 256;
        dim3 blockDim(TPB, 1, 1);
        dim3 gridDim((size + TPB - 1) / TPB, 1, 1);


        softmax_kernel<T><<<gridDim, blockDim>>>(d_output, d_output, sum, size);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result output from device to host
        if (!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying output from device to host" << endl;
        }
        // Free device memory
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        if (output == NULL)
        {
            cout << "Output of Softmax is NULL" << endl;
            exit(1);
        }
        memcpy(output, this->hidden_output, size * sizeof(T));
    }
    void backward(T *loss)
    {
        T *d_loss;
        T *d_temp_loss;
        T *d_out;
        int rows = this->rows;
        if (loss == NULL)
        {
            cout << "Loss of Softmax is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_temp_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying loss from host to device, Softmax loss" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        int twodthreadsPerBlock = 16;
        dim3 twodblockDim(twodthreadsPerBlock, twodthreadsPerBlock, 1);

        dim3 twodgridDim((rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, (rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, 1);

        // Launch the softmax derivative kernel
        softmax_derivative_kernel<T><<<twodgridDim, twodblockDim>>>(d_out, d_temp_loss, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        matrix_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_temp_loss, d_loss, d_loss, rows, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        if (this->next_loss == NULL)
        {
            cout << "Next Loss of Softmax is NULL" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host Softmax" << endl;
            // exit(1);
        }
        // cout<<"Softmax Next Loss"<<endl;
        // for(int i = 0; i<rows; i++) {
        //     cout<<"Next Loss["<<i<<"]"<<this->next_loss[i]<<endl;
        // }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_temp_loss)))
        {
            cout << "Error in freeing d_temp_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
__global__ void update_weights_kernel(T *weights, T *biases, T *d_weights, T *d_biases, T learning_rate, int input_size, int output_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < input_size)
    {
        weights[row * input_size + col] -= learning_rate * d_weights[row * input_size + col];
    }
}


template <typename T>
__global__ void update_bias_kernel(T *biases, T *d_biases, T learning_rate, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        biases[index] -= learning_rate * d_biases[index];
    }
}


template <typename T>
__global__ void Adam_Update_Weights(T *weights, T *d_weights, T *m_weights, T *v_weights, T beta1, T beta2, T epsilon, T learning_rate, int input_size, int output_size, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < input_size)
    {
        m_weights[row * input_size + col] = beta1 * m_weights[row * input_size + col] + (1 - beta1) * d_weights[row * input_size + col];
        v_weights[row * input_size + col] = beta2 * v_weights[row * input_size + col] + (1 - beta2) * d_weights[row * input_size + col] * d_weights[row * input_size + col];
        T m_hat = m_weights[row * input_size + col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row * input_size + col] / (1 - pow(beta2, epochs+1));
        weights[row * input_size + col] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        if(weights[row * input_size + col] != weights[row * input_size + col]) {
            printf("NAN detected in weights\n");
            printf("m_hat: %f\n", m_hat);
            printf("v_hat: %f\n", v_hat);
            printf("weights: %f\n", weights[row * input_size + col]);
            printf("d_weights: %f\n", d_weights[row * input_size + col]);
            printf("m_weights: %f\n", m_weights[row * input_size + col]);
            printf("v_weights: %f\n", v_weights[row * input_size + col]);
        }
    }
}

template <typename T>
__global__ void Adam_Update_Bias(T *biases, T *d_biases, T *m_biases, T *v_biases, T beta1, T beta2, T epsilon, T learning_rate, int size, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * d_biases[index];
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * d_biases[index] * d_biases[index];
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}


template <typename T>
class Linear : public Matrix<T>
{
public:
    Linear(int cols, int rows) : Matrix<T>(cols, rows)
    {
        InitMatrix_He<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->biases, rows,1);
        ZeroVector<T>(this->hidden_output, rows);
        ZeroVector<T>(this->input, cols);
        v_weights = (T*)malloc(rows * cols * sizeof(T));
        v_biases = (T*)malloc(rows * sizeof(T));
        m_weights = (T*)malloc(rows * cols * sizeof(T));
        m_biases = (T*)malloc(rows * sizeof(T));
        ZeroMatrix<T>(v_weights, rows, cols);
        ZeroVector<T>(v_biases, rows);
        ZeroMatrix<T>(m_weights, rows, cols);
        ZeroVector<T>(m_biases, rows);
        this->name = "linear";
    }
    ~Linear() override
    {
        free(this->weights);
        free(this->biases);
        free(this->d_weights);
        free(this->d_biases);
        free(this->hidden_output);
        free(this->input);
        cout << "Linear Destructor" << endl;
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
    T* v_weights;
    T* v_biases;
    T* m_weights;
    T* m_biases;
    
    void update_weights_RMSProp(T learning_rate, T decay_rate) override {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((rows + block_size - 1) / block_size, 1, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel

        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Update the weights and biases
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_v_weights, d_v_biases, d_d_weights, d_d_biases, decay_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_v_biases, d_d_biases, decay_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_Momentum(T learning_rate, T momentum) override {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);
        
        dim3 gridDim2D((rows + block_size - 1) / block_size, 1, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Update the weights and biases
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_v_weights, d_v_biases, d_d_weights, d_d_biases, momentum, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_v_biases, d_d_biases, momentum, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Copy the result weights and biases from device to host
        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }   
        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_Adam(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        cout<<"Epoch "<<epochs<<endl;
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((rows + block_size - 1) / block_size, 1, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }

        Adam_Update_Weights<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Adam_Update_Bias<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        
    }
    void update_weights_SGD(T learning_rate) override
    {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((rows + block_size - 1) / block_size, 1, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host
        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void set_weights(T *weights, T *biases);
};

template <typename T>
class Conv2D : public Matrix<T>
{
public:
    Conv2D(int cols, int rows)
    {
        this->rows = rows;
        this->cols = cols;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
    }
    int rows;
    int cols;
    T *weights;
    T *biases;
    ~Conv2D()
    {
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
    int get_cols()
    {
        return cols;
    }
};

template <typename T>
class MaxPooling2D : public Matrix<T>
{
public:
    MaxPooling2D(int cols, int rows)
    {
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
__global__ void Binary_Cross_Entropy_Kernel(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        loss[index] = -1 * (input[index] * log(output[index]) + (1 - input[index]) * log(1 - output[index]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy(T *input, T *output, T *loss, int size)
{
    /*def forward(self, bottom, top):
   labels = bottom[1].data
   scores = bottom[0].data
   # Normalizing to avoid instability
   scores -= np.max(scores, axis=1, keepdims=True)
   # Compute Softmax activations
   exp_scores = np.exp(scores)
   probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
   logprobs = np.zeros([bottom[0].num,1])
   # Compute cross-entropy loss
   for r in range(bottom[0].num): # For each element in the batch
       scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
       for c in range(len(labels[r,:])): # For each class
           if labels[r,c] != 0:  # Positive classes
               logprobs[r] += -np.log(probs[r,c]) * labels[r,c] * scale_factor # We sum the loss per class for each element of the batch

   data_loss = np.sum(logprobs) / bottom[0].num

   self.diff[...] = probs  # Store softmax activations
   top[0].data[...] = data_loss # Store loss*/
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        loss[index] = -1 * input[index] * log(output[index]);
    }
}

template <typename T>
__global__ void Mean_Squared_Error_Kernel(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        loss[index] = 0.5 * pow(input[index] - output[index], 2);
    }
}

template <typename T>
__global__ void Mean_Squared_Error_Derivative(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size)
    {
        loss[index] = input[index] - output[index];
    }
}

template <typename T>
__global__ void Binary_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size)
    {
        loss[index] = (output[index] - input[index]) / (output[index] * (1 - output[index]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size)
    {
        loss[index] = -1 * (output[index] / input[index]);
    }
    __syncthreads();
}

template <typename T>
class Loss : public Matrix<T>
{
public:
    Loss()
    {
        this->size = 0;
        this->loss = NULL;
        this->name = "loss";
    }
    Loss(int size)
    {
        cout << "Loss Constructor" << endl;
    }
    virtual ~Loss(){};
    virtual void forward(T *input, T *output) override {};
    virtual void backward(T *loss) override {};
};

template <typename T>
class Mean_Squared_Error : public Loss<T>
{
public:
    Mean_Squared_Error()
    {
        this->size = 0;
        this->rows = 0;
    }
    Mean_Squared_Error(int size)
    {
        this->size = size;
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "mean_squared_error";
    }
    ~Mean_Squared_Error() override
    {
        free(this->loss);
        free(this->input);
        free(this->output);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_input, *d_output, *d_loss;
        int size = this->rows;
        if (input == NULL)
        {
            cout << "Input of Categorical is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy kernel
        Mean_Squared_Error_Kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(output, this->output, size * sizeof(T));
        memcpy(input, this->input, size * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int size = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy derivative kernel
        Mean_Squared_Error_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
class Binary_CrossEntropy : public Loss<T>
{
public:
    Binary_CrossEntropy()
    {
        this->size = 0;
        this->rows = 0;
    }
    Binary_CrossEntropy(int size)
    {
        this->size = size;
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "binary_cross_entropy";
    }
    ~Binary_CrossEntropy() override
    {
        free(this->loss);
        free(this->input);
        free(this->output);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_input, *d_output, *d_loss;
        int size = this->rows;
        if (input == NULL)
        {
            cout << "Input of Categorical is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy kernel
        Binary_Cross_Entropy_Kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(output, this->output, size * sizeof(T));
        memcpy(input, this->input, size * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int size = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy derivative kernel
        Binary_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
class Categorical : public Loss<T>
{
public:
    T *labels;
    Categorical()
    {
        this->rows = 0;
    }
    Categorical(int size) : Loss<T>(size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->hidden_output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "categorical";
    }
    ~Categorical() override
    {
        free(this->loss);
        free(this->input);
        free(this->hidden_output);
        free(this->next_loss);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_input, *d_output, *d_loss;
        int rows = this->rows;
        // if(input == NULL){
        //     cout<<"Input of Categorical is NULL"<<endl;
        //     exit(1);
        // }

        // if(output == NULL){
        //     cout<<"Output of Categorical is NULL"<<endl;
        //     exit(1);
        // } else{
        //     cout<<"Output of Categorical is not NULL"<<endl;
        // }
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        // cout<<sizeof(output)<<endl;
        // cout<<rows*sizeof(T)<<endl;
        if (!HandleCUDAError(cudaMemcpy(d_output, output, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 256;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy kernel
        Categorical_Cross_Entropy<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(this->hidden_output, input, rows * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->labels, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy derivative kernel
        Categorical_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void set_labels(T *labels) override
    {
        this->labels = labels;
    }
};

template <typename T>
class Network
{
public:
    Network(int input_size, int *hidden_size, int output_size, int num_layers);
    Network(int input_size, int output_size, Optimizer<T>* optimizer){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->optim = optimizer;
    }
    ~Network(){};
    int input_size;
    int *hidden_size;
    int output_size;
    int num_layers;
    int num_activation;
    int num_derv;
    T *input;
    T *prediction;
    Optimizer<T>* optim;
    thrust::host_vector<Matrix<T> *> layers;
    thrust::host_vector<Matrix<T> *> activation;
    thrust::host_vector<T *> loss;
    thrust::host_vector<T *> hidden;
    thrust::host_vector<LayerMetadata> layerMetadata;
    void backward(T *input, T *output)
    {
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            if (i < layers.size())
            { // Ensure i is within bounds
                layers[i]->backward(loss[i]);
                if (i > 0)
                {
                    loss[i - 1] = layers[i]->next_loss;
                }
            }
            else
            {
                cout << "Index " << i << " out of bounds for layers vector." << endl;
            }
        }
    }
    void update_weights(T learning_rate);
    void update_weights(T learning_rate, int epochs);
    void addLayer(Linear<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        layer->name = "saved linear";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->cols * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        layerMetadata.push_back(LayerMetadata(num_layers, true)); // Assuming Linear layers are updateable
        num_layers++;
        num_derv++;
    }
    void addLayer(Conv2D<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLayer(MaxPooling2D<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLayer(Sigmoid<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
        num_layers++;
    }
    void addLayer(RELU_layer<T> *layer)
    {
        layers.push_back(layer);
        layer->name = "saved RELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLayer(LeakyRELU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved LeakyRELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLayer(Softmax<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        layer->name = "saved softmax";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLoss(Binary_CrossEntropy<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLoss(Mean_Squared_Error<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
    }
    void addLoss(Categorical<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        layer->name = "saved categorical";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
        num_layers++;
    }
    void train(T *input, T *output, int epochs, T learning_rate);
    void train(T **input, T **output, int epochs, T learning_rate, int size, int batch_size);
    void predict(T *input, T *output);
    void predict(T **input, T **output, int size);
    void set_input_size(int input_size);
    void set_hidden_size(int *hidden_size);
    void set_output_size(int output_size);
    void set_num_layers(int num_layers);
    int get_input_size();
    int *get_hidden_size();
    int get_output_size();
    void forward(T *input, T *output)
    {
        // for(int i = 0; i<layers.size();i++){
        //     cout<<this->layers[i]->name<<endl;
        //     cout<<this->layers[i]->rows<<endl;
        // }
        layers[0]->forward(input, layers[0]->hidden_output);
        for (int i = 1; i < layers.size() - 1; i++)
        {
            layers[i]->forward(layers[i - 1]->hidden_output, layers[i]->hidden_output);
        }
        // Should be the cost layer
        layers[layers.size() - 1]->forward(layers[layers.size() - 2]->hidden_output, output);
    }
    void getOutput(T *output)
    {
        memcpy(output, prediction, output_size * sizeof(T));
    }
};

template <typename T>
class Bernoulli_Network : public Network<T>
{
public:
    Bernoulli_Network(int input_size, int *hidden_size, int output_size, int num_layers);
    ~Bernoulli_Network();
    void forward(T *input, T *output);
    void backward(T *input, T *output);
    void update_weights(T learning_rate);
    void train(T *input, T *output, int epochs, T learning_rate);
    void predict(T *input, T *output);
};

template <typename T>
__global__ void matrix_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

template <typename T>
void Matrix<T>::matrix_multiply(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, cols * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, cols * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix multiplication kernel
    matrix_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_add_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_add(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix addition kernel
    matrix_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_subtract_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_subtract(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix subtraction kernel
    matrix_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_transpose_kernel(T *A, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[col * rows + row] = A[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_transpose(T *A, T *C)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, cols * rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix transpose kernel
    matrix_transpose_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, cols * rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_multiply_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] * scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_multiply(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix scalar multiplication kernel
    matrix_scalar_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_add_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_add(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix scalar addition kernel
    matrix_scalar_add_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_subtract_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_subtract(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix scalar subtraction kernel
    matrix_scalar_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_multiply(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix elementwise multiplication kernel
    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_divide_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] / B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_divide(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix elementwise division kernel
    matrix_elementwise_divide_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_add_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_add(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix elementwise addition kernel
    matrix_elementwise_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_subtract_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_subtract(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    // Launch the matrix elementwise subtraction kernel
    matrix_elementwise_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_sum_axis0_kernel(T *A, T *C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols)
    {
        T sum = 0;
        for (int row = 0; row < rows; row++)
        {
            sum += A[row * cols + col];
        }
        C[col] = sum;
    }
}

template <typename T>
__global__ void matrix_sum_axis1_kernel(T *A, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows)
    {
        T sum = 0;
        for (int col = 0; col < cols; col++)
        {
            sum += A[row * cols + col];
        }
        C[row] = sum;
    }
}

template <typename T>
void Matrix<T>::matrix_sum(T *A, T *C, int axis)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = (axis == 0) ? rows : cols;
    int output_size = (axis == 0) ? cols : rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);

    if (axis == 0)
    {
        // Launch the matrix sum along axis 0 kernel
        matrix_sum_axis0_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
    }
    else if (axis == 1)
    {
        // Launch the matrix sum along axis 1 kernel
        matrix_sum_axis1_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
    }

    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_divide_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] / scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_divide(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int output_size = rows;
    int input_size = cols;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);
    // Launch the matrix scalar division kernel
    matrix_scalar_divide_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Matrix<T>::set_cols(int cols)
{
    this->cols = cols;
}

template <typename T>
void Matrix<T>::set_rows(int rows)
{
    this->rows = rows;
}

template <typename T>
int Matrix<T>::get_cols()
{
    return cols;
}

template <typename T>
int Matrix<T>::get_rows()
{
    return rows;
}

template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = 1 / (1 + exp(-input[index]));
    }
}

template <typename T>
void Sigmoid<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->rows;
    T *d_input, *d_output;
    // this->input = input;
    if (input == NULL)
    {
        cout << "Input Sigmoid is NULL" << endl;
        input = (T *)malloc(size * sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of Sigmoid is NULL" << endl;
        output = (T *)malloc(size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of Sigmoid is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * sizeof(T));
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Launch the sigmoid kernel
    sigmoid_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    // this->hidden_output = output;
    memcpy(this->hidden_output, output, size * sizeof(T));
}

template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] * (1 - input[index]);
    }
}

template <typename T>
void Sigmoid<T>::backward(T *loss)
{
    // Allocate device memory for input and output
    cout << "Sigmoid Layer" << endl;
    T *d_input, *d_output;
    T *d_loss_mat;
    T *input = this->output;
    int rows = this->rows;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss_mat, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, rows);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    vector_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_output, d_loss_mat, d_loss_mat, rows);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(loss, d_loss_mat, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss_mat)))
    {
        cout << "Error in freeing d_loss_mat" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] > 0 ? input[index] : 0;
    }
}

template <typename T>
void RELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->rows;
    // this->input = input;
    if (input == NULL)
    {
        cout << "Input RELU is NULL" << endl;
        input = (T *)malloc(size * sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of RELU is NULL" << endl;
        output = (T *)malloc(size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of RELU is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * sizeof(T));
    T *d_input, *d_output;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Launch the RELU kernel
    RELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    // this->hidden_output = output;
    memcpy(this->hidden_output, output, size * sizeof(T));

}

template <typename T>
__global__ void RELU_derivative_kernel(T *input, T *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index] > 0 ? 1 : 0;
        // printf("output[%d] = %f\n", index, output[index]);
        // printf("input[%d] = %f\n", index, input[index]);
    }
}

template <typename T>
void RELU_layer<T>::backward(T *loss)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;

    int size = this->rows;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying loss from host to device, ReLU"<<endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    // Launch the sigmoid derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    vector_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_output, d_loss, d_loss, size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if (this->next_loss == NULL)
    {
        cout << "Next loss is NULL for ReLU" << endl;
        this->next_loss = (T *)malloc(size * sizeof(T));
    }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss_mat" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void linear_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < output_size)
    {
        T sum = 0;
        for (int i = 0; i < input_size; i++)
        {
            sum += weights[index * input_size + i] * input[i];
        }
        output[index] = sum + biases[index];
    }
}

template <typename T>
void Linear<T>::forward(T *input, T *output)
{
    // Allocate device memory for input, output, weights, and biases
    int input_size = this->cols;
    int output_size = this->rows;
    memcpy(this->input, input, input_size * sizeof(T));
    // for(int i = 0; i < input_size; i++){
    //     cout<<input[i]<<" ";
    // }
    T *d_input, *d_output, *dev_weights, *dev_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_weights, input_size * output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_biases, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Linear" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_weights, this->weights, input_size * output_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_biases, this->biases, output_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Launch the linear kernel
    linear_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, dev_weights, dev_biases, input_size, output_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of Linear is NULL" << endl;
        if (output == NULL)
        {
            cout << "Output of Linear is NULL" << endl;
        }
    }
    mempcpy(this->hidden_output, output, output_size * sizeof(T));

}

template <typename T>
__global__ void linear_derivative_kernel(T *loss, T *d_Weights, T *output, int rows, int cols)
{

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
    // Multiply the loss by the transpose of the weights (W^T)*delta
    // This is the derivative of the loss with respect to the input


    /*dL/dX, used to pass to the next layer
    If the Weight matrix is NxM, then this matrix will be Nx1
    */
    // Multiply the loss by the transpose of the input (x^T)*delta
    // This is the derivative of the loss with respect to the weights
    // Should be an outer product between o_i and delta_j
    if (row < rows && col < cols)
    {
        d_Weights[row * cols + col] = output[col] * loss[row];
        // if(row == 0){
        //     printf("d_input[%d] = %f\n", col, output[col]);
        // }
    }

    // Sum the loss to get the derivative of the loss with respect to the biases
}


template <typename T>
__global__ void linear_weight_derivatives(T *loss, T *Weights, T *d_F, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols)
    {
        T sum = 0;
        for (int i = 0; i < rows; i++)
        {
            sum += loss[i] * Weights[i * cols + col];
        }
        d_F[col] = sum;
        // printf("d_F[%d] = %f\n", col, d_F[col]);
    }

}

template <typename T>
__global__ void linear_bias_derivatives(T *loss, T *d_biases, int rows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        d_biases[row] = loss[row];
    }
}

template <typename T>
void Linear<T>::backward(T *loss)
{
    // Allocate device memory for input, output, weights, and biases
    // We need to take the loss from the previous layer and calculate the derivative of the loss with respect to the input, weights, and biases
    // Then we need to output the next loss for the layers behind this one
    // cout<<"Linear Backwards"<<endl;
    T *d_loss, *d_output, *dev_weights, *dev_biases;
    T *dd_weights, *dd_biases;
    T *d_F;
    // cout<<"Linear Backwards"<<endl;
    int rows = this->rows;
    int cols = this->cols;
    // cout<<"Linear Rows "<<rows<<endl;
    // cout<<"Linear Cols "<<cols<<endl;
    // for(int i = 0; i < rows; i++){
    //     cout<<loss[i]<<" ";
    // }
    // cout<<endl;
    // cout<<"Input for Linear Backwards"<<endl;
    // for(int i = 0; i < cols; i++){
    //     cout<<this->input[i]<<" ";
    // }
    cout<<endl;
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dd_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dd_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_F, cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_F" << endl;
        exit(1);
    }
    if (loss == NULL)
    {
        cout << "Loss is NULL" << endl;
    }
    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Linear Loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dd_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dd_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_output,this->input,cols*sizeof(T),cudaMemcpyHostToDevice))){
        cout<<"Error in copying output from host to device"<<endl;
        exit(1);
    }

    // Create three streams to (1) Find the weight update (2) Find the bias update (3) Find the next loss

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid_Rows = (rows + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_Cols = (cols + threadsPerBlock - 1) / threadsPerBlock;

    // Define the streams
    cudaStream_t stream1, stream2, stream3;
    if (!HandleCUDAError(cudaStreamCreate(&stream1)))
    {
        cout << "Error in creating stream1" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaStreamCreate(&stream2)))
    {
        cout << "Error in creating stream2" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaStreamCreate(&stream3)))
    {
        cout << "Error in creating stream3" << endl;
        exit(1);
    }

    int output_size = rows;
    int input_size = cols;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size,1);
    dim3 gridDim((output_size + block_size - 1) / block_size, (input_size + block_size - 1) / block_size, 1);
    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim,0,stream1>>>(d_loss, dd_weights, d_output, rows, cols);
    if (!HandleCUDAError(cudaStreamSynchronize(stream1)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_weight_derivatives<T><<<blocksPerGrid_Cols,threadsPerBlock,0,stream2>>>(d_loss,dev_weights,d_F,rows,cols);
    if (!HandleCUDAError(cudaStreamSynchronize(stream2)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_bias_derivatives<T><<<blocksPerGrid_Rows,threadsPerBlock,0,stream3>>>(d_loss,dd_biases,rows);
    if (!HandleCUDAError(cudaStreamSynchronize(stream3)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }


    //Handle the streams destruction
    if(!HandleCUDAError(cudaStreamDestroy(stream1))){
        cout<<"Error in destroying stream1"<<endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaStreamDestroy(stream2))){
        cout<<"Error in destroying stream2"<<endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaStreamDestroy(stream3))){
        cout<<"Error in destroying stream3"<<endl;
        exit(1);
    }

    if (!HandleCUDAError(cudaMemcpy(this->d_weights, dd_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->d_biases, dd_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_F, cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dd_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dd_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_F)))
    {
        cout << "Error in freeing d_F" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Linear<T>::set_weights(T *weights, T *biases)
{
    this->weights = weights;
    this->biases = biases;
}

// Assemble network

template <typename T>
Network<T>::Network(int input_size, int *hidden_size, int output_size, int num_layers)
{
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->num_layers = num_layers;
    this->prediction = (T *)malloc(output_size * sizeof(T));
}

template <typename T>
void Network<T>::predict(T *input, T *output)
{
    forward(input, output);
}

template <typename T>
void Network<T>::predict(T **input, T **output, int size)
{
    T *prediction = (T *)malloc(output_size * sizeof(T));
    float sum = 0;
    for (int i = 0; i < 100; i++)
    {
        forward(input[i], output[i]);
        // take the hidden output, and measure accuracy
        prediction = layers[layers.size() - 1]->hidden_output;
        for (int j = 0; j < output_size; j++)
        {
            cout << prediction[j] << " ";
            cout << output[i][j] << " ";
        }
        // Find the max value in the prediction
        int max_index = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (prediction[j] > prediction[max_index])
            {
                max_index = j;
            }
        }
        // Find the max value in the output
        int max_index_output = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (output[i][j] > output[i][max_index_output])
            {
                max_index_output = j;
            }
        }
        // check if the max index is the same as the max index output
        if (max_index == max_index_output)
        {
            sum++;
            cout << "Correct Prediction for input " << i << endl;
        }
        else
        {
            cout << "Incorrect Prediction for input " << i << endl;
        }
    }
    float accuracy = sum / size;
    cout << "Accuracy: " << accuracy << endl;
}

template <typename T>
void Network<T>::set_input_size(int input_size)
{
    this->input_size = input_size;
}

template <typename T>
void Network<T>::set_hidden_size(int *hidden_size)
{
    this->hidden_size = (T *)malloc((num_layers - 2) * sizeof(T));
    this->hidden_size = hidden_size;
}

template <typename T>
void Network<T>::set_output_size(int output_size)
{
    this->output_size = output_size;
}

template <typename T>
void Network<T>::set_num_layers(int num_layers)
{
    this->num_layers = num_layers;
}

template <typename T>
int Network<T>::get_input_size()
{
    return input_size;
}

template <typename T>
int *Network<T>::get_hidden_size()
{
    return hidden_size;
}

template <typename T>
int Network<T>::get_output_size()
{
    return output_size;
}

template <typename T>
void Network<T>::update_weights(T learning_rate, int epochs)
{
    // Ensure layers vector is not empty and is properly initialized
    if (this->layers.empty())
    {
        std::cerr << "Error: Layers vector is empty.\n";
        return;
    }

    // Iterate over each entry in the layerMetadata vector
    for (int i = 0; i < layerMetadata.size(); i++)
    {
        // Validate layerNumber is within bounds
        if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
        {
            // Check if the layer pointer is not null
            if (this->layers[layerMetadata[i].layerNumber] != nullptr)
            {
                // Check if the current layer is marked as updateable
                if (layerMetadata[i].isUpdateable)
                {
                    // Call the update_weights method on the updateable layer
                    if(this->optim->name == "SGD"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_SGD(learning_rate);
                    }
                    else if(this->optim->name == "Adam"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_Adam(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                    }
                    else if(this->optim->name == "RMSProp"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_RMSProp(learning_rate, this->optim->decay_rate);
                    }
                    else if(this->optim->name == "Momentum"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_Momentum(learning_rate, this->optim->momentum);
                    }
                    else{
                        cout<<"Optimizer not found"<<endl;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Layer at index " << layerMetadata[i].layerNumber << " is a null pointer.\n";
            }
        }
        else
        {
            std::cerr << "Error: layerNumber out of bounds: " << layerMetadata[i].layerNumber << "\n";
        }
    }
}



template <typename T>
int argmax(T *arr, int size)
{
     return static_cast<int>(std::distance(arr, max_element(arr, arr+size)));
}

template <typename T>
void Network<T>::train(T *input, T *output, int epochs, T learning_rate)
{
    for (int i = 0; i < epochs; i++)
    {
        cout << "Epoch: " << i << endl;
        forward(input, output);

        backward(input, output);

        update_weights(learning_rate);
        cout << endl;
    }
    for (int i = 0; i < output_size; i++)
    {
        cout << layers[layers.size() - 1]->hidden_output[i] << " ";
    }
    cout << endl;
    memcpy(this->prediction, layers[layers.size() - 1]->hidden_output, output_size * sizeof(T));
}

template <typename T>
void Network<T>::train(T **input, T **output, int epochs, T learning_rate, int size, int batch_size)
{
    // Find a random list of indices for the batch size
    //  Create a thrust vector of indices
    int* indices = (int*)malloc(batch_size*sizeof(int));
    // Fill the vector with random_indices
    // Iterate through the indices and train the network
    int pred_idx = 0;
    int gt_idx = 0;
    int sum = 0;

    for (int i = 0; i < 1; i++)
    {
        for (int k = 0; k < batch_size; k++)
        {
            indices[k] = rand() % size;
            // cout << "Index k is " << indices[k] << endl;
        }
        sum = 0;
        cout << "Epoch: " << i << endl;
        for (int j = 0; j < 2; j++)
        {
            cout << "Batch: " << j << endl;
            layers[layers.size() - 1]->set_labels(output[indices[j]]);
            cout<< "Input:"<<endl;
            for(int i = 0; i < input_size; i++){
                cout<<input[indices[j]][i]<<" ";
            }   
            cout<<endl;
            forward(input[indices[j]], output[indices[j]]);
            cout << "Prediction: " << endl;
            for (int k = 0; k < output_size; k++)
            {
                cout << layers[layers.size() - 1]->hidden_output[k] << ", ";
            }
            cout << endl;
            cout << "Ground Truth: ";
            for (int k = 0; k < output_size; k++)
            {
                cout << output[indices[j]][k] << " ";
            }
            cout << endl;
            for (int k = 0; k < layerMetadata.size(); k++)
            {
                // Validate layerNumber is within bounds
                if (layerMetadata[k].layerNumber >= 0 && layerMetadata[k].layerNumber < this->layers.size())
                {
                    // Check if the layer pointer is not null
                    if (this->layers[layerMetadata[k].layerNumber] != nullptr)
                    {
                        // Check if the current layer is marked as updateable
                        if (layerMetadata[k].isUpdateable)
                        {   
                            cout<<"Weight before update"<<endl;
                            cout<<"Layer "<<layerMetadata[k].layerNumber<<endl;
                            cout<<"Rows "<<layers[layerMetadata[k].layerNumber]->rows<<endl;
                            cout<<"Cols "<<layers[layerMetadata[k].layerNumber]->cols<<endl;
                            cout<<"[";
                            for(int l = 0; l<layers[layerMetadata[k].layerNumber]->rows; l++){
                                cout<<"[";
                                for(int m = 0; m<layers[layerMetadata[k].layerNumber]->cols; m++){
                                    cout<<layers[layerMetadata[k].layerNumber]->weights[l*layers[layerMetadata[k].layerNumber]->cols + m]<<",";
                                }
                                cout<<"]";
                                cout<<endl;
                                cout<<endl;
                            }
                            cout<<"]";
                            cout<<endl;

                            cout<<"Biases before update"<<endl;
                            cout<<"Layer "<<k<<endl;
                            cout<<"[";
                            for(int l = 0; l<layers[layerMetadata[k].layerNumber]->rows; l++){
                                cout<<layers[layerMetadata[k].layerNumber]->biases[l]<<", ";
                            }
                            cout<<"]";
                            cout<<endl;
                            
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Layer at index " << layerMetadata[i].layerNumber << " is a null pointer.\n";
                    }
                }
                else
                {
                    std::cerr << "Error: layerNumber out of bounds: " << layerMetadata[i].layerNumber << "\n";
                }
            }

            backward(input[indices[j]], output[indices[j]]);
            update_weights(learning_rate, i);
            // for (int k = 0; k < layerMetadata.size(); k++)
            // {
            //     // Validate layerNumber is within bounds
            //     if (layerMetadata[k].layerNumber >= 0 && layerMetadata[k].layerNumber < this->layers.size())
            //     {
            //         // Check if the layer pointer is not null
            //         if (this->layers[layerMetadata[k].layerNumber] != nullptr)
            //         {
            //             // Check if the current layer is marked as updateable
            //             if (layerMetadata[k].isUpdateable)
            //             {   
            //                 cout<<"Weight Gradient"<<endl;
            //                 cout<<"Layer "<<layerMetadata[k].layerNumber<<endl;
            //                 cout<<"Rows "<<layers[layerMetadata[k].layerNumber]->rows<<endl;
            //                 cout<<"Cols "<<layers[layerMetadata[k].layerNumber]->cols<<endl;
            //                 cout<<"[";
            //                 for(int l = 0; l<layers[layerMetadata[k].layerNumber]->rows; l++){
            //                     cout<<"[";
            //                     for(int m = 0; m<layers[layerMetadata[k].layerNumber]->cols; m++){
            //                         cout<<layers[layerMetadata[k].layerNumber]->weights[l*layers[layerMetadata[k].layerNumber]->cols + m]<<",";
            //                     }
            //                     cout<<"]";
            //                     cout<<endl;
            //                     cout<<endl;
            //                 }
            //                 cout<<"]";
            //                 cout<<endl;

            //                 cout<<"Bias Gradient"<<endl;
            //                 cout<<"Layer "<<k<<endl;
            //                 cout<<"[";
            //                 for(int l = 0; l<layers[layerMetadata[k].layerNumber]->rows; l++){
            //                     cout<<layers[layerMetadata[k].layerNumber]->biases[l]<<", ";
            //                 }
            //                 cout<<"]";
            //                 cout<<endl;
                            
            //             }
            //         }
            //         else
            //         {
            //             std::cerr << "Error: Layer at index " << layerMetadata[i].layerNumber << " is a null pointer.\n";
            //         }
            //     }
            //     else
            //     {
            //         std::cerr << "Error: layerNumber out of bounds: " << layerMetadata[i].layerNumber << "\n";
            //     }
            // }
            pred_idx = argmax<float>(layers[layers.size() - 1]->hidden_output, output_size);
            // cout<<"Prediction: "<<pred_idx<<endl;
            // for(int k = 0; k < output_size; k++){
            //     cout<<layers[layers.size() - 1]->hidden_output[k]<<" ";
            // }
            // cout<<endl;
            gt_idx = argmax<float>(output[indices[j]], output_size);
            // cout<<"Ground Truth: "<<gt_idx<<endl;
            // for(int k = 0; k < output_size; k++){
            //     cout<<output[indices[j]][k]<<" ";
            // }
            // cout<<endl;
            if (pred_idx == gt_idx)
            {
                sum++;
            }
        }
        cout << "Accuracy: " << sum / batch_size << endl;
        cout << endl;
    }
}


template <typename T>
void Conv2D<T>::set_weights(T *weights, T *biases)
{
    this->weights;
    this->biases;
}

template <typename T>
void Conv2D<T>::set_stride(int stride)
{
    this->stride = stride;
}

template <typename T>
void Conv2D<T>::set_padding(int padding)
{
    this->padding = padding;
}

template <typename T>
int Conv2D<T>::get_rows()
{
    return rows;
}

__global__ void d_Gauss_Filter(unsigned char *in, unsigned char *out, int h, int w)
{
    // __shared__ unsigned char in_shared[16][16];
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int idx = y * (w - 2) + x;
    if (x < (w - 2) && y < (h - 2))
    {
        out[idx] += .0625 * in[y * w + x];
        out[idx] += .125 * in[y * w + x + 1];
        out[idx] += .0625 * in[y * w + x + 2];
        out[idx] += .125 * in[(y + 1) * w + x];
        out[idx] += .25 * in[(y + 1) * w + x + 1];
        out[idx] += .125 * in[(y + 1) * w + x + 2];
        out[idx] += .0625 * in[(y + 2) * w + x];
        out[idx] += .125 * in[(y + 2) * w + x + 1];
        out[idx] += .0625 * in[(y + 2) * w + x + 2];
    }
}

#define TILE_WIDTH 16
__global__ void d_Gauss_Filter_v2(unsigned char *in, unsigned char *out, int h, int w)
{
    // __shared__ unsigned char in_shared[16][16];
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int idx = y * (w - 2) + x;
    __shared__ unsigned char in_s[TILE_WIDTH + 2][TILE_WIDTH + 2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (x < w && y < h)
    {
        in_s[ty][tx] = in[y * w + x];
        if (ty >= TILE_WIDTH - 2)
        {
            in_s[ty + 2][tx] = in[(y + 2) * w + x];
        }
        if (tx >= TILE_WIDTH - 2)
        {
            in_s[ty][tx + 2] = in[y * w + x + 2];
        }
        if (tx == TILE_WIDTH - 1 && ty == TILE_WIDTH - 1)
        {
            in_s[ty + 1][tx + 1] = in[(y + 1) * w + x + 1];
            in_s[ty + 1][tx + 2] = in[(y + 1) * w + x + 2];
            in_s[ty + 2][tx + 1] = in[(y + 2) * w + x + 1];
            in_s[ty + 2][tx + 2] = in[(y + 2) * w + x + 2];
        }
    }
    __syncthreads();
    if (x < (w - 2) && y < (h - 2))
    {
        out[idx] += .0625 * in_s[ty][tx];
        out[idx] += .125 * in_s[ty][tx + 1];
        out[idx] += .0625 * in_s[ty][tx + 2];
        out[idx] += .125 * in_s[ty + 1][tx];
        out[idx] += .25 * in_s[ty + 1][tx + 1];
        out[idx] += .125 * in_s[ty + 1][tx + 2];
        out[idx] += .0625 * in_s[(ty + 2)][tx];
        out[idx] += .125 * in_s[(ty + 2)][tx + 1];
        out[idx] += .0625 * in_s[(ty + 2)][tx + 2];
    }
}

template <typename T>
__global__ void conv2D_kernel(T *input, T *output, T *weights, T *biases, int radius, int width, int height, int out_width, int out_height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    T Pvalue = 0;
    for (int i = 0; i < 2 * radius + 1; i++)
    {
        for (int j = 0; j < 2 * radius + 1; j++)
        {
            int inRow = outRow - radius + i;
            int inCol = outCol - radius + j;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
            {
                Pvalue += input[inRow * width + inCol] * weights[i * (2 * radius + 1) + j];
            }
        }
    }
    output[outRow * out_width + outCol] = Pvalue + biases[outRow];
}

template <typename T>
__global__ void conv2D_backward_kernel(T *input, T *output, T *weights, T *biases, T *d_weights, T *d_biases, int radius, int width, int height, int out_width, int out_height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = 0; i < 2 * radius + 1; i++)
    {
        for (int j = 0; j < 2 * radius + 1; j++)
        {
            int inRow = outRow - radius + i;
            int inCol = outCol - radius + j;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
            {
                d_weights[i * (2 * radius + 1) + j] += input[inRow * width + inCol] * output[outRow * out_width + outCol];
            }
        }
    }
    d_biases[outRow] += output[outRow * out_width + outCol];
}

template <typename T>
void Conv2D<T>::forward(T *input, T *output, T *weights, T *biases, int input_size, int output_size)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);
    int radius = get_cols() / 2;
    // Launch the linear kernel
    conv2D_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, radius, input_size, input_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Conv2D<T>::backward(T *input, T *output, T *weights, T *biases, int input_size, int output_size)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    T *d_dweights, *d_dbiases, *d_dinput;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_dweights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_dweights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_dbiases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_dbiases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_dinput, input_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_dinput" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the linear kernel
    // Compute the gradients of the weights, biases, and input
    conv2D_backward_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, d_dweights, d_dbiases, d_dinput, input_size, output_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the gradients from device to host
    if (!HandleCUDAError(cudaMemcpy(d_weights, d_dweights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dweights from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, d_dbiases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dbiases from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_input, d_dinput, input_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dinput from device to host" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

// template <typename T>
// void Conv2D<T>::update_weights(T learning_rate){
//     this->weights = this->weights - learning_rate * this->dweights;
//     this->biases = this->biases - learning_rate * this->dbiases;
// }

template <typename T>
__global__ void max_pooling_kernel(T *input, T *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index] = input[index];
        for (int i = 0; i < size; i++)
        {
            if (input[i] > output[index])
            {
                output[index] = input[i];
            }
        }
    }
}

template <typename T>
void MaxPooling2D<T>::forward(T *input, T *output, int input_size, int output_size)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((output_size + block_size - 1) / block_size, 1, 1);

    // Launch the max pooling kernel
    max_pooling_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, input_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}