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
#include "CImg.h"
using namespace cimg_library;

#define WEATHER_DATA "../data/weather/weather_classification_data_cleaned.csv"
#define HEART_DATA "../data/heart/heart_classification_data_cleaned.csv"
#define DUMMY_DATA "../data/dummy/dummy.csv"
#define RICE_DATA_FOLDER "../data/Rice_Image_Dataset"
#define ARBORIO_RICE_FOLDER "/Arborio/"
#define BASMATI_RICE_FOLDER "/Basmati/"
#define IPSALA_RICE_FOLDER "/Ipsala/"
#define JASMINE_RICE_FOLDER "/Jasmine/"
#define KARACADAG_RICE_FOLDER "/Karacadag/"
#define ARBORIO_FILE RICE_DATA_FOLDER + ARBORIO_RICE_FOLDER + "Arborio ("
#define BASMATI_FILE RICE_DATA_FOLDER + BASMATI_RICE_FOLDER + "Basmati ("
#define IPSALA_FILE RICE_DATA_FOLDER + IPSALA_RICE_FOLDER + "Ipsala ("
#define JASMINE_FILE RICE_DATA_FOLDER + JASMINE_RICE_FOLDER + "Jasmine ("
#define KARACADAG_FILE RICE_DATA_FOLDER + KARACADAG_RICE_FOLDER + "Karacadag ("
#define RICE_TYPE_SIZE 15000
#define HEART_INPUT_SIZE 13
#define WEATHER_INPUT_SIZE 10
#define WEATHER_OUTPUT_SIZE 4
#define HEART_OUTPUT_SIZE 2
#define WEATHER_SIZE 13200
#define HEART_SIZE 303
#define DUMMY_SIZE 4
#define DUMMY_INPUT_SIZE 4
#define DUMMY_OUTPUT_SIZE 2
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

template <typename T>
void Read_Heart_Data(T **data, T **output)
{
    std::ifstream file(HEART_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 13;
    int classes = 2;

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

template <typename T>
void Read_Dummy_Data(T **data, T **output)
{
    std::ifstream file(DUMMY_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = DUMMY_INPUT_SIZE;
    int classes = DUMMY_OUTPUT_SIZE;

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
                    cout<<"Value: "<<std::stof(value)<<endl;
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

template <typename T>
void Train_Split_Test(T **data, T **output, T **train_data, T **train_output, T **test_data, T **test_output, int training_size, int test_size, int size, int input_size, int output_size)
{
    for (int i = 0; i < training_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            train_data[i][j] = data[i][j];
        }
        for (int j = 0; j < output_size; j++)
        {
            train_output[i][j] = output[i][j];
        }
    }
    if(test_size == 0) {
        return;
    }
    else {
        cout<<"Test size: "<<test_size<<endl;
        for (int i = training_size; i < size; i++)
        {
            for (int j = 0; j < input_size; j++)
            {
                test_data[i - training_size][j] = data[i][j];
            }
            for (int j = 0; j < output_size; j++)
            {
                test_output[i - training_size][j] = output[i][j];
            }
        }
    }
}

struct LayerMetadata
{
    int layerNumber;
    int LinNumber;
    bool isUpdateable;

    LayerMetadata(int number, bool updateable) : layerNumber(number), isUpdateable(updateable) {}
    LayerMetadata(int number, int linNumber, bool updateable) : layerNumber(number), LinNumber(linNumber), isUpdateable(updateable) {}
};

template <typename T>
struct Loc_Layer
{
    int row;
    int col;
    int layer;
    T weights_dW;
    int rank;
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
void Format_Batch_Data(T **data, T **output, T *batch_data, T *batch_output, int* indices, int batch_size, int input_size, int output_size)
{
    // Store the data in a 1D array, as an input to the neural network.
    //This should be like a matrix, where the entry of the first row includes the first entries of each input
    //The entry of the second row includes the second entries of each input, etc.
    // In the given data, each entry is given in a row
    for(int i = 0; i<input_size; i++){
        for(int j = 0; j<batch_size; j++){
            batch_data[i*batch_size + j] = data[indices[j]][i];
        }
    }
    for(int i = 0; i<output_size; i++){
        for(int j = 0; j<batch_size; j++){
            batch_output[i*batch_size + j] = output[indices[j]][i];
        }
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
        this->loss = (T *)malloc(rows * sizeof(T));
        this->B_weights = (T *)malloc(rows * cols * sizeof(T));
        this->B_biases = (T *)malloc(rows * sizeof(T));
        this->W_dW_weights = (T *)malloc(rows * cols * sizeof(T));
        this->W_dW_biases = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        // InitializeMatrix<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->weights, rows, cols);
        InitializeVector<T>(this->biases, rows);
        this->name = "full matrix";
        this->next_loss = (T *)malloc(cols * sizeof(T));
        this->d_biases = (T *)malloc(rows * sizeof(T));
        this->d_weights = (T *)malloc(rows * cols * sizeof(T));
    }
    Matrix(int cols, int rows, int batch_size)
    {
        this->batch_size = batch_size;
        this->rows = rows;
        this->cols = cols;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->B_weights = (T *)malloc(rows * cols * sizeof(T));
        this->B_biases = (T *)malloc(rows * sizeof(T));
        this->W_dW_weights = (T *)malloc(rows * cols * sizeof(T));
        this->W_dW_biases = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        // InitializeMatrix<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->weights, rows, cols);
        InitializeVector<T>(this->biases, rows);
        this->name = "full matrix";
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
    virtual void set_Bernoulli(int row, int col) {};
    virtual void Fill_Bernoulli(){};
    int rows;
    int cols;
    int batch_size;
    T *weights;
    T *biases;
    T *d_weights;
    T *d_biases;
    T *hidden_output;
    T *loss;
    T *next_loss;
    T* B_weights;
    T* B_biases;
    T* W_dW_weights;
    T* W_dW_biases;
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
    virtual void update_weights_AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void find_Loss_Metric() {};
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
class AdamWBernoulli: public Optimizer<T>
{
public:
    AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "AdamWBernoulli";};
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
__global__ void matrix3D_vector_multiply_kernel(T *A, T *B, T *C, int rows, int cols, int depth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    //The 3D matrix is composed as such: The first row*col entries correspond to the first batch, the second row*col entries correspond to the second batch, etc.
    //The entries of B are such that each column corresponds to a batch
    if (row < rows && batch < depth)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k + batch * rows * cols] * B[k*depth + batch];
        }
        C[row*depth+batch] = sum;
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
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "Sigmoid";
    }
    Sigmoid(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Sigmoid";
    }
    ~Sigmoid()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
class Tanh : public Matrix<T>
{
public:
    Tanh(int rows) : Matrix<T>(rows)
    {
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "Tanh";
    }
    Tanh(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Tanh";
    }
    ~Tanh()
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
    RELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"ReLU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        ZeroVector(this->loss, rows*batch_size);
        ZeroVector(this->next_loss, rows*batch_size);
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
__global__ void LeakyRELU_kernel(T *input, T *output, T alpha, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : alpha * input[index*batch_size + batch];
    }
}

template <typename T>
__global__ void ELU_kernel(T *input, T *output, T alpha, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : alpha * (exp(input[index*batch_size + batch])-1);
    }
}

template <typename T>
__global__ void SLU_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    //SLU(x) = max(0,min(1,x))
    if (index < size)
    {
        output[index*batch_size + batch] = max(0,min(1,input[index*batch_size + batch])); 
    }
}

template <typename T>
__global__ void Tanh_kernel(T* input, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = tanh(input[index*batch_size+batch]);
    }
}

template <typename T>
__global__ void LeakyRELU_derivative_kernel(T *input, T* loss, T *output, T alpha, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size + batch] : -alpha*loss[index*batch_size + batch];
    }
}

template <typename T>
__global__ void ELU_derivative_kernel(T *input, T* loss, T *output, T alpha, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size + batch] : (input[index*batch_size+batch] + alpha)*loss[index*batch_size + batch];
    }
}


template <typename T>
__global__ void SLU_derivative_kernel(T* input, T* loss, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0  && input[index*batch_size+batch] < 1 ? loss[index*batch_size+batch] : 0;
    }
}

template <typename T> 
__global__ void Tanh_derivative_kernel(T* input, T* loss, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = (1 - input[index*batch_size+batch]*input[index*batch_size+batch])*loss[index*batch_size+batch];
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
    LeakyRELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"LeakyRELU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
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
class ELU_layer : public Matrix<T>
{
public:
    ELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "ELU";
    }
    ELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"ELU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        this->name = "ELU";
    }
    T alpha = 0.01;
    ~ELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
class SLU_layer : public Matrix<T>
{
public:
    SLU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "ELU";
    }
    SLU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"SLU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        this->name = "ELU";
    }
    T alpha = 0.01;
    ~SLU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
void SLU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    SLU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void SLU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    SLU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void ELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    ELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, this->alpha, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void ELU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    ELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, this-> alpha, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void Tanh<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    Tanh_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void Tanh<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    Tanh_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
void LeakyRELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    LeakyRELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, this->alpha, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    LeakyRELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss,d_loss,this->alpha, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
__global__ void Col_Wise_Reduce(T* input, T* red, T* max, int cols, int batch_size){
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch<batch_size){
        for(int i = 0; i<cols; i++){
            input[i*batch_size + batch] = input[i*batch_size + batch] - max[batch];
            input[i*batch_size + batch] = exp(input[i*batch_size + batch]);
        }
    }
    __syncthreads();
    if(batch < batch_size){
        T sum = 0;
        for(int i = 0; i < cols; i++){
            sum += input[i*batch_size + batch];
        }
        red[batch] = sum;
    }
    __syncthreads();
    if(batch < batch_size){
        for(int i = 0; i < cols; i++){
            input[i*batch_size + batch] = input[i*batch_size + batch] / red[batch];
        }
    }
}

template <typename T>
__global__ void softmax_kernel(T *input, T *output, T* reduce, int size, int batch_size)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] / reduce[batch];
        // printf("Output[%d][%d] = %f\n", index, batch, output[index*batch_size + batch]);
    }


}

template <typename T>
__global__ void softmax_derivative_kernel(T *input, T *output, int size, int batch_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    if (row < size && col < size && batch < batch_size)
    {
        if(row == col) {
            output[row * size + col + batch*size*size] = input[row*batch_size+batch] * (1 - input[col*batch_size+batch]);
            // printf("Output[%d][%d][%d] = %f\n", row, col,batch, output[row * size + col + batch*size*size]);
        } else {
            output[row * size + col + batch*size*size] = -input[row*batch_size+batch] * input[col*batch_size+batch];
            // printf("Output[%d][%d][%d] = %f\n", row, col,batch, output[row * size + col + batch*size*size]);
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
    Softmax(int rows) : Matrix<T>(rows)
    {
        this->rows = rows;
        this->cols = 1;
        this->input = (T *)malloc(rows * sizeof(T));
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "softmax";
    }
    Softmax(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"Softmax layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Softmax";
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
        T* max = new T [this->batch_size];
        //Find the max of each column in input;
        for(int i = 0; i<this->batch_size; i++) {
            max[i] = input[i];
            for(int j = 1; j<this->rows; j++) {
                if(input[j*this->batch_size+i] > max[i]) {
                    max[i] = input[j*this->batch_size+i];
                }
            }
        }
        int size = this->rows;
        int batch_size = this->batch_size;
        T *d_input, *d_output;
        if (input == NULL)
        {
            cout << "Input Softmax is NULL" << endl;
            input = (T *)malloc(size * batch_size *sizeof(T));
            if (input == NULL)
            {
                cout << "Input of Softmax is NULL" << endl;
                exit(1);
            }
        } else {
            for(int i = 0; i<size*batch_size; i++) {
                this->input[i] = input[i];
                // cout<<"Input["<<i<<"]"<<input[i]<<endl;
            }
        }
        T* d_max;
        T* d_reduce;
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_max, batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_max"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_reduce, batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_reduce"<<endl;
            exit(1);
        }
        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_max, max, batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout<<"Error in copying max from host to device"<<endl;
            exit(1);
        }
        thrust::fill(thrust::device, d_output, d_output + size*batch_size, (T)0);

        // Define grid and block dimensions
        // Launch the softmax kernel
        // Corrected transformation for applying exp

        int threadsPerBlock = 16;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        int batchBlocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchBlocksPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        int TPB = 256;
        int blocks = (batch_size + TPB - 1) / TPB;
        Col_Wise_Reduce<<<blocks, TPB>>>(d_input, d_reduce, d_max, size, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // softmax_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_reduce, size, batch_size);
        // if (!HandleCUDAError(cudaDeviceSynchronize()))
        // {
        //     cout << "Error in synchronizing device" << endl;
        //     exit(1);
        // }
        // Copy the result output from device to host
        if (!HandleCUDAError(cudaMemcpy(output, d_input, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
        T *d_fin_loss;
        int rows = this->rows;
        int batch_size = this->batch_size;
        if (loss == NULL)
        {
            cout << "Loss of Softmax is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * rows * batch_size * sizeof(T))))
        {
            //May need to make a 3D matrix for this
            cout << "Error in allocating memory for d_temp_loss" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, rows * batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_fin_loss"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying loss from host to device, Softmax loss" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchBlocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchBlocksPerGrid, blocksPerGrid , 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        int twodthreadsPerBlock = 8;
        dim3 twodblockDim(twodthreadsPerBlock, twodthreadsPerBlock, twodthreadsPerBlock);

        dim3 twodgridDim((rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, (rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, (batch_size + twodthreadsPerBlock - 1) / twodthreadsPerBlock);

        // Launch the softmax derivative kernel
        softmax_derivative_kernel<T><<<twodgridDim, twodblockDim>>>(d_out, d_temp_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        matrix3D_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_temp_loss, d_loss, d_fin_loss, rows, rows, batch_size);
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
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_fin_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
        if(!HandleCUDAError(cudaFree(d_fin_loss)))
        {
            cout<<"Error in freeing d_fin_loss"<<endl;
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
__global__ void Adam_Update_Weights_Bernoulli(T *weights, T *d_weights, T *m_weights, T *v_weights, T *B_weights, T beta1, T beta2, T epsilon, T learning_rate, int input_size, int output_size, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < input_size)
    {
        d_weights[row * input_size + col] = B_weights[row * input_size + col] * d_weights[row * input_size + col];
        m_weights[row * input_size + col] = beta1 * m_weights[row * input_size + col] + (1 - beta1) * d_weights[row * input_size + col];
        v_weights[row * input_size + col] = beta2 * v_weights[row * input_size + col] + (1 - beta2) * d_weights[row * input_size + col] * d_weights[row * input_size + col];
        T m_hat = m_weights[row * input_size + col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row * input_size + col] / (1 - pow(beta2, epochs+1));
        T temp = weights[row * input_size + col];
        weights[row * input_size + col] -= learning_rate * (m_hat / (sqrt(v_hat) + epsilon)+ temp);
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
__global__ void Adam_Update_Bias_Bernoulli(T *biases, T *d_biases, T *m_biases, T *v_biases, T *B_biases, T beta1, T beta2, T epsilon, T learning_rate, int size, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        d_biases[index] = B_biases[index] * d_biases[index];
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * d_biases[index];
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * d_biases[index] * d_biases[index];
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        T temp = biases[index];
        biases[index] -= learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + temp);
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
    Linear(int cols, int rows,int batch_size) : Matrix<T>(cols, rows,batch_size)
    {
        cout << "Linear Constructor" << endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(cols * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(cols * batch_size * sizeof(T));
        InitMatrix_He<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->biases, rows,1);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        ZeroVector<T>(this->input, cols*batch_size);
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
    void set_weights(T *weights, T *biases)
    {
        this->weights = weights;
        this->biases = biases;
    }
    T* v_weights;
    T* v_biases;
    T* m_weights;
    T* m_biases;
    void Fill_Bernoulli() override{
        // Only fill with 0's and 1's at random
        for(int i = 0; i<this->rows * this->cols; i++) {
            this->B_weights[i] = 0;
        }
        for(int i = 0; i<this->rows; i++) {
            this->B_biases[i] = 0;
        }
    }
    void set_Bernoulli(int row, int col) override{
        this->B_weights[row*(this->cols) + col] = 1;
    }
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

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
        
        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    void update_weights_AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        T *d_B_weights, *d_B_biases;
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
        if(!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_m_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(T)))) {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
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
        if (!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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


        Adam_Update_Weights_Bernoulli<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, d_B_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Adam_Update_Bias_Bernoulli<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, d_B_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
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

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    void find_Loss_Metric() override {
        T *dev_Weights, *dev_Biases, *d_d_Weights, *d_d_Biases;
        T *d_wDw, *d_bDb;

        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_wDw, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_bDb, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_bDb" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_Weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_Biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_wDw, this->W_dW_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying wDw from host to device"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_bDb, this->W_dW_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying bDb from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
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


        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_d_Weights, d_wDw, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_d_Biases, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        //Delete streams
        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }

        //Transfer the result to host
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_weights, d_wDw, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying wDw from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_biases, d_bDb, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying bDb from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_wDw)))
        {
            cout << "Error in freeing d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_bDb)))
        {
            cout << "Error in freeing d_bDb" << endl;
            exit(1);
        }
    }
};


template <typename T>
class Conv1D : public Matrix<T>
{
    public:
    Conv1D(int width, int channels, int kernel_size, int stride, int padding, int filters, int batch_size) {
        this->width = width;
        this->channels = channels;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->filters = filters;
        this->rows = filters;
        this->cols = width;
        this->output_width = (width - kernel_size + 2 * padding) / stride + 1; 
        this->batch_size = batch_size;
        this->weights = (T *)malloc(filters * kernel_size * channels * sizeof(T));
        this->biases = (T *)malloc(filters * sizeof(T));
        this->input = (T *)malloc(width * channels * batch_size *  sizeof(T));
        this->hidden_output = (T *)malloc(output_width * filters * batch_size * sizeof(T));
    }
    int rows;
    int cols;
    int width;
    int batch_size;
    int channels;
    int kernel_size;
    int stride;
    int padding;
    int filters;
    T *biases;
    T *weights;
    T* input;
    T* hidden_output;
    int output_width;
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
    ~Conv1D() {
        free(this->weights);
        free(this->biases);
    }


};



template <typename T>
class Conv2D : public Matrix<T>
{
public:
    Conv2D(int width, int height, int channels, int kernel_width, int kernel_height, int stride, int padding, int filters, int batch_size)
    {
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->filters = filters;
        this->rows = filters;
        this->cols = width * height;
        this->output_width = (width - kernel_width + 2 * padding) / stride + 1;
        this->output_height = (height - kernel_height + 2 * padding) / stride + 1;
        this->weights = (T *)malloc(filters * kernel_width * kernel_height * channels * sizeof(T));
        this->biases = (T *)malloc(filters * sizeof(T));
        this->batch_size = batch_size;
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(output_width * output_height * filters * batch_size * sizeof(T));
    }
    int rows;
    int cols;
    int width;
    int batch_size;
    int height;
    int channels;
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int filters;
    int output_width;
    int output_height;
    T *weights;
    T *biases;
    T *input;
    T *hidden_output;
    ~Conv2D()
    {
        free(this->weights);
        free(this->biases);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
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
class MaxPooling1D : public Matrix<T>
{
public:
    MaxPooling1D(int kernel_width, int stride, int padding, int width, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    ~MaxPooling1D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class AvePooling1D : public Matrix<T>
{
public:
    AvePooling1D(int kernel_width, int stride, int padding, int width, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    ~AvePooling1D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class AvePooling2D : public Matrix<T>
{
public:
    AvePooling2D(int kernel_width, int kernel_height, int stride, int padding, int width, int height, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->output_height = floor((height - 2 * padding - (kernel_height-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int width;
    int height;
    int channels;
    int batch_size;
    int rows;
    int cols;
    int output_width;
    int output_height;
    T *input;
    T *hidden_output;
    ~AvePooling2D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class MaxPooling2D : public Matrix<T>
{
public:
    MaxPooling2D(int kernel_width, int kernel_height, int stride, int padding, int width, int height, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->batch_size = batch_size;
        this->rows = channels;
        this->cols = width * height;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->output_height = floor((height - 2 * padding - (kernel_height-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int width;
    int height;
    int channels;
    int batch_size;
    int rows;
    int cols;
    int output_width;
    int output_height;
    T *input;
    T *hidden_output;
    ~MaxPooling2D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
__global__ void Binary_Cross_Entropy_Kernel(T *label, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -1 * (label[index*batch_size + batch] * log(output[index*batch_size + batch]) + (1 - label[index*batch_size + batch]) * log(1 - output[index*batch_size + batch]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy(T *input, T *output, T *loss, int size, int batch_size)
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
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -1 * -log(input[index*batch_size + batch]) * output[index*batch_size + batch];
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
__global__ void Binary_Cross_Entropy_Derivative(T *label, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -(label[index*batch_size + batch] / output[index*batch_size + batch] - (1 - label[index*batch_size + batch]) / (1 - output[index*batch_size + batch]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    // The input is the output of the network and the output is the ground truth
    if (index < size && batch < batch_size)
    {
        loss[index* batch_size + batch] = -1 * (output[index* batch_size + batch] / input[index* batch_size + batch]);
    }
    __syncthreads();
}

template <typename T>
class Loss : public Matrix<T>
{
public:
    T* labels;
    T* output;
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
    Loss(int size, int batch_size): Matrix<T>(size,batch_size){
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
    };
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
    Binary_CrossEntropy(int size, int batch_size) : Loss<T>(size, batch_size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
        this->labels = (T *)malloc(size * batch_size * sizeof(T));
        this->name = "categorical";
        this->batch_size = batch_size;
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
        T *d_pred, *d_labels, *d_loss;
        int rows = this->rows;
        int batch_size = this->batch_size;
        //Output holds the labels
        //Input holds the predictions
        memcpy(this->labels, output, rows * batch_size * sizeof(T));
        memcpy(this->input, input, rows * batch_size * sizeof(T));
        if (!HandleCUDAError(cudaMalloc((void **)&d_pred, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_labels, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_pred, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_labels, output, rows * batch_size *  sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy kernel
        Binary_Cross_Entropy_Kernel<T><<<gridDim, blockDim>>>(d_labels, d_pred, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_pred)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_labels)))
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
        int batch_size = this->batch_size;

        //-label/output

        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }
        //Output holds the labels
        //Input holds the predictions

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->labels, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }
        //-d_gt/d_out

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy derivative kernel
        Binary_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_gt, d_out, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    Categorical(int size, int batch_size) : Loss<T>(size, batch_size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
        this->labels = (T *)malloc(size * batch_size * sizeof(T));
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
        int batch_size = this->batch_size;
        memcpy(this->labels, output, rows * batch_size * sizeof(T));
        memcpy(this->input, input, rows * batch_size * sizeof(T));
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_output, output, rows * batch_size *  sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy kernel
        Categorical_Cross_Entropy<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
        int batch_size = this->batch_size;

        //-label/output

        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->labels, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }
        //-d_gt/d_out

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy derivative kernel
        Categorical_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
        this->num_updateable = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
    }
    Network(int input_size, int output_size, Optimizer<T>* optimizer, int Q){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
        this->Q = Q;
    }
    Network(int input_size, int output_size, Optimizer<T>* optimizer, int Q, int batch_size){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
        this->Q = Q;
        this->batch_size = batch_size;
    }
    ~Network(){};
    int input_size;
    int *hidden_size;
    int output_size;
    int num_layers;
    int num_activation;
    int num_derv;
    int Q;
    int batch_size;
    int num_updateable;
    T *input;
    T *prediction;
    Optimizer<T>* optim;
    thrust::host_vector<Matrix<T> *> layers;
    thrust::host_vector<Matrix<T> *> activation;
    thrust::host_vector<Loc_Layer<T>*> bernoullie_w;
    thrust::host_vector<Loc_Layer<T>*> bernoullie_b;
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
                    memcpy(loss[i - 1], layers[i]->next_loss, layers[i-1]->rows * batch_size * sizeof(T));
                    //Display loss and layer name for debugging

                }
            }
            else
            {
                cout << "Index " << i << " out of bounds for layers vector." << endl;
            }
        }
    }
    void update_weights(T learning_rate){};
    void update_weights(T learning_rate, int epochs, int Q);
    void addLayer(Linear<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        if(this->optim->name == "AdamWBernoulli"){
            bernoullie_w.push_back((Loc_Layer<T> *)malloc(layer->rows * layer->cols * sizeof(Loc_Layer<T>)));
            bernoullie_b.push_back((Loc_Layer<T> *)malloc(layer->rows * sizeof(Loc_Layer<T>)));
        }
        layer->name = "saved linear";
        cout<<layer->name<<endl;
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->cols * this->batch_size  * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_updateable = bernoullie_w.size()-1;
        layerMetadata.push_back(LayerMetadata(num_layers,num_updateable, true)); // Assuming Linear layers are updateable
        num_layers++;
        num_derv++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Conv2D<T> *layer)
    {
        layers.push_back(layer);
        //this size below is not right either
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
        //The loss is the same size as the output of the layer
        layer->name = "saved conv2d";
        if (layer->next_loss == NULL)
        {
            //NOT CORRECT
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }

    }
    void addLayer(MaxPooling2D<T> *layer)
    {
        layers.push_back(layer);
        //The loss is the same size as the output of the layer
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * sizeof(T)));
        num_layers++;
        layer->name = "saved maxpooling2d";
        if (layer->next_loss == NULL)
        {
            //NOT CORRECT
            layer->next_loss = (T *)malloc(layer->rows * sizeof(T));
        }
    }
    void addLayer(Sigmoid<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(RELU_layer<T> *layer)
    {
        layers.push_back(layer);
        layer->name = "saved RELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(LeakyRELU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved LeakyRELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Tanh<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved Tanh";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(SLU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved SLU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(ELU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved Tanh";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Softmax<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved softmax";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLoss(Binary_CrossEntropy<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved Binary";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLoss(Mean_Squared_Error<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved MSE";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLoss(Categorical<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        layer->name = "saved categorical";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void train(T *input, T *output, int epochs, T learning_rate);
    void train(T **input, T **output, int epochs, T learning_rate, int size);
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
        layers[0]->forward(input, layers[0]->hidden_output);
        // cout<< "Layer 0 output"<<endl;
        //     for(int j = 0; j<layers[0]->rows; j++){
        //         for(int k = 0; k<batch_size; k++){
        //             cout<<layers[0]->hidden_output[j*batch_size + k]<<" ";
        //         }
        //         cout<<endl;
        //     }
        // cout<<endl;
        for (int i = 1; i < layers.size() - 1; i++)
        {   
            layers[i]->forward(layers[i - 1]->hidden_output, layers[i]->hidden_output);
            // cout<<"Layer "<<i<<" output"<<endl;
            // for(int j = 0; j<layers[i]->rows; j++){
            //     for(int k = 0; k<batch_size; k++){
            //         cout<<layers[i]->hidden_output[j*batch_size + k]<<" ";
            //     }
            //     cout<<endl;
            // }
        }
        // Should be the cost layer
        layers[layers.size() - 1]->forward(layers[layers.size() - 2]->hidden_output, output);
    }
    void getOutput(T *output)
    {
        memcpy(output, prediction, output_size * sizeof(T));
    }
    void Fill_Bern(Matrix<T>* Layer, int layer_num){
        if(this->optim->name == "AdamWBernoulli"){
            for(int i = 0; i<Layer->rows; i++){
                for(int j = 0; j<Layer->cols; j++){
                    bernoullie_w[layer_num][i*Layer->cols + j].row = i;
                    bernoullie_w[layer_num][i*Layer->cols + j].col = j;
                    bernoullie_w[layer_num][i*Layer->cols + j].layer = layer_num;
                    bernoullie_w[layer_num][i*Layer->cols + j].weights_dW = Layer->W_dW_weights[i*Layer->cols + j];
                }
            }
            for(int i = 0; i<Layer->rows; i++){
                bernoullie_b[layer_num][i].row = i;
                bernoullie_b[layer_num][i].col = 0;
                bernoullie_b[layer_num][i].layer = layer_num;
                bernoullie_b[layer_num][i].weights_dW = Layer->W_dW_biases[i];
            }
        }
    }
    thrust::host_vector<Loc_Layer<T>> flatten() {   
        thrust::host_vector<Loc_Layer<T>>  result;
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
                        result.insert(result.end(), bernoullie_w[layerMetadata[i].LinNumber], bernoullie_w[layerMetadata[i].LinNumber]+layers[layerMetadata[i].layerNumber]->rows*layers[layerMetadata[i].layerNumber]->cols); 
                    }
                }
            }
        }
        return result;
    }
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
    dim3 gridDim((input_size + block_size - 1) / block_size, (output_size + block_size - 1) / block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);
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
__global__ void sigmoid_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = 1 / (1 + exp(-input[index*batch_size + batch]));
    }
}

template <typename T>
void Sigmoid<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->rows;
    T *d_input, *d_output;
    // this->input = input;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input Sigmoid is NULL" << endl;
        input = (T *)malloc(size * batch_size * sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of Sigmoid is NULL" << endl;
        output = (T *)malloc(size * batch_size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of Sigmoid is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * batch_size * sizeof(T));
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the sigmoid kernel
    sigmoid_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    memcpy(this->hidden_output, output, size * batch_size * sizeof(T));
}

template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T* loss, T* fin_loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        fin_loss[index*batch_size+batch] = input[index*batch_size+batch] * (1 - input[index*batch_size+batch])*loss[index*batch_size+batch];
    }
}

template <typename T>
void Sigmoid<T>::backward(T *loss)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;
    T* d_fin_loss;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * batch_size* sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_fin_loss" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_loss, d_fin_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(loss, d_fin_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    if (!HandleCUDAError(cudaFree(d_fin_loss)))
    {
        cout << "Error in freeing d_fin_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : 0;
    }
}

template <typename T>
void RELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->rows;
    int batch_size = this->batch_size;
    // this->input = input;
    if (input == NULL)
    {
        cout << "Input RELU is NULL" << endl;
        input = (T *)malloc(size * batch_size* sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of RELU is NULL" << endl;
        output = (T *)malloc(size * batch_size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of RELU is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * batch_size * sizeof(T));
    T *d_input, *d_output;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the RELU kernel
    RELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    memcpy(this->hidden_output, output, size * batch_size * sizeof(T));

}

template <typename T>
__global__ void RELU_derivative_kernel(T* input, T* loss, T* fin_loss, int size, int batch_size)
{
    int batch = blockIdx.x*blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;

    if (index < size && batch < batch_size)
    {
        fin_loss[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size+batch] : 0;
    }
}

template <typename T>
void RELU_layer<T>::backward(T *loss)
{
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;
    T* d_fin_loss;
    int batch_size = this->batch_size;

    int size = this->rows;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_fin_loss, size * batch_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_fin_loss"<<endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, size * batch_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying loss from host to device, ReLU"<<endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_batch = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid_batch, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);
    // Launch the sigmoid derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_loss, d_fin_loss, size, batch_size);
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
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_fin_loss, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    if(!HandleCUDAError(cudaFree(d_fin_loss))){
        cout<<"Error in freeing d_fin_loss"<<endl;
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
__global__ void linear_kernel_batch(T* Input, T* Output, T* Weights, T* bias, const int nx, const int ny, const int batch_size) {
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	//This conditional is for debugging, even though done on the device
	if (row < ny && col < batch_size) { 
		for (int k = 0; k < nx; k++) {
			fSum += Weights[row * nx + k] * Input[k * batch_size + col];
		}
		Output[row * batch_size + col] = fSum + bias[row];
	}
}

template <typename T>
void Linear<T>::forward(T *input, T *output)
{
    // Allocate device memory for input, output, weights, and biases
    int input_size = this->cols;
    int output_size = this->rows;
    int batch_size = this->batch_size;
    memcpy(this->input, input, input_size * batch_size * sizeof(T));
    T *d_input, *d_output, *dev_weights, *dev_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * batch_size * sizeof(T))))
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
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
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
    int threadsPerBlock = 16;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the linear kernel
    linear_kernel_batch<T><<<gridDim, blockDim>>>(d_input, d_output, dev_weights, dev_biases, input_size, output_size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
__global__ void linear_derivative_kernel(T *loss, T *d_Weights, T *output, int rows, int cols, int batch_size)
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
    // Reference for batch_size = 1
    // if (row < rows && col < cols)
    // {
    //     d_Weights[row * cols + col] = output[col] * loss[row];
    // }
    if(row < rows && col < cols){
        T sum = 0;
        for (int k = 0 ; k < batch_size ; k++){
            sum += output[col * batch_size + k] * loss[row * batch_size + k];
        }
        d_Weights[row * cols + col] = sum/batch_size;
    }
    // Sum the loss to get the derivative of the loss with respect to the biases
}


template <typename T>
__global__ void linear_weight_derivatives(T *loss, T *Weights, T *d_F, int rows, int cols, int batch_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // columns
    int batch = blockIdx.y * blockDim.y + threadIdx.y; //batch_size
    //Compute gradient with respect to weights
    // dx = np.dot(dout, self.W.T)  
    //Now we must do this for a loss matrix of size cols x batch_size
    // if (col < cols)
    // {
    //     T sum = 0;
    //     for (int i = 0; i < rows; i++)
    //     {
    //         sum += loss[i] * Weights[col * rows + i]; // Gradient of the loss w.r.t. input x, delta* W^T
    //     }
    //     d_F[col] = sum;
    // }
    if(batch < batch_size && col < cols){
        T sum = 0;
        for (int k = 0 ; k < rows ; k++){
            sum += Weights[k* cols + col] * loss[k* batch_size + batch];
        }
        d_F[col * batch_size + batch] = sum;
    }

}

template <typename T>
__global__ void linear_bias_derivatives(T *loss, T *d_biases, int rows, int batch_size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        T sum = 0;
        for(int i = 0; i < batch_size; i++){
            sum += loss[row * batch_size + i];
        }
        d_biases[row] = sum/batch_size;
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
    int batch_size = this->batch_size;
    cout<<endl;
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, cols * batch_size * sizeof(T))))
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
    if (!HandleCUDAError(cudaMalloc((void **)&d_F, cols * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_F" << endl;
        exit(1);
    }
    if (loss == NULL)
    {
        cout << "Loss is NULL" << endl;
    }
    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
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
    if(!HandleCUDAError(cudaMemcpy(d_output,this->input,cols * batch_size * sizeof(T),cudaMemcpyHostToDevice))){
        cout<<"Error in copying output from host to device"<<endl;
        exit(1);
    }

    // Create three streams to (1) Find the weight update (2) Find the bias update (3) Find the next loss

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid_Rows = (rows + threadsPerBlock - 1) / threadsPerBlock;

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
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    dim3 gridDim_Cols((cols + block_size - 1) / block_size, (batch_size+block_size-1)/block_size, 1);
    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim,0,stream1>>>(d_loss, dd_weights, d_output, rows, cols, batch_size);
    if (!HandleCUDAError(cudaStreamSynchronize(stream1)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_weight_derivatives<T><<<gridDim_Cols,blockDim,0,stream2>>>(d_loss,dev_weights,d_F,rows,cols, batch_size);
    if (!HandleCUDAError(cudaStreamSynchronize(stream2)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_bias_derivatives<T><<<blocksPerGrid_Rows,threadsPerBlock,0,stream3>>>(d_loss,dd_biases,rows, batch_size);
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
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_F, cols * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
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
    int temp = this->batch_size;
    this->batch_size = 1;
    forward(input, output);
    this->batch_size = temp;
}

template <typename T>
void Network<T>::predict(T **input, T **output, int size)
{
    T *prediction = (T *)malloc(output_size * sizeof(T));
    float sum = 0;
    int temp = this->batch_size;
    this->batch_size = 1;
    for(int i=0; i<layers.size(); i++){
        layers[i]->batch_size = 1;
    }
    for (int i = 0; i < size; i++)
    {
        forward(input[i], output[i]);
        // take the hidden output, and measure accuracy
        prediction = layers[layers.size() - 1]->hidden_output;
        for(int j=0; j<output_size; j++){
            cout<<prediction[j]<<" ";
        }
        cout<<endl;
        // Find the max value in the prediction
        int max_index = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (prediction[j] >= prediction[max_index])
            {
                max_index = j;
            }
        }
        // Find the max value in the output
        int max_index_output = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (output[i][j] >= output[i][max_index_output])
            {
                max_index_output = j;
            }
        }
        // check if the max index is the same as the max index output
        if (max_index == max_index_output)
        {
            sum++;
            cout << "Correct Prediction for input " << i << endl;
            cout<<"Prediction: "<<max_index<<" Output: "<<max_index_output<<endl;
        }
        else
        {
            cout << "Incorrect Prediction for input " << i << endl;
            cout<<"Prediction: "<<max_index<<" Output: "<<max_index_output<<endl;
        }
    }
    float accuracy = sum / size;
    cout << "Accuracy: " << accuracy << endl;
    this->batch_size = temp;
    for(int i=0; i<layers.size(); i++){
        layers[i]->batch_size = temp;
    }
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
struct CompareBernoulliWeights {
    __host__ __device__
    bool operator()(const Loc_Layer<T>& lhs, const Loc_Layer<T>& rhs) const {
        // Assuming Loc_Layer has a member function or variable to get the weight
        return lhs.weights_dW > rhs.weights_dW; // Sort in ascending order
    }
};

template <typename T>
struct CompareBernoulliLayers {
    __host__ __device__
    bool operator()(const Loc_Layer<T>& lhs, const Loc_Layer<T>& rhs) const {
        // Assuming Loc_Layer has a member function or variable to get the weight
        return lhs.layer > rhs.layer; // Sort in ascending order
    }
};




template <typename T>
void Network<T>::update_weights(T learning_rate, int epochs, int Q)
{
    // Ensure layers vector is not empty and is properly initialized
    if (this->layers.empty())
    {
        std::cerr << "Error: Layers vector is empty.\n";
        return;
    }
    // for(int i = 0; i<this->layers.size(); i++){
    //     cout<<this->layers[i]->name<<endl;
    //     if(this->layers[i]->name.empty()){
    //         cout<<"Layer is null"<<endl;
    //         return;
    //     }
    //     else{
    //         cout<<"I have AIDS?"<<endl;
    //     }
    // }   
    if(this->optim->name == "AdamWBernoulli"){
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
                        this->layers[layerMetadata[i].layerNumber]->find_Loss_Metric();
                        Fill_Bern(this->layers[layerMetadata[i].layerNumber], layerMetadata[i].LinNumber);
                        
                    }
                }
            }
        }
        thrust::host_vector<Loc_Layer<T>> res = flatten();
        thrust::sort(res.begin(), res.end(), CompareBernoulliWeights<T>());
        thrust::sort(res.begin(), res.begin()+Q, CompareBernoulliLayers<T>());
        //Use the column and rows to set Bernoulli
        for(int i = 0; i < Q; i++){
            this->layers[res[i].layer]->set_Bernoulli(res[i].row, res[i].col);
        }
        
        
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
                    else if(this->optim->name == "AdamWBernoulli"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_AdamWBernoulli(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli();
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
    }
    for (int i = 0; i < output_size; i++)
    {
        cout << layers[layers.size() - 1]->hidden_output[i] << " ";
    }
    cout << endl;
    memcpy(this->prediction, layers[layers.size() - 1]->hidden_output, output_size * sizeof(T));
}

template <typename T>
void Network<T>::train(T **input, T **output, int epochs, T learning_rate, int size)
{
    // Find a random list of indices for the batch size
    //  Create a thrust vector of indices
    int* indices = (int*)malloc(batch_size*sizeof(int));
    // Fill the vector with random_indices
    // Iterate through the indices and train the network
    int pred_idx = 0;
    int gt_idx = 0;
    int sum = 0;
    T* batch_input = (T*)malloc(input_size*batch_size*sizeof(T));
    T* batch_output = (T*)malloc(output_size*batch_size*sizeof(T));
    for (int i = 0; i < epochs; i++)
    {
        cout<< "Epoch: " << i << endl;
        for (int k = 0; k < batch_size; k++)
        {
            indices[k] = rand() % size;
        }
        Format_Batch_Data(input,output,batch_input,batch_output,indices,batch_size,input_size,output_size);
        // cout<<"Batch Input: "<<endl;
        // for(int j = 0; j< input_size; j++){
        //     for(int q = 0; q < batch_size; q++){
        //         cout<<batch_input[j*batch_size + q]<<" ";
        //     }
        //     cout<<endl;
        // }
        // cout<<"OUTPUT: "<<endl;
        // for(int j = 0; j< output_size; j++){
        //     for(int q = 0; q < batch_size; q++){
        //         cout<<batch_output[j*batch_size + q]<<" ";
        //     }
        //     cout<<endl;
        // }
        // for (int m = 0; m < layerMetadata.size(); m++)
        // {
        //     // Validate layerNumber is within bounds
        //     if (layerMetadata[m].layerNumber >= 0 && layerMetadata[m].layerNumber < this->layers.size())
        //     {
        //         // Check if the layer pointer is not null
        //         if (this->layers[layerMetadata[m].layerNumber] != nullptr)
        //         {
        //             // Check if the current layer is marked as updateable
        //             if (layerMetadata[m].isUpdateable)
        //             {
        //                 cout<< "Weights: "<<endl;
        //                 for(int j = 0; j < this->layers[layerMetadata[m].layerNumber]->rows; j++){
        //                     for(int k = 0; k < this->layers[layerMetadata[m].layerNumber]->cols; k++){
        //                         cout<<this->layers[layerMetadata[m].layerNumber]->weights[j*this->layers[layerMetadata[m].layerNumber]->cols + k]<<" ";
        //                     }
        //                     cout<<endl;
        //                 }
        //                 cout<<endl;
        //                 cout<<"biases: "<<endl;
        //                 cout<<this->layers[layerMetadata[i].layerNumber]->rows<<endl;
        //                 for(int j = 0; j < this->layers[layerMetadata[i].layerNumber]->rows; j++){
        //                     cout<<this->layers[layerMetadata[i].layerNumber]->biases[j]<<" ";
        //                 }
        //                 cout<<endl;
        //             }
        //         }
        //     }
        //     cout<<endl;
        // }
        forward(batch_input, batch_output);
        // cout<<"Output: "<<endl;
        // cout<<layers[layers.size()-2]->name<<endl;
        // for(int j=0; j<output_size; j++){
        //     for(int k = 0; k < batch_size; k++){
        //         cout<<layers[layers.size() - 2]->hidden_output[j*batch_size + k]<<" ";
        //     }
        //     cout<<endl;
        // }
        backward(batch_input, batch_output);
        // //Display d_weights
        // for (int i = 0; i < layerMetadata.size(); i++)
        // {
        //     // Validate layerNumber is within bounds
        //     if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
        //     {
        //         // Check if the layer pointer is not null
        //         if (this->layers[layerMetadata[i].layerNumber] != nullptr)
        //         {
        //             // Check if the current layer is marked as updateable
        //             if (layerMetadata[i].isUpdateable)
        //             {
        //                 cout<<"d_weights: "<<endl;
        //                 for(int j = 0; j < this->layers[layerMetadata[i].layerNumber]->rows; j++){
        //                     for(int k = 0; k < this->layers[layerMetadata[i].layerNumber]->cols; k++){
        //                         cout<<this->layers[layerMetadata[i].layerNumber]->d_weights[j*this->layers[layerMetadata[i].layerNumber]->cols + k]<<" ";
        //                     }
        //                     cout<<endl;
        //                 }
        //                 cout<<endl;
        //                 cout<<"d_biases: "<<endl;
        //                 for(int j = 0; j < this->layers[layerMetadata[i].layerNumber]->rows; j++){
        //                     cout<<this->layers[layerMetadata[i].layerNumber]->d_biases[j]<<" ";
        //                 }
        //                 cout<<endl;
        //             }
        //         }
        //     }
        //     cout<<endl;
        // }

        update_weights(learning_rate, i, this->Q);
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
__global__ void conv2D_kernel(T *input, T *output, T *weights, T *biases, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int stride, int batch_size)
{
    /*The input has the shape of (batch_size, channel, height, width)
    The weights have the shape (filters, channel, filter_height, filter_width)
    The output has the shape of (batch_size, filter, output_height, output_width)
    We will have the dimy and dimx operate on the single image channel, i.e. deal with height and width
    For the sake of computatation, we will use the z dim to deal with batches and have them process each channel */
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z*blockDim.z + threadIdx.z;
    if(outCol < out_width && outRow < out_height && batch < batch_size){
        for (int filter = 0; filter < filters; filter++)
        {
            T sum = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                for (int i = 0; i < kernel_height; i++)
                {
                    for (int j = 0; j < kernel_width; j++)
                    {
                        int inRow = outRow * stride + i;
                        int inCol = outCol * stride + j;
                        sum += input[batch * channels * width * height + channel * width * height + inRow * width + inCol] * weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + i * kernel_width + j];
                    }
                }
            }
            output[batch * filters * out_width * out_height + filter * out_width * out_height + outRow * out_width + outCol] = sum + biases[filter];
        }
    }
}


template <typename T>
__global__ void conv2D_weight_update_kernel(T *input, T* this_loss, T* d_weights, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int stride, int batch_size)
{
    /*This kernel is tasked with finding dw for a convolutional layer
    The equation which will be used is:
    dW[filter,channel,k,m]= 1/(batch_size)\sum_{n=0}^{batch_size}\sum_{i=0}^{kernel_height}\sum_{j=0}^{kernel_width}x_n[channel,i+k,j+m]*thisloss_n[f,i,j]*/
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(filter < filters && channel < channels && k < kernel_height){
        T sum = 0;
        for(int m=0; m<kernel_width; m++){
            sum = 0;
            for(int batch = 0; batch < batch_size; batch++){
            //This is the sum over the batch
                for(int i = 0; i<out_height; i++){
                    //This is the sum over the height, and we will have i+k for the input, and i for the loss
                    for(int j = 0; j<out_width; j++){
                        //This is the sum over the width, and we will have j+m for the input, and j for the loss
                            sum += input[batch * channels * width * height + channel * width * height + (i+k) * width + j+m] * this_loss[batch * filters * out_width * out_height + filter * out_width * out_height + i * out_width + j];
                    }
                }
            }
            d_weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + k * kernel_width + m] = sum/batch_size;
        }
    }
    __syncthreads();
    //How does stride play a role?
}


template <typename T>
__global__ void conv2D_biases_update_kernel(T *this_loss, T* d_biases, int filters, int out_width, int out_height, int batch_size)
{
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    if(filter < filters){
        T sum = 0;
        for(int batch = 0; batch < batch_size; batch++){
            for(int i = 0; i<out_height; i++){
                for(int j = 0; j<out_width; j++){
                    sum += this_loss[batch * filters * out_width * out_height + filter * out_width * out_height + i * out_width + j];
                }
            }
        }
        d_biases[filter] = sum/batch_size;
    }
}

template <typename T>
__global__ void conv2D_next_loss_kernel(T *weights, T *this_loss, T *next_loss, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int stride, int batch_size)
{
    /*We need to find dx_n, where n corresponds to the batch*/
}




template <typename T>
void Conv2D<T>::forward(T *input, T *output)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, batch_size * width * height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, batch_size * output_width * output_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, filters * kernel_width * kernel_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, filters * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, batch_size * width * height * channels * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, filters * kernel_width * kernel_height * channels * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, biases, filters * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int TPB = 8;
    dim3 blockDim(TPB, TPB, TPB);
    dim3 gridDim((output_height + TPB - 1) / TPB,(output_width + TPB - 1) / TPB, (batch_size*channels + TPB - 1) / TPB);
    conv2D_kernel<T><<<gridDim,blockDim>>>(input, output, weights, biases, channels, filters, kernel_width, kernel_height, width, height, out_width, out_height, stride, batch_size);
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, batch_size * output_width * output_height * channels  * sizeof(T), cudaMemcpyDeviceToHost)))
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
void Conv2D<T>::backward(T * loss)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    T *d_dweights, *d_dbiases, *d_dinput;
    T* d_loss, *d_fin_loss;
    if(!HandleCUDAError(cudaMalloc((void **)&d_loss, batch_size * width * height * channels * sizeof(T)))){
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, batch_size * output_width * output_height * channels * sizeof(T)))){
        cout << "Error in allocating memory for d_fin_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, batch_size * width * height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, batch_size * output_width * output_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, filters * kernel_width * kernel_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, filters * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_dweights, filters * kernel_width * kernel_height * channels * sizeof(T)))){
        cout << "Error in allocating memory for d_dweights" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_dbiases, filters * sizeof(T)))){
        cout << "Error in allocating memory for d_dbiases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, batch_size * width * height * channels * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, filters * kernel_width * kernel_height * channels  * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, biases, filters * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, batch_size * width * height * channels * sizeof(T), cudaMemcpyHostToDevice))){
        cout << "Error in copying loss from host to device" << endl;
        exit(1);
    }



    if(!HandleCUDAError(cudaMemcpy(this->next_loss, d_fin_loss, batch_size * output_width * output_height * channels * sizeof(T), cudaMemcpyDeviceToHost))){
        cout << "Error in copying fin_loss from host to device" << endl;
        exit(1);
    }
    // Copy the gradients from device to host
    if (!HandleCUDAError(cudaMemcpy(d_weights, d_dweights, filters * kernel_width * kernel_height * channels * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dweights from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, d_dbiases, filters * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dbiases from device to host" << endl;
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
__global__ void max_pooling_kernel(T *input, T *output, int kernel_width, int kernel_height, int width, int height, int output_width, int output_height, int batch_size, int padding, int channels)
{
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int z = threadIdx.z + (blockIdx.z * blockDim.z);
    /*x corresponds to the column, y corresponds to the row, and z corresponds to the channel and batch*/
    int idx = z * channels * width * height + y * width + x;
    int out_idx = z * channels * output_width * output_height + y * output_width + x;
    if (x < output_width && y < output_height && z < batch_size * channels)
    {
        T max = input[idx];
        for (int i = 0; i < kernel_height; i++)
        {
            for (int j = 0; j < kernel_width; j++)
            {
                if (input[z * channels * width * height + (y * kernel_height + i) * width + (x * kernel_width + j)] > max)
                {
                    max = input[z * channels * width * height + (y * kernel_height + i) * width + (x * kernel_width + j)];
                }
            }
        }
        output[out_idx] = max;
    }
}


template <typename T>
void MaxPooling2D<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, batch_size * width * height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, batch_size * output_width * output_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, batch_size * width * height * channels * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    /*The equation for this is: out(N,C,h,w)= max_{0...kernel_H}max_{0...kernel_W}input(N,C,stride[0]xh+m,stride[1]xw+n)*/

    int TPB = 8;
    dim3 blockDim(TPB, TPB, TPB);
    dim3 gridDim((output_height + TPB - 1) / TPB,(output_width + TPB - 1) / TPB, (batch_size*channels + TPB - 1) / TPB);
    // Launch the max pooling kernel
    max_pooling_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, input_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, batch_size * output_width * output_height * channels * sizeof(T), cudaMemcpyDeviceToHost)))
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