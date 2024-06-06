#include "../include/include.h"
#include "../include/layers.h"


// Function to initialize the weights and biases of the network
template <typename T>
Matrix<T>::Matrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    this->d_data = (T*)malloc(rows * cols * sizeof(T));
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, T *data){
    this->rows = rows;
    this->cols = cols;
    this->d_data = data;
}

template <typename T>   
Matrix<T>::~Matrix(){
    free(this->d_data);
}


template <typename T>
void Matrix<T>::matrix_multiply(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, cols * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, cols * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix multiplication kernel
    matrix_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

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
void Matrix<T>::matrix_add(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix addition kernel
    matrix_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
void Matrix<T>::matrix_subtract(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix subtraction kernel
    matrix_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
void Matrix<T>::matrix_transpose(T *A, T *C){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, cols * rows * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix transpose kernel
    matrix_transpose_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, cols * rows * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
void Matrix<T>::matrix_scalar_multiply(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar multiplication kernel
    matrix_scalar_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
void Matrix<T>::matrix_scalar_add(T *A, T *C,T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar addition kernel
    matrix_scalar_add_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
void Matrix<T>::matrix_scalar_subtract(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar subtraction kernel
    matrix_scalar_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
void Matrix<T>::matrix_elementwise_multiply(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise multiplication kernel
    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_divide(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise division kernel
    matrix_elementwise_divide_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
void Matrix<T>::matrix_elementwise_add(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise addition kernel
    matrix_elementwise_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
void Matrix<T>::matrix_elementwise_subtract(T *A, T *B, T *C){
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix elementwise subtraction kernel
    matrix_elementwise_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
void Matrix<T>::matrix_sum(T *A, T *C, int axis){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    if (axis == 0) {
        // Launch the matrix sum along axis 0 kernel
        matrix_sum_axis0_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
    } else if (axis == 1) {
        // Launch the matrix sum along axis 1 kernel
        matrix_sum_axis1_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
    }

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
void Matrix<T>::matrix_scalar_divide(T *A, T *C, T scalar){
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(T));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the matrix scalar division kernel
    matrix_scalar_divide_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
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
    this->cols = cols;
}


template <typename T>
void Sigmoid<T>::forward(T *input, T *output, int size){
    // Allocate device memory for input and output
    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(T));
    cudaMalloc((void**)&d_output, size * sizeof(T));

    // Copy input from host to device
    cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the sigmoid kernel
    sigmoid_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = 1 / (1 + exp(-input[index]));
    }
}

template <typename T>
void Sigmoid<T>::backward(T *input, T *output, int size){
    // Allocate device memory for input and output
    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(T));
    cudaMalloc((void**)&d_output, size * sizeof(T));

    // Copy input from host to device
    cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] * (1 - input[index]);
    }
}

template <typename T>
RELU_layer<T>::RELU_layer(int rows, int cols):Matrix<T>(rows, cols){
    this->rows = rows;
    this->cols = cols;
}

template <typename T>
void RELU_layer<T>::forward(T *input, T *output, int size){
    // Allocate device memory for input and output
    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(T));
    cudaMalloc((void**)&d_output, size * sizeof(T));

    // Copy input from host to device
    cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the RELU kernel
    RELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = fmax(0, input[index]);
    }
}

template <typename T>
void RELU_layer<T>::backward(T *input, T *output, int size){
    // Allocate device memory for input and output
    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(T));
    cudaMalloc((void**)&d_output, size * sizeof(T));

    // Copy input from host to device
    cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the RELU derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <typename T>
__global__ void RELU_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] > 0 ? 1 : 0;
    }
}

template <typename T>
Linear<T>::Linear(int rows, int cols):Matrix<T>(rows, cols){
    this->rows = rows;
    this->cols = cols;
}

template <typename T>
void Linear<T>::forward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    cudaMalloc((void**)&d_input, input_size * sizeof(T));
    cudaMalloc((void**)&d_output, output_size * sizeof(T));
    cudaMalloc((void**)&d_weights, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_biases, rows * cols * sizeof(T));

    // Copy input, weights, and biases from host to device
    cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the linear kernel
    linear_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
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
void Linear<T>::backward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    cudaMalloc((void**)&d_input, input_size * sizeof(T));
    cudaMalloc((void**)&d_output, output_size * sizeof(T));
    cudaMalloc((void**)&d_weights, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_biases, rows * cols * sizeof(T));

    // Copy input, weights, and biases from host to device
    cudaMemcpy(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);

    // Copy the result output from device to host
    cudaMemcpy(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

template <typename T>
__global__ void linear_derivative_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
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
void Linear<T>::update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size){
    // Allocate device memory for weights, biases, d_weights, and d_biases
    T *d_weights, *d_biases, *d_d_weights, *d_d_biases;
    cudaMalloc((void**)&d_weights, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_biases, rows * sizeof(T));
    cudaMalloc((void**)&d_d_weights, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_d_biases, rows * sizeof(T));

    // Copy weights, biases, d_weights, and d_biases from host to device
    cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, rows * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_weights, d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_biases, d_biases, rows * sizeof(T), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(cols, rows, 1);

    // Launch the update weights kernel
    update_weights_kernel<T><<<gridDim, blockDim>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, input_size, output_size);

    // Copy the result weights and biases from device to host
    cudaMemcpy(weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_d_weights);
    cudaFree(d_d_biases);
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

    this->input = new T[input_size];
    for (int i = 0; i < num_layers; i++) {
        this->hidden[i] = new T[hidden_size[i]];
    }
    this->output = new T[output_size];
}

template <typename T>
void Network<T>::backward(T *input, T *output){
    this->activation[num_layers]->backward(this->output, this->output, this->output_size);
    this->output_layer->backward(this->hidden[num_layers-1], this->output, this->output_layer->weights, this->output_layer->biases, this->hidden_size[num_layers-1], this->output_size);
    for(int i = num_layers-1; i >= 0; i--){
        this->activation[i]->backward(this->hidden[i+1], this->hidden[i+1], this->hidden_size[i+1]);
        this->hidden_layer[i]->backward(this->hidden[i], this->hidden[i+1], this->hidden_layer[i]->weights, this->hidden_layer[i]->biases, this->hidden_size[i], this->hidden_size[i+1]);
    }
    this->input_layer->backward(input, this->hidden[0], this->input_layer->weights, this->input_layer->biases, this->input_size, this->hidden_size[0]);
}

template <typename T>
void Network<T>::update_weights(T learning_rate){
    this->input_layer->update_weights(this->input_layer->weights, this->input_layer->biases, learning_rate, this->input_size, this->hidden_size[0]);
    for(int i = 0; i < num_layers; i++){
        this->hidden_layer[i]->update_weights(this->hidden_layer[i]->weights, this->hidden_layer[i]->biases, learning_rate, this->hidden_size[i], this->hidden_size[i+1]);
    }
    this->output_layer->update_weights(this->output_layer->weights, this->output_layer->biases, learning_rate, this->hidden_size[num_layers-1], this->output_size);
}


template <typename T>
void Network<T>::train(T *input, T *output, int epochs, T learning_rate){
    for(int i = 0; i < epochs; i++){
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
void Network<T>::addLayer(Linear<T> *layer){
    layers.push_back(layer);
}

template <typename T>
void Network<T>::addLayer(Sigmoid<T> *layer){
    layers.push_back(layer);
}

template <typename T>
void Network<T>::forward(T *input, T *output){
    for(int i = 0; i < layers.size(); i++){
        layers[i]->forward(input, output);
    }
}

template <typename T>
Network<T>:: ~Network(){
    for(int i=0; i<num_layers; i++){
        delete[] layers[i];
    }

}



