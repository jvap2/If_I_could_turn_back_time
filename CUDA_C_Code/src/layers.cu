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

