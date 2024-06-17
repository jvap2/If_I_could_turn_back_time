#include "GPUErrors.h"
#include "include.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
class Matrix
{
public:
    Matrix(){
        this->rows = 0;
        this->cols = 0;
        this->weights = NULL;
        this->biases = NULL;
    }
    Matrix(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
        this->weights = (T*)malloc(rows * cols * sizeof(T));
        this->biases = (T*)malloc(rows * sizeof(T));
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
    void forward(T *input, T *output);
    void backward(T *input, T *output){};
    void backward(T *input, T *output, int size){};
    void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size){};
    void update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size){};
    void train(T *input, T *output, int epochs, T learning_rate){};
    int get_rows();
    int get_cols();
private:
    cudaError_t cudaStatus;
};

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
}


template <typename T>
class Sigmoid: public Matrix<T>
{
public:
    Sigmoid(int rows, int cols);
    int rows;
    int cols;
    ~Sigmoid();
    void forward(T *input, T *output, int size);
    void backward(T *input, T *output, int size);
};

template <typename T>
class RELU_layer: public Matrix<T>
{
public:
    RELU_layer(int rows, int cols);
    int rows;
    int cols;
    ~RELU_layer();
    void forward(T *input, T *output, int size);
    void backward(T *input, T *output, int size);
};


template <typename T>
__global__ void softmax_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = exp(input[index]) / thrust::reduce(input, input + size);
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
            this->cols = cols;
        }
        ~Softmax();
        void forward(T *input, T *output, int size){
            // Allocate device memory for input and output
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
            dim3 blockDim(size, 1, 1);

            // Launch the softmax kernel
            softmax_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
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
        Linear(int rows, int cols){
            this->rows = rows;
            this->cols = cols;
            this->weights = (T*)malloc(rows * cols * sizeof(T));
            this->biases = (T*)malloc(rows * sizeof(T));
        }
        int rows;
        int cols;
        T* weights;
        T* biases;
        ~Linear(){
            free(this->weights);
            free(this->biases);
        }
        void forward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
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
        float** hidden;
        thrust::host_vector<Matrix<T>*> layers;  // Change this line
        thrust::host_vector<Matrix<T>*> activation;  // Change this line
        void backward(T *input, T *output);
        void update_weights(T learning_rate);
        void addLayer(Linear<T> *layer){
            layers.push_back(layer);
            num_layers++;
        }
        void addLayer(Conv2D<T> *layer){
            layers.push_back(layer);
            num_layers++;
        }
        void addLayer(MaxPooling2D<T> *layer){
            layers.push_back(layer);
            num_layers++;
        }
        void addLayer(Sigmoid<T> *layer){
            activation.push_back(layer);
            num_activation++;
        }   
        void addLayer(RELU_layer<T>* layer){
            activation.push_back(layer);
            num_activation++;
        }
        void addLayer(Softmax<T>* layer){
            activation.push_back(layer);
            num_activation++;
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
            for(int i = 0; i < layers.size(); i++){
                layers[i]->forward(input, output);
            }
        }
        void forward(Matrix<T> input, Matrix<T> output){
            for(int i = 0; i < layers.size(); i++){
                layers[i]->forward(input, output);
            }
        }
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
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
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
    this->cols = cols;
}


template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = 1 / (1 + exp(-input[index]));
    }
}

template <typename T>
void Sigmoid<T>::forward(T *input, T *output, int size){
    // Allocate device memory for input and output
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
}



template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] * (1 - input[index]);
    }
}


template <typename T>
void Sigmoid<T>::backward(T *input, T *output, int size){
    // Allocate device memory for input and output
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

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
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


template <typename T>
RELU_layer<T>::RELU_layer(int rows, int cols):Matrix<T>(rows, cols){
    this->rows = rows;
    this->cols = cols;
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = fmax(0, input[index]);
    }
}


template <typename T>
void RELU_layer<T>::forward(T *input, T *output, int size){
    // Allocate device memory for input and output
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
}


template <typename T>
__global__ void RELU_derivative_kernel(T *input, T *output, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = input[index] > 0 ? 1 : 0;
    }
}

template <typename T>
void RELU_layer<T>::backward(T *input, T *output, int size){
    // Allocate device memory for input and output
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

    // Launch the RELU derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size);
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
void Linear<T>::forward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
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

    // Launch the linear kernel
    linear_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);
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
void Linear<T>::backward(T *input, T *output, T *weights, T *biases, int input_size, int output_size){
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

    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);
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
}

template <typename T>
void Network<T>::backward(T *input, T *output){
    this->activation[num_activation-1]->backward(this->output, this->output);
    this->layers[num_layers-1]->backward(this->hidden[num_layers-1], this->output, this->layers[num_layers-1]->weights, this->layers[num_layers-1]->biases, this->hidden_size[num_layers-1], this->output_size);
    for(int i = num_layers-2; i >= 0; i--){
        this->activation[i]->backward(this->hidden[i+1], this->hidden[i+1]);
        this->layers[i]->backward(this->hidden[i], this->hidden[i+1], this->layers[i]->weights, this->layers[i]->biases, this->hidden_size[i], this->hidden_size[i+1]);
    }
    this->layers[0]->backward(input, this->hidden[0], this->layers[0]->weights, this->layers[0]->biases, this->input_size, this->hidden_size[0]);
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
    d_biases[outRow]+=output[outRow*input_size+outCol];
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
    if (!HandleCUDAError(cudaMemcpy(dweights, d_dweights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost))) {
        cout << "Error in copying dweights from device to host" << endl;
        return;
    }
    if (!HandleCUDAError(cudaMemcpy(dbiases, d_dbiases, rows * sizeof(T), cudaMemcpyDeviceToHost))) {
        cout << "Error in copying dbiases from device to host" << endl;
        return;
    }
    if (!HandleCUDAError(cudaMemcpy(dinput, d_dinput, input_size * sizeof(T), cudaMemcpyDeviceToHost))) {
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