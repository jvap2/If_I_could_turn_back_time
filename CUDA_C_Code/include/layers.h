#include "GPUErrors.h"
#include "include.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
class Matrix
{
public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, T *data);
    ~Matrix();
    int rows;
    int cols;
    T *d_data;
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
    int get_rows();
    int get_cols();
private:
    cudaError_t cudaStatus;
};

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
class Softmax: public Matrix<T>
{
    public:
        Softmax();
        ~Softmax();
        void forward(T *input, T *output, int size);
        void backward(T *input, T *output, int size);
};

template <typename T>
class Linear: public Matrix<T>
{
    public:
        Linear(int rows, int cols);
        int rows;
        int cols;
        T* weights;
        T* biases;
        ~Linear();
        void forward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void update_weights(T *weights, T *biases, T learning_rate, int input_size, int output_size);
        void set_weights(T *weights, T *biases);
};

template <typename T>
class Network
{
    public:
        Network(int input_size, int* hidden_size, int output_size, int num_layers);
        ~Network();
        int input_size;
        int *hidden_size;
        int output_size;
        int num_layers;
        Matrix<T> input;
        Matrix<T> output;
        Matrix<T>* hidden;
        thrust::host_vector<Linear<T>*> layers;  // Change this line
        void forward(T *input, T *output);
        void backward(T *input, T *output);
        void update_weights(T learning_rate);
        void addLayer(Linear<T>* layer);
        void addLayer(Sigmoid<T>* layer);    
        void addLayer(RELU_layer<T>* layer);
        void addLayer(Softmax<T>* layer);
        void train(T *input, T *output, int epochs, T learning_rate);
        void predict(T *input, T *output);
        void set_input_size(int input_size);
        void set_hidden_size(int* hidden_size);
        void set_output_size(int output_size);
        void set_num_layers(int num_layers);
        int get_input_size();
        int* get_hidden_size();
        int get_output_size();
};

template <typename T>
__global__ void matrix_multiply_kernel(T *A, T *B, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_add_kernel(T *A, T *B, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_subtract_kernel(T *A, T *B, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_transpose_kernel(T *A, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_scalar_multiply_kernel(T *A, T *C, T scalar, int rows, int cols);

template <typename T>
__global__ void matrix_scalar_add_kernel(T *A, T *C, T scalar, int rows, int cols);

template <typename T>
__global__ void matrix_scalar_subtract_kernel(T *A, T *C, T scalar, int rows, int cols);

template <typename T>
__global__ void matrix_scalar_divide_kernel(T *A, T *C, T scalar, int rows, int cols);

template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_elementwise_divide_kernel(T *A, T *B, T *C, int rows, int cols);


template <typename T>
__global__ void matrix_elementwise_add_kernel(T *A, T *B, T *C, int rows, int cols);


template <typename T>
__global__ void matrix_elementwise_subtract_kernel(T *A, T *B, T *C, int rows, int cols);

template <typename T>
__global__ void matrix_sum_kernel(T *A, T *C, int rows, int cols, int axis);

template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size);

template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T *output, int size);

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size);

template <typename T>
__global__ void RELU_derivative_kernel(T *input, T *output, int size);

template <typename T>
__global__ void linear_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size);

template <typename T>
__global__ void linear_derivative_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size);

template <typename T>
__global__ void update_weights_kernel(T *weights, T *biases, T *d_weights, T *d_biases, T learning_rate, int input_size, int output_size);

template <typename T>
__global__ void matrix_sum_axis0_kernel(T *A, T *C, int rows, int cols);


template <typename T>
__global__ void matrix_sum_axis1_kernel(T *A, T *C, int rows, int cols);
