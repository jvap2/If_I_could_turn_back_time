#include "GPUErrors.h"

template <typename T>
class Matrix
{
public:
    Matrix();
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
    void set_rows(int rows);
    void set_cols(int cols);
    int get_rows();
    int get_cols();
private:
    cudaError_t cudaStatus;
};

template <typename T>
class Sigmoid: public Matrix
{
public:
    Sigmoid();
    ~Sigmoid();
    void forward(T *input, T *output, int size);
    void backward(T *input, T *output, int size);
};

template <typename T>
class RELU_layer: public Matrix
{
public:
    RELU_layer();
    ~RELU_layer();
    void forward(T *input, T *output, int size);
    void backward(T *input, T *output, int size);
};

template <typename T>
class Linear: public Matrix
{
    public:
        Linear();
        ~Linear();
        void forward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size);
        void update(T *weight, T *bias, T *d_weight, T *d_bias, int input_size, int output_size, T learning_rate);
};

template <typename T>
class Softmax: public Matrix
{
    public:
        Softmax();
        ~Softmax();
        void forward(T *input, T *output, int size);
        void backward(T *input, T *output, int size);
};

