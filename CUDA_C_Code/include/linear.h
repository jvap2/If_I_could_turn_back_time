#include "GPUErrors.h"


class Linear
{
    public:
        Linear();
        ~Linear();
        void forward(float *input, float *output, float *weight, float *bias, int input_size, int output_size);
        void backward(float *input, float *output, float *weight, float *bias, int input_size, int output_size);
        void update(float *weight, float *bias, float *d_weight, float *d_bias, int input_size, int output_size, float learning_rate);
};