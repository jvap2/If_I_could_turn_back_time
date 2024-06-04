#include "GPUErrors.h"

class RELU_layer
{
public:
    RELU_layer();
    ~RELU_layer();
    void forward(float *input, float *output, int size);
    void backward(float *input, float *output, int size);
};