#include "GPUErrors.h"

class Sigmoid
{
public:
    Sigmoid();
    ~Sigmoid();
    void forward(float *input, float *output, int size);
    void backward(float *input, float *output, int size);
};

