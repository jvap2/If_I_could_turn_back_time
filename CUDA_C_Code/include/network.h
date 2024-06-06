#include "GPUErrors.h"
#include "layers.h"

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
        Matrix<T> **hidden_layers;
        Matrix<T> **input_layer;
        Matrix<T> **output_layer;
        void forward(T *input, T *output, int size);
        void backward(T *input, T *output, int size);
        void update_weights(T *input, T *output, int size);
        void train(T *input, T *output, int size);
        void predict(T *input, T *output, int size);
        void set_input_size(int input_size);
        void set_hidden_size(int hidden_size);
        void set_output_size(int output_size);
        int get_input_size();
        int get_hidden_size();
        int get_output_size();
};