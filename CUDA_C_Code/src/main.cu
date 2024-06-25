#include "../include/GPUErrors.h"
#include "../include/layers.h"



int main(){
    int input_size = 784;
    int output_size = 10;
    int num_layers = 4;
    int* hidden_layers = new int[num_layers-2];
    hidden_layers[0] = 512;
    hidden_layers[1] = 256;
    // Create a network
    float* input = new float[input_size];
    float* target = new float[output_size];
    InitializeVector<float>(input, input_size);
    ZeroVector(target, output_size);
    Network<float> net(input_size,  hidden_layers, output_size, 4);
    net.addLayer(new Linear<float>(input_size, hidden_layers[0]));
    net.addLayer(new RELU_layer<float>(hidden_layers[0], hidden_layers[0]));
    net.addLayer(new Linear<float>(hidden_layers[0], hidden_layers[1]));
    net.addLayer(new RELU_layer<float>(hidden_layers[1], hidden_layers[1]));
    net.addLayer(new Linear<float>(hidden_layers[1], output_size));
    net.addLayer(new Softmax<float>(output_size, output_size));
    net.addLoss(new Categorical<float>(output_size));

    net.train(input, target,10,.01);



    // Add layers to the network
    







    return 0;
}