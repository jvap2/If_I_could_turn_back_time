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
    InitializeVector<float>(target, output_size);
    Network<float> net(input_size,  hidden_layers, output_size, 0);
    net.addLayer(new Linear<float>(input_size, hidden_layers[0]));
    net.addLayer(new RELU_layer<float>(hidden_layers[0]));
    net.addLayer(new Linear<float>(hidden_layers[0], hidden_layers[1]));
    net.addLayer(new RELU_layer<float>(hidden_layers[1])); //NULL layer for backprop
    net.addLayer(new Linear<float>(hidden_layers[1], output_size)); //NULL layer for backprop
    net.addLayer(new Softmax<float>(output_size));
    net.addLoss(new Categorical<float>(output_size));
    //Print out the size of the categorical layer

    net.train(input, target,10,.01);
    // net.forward(input,target);
    float* output = new float[output_size];
    cout<<endl;
    net.getOutput(output);
    for(int i = 0; i < output_size; i++){
        std::cout << output[i] << " ";
    }
    cout<<endl;
    // for(int i = 0; i < output_size; i++){
    //     std::cout << target[i] << " ";
    // }
    cout<<endl;


    // Add layers to the network
    







    return 0;
}