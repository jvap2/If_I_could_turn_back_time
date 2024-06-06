#include "../include/include.h"
#include "../include/GPUErrors.h"
#include "../include/layers.h"



int main(){
    int input_size = 784;
    int output_size = 10;
    int* hidden_layers = new int[2];
    hidden_layers[0] = 512;
    hidden_layers[1] = 256;
    // Create a network
    Network<float> net(input_size,  hidden_layers, output_size, 2);



    // Add layers to the network
    







    return 0;
}