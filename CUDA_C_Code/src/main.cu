#include "../include/GPUErrors.h"
#include "../include/layers.h"



int main(){
    int input_size = WEATHER_INPUT_SIZE;
    int output_size = WEATHER_OUTPUT_SIZE;
    int num_layers = 4;
    int* hidden_layers = new int[num_layers-2];
    hidden_layers[0] = 512;
    hidden_layers[1] = 256;
    int batch_size = 64;
    // Create a network
    float** input = new float*[WEATHER_SIZE];
    float** target = new float*[WEATHER_SIZE];
    for(int i = 0; i < input_size; i++){
        input[i] = new float[input_size]{};
        target[i] = new float[output_size]{};
    }
    Read_Weather_Data(input, target);
    Network<float> net(input_size,  hidden_layers, output_size, 0);
    net.addLayer(new Linear<float>(input_size, hidden_layers[0]));
    net.addLayer(new RELU_layer<float>(hidden_layers[0]));
    net.addLayer(new Linear<float>(hidden_layers[0], hidden_layers[1]));
    net.addLayer(new RELU_layer<float>(hidden_layers[1])); //NULL layer for backprop
    net.addLayer(new Linear<float>(hidden_layers[1], output_size)); //NULL layer for backprop
    net.addLayer(new Softmax<float>(output_size));
    net.addLoss(new Categorical<float>(output_size));
    //Print out the size of the categorical layer

    net.train(input, target, 15, .001, WEATHER_SIZE, batch_size);


    return 0;
}