#include "../include/GPUErrors.h"
#include "../include/layers.h"



int main(){
    int input_size = WEATHER_INPUT_SIZE;
    int output_size = WEATHER_OUTPUT_SIZE;
    int training_size = (int)WEATHER_SIZE*TRAIN;
    int test_size = WEATHER_SIZE-training_size;
    int num_layers = 3;
    int* hidden_layers = new int[num_layers-2];
    hidden_layers[0] = 16;
    int batch_size = 32;
    int Q = 64;
    // Create a network
    float** input = new float*[WEATHER_SIZE];
    float** target = new float*[WEATHER_SIZE];
    for(int i = 0; i < WEATHER_SIZE; i++){
        input[i] = new float[input_size]{};
        target[i] = new float[output_size]{};
    }
    float** test_input = new float*[test_size];
    float** test_target = new float*[test_size];
    for(int i = 0; i < test_size; i++){
        test_input[i] = new float[input_size]{};
        test_target[i] = new float[output_size]{};
    }
    float** train_input = new float*[training_size];
    float** train_target = new float*[training_size];
    for(int i = 0; i < training_size; i++){
        train_input[i] = new float[input_size]{};
        train_target[i] = new float[output_size]{};
    }
    Read_Weather_Data_Norm(input, target);
    Train_Split_Test(input, target, train_input, train_target, test_input, test_target, WEATHER_SIZE);
    // AdamOptimizer<float>* optimizer = new AdamOptimizer<float>(.001, .9, .999, 1e-8);
    AdamWBernoulli<float>* optimizer = new AdamWBernoulli<float>(.001, .9, .999, 1e-8);
    Network<float> net(input_size, output_size, optimizer,Q);
    net.addLayer(new Linear<float>(input_size, 16));
    net.addLayer(new RELU_layer<float>(16));
    net.addLayer(new Linear<float>(16, 32));
    net.addLayer(new RELU_layer<float>(32));
    // net.addLayer(new Linear<float>(512, 128));
    // net.addLayer(new RELU_layer<float>(128));
    net.addLayer(new Linear<float>(32, 16));
    net.addLayer(new RELU_layer<float>(16));
    net.addLayer(new Linear<float>(16, output_size));
    net.addLayer(new Softmax<float>(output_size));
    net.addLoss(new Categorical<float>(output_size));
    //Print out the size of the categorical layer

    net.train(train_input, train_target, 100, .001, training_size, batch_size);

    // net.predict(test_input,test_target, test_size);


    return 0;
}