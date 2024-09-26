#include "../include/GPUErrors.h"
#include "../include/layers.h"


int main(int argc, char** argv){
	int input_size, output_size, training_size, test_size, num_layers, batch_size, Q, size;
    int height = MNIST_HEIGHT;
    int width = MNIST_WIDTH;
    int depth = 1;
    batch_size = 8;
	input_size = height*width*depth;
	output_size = 10;
	training_size = MNIST_TRAIN_DATA;
	test_size = MNIST_TEST_DATA;
    size = training_size + test_size;
	Q = 128;
    // Create a network
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
    int kernel_width = 3;
    int kernel_height = 3;
    int stride = 1;
    int padding = 0;
    int filters = 1;
    int channels = 1;
    Read_MNIST_train_data(train_input, train_target, input_size,output_size);
    Read_MNIST_test_data(test_input, test_target, input_size,output_size);
    // AdamOptimizer<float>* optimizer = new AdamOptimizer<float>(.001, .9, .999, 1e-8);
    // AdamOptimizer<float>* optimizer = new AdamOptimizer<float>(.0001, .9, .999, 1e-8);
    AdamOptimizer <float>* optimizer = new AdamOptimizer<float>(.001, .9, .999, 1e-8);
    Network<float> net(input_size, output_size, optimizer,Q,batch_size);
    net.addLayer(new Conv2D<float>(width, height, channels, kernel_width, kernel_height, stride, padding, filters, batch_size));
    net.addLayer(new RELU_layer<float>(height-2, width-2, filters, batch_size));
    net.addLayer(new MaxPooling2D<float>(kernel_width, kernel_height, stride, padding, width-2, height-2, filters, batch_size));  
    net.addLayer(new Conv2D<float>((height-2)/2, (width-2)/2, filters, kernel_width, kernel_height, stride, padding, filters, batch_size));
    net.addLayer(new RELU_layer<float>((height-2)/2, (width-2)/2, filters, batch_size));
    net.addLayer(new MaxPooling2D<float>(kernel_width, kernel_height, stride, padding, (width-2)/2, (height-2)/2, filters, batch_size));
    net.addLayer(new Flatten<float>((width-2)/4, (height-2)/4,filters, batch_size));
    net.addLayer(new Linear<float>((height-2)/4*(width-2)/4*filters, 64, batch_size));
    net.addLayer(new RELU_layer<float>(64, batch_size));
    net.addLayer(new Softmax<float>(64, batch_size));
    net.addLoss(new Categorical<float>(64, batch_size));
    cout<<"Training Network"<<endl;
    net.train(train_input, train_target, 75, .001, training_size);
    // cout<<"Training Complete"<<endl;
    // // cout<<"Results on Training Data"<<endl;
    // // net.predict(train_input,train_target, training_size);
    // cout<<"Results on Test Data"<<endl;
    // net.predict(test_input,test_target, test_size);

    // Your additional code here

    return 0;
}