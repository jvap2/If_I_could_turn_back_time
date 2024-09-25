#include "../include/GPUErrors.h"
#include "../include/layers.h"


int main(int argc, char** argv){
    double cpuComputeTime = 0.0f;
    double gpuComputeTime = 0.0f;
    char mask[]={0,-1,0,-1,5,-1,0,-1};
    cout << "Loading Images" << endl;

    int imageIndex = 1;
    std::string filePath = std::string(ARBORIO_FILE) + std::to_string(imageIndex) + ").jpg";
    CImg<unsigned char> imgRGB(filePath.c_str());

    // Display Image
    // CImgDisplay dispRGB(imgRGB, "Grey");

    // Display Image Size
    cout << "Image Height: " << imgRGB.height() << " pixels" << endl;
    cout << "Image Width: " << imgRGB.width() << " pixels" << endl;
	cout << "Image Depth: " << imgRGB.depth() << " pixels" << endl;
	int height = imgRGB.height();
	int width = imgRGB.width();
	int depth = imgRGB.depth();

    // Store RGB image size in bytes
    unsigned int greySize = imgRGB.width() * imgRGB.height() * sizeof(unsigned char);
    unsigned int blurSize = (imgRGB.width() - 2) * (imgRGB.height() - 2) * sizeof(unsigned char);

    // Initialize a pointer to the RGB image data stored by CImg
    unsigned char* ptrRGB = imgRGB.data();

    // Create an empty image with a single channel - GrayScale
    CImg<unsigned char> imgGrayScale(imgRGB.width(), imgRGB.height());

    // Create an empty image for blurring
    CImg<unsigned char> imgBlur(imgRGB.width() - 2, imgRGB.height() - 2);
    CImg<unsigned char> imgBlur_GPU(imgRGB.width() - 2, imgRGB.height() - 2);
	int input_size, output_size, training_size, test_size, num_layers, batch_size, Q, size;

	cout<<argv[1]<<endl;
	size = NUM_RICE*RICE_TYPE_SIZE;
	input_size = height*width*depth;
	output_size = NUM_RICE;
	training_size = (int)size*TRAIN;
	test_size = size-training_size;
	num_layers = 3;
	int* hidden_layers = new int[num_layers-2];
	hidden_layers[0] = 16;
	batch_size = 128;
	Q = 128;
    // Create a network
    float** input = new float*[size];
    float** target = new float*[size];
    for(int i = 0; i < size; i++){
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
	Read_Rice_Data<float>(input, target,input_size, output_size);
    Train_Split_Test<float>(input, target, train_input, train_target, test_input, test_target, training_size,test_size,size, input_size, output_size);
    int kernel_width = 3;
    int kernel_height = 3;
    int stride = 1;
    int padding = 1;
    int filters = 1;
    int channels = 1;
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
    net.train(train_input, train_target, 75, .001, training_size);
    // cout<<"Training Complete"<<endl;
    // // cout<<"Results on Training Data"<<endl;
    // // net.predict(train_input,train_target, training_size);
    // cout<<"Results on Test Data"<<endl;
    // net.predict(test_input,test_target, test_size);

    // Your additional code here

    return 0;
}