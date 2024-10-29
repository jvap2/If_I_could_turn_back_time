#include "../include/GPUErrors.h"
#include "../include/layers.h"



int main(int argc, char** argv){
    int input_size, output_size, training_size, test_size, num_layers, batch_size, Q, size;
    bool weather;
    if(argc != 2){
        std::cout << "Usage: ./main <dataset>" << std::endl;
        return 1;
    }
    if(strcmp(argv[1],"weather")==0){
        cout<<argv[1]<<endl;
        size = WEATHER_SIZE;
        input_size = WEATHER_INPUT_SIZE;
        output_size = WEATHER_OUTPUT_SIZE;
        training_size = (int)WEATHER_SIZE*TRAIN;
        test_size = WEATHER_SIZE-training_size;
        num_layers = 3;
        int* hidden_layers = new int[num_layers-2];
        hidden_layers[0] = 16;
        batch_size = 128;
        Q = 128;
        weather = true;
    }
    else if(strcmp(argv[1],"heart")==0){
        cout<<argv[1]<<endl;
        size = HEART_SIZE;
        input_size = HEART_INPUT_SIZE;
        output_size = HEART_OUTPUT_SIZE;
        training_size = (int)HEART_SIZE*TRAIN;
        test_size = HEART_SIZE-training_size;
        num_layers = 3;
        int* hidden_layers = new int[num_layers-2];
        hidden_layers[0] = 16;
        batch_size = 128;
        Q = 128;
        weather = false;
        cout<<"Heart"<<endl;
    }
    else if(strcmp(argv[1],"dummy")==0){
        cout<<argv[1]<<endl;
        size = DUMMY_SIZE;
        input_size = DUMMY_INPUT_SIZE;
        output_size = DUMMY_OUTPUT_SIZE;
        training_size = DUMMY_SIZE;
        test_size = DUMMY_SIZE-training_size;
        num_layers = 3;
        int* hidden_layers = new int[num_layers-2];
        hidden_layers[0] = 16;
        batch_size = DUMMY_SIZE;
        Q = 128;
        weather = false;
        cout<<"Dummy"<<endl;
    }
    else{
        std::cout << "Invalid dataset" << std::endl;
        return 1;
    }
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
    if(strcmp(argv[1],"weather")==0){
        cout<<"Reading Weather Data"<<endl;
        Read_Weather_Data_Norm(input, target);
    }
    if(strcmp(argv[1],"heart")==0){
        cout<<"Reading Heart Data"<<endl;
        Read_Heart_Data(input, target);
    }
    if(strcmp(argv[1],"dummy")==0){
        cout<<"Reading Dummy Data"<<endl;
        Read_Dummy_Data(input, target);
    }
    Train_Split_Test(input, target, train_input, train_target, test_input, test_target, training_size,test_size,size, input_size, output_size);
    // AdamOptimizer<float>* optimizer = new AdamOptimizer<float>(.001, .9, .999, 1e-8);
    // AdamOptimizer<float>* optimizer = new AdamOptimizer<float>(.0001, .9, .999, 1e-8);
    AdamJenksOptimizer <float>* optimizer = new AdamJenksOptimizer<float>(.001, .9, .999, 1e-8);
    Network<float> net(input_size, output_size, optimizer,Q,batch_size);
    net.addLayer(new Linear<float>(input_size, 128,batch_size));
    net.addLayer(new Tanh<float>(128,batch_size));
    net.addLayer(new Linear<float>(128, 256, batch_size));
    net.addLayer(new Tanh<float>(256,batch_size));
    net.addLayer(new Linear<float>(256, 64, batch_size));
    net.addLayer(new Tanh<float>(64,batch_size));
    net.addLayer(new Linear<float>(64, output_size, batch_size));
    net.addLayer(new Softmax<float>(output_size,batch_size));
    if(strcmp(argv[1],"weather")==0){
        cout<<"Adding Categorical Loss"<<endl;
        net.addLoss(new Categorical<float>(output_size,batch_size));
    }
    if(strcmp(argv[1],"heart")==0){
        cout<<"Adding Binary Cross Entropy Loss"<<endl;
        net.addLoss(new Binary_CrossEntropy<float>(output_size,batch_size));
    }
    if(strcmp(argv[1],"dummy")==0){
        cout<<"Adding MSE Loss"<<endl;
        net.addLoss(new Binary_CrossEntropy<float>(output_size,batch_size));
    }
    //Print out the size of the categorical layer

    net.train(train_input, train_target, 75, .001, training_size);
    cout<<"Training Complete"<<endl;
    // cout<<"Results on Training Data"<<endl;
    // net.predict(train_input,train_target, training_size);
    cout<<"Results on Test Data"<<endl;
    net.predict(test_input,test_target, test_size);


    return 0;
}