#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/set_operations.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <cusolverDn.h>
#include <cmath>
#include "GPUErrors.h"
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

#define WEATHER_DATA "../data/weather/weather_classification_data_cleaned.csv"
#define HEART_DATA "../data/heart/heart_classification_data_cleaned.csv"
#define DUMMY_DATA "../data/dummy/dummy.csv"
#define RICE_DATA_FOLDER "../data/Rice_Image_Dataset"
#define ARBORIO_RICE_FOLDER "/Arborio/"
#define BASMATI_RICE_FOLDER "/Basmati/"
#define IPSALA_RICE_FOLDER "/Ipsala/"
#define JASMINE_RICE_FOLDER "/Jasmine/"
#define KARACADAG_RICE_FOLDER "/Karacadag/"
#define ARBORIO_FILE "../data/Rice_Image_Dataset/Arborio/Arborio ("
#define BASMATI_FILE "../data/Rice_Image_Dataset/Basmati/basmati ("
#define IPSALA_FILE "../data/Rice_Image_Dataset/Ipsala/Ipsala ("
#define JASMINE_FILE "../data/Rice_Image_Dataset/Jasmine/Jasmine ("
#define KARACADAG_FILE "../data/Rice_Image_Dataset/Karacadag/Karacadag ("
#define MNIST_CSV_TRAIN "../data/mnist/MNISTDatasetJPG/mnist_file_info_training.csv"
#define MNIST_CSV_TEST "../data/mnist/MNISTDatasetJPG/mnist_file_info_testing.csv"
#define MNIST_TRAIN "../data/mnist/MNISTDatasetJPG/training/"
#define MNIST_TEST "../data/mnist/MNISTDatasetJPG/testing/"
#define MNIST_TEST_DATA 10000
#define MNIST_TRAIN_DATA 60000
#define IMAGE_HEIGHT 250
#define IMAGE_WIDTH 250
#define MNIST_HEIGHT 28
#define MNIST_WIDTH 28
#define MNIST_SIZE 70000
#define RICE_TYPE_SIZE 15000
#define RICE_TYPE_SIZE_SMALL 100
#define NUM_RICE 5
#define HEART_INPUT_SIZE 13
#define WEATHER_INPUT_SIZE 10
#define WEATHER_OUTPUT_SIZE 4
#define HEART_OUTPUT_SIZE 2
#define WEATHER_SIZE 13200
#define HEART_SIZE 303
#define DUMMY_SIZE 4
#define DUMMY_INPUT_SIZE 4
#define DUMMY_OUTPUT_SIZE 2
#define RANGE_MAX 0.5
#define RANGE_MIN -0.5
#define TRAIN .9
#define TEST .1



void Read_Weather_Data(float **data, float **output)
{
    std::ifstream file(WEATHER_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 10;
    int classes = 4;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}


void Read_Weather_Data_Norm(float **data, float **output)
{
    std::ifstream file(WEATHER_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 10;
    int classes = 4;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
    //Go through the data and normalize it
    //Find the max and min of each column
    float max[WEATHER_INPUT_SIZE];  
    float min[WEATHER_INPUT_SIZE];
    for(int i = 0; i < WEATHER_INPUT_SIZE; i++) {
        max[i] = -1000000;
        min[i] = 1000000;
    }
    for(int i = 0; i < WEATHER_SIZE; i++) {
        for(int j = 0; j < WEATHER_INPUT_SIZE; j++) {
            if(data[i][j] > max[j]) {
                max[j] = data[i][j];
            }
            if(data[i][j] < min[j]) {
                min[j] = data[i][j];
            }
        }
    }
    //Normalize the data
    for(int i = 0; i < WEATHER_SIZE; i++) {
        for(int j = 0; j < WEATHER_INPUT_SIZE; j++) {
            data[i][j] = (data[i][j] - min[j])/(max[j] - min[j]);
        }
    }
}

template <typename T>
void Read_Heart_Data(T **data, T **output)
{
    std::ifstream file(HEART_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = 13;
    int classes = 2;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}

template <typename T>
void Read_Dummy_Data(T **data, T **output)
{
    std::ifstream file(DUMMY_DATA);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = DUMMY_INPUT_SIZE;
    int classes = DUMMY_OUTPUT_SIZE;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    cout<<"Value: "<<std::stof(value)<<endl;
                    data[row - 1][col] = std::stof(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}


template <typename T>
void Read_Rice_Data(T **data, T **output, int input_size, int output_size)
{
    int row = 0;
    int col = 0;
    int col_max = input_size;
    int classes = output_size;
    for (int i = 0; i < NUM_RICE; i++)
    {
        std::string file;
        switch (i)
        {
        case 0:
            file = ARBORIO_FILE;
            break;
        case 1:
            file = BASMATI_FILE;
            break;
        case 2:
            file = IPSALA_FILE;
            break;
        case 3:
            file = JASMINE_FILE;
            break;
        case 4:
            file = KARACADAG_FILE;
            break;
        }
        for (int j = 1; j <= RICE_TYPE_SIZE_SMALL; j++)
        {
            std::string file_name = file + std::to_string(j) + ").jpg";
            CImg<T> image(file_name.c_str());
            for (int k = 0; k < IMAGE_HEIGHT; k++)
            {
                for (int l = 0; l < IMAGE_WIDTH; l++)
                {
                    data[row][k * IMAGE_WIDTH + l] = image(l, k, 0, 0); //goes x,y,z,c
                }
            }
            for (int k = 0; k < classes; k++)
            {
                output[row][k] = (k == i) ? 1.0f : 0.0f;
            }
            row++;
        }
    }
}

template <typename T>
void Read_MNIST_train_data(T **data, T **output, int input_size, int output_size)
{
    //Read the csv files which have the columns label,file_name
    //Read the images and store them in the data array
    //Store the labels in the output array
    std::ifstream file_train(MNIST_CSV_TRAIN);
    std::ifstream file_test(MNIST_CSV_TEST);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = MNIST_HEIGHT * MNIST_WIDTH;
    int classes = 10;
    int label;
    while (std::getline(file_train, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    label = std::stoi(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    //Take label and read the image
                    std::string file_name = MNIST_TRAIN + std::to_string(temp) + "/" + std::to_string(label) + ".jpg";
                    CImg<T> image(file_name.c_str());
                    for (int i = 0; i < MNIST_HEIGHT; i++)
                    {
                        for (int j = 0; j < MNIST_WIDTH; j++)
                        {
                            data[row - 1][i * MNIST_WIDTH + j] = image(j, i, 0, 0); //goes x,y,z,c
                        }
                    }
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                    
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}

template <typename T>
void Read_MNIST_test_data(T **data, T **output, int input_size, int output_size)
{
    //Read the csv files which have the columns label,file_name
    //Read the images and store them in the data array
    //Store the labels in the output array
    std::ifstream file_test(MNIST_CSV_TEST);
    std::string line;
    int row = 0;
    int col = 0;
    int col_max = MNIST_HEIGHT * MNIST_WIDTH;
    int classes = 10;
    int label;
    while (std::getline(file_test, line))
    {
        std::stringstream ss(line);
        std::string value;
        if (row == 0)
        {
            // Skip header or initial row if necessary
            row++;
            continue;
        }
        while (std::getline(ss, value, ','))
        {
            try
            {
                if (col < col_max)
                {
                    // Convert string to float safely
                    label = std::stoi(value);
                }
                else
                {
                    // Convert string to int safely and update output array
                    int temp = std::stoi(value);
                    //Take label and read the image
                    std::string file_name = MNIST_TRAIN + std::to_string(temp) + "/" + std::to_string(label) + ".jpg";
                    CImg<T> image(file_name.c_str());
                    for (int i = 0; i < MNIST_HEIGHT; i++)
                    {
                        for (int j = 0; j < MNIST_WIDTH; j++)
                        {
                            data[row - 1][i * MNIST_WIDTH + j] = image(j, i, 0, 0); //goes x,y,z,c
                        }
                    }
                    for (int i = 0; i < classes; i++)
                    {
                        output[row - 1][i] = (i == temp) ? 1.0f : 0.0f;
                    }
                    
                }
            }
            catch (const std::exception &e)
            {
                // Handle or log conversion error
                std::cerr << "Conversion error: " << e.what() << '\n';
                // Consider setting a default value or skipping this value
            }
            col++;
        }
        col = 0;
        row++;
    }
}


template <typename T>
void Train_Split_Test(T **data, T **output, T **train_data, T **train_output, T **test_data, T **test_output, int training_size, int test_size, int size, int input_size, int output_size)
{
    for (int i = 0; i < training_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            train_data[i][j] = data[i][j];
        }
        for (int j = 0; j < output_size; j++)
        {
            train_output[i][j] = output[i][j];
        }
    }
    if(test_size == 0) {
        return;
    }
    else {
        cout<<"Test size: "<<test_size<<endl;
        for (int i = training_size; i < size; i++)
        {
            for (int j = 0; j < input_size; j++)
            {
                test_data[i - training_size][j] = data[i][j];
            }
            for (int j = 0; j < output_size; j++)
            {
                test_output[i - training_size][j] = output[i][j];
            }
        }
    }
}

struct LayerMetadata
{
    int layerNumber;
    int LinNumber;
    bool isUpdateable;

    LayerMetadata(int number, bool updateable) : layerNumber(number), isUpdateable(updateable) {}
    LayerMetadata(int number, int linNumber, bool updateable) : layerNumber(number), LinNumber(linNumber), isUpdateable(updateable) {}
};

template <typename T>
struct Loc_Layer
{
    int row;
    int col;
    int layer;
    T weights_dW;
    int rank;
};



template <typename T>
struct CompareBernoulliWeights {
    __host__ __device__
    bool operator()(const Loc_Layer<T>& lhs, const Loc_Layer<T>& rhs) const {
        // Assuming Loc_Layer has a member function or variable to get the weight
        return lhs.weights_dW < rhs.weights_dW; // Sort in ascending order
    }
};

template <typename T>
struct IsZero
{
    __host__ __device__
    bool operator()(const T& x) const
    {
        return fabsf(x)<=1e-8;
    }
};

template <typename T>
void InitializeMatrix(T *matrix, int ny, int nx)
{

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            matrix[j] = ((T)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
            matrix[j]/=nx;
        }
        matrix += nx;
    }
}

template <typename T>
void InitMatrix_Xavier(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            // srand(time(NULL));
            matrix[j] = ((T)rand() / (RAND_MAX + 1) * (upper - lower) + lower);
        }
        matrix += nx;
    }
}

template <typename T>
void InitMatrix_He(T* matrix, int ny, int nx){
    T upper = sqrt(2.0 / nx);
    T lower = -upper;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            matrix[j] = ((T)rand() / RAND_MAX * (upper - lower) + lower);
        }
        matrix += nx;
    }
}

template <typename T>
void InitMatrix_Xavier_Norm(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            // srand(time(NULL));
            matrix[j] = ((T)rand() / (RAND_MAX + 1) * (upper - lower) + lower);
            matrix[j] /= nx;
        }
        matrix += nx;
    }
}

template <typename T>
void ZeroMatrix(T *temp, const int ny, const int nx)
{
    T *p = temp;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            p[j] = 0.0f;
        }
        p += nx;
    }
}

template <typename T>
void InitializeVector(T *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = ((T)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
    }
}


template <typename T>
void Init_Bias_Xavier_Norm(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;

    for (int i = 0; i < ny; i++)
    {
        matrix[i] = ((T)rand() / (RAND_MAX + 1) * (upper - lower) + lower);
        matrix[i] /= nx;
    }
}

template <typename T>
void Init_Bias_He(T* matrix, int ny, int nx){
    T upper = sqrt(2.0 / nx);
    T lower = -upper;
    for (int i = 0; i < ny; i++) {
        matrix[i] = ((T)rand() / RAND_MAX * (upper - lower) + lower);
    }
}

template <typename T>
void Init_Weights_Same_Xavier(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;
    T val = .05;
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            // srand(time(NULL));
            matrix[j] = val;
        }
        matrix += nx;
    }
}

template <typename T>
void Init_Bias_Same_Xavier(T* matrix, int ny, int nx){
    T upper,lower;
    upper = sqrt(6.0/(nx+ny));
    lower = -upper;
    T val = .05;
    for (int i = 0; i < ny; i++)
    {
        matrix[i] = val;
    }
}

template <typename T>
void ZeroVector(T *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = 0.0f;
    }
}


template <typename T>
void Format_Batch_Data(T **data, T **output, T *batch_data, T *batch_output, int* indices, int batch_size, int input_size, int output_size)
{
    // Store the data in a 1D array, as an input to the neural network.
    //This should be like a matrix, where the entry of the first row includes the first entries of each input
    //The entry of the second row includes the second entries of each input, etc.
    // In the given data, each entry is given in a row
    for(int i = 0; i<input_size; i++){
        for(int j = 0; j<batch_size; j++){
            batch_data[i*batch_size + j] = data[indices[j]][i];
        }
    }
    for(int i = 0; i<output_size; i++){
        for(int j = 0; j<batch_size; j++){
            batch_output[i*batch_size + j] = output[indices[j]][i];
        }
    }
}

template <typename T>
class Matrix
{
public:
    T *input;
    string name;
    T *weights;
    T *biases;
    T *d_weights;
    T *d_biases;
    T *hidden_output;
    T *loss;
    T *next_loss;
    int* B_weights;
    int* B_biases;
    T* W_dW_weights;
    T* W_dW_biases;
    Matrix(){};
    Matrix(int cols, int rows)
    {
        this->rows = rows;
        this->cols = cols;
        this->channels = 1;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(cols * sizeof(T));
        this->loss = (T *)malloc(rows * sizeof(T));
        this->B_weights = (int *)malloc(rows * cols * sizeof(int));
        this->B_biases = (int *)malloc(rows * sizeof(int));
        this->W_dW_weights = (T *)malloc(rows * cols * sizeof(T));
        this->W_dW_biases = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        // InitializeMatrix<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->weights, rows, cols);
        Init_Bias_He<T>(this->biases, rows,cols);
        this->name = "full matrix";
        this->next_loss = (T *)malloc(cols * sizeof(T));
        this->d_biases = (T *)malloc(rows * sizeof(T));
        this->d_weights = (T *)malloc(rows * cols * sizeof(T));
    }
    Matrix(int cols, int rows, int batch_size)
    {
        this->batch_size = batch_size;
        this->rows = rows;
        this->cols = cols;
        cout<<"Rows: "<<rows<<" Cols: "<<cols<<endl;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->B_weights = (int *)malloc(rows * cols * sizeof(int));
        this->B_biases = (int *)malloc(rows * sizeof(int));
        this->W_dW_weights = (T *)malloc(rows * cols * sizeof(T));
        this->W_dW_biases = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        // InitializeMatrix<T>(this->weights, rows, cols);
        InitMatrix_He<T>(this->weights, rows, cols);
        Init_Bias_He<T>(this->biases, rows,cols);
        this->name = "full matrix";
        this->d_biases = (T *)malloc(rows * sizeof(T));
        this->d_weights = (T *)malloc(rows * cols * sizeof(T));
    }
    Matrix(int rows)
    {
        this->rows = rows;
        this->cols = 1;
        this->weights = (T *)malloc(rows * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(rows * sizeof(T));
        this->next_loss = (T *)malloc(rows * sizeof(T));
        // Create random weights and biases
        InitializeVector<T>(this->weights, rows);
        InitializeVector<T>(this->biases, rows);
        ZeroVector<T>(this->input, rows);
        this->name = "vector";
    }

    Matrix(int rows, int cols, T *weights, T *biases)
    {
        this->rows = rows;
        this->cols = cols;
        this->weights = (T *)malloc(rows * cols * sizeof(T));
        this->biases = (T *)malloc(rows * sizeof(T));
        this->weights = weights;
        this->biases = biases;
        this->hidden_output = (T *)malloc(rows * sizeof(T));
        this->input = (T *)malloc(cols * sizeof(T));
        this->name = "matrix";
    }
    Matrix(int width, int height, int channels, int kernel_width, int kernel_height, int stride, int padding, int filters, int batch_size)
    {
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->filters = filters;
        this->rows = filters;
        this->cols = width * height;
        this->weights = (T *)malloc(filters * kernel_width * kernel_height * channels * sizeof(T));
        this->biases = (T *)malloc(filters * sizeof(T));
        this -> d_weights = (T *)malloc(filters * kernel_width * kernel_height * channels * sizeof(T));
        this -> d_biases = (T *)malloc(filters * sizeof(T));
        this->B_weights = (int *)malloc(filters * kernel_width * kernel_height * channels * sizeof(int));
        this->B_biases = (int *)malloc(filters * sizeof(int));
        this->W_dW_weights = (T *)malloc(filters * kernel_width * kernel_height * channels * sizeof(T));
        this->W_dW_biases = (T *)malloc(filters * sizeof(T));
        InitMatrix_He<T>(this->weights, filters * kernel_width * kernel_height * channels,1);
        InitMatrix_He<T>(this->biases, filters,1);
        this->batch_size = batch_size;
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        // Calculate output dimensions
        this->output_width = (width - kernel_width + 2 * padding) / stride + 1;
        this->output_height = (height - kernel_height + 2 * padding) / stride + 1;

        // Allocate memory for the output
        this->hidden_output = (T*)malloc(output_width * output_height * filters * batch_size);
        std::cout << "Output dimensions: " << output_width << "x" << output_height << "x" << filters << "x" << batch_size << std::endl;
        if (this->hidden_output == nullptr) {
            std::cerr << "Hidden output is null" << std::endl;
        } else {
            std::cout << "Hidden output allocated successfully" << std::endl;
        }
        cout<<"Conv2D constructor called"<<endl;
        this->name = "Conv2D";
    }
    virtual ~Matrix()
    {
        free(this->weights);
        free(this->biases);
        cout << "Matrix Destructor" << endl;
    }
    void randomize()
    {
        for (int i = 0; i < rows * cols; i++)
        {
            weights[i] = (T)rand() / RAND_MAX;
            if (i < rows)
            {
                biases[i] = (T)rand() / RAND_MAX;
            }
        }
    }
    virtual void set_labels(float *labels) {};
    virtual void set_Bernoulli(int row, int col) {};
    virtual void Fill_Bernoulli(){};
    virtual void Fill_Bernoulli_Ones(){};
    virtual void Fill_Activ() {};
    virtual void Fill_Loss_data() {};
    virtual void find_Loss_Metric_Jenks_Aggressive_Single() {};
    virtual void Agg_Jenks_Loss(){};
    int rows;
    int cols;
    int batch_size;
    int channels;
    int filters;
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int width;
    int height;
    int output_width;
    int output_height;
    int* num_ones;
    Loc_Layer<T> *loss_data;
    void matrix_multiply(T *A, T *B, T *C);
    void matrix_add(T *A, T *B, T *C);
    void matrix_subtract(T *A, T *B, T *C);
    void matrix_transpose(T *A, T *C);
    void matrix_scalar_multiply(T *A, T *C, T scalar);
    void matrix_scalar_add(T *A, T *C, T scalar);
    void matrix_scalar_subtract(T *A, T *C, T scalar);
    void matrix_scalar_divide(T *A, T *C, T scalar);
    void matrix_elementwise_multiply(T *A, T *B, T *C);
    void matrix_elementwise_divide(T *A, T *B, T *C);
    void matrix_elementwise_add(T *A, T *B, T *C);
    void matrix_elementwise_subtract(T *A, T *B, T *C);
    void matrix_sum(T *A, T *C, int axis);
    void set_rows(int rows);
    void set_cols(int cols);
    void virtual forward(T *input, T *output);
    void virtual backward(T *loss){};
    void backward(T *loss, int size) {};
    void backward(T *input, T *output, T *weight, T *bias, int input_size, int output_size) {};
    virtual void update_weights_SGD(T learning_rate) { cout << "Hello" << endl; };
    virtual void update_weights_AdamJenks(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_SGDJenks(T learning_rate) {};
    virtual void update_weights_Momentum(T learning_rate, T momentum) {};
    virtual void update_weights_RMSProp(T learning_rate, T decay_rate) {};
    virtual void update_weights_Adam(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_AdamActiv(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_AdamWJenks(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_AdamDecay(T learning_rate, T beta1, T beta2, T epsilon, int epochs) {};
    virtual void update_weights_SGDMomentum_Jenks(T learning_rate, T momentum) {};
    virtual void find_Loss_Metric() {};
    virtual void find_Loss_Metric_Jenks() {};   
    virtual void find_Loss_Metric_Jenks_Aggressive() {};    
    virtual void find_Loss_Metric_Jenks_Prune() {};
    void train(T *input, T *output, int epochs, T learning_rate) {};
    int get_rows();
    int get_cols();

private:
    cudaError_t cudaStatus;
};

template <typename T>
class Optimizer
{
public:
    Optimizer(T learning_rate, T momentum, T decay_rate, T beta1, T beta2, T epsilon)
    {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        this->decay_rate = decay_rate;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->name = "Optimizer";
    }
    Optimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {};
    Optimizer(T learning_rate, T decay_rate, T epsilon) : Optimizer(learning_rate, 0.0, decay_rate, 0.0, 0.0, epsilon) {};
    Optimizer(T learning_rate) : Optimizer(learning_rate, 0.0, 0.0, 0.0, 0.0, 0.0) {};
    T learning_rate;
    T momentum;
    T decay_rate;
    T beta1;
    T beta2;
    T epsilon;
    string name;
    virtual void update_weights(T *weights, T *d_weights, T *biases, T *d_biases, int rows, int cols){};
};

template <typename T>
class AdamOptimizer : public Optimizer<T>
{
public:
    AdamOptimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "Adam";};
};

template <typename T>
class AdamJenksOptimizer : public Optimizer<T>
{
    public:
    AdamJenksOptimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "AdamJenks";};
};

template <typename T>
class AdamJenksDecayOptimizer : public Optimizer<T>
{
    public:
    AdamJenksDecayOptimizer(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "AdamDecay";};
};

template <typename T>
class SGDMomentumJenksOptimizer : public Optimizer<T>
{
    public:
    SGDMomentumJenksOptimizer(T learning_rate, T momentum) : Optimizer<T>(learning_rate, momentum, 0.0, 0.0, 0.0, 0.0) {this->name = "SGDMomentumJenks";};
};

template <typename T>
class SGDJenksOptimizer : public Optimizer<T>
{
    public:
    SGDJenksOptimizer(T learning_rate) : Optimizer<T>(learning_rate, 0.0, 0.0, 0.0, 0.0, 0.0) {this->name = "SGDJenks";};
};

template <typename T>
class SGD_Optimizer : public Optimizer<T>
{
public:
    SGD_Optimizer(T learning_rate) : Optimizer<T>(learning_rate, 0.0, 0.0, 0.0, 0.0, 0.0) {this->name = "SGD";};
};

template <typename T>
class MomentumOptimizer : public Optimizer<T>
{
public:
    MomentumOptimizer(T learning_rate, T momentum) : Optimizer<T>(learning_rate, momentum, 0.0, 0.0, 0.0, 0.0) {this->name = "Momentum";};
};

template <typename T>
class RMSPropOptimizer : public Optimizer<T>
{
public:
    RMSPropOptimizer(T learning_rate, T decay_rate, T epsilon) : Optimizer<T>(learning_rate, 0.0, decay_rate, 0.0, 0.0, epsilon) {this -> name = "RMSProp";};
};

template <typename T>
class AdamWBernoulli: public Optimizer<T>
{
public:
    AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon) : Optimizer<T>(learning_rate, 0.0, 0.0, beta1, beta2, epsilon) {this->name = "AdamWBernoulli";};
};


template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

template <typename T>
__global__ void vector_elementwise_multiply_kernel(T *A, T *B, T *C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] * B[index];
    }
}

template <typename T>
__global__ void matrix_elementwise_multiply_kernel(T *A, int *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

template <typename T>
__global__ void vector_elementwise_multiply_kernel(T *A, int *B, T *C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] * B[index];
    }
}

template <typename T>
__global__ void matrix_multiply(T *A, T *B, T *C, int rows, int cols, int inter_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        T sum = 0;
        for (int i = 0; i < inter_size; i++)
        {
            sum += A[row * cols + i] * B[i * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k] * B[k];
        }
        C[row] = sum;
    }
}

template <typename T>
__global__ void matrix_matrix_addition_kernel(T* A, T* B, T* C, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols){
        C[row*cols + col] = A[row*cols + col] + B[row*cols + col];
    }
}

template <typename T>
__global__ void matrix_accum_kernel(T* A, T* B, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols){
        A[row*cols + col] += B[row*cols + col];
    }
}

template <typename T>
__global__ void vector_accum_kernel(T* A, T* B, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size){
        A[index] += B[index];
    }
}

template <typename T>
__global__ void matrix3D_vector_multiply_kernel(T *A, T *B, T *C, int rows, int cols, int depth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    //The 3D matrix is composed as such: The first row*col entries correspond to the first batch, the second row*col entries correspond to the second batch, etc.
    //The entries of B are such that each column corresponds to a batch
    if (row < rows && batch < depth)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k + batch * rows * cols] * B[k*depth + batch];
        }
        C[row*depth+batch] = sum;
    }
}

template <typename T>
__global__ void matrix_vector_addition_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        C[row] = A[row] + B[row];
    }
}

template <typename T>
void Matrix<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    cout << "Matrix forward" << endl;
    memcpy(this->input, input, cols * sizeof(T));
    // this->input = input;
    T *d_input, *d_output;
    T *d_weights;
    T *d_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for device_data" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Matrix" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying d_data from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions

    int blocksize = 16;
    dim3 blockDim(blocksize, blocksize, 1);

    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the matrix multiplication kernel
    matrix_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_weights, d_input, d_output, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    matrix_vector_addition_kernel<T><<<gridDim, blockDim>>>(d_output, d_biases, d_output, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing device_data" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    memcpy(output, this->hidden_output, rows * sizeof(T));
}

template <typename T>
class Sigmoid : public Matrix<T>
{
public:
    Sigmoid(int rows) : Matrix<T>(rows)
    {
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "Sigmoid";
    }
    Sigmoid(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Sigmoid";
    }
    Sigmoid(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * batch_size * cols * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * cols * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * cols * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * cols * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows * batch_size * cols * batch_size);
        ZeroVector<T>(this->hidden_output, rows * batch_size * cols * batch_size);
        this->name = "Sigmoid";
    }
    ~Sigmoid()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
class Tanh : public Matrix<T>
{
public:
    Tanh(int rows) : Matrix<T>(rows)
    {
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "Tanh";
    }
    Tanh(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Tanh";
    }
    Tanh(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*cols*channels*batch_size);
        ZeroVector<T>(this->hidden_output, rows*cols*channels*batch_size);
        this->name = "Tanh";
    }
    ~Tanh()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
class SoftPlus : public Matrix<T>
{
    public:
    SoftPlus(int rows) : Matrix<T>(rows)
    {
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "SoftPlus";
    }
    SoftPlus(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "SoftPlus";
    }
    SoftPlus(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows * cols * channels* batch_size);
        ZeroVector<T>(this->hidden_output, rows * cols * channels* batch_size);
        this->name = "SoftPlus";
    }
    ~SoftPlus()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
class RELU_layer : public Matrix<T>
{
public:
    RELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "RELU";
    }
    RELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Tanh";
        this->size = rows;
    }
    RELU_layer(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector(this->input, rows * cols * channels* batch_size);
        ZeroVector(this->hidden_output, rows * cols * channels* batch_size);
        ZeroVector(this->loss, rows * cols * channels* batch_size);
        ZeroVector(this->next_loss, rows * cols * channels* batch_size);
        this->name = "RELU";
        this->batch_size = batch_size;
        this->size = rows * cols * channels;
    }
    ~RELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    int size;
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};


template <typename T>
__global__ void LeakyRELU_kernel(T *input, T *output, T alpha, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : alpha * input[index*batch_size + batch];
    }
}

template <typename T>
__global__ void ELU_kernel(T *input, T *output, T alpha, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : alpha * (exp(input[index*batch_size + batch])-1);
    }
}

template <typename T>
__global__ void SLU_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    //SLU(x) = max(0,min(1,x))
    if (index < size)
    {
        output[index*batch_size + batch] = max(0,min(1,input[index*batch_size + batch])); 
    }
}

template <typename T>
__global__ void Tanh_kernel(T* input, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = tanh(input[index*batch_size+batch]);
    }
}

template <typename T>
__global__ void LeakyRELU_derivative_kernel(T *input, T* loss, T *output, T alpha, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size + batch] : -alpha*loss[index*batch_size + batch];
    }
}

template <typename T>
__global__ void ELU_derivative_kernel(T *input, T* loss, T *output, T alpha, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size + batch] : (input[index*batch_size+batch] + alpha)*loss[index*batch_size + batch];
    }
}


template <typename T>
__global__ void SLU_derivative_kernel(T* input, T* loss, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = input[index*batch_size+batch] > 0  && input[index*batch_size+batch] < 1 ? loss[index*batch_size+batch] : 0;
    }
}

template <typename T> 
__global__ void Tanh_derivative_kernel(T* input, T* loss, T* output, int size, int batch_size){
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size+batch] = (1 - input[index*batch_size+batch]*input[index*batch_size+batch])*loss[index*batch_size+batch];
    }
}


template <typename T>
class LeakyRELU_layer : public Matrix<T>
{
public:
    LeakyRELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "LeakyRELU";
    }
    LeakyRELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"LeakyRELU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        this->name = "LeakyRELU";
    }
    T alpha = 0.01;
    LeakyRELU_layer(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector(this->input, rows * cols * channels* batch_size);
        ZeroVector(this->hidden_output, rows * cols * channels* batch_size);
        this->name = "LeakyRELU";
    }
    ~LeakyRELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
class ELU_layer : public Matrix<T>
{
public:
    ELU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "ELU";
    }
    ELU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"ELU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        this->name = "ELU";
    }
    ELU_layer(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector(this->input, rows * cols * channels* batch_size);
        ZeroVector(this->hidden_output, rows * cols * channels* batch_size);
        this->name = "ELU";
    }
    T alpha = 0.01;
    ~ELU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
class SLU_layer : public Matrix<T>
{
public:
    SLU_layer(int rows) : Matrix<T>(rows)
    {
        ZeroVector(this->input, rows);
        ZeroVector(this->hidden_output, rows);
        this->name = "ELU";
    }
    SLU_layer(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"SLU layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector(this->input, rows*batch_size);
        ZeroVector(this->hidden_output, rows*batch_size);
        this->name = "ELU";
    }
    T alpha = 0.01;
    SLU_layer(int rows, int cols, int channels, int batch_size) : Matrix<T>(cols, rows, channels*batch_size)
    {
        this->channels = channels;
        this->hidden_output = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->input = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * cols * channels* batch_size * sizeof(T));
        ZeroVector(this->input, rows * cols * channels* batch_size);
        ZeroVector(this->hidden_output, rows * cols * channels* batch_size);
        this->name = "ELU";
    }
    ~SLU_layer()
    {
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;


};


template <typename T>
void SLU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    SLU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of LeakyRELU is NULL" << endl;
        exit(1);
    }   
    memcpy(output, this->hidden_output, size * sizeof(T));

}


template <typename T>
void SLU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    SLU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_out)))
    {
        cout << "Error in freeing d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_temp_loss)))
    {
        cout << "Error in freeing d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }

}



template <typename T>
void ELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    ELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, this->alpha, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of LeakyRELU is NULL" << endl;
        exit(1);
    }   
    memcpy(output, this->hidden_output, size * sizeof(T));

}


template <typename T>
void ELU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    ELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, this-> alpha, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_out)))
    {
        cout << "Error in freeing d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_temp_loss)))
    {
        cout << "Error in freeing d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }

}


template <typename T>
void Tanh<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    Tanh_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of LeakyRELU is NULL" << endl;
        exit(1);
    }   
    memcpy(output, this->hidden_output, size * sizeof(T));

}


template <typename T>
void Tanh<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    Tanh_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss, d_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_out)))
    {
        cout << "Error in freeing d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_temp_loss)))
    {
        cout << "Error in freeing d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }

}


template <typename T>
void LeakyRELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int size = this->rows;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int TPB = 16;
    dim3 blockDim(TPB, TPB, 1);
    dim3 gridDim((batch_size + TPB -1 ) / TPB, (size + TPB - 1) / TPB, 1);
    // Launch the LeakyRELU kernel
    LeakyRELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, this->alpha, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of LeakyRELU is NULL" << endl;
        exit(1);
    }   
    memcpy(output, this->hidden_output, size * sizeof(T));

}


template <typename T>
void LeakyRELU_layer<T>::backward(T* loss){
    T *d_loss;
    T *d_temp_loss;
    T *d_out;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (loss == NULL)
    {
        cout << "Loss of LeakyRELU is NULL" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_temp_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device, LeakyRELU loss" << endl;
        exit(1);
    }
    // Define grid and block dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);
    // Launch the LeakyRELU derivative kernel
    LeakyRELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_out, d_temp_loss,d_loss,this->alpha, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    if(this->next_loss == NULL) {
        cout<<"Next loss is NULL"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_out)))
    {
        cout << "Error in freeing d_out" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_temp_loss)))
    {
        cout << "Error in freeing d_temp_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }

}


template <typename T>
__global__ void Col_Wise_Reduce(T* input, T* red, T* max, int cols, int batch_size){
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch<batch_size){
        for(int i = 0; i<cols; i++){
            input[i*batch_size + batch] = input[i*batch_size + batch] - max[batch];
            input[i*batch_size + batch] = exp(input[i*batch_size + batch]);
        }
    }
    __syncthreads();
    if(batch < batch_size){
        T sum = 0;
        for(int i = 0; i < cols; i++){
            sum += input[i*batch_size + batch];
        }
        red[batch] = sum;
    }
    __syncthreads();
    if(batch < batch_size){
        for(int i = 0; i < cols; i++){
            input[i*batch_size + batch] = input[i*batch_size + batch] / red[batch];
        }
    }
}

template <typename T>
__global__ void softmax_kernel(T *input, T *output, T* reduce, int size, int batch_size)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] / reduce[batch];
        // printf("Output[%d][%d] = %f\n", index, batch, output[index*batch_size + batch]);
    }


}

template <typename T>
__global__ void softmax_derivative_kernel(T *input, T *output, int size, int batch_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    if (row < size && col < size && batch < batch_size)
    {
        if(row == col) {
            output[row * size + col + batch*size*size] = input[row*batch_size+batch] * (1 - input[col*batch_size+batch]);
            // printf("Output[%d][%d][%d] = %f\n", row, col,batch, output[row * size + col + batch*size*size]);
        } else {
            output[row * size + col + batch*size*size] = -input[row*batch_size+batch] * input[col*batch_size+batch];
            // printf("Output[%d][%d][%d] = %f\n", row, col,batch, output[row * size + col + batch*size*size]);
        }  

    }
}

template <typename T>
__global__ void Print_Loss_Data(Loc_Layer<T>* lof, int rows, int cols){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col <= cols && row < rows){
        printf("Loss_Data[%d][%d] = %f\n", lof[row*cols+col].row, lof[row*cols+col].col, lof[row*cols+col].weights_dW);
    }
}

template <typename T>
__global__ void Fill_Jenks_Device(Loc_Layer<T>* loc, T* d_Wdw, T* d_Bbw, int cols, int rows){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < cols && row < rows){
        loc[row*cols+col].row = row;
        loc[row*cols+col].col = col;
        loc[row*cols+col].weights_dW = fabsf(d_Wdw[row*cols+col]);
        printf("loc[%d][%d] = %f\n", loc[row*cols+col].row, loc[row*cols+col].col, loc[row*cols+col].weights_dW);
    }   
    else if(col == cols && row < rows){
        loc[row*cols+col].row = row;
        loc[row*cols+col].col = col;
        loc[row*cols+col].weights_dW= fabsf(d_Bbw[row]);
        printf("Bias loc[%d][%d] = %f\n", loc[row*cols+col].row, loc[row*cols+col].col, loc[row*cols+col].weights_dW);
    }
}

template <typename T>
__global__ void Fill_Jenks_Device_Prune(Loc_Layer<T>* loc, T* d_Wdw_agg, int cols, int rows){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col <= cols && row < rows){
        loc[row*cols+col].row = row;
        loc[row*cols+col].col = col;
        loc[row*cols+col].weights_dW = d_Wdw_agg[row*cols+col];
    }   
}

template <typename T>
__global__ void Fill_Agg_Device(T* Agg, T* d_Wdw, T* d_Bbw, int cols, int rows){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < cols && row < rows){
        Agg[row*cols+col] = d_Wdw[row*cols+col];
    }   
    else if(col == cols && row < rows){
        Agg[row*cols+col] = d_Bbw[row];
    }
}

template <typename T>
__global__ void Fill_WB_device(T* d_WB, Loc_Layer<T>* d_Loss_Data, int rows, int cols){

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col <= cols && row < rows){
        int temp_row = d_Loss_Data[row*cols+col].row;
        int temp_col = d_Loss_Data[row*cols+col].col;
        d_WB[temp_row*cols+temp_col] = fabsf(d_Loss_Data[row*cols+col].weights_dW);
        printf("d_WB[%d][%d] = %f\n", temp_row, temp_col, d_WB[row*cols+col]);
    }
}

template <typename T>
__global__ void Fill_WBW_device_array(T* d_WB, int* indices, T* d_wDw, int rows, int cols){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < cols && row < rows){
        d_WB[row*(cols)+col] = fabsf(d_wDw[row*cols+col]);
        indices[row*(cols)+col] = row*(cols)+col;
    }
}

template <typename T>
__global__ void Fill_WBB_device_array(T* d_WB, int* indices, T* d_Bbw, int rows){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < rows){
        d_WB[row] = fabsf(d_Bbw[row]);
        indices[row] = row;
    }
}


template <typename T>
__global__ void Jenks_Optimization(T* d_WB, T* d_var, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows*(cols+1))){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_WB[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows*(cols+1); i++){
            mean_2 += d_WB[i];
        }
        mean_2 = mean_2/(rows*(cols+1)-idx);
    }
    __syncthreads();
    if(idx<rows*(cols+1)){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_WB[i] - mean_1)*(d_WB[i] - mean_1);
        }
        for(int i = idx; i<rows*(cols+1); i++){
            var += (d_WB[i] - mean_2)*(d_WB[i] - mean_2);
        }
        d_var[idx] = var;
    }
}


template <typename T>
__global__ void Jenks_Optimization_Weights(T* d_WB, T* d_var, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows*(cols))){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_WB[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows*(cols); i++){
            mean_2 += d_WB[i];
        }
        mean_2 = mean_2/(rows*(cols)-idx);
    }
    __syncthreads();
    if(idx<rows*(cols)){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_WB[i] - mean_1)*(d_WB[i] - mean_1);
        }
        for(int i = idx; i<rows*(cols); i++){
            var += (d_WB[i] - mean_2)*(d_WB[i] - mean_2);
        }
        d_var[idx] = var;
    }
}

template <typename T>
__global__ void Jenks_Optimization_Biases(T* d_B, T* d_var, int rows){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows)){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_B[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows; i++){
            mean_2 += d_B[i];
        }
        mean_2 = mean_2/(rows-idx);
    }
    __syncthreads();
    if(idx<rows){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_B[i] - mean_1)*(d_B[i] - mean_1);
        }
        for(int i = idx; i<rows; i++){
            var += (d_B[i] - mean_2)*(d_B[i] - mean_2);
        }
        d_var[idx] = var;
    }
}

template <typename T>
__global__ void Jenks_Optimization_CutOff(T* d_WB, T* d_var, int rows, int cols, int index){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows*(cols+1)-index)){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = index; i<(idx+index); i++){
            mean_1 += d_WB[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = (idx+index); i<rows*(cols+1); i++){
            mean_2 += d_WB[i];
        }
        mean_2 = mean_2/(rows*(cols+1)-idx-index);
    }
    __syncthreads();
    if(idx<rows*(cols+1)-index){
        T var = 0;
        for(int i = index; i<(idx+index); i++){
            var += (d_WB[i] - mean_1)*(d_WB[i] - mean_1);
        }
        for(int i = (idx+index); i<rows*(cols+1); i++){
            var += (d_WB[i] - mean_2)*(d_WB[i] - mean_2);
        }
        d_var[idx] = var;
    }
}

template <typename T>
class Softmax : public Matrix<T>
{
public:
    //Need to edit for stability by doing e^(x-max(x))
    Softmax()
    {
        this->rows = 0;
        this->cols = 0;
    }
    Softmax(int rows) : Matrix<T>(rows)
    {
        this->rows = rows;
        this->cols = 1;
        this->input = (T *)malloc(rows * sizeof(T));
        ZeroVector<T>(this->input, rows);
        ZeroVector<T>(this->hidden_output, rows);
        this->name = "softmax";
    }
    Softmax(int rows, int batch_size) : Matrix<T>(1, rows, batch_size)
    {
        cout<<"Softmax layer constructor"<<endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(rows * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(rows * batch_size * sizeof(T));
        ZeroVector<T>(this->input, rows*batch_size);
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        this->name = "Softmax";
    }
    ~Softmax()
    {
        free(this->input);
        free(this->hidden_output);
    }
    // T* input;
    // T* hidden_output;
    // T* loss;
    // T* next_loss;
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T* max = new T [this->batch_size];
        //Find the max of each column in input;
        for(int i = 0; i<this->batch_size; i++) {
            max[i] = input[i];
            for(int j = 1; j<this->rows; j++) {
                if(input[j*this->batch_size+i] > max[i]) {
                    max[i] = input[j*this->batch_size+i];
                }
            }
        }
        int size = this->rows;
        int batch_size = this->batch_size;
        T *d_input, *d_output;
        if (input == NULL)
        {
            cout << "Input Softmax is NULL" << endl;
            input = (T *)malloc(size * batch_size *sizeof(T));
            if (input == NULL)
            {
                cout << "Input of Softmax is NULL" << endl;
                exit(1);
            }
        } else {
            for(int i = 0; i<size*batch_size; i++) {
                this->input[i] = input[i];
                // cout<<"Input["<<i<<"]"<<input[i]<<endl;
            }
        }
        T* d_max;
        T* d_reduce;
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_max, batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_max"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_reduce, batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_reduce"<<endl;
            exit(1);
        }
        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_max, max, batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout<<"Error in copying max from host to device"<<endl;
            exit(1);
        }
        thrust::fill(thrust::device, d_output, d_output + size*batch_size, (T)0);

        // Define grid and block dimensions
        // Launch the softmax kernel
        // Corrected transformation for applying exp

        int threadsPerBlock = 16;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        int batchBlocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchBlocksPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        int TPB = 256;
        int blocks = (batch_size + TPB - 1) / TPB;
        Col_Wise_Reduce<<<blocks, TPB>>>(d_input, d_reduce, d_max, size, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // softmax_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_reduce, size, batch_size);
        // if (!HandleCUDAError(cudaDeviceSynchronize()))
        // {
        //     cout << "Error in synchronizing device" << endl;
        //     exit(1);
        // }
        // Copy the result output from device to host
        if (!HandleCUDAError(cudaMemcpy(output, d_input, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying output from device to host" << endl;
        }
        // Free device memory
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        if (output == NULL)
        {
            cout << "Output of Softmax is NULL" << endl;
            exit(1);
        }
        memcpy(output, this->hidden_output, size * sizeof(T));
    }
    void backward(T *loss)
    {
        T *d_loss;
        T *d_temp_loss;
        T *d_out;
        T *d_fin_loss;
        int rows = this->rows;
        int batch_size = this->batch_size;
        if (loss == NULL)
        {
            cout << "Loss of Softmax is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_temp_loss, rows * rows * batch_size * sizeof(T))))
        {
            //May need to make a 3D matrix for this
            cout << "Error in allocating memory for d_temp_loss" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, rows * batch_size * sizeof(T))))
        {
            cout<<"Error in allocating memory for d_fin_loss"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_out, this->hidden_output, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Softmax loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying loss from host to device, Softmax loss" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchBlocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchBlocksPerGrid, blocksPerGrid , 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        int twodthreadsPerBlock = 8;
        dim3 twodblockDim(twodthreadsPerBlock, twodthreadsPerBlock, twodthreadsPerBlock);

        dim3 twodgridDim((rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, (rows + twodthreadsPerBlock - 1) / twodthreadsPerBlock, (batch_size + twodthreadsPerBlock - 1) / twodthreadsPerBlock);

        // Launch the softmax derivative kernel
        softmax_derivative_kernel<T><<<twodgridDim, twodblockDim>>>(d_out, d_temp_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        matrix3D_vector_multiply_kernel<T><<<gridDim, blockDim>>>(d_temp_loss, d_loss, d_fin_loss, rows, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        if (this->next_loss == NULL)
        {
            cout << "Next Loss of Softmax is NULL" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_fin_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host Softmax" << endl;
            // exit(1);
        }
        // cout<<"Softmax Next Loss"<<endl;
        // for(int i = 0; i<rows; i++) {
        //     cout<<"Next Loss["<<i<<"]"<<this->next_loss[i]<<endl;
        // }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_temp_loss)))
        {
            cout << "Error in freeing d_temp_loss" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_fin_loss)))
        {
            cout<<"Error in freeing d_fin_loss"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
__global__ void update_weights_kernel(T *weights, T *biases, T *d_weights, T *d_biases, T learning_rate, int input_size, int output_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < input_size)
    {
        weights[row * input_size + col] -= learning_rate * d_weights[row * input_size + col];
    }
}


template <typename T>
__global__ void update_bias_kernel(T *biases, T *d_biases, T learning_rate, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        biases[index] -= learning_rate * d_biases[index];
    }
}

template <typename T>
__global__ void update_weights_kernel_Jenks(T *weights, T *d_weights, T* B_W, T learning_rate, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows && col < cols){
        weights[row*cols+col] -= learning_rate * d_weights[row*cols+col] * B_W[row*cols+col];
    }
}

template <typename T>
__global__ void update_bias_kernel_Jenks(T *biases, T *d_biases, T* B_B, T learning_rate, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        biases[index] -= learning_rate * d_biases[index] * B_B[index];
    }
}

template <typename T>
__global__ void AdamDecay_update_weights_kernel_Jenks(T *weights, T *d_weights,T* m_weights, T* v_weights, int* B_W, T beta1, T beta2, T epsilon, T learning_rate, int cols, int rows, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows && col < cols){
        m_weights[row*cols+col] = beta1 * m_weights[row*cols+col] + (1 - beta1) * d_weights[row*cols+col];
        v_weights[row*cols+col] = beta2 * v_weights[row*cols+col] + (1 - beta2) * d_weights[row*cols+col] * d_weights[row*cols+col];
        T m_hat = m_weights[row*cols+col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row*cols+col] / (1 - pow(beta2, epochs+1));
        if(B_W[row*cols+col]){
            weights[row*cols+col] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon) * B_W[row*cols+col];
        }
        else{
            //Decay the weights
            weights[row*cols+col] = weights[row*cols+col] * beta1;
        }

    }
}

template <typename T>
__global__ void SGDMomentum_Decay_update_weights_kernel_Jenks(T *weights, T *d_weights,T* m_weights, int* B_W, T beta1, T epsilon, T learning_rate, int cols, int rows)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows && col < cols){
        m_weights[row*cols+col] = beta1 * m_weights[row*cols+col] + epsilon * weights[row*cols+col] + B_W[row*cols+col] * d_weights[row*cols+col];
        weights[row*cols+col] -= learning_rate * m_weights[row*cols+col];
    }
}

template <typename T>
__global__ void SGDMomentum_Decay_update_bias_kernel_Jenks(T *biases, T *d_biases,T* m_biases, int* B_B, T beta1, T epsilon, T learning_rate, int rows)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < rows){
        m_biases[index] = beta1 * m_biases[index] + epsilon * biases[index] + B_B[index] * d_biases[index];
        m_biases[index] -= learning_rate * m_biases[index];
    }
}

template <typename T>
__global__ void AdamDecay_update_bias_kernel_Jenks(T *biases, T *d_biases,T* m_biases, T* v_biases, int* B_B, T beta1, T beta2, T epsilon, T learning_rate, int rows, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < rows){
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * d_biases[index];
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * d_biases[index] * d_biases[index];
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        if(B_B[index]){
            biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon) * B_B[index];
        }
        else{
            //Decay the biases
            biases[index] = biases[index] * beta1;
        }
    }
}

template <typename T>
__global__ void Adam_Update_Weights(T *weights, T *d_weights, T *m_weights, T *v_weights, T beta1, T beta2, T epsilon, T learning_rate, int input_size, int output_size, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < input_size)
    {
        m_weights[row * input_size + col] = beta1 * m_weights[row * input_size + col] + (1 - beta1) * d_weights[row * input_size + col];
        v_weights[row * input_size + col] = beta2 * v_weights[row * input_size + col] + (1 - beta2) * d_weights[row * input_size + col] * d_weights[row * input_size + col];
        T m_hat = m_weights[row * input_size + col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row * input_size + col] / (1 - pow(beta2, epochs+1));
        weights[row * input_size + col] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        // if(weights[row * input_size + col] != weights[row * input_size + col]) {
        //     printf("NAN detected in weights\n");
        //     printf("m_hat: %f\n", m_hat);
        //     printf("v_hat: %f\n", v_hat);
        //     printf("weights: %f\n", weights[row * input_size + col]);
        //     printf("d_weights: %f\n", d_weights[row * input_size + col]);
        //     printf("m_weights: %f\n", m_weights[row * input_size + col]);
        //     printf("v_weights: %f\n", v_weights[row * input_size + col]);
        // }
    }
}

template <typename T>
__global__ void Adam_Update_Weights_Bernoulli(T *weights, T *d_weights, T *m_weights, T *v_weights, int *B_weights, T beta1, T beta2, T epsilon, T learning_rate, int input_size, int output_size, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < output_size && col < input_size)
    {
        T temp = B_weights[row * input_size + col] * d_weights[row * input_size + col];
        // printf("B_weights[%d][%d] = %d\n", row, col, B_weights[row * input_size + col]);
        m_weights[row * input_size + col] = beta1 * m_weights[row * input_size + col] + (1 - beta1) * temp;
        v_weights[row * input_size + col] = beta2 * v_weights[row * input_size + col] + (1 - beta2) * temp * temp;
        T m_hat = m_weights[row * input_size + col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row * input_size + col] / (1 - pow(beta2, epochs+1));
        weights[row * input_size + col] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}


template <typename T>
__global__ void Adam_Update_Weights_Bernoulli_Zero(T *weights, T *d_weights, T *m_weights, T *v_weights, int *B_weights, T beta1, T beta2, T epsilon, T learning_rate, int input_size, int output_size, int epochs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < output_size && col < input_size)
    {
        T temp = B_weights[row * input_size + col] * d_weights[row * input_size + col];
        // printf("B_weights[%d][%d] = %d\n", row, col, B_weights[row * input_size + col]);
        m_weights[row * input_size + col] = beta1 * m_weights[row * input_size + col] + (1 - beta1) * temp;
        v_weights[row * input_size + col] = beta2 * v_weights[row * input_size + col] + (1 - beta2) * temp * temp;
        T m_hat = m_weights[row * input_size + col] / (1 - pow(beta1, epochs+1));
        T v_hat = v_weights[row * input_size + col] / (1 - pow(beta2, epochs+1));
        weights[row * input_size + col] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        weights[row * input_size + col] = weights[row * input_size + col] * B_weights[row * input_size + col];
    }
}

template <typename T>
__global__ void Adam_Update_Bias(T *biases, T *d_biases, T *m_biases, T *v_biases, T beta1, T beta2, T epsilon, T learning_rate, int size, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * d_biases[index];
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * d_biases[index] * d_biases[index];
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

template <typename T>
__global__ void Adam_Update_Bias_Bernoulli(T *biases, T *d_biases, T *m_biases, T *v_biases, int *B_biases, T beta1, T beta2, T epsilon, T learning_rate, int size, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        T temp = B_biases[index] * d_biases[index];
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * temp;
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * temp * temp;
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

template <typename T>
__global__ void Adam_Update_Bias_Bernoulli_Zero(T *biases, T *d_biases, T *m_biases, T *v_biases, int *B_biases, T beta1, T beta2, T epsilon, T learning_rate, int size, int epochs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        T temp = B_biases[index] * d_biases[index];
        m_biases[index] = beta1 * m_biases[index] + (1 - beta1) * temp;
        v_biases[index] = beta2 * v_biases[index] + (1 - beta2) * temp * temp;
        T m_hat = m_biases[index] / (1 - pow(beta1, epochs+1));
        T v_hat = v_biases[index] / (1 - pow(beta2, epochs+1));
        biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        biases[index] = biases[index] * B_biases[index];
    }
}

int* return_col_row(int index, int cols){
    int* temp = (int*)malloc(2*sizeof(int));
    temp[0] = index/cols; //row
    temp[1] = index%cols; //col
    return temp;        
}


template <typename T>
class Linear : public Matrix<T>
{
public:
    Linear(int cols, int rows) : Matrix<T>(cols, rows)
    {
        InitMatrix_He<T>(this->weights, rows, cols);
        Init_Bias_He<T>(this->biases, rows,cols);
        ZeroVector<T>(this->hidden_output, rows);
        ZeroVector<T>(this->input, cols);
        v_weights = (T*)malloc(rows * cols * sizeof(T));
        v_biases = (T*)malloc(rows * sizeof(T));
        m_weights = (T*)malloc(rows * cols * sizeof(T));
        m_biases = (T*)malloc(rows * sizeof(T));
        ZeroMatrix<T>(v_weights, rows, cols);
        ZeroVector<T>(v_biases, rows);
        ZeroMatrix<T>(m_weights, rows, cols);
        ZeroVector<T>(m_biases, rows);
        this->name = "linear";
    }
    Linear(int cols, int rows,int batch_size) : Matrix<T>(cols, rows,batch_size)
    {
        cout << "Linear Constructor" << endl;
        this->hidden_output = (T *)malloc(rows * batch_size * sizeof(T));
        this->input = (T *)malloc(cols * batch_size * sizeof(T));
        this->loss = (T *)malloc(rows * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(cols * batch_size * sizeof(T));
        this->loss_data = (Loc_Layer<T>*)malloc(rows * (cols + 1) * sizeof(Loc_Layer<T>));
        ZeroVector<T>(this->hidden_output, rows*batch_size);
        ZeroVector<T>(this->input, cols*batch_size);
        v_weights = (T*)malloc(rows * cols * sizeof(T));
        v_biases = (T*)malloc(rows * sizeof(T));
        m_weights = (T*)malloc(rows * cols * sizeof(T));
        m_biases = (T*)malloc(rows * sizeof(T));
        this->num_ones = (int*)malloc(rows * (cols+1) * sizeof(int));
        WB_agg = (T*)malloc(rows * (cols) * sizeof(T));
        BB_agg = (T*)malloc(rows * sizeof(T));
        ZeroMatrix<T>(v_weights, rows, cols);
        ZeroVector<T>(v_biases, rows);
        ZeroMatrix<T>(m_weights, rows, cols);
        ZeroVector<T>(m_biases, rows);
        ZeroMatrix<T>(WB_agg, rows, cols);
        ZeroVector<T>(BB_agg, rows);
        this->name = "linear";
    }
    ~Linear() override
    {
        free(this->weights);
        free(this->biases);
        free(this->d_weights);
        free(this->d_biases);
        free(this->hidden_output);
        free(this->input);
        cout << "Linear Destructor" << endl;
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
    void set_weights(T *weights, T *biases)
    {
        this->weights = weights;
        this->biases = biases;
    }
    T* v_weights;
    T* v_biases;
    T* m_weights;
    T* m_biases;
    T* WB_agg;
    T* BB_agg;
    Loc_Layer<T>* loss_data;
    void Fill_Loss_data() override{
        for(int i = 0; i< this->rows*(this->cols+1); i++) {
            this->num_ones[i] = 0;   
        }
    }
    void Fill_Bernoulli_Ones() override{
        // All weights initially set to 1
        for(int i = 0; i<this->rows * this->cols; i++) {
            this->B_weights[i] = 1;
        }
        for(int i = 0; i<this->rows; i++) {
            this->B_biases[i] = 1;
        }
    }
    void Fill_Bernoulli() override{
        // All weights initially set to 1
        for(int i = 0; i<this->rows * this->cols; i++) {
            this->B_weights[i] = 0;
        }
        for(int i = 0; i<this->rows; i++) {
            this->B_biases[i] = 0;
        }
    }
    void Fill_Activ() override{
        // Only fill with 0's and 1's at random
        for(int i = 0; i<this->rows * this->cols; i++) {
            this->B_weights[i] = 1;
        }
        for(int i = 0; i<this->rows; i++) {
            this->B_biases[i] = 1;
        }
    }
    void set_Bernoulli(int row, int col) override{
        this->B_weights[row*(this->cols) + col] = 1;
    }
    void update_weights_SGDMomentum_Jenks(T learning_rate, T beta1) override {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_m_weights, *d_m_biases;
        int* d_B_weights, *d_B_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_biases" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(int))))
        {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(int))))
        {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" <<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" <<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" <<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" <<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout<<"Error in copying B_weights from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout<<"Error in copying B_biases from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);
        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel

        SGDMomentum_Decay_update_weights_kernel_Jenks<T><<<gridDim2D, blockDim2D>>>(d_weights, d_d_weights, d_m_weights, d_B_weights, beta1, 0.0001, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        SGDMomentum_Decay_update_bias_kernel_Jenks<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, d_m_biases, d_B_biases, beta1, 0.0001, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_weights))) {
            cout<<"Error in freeing d_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_biases))) {
            cout<<"Error in freeing d_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_d_weights))) {
            cout<<"Error in freeing d_d_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_d_biases))) {
            cout<<"Error in freeing d_d_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_m_weights))) {
            cout<<"Error in freeing d_m_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_m_biases))) {
            cout<<"Error in freeing d_m_biases"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void update_weights_RMSProp(T learning_rate, T decay_rate) override {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel

        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Update the weights and biases
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_v_weights, d_v_biases, d_d_weights, d_d_biases, decay_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_v_biases, d_d_biases, decay_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_Momentum(T learning_rate, T momentum) override {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);
        
        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Update the weights and biases
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_v_weights, d_v_biases, d_d_weights, d_d_biases, momentum, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_v_biases, d_d_biases, momentum, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        // Copy the result weights and biases from device to host
        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }   
        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_Adam(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }

        Adam_Update_Weights<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Adam_Update_Bias<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        
    }
    void update_weights_AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        int *d_B_weights, *d_B_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_m_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        Adam_Update_Weights_Bernoulli<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, d_B_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Adam_Update_Bias_Bernoulli<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, d_B_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_AdamActiv(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        T *d_A_weights, *d_A_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights"<<endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_A_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_A_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_A_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_A_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_A_weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying A_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_A_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying A_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }

        // Adam_Update_Weights_Activ<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, d_A_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        // if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
        //     cout<<"Error in synchronizing device"<<endl;
        //     exit(1);
        // }

        // Adam_Update_Bias_Activ<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, d_A_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        // if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
        //     cout<<"Error in synchronizing device"<<endl;
        //     exit(1);
        // }

        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_A_weights)))
        {
            cout << "Error in freeing d_A_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_A_biases)))
        {
            cout << "Error in freeing d_A_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void update_weights_SGD(T learning_rate) override
    {
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel
        update_weights_kernel<T><<<gridDim2D, blockDim2D>>>(d_weights, d_biases, d_d_weights, d_d_biases, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        update_bias_kernel<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host
        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void update_weights_AdamDecay(T learning_rate, T beta1, T beta2, T epsilon, int epochs){
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        int *d_B_weights, *d_B_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_m_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        AdamDecay_update_weights_kernel_Jenks<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, d_B_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        AdamDecay_update_bias_kernel_Jenks<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, d_B_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void update_weights_SGDJenks(T learning_rate){
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases;
        T* d_B_weights, *d_B_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(T)))) {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying B_weights from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying B_biases from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        // Launch the update weights kernel
        update_weights_kernel_Jenks<T><<<gridDim2D, blockDim2D>>>(d_weights, d_d_weights, d_B_weights, learning_rate, cols, rows);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        update_bias_kernel_Jenks<T><<<gridDim1D, blockDim1D>>>(d_biases, d_d_biases, d_B_biases, learning_rate, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host
        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void update_weights_AdamJenks(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override {
        /*
        Algorithm:
        m = beta1 * m + (1 - beta1) * d_weights
        v = beta2 * v + (1 - beta2) * d_weights^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        
        */
        T *d_weights, *d_biases, *d_d_weights, *d_d_biases, *d_v_weights, *d_v_biases, *d_m_weights, *d_m_biases;
        int *d_B_weights, *d_B_biases;
        int cols = this->cols;
        int rows = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_v_biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_m_weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_m_weights" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_m_biases, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_m_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_weights, rows * cols * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(int)))) {
            cout<<"Error in allocating memory for d_B_biases"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_weights, this->v_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_v_biases, this->v_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying v_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_weights, this->m_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_m_biases, this->m_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying m_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying B_biases from host to device" << endl;
            exit(1);
        }
        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);

        //Follow the algorithm  from line 1529 to 1538

        //Calculate m = beta1 * m + (1 - beta1) * d_weights using thrust

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        Adam_Update_Weights_Bernoulli_Zero<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_weights, d_d_weights, d_m_weights, d_v_weights, d_B_weights, beta1, beta2, epsilon, learning_rate, cols, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Adam_Update_Bias_Bernoulli_Zero<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_biases, d_d_biases, d_m_biases, d_v_biases, d_B_biases, beta1, beta2, epsilon, learning_rate, rows, epochs);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        // Copy the result weights and biases from device to host

        if (!HandleCUDAError(cudaMemcpy(this->weights, d_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->biases, d_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_weights, d_v_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->v_biases, d_v_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying v_biases from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_weights, d_m_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_weights from device to host" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(this->m_biases, d_m_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying m_biases from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(d_weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_weights)))
        {
            cout << "Error in freeing d_v_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_v_biases)))
        {
            cout << "Error in freeing d_v_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_weights)))
        {
            cout << "Error in freeing d_m_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_m_biases)))
        {
            cout << "Error in freeing d_m_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }

    }
    void find_Loss_Metric() override {
        T *dev_Weights, *dev_Biases, *d_d_Weights, *d_d_Biases;
        T *d_wDw, *d_bDb;

        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_wDw, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_bDb, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_bDb" << endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_Weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_Biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_wDw, this->W_dW_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying wDw from host to device"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_bDb, this->W_dW_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying bDb from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_d_Weights, d_wDw, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_d_Biases, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        //Delete streams
        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }

        //Transfer the result to host
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_weights, d_wDw, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying wDw from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_biases, d_bDb, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying bDb from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_wDw)))
        {
            cout << "Error in freeing d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_bDb)))
        {
            cout << "Error in freeing d_bDb" << endl;
            exit(1);
        }
    }
    void Agg_Jenks_Loss() override {
        T *dev_Weights, *dev_Biases, *d_d_Weights, *d_d_Biases;
        T *d_wDw, *d_bDb;
        T* d_WB_agg;
        T* d_BB_agg;
        //WB_agg for this function and BB_agg 

        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_wDw, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_bDb, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_bDb" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_WB_agg, rows * (cols) * sizeof(T)))) {
            cout<<"Error in allocating memory for d_WB_agg"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_BB_agg, rows * sizeof(T)))) {
            cout<<"Error in allocating memory for d_BB_agg"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_Weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_Biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_wDw, this->W_dW_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying wDw from host to device"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_bDb, this->W_dW_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying bDb from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_WB_agg, this->WB_agg, rows * (cols) * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying WB_agg from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_BB_agg, this->BB_agg, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying BB_agg from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_d_Weights, d_wDw, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_d_Biases, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        //Accumulate the results
        matrix_accum_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(d_WB_agg,d_wDw, rows, cols);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_accum_kernel<T><<<gridDim1D,blockDim1D,0,stream_bias>>>(d_BB_agg,d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }


        //Delete streams

        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }

        if(!HandleCUDAError(cudaMemcpy(this->WB_agg, d_WB_agg, rows * (cols) * sizeof(T), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying WB_agg from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->BB_agg, d_BB_agg, rows * sizeof(T), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying BB_agg from device to host"<<endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_wDw)))
        {
            cout << "Error in freeing d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_bDb)))
        {
            cout << "Error in freeing d_bDb" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_WB_agg))) {
            cout<<"Error in freeing d_WB_agg"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_BB_agg))) {
            cout<<"Error in freeing d_BB_agg"<<endl;
            exit(1);
        }
        // if(!HandleCUDAError(cudaFree(d_Loss_Data))) {
        //     cout<<"Error in freeing d_Loss_Data"<<endl;
        //     exit(1);
        // }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
    void find_Loss_Metric_Jenks_Aggressive_Single() override {
        T *d_wDw, *d_bDb;
        T* d_WB_agg;
        T* d_BB_agg;

        int cols = this->cols;
        int rows = this->rows;

        if(!HandleCUDAError(cudaMalloc((void **)&d_WB_agg, rows * (cols) * sizeof(T)))) {
            cout<<"Error in allocating memory for d_WB_agg"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_BB_agg, rows * sizeof(T))) ) {
            cout<<"Error in allocating memory for d_BB_agg"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if(!HandleCUDAError(cudaMemcpy(d_WB_agg, this->WB_agg, rows * (cols) * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying WB_agg from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_BB_agg, this->BB_agg, rows * sizeof(T), cudaMemcpyHostToDevice)) ) {
            cout<<"Error in copying BB_agg from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }

        //Delete streams


        //Fill the WB structure
        int size = rows * (cols + 1);

        T* d_WBW;
        T* d_BB;
        int* d_indices_W;
        int* d_indices_B;
        if (!HandleCUDAError(cudaMalloc((void **)&d_WBW, rows * (cols) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_WB" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_BB, rows*sizeof(T))) ) {
            cout<<"Error in allocating memory for d_BB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_indices_W, rows * (cols) * sizeof(int))) ) {
            cout<<"Error in allocating memory for d_indices"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_indices_B, rows*sizeof(int))) ) {
            cout<<"Error in allocating memory for d_indices_B"<<endl;
            exit(1);
        }

        Fill_WBW_device_array<T><<<gridDim2D, blockDim2D,0,stream_weights>>>(d_WBW, d_indices_W,d_WB_agg, rows, cols);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        Fill_WBB_device_array<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(d_BB, d_indices_B, d_BB_agg, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        thrust::sort_by_key(thrust::device, d_WBW, d_WBW + rows*cols, d_indices_W);
        thrust::sort_by_key(thrust::device, d_BB, d_BB + rows, d_indices_B);
        int* h_indices = (int*)malloc(rows*cols*sizeof(int));
        int* h_indices_B = (int*)malloc(rows*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_indices, d_indices_W, rows*cols*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_indices from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(h_indices_B, d_indices_B, rows*sizeof(int), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying d_indices_B from device to host"<<endl;
            exit(1);
        }
        // }
        //Perfrom the Jenks natural breaks optimization
        //Define the threads and block size
        //We will launch a kernel with as many threads as there are entries in the matrix
        T* d_var_W;
        T* d_var_B;

        if (!HandleCUDAError(cudaMalloc((void **)&d_var_W, ((rows*(cols))) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_var" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_var_B, rows * sizeof(T))) ) {
            cout<<"Error in allocating memory for d_var_B"<<endl;
            exit(1);
        }
        int TPB_3 = 256;
        dim3 blockDim(TPB_3,1, 1);
        dim3 gridDim((((rows*(cols))) + TPB_3 - 1) / TPB_3, 1, 1);  

        //Launch the kernel
        Jenks_Optimization_Weights<T><<<gridDim, blockDim,0,stream_weights>>>(d_WBW, d_var_W, rows, cols);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        Jenks_Optimization_Biases<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(d_BB, d_var_B, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        //Find the minimum of d_var
        int* d_min_W;
        int* d_min_B;
        if (!HandleCUDAError(cudaMalloc((void **)&d_min_W, ((rows*(cols)))*sizeof(int))))
        {
            cout << "Error in allocating memory for d_min" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_min_B, rows*sizeof(int))) ) {
            cout<<"Error in allocating memory for d_min_B"<<endl;
            exit(1);
        }
        thrust::sequence(thrust::device, d_min_W, d_min_W + ((rows*(cols))));
        thrust::sequence(thrust::device, d_min_B, d_min_B + rows);
        thrust::sort_by_key(thrust::device, d_var_W, d_var_W + ((rows*(cols))), d_min_W);   
        thrust::sort_by_key(thrust::device, d_var_B, d_var_B + rows, d_min_B);
        //The first entry of d_min will be the index of the minimum value of d_var
        // This will be the break point

        int* h_min_W = (int*)malloc((rows*(cols))*sizeof(int));
        int* h_min_B = (int*)malloc(rows*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_min_W, d_min_W, ((rows*(cols)))*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_min from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(h_min_B, d_min_B, rows*sizeof(int), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying d_min_B from device to host"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }


        //Transfer the result to host
        // if (!HandleCUDAError(cudaMemcpy(this->W_dW_weights, d_wDw, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        // {
        //     cout << "Error in copying wDw from device to host" << endl;
        //     exit(1);
        // }
        // if (!HandleCUDAError(cudaMemcpy(this->W_dW_biases, d_bDb, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        // {
        //     cout << "Error in copying bDb from device to host" << endl;
        //     exit(1);
        // }

        // for(int i = 0; i<this->rows; i++){
        //     cout<<"Score biases: "<<this->W_dW_biases[i]<<endl;
        // }
        // cout<<"Score weights"<<endl;
        // for(int i = 0; i<this->rows; i++){
        //     for(int j  = 0; j<this->cols; j++){
        //         cout<<this->W_dW_weights[i*this->cols + j]<<" ";
        //     }
        //     cout<<endl;
        // }
        int break_point = h_min_W[0];// Will need to add the offset of non zeros here
        int break_point_B = h_min_B[0];
        cout<<"The Weight break point is "<<break_point<<endl;
        cout<< "The Bias break point is "<<break_point_B<<endl;
        //Now we need to set the B matrix values to 0 before the break point and 1 after the break point
        //We will do this serially, and save in this->B_weights and this->B_biases
        int r_r, c_r;
        int* place = new int[2];
        for(int i =0; i<break_point;i++){
            //Now we need to use this->loss_data which has been sorted according to the loss values
            place = return_col_row(h_indices[i],cols);
            r_r = place[0];
            c_r = place[1];
            if (r_r > this->rows || c_r > this->cols) {
                cout<<"Error in accessing loss_data"<<endl;
                cout<<"r_r: "<<r_r<<endl;
                cout<<"c_r: "<<c_r<<endl;
                exit(1);
            }
            this->B_weights[r_r * this->cols + c_r] = 0;
        }
        for(int i = 0 ; i<break_point_B; i++){
            r_r = h_indices_B[i];
            if (r_r > this->rows) {
                cout<<"Error in accessing loss_data"<<endl;
                cout<<"r_r: "<<r_r<<endl;
                exit(1);
            }
            this->B_biases[r_r] = 0;
        }

        // Free device memory
        // if (!HandleCUDAError(cudaFree(d_wDw)))
        // {
        //     cout << "Error in freeing d_wDw" << endl;
        //     exit(1);
        // }
        // if (!HandleCUDAError(cudaFree(d_bDb)))
        // {
        //     cout << "Error in freeing d_bDb" << endl;
        //     exit(1);
        // }
        if (!HandleCUDAError(cudaFree(d_WB_agg)))
        {
            cout << "Error in freeing d_WB_agg" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_BB_agg)))
        {
            cout << "Error in freeing d_BB_agg" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_WBW))) {
            cout<<"Error in freeing d_WB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_BB))) {
            cout<<"Error in freeing d_BB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var_W))) {
            cout<<"Error in freeing d_var"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var_B))) {
            cout<<"Error in freeing d_var_B"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min_W))) {
            cout<<"Error in freeing d_min"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min_B))) {
            cout<<"Error in freeing d_min_B"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
    void find_Loss_Metric_Jenks_Aggressive() override {
        T *dev_Weights, *dev_Biases, *d_d_Weights, *d_d_Biases;
        T *d_wDw, *d_bDb;
        T* d_WB_agg;

        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_wDw, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_bDb, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_bDb" << endl;
            exit(1);
        }
        // if(!HandleCUDAError(cudaMalloc((void **)&d_WB_agg, rows * (cols+1) * sizeof(T)))) {
        //     cout<<"Error in allocating memory for d_WB_agg"<<endl;
        //     exit(1);
        // }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_Weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_Biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_wDw, this->W_dW_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying wDw from host to device"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_bDb, this->W_dW_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying bDb from host to device"<<endl;
            exit(1);
        }
        // if(!HandleCUDAError(cudaMemcpy(d_WB_agg, this->WB_agg, rows * (cols+1) * sizeof(T), cudaMemcpyHostToDevice))) {
        //     cout<<"Error in copying WB_agg from host to device"<<endl;
        //     exit(1);
        // }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_d_Weights, d_wDw, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_d_Biases, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        //Delete streams

        int TPB_2 = 16;
        dim3 blockDim2D_diff(TPB_2, TPB_2, 1);
        dim3 gridDim2D_diff((cols+1 + TPB_2 - 1) / TPB_2, (rows+TPB_2-1)/TPB_2, 1);

        //Fill the WB structure
        int size = rows * (cols + 1);

        T* d_WBW;
        T* d_BB;
        int* d_indices_W;
        int* d_indices_B;
        if (!HandleCUDAError(cudaMalloc((void **)&d_WBW, rows * (cols) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_WB" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_BB, rows*sizeof(T))) ) {
            cout<<"Error in allocating memory for d_BB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_indices_W, rows * (cols) * sizeof(int))) ) {
            cout<<"Error in allocating memory for d_indices"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_indices_B, rows*sizeof(int))) ) {
            cout<<"Error in allocating memory for d_indices_B"<<endl;
            exit(1);
        }

        Fill_WBW_device_array<T><<<gridDim2D, blockDim2D,0,stream_weights>>>(d_WBW, d_indices_W,d_wDw, rows, cols);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        Fill_WBB_device_array<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(d_BB, d_indices_B, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        thrust::sort_by_key(thrust::device, d_WBW, d_WBW + rows*cols, d_indices_W);
        thrust::sort_by_key(thrust::device, d_BB, d_BB + rows, d_indices_B);
        int* h_indices = (int*)malloc(rows*cols*sizeof(int));
        int* h_indices_B = (int*)malloc(rows*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_indices, d_indices_W, rows*cols*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_indices from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(h_indices_B, d_indices_B, rows*sizeof(int), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying d_indices_B from device to host"<<endl;
            exit(1);
        }
        // }
        //Perfrom the Jenks natural breaks optimization
        //Define the threads and block size
        //We will launch a kernel with as many threads as there are entries in the matrix
        T* d_var_W;
        T* d_var_B;

        if (!HandleCUDAError(cudaMalloc((void **)&d_var_W, ((rows*(cols))) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_var" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_var_B, rows * sizeof(T))) ) {
            cout<<"Error in allocating memory for d_var_B"<<endl;
            exit(1);
        }
        int TPB_3 = 256;
        dim3 blockDim(TPB_3,1, 1);
        dim3 gridDim((((rows*(cols+1))) + TPB_3 - 1) / TPB_3, 1, 1);  

        //Launch the kernel
        Jenks_Optimization_Weights<T><<<gridDim, blockDim,0,stream_weights>>>(d_WBW, d_var_W, rows, cols);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        Jenks_Optimization_Biases<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(d_BB, d_var_B, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        //Find the minimum of d_var
        int* d_min_W;
        int* d_min_B;
        if (!HandleCUDAError(cudaMalloc((void **)&d_min_W, ((rows*(cols)))*sizeof(int))))
        {
            cout << "Error in allocating memory for d_min" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_min_B, rows*sizeof(int))) ) {
            cout<<"Error in allocating memory for d_min_B"<<endl;
            exit(1);
        }
        thrust::sequence(thrust::device, d_min_W, d_min_W + ((rows*(cols))));
        thrust::sequence(thrust::device, d_min_B, d_min_B + rows);
        thrust::sort_by_key(thrust::device, d_var_W, d_var_W + ((rows*(cols))), d_min_W);   
        thrust::sort_by_key(thrust::device, d_var_B, d_var_B + rows, d_min_B);
        //The first entry of d_min will be the index of the minimum value of d_var
        // This will be the break point

        int* h_min_W = (int*)malloc((rows*(cols))*sizeof(int));
        int* h_min_B = (int*)malloc(rows*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_min_W, d_min_W, ((rows*(cols)))*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_min from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(h_min_B, d_min_B, rows*sizeof(int), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying d_min_B from device to host"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }


        //Transfer the result to host
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_weights, d_wDw, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying wDw from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_biases, d_bDb, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying bDb from device to host" << endl;
            exit(1);
        }

        // for(int i = 0; i<this->rows; i++){
        //     cout<<"Score biases: "<<this->W_dW_biases[i]<<endl;
        // }
        // cout<<"Score weights"<<endl;
        // for(int i = 0; i<this->rows; i++){
        //     for(int j  = 0; j<this->cols; j++){
        //         cout<<this->W_dW_weights[i*this->cols + j]<<" ";
        //     }
        //     cout<<endl;
        // }
        int break_point = h_min_W[0];// Will need to add the offset of non zeros here
        int break_point_B = h_min_B[0];
        cout<<"The Weight break point is "<<break_point<<endl;
        cout<< "The Bias break point is "<<break_point_B<<endl;
        //Now we need to set the B matrix values to 0 before the break point and 1 after the break point
        //We will do this serially, and save in this->B_weights and this->B_biases
        int r_r, c_r;
        int* place = new int[2];
        for(int i =0; i<break_point;i++){
            //Now we need to use this->loss_data which has been sorted according to the loss values
            place = return_col_row(h_indices[i],cols);
            r_r = place[0];
            c_r = place[1];
            if (r_r > this->rows || c_r > this->cols) {
                cout<<"Error in accessing loss_data"<<endl;
                cout<<"r_r: "<<r_r<<endl;
                cout<<"c_r: "<<c_r<<endl;
                exit(1);
            }
            this->B_weights[r_r * this->cols + c_r] = 0;
        }
        for(int i = 0 ; i<break_point_B; i++){
            r_r = h_indices_B[i];
            if (r_r > this->rows) {
                cout<<"Error in accessing loss_data"<<endl;
                cout<<"r_r: "<<r_r<<endl;
                exit(1);
            }
            this->B_biases[r_r] = 0;
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_wDw)))
        {
            cout << "Error in freeing d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_bDb)))
        {
            cout << "Error in freeing d_bDb" << endl;
            exit(1);
        }
        // if(!HandleCUDAError(cudaFree(d_Loss_Data))) {
        //     cout<<"Error in freeing d_Loss_Data"<<endl;
        //     exit(1);
        // }
        if(!HandleCUDAError(cudaFree(d_WBW))) {
            cout<<"Error in freeing d_WB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_BB))) {
            cout<<"Error in freeing d_BB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var_W))) {
            cout<<"Error in freeing d_var"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var_B))) {
            cout<<"Error in freeing d_var_B"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min_W))) {
            cout<<"Error in freeing d_min"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min_B))) {
            cout<<"Error in freeing d_min_B"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
    void find_Loss_Metric_Jenks() override {
        T *dev_Weights, *dev_Biases, *d_d_Weights, *d_d_Biases;
        T *d_wDw, *d_bDb;
        T* d_WB_agg;

        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_d_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_wDw, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_bDb, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_bDb" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_WB_agg, rows * (cols+1) * sizeof(T)))) {
            cout<<"Error in allocating memory for d_WB_agg"<<endl;
            exit(1);
        }

        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_d_Weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_weights from host to device" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_d_Biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying d_biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_wDw, this->W_dW_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying wDw from host to device"<<endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_bDb, this->W_dW_biases, rows * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying bDb from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_WB_agg, this->WB_agg, rows * (cols+1) * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying WB_agg from host to device"<<endl;
            exit(1);
        }

        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }


        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_d_Weights, d_wDw, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_d_Biases, d_bDb, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        //Delete streams
        if (!HandleCUDAError(cudaStreamDestroy(stream_weights)))
        {
            cout << "Error in destroying stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamDestroy(stream_bias)))
        {
            cout << "Error in destroying stream for bias" << endl;
            exit(1);
        }
        // Now we need to sort the weights and biases in descending order, and then perform Jenks natural breaks optimization
        //We will want to use Loc_Layers to keep track of the weights and biases

        //Fill the Loss_Data structure
        Loc_Layer<T> *d_Loss_Data;
        if (!HandleCUDAError(cudaMalloc((void **)&d_Loss_Data, (rows * (cols + 1)) * sizeof(Loc_Layer<T>))))
        {
            cout << "Error in allocating memory for d_Loss_Data" << endl;
            exit(1);
        }

        int TPB_2 = 16;
        dim3 blockDim2D_diff(TPB_2, TPB_2, 1);
        dim3 gridDim2D_diff((cols+1 + TPB_2 - 1) / TPB_2, (rows+TPB_2-1)/TPB_2, 1);
        cudaStream_t fill_agg;
        cudaStream_t fill_loss;

        if (!HandleCUDAError(cudaStreamCreate(&fill_agg)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&fill_loss)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }       

        Fill_Agg_Device<T><<<gridDim2D_diff, blockDim2D_diff,0,fill_agg>>>(d_WB_agg, d_wDw, d_bDb, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(fill_agg))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        Fill_Jenks_Device<T><<<gridDim2D_diff, blockDim2D_diff,0,fill_loss>>>(d_Loss_Data,d_wDw, d_bDb, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(fill_loss))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        if(!HandleCUDAError(cudaStreamDestroy(fill_agg))) {
            cout<<"Error in destroying stream"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaStreamDestroy(fill_loss))) {
            cout<<"Error in destroying stream"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->WB_agg, d_WB_agg, rows * (cols+1) * sizeof(T), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_WB_agg from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_WB_agg))) {
            cout<<"Error in freeing d_WB_agg"<<endl;
            exit(1);
        }

        //Fill the WB structure
        thrust::sort(thrust::device, d_Loss_Data, d_Loss_Data+(rows*(cols+1)), CompareBernoulliWeights<T>());



        T* d_WB;
        if (!HandleCUDAError(cudaMalloc((void **)&d_WB, rows * (cols + 1) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_WB" << endl;
            exit(1);
        }

        Fill_WB_device<T><<<gridDim2D, blockDim2D>>>(d_WB, d_Loss_Data, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }



        //Perfrom the Jenks natural breaks optimization
        //Define the threads and block size
        //We will launch a kernel with as many threads as there are entries in the matrix
        T* d_var;

        if (!HandleCUDAError(cudaMalloc((void **)&d_var, (rows*(cols+1)) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_var" << endl;
            exit(1);
        }
        int TPB_3 = 256;
        dim3 blockDim(TPB_3,1, 1);
        dim3 gridDim((rows*(cols+1) + TPB_2 - 1) / TPB_2, 1, 1);  

        //Launch the kernel
        Jenks_Optimization<T><<<gridDim, blockDim>>>(d_WB, d_var, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        //Find the minimum of d_var
        int* d_min;
        if (!HandleCUDAError(cudaMalloc((void **)&d_min, (rows*(cols+1))*sizeof(int))))
        {
            cout << "Error in allocating memory for d_min" << endl;
            exit(1);
        }
        thrust::sequence(thrust::device, d_min, d_min + rows*(cols+1));
        thrust::sort_by_key(thrust::device, d_var, d_var + rows*(cols+1), d_min);   
        //The first entry of d_min will be the index of the minimum value of d_var
        // This will be the break point

        int* h_min = (int*)malloc(rows*(cols+1)*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_min, d_min, rows*(cols+1)*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_min from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->loss_data, d_Loss_Data, (rows*(cols+1))*sizeof(Loc_Layer<T>), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_Loss_Data from device to host"<<endl;
            exit(1);
        }

        int break_point = h_min[0];
        cout<<"The break point is "<<break_point<<endl;
        //Now we need to set the B matrix values to 0 before the break point and 1 after the break point
        //We will do this serially, and save in this->B_weights and this->B_biases
        int r_r, c_r;
        for(int i =0; i<break_point;i++){
            //Now we need to use this->loss_data which has been sorted according to the loss values
            r_r = this->loss_data[i].row;
            c_r = this->loss_data[i].col;
            if (r_r > this->rows || c_r > this->cols) {
                cout<<"Error in accessing loss_data"<<endl;
                cout<<"r_r: "<<r_r<<endl;
                cout<<"c_r: "<<c_r<<endl;
                exit(1);
            }
            if(c_r == cols) {
                this->B_biases[r_r] = 0;
            } else {
                this->B_weights[r_r * this->cols + c_r] = 0;
            }
        }
        for(int i=break_point; i<(rows*(cols+1));i++){
            r_r = this->loss_data[i].row;
            c_r = this->loss_data[i].col;
            this->num_ones[r_r*cols + c_r]++;
            if(c_r == cols) {
                this->B_biases[r_r] = 1;
            } else {
                this->B_weights[r_r * this->cols + c_r] = 1;
            }
        }


        //Transfer the result to host
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_weights, d_wDw, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying wDw from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(this->W_dW_biases, d_bDb, rows * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying bDb from device to host" << endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Weights)))
        {
            cout << "Error in freeing d_d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_d_Biases)))
        {
            cout << "Error in freeing d_d_biases" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_wDw)))
        {
            cout << "Error in freeing d_wDw" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_bDb)))
        {
            cout << "Error in freeing d_bDb" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_Loss_Data))) {
            cout<<"Error in freeing d_Loss_Data"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_WB))) {
            cout<<"Error in freeing d_WB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var))) {
            cout<<"Error in freeing d_var"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min))) {
            cout<<"Error in freeing d_min"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
    void find_Loss_Metric_Jenks_Prune() override {
        T *dev_Weights, *dev_Biases;
        T* d_WB_agg;
        int cols = this->cols;
        int rows = this->rows;

        if (!HandleCUDAError(cudaMalloc((void **)&dev_Weights, rows * cols * sizeof(T))))
        {
            cout << "Error in allocating memory for d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&dev_Biases, rows * sizeof(T))))
        {
            cout << "Error in allocating memory for d_biases" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_WB_agg, rows * (cols+1) * sizeof(T)))) {
            cout<<"Error in allocating memory for d_WB_agg"<<endl;
            exit(1);
        }


        // Copy weights, biases, d_weights, and d_biases from host to device
        if (!HandleCUDAError(cudaMemcpy(dev_Weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying weights from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(dev_Biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying biases from host to device" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_WB_agg, this->WB_agg, rows * (cols+1) * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying WB_agg from host to device"<<endl;
            exit(1);
        }



        // Define grid and block dimensions
        int block_size = 16;
        dim3 blockDim2D(block_size, block_size);

        dim3 gridDim2D((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

        int TPB = 256;
        dim3 blockDim1D(TPB, 1, 1);
        dim3 gridDim1D((rows + TPB - 1) / TPB, 1, 1);
        // Now we need to sort the weights and biases in descending order, and then perform Jenks natural breaks optimization
        //We will want to use Loc_Layers to keep track of the weights and biases

        //Fill the Loss_Data structure
        Loc_Layer<T> *d_Loss_Data;
        if (!HandleCUDAError(cudaMalloc((void **)&d_Loss_Data, (rows * (cols + 1)) * sizeof(Loc_Layer<T>))))
        {
            cout << "Error in allocating memory for d_Loss_Data" << endl;
            exit(1);
        }

        int TPB_2 = 16;
        dim3 blockDim2D_diff(TPB_2, TPB_2, 1);
        dim3 gridDim2D_diff((cols+1 + TPB_2 - 1) / TPB_2, (rows+TPB_2-1)/TPB_2, 1);       

        Fill_Jenks_Device_Prune<T><<<gridDim2D_diff, blockDim2D_diff>>>(d_Loss_Data,d_WB_agg, cols, rows);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }

        //Fill the WB structure
        thrust::sort(thrust::device, d_Loss_Data, d_Loss_Data+(rows*(cols+1)), CompareBernoulliWeights<T>());



        T* d_WB;
        if (!HandleCUDAError(cudaMalloc((void **)&d_WB, rows * (cols + 1) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_WB" << endl;
            exit(1);
        }

        Fill_WB_device<T><<<gridDim2D, blockDim2D>>>(d_WB, d_Loss_Data, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }



        //Perfrom the Jenks natural breaks optimization
        //Define the threads and block size
        //We will launch a kernel with as many threads as there are entries in the matrix
        T* d_var;

        if (!HandleCUDAError(cudaMalloc((void **)&d_var, (rows*(cols+1)) * sizeof(T))))
        {
            cout << "Error in allocating memory for d_var" << endl;
            exit(1);
        }
        int TPB_3 = 256;
        dim3 blockDim(TPB_3,1, 1);
        dim3 gridDim((rows*(cols+1) + TPB_2 - 1) / TPB_2, 1, 1);  

        //Launch the kernel
        Jenks_Optimization<T><<<gridDim, blockDim>>>(d_WB, d_var, rows, cols);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        //Find the minimum of d_var
        int* d_min;
        if (!HandleCUDAError(cudaMalloc((void **)&d_min, (rows*(cols+1))*sizeof(int))))
        {
            cout << "Error in allocating memory for d_min" << endl;
            exit(1);
        }
        thrust::sequence(thrust::device, d_min, d_min + rows*(cols+1));
        thrust::sort_by_key(thrust::device, d_var, d_var + rows*(cols+1), d_min);   
        //The first entry of d_min will be the index of the minimum value of d_var
        // This will be the break point

        int* h_min = (int*)malloc(rows*(cols+1)*sizeof(int));
        if(!HandleCUDAError(cudaMemcpy(h_min, d_min, rows*(cols+1)*sizeof(int), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_min from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->loss_data, d_Loss_Data, (rows*(cols+1))*sizeof(Loc_Layer<T>), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying d_Loss_Data from device to host"<<endl;
            exit(1);
        }

        int break_point = h_min[0];
        cout<<"The break point is "<<break_point<<endl;
        //Now we need to set the B matrix values to 0 before the break point and 1 after the break point
        //We will do this serially, and save in this->B_weights and this->B_biases
        int r_r, c_r;
        for(int i =0; i<break_point;i++){
            //Now we need to use this->loss_data which has been sorted according to the loss values
            r_r = this->loss_data[i].row;
            c_r = this->loss_data[i].col;
            if(c_r == cols) {
                this->B_biases[r_r] = 0;
            } else {
                this->B_weights[r_r * this->cols + c_r] = 0;
            }
        }
        for(int i=break_point; i<(rows*(cols+1));i++){
            r_r = this->loss_data[i].row;
            c_r = this->loss_data[i].col;
            if(c_r == cols) {
                this->B_biases[r_r] = 1;
            } else {
                this->B_weights[r_r * this->cols + c_r] = 1;
            }
        }

        cudaStream_t stream_weights;
        cudaStream_t stream_bias;

        if (!HandleCUDAError(cudaStreamCreate(&stream_weights)))
        {
            cout << "Error in creating stream for weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaStreamCreate(&stream_bias)))
        {
            cout << "Error in creating stream for bias" << endl;
            exit(1);
        }

        int* d_B_Weights, *d_B_biases;
        if (!HandleCUDAError(cudaMalloc((void **)&d_B_Weights, rows * cols * sizeof(int))))
        {
            cout << "Error in allocating memory for d_B_Weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_B_biases, rows * sizeof(int))))
        {
            cout << "Error in allocating memory for d_B_biases" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_Weights, this->B_weights, rows * cols * sizeof(int), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying B_weights from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_B_biases, this->B_biases, rows * sizeof(int), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying B_biases from host to device"<<endl;
            exit(1);
        }
        //Perform elementwise multiplication of d_weights and W_dW_weights and d_biases and W_dW_biases

        matrix_elementwise_multiply_kernel<T><<<gridDim2D,blockDim2D,0,stream_weights>>>(dev_Weights, d_B_Weights, dev_Weights, cols, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_weights))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        vector_elementwise_multiply_kernel<T><<<gridDim1D, blockDim1D,0,stream_bias>>>(dev_Biases, d_B_biases, dev_Biases, rows);
        if(!HandleCUDAError(cudaStreamSynchronize(stream_bias))) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }


        //Transfer the result to host
        if(!HandleCUDAError(cudaMemcpy(this->weights, dev_Weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying weights from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->biases, dev_Biases, rows * sizeof(T), cudaMemcpyDeviceToHost)) ) {
            cout<<"Error in copying biases from device to host"<<endl;
            exit(1);
        }

        // Free device memory
        if (!HandleCUDAError(cudaFree(dev_Weights)))
        {
            cout << "Error in freeing d_weights" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(dev_Biases)))
        {
            cout << "Error in freeing d_biases" << endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_Loss_Data))) {
            cout<<"Error in freeing d_Loss_Data"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_WB))) {
            cout<<"Error in freeing d_WB"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_var))) {
            cout<<"Error in freeing d_var"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_min))) {
            cout<<"Error in freeing d_min"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_B_Weights))) {
            cout<<"Error in freeing d_B_Weights"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_B_biases))) {
            cout<<"Error in freeing d_B_biases"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
};


template <typename T>
class Conv1D : public Matrix<T>
{
    public:
    Conv1D(int width, int channels, int kernel_size, int stride, int padding, int filters, int batch_size) {
        this->width = width;
        this->channels = channels;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->filters = filters;
        this->rows = filters;
        this->cols = width;
        this->output_width = (width - kernel_size + 2 * padding) / stride + 1; 
        this->batch_size = batch_size;
        this->weights = (T *)malloc(filters * kernel_size * channels * sizeof(T));
        this->biases = (T *)malloc(filters * sizeof(T));
        this->input = (T *)malloc(width * channels * batch_size *  sizeof(T));
        this->hidden_output = (T *)malloc(output_width * filters * batch_size * sizeof(T));
    }
    int rows;
    int cols;
    int width;
    int batch_size;
    int channels;
    int kernel_size;
    int stride;
    int padding;
    int filters;
    T *biases;
    T *weights;
    T* input;
    T* hidden_output;
    int output_width;
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
    ~Conv1D() {
        free(this->weights);
        free(this->biases);
    }


};



template <typename T>
class Conv2D : public Matrix<T>
{
public:
    Conv2D(int width, int height, int channels, int kernel_width, int kernel_height, int stride, int padding, int filters, int batch_size): Matrix<T>(width, height, channels, kernel_width, kernel_height, stride, padding, filters, batch_size){};
    ~Conv2D()
    {
        free(this->weights);
        free(this->biases);
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
    void set_weights(T *weights, T *biases);
    void set_kernel_size(int kernel_size);
    void set_weights(T *weights);
    void set_stride(int stride);
    void set_padding(int padding);
    void update_weights_SGD(T learning_rate) override;
    void update_weights_AdamJenks(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_SGDJenks(T learning_rate) override;
    void update_weights_Momentum(T learning_rate, T momentum) override;
    void update_weights_RMSProp(T learning_rate, T decay_rate) override;
    void update_weights_Adam(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_AdamWBernoulli(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_AdamActiv(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_AdamWJenks(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_AdamDecay(T learning_rate, T beta1, T beta2, T epsilon, int epochs) override;
    void update_weights_SGDMomentum_Jenks(T learning_rate, T momentum) override;
    int get_rows();
    int get_cols()
    {
        return this->cols;
    }
};


template <typename T>
class MaxPooling1D : public Matrix<T>
{
public:
    MaxPooling1D(int kernel_width, int stride, int padding, int width, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    ~MaxPooling1D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class AvePooling1D : public Matrix<T>
{
public:
    AvePooling1D(int kernel_width, int stride, int padding, int width, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    ~AvePooling1D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
class AvePooling2D : public Matrix<T>
{
public:
    AvePooling2D(int kernel_width, int kernel_height, int stride, int padding, int width, int height, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->batch_size = batch_size;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->output_height = floor((height - 2 * padding - (kernel_height-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
    }
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int width;
    int height;
    int channels;
    int batch_size;
    int rows;
    int cols;
    int output_width;
    int output_height;
    T *input;
    T *hidden_output;
    ~AvePooling2D();
    void forward(T *input, T *output) override;
    void backward(T *loss) override;
};

template <typename T>
__global__ void MaxPooling2D_Backward_Kernel(int* d_max_indices, T* d_loss, T* d_next_loss, int output_width, int output_height, int channels, int batch_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x < output_width && y < output_height && z < channels) {
        for(int batch = 0; batch < batch_size; batch++) {
            int index = x + y * output_width + z * output_width * output_height + batch * output_width * output_height * channels;
            int max_index = d_max_indices[index];
            d_next_loss[max_index] = d_loss[index];
        }
    }
}


template <typename T>
class MaxPooling2D : public Matrix<T>
{
public:
    MaxPooling2D(int kernel_width, int kernel_height, int stride, int padding, int width, int height, int channels, int batch_size)
    {
        this->kernel_width = kernel_width;
        this->kernel_height = kernel_height;
        this->stride = stride;
        this->padding = padding;
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->batch_size = batch_size;
        this->rows = channels;
        this->cols = width * height;
        this->output_width = floor((width - 2 * padding - (kernel_width-1) - 1) / stride + 1);
        this->output_height = floor((height - 2 * padding - (kernel_height-1) - 1) / stride + 1);
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->loss = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(T));
        this->max_indices = (int *)malloc(this->output_width * this->output_height * channels * batch_size * sizeof(int));
    }
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
    int width;
    int height;
    int channels;
    int batch_size;
    int rows;
    int cols;
    int output_width;
    int output_height;
    int* max_indices;
    T *input;
    T *hidden_output;
    ~MaxPooling2D(){
        free(this->max_indices);
        free(this->input);
        free(this->hidden_output);
        free(this->next_loss);
        free(this->loss);
    }
    void forward(T *input, T *output) override;
    void backward(T *loss) override{
        /*For this, we need to only pass the gradient to the coordinates of max coordinates*/
        int* d_max_indices;
        T* d_loss;
        T* d_next_loss;
        int size = this->output_width * this->output_height * this->channels * this->batch_size;
        int next_size = this->width * this->height * this->channels * this->batch_size;
        if(!HandleCUDAError(cudaMalloc((void **)&d_max_indices, size * sizeof(int))) ) {
            cout<<"Error in allocating memory for d_max_indices"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))) ) {
            cout<<"Error in allocating memory for d_loss"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMalloc((void **)&d_next_loss, next_size * sizeof(T))) ) {
            cout<<"Error in allocating memory for d_next_loss"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_max_indices, this->max_indices, size * sizeof(int), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying max_indices from host to device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(d_loss, loss, size * sizeof(T), cudaMemcpyHostToDevice))) {
            cout<<"Error in copying loss from host to device"<<endl;
            exit(1);
        }
        int TPB = 8;
        //Define grid and block dimensions
        //We want to have the coordinates correlate to the max indices
        dim3 blockDim3D(TPB, TPB, TPB);
        dim3 gridDim_padding((output_width + TPB - 1) / TPB,(output_height + TPB - 1) / TPB, (channels + TPB - 1) / TPB);

        //Launch the kernel
        MaxPooling2D_Backward_Kernel<T><<<gridDim_padding, blockDim3D>>>(d_max_indices, d_loss, d_next_loss, output_width, output_height, channels, batch_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())) {
            cout<<"Error in synchronizing device"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaMemcpy(this->next_loss, d_next_loss, next_size * sizeof(T), cudaMemcpyDeviceToHost))) {
            cout<<"Error in copying next_loss from device to host"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_max_indices))) {
            cout<<"Error in freeing d_max_indices"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_loss))) {
            cout<<"Error in freeing d_loss"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaFree(d_next_loss))) {
            cout<<"Error in freeing d_next_loss"<<endl;
            exit(1);
        }
        if(!HandleCUDAError(cudaDeviceReset())) {
            cout<<"Error in resetting device"<<endl;
            exit(1);
        }
    }
};

template <typename T>
class Flatten : public Matrix<T>
{
public:
    Flatten(int width, int height, int channels, int batch_size)
    {
        this->width = width;
        this->height = height;
        this->channels = channels;
        this->batch_size = batch_size;
        this->rows = width * height * channels;
        this->cols = 1;
        this->input = (T *)malloc(width * height * channels * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(width * height * channels * batch_size * sizeof(T));
    }
    int width;
    int height;
    int channels;
    int batch_size;
    int rows;
    int cols;
    T *input;
    T *hidden_output;
    ~Flatten(){
        free(this->input);
        free(this->hidden_output);
    }
    void forward(T *input, T *output) override{
        for(int i = 0; i < this->rows * this->batch_size; i++){
            output[i] = input[i];
        }
    }
    void backward(T *loss) override{
        for(int i = 0; i < this->rows * this->batch_size; i++){
            this->next_loss[i]= loss[i];
        }
    }
};



template <typename T>
__global__ void Binary_Cross_Entropy_Kernel(T *label, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -1 * (label[index*batch_size + batch] * log(output[index*batch_size + batch]) + (1 - label[index*batch_size + batch]) * log(1 - output[index*batch_size + batch]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy(T *input, T *output, T *loss, int size, int batch_size)
{
    /*def forward(self, bottom, top):
   labels = bottom[1].data
   scores = bottom[0].data
   # Normalizing to avoid instability
   scores -= np.max(scores, axis=1, keepdims=True)
   # Compute Softmax activations
   exp_scores = np.exp(scores)
   probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
   logprobs = np.zeros([bottom[0].num,1])
   # Compute cross-entropy loss
   for r in range(bottom[0].num): # For each element in the batch
       scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
       for c in range(len(labels[r,:])): # For each class
           if labels[r,c] != 0:  # Positive classes
               logprobs[r] += -np.log(probs[r,c]) * labels[r,c] * scale_factor # We sum the loss per class for each element of the batch

   data_loss = np.sum(logprobs) / bottom[0].num

   self.diff[...] = probs  # Store softmax activations
   top[0].data[...] = data_loss # Store loss*/
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -1 * -log(input[index*batch_size + batch]) * output[index*batch_size + batch];
    }
}

template <typename T>
__global__ void Mean_Squared_Error_Kernel(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        loss[index] = 0.5 * pow(input[index] - output[index], 2);
    }
}

template <typename T>
__global__ void Mean_Squared_Error_Derivative(T *input, T *output, T *loss, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size)
    {
        loss[index] = input[index] - output[index];
    }
}

template <typename T>
__global__ void Binary_Cross_Entropy_Derivative(T *label, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    // The input is the output of the network and the output is the ground truth
    if (index < size && batch < batch_size)
    {
        loss[index*batch_size + batch] = -(label[index*batch_size + batch] / output[index*batch_size + batch] - (1 - label[index*batch_size + batch]) / (1 - output[index*batch_size + batch]));
    }
}

template <typename T>
__global__ void Categorical_Cross_Entropy_Derivative(T *input, T *output, T *loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    // The input is the output of the network and the output is the ground truth
    if (index < size && batch < batch_size)
    {
        loss[index* batch_size + batch] = -1 * (output[index* batch_size + batch] / input[index* batch_size + batch]);
    }
    __syncthreads();
}

template <typename T>
class Loss : public Matrix<T>
{
public:
    T* labels;
    T* output;
    Loss()
    {
        this->size = 0;
        this->loss = NULL;
        this->name = "loss";
    }
    Loss(int size)
    {
        cout << "Loss Constructor" << endl;
    }
    Loss(int size, int batch_size): Matrix<T>(size,batch_size){
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
    };
    virtual ~Loss(){};
    virtual void forward(T *input, T *output) override {};
    virtual void backward(T *loss) override {};
    virtual void set_labels(T *labels)  {};
};

template <typename T>
class Mean_Squared_Error : public Loss<T>
{
public:
    Mean_Squared_Error()
    {
        this->size = 0;
        this->rows = 0;
    }
    Mean_Squared_Error(int size)
    {
        this->size = size;
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "mean_squared_error";
    }
    ~Mean_Squared_Error() override
    {
        free(this->loss);
        free(this->input);
        free(this->output);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_input, *d_output, *d_loss;
        int size = this->rows;
        if (input == NULL)
        {
            cout << "Input of Categorical is NULL" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_output, output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy kernel
        Mean_Squared_Error_Kernel<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(output, this->output, size * sizeof(T));
        memcpy(input, this->input, size * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int size = this->rows;
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->output, size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        // Launch the categorical cross entropy derivative kernel
        Mean_Squared_Error_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
class Binary_CrossEntropy : public Loss<T>
{
public:
    Binary_CrossEntropy()
    {
        this->size = 0;
        this->rows = 0;
    }
    Binary_CrossEntropy(int size)
    {
        this->size = size;
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "binary_cross_entropy";
    }
    Binary_CrossEntropy(int size, int batch_size) : Loss<T>(size, batch_size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
        this->labels = (T *)malloc(size * batch_size * sizeof(T));
        this->name = "categorical";
        this->batch_size = batch_size;
    }
    ~Binary_CrossEntropy() override
    {
        free(this->loss);
        free(this->input);
        free(this->output);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_pred, *d_labels, *d_loss;
        int rows = this->rows;
        int batch_size = this->batch_size;
        //Output holds the labels
        //Input holds the predictions
        memcpy(this->labels, output, rows * batch_size * sizeof(T));
        memcpy(this->input, input, rows * batch_size * sizeof(T));
        if (!HandleCUDAError(cudaMalloc((void **)&d_pred, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_labels, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_pred, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_labels, output, rows * batch_size *  sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy kernel
        Binary_Cross_Entropy_Kernel<T><<<gridDim, blockDim>>>(d_labels, d_pred, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_pred)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_labels)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(this->hidden_output, input, rows * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int rows = this->rows;
        int batch_size = this->batch_size;

        //-label/output

        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }
        //Output holds the labels
        //Input holds the predictions

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->labels, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }
        //-d_gt/d_out

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy derivative kernel
        Binary_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_gt, d_out, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
};

template <typename T>
class Categorical : public Loss<T>
{
public:
    Categorical()
    {
        this->rows = 0;
    }
    Categorical(int size) : Loss<T>(size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * sizeof(T));
        this->input = (T *)malloc(size * sizeof(T));
        this->hidden_output = (T *)malloc(size * sizeof(T));
        this->next_loss = (T *)malloc(size * sizeof(T));
        this->name = "categorical";
    }
    Categorical(int size, int batch_size) : Loss<T>(size, batch_size)
    {
        this->rows = size;
        this->loss = (T *)malloc(size * batch_size * sizeof(T));
        this->input = (T *)malloc(size * batch_size * sizeof(T));
        this->hidden_output = (T *)malloc(size * batch_size * sizeof(T));
        this->next_loss = (T *)malloc(size * batch_size * sizeof(T));
        this->labels = (T *)malloc(size * batch_size * sizeof(T));
        this->name = "categorical";
    }
    ~Categorical() override
    {
        free(this->loss);
        free(this->input);
        free(this->hidden_output);
        free(this->next_loss);
    }
    void forward(T *input, T *output) override
    {
        // Allocate device memory for input and output
        T *d_input, *d_output, *d_loss;
        int rows = this->rows;
        int batch_size = this->batch_size;
        memcpy(this->labels, output, rows * batch_size * sizeof(T));
        memcpy(this->input, input, rows * batch_size * sizeof(T));
        if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }

        // Copy input from host to device
        if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_output, output, rows * batch_size *  sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }

        // Define grid and block dimensions

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy kernel
        Categorical_Cross_Entropy<T><<<gridDim, blockDim>>>(d_input, d_output, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }

        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_input)))
        {
            cout << "Error in freeing d_input" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_output)))
        {
            cout << "Error in freeing d_output" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
        memcpy(this->hidden_output, input, rows * sizeof(T));
    }
    void backward(T *lss) override
    {
        /*Calculate the derivative of the Cost with respect to the last output to begin backpropogation*/
        T *d_loss;
        T *d_out, *d_gt;
        int rows = this->rows;
        int batch_size = this->batch_size;

        //-label/output

        if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_out, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMalloc((void **)&d_gt, rows * batch_size * sizeof(T))))
        {
            cout << "Error in allocating memory for d_gt" << endl;
            exit(1);
        }

        if (!HandleCUDAError(cudaMemcpy(d_out, this->input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying input from host to device, Categorical Loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaMemcpy(d_gt, this->labels, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
        {
            cout << "Error in copying output from host to device" << endl;
            exit(1);
        }
        //-d_gt/d_out

        int threadsPerBlock = 16;
        int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
        dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

        // Launch the categorical cross entropy derivative kernel
        Categorical_Cross_Entropy_Derivative<T><<<gridDim, blockDim>>>(d_out, d_gt, d_loss, rows, batch_size);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
        // Copy the result loss from device to host
        if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
        {
            cout << "Error in copying loss from device to host" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_out)))
        {
            cout << "Error in freeing d_out" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_gt)))
        {
            cout << "Error in freeing d_gt" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaFree(d_loss)))
        {
            cout << "Error in freeing d_loss" << endl;
            exit(1);
        }
        if (!HandleCUDAError(cudaDeviceReset()))
        {
            cout << "Error in resetting device" << endl;
            exit(1);
        }
    }
    void set_labels(T *labels) override
    {
        this->labels = labels;
    }
};

template <typename T>
class Network
{
public:
    Network(int input_size, int *hidden_size, int output_size, int num_layers);
    Network(int input_size, int output_size, Optimizer<T>* optimizer){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->num_updateable = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
    }
    Network(int input_size, int output_size, Optimizer<T>* optimizer, int Q){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
        this->Q = Q;
    }
    Network(int input_size, int output_size, Optimizer<T>* optimizer, int Q, int batch_size, string name){
        this->input_size = input_size;
        this->output_size = output_size;
        this->num_layers = 0;
        this->num_activation = 0;
        this->num_derv = 0;
        this->optim = optimizer;
        if(optimizer == NULL){
            cout<<"Optimizer is NULL"<<endl;
            exit(1);
        }
        this->Q = Q;
        this->batch_size = batch_size;
        this->data_set = name;
        this->compression_ratio = 0;
    }
    ~Network(){};
    int input_size;
    int *hidden_size;
    int output_size;
    int num_layers;
    int num_activation;
    int num_derv;
    int Q;
    int batch_size;
    int num_updateable;
    string data_set;
    float accuracy;
    float compression_ratio;
    string loss_function;
    int epochs;
    T learning_rate;
    T *input;
    T *prediction;
    Optimizer<T>* optim;
    thrust::host_vector<Matrix<T> *> layers;
    thrust::host_vector<Matrix<T> *> activation;
    thrust::host_vector<Loc_Layer<T>*> bernoullie_w;
    thrust::host_vector<Loc_Layer<T>*> bernoullie_b;
    thrust::host_vector<T *> loss;
    thrust::host_vector<T *> hidden;
    thrust::host_vector<LayerMetadata> layerMetadata;
    void backward(T *input, T *output)
    {
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            if (i < layers.size())
            { // Ensure i is within bounds
                layers[i]->backward(loss[i]);
                if (i > 0)
                {
                    memcpy(loss[i - 1], layers[i]->next_loss, layers[i-1]->rows * batch_size * sizeof(T));
                    //Display loss and layer name for debugging

                }
            }
            else
            {
                cout << "Index " << i << " out of bounds for layers vector." << endl;
            }
        }
    }
    void update_weights(T learning_rate){};
    void update_weights(T learning_rate, int epochs, int Q);
    void update_weights(T learning_rate, int epochs, int Q, int total_epochs);
    void addLayer(Linear<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        if(this->optim->name == "AdamWBernoulli"){
            bernoullie_w.push_back((Loc_Layer<T> *)malloc((layer->rows * layer->cols + layer->rows) * sizeof(Loc_Layer<T>)));
        }
        layer->name = "saved linear";
        cout<<layer->name<<endl;
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->cols * this->batch_size  * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_updateable = bernoullie_w.size()-1;
        layerMetadata.push_back(LayerMetadata(num_layers,num_updateable, true)); // Assuming Linear layers are updateable
        num_layers++;
        num_derv++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Conv2D<T> *layer)
    {
        //Examine the layers structure

        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        if(this->optim->name == "AdamWBernoulli"){
            bernoullie_w.push_back((Loc_Layer<T> *)malloc(layer->kernel_width * layer->kernel_height * layer->filters * layer->channels * sizeof(Loc_Layer<T>)));
            bernoullie_b.push_back((Loc_Layer<T> *)malloc(layer->filters* sizeof(Loc_Layer<T>)));
        }
        layer->name = "saved conv2D";
        cout<<layer->name<<endl;
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->cols * this->batch_size  * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        num_updateable = bernoullie_w.size()-1;
        layerMetadata.push_back(LayerMetadata(num_layers,num_updateable, true)); // Assuming Linear layers are updateable
        num_layers++;
        num_derv++;
        layer->batch_size = this->batch_size;

    }
    void addLayer(MaxPooling2D<T> *layer)
    {
        layers.push_back(layer);
        //The loss is the same size as the output of the layer
        loss.push_back((T*)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        hidden.push_back((T*)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        num_layers++;
        layer->name = "saved maxpooling2d";
        if (layer->next_loss == NULL)
        {
            //NOT CORRECT
            layer->next_loss = (T *)malloc(layer->width * layer->height * layer->channels * this->batch_size * sizeof(T));
        }
    }
    void addLayer(AvePooling2D<T> *layer){
        layers.push_back(layer);
        //The loss is the same size as the output of the layer
        loss.push_back((T*)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        hidden.push_back((T*)malloc(layer->output_width * layer->output_height * layer->filters * this->batch_size * sizeof(T)));
        num_layers++;
        layer->name = "saved avepooling2d";
        if (layer->next_loss == NULL)
        {
            //NOT CORRECT
            layer->next_loss = (T *)malloc(layer->width * layer->height * layer->channels * this->batch_size * sizeof(T));
        }
    }
    void addLayer(Sigmoid<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(RELU_layer<T> *layer)
    {
        layers.push_back(layer);
        layer->name = "saved RELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(LeakyRELU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved LeakyRELU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Tanh<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved Tanh";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(SLU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved SLU";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(ELU_layer<T> * layer)
    {
        layers.push_back(layer);
        layer->name = "saved Tanh";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLayer(Softmax<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved softmax";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void addLoss(Binary_CrossEntropy<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved Binary";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
        this->loss_function = "Binary_Cross_Entropy";
    }
    void addLoss(Mean_Squared_Error<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        layer->name = "saved MSE";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        hidden.push_back((T *)malloc(layer->rows * this->batch_size * sizeof(T)));
        num_layers++;
        layer->batch_size = this->batch_size;
        this->loss_function = "Mean_Squared_Error";
    }
    void addLoss(Categorical<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * sizeof(T)));
        layer->name = "saved categorical";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * this->batch_size * sizeof(T));
        }
        num_layers++;
        layer->batch_size = this->batch_size;
        this->loss_function = "Categorical";
    }
    void addLayer(Flatten<T> *layer)
    {
        layers.push_back(layer);
        loss.push_back((T *)malloc(layer->rows * layer->cols * layer->channels * this->batch_size * sizeof(T)));
        hidden.push_back((T *)malloc(layer->rows * layer->cols * layer->channels * this->batch_size * sizeof(T)));
        layer->name = "saved flatten";
        if (layer->next_loss == NULL)
        {
            layer->next_loss = (T *)malloc(layer->rows * layer->cols * layer->channels * this->batch_size * sizeof(T));
        }
        num_layers++;
        layer->batch_size = this->batch_size;
    }
    void train(T *input, T *output, int epochs, T learning_rate);
    void train(T **input, T **output, int epochs, T learning_rate, int size);
    void predict(T *input, T *output);
    void predict(T **input, T **output, int size);
    void set_input_size(int input_size);
    void set_hidden_size(int *hidden_size);
    void set_output_size(int output_size);
    void set_num_layers(int num_layers);
    int get_input_size();
    int *get_hidden_size();
    int get_output_size();
    void forward(T *input, T *output)
    {
        if(layers[0]->hidden_output == nullptr){
            cout<<"Hidden output is null"<<endl;
            exit(1);
        }
        layers[0]->forward(input, layers[0]->hidden_output);
        // cout<< "Layer 0 output"<<endl;
        //     for(int j = 0; j<layers[0]->rows; j++){
        //         for(int k = 0; k<batch_size; k++){
        //             cout<<layers[0]->hidden_output[j*batch_size + k]<<" ";
        //         }
        //         cout<<endl;
        //     }
        // cout<<endl;
        for (int i = 1; i < layers.size() - 1; i++)
        {   
            layers[i]->forward(layers[i - 1]->hidden_output, layers[i]->hidden_output);
            // cout<<"Layer "<<i<<" output"<<endl;
            // for(int j = 0; j<layers[i]->rows; j++){
            //     for(int k = 0; k<batch_size; k++){
            //         cout<<layers[i]->hidden_output[j*batch_size + k]<<" ";
            //     }
            //     cout<<endl;
            // }
        }
        // Should be the cost layer
        layers[layers.size() - 1]->forward(layers[layers.size() - 2]->hidden_output, output);
    }
    void getOutput(T *output)
    {
        memcpy(output, prediction, output_size * sizeof(T));
    }
    void Fill_Bern(Matrix<T>* Layer, int layer_num){
        if(this->optim->name == "AdamWBernoulli"){
            for(int i = 0; i<Layer->rows; i++){
                for(int j = 0; j<Layer->cols; j++){
                    bernoullie_w[layer_num][i*Layer->cols + j].row = i;
                    bernoullie_w[layer_num][i*Layer->cols + j].col = j;
                    bernoullie_w[layer_num][i*Layer->cols + j].layer = layer_num;
                    bernoullie_w[layer_num][i*Layer->cols + j].weights_dW = Layer->W_dW_weights[i*Layer->cols + j];
                }
            }
            for(int i = 0; i<Layer->rows; i++){
                bernoullie_w[layer_num][i].row = i;
                bernoullie_w[layer_num][i].col = Layer->cols;
                bernoullie_w[layer_num][i].layer = layer_num;
                bernoullie_w[layer_num][i].weights_dW = Layer->W_dW_biases[i];
            }
        }
    }
    thrust::host_vector<Loc_Layer<T>> flatten() {   
        thrust::host_vector<Loc_Layer<T>>  result;
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        result.insert(result.end(), bernoullie_w[layerMetadata[i].LinNumber], bernoullie_w[layerMetadata[i].LinNumber]+layers[layerMetadata[i].layerNumber]->rows*layers[layerMetadata[i].layerNumber]->cols); 
                        //Append the biases
                        result.insert(result.end(), bernoullie_b[layerMetadata[i].LinNumber], bernoullie_b[layerMetadata[i].LinNumber]+layers[layerMetadata[i].layerNumber]->rows);
                    }
                }
            }
        }
        return result;
    }
    void Save_Data_to_CSV();
};



template <typename T>
__global__ void matrix_multiply_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        T sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

template <typename T>
void Matrix<T>::matrix_multiply(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, cols * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, cols * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((input_size + block_size - 1) / block_size, (output_size + block_size - 1) / block_size, 1);

    // Launch the matrix multiplication kernel
    matrix_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_add_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_add(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix addition kernel
    matrix_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }
    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_subtract_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_subtract(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix subtraction kernel
    matrix_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_transpose_kernel(T *A, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[col * rows + row] = A[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_transpose(T *A, T *C)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, cols * rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix transpose kernel
    matrix_transpose_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, cols * rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_multiply_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] * scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_multiply(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix scalar multiplication kernel
    matrix_scalar_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_add_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_add(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix scalar addition kernel
    matrix_scalar_add_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_subtract_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_subtract(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix scalar subtraction kernel
    matrix_scalar_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_multiply(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix elementwise multiplication kernel
    matrix_elementwise_multiply_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_divide_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] / B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_divide(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix elementwise division kernel
    matrix_elementwise_divide_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_add_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_add(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix elementwise addition kernel
    matrix_elementwise_add_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_elementwise_subtract_kernel(T *A, T *B, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
    }
}

template <typename T>
void Matrix<T>::matrix_elementwise_subtract(T *A, T *B, T *C)
{
    // Allocate device memory for matrices A, B, and C
    T *d_A, *d_B, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_B, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrices A and B from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying B from host to device" << endl;
        exit(1);
    }
    int input_size = cols;
    int output_size = rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);

    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    // Launch the matrix elementwise subtraction kernel
    matrix_elementwise_subtract_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_B)))
    {
        cout << "Error in freeing d_B" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_sum_axis0_kernel(T *A, T *C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols)
    {
        T sum = 0;
        for (int row = 0; row < rows; row++)
        {
            sum += A[row * cols + col];
        }
        C[col] = sum;
    }
}

template <typename T>
__global__ void matrix_sum_axis1_kernel(T *A, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows)
    {
        T sum = 0;
        for (int col = 0; col < cols; col++)
        {
            sum += A[row * cols + col];
        }
        C[row] = sum;
    }
}

template <typename T>
void Matrix<T>::matrix_sum(T *A, T *C, int axis)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int input_size = (axis == 0) ? rows : cols;
    int output_size = (axis == 0) ? cols : rows;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    if (axis == 0)
    {
        // Launch the matrix sum along axis 0 kernel
        matrix_sum_axis0_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
    }
    else if (axis == 1)
    {
        // Launch the matrix sum along axis 1 kernel
        matrix_sum_axis1_kernel<T><<<gridDim, blockDim>>>(d_A, d_C, rows, cols);
        if (!HandleCUDAError(cudaDeviceSynchronize()))
        {
            cout << "Error in synchronizing device" << endl;
            exit(1);
        }
    }

    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void matrix_scalar_divide_kernel(T *A, T scalar, T *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        C[row * cols + col] = A[row * cols + col] / scalar;
    }
}

template <typename T>
void Matrix<T>::matrix_scalar_divide(T *A, T *C, T scalar)
{
    // Allocate device memory for matrices A and C
    T *d_A, *d_C;
    if (!HandleCUDAError(cudaMalloc((void **)&d_A, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_C, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_C" << endl;
        exit(1);
    }

    // Copy matrix A from host to device
    if (!HandleCUDAError(cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying A from host to device" << endl;
        exit(1);
    }
    int output_size = rows;
    int input_size = cols;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);
    // Launch the matrix scalar division kernel
    matrix_scalar_divide_kernel<T><<<gridDim, blockDim>>>(d_A, scalar, d_C, rows, cols);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    // Copy the result matrix C from device to host
    if (!HandleCUDAError(cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying C from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_A)))
    {
        cout << "Error in freeing d_A" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_C)))
    {
        cout << "Error in freeing d_C" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Matrix<T>::set_cols(int cols)
{
    this->cols = cols;
}

template <typename T>
void Matrix<T>::set_rows(int rows)
{
    this->rows = rows;
}

template <typename T>
int Matrix<T>::get_cols()
{
    return cols;
}

template <typename T>
int Matrix<T>::get_rows()
{
    return rows;
}

template <typename T>
__global__ void sigmoid_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = 1 / (1 + exp(-input[index*batch_size + batch]));
    }
}

template <typename T>
void Sigmoid<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->rows;
    T *d_input, *d_output;
    // this->input = input;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input Sigmoid is NULL" << endl;
        input = (T *)malloc(size * batch_size * sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of Sigmoid is NULL" << endl;
        output = (T *)malloc(size * batch_size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of Sigmoid is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * batch_size * sizeof(T));
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the sigmoid kernel
    sigmoid_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    memcpy(this->hidden_output, output, size * batch_size * sizeof(T));
}

template <typename T>
__global__ void sigmoid_derivative_kernel(T *input, T* loss, T* fin_loss, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        fin_loss[index*batch_size+batch] = input[index*batch_size+batch] * (1 - input[index*batch_size+batch])*loss[index*batch_size+batch];
    }
}

template <typename T>
void Sigmoid<T>::backward(T *loss)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;
    T* d_fin_loss;
    int rows = this->rows;
    int batch_size = this->batch_size;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * batch_size* sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_fin_loss" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the sigmoid derivative kernel
    sigmoid_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_loss, d_fin_loss, rows, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(loss, d_fin_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss_mat" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_fin_loss)))
    {
        cout << "Error in freeing d_fin_loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void RELU_kernel(T *input, T *output, int size, int batch_size)
{
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && batch < batch_size)
    {
        output[index*batch_size + batch] = input[index*batch_size + batch] > 0 ? input[index*batch_size + batch] : 0;
    }
}

template <typename T>
void SoftPlus<T>::forward(T* input, T* output){
    // Allocate device memory for input and output
    int size = this->rows;
    T *d_input, *d_output;
    // this->input = input;
    int batch_size = this->batch_size;
    if (input == NULL)
    {
        cout << "Input SoftPlus is NULL" << endl;
        input = (T *)malloc(size * batch_size * sizeof(T));
        if (input == NULL)
        {
            cout << "Input of SoftPlus is NULL" << endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of SoftPlus is NULL" << endl;
        output = (T *)malloc(size * batch_size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of SoftPlus is NULL" << endl;
            exit(1);
        }
    }
    memcpy(this->input, input, size * batch_size * sizeof(T));
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the SoftPlus kernel
    // SoftPlus_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    // if (!HandleCUDAError(cudaDeviceSynchronize()))
    // {
    //     cout << "Error in synchronizing device" <<endl;
    // }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void SoftPlus<T>::backward(T* loss){
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;
    T* d_fin_loss;
    int rows = this->rows;
    int batch_size = this->batch_size;

    if (!HandleCUDAError(cudaMalloc((void **)&d_input, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_fin_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_fin_loss" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, SoftPlus loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying loss from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid2, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the SoftPlus derivative kernel
    // SoftPlus_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_loss, d_fin_loss, rows, batch_size);
    // if (!HandleCUDAError(cudaDeviceSynchronize()))
    // {
    //     cout << "Error in synchronizing device" << endl;
    //     exit(1);
    // }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(loss, d_fin_loss, rows * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss_mat" << endl;
        exit(1);
    }
}

template <typename T>
void RELU_layer<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    int size = this->size;
    int batch_size = this->batch_size;
    // this->input = input;
    if (input == NULL)
    {
        cout << "Input RELU is NULL" << endl;
        input = (T *)malloc(size * batch_size* sizeof(T));
        if (input == NULL)
        {
            cout << "Input of RELU is NULL" << endl;
            exit(1);
        }
    }
    if(this->input == NULL){
        cout<<"Input of RELU is NULL"<<endl;
        this->input = (T*)malloc(size * batch_size * sizeof(T));
        if(this->input == NULL){
            cout<<"Input of RELU is NULL"<<endl;
            exit(1);
        }
    }
    if (output == NULL)
    {
        cout << "Output of RELU is NULL" << endl;
        output = (T *)malloc(size * batch_size * sizeof(T));
        if (output == NULL)
        {
            cout << "Output of RELU is NULL" << endl;
            exit(1);
        }
    }
    //Set this->input equal to input
    memcpy(this->input, input, size * batch_size * sizeof(T));
    T *d_input, *d_output;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    printCudaMemoryUsage();
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the RELU kernel
    RELU_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    // this->hidden_output = output;
    memcpy(this->hidden_output, output, size * batch_size * sizeof(T));

}

template <typename T>
__global__ void RELU_derivative_kernel(T* input, T* loss, T* fin_loss, int size, int batch_size)
{
    int batch = blockIdx.x*blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;

    if (index < size && batch < batch_size)
    {
        fin_loss[index*batch_size+batch] = input[index*batch_size+batch] > 0 ? loss[index*batch_size+batch] : 0;
    }
}

template <typename T>
void RELU_layer<T>::backward(T *loss)
{
    T *d_input, *d_output;
    T *d_loss;
    T *input = this->hidden_output;
    T* d_fin_loss;
    int batch_size = this->batch_size;

    int size = this->rows;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_fin_loss, size * batch_size * sizeof(T)))){
        cout<<"Error in allocating memory for d_fin_loss"<<endl;
        exit(1);
    }

    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, ReLU loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_loss, loss, size * batch_size * sizeof(T), cudaMemcpyHostToDevice))){
        cout<<"Error in copying loss from host to device, ReLU"<<endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_batch = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocksPerGrid_batch, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);
    // Launch the sigmoid derivative kernel
    RELU_derivative_kernel<T><<<gridDim, blockDim>>>(d_input, d_loss, d_fin_loss, size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }

    if (this->next_loss == NULL)
    {
        cout << "Next loss is NULL for ReLU" << endl;
        this->next_loss = (T *)malloc(size * sizeof(T));
    }

    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_fin_loss, size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_loss_mat" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaFree(d_fin_loss))){
        cout<<"Error in freeing d_fin_loss"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
__global__ void linear_kernel(T *input, T *output, T *weights, T *biases, int input_size, int output_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < output_size)
    {
        T sum = 0;
        for (int i = 0; i < input_size; i++)
        {
            sum += weights[index * input_size + i] * input[i];
        }
        output[index] = sum + biases[index];
    }
}

template <typename T>
__global__ void linear_kernel_batch(T* Input, T* Output, T* Weights, T* bias, const int nx, const int ny, const int batch_size) {
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	//This conditional is for debugging, even though done on the device
	if (row < ny && col < batch_size) { 
		for (int k = 0; k < nx; k++) {
			fSum += Weights[row * nx + k] * Input[k * batch_size + col];
		}
		Output[row * batch_size + col] = fSum + bias[row];
	}
}

template <typename T>
void Linear<T>::forward(T *input, T *output)
{
    // Allocate device memory for input, output, weights, and biases
    int input_size = this->cols;
    int output_size = this->rows;
    int batch_size = this->batch_size;
    memcpy(this->input, input, input_size * batch_size * sizeof(T));
    T *d_input, *d_output, *dev_weights, *dev_biases;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_weights, input_size * output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_biases, output_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Linear" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_weights, this->weights, input_size * output_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_biases, this->biases, output_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int batchPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(batchPerGrid, blocksPerGrid, 1);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);

    // Launch the linear kernel
    linear_kernel_batch<T><<<gridDim, blockDim>>>(d_input, d_output, dev_weights, dev_biases, input_size, output_size, batch_size);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
    if (output == NULL)
    {
        cout << "Output of Linear is NULL" << endl;
        if (output == NULL)
        {
            cout << "Output of Linear is NULL" << endl;
        }
    }
    mempcpy(this->hidden_output, output, output_size * sizeof(T));

}

template <typename T>
__global__ void linear_derivative_kernel(T *loss, T *d_Weights, T *output, int rows, int cols, int batch_size)
{

    /*Calculate the weight update for the backward step

    We need to update the loss being propogated and also save the change in weights
    This means we will need to pass in the activation from the prior layer, the weights (For W^T), and the loss from the last layer (delta)
    */

    /* Python code to describe:
         x = self.cache  # Retrieve the cached input

         # Compute gradient with respect to input
         dx = np.dot(dout, self.W.T)  # Gradient of the loss w.r.t. input x, delta* W^T

         # Compute gradient with respect to weights
         dW = np.dot(x.T, dout)  # Gradient of the loss w.r.t. weights W, delta* x^T(activation of previous layer)

         # Compute gradient with respect to biases
         db = np.sum(dout, axis=0, keepdims=True)  # Gradient of the loss w.r.t. biases b

         # Update weights and biases (if using a simple gradient descent step)
         learning_rate = 0.01  # Example learning rate
         self.W -= learning_rate * dW  # Update weights
         self.b -= learning_rate * db  # Update biases

         Rows represent the output shape, i.e. the shape of the delta function for the loss
         Columns represent the input shape, i.e. the shape of the activation from the previous layer
     */

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Multiply the loss by the transpose of the weights (W^T)*delta
    // This is the derivative of the loss with respect to the input


    /*dL/dX, used to pass to the next layer
    If the Weight matrix is NxM, then this matrix will be Nx1
    */
    // Multiply the loss by the transpose of the input (x^T)*delta
    // This is the derivative of the loss with respect to the weights
    // Should be an outer product between o_i and delta_j
    // Reference for batch_size = 1
    // if (row < rows && col < cols)
    // {
    //     d_Weights[row * cols + col] = output[col] * loss[row];
    // }
    if(row < rows && col < cols){
        T sum = 0;
        for (int k = 0 ; k < batch_size ; k++){
            sum += output[col * batch_size + k] * loss[row * batch_size + k];
        }
        d_Weights[row * cols + col] = sum/batch_size;
    }
    // Sum the loss to get the derivative of the loss with respect to the biases
}


template <typename T>
__global__ void linear_weight_derivatives(T *loss, T *Weights, T *d_F, int rows, int cols, int batch_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // columns
    int batch = blockIdx.y * blockDim.y + threadIdx.y; //batch_size
    //Compute gradient with respect to weights
    // dx = np.dot(dout, self.W.T)  
    //Now we must do this for a loss matrix of size cols x batch_size
    // if (col < cols)
    // {
    //     T sum = 0;
    //     for (int i = 0; i < rows; i++)
    //     {
    //         sum += loss[i] * Weights[col * rows + i]; // Gradient of the loss w.r.t. input x, delta* W^T
    //     }
    //     d_F[col] = sum;
    // }
    if(batch < batch_size && col < cols){
        T sum = 0;
        for (int k = 0 ; k < rows ; k++){
            sum += Weights[k* cols + col] * loss[k* batch_size + batch];
        }
        d_F[col * batch_size + batch] = sum;
    }

}

template <typename T>
__global__ void linear_bias_derivatives(T *loss, T *d_biases, int rows, int batch_size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        T sum = 0;
        for(int i = 0; i < batch_size; i++){
            sum += loss[row * batch_size + i];
        }
        d_biases[row] = sum/batch_size;
    }
}

template <typename T>
void Linear<T>::backward(T *loss)
{
    // Allocate device memory for input, output, weights, and biases
    // We need to take the loss from the previous layer and calculate the derivative of the loss with respect to the input, weights, and biases
    // Then we need to output the next loss for the layers behind this one
    // cout<<"Linear Backwards"<<endl;
    T *d_loss, *d_output, *dev_weights, *dev_biases;
    T *dd_weights, *dd_biases;
    T *d_F;
    // cout<<"Linear Backwards"<<endl;
    int rows = this->rows;
    int cols = this->cols;
    int batch_size = this->batch_size;
    cout<<endl;
    if (!HandleCUDAError(cudaMalloc((void **)&d_loss, rows * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, cols * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dev_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dd_weights, rows * cols * sizeof(T))))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&dd_biases, rows * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_F, cols * batch_size * sizeof(T))))
    {
        cout << "Error in allocating memory for d_F" << endl;
        exit(1);
    }
    if (loss == NULL)
    {
        cout << "Loss is NULL" << endl;
    }
    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_loss, loss, rows * batch_size * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device, Linear Loss" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_weights, this->weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dev_biases, this->biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dd_weights, this->d_weights, rows * cols * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(dd_biases, this->d_biases, rows * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_output,this->input,cols * batch_size * sizeof(T),cudaMemcpyHostToDevice))){
        cout<<"Error in copying output from host to device"<<endl;
        exit(1);
    }

    // Create three streams to (1) Find the weight update (2) Find the bias update (3) Find the next loss

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid_Rows = (rows + threadsPerBlock - 1) / threadsPerBlock;

    // Define the streams
    cudaStream_t stream1, stream2, stream3;
    if (!HandleCUDAError(cudaStreamCreate(&stream1)))
    {
        cout << "Error in creating stream1" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaStreamCreate(&stream2)))
    {
        cout << "Error in creating stream2" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaStreamCreate(&stream3)))
    {
        cout << "Error in creating stream3" << endl;
        exit(1);
    }

    int output_size = rows;
    int input_size = cols;
    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size,1);
    dim3 gridDim((cols + block_size - 1) / block_size, (rows+block_size-1)/block_size, 1);

    dim3 gridDim_Cols((cols + block_size - 1) / block_size, (batch_size+block_size-1)/block_size, 1);
    // Launch the linear derivative kernel
    linear_derivative_kernel<T><<<gridDim, blockDim,0,stream1>>>(d_loss, dd_weights, d_output, rows, cols, batch_size);
    if (!HandleCUDAError(cudaStreamSynchronize(stream1)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_weight_derivatives<T><<<gridDim_Cols,blockDim,0,stream2>>>(d_loss,dev_weights,d_F,rows,cols, batch_size);
    if (!HandleCUDAError(cudaStreamSynchronize(stream2)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    linear_bias_derivatives<T><<<blocksPerGrid_Rows,threadsPerBlock,0,stream3>>>(d_loss,dd_biases,rows, batch_size);
    if (!HandleCUDAError(cudaStreamSynchronize(stream3)))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }


    //Handle the streams destruction
    if(!HandleCUDAError(cudaStreamDestroy(stream1))){
        cout<<"Error in destroying stream1"<<endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaStreamDestroy(stream2))){
        cout<<"Error in destroying stream2"<<endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaStreamDestroy(stream3))){
        cout<<"Error in destroying stream3"<<endl;
        exit(1);
    }

    if (!HandleCUDAError(cudaMemcpy(this->d_weights, dd_weights, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->d_biases, dd_biases, rows * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(this->next_loss, d_F, cols * batch_size * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_loss)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dev_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dd_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(dd_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_F)))
    {
        cout << "Error in freeing d_F" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}



// Assemble network

template <typename T>
Network<T>::Network(int input_size, int *hidden_size, int output_size, int num_layers)
{
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->num_layers = num_layers;
    this->prediction = (T *)malloc(output_size * sizeof(T));
}

template <typename T>
void Network<T>::predict(T *input, T *output)
{   
    int temp = this->batch_size;
    this->batch_size = 1;
    forward(input, output);
    this->batch_size = temp;
}

template <typename T>
void Network<T>::predict(T **input, T **output, int size)
{
    T *prediction = (T *)malloc(output_size * sizeof(T));
    float sum = 0;
    int temp = this->batch_size;
    this->batch_size = 1;
    for(int i=0; i<layers.size(); i++){
        layers[i]->batch_size = 1;
    }
    for (int i = 0; i < size; i++)
    {
        forward(input[i], output[i]);
        // take the hidden output, and measure accuracy
        prediction = layers[layers.size() - 1]->hidden_output;
        for(int j=0; j<output_size; j++){
            cout<<prediction[j]<<" ";
        }
        cout<<endl;
        // Find the max value in the prediction
        int max_index = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (prediction[j] >= prediction[max_index])
            {
                max_index = j;
            }
        }
        // Find the max value in the output
        int max_index_output = 0;
        for (int j = 0; j < output_size; j++)
        {
            if (output[i][j] >= output[i][max_index_output])
            {
                max_index_output = j;
            }
        }
        // check if the max index is the same as the max index output
        if (max_index == max_index_output)
        {
            sum++;
            cout << "Correct Prediction for input " << i << endl;
            cout<<"Prediction: "<<max_index<<" Output: "<<max_index_output<<endl;
        }
        else
        {
            cout << "Incorrect Prediction for input " << i << endl;
            cout<<"Prediction: "<<max_index<<" Output: "<<max_index_output<<endl;
        }
    }
    this->accuracy = sum / size;
    cout << "Accuracy: " << this->accuracy << endl;
    this->batch_size = temp;
    for(int i=0; i<layers.size(); i++){
        layers[i]->batch_size = temp;
    }
}


template <typename T>
void Network<T>::Save_Data_to_CSV(){
    /*Save for later
    Need to save the weights and biases of each layer
    Need to save the activation function of each layer
    Need to save the loss function of each layer
    Need to save the optimizer of the network
    Need to save the learning rate of the network
    Need to save the batch size of the network
    Need to save the number of epochs of the network
    Main things to help differentiate is the compression ratio of the network, the num of ones and zeros saved in each layers loss_data structure*/

    /*Need to create a new name for the folders everytime we run
    Folder Name:(involve the time of the run)
        - Info.csv holds loss function, optimizer, epochs, learning rate, batch size, compression ratio, and accuracy
        - Layer_Info.csv holds the activation function of each layer, as well as Linear layers, with sizes
        - Layer_Weights.csv holds the weights of each layer, as well as the biases, and the num of zeros and ones in the loss_data
    */
    //Create the folder
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string folder_name = "../exp_data/Run_";
    folder_name += this->data_set;
    folder_name += "_";
    folder_name += to_string(1900 + ltm->tm_year);
    folder_name += "_";
    folder_name += to_string(1 + ltm->tm_mon);
    folder_name += "_";
    folder_name += to_string(ltm->tm_mday);
    folder_name += "_";
    folder_name += to_string(ltm->tm_hour);
    folder_name += "_";
    folder_name += to_string(ltm->tm_min);
    folder_name += "_";
    folder_name += to_string(ltm->tm_sec);
    cout<<"Folder Name: "<<folder_name<<endl;
    if(mkdir(folder_name.c_str(),0777) == -1){
        cout<<"Error in creating directory"<<endl;
        exit(1);
    }
    //Create the Info.csv file
    ofstream info_file;
    info_file.open(folder_name+"/Info.csv");
    info_file<<"Loss Function,Optimizer,Epochs,Learning Rate,Batch Size,Compression Ratio,Accuracy"<<endl;
    info_file<<this->loss_function<<","<<this->optim->name<<","<<this->epochs<<","<<this->learning_rate<<","<<this->batch_size<<","<<this->compression_ratio<<","<<this->accuracy<<endl;
    info_file.close();

    //Create the Layer_Info.csv file
    ofstream layer_info_file;
    layer_info_file.open(folder_name+"/Layer_Info.csv");
    layer_info_file<<"Layer,Name,Rows,Cols"<<endl;
    for(int i=0; i<layers.size(); i++){
        layer_info_file<<i<<","<<layers[i]->name<<","<<layers[i]->rows<< "," <<layers[i]->cols<<endl;
    }
    layer_info_file.close();

    //Create the Layer_Weights.csv file
    ofstream layer_weights_file;
    layer_weights_file.open(folder_name+"/Layer_Weights.csv");
    layer_weights_file<<"Layer,Row,Col,Value,Num_Zeros,Num_Ones"<<endl;
    //Can only do this for the linear layers
    for (int i = 0; i < layerMetadata.size(); i++)
    {
        // Validate layerNumber is within bounds
        if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
        {
            // Check if the layer pointer is not null
            if (this->layers[layerMetadata[i].layerNumber] != nullptr)
            {
                // Check if the current layer is marked as updateable
                if (layerMetadata[i].isUpdateable)
                {
                    for(int j=0; j<layers[layerMetadata[i].layerNumber]->rows; j++){
                        for(int k=0; k<layers[layerMetadata[i].layerNumber]->cols; k++){
                            layer_weights_file<<layerMetadata[i].layerNumber<<","<<j<<","<<k<<","<<layers[layerMetadata[i].layerNumber]->weights[j*layers[layerMetadata[i].layerNumber]->cols + k]<<","<<epochs-layers[layerMetadata[i].layerNumber]->num_ones[j*layers[layerMetadata[i].layerNumber]->cols + k]<<","<<layers[layerMetadata[i].layerNumber]->num_ones[j*layers[layerMetadata[i].layerNumber]->cols + k]<<endl;
                        }
                    }
                    //Save the biases
                    for(int j=0; j<layers[layerMetadata[i].layerNumber]->rows; j++){
                            layer_weights_file<<layerMetadata[i].layerNumber<<","<<j<<","<<layers[layerMetadata[i].layerNumber]->cols<<","<<layers[layerMetadata[i].layerNumber]->biases[j]<<","<<epochs-layers[layerMetadata[i].layerNumber]->num_ones[j*layers[layerMetadata[i].layerNumber]->cols + layers[layerMetadata[i].layerNumber]->cols]<<","<<layers[layerMetadata[i].layerNumber]->num_ones[j*layers[layerMetadata[i].layerNumber]->cols + layers[layerMetadata[i].layerNumber]->cols]<<endl;
                    }
                }
            }
        }
    }
}

template <typename T>
void Network<T>::set_input_size(int input_size)
{
    this->input_size = input_size;
}

template <typename T>
void Network<T>::set_hidden_size(int *hidden_size)
{
    this->hidden_size = (T *)malloc((num_layers - 2) * sizeof(T));
    this->hidden_size = hidden_size;
}

template <typename T>
void Network<T>::set_output_size(int output_size)
{
    this->output_size = output_size;
}

template <typename T>
void Network<T>::set_num_layers(int num_layers)
{
    this->num_layers = num_layers;
}

template <typename T>
int Network<T>::get_input_size()
{
    return input_size;
}

template <typename T>
int *Network<T>::get_hidden_size()
{
    return hidden_size;
}

template <typename T>
int Network<T>::get_output_size()
{
    return output_size;
}


template <typename T>
struct CompareBernoulliLayers {
    __host__ __device__
    bool operator()(const Loc_Layer<T>& lhs, const Loc_Layer<T>& rhs) const {
        // Assuming Loc_Layer has a member function or variable to get the weight
        return lhs.layer > rhs.layer; // Sort in ascending order
    }
};




template <typename T>
void Network<T>::update_weights(T learning_rate, int epochs, int Q, int total_epochs)
{
    // Ensure layers vector is not empty and is properly initialized
    if (this->layers.empty())
    {
        std::cerr << "Error: Layers vector is empty.\n";
        return;
    }
    if(this->optim->name == "AdamDecay" || this->optim->name == "SGDMomentumJenks"){
        // find_Loss_Metric_Jenks_Aggressive();
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        //Can we change this to find the cutoff for weights and biases seperately?
                        //We need a function to aggregate the loss over time
                        this->layers[layerMetadata[i].layerNumber]->find_Loss_Metric_Jenks_Aggressive();  
                    }
                }
            }
        }
    }
    if(this->optim->name == "AdamWBernoulli"){
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        this->layers[layerMetadata[i].layerNumber]->find_Loss_Metric();
                        Fill_Bern(this->layers[layerMetadata[i].layerNumber], layerMetadata[i].LinNumber);
                        
                    }
                }
            }
        }
        thrust::host_vector<Loc_Layer<T>> res = flatten();
        thrust::sort(res.begin(), res.end(), CompareBernoulliWeights<T>());
        thrust::sort(res.begin(), res.begin()+Q, CompareBernoulliLayers<T>());
        //Use the column and rows to set Bernoulli
        for(int i = 0; i < Q; i++){
            this->layers[res[i].layer]->set_Bernoulli(res[i].row, res[i].col);
        }
        
        
    }
    if(this->optim->name == "AdamJenks" || this->optim->name == "SGDJenks"){
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        //Can we change this to find the cutoff for weights and biases seperately?
                        //We need a function to aggregate the loss over time
                        this->layers[layerMetadata[i].layerNumber]->Agg_Jenks_Loss();
                        if(epochs == total_epochs/2){
                            this->layers[layerMetadata[i].layerNumber]->find_Loss_Metric_Jenks_Aggressive_Single();  
                        }   
                    }
                }
            }
        }
    }
    // this->learning_rate = learning_rate/2;

    // Iterate over each entry in the layerMetadata vector
    for (int i = 0; i < layerMetadata.size(); i++)
    {
        // Validate layerNumber is within bounds
        if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
        {
            // Check if the layer pointer is not null
            if (this->layers[layerMetadata[i].layerNumber] != nullptr)
            {
                // Check if the current layer is marked as updateable
                if (layerMetadata[i].isUpdateable)
                {
                    // Call the update_weights method on the updateable layer
                    if(this->optim->name == "SGD"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_SGD(learning_rate);
                    }
                    else if(this->optim->name == "Adam"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_Adam(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                    }
                    else if(this->optim->name == "RMSProp"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_RMSProp(learning_rate, this->optim->decay_rate);
                    }
                    else if(this->optim->name == "Momentum"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_Momentum(learning_rate, this->optim->momentum);
                    }
                    else if(this->optim->name == "AdamWBernoulli"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_AdamWBernoulli(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli();
                    }
                    else if(this->optim->name == "AdamActiv"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_AdamActiv(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                        this->layers[layerMetadata[i].layerNumber]->Fill_Activ();
                    }
                    else if(this->optim->name == "AdamJenks"){
                        if(epochs==0){
                            this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();
                        }
                        this->layers[layerMetadata[i].layerNumber]->update_weights_AdamJenks(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                        // this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();
                    }
                    else if(this->optim->name == "SGDJenks"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_SGDJenks(learning_rate);
                        // this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli();
                    }
                    else if(this->optim->name == "AdamDecay"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_AdamDecay(learning_rate, this->optim->beta1, this->optim->beta2, this->optim->epsilon, epochs);
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();
                    }
                    else if (this->optim->name == "SGDMomentumJenks"){
                        this->layers[layerMetadata[i].layerNumber]->update_weights_SGDMomentum_Jenks(learning_rate, this->optim->momentum);
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();
                    }
                    else{
                        cout<<"Optimizer not found"<<endl;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Layer at index " << layerMetadata[i].layerNumber << " is a null pointer.\n";
            }
        }
        else
        {
            std::cerr << "Error: layerNumber out of bounds: " << layerMetadata[i].layerNumber << "\n";
        }
    }
}



template <typename T>
int argmax(T *arr, int size)
{
     return static_cast<int>(std::distance(arr, max_element(arr, arr+size)));
}

template <typename T>
void Network<T>::train(T *input, T *output, int epochs, T learning_rate)
{
    for (int i = 0; i < epochs; i++)
    {
        cout << "Epoch: " << i << endl;
        forward(input, output);

        backward(input, output);

        update_weights(learning_rate);
    }
    for (int i = 0; i < output_size; i++)
    {
        cout << layers[layers.size() - 1]->hidden_output[i] << " ";
    }
    cout << endl;
    memcpy(this->prediction, layers[layers.size() - 1]->hidden_output, output_size * sizeof(T));
}

template <typename T>
void Network<T>::train(T **input, T **output, int epochs, T learning_rate, int size)
{
    // Find a random list of indices for the batch size
    //  Create a thrust vector of indices
    this->learning_rate = learning_rate;
    this->epochs = epochs;
    int* indices = (int*)malloc(batch_size*sizeof(int));
    // Fill the vector with random_indices
    // Iterate through the indices and train the network
    int pred_idx = 0;
    int gt_idx = 0;
    int sum = 0;
    T* batch_input = (T*)malloc(input_size*batch_size*sizeof(T));
    T* batch_output = (T*)malloc(output_size*batch_size*sizeof(T));
    if(this->optim->name == "AdamJenks" || this->optim->name == "SGDJenks" || this->optim->name == "SGDMomentumJenks"){
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        //Something to this nature
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();    
                    }
                }
            }
        }
    }
    if(this->optim->name == "AdamDecay"){
        for (int i = 0; i < layerMetadata.size(); i++)
        {
            // Validate layerNumber is within bounds
            if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
            {
                // Check if the layer pointer is not null
                if (this->layers[layerMetadata[i].layerNumber] != nullptr)
                {
                    // Check if the current layer is marked as updateable
                    if (layerMetadata[i].isUpdateable)
                    {
                        //Something to this nature
                        this->layers[layerMetadata[i].layerNumber]->Fill_Loss_data(); 
                        this->layers[layerMetadata[i].layerNumber]->Fill_Bernoulli_Ones();    
                    }
                }
            }
        }
    }
    for (int i = 0; i < epochs; i++)
    {
        cout<< "Epoch: " << i << endl;
        for (int k = 0; k < batch_size; k++)
        {
            indices[k] = rand() % size;
        }
        Format_Batch_Data(input,output,batch_input,batch_output,indices,batch_size,input_size,output_size);
        forward(batch_input, batch_output);
        backward(batch_input, batch_output);
        update_weights(learning_rate, i, this->Q, epochs);
    }
    // If the optimizer is AdamJenks, we need to prune the weight at the end of the training
    // if(this->optim->name == "AdamJenks"){
    //     for (int i = 0; i < layerMetadata.size(); i++)
    //     {
    //         // Validate layerNumber is within bounds
    //         if (layerMetadata[i].layerNumber >= 0 && layerMetadata[i].layerNumber < this->layers.size())
    //         {
    //             // Check if the layer pointer is not null
    //             if (this->layers[layerMetadata[i].layerNumber] != nullptr)
    //             {
    //                 // Check if the current layer is marked as updateable
    //                 if (layerMetadata[i].isUpdateable)
    //                 {
    //                     //Something to this nature
    //                     this->layers[layerMetadata[i].layerNumber]->find_Loss_Metric_Jenks_Prune();     
    //                 }
    //             }
    //         }
    //     }
    // }
}


template <typename T>
void Conv2D<T>::set_weights(T *weights, T *biases)
{
    this->weights;
    this->biases;
}

template <typename T>
void Conv2D<T>::set_stride(int stride)
{
    this->stride = stride;
}

template <typename T>
void Conv2D<T>::set_padding(int padding)
{
    this->padding = padding;
}

template <typename T>
int Conv2D<T>::get_rows()
{
    return this->rows;
}

__global__ void d_Gauss_Filter(unsigned char *in, unsigned char *out, int h, int w)
{
    // __shared__ unsigned char in_shared[16][16];
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int idx = y * (w - 2) + x;
    if (x < (w - 2) && y < (h - 2))
    {
        out[idx] += .0625 * in[y * w + x];
        out[idx] += .125 * in[y * w + x + 1];
        out[idx] += .0625 * in[y * w + x + 2];
        out[idx] += .125 * in[(y + 1) * w + x];
        out[idx] += .25 * in[(y + 1) * w + x + 1];
        out[idx] += .125 * in[(y + 1) * w + x + 2];
        out[idx] += .0625 * in[(y + 2) * w + x];
        out[idx] += .125 * in[(y + 2) * w + x + 1];
        out[idx] += .0625 * in[(y + 2) * w + x + 2];
    }
}

#define TILE_WIDTH 16
__global__ void d_Gauss_Filter_v2(unsigned char *in, unsigned char *out, int h, int w)
{
    // __shared__ unsigned char in_shared[16][16];
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int idx = y * (w - 2) + x;
    __shared__ unsigned char in_s[TILE_WIDTH + 2][TILE_WIDTH + 2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (x < w && y < h)
    {
        in_s[ty][tx] = in[y * w + x];
        if (ty >= TILE_WIDTH - 2)
        {
            in_s[ty + 2][tx] = in[(y + 2) * w + x];
        }
        if (tx >= TILE_WIDTH - 2)
        {
            in_s[ty][tx + 2] = in[y * w + x + 2];
        }
        if (tx == TILE_WIDTH - 1 && ty == TILE_WIDTH - 1)
        {
            in_s[ty + 1][tx + 1] = in[(y + 1) * w + x + 1];
            in_s[ty + 1][tx + 2] = in[(y + 1) * w + x + 2];
            in_s[ty + 2][tx + 1] = in[(y + 2) * w + x + 1];
            in_s[ty + 2][tx + 2] = in[(y + 2) * w + x + 2];
        }
    }
    __syncthreads();
    if (x < (w - 2) && y < (h - 2))
    {
        out[idx] += .0625 * in_s[ty][tx];
        out[idx] += .125 * in_s[ty][tx + 1];
        out[idx] += .0625 * in_s[ty][tx + 2];
        out[idx] += .125 * in_s[ty + 1][tx];
        out[idx] += .25 * in_s[ty + 1][tx + 1];
        out[idx] += .125 * in_s[ty + 1][tx + 2];
        out[idx] += .0625 * in_s[(ty + 2)][tx];
        out[idx] += .125 * in_s[(ty + 2)][tx + 1];
        out[idx] += .0625 * in_s[(ty + 2)][tx + 2];
    }
}

template <typename T>
__global__ void conv2D_kernel(T *input, T *output, T *weights, T *biases, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int stride, int batch_size)
{
    /*The input has the shape of (batch_size, channel, height, width)
    The weights have the shape (filters, channel, filter_height, filter_width)
    The output has the shape of (batch_size, filter, output_height, output_width)
    We will have the dimy and dimx operate on the single image channel, i.e. deal with height and width
    For the sake of computatation, we will use the z dim to deal with batches and have them process each channel */
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    if(outCol < out_width && outRow < out_height && batch < batch_size){
        for (int filter = 0; filter < filters; filter++)
        {
            T sum = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                for (int i = 0; i < kernel_height; i++)
                {
                    for (int j = 0; j < kernel_width; j++)
                    {
                        int inRow = outRow * stride + i;
                        int inCol = outCol * stride + j;
                        // printf("Input[%d][%d][%d][%d]: \n", batch, channel, inRow, inCol);
                        // printf("Weights[%d][%d][%d][%d]: \n", filter, channel, i, j);
                        sum += input[batch * channels * width * height + channel * width * height + inRow * width + inCol] * weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + i * kernel_width + j];
                    }
                }
            }
            output[batch * filters * out_width * out_height + filter * out_width * out_height + outRow * out_width + outCol] = sum + biases[filter];
        }
    }
}


template <typename T>
__global__ void conv2D_weight_update_kernel(T *input, T* this_loss, T* d_weights, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int batch_size)
{
    /*This kernel is tasked with finding dw for a convolutional layer
    The equation which will be used is:
    dW[filter,channel,k,m]= 1/(batch_size)\sum_{n=0}^{batch_size}\sum_{i=0}^{kernel_height}\sum_{j=0}^{kernel_width}x_n[channel,i+k,j+m]*thisloss_n[f,i,j]
    We need to dilate the current loss according to the stride*/
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(filter < filters && channel < channels && k < kernel_height){
        T sum = 0;
        for(int m=0; m<kernel_width; m++){
            sum = 0;
            for(int batch = 0; batch < batch_size; batch++){
            //This is the sum over the batch
                for(int i = 0; i<out_height; i++){
                    //This is the sum over the height, and we will have i+k for the input, and i for the loss
                    for(int j = 0; j<out_width; j++){
                        //This is the sum over the width, and we will have j+m for the input, and j for the loss
                            sum += input[batch * channels * width * height + channel * width * height + (i+k) * width + j+m] * this_loss[batch * filters * out_width * out_height + filter * out_width * out_height + i * out_width + j];
                    }
                }
            }
            d_weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + k * kernel_width + m] = sum/batch_size;
        }
    }
    __syncthreads();
    //How does stride play a role?
}


template <typename T>
__global__ void conv2D_biases_update_kernel(T *this_loss, T* d_biases, int filters, int out_width, int out_height, int batch_size)
{
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    if(filter < filters){
        T sum = 0;
        for(int batch = 0; batch < batch_size; batch++){
            for(int i = 0; i<out_height; i++){
                for(int j = 0; j<out_width; j++){
                    sum += this_loss[batch * filters * out_width * out_height + filter * out_width * out_height + i * out_width + j];
                }
            }
        }
        d_biases[filter] = sum/batch_size;
    }
}

template <typename T>
__global__ void conv2D_next_loss_kernel(T *weights, T *this_loss, T *next_loss, int channels, int filters, int kernel_width, int kernel_height, int width, int height, int out_width, int out_height, int stride, int batch_size)
{
    /*We need to find dx_n, where n corresponds to the batch*/
    /*This will have the form of dy_n * w^T, where dy_n is padded
    dy_n is padded according to the stride in the forward pass, namely, the amount of padding on the top/bottom and left/right is equal tot the strides
    This allows us to recover the input size*/ 
    /*The loss will be for each batch and channel- it will have the same shape as the input to this layer*/
    int inCol = blockIdx.x * blockDim.x + threadIdx.x; //inCol w.r.t. the input size to this layer in the forward pass
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z*blockDim.z + threadIdx.z;
    if(inCol < width && inRow < height && batch < batch_size){
        for(int channel = 0; channel < channels; channel++){
            for(int filter = 0; filter < filters; filter++){
                for(int i = 0; i < kernel_height; i++){
                    for(int j = 0; j < kernel_width; j++){
                        int outRow = inRow - i;
                        int outCol = inCol - j;
                        if(outRow >= 0 && outRow < out_height && outCol >= 0 && outCol < out_width){
                            next_loss[batch * channels * width * height + channel * width * height + inRow * width + inCol] += weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + i * kernel_width + j] * this_loss[batch * filters * out_width * out_height + filter * out_width * out_height + outRow * out_width + outCol];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ void conv2D_rotate_filter(T *weights, T *weights_rot, int channels, int filters, int kernel_width, int kernel_height)
{
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if(filter < filters && channel < channels && i < kernel_height){
        for(int j = 0; j < kernel_width; j++){
            weights_rot[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + i * kernel_width + j] = weights[filter * channels * kernel_height * kernel_width + channel * kernel_height * kernel_width + (kernel_height - i - 1) * kernel_width + (kernel_width - j - 1)];
        }
    }
}

template <typename T>
__global__ void conv2D_dilate_loss(T* loss, T* loss_padded, int batch_size, int channels, int width, int height, int dilate_width, int dilate_height){
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int chan = blockIdx.z * blockDim.z + threadIdx.z;
    int outwidth = width*(dilate_width+1)-dilate_width;
    int outheight = height*(dilate_height+1)-dilate_height;
    int outRow = inRow*(dilate_height+1);
    int outCol = inCol*(dilate_width+1);
    if(inCol < width && inRow < height && chan < channels){
        for(int i = 0; i<batch_size;i++){
            loss_padded[i * channels * outwidth * outheight + chan * outwidth * outheight + outRow*outwidth + outCol] = loss[i * channels * width * height + chan * width * height + inRow * width + inCol];
        }
    }
}

template <typename T>
__global__ void conv2D_dilate_pad_loss(T* loss, T* loss_padded, int batch_size, int channels, int width, int height, int dilate_width, int dilate_height, int pad_width, int pad_height){
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int chan = blockIdx.z * blockDim.z + threadIdx.z;
    int outwidth = width*(dilate_width+1)-dilate_width + 2*pad_width;
    int outheight = height*(dilate_height+1)-dilate_height + 2*pad_height;
    int outRow = inRow*(dilate_height+1) + pad_height;
    int outCol = inCol*(dilate_width+1) + pad_width;
    if(inCol < width && inRow < height && chan < channels){
        for(int i = 0; i<batch_size;i++){
            loss_padded[i * channels * outwidth * outheight + chan * outwidth * outheight + outRow*outwidth + outCol] = loss[i * channels * width * height + chan * width * height + inRow * width + inCol];
            //double check
        }
    }
}



template <typename T>
void Conv2D<T>::forward(T *input, T *output)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    if(input == nullptr){
        cout<<"Input is null"<<endl;
        exit(1);
    }
    printCudaMemoryUsage();
    cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // if (!prop.canMapHostMemory)
    //     cout << "Device cannot map host memory" << endl;
    // cudaSetDeviceFlags(cudaDeviceMapHost);
    cout<<"The width is "<<this->width<<endl;
    cout<<"The height is "<<this->height<<endl;
    int input_size = this->batch_size * this->width * this->height * this->channels * sizeof(T);
    cout<<"The size of the input is "<<this->batch_size * this->width * this->height * this->channels * sizeof(T)<<endl;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size)))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    int output_size = this->batch_size * this->output_width * this->output_height * this->filters * sizeof(T);
    printCudaMemoryUsage();
    cout<< "The size of the output is "<<putput_size<<endl;
    cout<< "Output width is "<<output_width<<endl;
    cout<< "Output height is "<<output_height<<endl;
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size)))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    printCudaMemoryUsage();
    int weight_size = this->filters * this->kernel_width * this->kernel_height * this->channels * sizeof(T);
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, weight_size)))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, this_.filters * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if(weights == nullptr){
        cout<<"Weights is null"<<endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, this->weights, weight_size, cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, this->biases, this->filters * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }

    // Define grid and block dimensions
    int TPB = 8;
    dim3 blockDim(TPB, TPB, TPB);
    dim3 gridDim((this->output_width + TPB - 1) / TPB,(this->output_height + TPB - 1) / TPB, (this->batch_size + TPB - 1) / TPB);
    conv2D_kernel<T><<<gridDim,blockDim>>>(d_input, d_output, d_weights, d_biases, this->channels, this->filters, this->kernel_width, this->kernel_height, this->width, this->height, this->output_width, this->output_height, this->stride, this->batch_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
void Conv2D<T>::backward(T * loss)
{
    // Allocate device memory for input, output, weights, and biases
    T *d_input, *d_output, *d_weights, *d_biases;
    T *d_dweights, *d_dbiases, *d_dinput;
    T* d_loss, *d_fin_loss;
    T* d_temp_loss_dilate, *d_temp_loss_d_pad, *d_w_prime;
    int loss_size = this->batch_size * this->output_width * this->output_height * this->filters * sizeof(T);
    int fin_loss_size = this->batch_size * this->output_width * this->output_height * this->channels * sizeof(T);
    int temp_loss_dilate_size = this->batch_size * (this->output_width+(this->stride-1)) * (this->output_height+(this->stride-1)) * this->filters * sizeof(T);
    int temp_loss_pad_dilate_size = this->batch_size * (this->output_width+(this->stride-1)+2*(this->kernel_width-1)) * (this->output_height+(this->stride-1)+2*(this->kernel_width-1)) * this->filters * sizeof(T);
    int input_size = this->batch_size * this->width * this->height * this->channels * sizeof(T);
    int output_size = this->batch_size * this->output_width * this->output_height * this->filters * sizeof(T);
    int weight_size = this->filters * this->kernel_width * this->kernel_height * this->channels * sizeof(T);
    if(!HandleCUDAError(cudaMalloc((void **)&d_loss, loss_size))){
        cout << "Error in allocating memory for d_loss" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_fin_loss, fin_loss_size))){
        cout << "Error in allocating memory for d_fin_loss" << endl;
        exit(1);
    } 
    if(!HandleCUDAError(cudaMalloc((void **)&d_temp_loss_dilate, temp_loss_dilate_size))){
        cout << "Error in allocating memory for d_temp_loss_dilate" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_temp_loss_d_pad, temp_loss_pad_dilate_size))){
        cout << "Error in allocating memory for d_temp_loss_d_pad" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, input_size)))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, output_size)))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_weights, weight_size)))
    {
        cout << "Error in allocating memory for d_weights" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_w_prime, weight_size))){
        cout << "Error in allocating memory for d_w_prime" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMalloc((void **)&d_biases, this->filters * sizeof(T))))
    {
        cout << "Error in allocating memory for d_biases" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_dweights, weight_size))){
        cout << "Error in allocating memory for d_dweights" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_dbiases, this->filters * sizeof(T)))){
        cout << "Error in allocating memory for d_dbiases" << endl;
        exit(1);
    }

    // Copy input, weights, and biases from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_weights, weights, weight_size, cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying weights from host to device" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, biases, this->filters * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying biases from host to device" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMemcpy(d_fin_loss, loss, loss_size, cudaMemcpyHostToDevice))){
        cout << "Error in copying loss from host to device" << endl;
        exit(1);
    }

    /*Need to dilate the loss, as well as dilate and pad the loss for different components of the backward pass
    
    conv2D_dilate_loss(T* loss, T* loss_padded, int batch_size, int channels, int width, int height, int dilate_width, int dilate_height)
    conv2D_dilate_pad_loss(T* loss, T* loss_padded, int batch_size, int channels, int width, int height, int dilate_width, int dilate_height, int pad_width, int pad_height)
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;


    conv2D_rotate_filter(T *weights, T *weights_rot, int channels, int filters, int kernel_width, int kernel_height)

    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    
    */
    int TPB_padding = 8;
    dim3 blockDim_padding(TPB_padding, TPB_padding, TPB_padding);
    dim3 gridDim_padding((this->output_width + TPB_padding - 1) / TPB_padding,(this->output_height + TPB_padding - 1) / TPB_padding, (this->channels + TPB_padding - 1) / TPB_padding);
    dim3 blockDim_rotate(TPB_padding, TPB_padding, TPB_padding);
    dim3 gridDim_rotate((this->filters+ TPB_padding - 1) / TPB_padding, (this->channels + TPB_padding - 1) / TPB_padding, (this->kernel_height + TPB_padding - 1) / TPB_padding);
    conv2D_dilate_loss<T><<<gridDim_padding, blockDim_padding>>>(d_fin_loss, d_temp_loss_dilate, this->batch_size, this->channels, this->output_width, this->output_height, this->stride, this->stride);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
    conv2D_dilate_pad_loss<T><<<gridDim_padding, blockDim_padding>>>(d_fin_loss, d_temp_loss_d_pad, this->batch_size, this->channels, this->output_width, this->output_height, this->stride, this->stride, this->kernel_width-1, this->kernel_height-1);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
    conv2D_rotate_filter<T><<<gridDim_rotate, blockDim_rotate>>>(d_weights, d_w_prime, this->channels, this->filters, this->kernel_width, this->kernel_height);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
    //Define the grid dimensions
    /* For the weight update kernel
    int filter = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;*/

    int tpb_bias = this->filters;
    dim3 blockDim_bias(tpb_bias);
    dim3 gridDim_bias((this->filters + tpb_bias - 1) / tpb_bias);
    int TPB = 8;
    dim3 blockDim(TPB, TPB, TPB);
    dim3 gridDim((this->filters + TPB - 1) / TPB,(this->channels + TPB - 1) / TPB, (this->height + TPB - 1) / TPB);
    dim3 blockDim_nextloss(TPB, TPB, TPB);
    dim3 gridDim_nextloss((width + TPB - 1) / TPB,(height + TPB - 1) / TPB, (batch_size + TPB - 1) / TPB);
    /* For Next loss 
    int inCol = blockIdx.x * blockDim.x + threadIdx.x; //inCol w.r.t. the input size to this layer in the forward pass
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z*blockDim.z + threadIdx.z;*/
    conv2D_next_loss_kernel<T><<<gridDim_nextloss, blockDim_nextloss>>>(d_w_prime, d_loss, d_temp_loss_d_pad, channels, filters, kernel_width, kernel_height, width, height, output_width, output_height, stride, batch_size);   
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }

    if(!HandleCUDAError(cudaMemcpy(this->next_loss, d_loss, batch_size * width * height * channels * sizeof(T), cudaMemcpyDeviceToHost))){
        cout << "Error in copying fin_loss from host to device" << endl;
        exit(1);
    }
    // Copy the gradients from device to host
    if (!HandleCUDAError(cudaMemcpy(d_weights, d_dweights, filters * kernel_width * kernel_height * channels * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dweights from device to host" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaMemcpy(d_biases, d_dbiases, filters * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying dbiases from device to host" << endl;
        exit(1);
    }


    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_weights)))
    {
        cout << "Error in freeing d_weights" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_biases)))
    {
        cout << "Error in freeing d_biases" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}

template <typename T>
Conv2D<T>::update_weights(){
    conv2D_weight_update_kernel<T><<<gridDim, blockDim>>>(d_input, d_temp_loss_dilate, d_dweights, channels, filters, kernel_width, kernel_height, width, height, output_width, output_height, batch_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
    /*For the bias update
    int filter = blockIdx.x * blockDim.x + threadIdx.x;*/
    conv2D_biases_update_kernel<T><<<gridDim_bias, blockDim_bias>>>(d_temp_loss_d_pad, d_dbiases, filters, output_width, output_height, batch_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error in running kernel"<<endl;
        exit(1);
    }
}



template <typename T>
__global__ void max_pooling_kernel(T *input, T *output, int kernel_width, int kernel_height, int width, int height, int output_width, int output_height, int batch_size, int padding, int channels, int* coordinates)
{
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int z = threadIdx.z + (blockIdx.z * blockDim.z);
    /*x corresponds to the column, y corresponds to the row, and z corresponds to the channel and batch*/
    if (x < output_width && y < output_height && z < channels)
    {
        for(int batch = 0; batch < batch_size; batch++){
            int idx = batch * channels * width * height + z * width * height + y * width + x;
            int out_idx = batch * channels * output_width * output_height + z * output_width * output_height + y * output_width + x;
            T max = input[idx];
            int coord = 0;
            for (int i = 0; i < kernel_height; i++)
            {
                for (int j = 0; j < kernel_width; j++)
                {
                    if(input[batch * channels * width * height + z * width * height + (y + i) * width + x + j] > max){
                        max = input[batch * channels * width * height + z * width * height + (y + i) * width + x + j];
                        coord = batch * channels * width * height + z * width * height + (y + i) * width + x + j;
                    }
                }
            }
            output[out_idx] = max;
            coordinates[out_idx] = coord;
        }
    }
}


template <typename T>
void MaxPooling2D<T>::forward(T *input, T *output)
{
    // Allocate device memory for input and output
    T *d_input, *d_output;
    int *d_coordinates;
    if (!HandleCUDAError(cudaMalloc((void **)&d_input, batch_size * width * height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_input" << endl;
        exit(1);
    }
    printCudaMemoryUsage();
    if (!HandleCUDAError(cudaMalloc((void **)&d_output, batch_size * output_width * output_height * channels * sizeof(T))))
    {
        cout << "Error in allocating memory for d_output" << endl;
        exit(1);
    }
    if(!HandleCUDAError(cudaMalloc((void **)&d_coordinates, batch_size * output_width * output_height * channels * sizeof(int)))){
        cout << "Error in allocating memory for d_coordinates" << endl;
        exit(1);
    }
    // Copy input from host to device
    if (!HandleCUDAError(cudaMemcpy(d_input, input, batch_size * width * height * channels * sizeof(T), cudaMemcpyHostToDevice)))
    {
        cout << "Error in copying input from host to device" << endl;
        exit(1);
    }

    /*The equation for this is: out(N,C,h,w)= max_{0...kernel_H}max_{0...kernel_W}input(N,C,stride[0]xh+m,stride[1]xw+n)*/

    int TPB = 8;
    dim3 blockDim(TPB, TPB, TPB);
    dim3 gridDim((output_height + TPB - 1) / TPB,(output_width + TPB - 1) / TPB, (channels + TPB - 1) / TPB);
    // Launch the max pooling kernel
    max_pooling_kernel<T><<<gridDim, blockDim>>>(input, output, kernel_width, kernel_height, width, height, output_width, output_height, batch_size, padding, channels, d_coordinates);
    if (!HandleCUDAError(cudaDeviceSynchronize()))
    {
        cout << "Error in synchronizing device" << endl;
        exit(1);
    }
    // Copy the result output from device to host
    if (!HandleCUDAError(cudaMemcpy(output, d_output, batch_size * output_width * output_height * channels * sizeof(T), cudaMemcpyDeviceToHost)))
    {
        cout << "Error in copying output from device to host" << endl;
        exit(1);
    }

    // Free device memory
    if (!HandleCUDAError(cudaFree(d_input)))
    {
        cout << "Error in freeing d_input" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaFree(d_output)))
    {
        cout << "Error in freeing d_output" << endl;
        exit(1);
    }
    if (!HandleCUDAError(cudaDeviceReset()))
    {
        cout << "Error in resetting device" << endl;
        exit(1);
    }
}