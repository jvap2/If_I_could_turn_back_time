#include "../include/include.h"
#include "../include/layers.h"

// Function to initialize the weights and biases of the network

template <typename T>
Matrix<T>:: Matrix()
{
    this->d_data = (T*)malloc(rows*cols*sizeof(T));
    for(int i=0; i<rows*cols; i++)
    {
        this->d_data[i] = (T)rand()/(T)RAND_MAX;
    }
}

template <typename T>
void Matrix<T>::set_rows(int rows)
{
    this->rows = rows;
}

template <typename T>
void Matrix<T>::set_cols(int cols)
{
    this->cols = cols;
}

template <typename T>
int Matrix<T>::get_rows()
{
    return this->rows;
}

template <typename T>
int Matrix<T>::get_cols()
{
    return this->cols;
}
