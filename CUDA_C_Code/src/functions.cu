#include "../include/include.h"

void InitializeMatrix(float *matrix, int ny, int nx)
{
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += nx;
	}
}

void ZeroMatrix(float *temp, const int ny, const int nx)
{
	float *p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = 0.0f;
		}
		p += nx;
	}
}

void InitializeVector(float* vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}

void ZeroVector(float* vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = 0.0f;
	}
}