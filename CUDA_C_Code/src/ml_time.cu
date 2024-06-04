#include "../include/include.h"


__global__ void MatMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	if (row < ny && col < nx)
	{
		for (int k = 0; k < nx; k++)
		{
			fSum += g_A[row * nx + k] * g_B[k * nx + col];
		}
		g_C[row * nx + col] = fSum;
	}
}

//Tiled Kernel
#define TILE_WIDTH 32
__global__ void TiledMult(float* g_A, float* g_B, float* g_C, const int Width)
{
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix A
	//We do not need to access the static array with one index, memory is stored as matrix and since static. If it is dynamic, we have to access with 1D
	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix B
	//TILE_WIDTH is equal to block width since this is square matrices
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];//Ads and Bds are the portions of data we are going to allocate into the shared mem
	//Shared memory is only around for the lifetime of a block. Once evicted, shared memory is gone
	// If we have 4 blocks running on an SM, there are 4 copies of Ads and Bds on SM
	//Write code to store locally the thread and block indices
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//This is for ease of typing these- we are wasting registers, but for this problem it won't matter
	//Compute the row and column indices of the C Element to store the dot product result
	int row = ty + (by * TILE_WIDTH);
	int col = tx + (bx * TILE_WIDTH);//Note the difference, the width here is tile_width. In the naive, we use blockDim as the width

	//Loop over the A and B tiles to compute the C Element
	float cValue = 0.0f;
	for (int ph{}; ph < Width / TILE_WIDTH;++ph) {
		//Load A and B tiles into the shared memory collaboratively
		//Note, we are shifting by ph*TILE_WIDTH so when the next phase happens, we can get one TILE_WIDTH away
		Ads[ty][tx] = g_A[row * Width + ph * TILE_WIDTH + tx];//Add tx to get the column in a tile
		Bds[ty][tx] = g_B[(ph * TILE_WIDTH + ty) * Width+col];//The term in () shifts us down the column, we are using column to shift across the tile horizontally
		//One thread gets one element
		//col is within the size of the tile width, so we will stay within a tile as this increments
		//This is called dynamic core model
		//Wait for threads of the block to fetch their specific element of block (TILE_WIDTH) to complete loading to shared memory
		__syncthreads(); 
		//Perform the partial dot product in the phase
		for (int k{}; k < TILE_WIDTH;k++) {
			cValue += Ads[ty][k] * Bds[k][tx];
			//We access A in a coalesced access in the shared memory
			//We are only doing this across the tile
		}
		//Wait for all threads in the block to complete partial dot product in a phase
		__syncthreads();
	}
	//We have now finished the dot product
	g_C[row * Width + col] = cValue;
}
