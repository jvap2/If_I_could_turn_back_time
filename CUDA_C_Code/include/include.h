#pragma once



__global__ void TiledMult(float* g_A, float* g_B, float* g_C, const int Width);
__global__ void MatMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx);

