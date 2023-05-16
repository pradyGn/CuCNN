#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void transpose(float *input, float* output, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            output[i * N + j] = input[j * M + i];
        }
    }
}

__global__ void transpose_cuda(float *input, float* output){
    int i = threadIdx.x;

    for (int j = 0; j < dense_output_M; j++){
        output[i * dense_output_M + j] = input[j * (output_M * output_M) + i];
    }
    
}