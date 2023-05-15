#include "constants.h"
#include <cuda.h>
#include <stdlib.h>


void initialize_dense_weights_and_bias(float* weights, float* bias){
for (int i = 0; i < output_M; i++) {
    for(int j = 0; j < output_N; j++) {
        weights[i*N + j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    
    bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
    }
}

void initialize_dense_output(float* output){
for (int i = 0; i < output_M; i++) {
        output[i] = 0.0f;
    }
}



__global__ void forward_propagation_fc(float* input, float* weights, float* bias, float* output) {
         int i = threadIdx.x;
         float sum = 0.0f;
         for(int j = 0; j < N; j++){
         sum += bias[j] + weights[i*N + j] * input[j];
        }
        output[i] = sum;
}

