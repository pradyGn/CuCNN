#include "constants.h"
#include <cuda.h>
#include <stdlib.h>

void initialize_dense_weights_and_bias(float* weights, float* bias){
for (int i = 0; i < dense_output_M; i++) {
    for(int j = 0; j < output_N*output_N; j++) {
        weights[i*dense_output_M + j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    
    bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
    }
}

void initialize_dense_output(float* output){
for (int i = 0; i < dense_output_M; i++) {
        output[i] = 0.0f;
    }
}

__global__ void forward_propagation_fc(float* input, float* weights, float* bias, float* output) {
         int i = threadIdx.x;
         float sum = 0.0f;
         for(int j = 0; j < output_N*output_N; j++){
         sum += bias[i] + weights[i*dense_output_M + j] * input[j];
        }
        output[i] = sum;
}

__global__ void backward_propagation_fc_lastlayer(float* sigmoid_output,int* labels,float* delta)
{
int i = threadIdx.x;
delta[i] = (sigmoid_output[i] - labels[i])/bs;
}

__global__ void backward_propagation_fc(float* sigmoid_output,float* delta_next,float* delta_curr,float* weights)
{
 int i = blockIdx.x * blockDim.x  + threadIdx.x;
 int j = blockIdx.x;
 delta_curr[i] = sigmoid_output[j]*delta_next[i % blockDim.x]; 
 delta_curr[i] /= bs;
 delta_curr[i] += lambda*weights[i];

//delta_curr[(i % blockDim.x) + j*blockDim.x] += lambda*weights[(i % blockDim.x) + j*blockDim.x];
}

__global__ void weight_update(float* delta_curr,float* weights)
{
 int i = threadIdx.x;
 weights[i] -= lr*delta_curr[i];
 }