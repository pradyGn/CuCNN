#include <stdio.h>
#include <iostream>
#include <cmath>


__global__ void sigmoid_function(float* input, float* output){
    int i = threadIdx.x;

    output[i] = 1/(1 + exp(-1*input[i]));
}


__global__ void softmax_denom(float* denom, float* input){
    int i = threadIdx.x;

    for (int j = 0; j < dense_output_M; j++){
        denom[0] += exp(input[j]);
    }

}

__global__ void softmax(float* denom, float* input, float* softmax_return){
    int i = threadIdx.x;

    softmax_return[i] = exp(input[i])/denom[0];

}

__global__ void cross_entropy_loss(float *last_layer, int* labels, float *loss){
    int i = threadIdx.x;
    loss[0] += -1*labels[i]*log(last_layer[i]);
}