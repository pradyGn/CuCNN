#include <stdio.h>
#include <iostream>
#include <cmath>


__global__ void sigmoid_function(float* input, float* output){
    int i = threadIdx.x;

    output[i] = 1/(1 + exp(-1*input[i]));
}


__global__ void softmax_denom(float* denom, float* input){
    int i = threadIdx.x;

    denom[i] += exp(input[i]);

}

__global__ void softmax(float* denom, float* input, float* softmax_return){
    int i = threadIdx.x;

    softmax_return[i] = exp(input[i])/denom[0];

}