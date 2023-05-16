#include <stdio.h>
#include <iostream>
#include <cmath>


__global__ void sigmoid_function(float* input, float* output){
    int i = threadIdx.x;

    output[i] = 1/(1 + exp(-1*input[i]));
}

