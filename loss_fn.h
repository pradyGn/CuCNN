#include <stdio.h>
#include <iostream>
#include <cmath>


__global__ void cross_entropy_loss(float *last_layer, int* labels, float *loss){
    int i = threadIdx.x;
    loss[0] += -1*labels[i]*log(last_layer[i]);
}