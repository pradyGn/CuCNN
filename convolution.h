#include <stdio.h>
#include <iostream>
#include "constants.h"


// input_N, filter_M, output_N needs to be defined.


__global__ void convolutional_layer2D (float *filter, float *input, float *output, float *bias)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int input_pos = i + (j*input_N);

    float sum = 0;
    
    for (int m = 0; m < filter_M; m++){
        for (int n = 0; n < filter_M; n++){

            sum += filter[m * filter_M + n] * input[input_pos + n + m*(input_N)]; //+ bias[m * filter_M + n];

        }
    }
    
    int output_pos = i + (j*output_N);
    output[output_pos] = sum + bias[output_pos];

}

__global__ void filter_grad_func (float *filter, float *input, float *output)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int input_pos = i + (j*input_N);

    float sum = 0;
    
    for (int m = 0; m < output_M; m++){
        for (int n = 0; n < output_M; n++){

            sum += filter[m * output_M + n] * input[input_pos + n + m*(input_N)];

        }
    }
    
    int output_pos = i + (j*filter_N);
    output[output_pos] = sum;

}


void initialize_filter(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    }
}

void initialize_output(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = 0;
        }
    }
}


void check_matrix(float *matrix, int matrix_M, int matrix_N){
    for (int i=0; i<matrix_M; i++){
        for (int j=0; j<matrix_N; j++)
        {
                printf("%.2f", matrix[(i*matrix_N)+j]);
                printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

