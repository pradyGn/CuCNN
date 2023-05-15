#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define input_M 8
#define input_N 8
#define filter_M 3
#define filter_N 3
#define output_M 6
#define output_N 6

__global__ void convolutional_layer2D (float *filter, float *input, float *output, float bias)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int input_pos = i + (j*input_N);

    float sum = 0;
    //int output_pos = (i + (filter_N - 1) - n) + (j + (filter_M - 1) - m) * output_N;
    //int filter_pos = (m*filter_N) + n;
    
    for (int m = 0; m < filter_M; m++){
        for (int n = 0; n < filter_N; n++){

            sum += filter[m * filter_N + n] * input[input_pos + n + m*(input_N)];

        }
    }
    
    int output_pos = i + (j*output_N);

    output[output_pos] = sum + bias;
    
    


}

void initialize(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = j + i;
        }
    }
}

void check_matrix(float *matrix, int matrix_M, int matrix_N){
    for (int i=0; i<matrix_M; i++){
        for (int j=0; j<matrix_N; j++)
        {
                printf("%.2f", matrix[(i*matrix_M)+j]);
                printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}


int main(){

    float *d_output, *h_output, *d_filter, *h_filter, *d_input, *h_input;
    float *h_bias, *d_bias;

    h_output = (float*)malloc(sizeof(float) * (output_M * output_M));
    h_filter = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    h_input = (float*)malloc(sizeof(float) * (input_M * input_M));
    h_bais = (float*)malloc(sizeof(float));

    h_bias = 0.1;
    initialize(h_filter, filter_M, filter_M);
    initialize(h_input, input_M, input_M);
    
    for (int i = 0; i < output_N; i++){
        for (int j = 0; j < output_N; j++){
            h_output[(i*output_N) + j] = 0;
        }
    }

    check_matrix(h_filter, filter_M, filter_M);
    check_matrix(h_input, input_M, input_M);


    cudaMalloc((void**)&d_output, sizeof(float) * (output_M * output_M));
    cudaMalloc((void**)&d_filter, sizeof(float) * (filter_M * filter_M));
    cudaMalloc((void**)&d_input, sizeof(float) * (input_M * input_M));
    cudaMalloc((void**)&d_bias, sizeof(float));

    cudaMemcpy(d_filter, h_filter, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, sizeof(float) * (input_M * input_M), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(float) * (output_M * output_M), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 gridsize(output_M);
    dim3 blocksize(output_M);

    convolutional_layer2D <<<gridsize, blocksize>>>(d_filter, d_input, d_output, d_bias);

    cudaMemcpy(h_output, d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);

    for (int i=0; i<output_M; i++){
        for (int j=0; j<output_M; j++)
        {
                printf("%.2f", h_output[(i*output_M)+j]);
                printf(" ");
        }
        printf("\n");
    }
    

    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_input);

    free(h_output);
    free(h_filter);
    free(h_input);

    

}