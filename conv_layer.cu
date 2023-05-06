#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define input_M 3
#define input_N 3
#define filter_M 2
#define filter_N 2
#define output_M 4
#define output_N 4

__global__ void convolutional_layer2D (float *filter, float *input, float *output)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int input_pos = i + (j*input_N);
    //int output_pos = (i + (filter_N - 1)) + ((j + (filter_M - 1)) * output_N);
    
    for (int m = 0; m < filter_M; m++){
        for (int n = 0; n < filter_N; n++){
            //int output_pos = (i + (filter_N - 1) - n) + (j + (filter_M - 1) - m) * output_N;
            int output_pos = (i + (filter_N - 1) - n) + ((j + (filter_M - 1)) * output_N);
            int filter_pos = (m*filter_N) + n;
            output[output_pos] += filter[filter_pos] * input[input_pos];
        }
    }
    
    
    


}

void initialize(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = j + i;
        }
    }
}


int main(){

    float *d_output, *h_output, *d_filter, *h_filter, *d_input, *h_input;

    h_output = (float*)malloc(sizeof(float) * (output_M * output_M));
    h_filter = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    h_input = (float*)malloc(sizeof(float) * (input_M * input_M));

    initialize(h_filter, filter_M, filter_M);
    initialize(h_input, input_M, input_M);
    
    for (int i = 0; i < output_N; i++){
        for (int j = 0; j < output_N; j++){
            h_output[(i*output_N) + j] = 0;
        }
    }

    for (int i=0; i<filter_M; i++){
        for (int j=0; j<filter_M; j++)
        {
                printf("%.2f", h_filter[(i*filter_M)+j]);
                printf(" ");
        }
        printf("\n");
    }

    printf("\n");

    for (int i=0; i<input_M; i++){
        for (int j=0; j<input_M; j++)
        {
                printf("%.2f", h_input[(i*input_M)+j]);
                printf(" ");
        }
        printf("\n");
    }

    printf("\n");

    cudaMalloc((void**)&d_output, sizeof(float) * (output_M * output_M));
    cudaMalloc((void**)&d_filter, sizeof(float) * (filter_M * filter_M));
    cudaMalloc((void**)&d_input, sizeof(float) * (input_M * input_M));

    cudaMemcpy(d_filter, h_filter, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, sizeof(float) * (input_M * input_M), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(float) * (output_M * output_M), cudaMemcpyHostToDevice);

    
    dim3 gridsize(input_M);
    dim3 blocksize(input_M);

    convolutional_layer2D <<<gridsize, blocksize>>>(d_filter, d_input, d_output);

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