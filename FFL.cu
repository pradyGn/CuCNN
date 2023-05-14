#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


__global__ void forward_propagation_fc(float* input, float* weights, float* bias, float* output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output[i] = bias[i] + weights[i] * input[i];
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

     // Allocate the input and output arrays.
    float* input = (float*)malloc(N * sizeof(float));
    float* weights = (float*)malloc(N*N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* biases = (float*)malloc(N * sizeof(float));
    // Initialize the input and output arrays.
    for (int i = 0; i < N; i++) {
        input[i] = i;
        for(int j = 0; j < N; j++){
        weights[i*N + j] = 0.5f;}
        output[i] = 0.0f;
        biases[i] = 0.0f;
    }

    // Allocate the CUDA memory for the input and output arrays.
    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    float* d_weights;
    cudaMalloc(&d_weights, N*N * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, N * sizeof(float));
    float* d_biases;
    cudaMalloc(&d_biases, N * sizeof(float));
    // Copy the input and output arrays to the CUDA device.
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, N*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, N * sizeof(float), cudaMemcpyHostToDevice);
    // Launch the kernel.
    dim3 gridsize(1);
    dim3 blocksize(N);
    //fully_connected_forward<<<blocks, threads>>>(d_input, d_weights, d_output, 1, N, N);
    forward_propagation_fc<<<gridsize, blocksize>>>(d_input, d_weights, d_biases, d_output);
 // Copy the output array back to the host.
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%f ", input[i]);
    }
    printf("\n\n\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0;j < N; j++){
        printf("%f ", weights[i*N + j]);
    }}
    printf("\n\n\n");
    // Print the output array.
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }

    // Free the CUDA memory.
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_biases);
    // Free the host memory.
    free(input);
    free(weights);
    free(output);
    free(biases);

    return 0;



}
