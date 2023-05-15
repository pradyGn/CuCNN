#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int N = 4;
__global__ void forward_propagation_fc(float* input, float* weights, float* bias, float* output) {
         int i = threadIdx.x;
         float sum = 0.0f;
         for(int j = 0; j < N; j++){
         sum += bias[j] + weights[i*N + j] * input[j];
        }
        output[i] = sum;
}

int main()
{
     // Allocate memory for arrays
    float* input = (float*)malloc(N * sizeof(float));
    float* weights = (float*)malloc(N*N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* biases = (float*)malloc(N * sizeof(float));
    // Initialize all arrays
    for (int i = 0; i < N; i++) {
        input[i] = i;
        for(int j = 0; j < N; j++){
        weights[i*N + j] = 0.5f;}
        output[i] = 0.0f;
        biases[i] = 0.0f;
    }
    // Allocate CUDA Memory
    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    float* d_weights;
    cudaMalloc(&d_weights, N*N * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, N * sizeof(float));
    float* d_biases;
    cudaMalloc(&d_biases, N * sizeof(float));
    // Copy the required parameters to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, N*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, N * sizeof(float), cudaMemcpyHostToDevice);
    // Launch the kernel.
    dim3 blocks(1);
    dim3 threads(N);
    forward_propagation_fc<<<blocks, threads>>>(d_input, d_weights, d_biases, d_output);
    // Copy the output array back to the host.
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    // Print input array
    for (int i = 0; i < N; i++) {
        printf("%f ", input[i]);
    }
    printf("\n\n\n");
    // Print weights
    for (int i = 0; i < N; i++) {
        for (int j = 0;j < N; j++){
        printf("%f ", weights[i*N + j]);
    }}
    printf("\n\n\n");
    // Print the output array.
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }

    // Free CUDA memory.
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_biases);
    // Free host memory.
    free(input);
    free(weights);
    free(output);
    free(biases);
    return 0;
}