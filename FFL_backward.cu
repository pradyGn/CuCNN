#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

using namespace std;

const int N = 4;
const int bs = 1;
const int lambda = 14;

__global__ void backward_propagation_fc_lastlayer(float* sigmoid_output,int* labels,float* delta)
{
int i = threadIdx.x;
delta[i] = (sigmoid_output[i] - labels[i])/bs;
}

__global__ void backward_propagation_fc(float* sigmoid_output,float* delta_next,float* delta_curr,float* weights)
{
 int i = blockIdx.x * blockDim.x  + threadIdx.x;
 int j = blockIdx.x;
 delta_curr[i] += sigmoid_output[j]*delta_next[i % blockDim.x]; 
 delta_curr[i] /= bs;
 //delta_curr[(i % blockDim.x) + j*blockDim.x] += lambda*weights[(i % blockDim.x) + j*blockDim.x];
 delta_curr[i] += lambda*weights[i*N + j];
}


int main(){
float* sigmoid_output = (float*)malloc(N * sizeof(float));
float* delta_curr = (float*)malloc(N*N * sizeof(float));
float* delta_next = (float*)malloc(N * sizeof(float));
float* weights = (float*)malloc(N*N * sizeof(float));
//float* lambda = (float*)malloc(1 * sizeof(float));

//lambda[0] = 0.1;
for (int i = 0; i < N; i++){
    sigmoid_output[i] = i;
    cout << i << endl;
    delta_next[i] = N + i;
    for (int j = 0; j < N; j++){ 
        delta_curr[i*N + j] = 0.0f;
        weights[i*N + j] = 1.0f;
    }
}
float* d_sig_op;
cudaMalloc(&d_sig_op, N * sizeof(float));
float* d_delta_curr;
cudaMalloc(&d_delta_curr, N*N * sizeof(float));
float* d_delta_next;
cudaMalloc(&d_delta_next, N * sizeof(float));
float* d_weights;
cudaMalloc(&d_weights, N*N * sizeof(float));


cudaMemcpy(d_sig_op, sigmoid_output, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_delta_curr, delta_curr, N*N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_delta_next, delta_next, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_weights, weights, N*N * sizeof(float), cudaMemcpyHostToDevice);

dim3 blocks(N);
dim3 threads(N);
backward_propagation_fc<<<blocks, threads>>>(d_sig_op, d_delta_next, d_delta_curr,d_weights);
cudaMemcpy(delta_curr, d_delta_curr, N * N *sizeof(float), cudaMemcpyDeviceToHost);

// Print input array
    for (int i = 0; i < N; i++) {
        printf("%f ", sigmoid_output[i]);
    }
printf("\n\n\n");
// Delta_Curr_next
for (int i = 0; i < N; i++) {
    printf("%f ", delta_next[i]);
}
printf("\n\n");
// Print weights
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++){
    printf("%f ", weights[i*N + j]);
}}
printf("\n\n");
// Print output
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++){
    printf("%f ", delta_curr[i*N + j]);
}}
// Free CUDA memory.
cudaFree(d_sig_op);
cudaFree(d_delta_next);
cudaFree(d_delta_curr);
cudaFree(d_weights);
// Free host memory.
free(sigmoid_output);
free(delta_curr);
free(delta_next);
free(weights);
return 0;
}