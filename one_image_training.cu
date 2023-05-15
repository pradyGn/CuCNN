#include "Mnist_test.h"
#include "convolution.h"
#include "constants.h"
#include "dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;

int main(){

    float* h_train_images = (float*)malloc(sizeof(float) * 60000 * 784);
    float* h_train_labels = (float*)malloc(sizeof(float) * 60000);
    float* h_test_images = (float*)malloc(sizeof(float) * 10000 * 784);
    float* h_test_labels = (float*)malloc(sizeof(float) * 10000);

    float  *h_output, *d_filter, *h_filter, *h_bias_conv, *d_bias_conv, *d_bias_dense, *h_bias_dense, *h_weights, *d_weights;

    float *h_dense_output;


    get_image_data(h_train_images, h_train_labels, h_test_images, h_test_labels);

    h_output = (float*)malloc(sizeof(float) * 60000 * (output_N * output_N));
    h_dense_output = (float*)malloc(sizeof(float) * 60000 * (dense_output_M * 1));

    // bias initization and allocation
    h_bias_conv = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    initialize_filter(h_bias_conv, filter_M, filter_M);
    cudaMalloc((void**)&d_bias_conv, sizeof(float) * (filter_M * filter_M));
    cudaMemcpy(d_bias_conv, h_bias_conv, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    //check_matrix(h_bias_conv, filter_M, filter_M);




    // filter initization and allocation
    h_filter = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    initialize_filter(h_filter, filter_M, filter_M);
    cudaMalloc((void**)&d_filter, sizeof(float) * (filter_M * filter_M));
    cudaMemcpy(d_filter, h_filter, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    //check_matrix(h_filter, filter_M, filter_M);

    // Initialize and allocate weights and bias for Dense layer
    h_weights = (float*)malloc(sizeof(float) * (dense_output_M * (output_M * output_M)));
    h_bias_dense = (float*)malloc(sizeof(float) * (dense_output_M));
    initialize_dense_weights_and_bias(h_weights, h_bias_dense);
    cudaMalloc((void**)&d_weights, sizeof(float) * (dense_output_M * (output_M * output_M)));
    cudaMalloc((void**)&d_bias_dense, sizeof(float) * dense_output_M);
    cudaMemcpy(d_weights, h_weights, sizeof(float) * (dense_output_M * (output_M * output_M)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_dense, h_bias_dense, sizeof(float) * dense_output_M, cudaMemcpyHostToDevice);
    check_matrix(h_bias_dense, 1, dense_output_M);
    check_matrix(h_weights, dense_output_M, (output_M * output_M));

    for (int i = 0; i < 2; i++){

        initialize_output(&h_output[784*i], output_N, output_N);
        
        initialize_dense_output(&h_dense_output[10*i]);
        
        float *d_train_image;
        float *d_dense_output, *d_output;

        cudaMalloc((void**)&d_output, sizeof(float) * (output_N * output_N));
        cudaMalloc((void**)&d_train_image, sizeof(float) * 784);
        cudaMalloc((void**)&d_dense_output, sizeof(float) * dense_output_M);


        cudaMemcpy(d_output, &h_output[784*i], sizeof(float) * (output_N * output_N), cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_image, &h_train_images[784*i], sizeof(float) * 784, cudaMemcpyHostToDevice);

        cudaMemcpy(d_dense_output, &h_dense_output[10*i], sizeof(float) * (dense_output_M * 1), cudaMemcpyHostToDevice);

        
        dim3 gridsize(output_M);
        dim3 blocksize(output_M);

        convolutional_layer2D <<<gridsize, blocksize>>>(d_filter, d_train_image, d_output, d_bias_conv);

        //cudaMemcpy(&h_output[784*i], d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);
        
        dim3 gridsize_dense(1);
        dim3 blocksize_dense(dense_output_M);

        forward_propagation_fc<<<gridsize_dense, blocksize_dense>>>(d_output, d_weights, d_bias_dense, d_dense_output);
        cudaMemcpy(&h_output[784*i], d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_dense_output[10*i], d_dense_output, sizeof(float) * (dense_output_M * 1), cudaMemcpyDeviceToHost);
        
        if (i == 1){
            //check_matrix(&h_train_images[784*i], input_M, input_M);
            check_matrix(&h_output[784*i], output_M, output_M);
            check_matrix(&h_dense_output[10*i], 1, dense_output_M);
        }

        cudaFree(d_output);
        cudaFree(d_dense_output);
        cudaFree(d_train_image);

    }
    
    cudaFree(d_filter);
    cudaFree(d_bias_conv);
    cudaFree(d_weights);
    cudaFree(d_bias_dense);

    // Free the memory.
    free(h_train_images);
    free(h_train_labels);
    free(h_test_images);
    free(h_test_labels);
    free(h_output);
    free(h_bias_conv);
    free(h_filter);
    free(h_weights);
    free(h_bias_dense);
    free(h_output);

    return 0;

}