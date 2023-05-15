#include "Mnist_test.h"
#include "convolution.h"
#include "constants.h"
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

    float *d_output, *h_output, *d_filter, *h_filter, *h_bias, *d_bias;


    get_image_data(h_train_images, h_train_labels, h_test_images, h_test_labels);

    h_output = (float*)malloc(sizeof(float) * 60000 * (output_N * output_N));
    


    // bias initization and allocation
    h_bias = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    initialize_filter(h_bias, filter_M, filter_M);
    cudaMalloc((void**)&d_bias, sizeof(float) * (filter_M * filter_M));
    cudaMemcpy(d_bias, h_bias, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    check_matrix(h_bias, filter_M, filter_M);




    // filter initization and allocation
    h_filter = (float*)malloc(sizeof(float) * (filter_M * filter_M));
    initialize_filter(h_filter, filter_M, filter_M);
    cudaMalloc((void**)&d_filter, sizeof(float) * (filter_M * filter_M));
    cudaMemcpy(d_filter, h_filter, sizeof(float) * (filter_M * filter_M), cudaMemcpyHostToDevice);
    check_matrix(h_filter, filter_M, filter_M);


    for (int i = 0; i < 2; i++){

        initialize_output(h_output[784*i], output_N, output_N);


        cudaMalloc((void**)&d_output, sizeof(float) * (output_N * output_N));
        cudaMalloc((void**)&d_train_image, sizeof(float) * 784);


        cudaMemcpy(d_output, h_output[784*i], sizeof(float) * (output_N * output_N), cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_image, h_train_images[784*i], sizeof(float) * 784, cudaMemcpyHostToDevice);
        

        dim3 gridsize(output_M);
        dim3 blocksize(output_M);

        convolutional_layer2D <<<gridsize, blocksize>>>(d_filter, d_train_image, d_output, d_bias);

        cudaMemcpy(h_output[784*i], d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);

        if (i == 1){
            check_matrix(h_train_images[784*i], input_M, input_M);
            check_matrix(h_output[784*i], output_M, output_M);
        }


        cudaFree(d_output);
        cudaFree(d_train_image);

    }
    
    cudaFree(d_filter);
    cudaFree(d_bias);



    // Free the memory.
    free(h_train_images);
    free(h_train_labels);
    free(h_test_images);
    free(h_test_labels);
    free(h_output);
    free(h_bias);
    free(h_filter);

    return 0;

}