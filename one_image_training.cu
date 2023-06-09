#include "Mnist_test.h"
#include "transpose.h"
#include "convolution.h"
#include "constants.h"
#include "dense.h"
#include "activation_fn.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include<limits>
#include <sys/time.h>

using namespace std;

int main(){

    float* h_train_images = (float*)malloc(sizeof(float) * 60000 * 784);
    int* h_train_labels = (int*)malloc(sizeof(int) * 60000);
    float* h_test_images = (float*)malloc(sizeof(float) * 10000 * 784);
    int* h_test_labels = (int*)malloc(sizeof(int) * 10000);

    float  *h_output, *d_filter, *h_filter, *h_bias_conv, *d_bias_conv, *d_bias_dense, *h_bias_dense, *h_weights, *d_weights;


    float *h_dense_output;


    get_image_data(h_train_images, h_train_labels, h_test_images, h_test_labels);

    h_output = (float*)malloc(sizeof(float) * 60000 * (output_N * output_N));
    h_dense_output = (float*)malloc(sizeof(float) * 60000 * (dense_output_M * 1));

    // bias initization and allocation
    h_bias_conv = (float*)malloc(sizeof(float) * (output_M * output_M));
    initialize_filter(h_bias_conv, output_M, output_M);
    cudaMalloc((void**)&d_bias_conv, sizeof(float) * (output_M * output_M));
    cudaMemcpy(d_bias_conv, h_bias_conv, sizeof(float) * (output_M * output_M), cudaMemcpyHostToDevice);



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
    //check_matrix(h_bias_dense, 1, dense_output_M);
    //check_matrix(h_weights, dense_output_M, (output_M * output_M));

    float *h_loss, *d_loss, *d_loss_arr, *h_loss_arr;
    h_loss_arr = (float*)malloc(sizeof(float) * 100);
    cudaMalloc((void**)&d_loss_arr, sizeof(float)* 100);
    h_loss = (float*)malloc(sizeof(float));
    h_loss[0] = 0;
    cudaMalloc((void**)&d_loss, sizeof(float));
    cudaMemcpy(d_loss, h_loss, sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    for (int i = 0; i < 100; i++){

        initialize_output(&h_output[784*i], output_N, output_N);
        initialize_dense_output(&h_dense_output[10*i]);
        



        int *d_train_label;
        float *d_train_image, *h_delta_ll, *d_delta_ll, *d_delta_curr, *h_delta_curr;
        float *d_dense_output, *d_output, *d_dense_grad_input;




        h_delta_ll = (float*)malloc(sizeof(float) * dense_output_M*1);
        h_delta_curr = (float*)malloc(sizeof(float) * dense_output_M*(output_M*output_M));
        initialize_dense_output(h_delta_ll);




        cudaMalloc((void**)&d_delta_curr, sizeof(float) * dense_output_M*(output_M*output_M));
        cudaMalloc((void**)&d_delta_ll, sizeof(float) * dense_output_M*1);
        cudaMalloc((void**)&d_output, sizeof(float) * (output_N * output_N));
        cudaMalloc((void**)&d_train_image, sizeof(float) * 784);
        cudaMalloc((void**)&d_dense_output, sizeof(float) * dense_output_M);
        cudaMalloc((void**)&d_train_label, sizeof(int) * dense_output_M);
        cudaMalloc((void**)&d_dense_grad_input, sizeof(float) * dense_output_M);

        // One hot labels
        int* one_hot_label = (int*)malloc(sizeof(int) * dense_output_M);
        for (int j = 0; j < dense_output_M; j++) {
            one_hot_label[j] = 0;
        }
        one_hot_label[h_train_labels[i]] = 1;

        cudaMemcpy(d_delta_ll, h_delta_ll, sizeof(float) * (dense_output_M * 1), cudaMemcpyHostToDevice);       
        cudaMemcpy(d_output, &h_output[784*i], sizeof(float) * (output_N * output_N), cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_image, &h_train_images[784*i], sizeof(float) * 784, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dense_output, &h_dense_output[10*i], sizeof(float) * (dense_output_M * 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_label, one_hot_label, sizeof(int) * dense_output_M, cudaMemcpyHostToDevice);

        
        dim3 gridsize(output_M);
        dim3 blocksize(output_M);
        convolutional_layer2D <<<gridsize, blocksize>>>(d_filter, d_train_image, d_output, d_bias_conv);
        //cudaFree(d_train_image);


        dim3 gridsize_sig(1);
        dim3 blocksize_sig(output_M*output_M);
        sigmoid_function<<<gridsize_sig, blocksize_sig>>>(d_output,d_output);

        //cudaMemcpy(&h_output[784*i], d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);
        //cudaMemcpy(&h_output[784*i], d_output, sizeof(float) * (output_M * output_M), cudaMemcpyDeviceToHost);
        
        dim3 gridsize_dense(1);
        dim3 blocksize_dense(dense_output_M);
        forward_propagation_fc<<<gridsize_dense, blocksize_dense>>>(d_output, d_weights, d_bias_dense, d_dense_output);
        
        float *h_min, *h_max, *d_max, *d_min;
        h_min = (float*)malloc(sizeof(float));
        h_max = (float*)malloc(sizeof(float));
        h_min[0] = (numeric_limits<float>::infinity());
        h_max[0] = -1*numeric_limits<float>::infinity();

        cudaMalloc((void**)&d_max, sizeof(float));
        cudaMalloc((void**)&d_min, sizeof(float));

        cudaMemcpy(d_max, h_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_min, h_min, sizeof(float), cudaMemcpyHostToDevice);

        calulate_min_max<<<1, dense_output_M>>>(d_dense_output, d_min, d_max);

        cudaMemcpy(h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

        min_max_normalization<<<1, dense_output_M>>>(d_dense_output, d_min, d_max);


        dim3 gridsize_sig_dense(1);
        dim3 blocksize_sig_dense(dense_output_M * 1);
        float *h_denom, *d_denom;
        h_denom = (float*)malloc(sizeof(float));
        h_denom[0] = 0;
        cudaMalloc((void**)&d_denom, sizeof(float));
        cudaMemcpy(d_denom, h_denom, sizeof(float), cudaMemcpyHostToDevice);
        softmax_denom<<<gridsize_sig_dense, blocksize_sig_dense>>>(d_denom, d_dense_output);


        softmax<<<gridsize_sig_dense, blocksize_sig_dense>>>(d_denom, d_dense_output, d_dense_output);


        dim3 gridsize_loss_dense(1);
        dim3 blocksize_loss_dense(dense_output_M);
        cross_entropy_loss<<<gridsize_loss_dense, blocksize_loss_dense>>>(d_dense_output, d_train_label, d_loss, d_loss_arr);
        
        
        // Backprop for last layer
        dim3 gridsize_ll(1);
        dim3 blocksize_ll(dense_output_M * 1);
        backward_propagation_fc_lastlayer<<<gridsize_ll,blocksize_ll>>>(d_dense_output,d_train_label,d_delta_ll);
        
        
        // Backprop for previous layers
        dim3 gridsize_dense_bp(output_M*output_M);
        dim3 blocksize_dense_bp(dense_output_M * 1);
        backward_propagation_fc<<<gridsize_dense_bp,blocksize_dense_bp>>>(d_output,d_delta_ll,d_delta_curr,d_weights);
        cudaMemcpy(h_delta_curr, d_delta_curr, sizeof(float) * (dense_output_M * (output_M * output_M)), cudaMemcpyDeviceToHost);
        //cudaMemcpy(&h_dense_output[10*i], d_dense_output, sizeof(float) * (dense_output_M * 1), cudaMemcpyDeviceToHost);
        
        
        dim3 gridsize_wts_update(dense_output_M);
        dim3 blocksize_wts_update(output_M*output_M);
        weight_update<<<gridsize_wts_update,blocksize_wts_update>>>(d_delta_curr,d_weights);
        

        dim3 gridsize_dense_grad_input(1);
        dim3 blocksize_dense_grad_input(dense_output_M);

        input_grad<<<gridsize_dense_grad_input, blocksize_dense_grad_input>>>(d_dense_grad_input, d_dense_output);


        float *h_weights_T, *d_weights_T;
        h_weights_T = (float*)malloc(sizeof(float) * dense_output_M * (output_M * output_M));
        cudaMemcpy(h_weights, d_weights, sizeof(float) * (dense_output_M * (output_M * output_M)), cudaMemcpyDeviceToHost);
        

        transpose(h_weights, h_weights_T, (output_M * output_M), dense_output_M);

        cudaMalloc((void**)&d_weights_T, sizeof(float) * (dense_output_M * (output_M * output_M)));
        cudaMemcpy(d_weights_T, h_weights_T, sizeof(float) * (dense_output_M * (output_M * output_M)), cudaMemcpyHostToDevice);






        dim3 gridsize_dense_grad_mm(1);
        dim3 blocksize_dense_grad_mm((output_M*output_M));
        float *d_dense_grad_input_act;
        cudaMalloc((void**)&d_dense_grad_input_act, sizeof(float) * (output_M * output_M));
        matrix_mul<<<gridsize_dense_grad_mm, blocksize_dense_grad_mm>>>(d_dense_grad_input, d_weights_T, d_dense_grad_input_act);


        dim3 gridsize_fg(filter_M);
        dim3 blocksize_fg(filter_M);

        float *d_filter_grad; //h_filter_grad
        cudaMalloc((void**)&d_filter_grad, sizeof(float) * (filter_M * filter_M));


        filter_grad_func <<<gridsize_fg, blocksize_fg>>>(d_dense_grad_input_act, d_train_image, d_filter_grad);

        dim3 gridsize_fup(1);
        dim3 blocksize_fup(filter_M*filter_M);
        weight_update<<<gridsize_fup, blocksize_fup>>>(d_filter_grad, d_filter);

        dim3 gridsize_bup(1);
        dim3 blocksize_bup(output_M*output_M);
        weight_update<<<gridsize_bup, blocksize_bup>>>(d_dense_grad_input_act, d_bias_conv);

        
        cudaFree(d_output);
        cudaFree(d_train_image);
        cudaFree(d_train_label);
        cudaFree(d_dense_output);
        cudaFree(d_delta_ll);
        cudaFree(d_delta_curr);
        cudaFree(d_dense_grad_input_act);
        cudaFree(d_weights_T);
        cudaFree(d_dense_grad_input);
        free(h_delta_ll);
        free(one_hot_label);
        free(h_delta_curr);
        free(h_weights_T);
        cudaFree(d_min);
        cudaFree(d_max);
        free(h_max);
        free(h_min);
        cudaFree(d_denom);
        free(h_denom);
        cudaFree(d_filter_grad);


    }

    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    double flops = 60000*(filter_N * filter_N * output_M * 2 + (output_M * output_M) * dense_output_M * 5 + output_M * output_M * filter_N * 2); // TODO: calculate from m, n, k, NREPEATS, time
    double flops_p_sec = (flops/time);
    double bandwidth = 0;
    cout << "Total training time (ms): " << time << endl;
    cout << "Total Flops: " << flops << endl;
    cout << "flops / ms: " << flops_p_sec << endl;

    cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_loss_arr, d_loss_arr, sizeof(float), cudaMemcpyDeviceToHost);
    check_matrix(h_loss_arr,1,100);

    cout << "Average negative log train loss: " << h_loss[0]/60000 << endl;

    



    cudaFree(d_filter);
    cudaFree(d_bias_conv);
    cudaFree(d_weights);
    cudaFree(d_bias_dense);
    free(h_train_images);
    free(h_train_labels);
    free(h_test_images);
    free(h_test_labels);
    free(h_output);
    free(h_bias_conv);
    free(h_filter);
    free(h_weights);
    free(h_bias_dense);
    free(h_dense_output);
    return 0;

}