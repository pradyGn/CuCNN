#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void add_padding(float *input, float *padded_input, int padding_dim, int input_N, int padded_input_N){

    for (int i = 0; i<padded_input_N; i++){
        for (int j = 0; j<padded_input_N; j++){
            padded_input[i*padded_input_N + j] = 0;
        }
    }

    for (int i = padding_dim; i<(input_N + padding_dim); i++){
        cout << i << endl;
        for (int j = padding_dim; j<(input_N + padding_dim); j++){
            padded_input[i*padded_input_N + j] = input[(i-padding_dim)*input_N + (j-padding_dim)];
        }
    }


}

void initialize(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = j + i + 1;
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
    float *input, *padded_input;

    int padding_dim = 2;
    int input_N = 2;
    int padded_input_N = padding_dim + input_N;

    padded_input = (float*)malloc(sizeof(float) * (padded_input_N * padded_input_N));
    input = (float*)malloc(sizeof(float) * (input_N * input_N));

    initialize(input, input_N, input_N);
    check_matrix(input, input_N, input_N);

    add_padding(input, padded_input, padding_dim, input_N, padded_input_N);
    check_matrix(padded_input, padded_input_N, padded_input_N);


}