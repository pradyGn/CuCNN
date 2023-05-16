#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void transpose(float *input, float* output, int M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            output[i * M + j] = input[j * N + i];
            cout << input[j * M + i] << endl;
        }
    }
}

void initialize_output(float *matrix, int matrix_M, int matrix_N){
    for (int i = 0; i < matrix_M; i++){
        for (int j = 0; j < matrix_N; j++){
            matrix[(i*matrix_N) + j] = i + j;
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