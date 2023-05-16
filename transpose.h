#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void transpose(float *input, float* output, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            output[i * M + j] = input[j * N + i];
        }
    }
}