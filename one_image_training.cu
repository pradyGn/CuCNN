#include <Mnist_test.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

using namespace std;

int main(){

    float* train_images = (float*)malloc(sizeof(float) * 60000 * 784);
    float* train_labels = (float*)malloc(sizeof(float) * 60000);
    float* test_images = (float*)malloc(sizeof(float) * 10000 * 784);
    float* test_labels = (float*)malloc(sizeof(float) * 10000);

    get_image_data(train_images, train_labels, test_images, test_labels)

    //Print out the first image.
    for (int i = 0; i < 784; i++) {
      printf("%f ", train_images[i]);
    }
    printf("\n");

}