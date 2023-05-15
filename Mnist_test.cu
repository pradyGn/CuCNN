#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Function to read the MNIST dataset.
void read_mnist_dataset(float* train_images, float* train_labels, float* test_images, float* test_labels) {

  // Open the MNIST files.
  FILE* train_images_file = fopen("./MNIST/MNIST_for_C/data/train-images.idx3-ubyte", "rb");
  FILE* train_labels_file = fopen("./MNIST/MNIST_for_C/data/train-labels.idx1-ubyte", "rb");
  FILE* test_images_file = fopen("./MNIST/MNIST_for_C/data/t10k-images.idx3-ubyte", "rb");
  FILE* test_labels_file = fopen("./MNIST/MNIST_for_C/data/t10k-labels.idx1-ubyte", "rb");

  // Check if the files were opened successfully.
  if (train_images_file == NULL || train_labels_file == NULL || test_images_file == NULL || test_labels_file == NULL) {
    printf("Could not open MNIST files.\n");
    exit(1);
  }

  // Read the number of training images.
  int num_train_images;
  fread(&num_train_images, sizeof(int), 1, train_images_file);

  // Read the number of test images.
  int num_test_images;
  fread(&num_test_images, sizeof(int), 1, test_images_file);

  // Read the training images.
  for (int i = 0; i < num_train_images; i++) {
    unsigned char image[784];
    fread(image, sizeof(unsigned char), 784, train_images_file);
    for (int j = 0; j < 784; j++) {
      train_images[i * 784 + j] = image[j] / 255.0f;
    }
  }

  // Read the training labels.
  for (int i = 0; i < num_train_images; i++) {
    unsigned char label;
    fread(&label, sizeof(unsigned char), 1, train_labels_file);
    train_labels[i] = label;
  }

  // Read the test images.
  for (int i = 0; i < num_test_images; i++) {
    unsigned char image[784];
    fread(image, sizeof(unsigned char), 784, test_images_file);
    for (int j = 0; j < 784; j++) {
      test_images[i * 784 + j] = image[j] / 255.0f;
    }
  }

  // Read the test labels.
  for (int i = 0; i < num_test_images; i++) {
    unsigned char label;
    fread(&label, sizeof(unsigned char), 1, test_labels_file);
    test_labels[i] = label;
  }

  // Close the files.
  fclose(train_images_file);
  fclose(train_labels_file);
  fclose(test_images_file);
  fclose(test_labels_file);
}

int main() {

  // Initialize the arrays.
  float* train_images = (float*)malloc(sizeof(float) * 60000 * 784);
  float* train_labels = (float*)malloc(sizeof(float) * 60000);
  float* test_images = (float*)malloc(sizeof(float) * 10000 * 784);
  float* test_labels = (float*)malloc(sizeof(float) * 10000);

  // Read the MNIST dataset.
  read_mnist_dataset(train_images, train_labels, test_images, test_labels);

  // Print out the first image.
  for (int i = 0; i < 784; i++) {
    printf("%f ", train_images[i]);
  }
  printf("\n");

  // Print out the first label.
  printf("%f\n", train_labels[0]);

  // Free the memory.
  free(train_images);
  free(train_labels);
  free(test_images);
  free(test_labels);

  return 0;
}