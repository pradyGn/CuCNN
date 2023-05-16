#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


__global__ void transpose(float *d_odata, float *d_idata, int width, int height) {
  // Get the thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate the global matrix indices
  int x = tid % width;
  int y = tid / width;

  // Calculate the 1D index of the element in the output matrix
  int index = y * width + x;

  // Write the element from the input matrix to the output matrix
  d_odata[index] = d_idata[tid];
}


int main() {

// Allocate memory on the device for the input and output matrices
float *idata, *odata;
float *d_idata, *d_odata;
int width = 4;
int height = 7;
idata = (float*)malloc(width * height * sizeof(float));
odata = (float*)malloc(7*4* sizeof(float));
cudaMalloc((void **)&d_idata, 4 * 7 * sizeof(float));
cudaMalloc((void **)&d_odata, 4 * 7 * sizeof(float));

initialize_filter(idata, 4,7);
check_matrix(idata,4,7);
// Copy the input matrix to the device
cudaMemcpy(d_idata, idata, width * height * sizeof(float), cudaMemcpyHostToDevice);

// Launch the kernel
transpose<<<4, 7>>>(d_odata, d_idata, width, height);

// Copy the output matrix back to the host
cudaMemcpy(odata, d_odata, width * height * sizeof(float), cudaMemcpyDeviceToHost);
check_matrix(odata,7,4);
// Free the device memory
cudaFree(d_idata);
cudaFree(d_odata);

free(idata);
free(odata);

return 0;
}