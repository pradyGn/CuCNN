#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>


class FeedForwardLayer
{
public:

float *weights, *bias, *Op, lr, *dev_op,*dev_wt;
int ht, wdth, op_sz;

FeedForwardLayer::FeedForwardLayer(){
    wdth = 28;
    ht = 28;
    int op_sz = 28*28;

    float h_biases[28];
    float h_weights[28][28];
    
    cudaMalloc(&Op, sizeof(float)*op_sz);
    cudaMalloc(&weights, sizeof(float)*wdth*ht);
    cudaMalloc(&bias, sizeof(float)*ht);
    cudaMalloc(&dev_op, sizeof(float) * op_sz);
    cudaMalloc(&dev_wt, sizeof(float) * wdth*ht);

    for (int i = 0; i < ht; ++i) {
		h_biases[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < wdth; ++j) {
			h_weights[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}
    cudaMemcpy(bias, h_biases, sizeof(float) * ht, cudaMemcpyHostToDevice);
	cudaMemcpy(weights, h_weights, sizeof(float) * wdth * ht, cudaMemcpyHostToDevice);
}
__global__ void forward(float* O, float* X, float *W, float *b, int W_x, int W_y, int X_x, int X_y){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int O_x = X_x; // Is it transposed?
    int O_y = W_y; // Is it transposed?

    //O[i + j*O_x] = W[i + j*O_x]*X[i + j*O_x] + b[i + j*O_x];
    float O_val = 0;
    if (row < O_y && col < O_x){
        for (int i = 0; i < W_x; i++){
            O_val += W[row * W_x + i] * X[i * X_x + col];
        }
        O[row*O_x + col] = O_val + b[row];
    }
    Op = O;
}
};
