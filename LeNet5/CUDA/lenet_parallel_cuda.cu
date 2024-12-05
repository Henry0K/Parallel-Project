/*
 Authors: Henry Hardane, Alex Kheir, Lori Yahnian, and Alain Samaha
 Title: Reduced Model LeNet-5 Parallelized with CUDA
 Description: CUDA parallelized implementation of the LeNet-5 architecture.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// Neural Network Parameters
#define IMAGE_DIMENSION 32
#define KERNEL_SIZE 5
#define LAYER1_CHANNELS 6
#define LAYER2_CHANNELS 16
#define POOL_KERNEL 2
#define FULLY_CONNECTED1_NEURONS 120
#define OUTPUT_NEURONS 10

// Macro Utilities
#define COORD(row, col, stride) ((row) * (stride) + (col))
#define MAX_VAL(a, b) (((a) > (b)) ? (a) : (b))

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Randomized weight initialization with Uniform distribution
float generate_weight(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return ((float)rand() / (float)RAND_MAX) * 2 * limit - limit;
}

// Activation Functions Kernels
__global__ void relu_activation(float *data, int length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length){
        data[idx] = data[idx] > 0 ? data[idx] : 0;
    }
}

__global__ void softmax_activation(float *input, int length){
    extern __shared__ float shared_data[];
    // Find max value
    float max_val = input[0];
    for(int i = threadIdx.x; i < length; i += blockDim.x){
        if(input[i] > max_val){
            max_val = input[i];
        }
    }
    // Reduction to find the max
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    // Parallel reduction
    for(int s = blockDim.x / 2; s > 0; s >>=1){
        if(threadIdx.x < s){
            if(shared_data[threadIdx.x + s] > shared_data[threadIdx.x]){
                shared_data[threadIdx.x] = shared_data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_data[0];
    // Compute exponentials and sum
    float sum = 0.0f;
    for(int i = threadIdx.x; i < length; i += blockDim.x){
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    // Parallel reduction to find the sum
    for(int s = blockDim.x / 2; s > 0; s >>=1){
        if(threadIdx.x < s){
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total = shared_data[0];
    // Normalize
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x){
        input[i] /= total;
    }
}

// Convolution Operation Kernel using shared memory tiling
__global__ void convolution_kernel(float *input, float *filters, float *output, int input_dim, int num_filters, int kernel_dim, int output_dim){
    // Calculate the filter and output indices
    int f = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < output_dim && col < output_dim){
        float acc = 0.0f;
        for(int k = 0; k < kernel_dim; ++k){
            for(int l = 0; l < kernel_dim; ++l){
                acc += input[COORD(row + k, col + l, input_dim)] *
                       filters[f * kernel_dim * kernel_dim + COORD(k, l, kernel_dim)];
            }
        }
        output[f * output_dim * output_dim + COORD(row, col, output_dim)] = acc;
    }
}

// Pooling Operation Kernel (Max Pooling)
__global__ void pooling_kernel(float *input, float *output, int input_dim, int pool_dim, int num_channels, int output_dim){
    int c = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < output_dim && col < output_dim){
        float max_val = -INFINITY;
        for(int i = 0; i < pool_dim; ++i){
            for(int j = 0; j < pool_dim; ++j){
                float val = input[c * input_dim * input_dim + COORD(row * pool_dim + i, col * pool_dim + j, input_dim)];
                max_val = MAX_VAL(max_val, val);
            }
        }
        output[c * output_dim * output_dim + COORD(row, col, output_dim)] = max_val;
    }
}

// Fully Connected Layer Kernel (Matrix-Vector Multiplication)
__global__ void dense_layer_kernel(float *input, float *weights, float *biases, float *output, int input_length, int output_length){
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o < output_length){
        float sum = biases[o];
        for(int i = 0; i < input_length; ++i){
            sum += input[i] * weights[o * input_length + i];
        }
        output[o] = sum;
    }
}

// Complete Model Function (Runs on Host)
void neural_network_cuda(float *d_input, float *d_conv1_weights, float *d_conv2_weights,
                        float *d_fc1_weights, float *d_fc1_biases,
                        float *d_fc2_weights, float *d_fc2_biases,
                        float *d_final_output,
                        int image_dim, int kernel_size, int layer1_channels,
                        int layer2_channels, int pool_kernel,
                        int fully_connected1_neurons, int output_neurons){

    // Define dimensions
    int conv1_output_dim = image_dim - kernel_size + 1;
    int conv1_output_size = layer1_channels * conv1_output_dim * conv1_output_dim;

    // Allocate device memory for conv1 output
    float *d_conv1_output;
    cudaCheckError(cudaMalloc((void**)&d_conv1_output, conv1_output_size * sizeof(float)));

    // Define CUDA grid and block dimensions for convolution
    dim3 blockDim_conv(16, 16);
    dim3 gridDim_conv((conv1_output_dim + blockDim_conv.x -1)/blockDim_conv.x,
                     (conv1_output_dim + blockDim_conv.y -1)/blockDim_conv.y,
                     layer1_channels);

    // Create CUDA events for timing
    cudaEvent_t conv1_start, conv1_stop;
    cudaEventCreate(&conv1_start);
    cudaEventCreate(&conv1_stop);
    cudaEventRecord(conv1_start, 0);

    // Launch convolution kernel for conv1
    convolution_kernel<<<gridDim_conv, blockDim_conv>>>(d_input, d_conv1_weights, d_conv1_output,
                                                        image_dim, layer1_channels, kernel_size, conv1_output_dim);
    cudaCheckError(cudaGetLastError());

    // Record end time
    cudaEventRecord(conv1_stop, 0);
    cudaEventSynchronize(conv1_stop);
    float conv1_time;
    cudaEventElapsedTime(&conv1_time, conv1_start, conv1_stop);

    // Apply ReLU activation
    int threads = 256;
    int blocks = (conv1_output_size + threads -1)/threads;
    relu_activation<<<blocks, threads>>>(d_conv1_output, conv1_output_size);
    cudaCheckError(cudaGetLastError());

    // Pooling Layer 1
    int pool1_output_dim = conv1_output_dim / pool_kernel;
    int pool1_output_size = layer1_channels * pool1_output_dim * pool1_output_dim;
    float *d_pool1_output;
    cudaCheckError(cudaMalloc((void**)&d_pool1_output, pool1_output_size * sizeof(float)));

    dim3 blockDim_pool(16, 16);
    dim3 gridDim_pool((pool1_output_dim + blockDim_pool.x -1)/blockDim_pool.x,
                     (pool1_output_dim + blockDim_pool.y -1)/blockDim_pool.y,
                     layer1_channels);

    // Timing for Pooling Layer 1
    cudaEvent_t pool1_start, pool1_stop;
    cudaEventCreate(&pool1_start);
    cudaEventCreate(&pool1_stop);
    cudaEventRecord(pool1_start, 0);

    pooling_kernel<<<gridDim_pool, blockDim_pool>>>(d_conv1_output, d_pool1_output,
                                                    conv1_output_dim, pool_kernel, layer1_channels, pool1_output_dim);
    cudaCheckError(cudaGetLastError());

    cudaEventRecord(pool1_stop, 0);
    cudaEventSynchronize(pool1_stop);
    float pool1_time;
    cudaEventElapsedTime(&pool1_time, pool1_start, pool1_stop);

    // Convolution Layer 2
    int conv2_output_dim = pool1_output_dim - kernel_size + 1;
    int conv2_output_size = layer2_channels * conv2_output_dim * conv2_output_dim;
    float *d_conv2_output;
    cudaCheckError(cudaMalloc((void**)&d_conv2_output, conv2_output_size * sizeof(float)));

    dim3 gridDim_conv2((conv2_output_dim + blockDim_conv.x -1)/blockDim_conv.x,
                      (conv2_output_dim + blockDim_conv.y -1)/blockDim_conv.y,
                      layer2_channels);

    // Timing for Convolution Layer 2
    cudaEvent_t conv2_start, conv2_stop;
    cudaEventCreate(&conv2_start);
    cudaEventCreate(&conv2_stop);
    cudaEventRecord(conv2_start, 0);

    convolution_kernel<<<gridDim_conv2, blockDim_conv>>>(d_pool1_output, d_conv2_weights, d_conv2_output,
                                                         pool1_output_dim, layer2_channels, kernel_size, conv2_output_dim);
    cudaCheckError(cudaGetLastError());

    cudaEventRecord(conv2_stop, 0);
    cudaEventSynchronize(conv2_stop);
    float conv2_time;
    cudaEventElapsedTime(&conv2_time, conv2_start, conv2_stop);

    // Apply ReLU activation
    relu_activation<<<(conv2_output_size + threads -1)/threads, threads>>>(d_conv2_output, conv2_output_size);
    cudaCheckError(cudaGetLastError());

    // Pooling Layer 2
    int pool2_output_dim = conv2_output_dim / pool_kernel;
    int pool2_output_size = layer2_channels * pool2_output_dim * pool2_output_dim;
    float *d_pool2_output;
    cudaCheckError(cudaMalloc((void**)&d_pool2_output, pool2_output_size * sizeof(float)));

    dim3 gridDim_pool2((pool2_output_dim + blockDim_pool.x -1)/blockDim_pool.x,
                      (pool2_output_dim + blockDim_pool.y -1)/blockDim_pool.y,
                      layer2_channels);

    // Timing for Pooling Layer 2
    cudaEvent_t pool2_start, pool2_stop;
    cudaEventCreate(&pool2_start);
    cudaEventCreate(&pool2_stop);
    cudaEventRecord(pool2_start, 0);

    pooling_kernel<<<gridDim_pool2, blockDim_pool>>>(d_conv2_output, d_pool2_output,
                                                    conv2_output_dim, pool_kernel, layer2_channels, pool2_output_dim);
    cudaCheckError(cudaGetLastError());

    cudaEventRecord(pool2_stop, 0);
    cudaEventSynchronize(pool2_stop);
    float pool2_time;
    cudaEventElapsedTime(&pool2_time, pool2_start, pool2_stop);

    // Fully Connected Layer 1
    int flattened_length = layer2_channels * pool2_output_dim * pool2_output_dim;
    float *d_fc1_output;
    cudaCheckError(cudaMalloc((void**)&d_fc1_output, fully_connected1_neurons * sizeof(float)));

    dim3 blockDim_fc(256);
    dim3 gridDim_fc((fully_connected1_neurons + blockDim_fc.x -1)/blockDim_fc.x);

    // Timing for Fully Connected Layer 1
    cudaEvent_t fc1_start, fc1_stop;
    cudaEventCreate(&fc1_start);
    cudaEventCreate(&fc1_stop);
    cudaEventRecord(fc1_start, 0);

    dense_layer_kernel<<<gridDim_fc, blockDim_fc>>>(d_pool2_output, d_fc1_weights, d_fc1_biases,
                                                   d_fc1_output, flattened_length, fully_connected1_neurons);
    cudaCheckError(cudaGetLastError());

    cudaEventRecord(fc1_stop, 0);
    cudaEventSynchronize(fc1_stop);
    float fc1_time;
    cudaEventElapsedTime(&fc1_time, fc1_start, fc1_stop);

    // Apply ReLU activation
    relu_activation<<<(fully_connected1_neurons + threads -1)/threads, threads>>>(d_fc1_output, fully_connected1_neurons);
    cudaCheckError(cudaGetLastError());

    // Fully Connected Layer 2
    // OUTPUT_NEURONS
    float *d_fc2_output;
    cudaCheckError(cudaMalloc((void**)&d_fc2_output, output_neurons * sizeof(float)));

    dim3 gridDim_fc2((output_neurons + blockDim_fc.x -1)/blockDim_fc.x);

    // Timing for Fully Connected Layer 2
    cudaEvent_t fc2_start, fc2_stop;
    cudaEventCreate(&fc2_start);
    cudaEventCreate(&fc2_stop);
    cudaEventRecord(fc2_start, 0);

    dense_layer_kernel<<<gridDim_fc2, blockDim_fc>>>(d_fc1_output, d_fc2_weights, d_fc2_biases,
                                                   d_fc2_output, fully_connected1_neurons, output_neurons);
    cudaCheckError(cudaGetLastError());

    cudaEventRecord(fc2_stop, 0);
    cudaEventSynchronize(fc2_stop);
    float fc2_time;
    cudaEventElapsedTime(&fc2_time, fc2_start, fc2_stop);

    // Apply Softmax activation
    // Assuming block size of 256 and grid size calculated accordingly
    int softmax_threads = 256;
    int softmax_blocks = (output_neurons + softmax_threads -1)/softmax_threads;
    softmax_activation<<<softmax_blocks, softmax_threads, softmax_threads * sizeof(float)>>>(d_fc2_output, output_neurons);
    cudaCheckError(cudaGetLastError());

    // Copy final output back to host
    cudaCheckError(cudaMemcpy(d_final_output, d_fc2_output, output_neurons * sizeof(float), cudaMemcpyDeviceToDevice));

    // Print timing information
    printf("Layer Execution Times (seconds):\n");
    printf("Convolution Layer 1: %f s\n", conv1_time/1000);
    printf("Pooling Layer 1:     %f s\n", pool1_time/ 1000);
    printf("Convolution Layer 2: %f s\n", conv2_time/ 1000);
    printf("Pooling Layer 2:     %f s\n", pool2_time/ 1000);
    printf("Fully Connected 1:   %f s\n", fc1_time/ 1000);
    printf("Fully Connected 2:   %f s\n", fc2_time/ 1000);
    printf("Total Execution Time: %f s\n", conv1_time/1000 + pool1_time/1000 + conv2_time/1000 + pool2_time/1000 + fc1_time/1000 + fc2_time/1000);

    // Free allocated device memory
    cudaFree(d_conv1_output);
    cudaFree(d_pool1_output);
    cudaFree(d_conv2_output);
    cudaFree(d_pool2_output);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_output);

    // Destroy CUDA events
    cudaEventDestroy(conv1_start);
    cudaEventDestroy(conv1_stop);
    cudaEventDestroy(pool1_start);
    cudaEventDestroy(pool1_stop);
    cudaEventDestroy(conv2_start);
    cudaEventDestroy(conv2_stop);
    cudaEventDestroy(pool2_start);
    cudaEventDestroy(pool2_stop);
    cudaEventDestroy(fc1_start);
    cudaEventDestroy(fc1_stop);
    cudaEventDestroy(fc2_start);
    cudaEventDestroy(fc2_stop);
}

// Main Function
int main(){
    srand(42); // Fixed seed to compare with sequential

    // Input and Weights Allocation on Host
    float *image = (float *)malloc(IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float));
    float *conv1_weights = (float *)malloc(LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *conv2_weights = (float *)malloc(LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    int fc1_input_size = LAYER2_CHANNELS * ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4) * ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4);
    float *fc1_weights = (float *)malloc(FULLY_CONNECTED1_NEURONS * fc1_input_size * sizeof(float));
    float *fc1_biases = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    float *fc2_weights = (float *)malloc(OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float));
    float *fc2_biases = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    float *final_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Random Initialization
    for(int i = 0; i < IMAGE_DIMENSION * IMAGE_DIMENSION; ++i){
        image[i] = ((float)rand() / RAND_MAX);
    }
    for(int i = 0; i < LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i){
        conv1_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    }
    for(int i = 0; i < LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i){
        conv2_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    }
    for(int i = 0; i < FULLY_CONNECTED1_NEURONS * fc1_input_size; ++i){
        fc1_weights[i] = generate_weight(fc1_input_size);
    }
    for(int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i){
        fc1_biases[i] = 0.05f;
    }
    for(int i = 0; i < OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS; ++i){
        fc2_weights[i] = generate_weight(FULLY_CONNECTED1_NEURONS);
    }
    for(int i = 0; i < OUTPUT_NEURONS; ++i){
        fc2_biases[i] = 0.05f;
    }

    // Device memory pointers
    float *d_image, *d_conv1_weights, *d_conv2_weights;
    float *d_fc1_weights, *d_fc1_biases, *d_fc2_weights, *d_fc2_biases, *d_final_output;

    // Allocate device memory
    cudaCheckError(cudaMalloc((void**)&d_image, IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_fc1_weights, FULLY_CONNECTED1_NEURONS * fc1_input_size * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_fc1_biases, FULLY_CONNECTED1_NEURONS * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_fc2_biases, OUTPUT_NEURONS * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_final_output, OUTPUT_NEURONS * sizeof(float)));

    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_image, image, IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_conv1_weights, conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_conv2_weights, conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_fc1_weights, fc1_weights, FULLY_CONNECTED1_NEURONS * fc1_input_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_fc1_biases, fc1_biases, FULLY_CONNECTED1_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_fc2_weights, fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_fc2_biases, fc2_biases, OUTPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));

    // Execute the Neural Network
    printf("Running CUDA Neural Network...\n");
    neural_network_cuda(d_image, d_conv1_weights, d_conv2_weights,
                       d_fc1_weights, d_fc1_biases,
                       d_fc2_weights, d_fc2_biases,
                       d_final_output,
                       IMAGE_DIMENSION, KERNEL_SIZE, LAYER1_CHANNELS,
                       LAYER2_CHANNELS, POOL_KERNEL,
                       FULLY_CONNECTED1_NEURONS, OUTPUT_NEURONS);

    // Copy final output back to host
    cudaCheckError(cudaMemcpy(final_output, d_final_output, OUTPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));

    // Display Output
    printf("Output Probabilities:\n");
    for(int i = 0; i < OUTPUT_NEURONS; ++i){
        printf("Class %d: %f\n", i, final_output[i]);
    }

    // Free Allocated Memory
    free(image);
    free(conv1_weights);
    free(conv2_weights);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    free(final_output);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_conv1_weights);
    cudaFree(d_conv2_weights);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_biases);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc2_biases);
    cudaFree(d_final_output);

    return 0;
}
