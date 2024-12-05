#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Neural Network Parameters
#define IMAGE_HEIGHT 227
#define IMAGE_WIDTH 227
#define IMAGE_CHANNELS 3

// AlexNet Layer Parameters
#define CONV1_FILTERS 96
#define CONV1_KERNEL_SIZE 11
#define CONV1_STRIDE 4
#define CONV1_PADDING 0

#define POOL1_KERNEL_SIZE 3
#define POOL1_STRIDE 2

#define CONV2_FILTERS 256
#define CONV2_KERNEL_SIZE 5
#define CONV2_STRIDE 1
#define CONV2_PADDING 2

#define POOL2_KERNEL_SIZE 3
#define POOL2_STRIDE 2

#define CONV3_FILTERS 384
#define CONV3_KERNEL_SIZE 3
#define CONV3_STRIDE 1
#define CONV3_PADDING 1

#define CONV4_FILTERS 384
#define CONV4_KERNEL_SIZE 3
#define CONV4_STRIDE 1
#define CONV4_PADDING 1

#define CONV5_FILTERS 256
#define CONV5_KERNEL_SIZE 3
#define CONV5_STRIDE 1
#define CONV5_PADDING 1

#define POOL3_KERNEL_SIZE 3
#define POOL3_STRIDE 2

#define FC6_NEURONS 4096
#define FC7_NEURONS 4096
#define OUTPUT_NEURONS 1000

// Macro Utilities
#define INDEX3(c, h, w, H, W) ((c)*(H)*(W) + (h)*(W) + (w))
#define FILTER_INDEX(o, c, k_h, k_w, C, K_H, K_W) ((o)*(C)*(K_H)*(K_W) + (c)*(K_H)*(K_W) + (k_h)*(K_W) + (k_w))
#define OUTPUT_INDEX(o, h, w, H, W) ((o)*(H)*(W) + (h)*(W) + (w))

// Randomized weight initialization with Uniform distribution
float generate_weight(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return ((float)rand() / (float)RAND_MAX) * 2 * limit - limit;
}

void load_data(const char *filename, float *data, size_t size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }
    fread(data, sizeof(float), size, fp);
    fclose(fp);
}

// Function to save timing information to a file
void save_timing_info(const char *filename, float conv1_time, float pool1_time, float conv2_time, float pool2_time, float conv3_time, float conv4_time, float conv5_time, float pool3_time, float fc6_time, float fc7_time, float fc8_time, float total_time) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "Layer Execution Times (seconds):\n");
    fprintf(fp, "Convolution Layer 1: %f\n", conv1_time);
    fprintf(fp, "Pooling Layer 1:     %f\n", pool1_time);
    fprintf(fp, "Convolution Layer 2: %f\n", conv2_time);
    fprintf(fp, "Pooling Layer 2:     %f\n", pool2_time);
    fprintf(fp, "Convolution Layer 3: %f\n", conv3_time);
    fprintf(fp, "Convolution Layer 4: %f\n", conv4_time);
    fprintf(fp, "Convolution Layer 5: %f\n", conv5_time);
    fprintf(fp, "Pooling Layer 3:     %f\n", pool3_time);
    fprintf(fp, "Fully Connected 6:   %f\n", fc6_time);
    fprintf(fp, "Fully Connected 7:   %f\n", fc7_time);
    fprintf(fp, "Fully Connected 8:   %f\n", fc8_time);
    fprintf(fp, "Total Execution Time: %f\n", total_time);
    fclose(fp);
}

// Activation Functions
__global__ void activate_relu(float *vector, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        vector[idx] = (vector[idx] > 0) ? vector[idx] : 0;
    }
}

// Add biases and apply ReLU activation
__global__ void add_bias_and_relu(float *output, float *biases, int output_channels, int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = output_channels * output_height * output_width;
    if (idx < total_elements) {
        int o = idx / (output_height * output_width);
        output[idx] += biases[o];
        output[idx] = fmaxf(0.0f, output[idx]); // ReLU
    }
}

// Max Pooling Operation
__global__ void pooling_operation(float *input, float *output, int input_channels, int input_height, int input_width,
                                  int pool_size, int stride, int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = input_channels * output_height * output_width;
    if (idx < total_elements) {
        int c = idx / (output_height * output_width);
        int h = (idx % (output_height * output_width)) / output_width;
        int w = idx % output_width;

        float max_value = -INFINITY;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int in_h = h * stride + i;
                int in_w = w * stride + j;
                if (in_h < input_height && in_w < input_width) {
                    float value = input[INDEX3(c, in_h, in_w, input_height, input_width)];
                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }
        }
        output[INDEX3(c, h, w, output_height, output_width)] = max_value;
    }
}

// Convolution Operation
__global__ void convolution_operation(float *input, float *filters, float *output,
                                      int input_height, int input_width, int input_channels,
                                      int output_channels, int kernel_size, int stride, int padding,
                                      int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = output_channels * output_height * output_width;
    if (idx < total_elements) {
        int o = idx / (output_height * output_width);
        int h = (idx % (output_height * output_width)) / output_width;
        int w = idx % output_width;

        float acc = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int in_h = h * stride + k_h - padding;
                    int in_w = w * stride + k_w - padding;
                    if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        float input_value = input[INDEX3(c, in_h, in_w, input_height, input_width)];
                        float filter_value = filters[FILTER_INDEX(o, c, k_h, k_w, input_channels, kernel_size, kernel_size)];
                        acc += input_value * filter_value;
                    }
                }
            }
        }
        output[OUTPUT_INDEX(o, h, w, output_height, output_width)] = acc;
    }
}

// Fully Connected Layer
__global__ void dense_layer(float *input, float *weights, float *biases, float *output, int input_length, int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_length) {
        float sum = biases[idx];
        for (int i = 0; i < input_length; ++i) {
            sum += input[i] * weights[idx * input_length + i];
        }
        output[idx] = sum;
    }
}

// Apply softmax on host
void apply_softmax_function(float *input, int length) {
    float max_value = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_value) max_value = input[i];
    }

    float sum_exponentials = 0.0f;
    for (int i = 0; i < length; ++i) {
        input[i] = expf(input[i] - max_value);
        sum_exponentials += input[i];
    }
    for (int i = 0; i < length; ++i) {
        input[i] /= sum_exponentials;
    }
}

// Complete Model
void neural_network(float *d_input,
                    float *d_conv1_weights, float *d_conv1_biases,
                    float *d_conv2_weights, float *d_conv2_biases,
                    float *d_conv3_weights, float *d_conv3_biases,
                    float *d_conv4_weights, float *d_conv4_biases,
                    float *d_conv5_weights, float *d_conv5_biases,
                    float *d_fc6_weights, float *d_fc6_biases,
                    float *d_fc7_weights, float *d_fc7_biases,
                    float *d_fc8_weights, float *d_fc8_biases,
                    float *final_output) {

    cudaEvent_t total_start, total_end;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);
    cudaEventRecord(total_start);

    float conv1_time = 0, pool1_time = 0, conv2_time = 0, pool2_time = 0;
    float conv3_time = 0, conv4_time = 0, conv5_time = 0, pool3_time = 0;
    float fc6_time = 0, fc7_time = 0, fc8_time = 0, total_time = 0;

    // Layer 1: Convolution + ReLU + Pooling
    cudaEvent_t conv1_start, conv1_end;
    cudaEventCreate(&conv1_start);
    cudaEventCreate(&conv1_end);
    cudaEventRecord(conv1_start);

    int conv1_output_height = (IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1;
    int conv1_output_width = (IMAGE_WIDTH + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1;
    int conv1_output_size = CONV1_FILTERS * conv1_output_height * conv1_output_width;

    float *d_conv1_output;
    cudaMalloc(&d_conv1_output, conv1_output_size * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (conv1_output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_operation<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_conv1_weights, d_conv1_output,
                              IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                              CONV1_FILTERS, CONV1_KERNEL_SIZE, CONV1_STRIDE, CONV1_PADDING,
                              conv1_output_height, conv1_output_width);

    // Add biases and ReLU
    add_bias_and_relu<<<blocksPerGrid, threadsPerBlock>>>(d_conv1_output, d_conv1_biases, CONV1_FILTERS, conv1_output_height, conv1_output_width);

    cudaEventRecord(conv1_end);
    cudaEventSynchronize(conv1_end);
    cudaEventElapsedTime(&conv1_time, conv1_start, conv1_end);
    conv1_time /= 1000.0f; // Convert to seconds

    // Pooling
    cudaEvent_t pool1_start, pool1_end;
    cudaEventCreate(&pool1_start);
    cudaEventCreate(&pool1_end);
    cudaEventRecord(pool1_start);

    int pool1_output_height = (conv1_output_height - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1;
    int pool1_output_width = (conv1_output_width - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1;
    int pool1_output_size = CONV1_FILTERS * pool1_output_height * pool1_output_width;
    float *d_pool1_output;
    cudaMalloc(&d_pool1_output, pool1_output_size * sizeof(float));

    int pool1_threadsPerBlock = 256;
    int pool1_blocksPerGrid = (pool1_output_size + pool1_threadsPerBlock - 1) / pool1_threadsPerBlock;

    pooling_operation<<<pool1_blocksPerGrid, pool1_threadsPerBlock>>>(d_conv1_output, d_pool1_output, CONV1_FILTERS, conv1_output_height, conv1_output_width,
                          POOL1_KERNEL_SIZE, POOL1_STRIDE, pool1_output_height, pool1_output_width);

    cudaFree(d_conv1_output);

    cudaEventRecord(pool1_end);
    cudaEventSynchronize(pool1_end);
    cudaEventElapsedTime(&pool1_time, pool1_start, pool1_end);
    pool1_time /= 1000.0f; // Convert to seconds

    // Layer 2: Convolution + ReLU + Pooling
    cudaEvent_t conv2_start, conv2_end;
    cudaEventCreate(&conv2_start);
    cudaEventCreate(&conv2_end);
    cudaEventRecord(conv2_start);

    int conv2_output_height = (pool1_output_height + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
    int conv2_output_width = (pool1_output_width + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
    int conv2_output_size = CONV2_FILTERS * conv2_output_height * conv2_output_width;
    float *d_conv2_output;
    cudaMalloc(&d_conv2_output, conv2_output_size * sizeof(float));

    int conv2_threadsPerBlock = 256;
    int conv2_blocksPerGrid = (conv2_output_size + conv2_threadsPerBlock - 1) / conv2_threadsPerBlock;

    convolution_operation<<<conv2_blocksPerGrid, conv2_threadsPerBlock>>>(d_pool1_output, d_conv2_weights, d_conv2_output,
                              pool1_output_height, pool1_output_width, CONV1_FILTERS,
                              CONV2_FILTERS, CONV2_KERNEL_SIZE, CONV2_STRIDE, CONV2_PADDING,
                              conv2_output_height, conv2_output_width);

    // Add biases and ReLU
    add_bias_and_relu<<<conv2_blocksPerGrid, conv2_threadsPerBlock>>>(d_conv2_output, d_conv2_biases, CONV2_FILTERS, conv2_output_height, conv2_output_width);

    cudaEventRecord(conv2_end);
    cudaEventSynchronize(conv2_end);
    cudaEventElapsedTime(&conv2_time, conv2_start, conv2_end);
    conv2_time /= 1000.0f; // Convert to seconds

    // Pooling
    cudaEvent_t pool2_start, pool2_end;
    cudaEventCreate(&pool2_start);
    cudaEventCreate(&pool2_end);
    cudaEventRecord(pool2_start);

    int pool2_output_height = (conv2_output_height - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    int pool2_output_width = (conv2_output_width - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    int pool2_output_size = CONV2_FILTERS * pool2_output_height * pool2_output_width;
    float *d_pool2_output;
    cudaMalloc(&d_pool2_output, pool2_output_size * sizeof(float));

    int pool2_threadsPerBlock = 256;
    int pool2_blocksPerGrid = (pool2_output_size + pool2_threadsPerBlock - 1) / pool2_threadsPerBlock;

    pooling_operation<<<pool2_blocksPerGrid, pool2_threadsPerBlock>>>(d_conv2_output, d_pool2_output, CONV2_FILTERS, conv2_output_height, conv2_output_width,
                          POOL2_KERNEL_SIZE, POOL2_STRIDE, pool2_output_height, pool2_output_width);

    cudaFree(d_conv2_output);
    cudaFree(d_pool1_output);

    cudaEventRecord(pool2_end);
    cudaEventSynchronize(pool2_end);
    cudaEventElapsedTime(&pool2_time, pool2_start, pool2_end);
    pool2_time /= 1000.0f; // Convert to seconds

    // Layer 3: Convolution + ReLU
    cudaEvent_t conv3_start, conv3_end;
    cudaEventCreate(&conv3_start);
    cudaEventCreate(&conv3_end);
    cudaEventRecord(conv3_start);

    int conv3_output_height = (pool2_output_height + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1;
    int conv3_output_width = (pool2_output_width + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1;
    int conv3_output_size = CONV3_FILTERS * conv3_output_height * conv3_output_width;
    float *d_conv3_output;
    cudaMalloc(&d_conv3_output, conv3_output_size * sizeof(float));

    int conv3_threadsPerBlock = 256;
    int conv3_blocksPerGrid = (conv3_output_size + conv3_threadsPerBlock - 1) / conv3_threadsPerBlock;

    convolution_operation<<<conv3_blocksPerGrid, conv3_threadsPerBlock>>>(d_pool2_output, d_conv3_weights, d_conv3_output,
                              pool2_output_height, pool2_output_width, CONV2_FILTERS,
                              CONV3_FILTERS, CONV3_KERNEL_SIZE, CONV3_STRIDE, CONV3_PADDING,
                              conv3_output_height, conv3_output_width);

    // Add biases and ReLU
    add_bias_and_relu<<<conv3_blocksPerGrid, conv3_threadsPerBlock>>>(d_conv3_output, d_conv3_biases, CONV3_FILTERS, conv3_output_height, conv3_output_width);

    cudaFree(d_pool2_output);

    cudaEventRecord(conv3_end);
    cudaEventSynchronize(conv3_end);
    cudaEventElapsedTime(&conv3_time, conv3_start, conv3_end);
    conv3_time /= 1000.0f; // Convert to seconds

    // Layer 4: Convolution + ReLU
    cudaEvent_t conv4_start, conv4_end;
    cudaEventCreate(&conv4_start);
    cudaEventCreate(&conv4_end);
    cudaEventRecord(conv4_start);

    int conv4_output_height = (conv3_output_height + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1;
    int conv4_output_width = (conv3_output_width + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1;
    int conv4_output_size = CONV4_FILTERS * conv4_output_height * conv4_output_width;
    float *d_conv4_output;
    cudaMalloc(&d_conv4_output, conv4_output_size * sizeof(float));

    int conv4_threadsPerBlock = 256;
    int conv4_blocksPerGrid = (conv4_output_size + conv4_threadsPerBlock - 1) / conv4_threadsPerBlock;

    convolution_operation<<<conv4_blocksPerGrid, conv4_threadsPerBlock>>>(d_conv3_output, d_conv4_weights, d_conv4_output,
                              conv3_output_height, conv3_output_width, CONV3_FILTERS,
                              CONV4_FILTERS, CONV4_KERNEL_SIZE, CONV4_STRIDE, CONV4_PADDING,
                              conv4_output_height, conv4_output_width);

    // Add biases and ReLU
    add_bias_and_relu<<<conv4_blocksPerGrid, conv4_threadsPerBlock>>>(d_conv4_output, d_conv4_biases, CONV4_FILTERS, conv4_output_height, conv4_output_width);

    cudaFree(d_conv3_output);

    cudaEventRecord(conv4_end);
    cudaEventSynchronize(conv4_end);
    cudaEventElapsedTime(&conv4_time, conv4_start, conv4_end);
    conv4_time /= 1000.0f; // Convert to seconds

    // Layer 5: Convolution + ReLU + Pooling
    cudaEvent_t conv5_start, conv5_end;
    cudaEventCreate(&conv5_start);
    cudaEventCreate(&conv5_end);
    cudaEventRecord(conv5_start);

    int conv5_output_height = (conv4_output_height + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1;
    int conv5_output_width = (conv4_output_width + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1;
    int conv5_output_size = CONV5_FILTERS * conv5_output_height * conv5_output_width;
    float *d_conv5_output;
    cudaMalloc(&d_conv5_output, conv5_output_size * sizeof(float));

    int conv5_threadsPerBlock = 256;
    int conv5_blocksPerGrid = (conv5_output_size + conv5_threadsPerBlock - 1) / conv5_threadsPerBlock;

    convolution_operation<<<conv5_blocksPerGrid, conv5_threadsPerBlock>>>(d_conv4_output, d_conv5_weights, d_conv5_output,
                              conv4_output_height, conv4_output_width, CONV4_FILTERS,
                              CONV5_FILTERS, CONV5_KERNEL_SIZE, CONV5_STRIDE, CONV5_PADDING,
                              conv5_output_height, conv5_output_width);

    // Add biases and ReLU
    add_bias_and_relu<<<conv5_blocksPerGrid, conv5_threadsPerBlock>>>(d_conv5_output, d_conv5_biases, CONV5_FILTERS, conv5_output_height, conv5_output_width);

    cudaFree(d_conv4_output);

    cudaEventRecord(conv5_end);
    cudaEventSynchronize(conv5_end);
    cudaEventElapsedTime(&conv5_time, conv5_start, conv5_end);
    conv5_time /= 1000.0f; // Convert to seconds

    // Pooling
    cudaEvent_t pool3_start, pool3_end;
    cudaEventCreate(&pool3_start);
    cudaEventCreate(&pool3_end);
    cudaEventRecord(pool3_start);

    int pool3_output_height = (conv5_output_height - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;
    int pool3_output_width = (conv5_output_width - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;
    int pool3_output_size = CONV5_FILTERS * pool3_output_height * pool3_output_width;
    float *d_pool3_output;
    cudaMalloc(&d_pool3_output, pool3_output_size * sizeof(float));

    int pool3_threadsPerBlock = 256;
    int pool3_blocksPerGrid = (pool3_output_size + pool3_threadsPerBlock - 1) / pool3_threadsPerBlock;

    pooling_operation<<<pool3_blocksPerGrid, pool3_threadsPerBlock>>>(d_conv5_output, d_pool3_output, CONV5_FILTERS, conv5_output_height, conv5_output_width,
                          POOL3_KERNEL_SIZE, POOL3_STRIDE, pool3_output_height, pool3_output_width);

    cudaFree(d_conv5_output);

    cudaEventRecord(pool3_end);
    cudaEventSynchronize(pool3_end);
    cudaEventElapsedTime(&pool3_time, pool3_start, pool3_end);
    pool3_time /= 1000.0f; // Convert to seconds

    // Flatten the output for Fully Connected Layers
    int flattened_length = CONV5_FILTERS * pool3_output_height * pool3_output_width;
    float *h_flattened_output = (float *)malloc(flattened_length * sizeof(float));
    cudaMemcpy(h_flattened_output, d_pool3_output, flattened_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_pool3_output);

    // Fully Connected Layer 6
    cudaEvent_t fc6_start, fc6_end;
    cudaEventCreate(&fc6_start);
    cudaEventCreate(&fc6_end);
    cudaEventRecord(fc6_start);

    float *d_fc6_output;
    cudaMalloc(&d_fc6_output, FC6_NEURONS * sizeof(float));

    int fc6_blocks = (FC6_NEURONS + 255) / 256;
    float *d_flattened_output;
    cudaMalloc(&d_flattened_output, flattened_length * sizeof(float));
    cudaMemcpy(d_flattened_output, h_flattened_output, flattened_length * sizeof(float), cudaMemcpyHostToDevice);
    free(h_flattened_output);

    dense_layer<<<fc6_blocks, 256>>>(d_flattened_output, d_fc6_weights, d_fc6_biases, d_fc6_output, flattened_length, FC6_NEURONS);

    activate_relu<<<(FC6_NEURONS + 255) / 256, 256>>>(d_fc6_output, FC6_NEURONS);

    cudaFree(d_flattened_output);

    cudaEventRecord(fc6_end);
    cudaEventSynchronize(fc6_end);
    cudaEventElapsedTime(&fc6_time, fc6_start, fc6_end);
    fc6_time /= 1000.0f; // Convert to seconds

    // Fully Connected Layer 7
    cudaEvent_t fc7_start, fc7_end;
    cudaEventCreate(&fc7_start);
    cudaEventCreate(&fc7_end);
    cudaEventRecord(fc7_start);

    float *d_fc7_output;
    cudaMalloc(&d_fc7_output, FC7_NEURONS * sizeof(float));

    int fc7_blocks = (FC7_NEURONS + 255) / 256;
    dense_layer<<<fc7_blocks, 256>>>(d_fc6_output, d_fc7_weights, d_fc7_biases, d_fc7_output, FC6_NEURONS, FC7_NEURONS);

    activate_relu<<<(FC7_NEURONS + 255) / 256, 256>>>(d_fc7_output, FC7_NEURONS);

    cudaFree(d_fc6_output);

    cudaEventRecord(fc7_end);
    cudaEventSynchronize(fc7_end);
    cudaEventElapsedTime(&fc7_time, fc7_start, fc7_end);
    fc7_time /= 1000.0f; // Convert to seconds

    // Fully Connected Layer 8 (Output Layer)
    cudaEvent_t fc8_start, fc8_end;
    cudaEventCreate(&fc8_start);
    cudaEventCreate(&fc8_end);
    cudaEventRecord(fc8_start);

    float *d_fc8_output;
    cudaMalloc(&d_fc8_output, OUTPUT_NEURONS * sizeof(float));

    int fc8_blocks = (OUTPUT_NEURONS + 255) / 256;
    dense_layer<<<fc8_blocks, 256>>>(d_fc7_output, d_fc8_weights, d_fc8_biases, d_fc8_output, FC7_NEURONS, OUTPUT_NEURONS);

    cudaFree(d_fc7_output);

    cudaEventRecord(fc8_end);
    cudaEventSynchronize(fc8_end);
    cudaEventElapsedTime(&fc8_time, fc8_start, fc8_end);
    fc8_time /= 1000.0f; // Convert to seconds

    // Copy final output to host
    float *h_fc8_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    cudaMemcpy(h_fc8_output, d_fc8_output, OUTPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fc8_output);

    // Apply softmax on host
    apply_softmax_function(h_fc8_output, OUTPUT_NEURONS);

    // Copy to final_output
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        final_output[i] = h_fc8_output[i];
    }

    free(h_fc8_output);

    cudaEventRecord(total_end);
    cudaEventSynchronize(total_end);
    cudaEventElapsedTime(&total_time, total_start, total_end);
    total_time /= 1000.0f; // Convert to seconds

    // Print timing information and save them in a file
    printf("Layer Execution Times (seconds):\n");
    printf("Convolution Layer 1: %f\n", conv1_time);
    printf("Pooling Layer 1:     %f\n", pool1_time);
    printf("Convolution Layer 2: %f\n", conv2_time);
    printf("Pooling Layer 2:     %f\n", pool2_time);
    printf("Convolution Layer 3: %f\n", conv3_time);
    printf("Convolution Layer 4: %f\n", conv4_time);
    printf("Convolution Layer 5: %f\n", conv5_time);
    printf("Pooling Layer 3:     %f\n", pool3_time);
    printf("Fully Connected 6:   %f\n", fc6_time);
    printf("Fully Connected 7:   %f\n", fc7_time);
    printf("Fully Connected 8:   %f\n", fc8_time);
    printf("Total Execution Time: %f\n", total_time);

    // Save timing information to a file
    save_timing_info("../Data/cuda_execution_time.txt", conv1_time, pool1_time, conv2_time, pool2_time, conv3_time, conv4_time, conv5_time, pool3_time, fc6_time, fc7_time, fc8_time, total_time);

}



// Function to save output to a text file
void save_output(const char *filename, float *data, size_t size) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < size; ++i) {
        fprintf(fp, "%f\n", data[i]);
    }
    fclose(fp);
}

// Main Function
int main() {
    srand(42);  // Fixed seed for reproducibility

    // Input and Weights Allocation
    size_t image_size = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
    float *image = (float *)malloc(image_size * sizeof(float));

    // Allocate weights and biases for all layers
    size_t conv1_weights_size = CONV1_FILTERS * IMAGE_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    float *conv1_weights = (float *)malloc(conv1_weights_size * sizeof(float));
    float *conv1_biases = (float *)malloc(CONV1_FILTERS * sizeof(float));

    size_t conv2_weights_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    float *conv2_weights = (float *)malloc(conv2_weights_size * sizeof(float));
    float *conv2_biases = (float *)malloc(CONV2_FILTERS * sizeof(float));

    size_t conv3_weights_size = CONV3_FILTERS * CONV2_FILTERS * CONV3_KERNEL_SIZE * CONV3_KERNEL_SIZE;
    float *conv3_weights = (float *)malloc(conv3_weights_size * sizeof(float));
    float *conv3_biases = (float *)malloc(CONV3_FILTERS * sizeof(float));

    size_t conv4_weights_size = CONV4_FILTERS * CONV3_FILTERS * CONV4_KERNEL_SIZE * CONV4_KERNEL_SIZE;
    float *conv4_weights = (float *)malloc(conv4_weights_size * sizeof(float));
    float *conv4_biases = (float *)malloc(CONV4_FILTERS * sizeof(float));

    size_t conv5_weights_size = CONV5_FILTERS * CONV4_FILTERS * CONV5_KERNEL_SIZE * CONV5_KERNEL_SIZE;
    float *conv5_weights = (float *)malloc(conv5_weights_size * sizeof(float));
    float *conv5_biases = (float *)malloc(CONV5_FILTERS * sizeof(float));

    int conv1_output_height = (IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1;
    int pool1_output_height = (conv1_output_height - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1;
    int conv2_output_height = (pool1_output_height + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
    int pool2_output_height = (conv2_output_height - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    int conv3_output_height = (pool2_output_height + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1;
    int conv4_output_height = (conv3_output_height + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1;
    int conv5_output_height = (conv4_output_height + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1;
    int pool3_output_height = (conv5_output_height - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;

    int pool3_output_width = pool3_output_height;  // Since width and height remain equal
    int flattened_size = CONV5_FILTERS * pool3_output_height * pool3_output_width;

    size_t fc6_weights_size = FC6_NEURONS * flattened_size;
    float *fc6_weights = (float *)malloc(fc6_weights_size * sizeof(float));
    float *fc6_biases = (float *)malloc(FC6_NEURONS * sizeof(float));

    size_t fc7_weights_size = FC7_NEURONS * FC6_NEURONS;
    float *fc7_weights = (float *)malloc(fc7_weights_size * sizeof(float));
    float *fc7_biases = (float *)malloc(FC7_NEURONS * sizeof(float));

    size_t fc8_weights_size = OUTPUT_NEURONS * FC7_NEURONS;
    float *fc8_weights = (float *)malloc(fc8_weights_size * sizeof(float));
    float *fc8_biases = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    float *final_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Random Initialization
    for (size_t i = 0; i < image_size; ++i)
        image[i] = ((float)rand() / RAND_MAX);

    for (size_t i = 0; i < conv1_weights_size; ++i)
        conv1_weights[i] = generate_weight(IMAGE_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE);
    for (int i = 0; i < CONV1_FILTERS; ++i)
        conv1_biases[i] = 0.0f;

    for (size_t i = 0; i < conv2_weights_size; ++i)
        conv2_weights[i] = generate_weight(CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE);
    for (int i = 0; i < CONV2_FILTERS; ++i)
        conv2_biases[i] = 0.0f;

    for (size_t i = 0; i < conv3_weights_size; ++i)
        conv3_weights[i] = generate_weight(CONV2_FILTERS * CONV3_KERNEL_SIZE * CONV3_KERNEL_SIZE);
    for (int i = 0; i < CONV3_FILTERS; ++i)
        conv3_biases[i] = 0.0f;

    for (size_t i = 0; i < conv4_weights_size; ++i)
        conv4_weights[i] = generate_weight(CONV3_FILTERS * CONV4_KERNEL_SIZE * CONV4_KERNEL_SIZE);
    for (int i = 0; i < CONV4_FILTERS; ++i)
        conv4_biases[i] = 0.0f;

    for (size_t i = 0; i < conv5_weights_size; ++i)
        conv5_weights[i] = generate_weight(CONV4_FILTERS * CONV5_KERNEL_SIZE * CONV5_KERNEL_SIZE);
    for (int i = 0; i < CONV5_FILTERS; ++i)
        conv5_biases[i] = 0.0f;

    for (size_t i = 0; i < fc6_weights_size; ++i)
        fc6_weights[i] = generate_weight(flattened_size);
    for (int i = 0; i < FC6_NEURONS; ++i)
        fc6_biases[i] = 0.0f;

    for (size_t i = 0; i < fc7_weights_size; ++i)
        fc7_weights[i] = generate_weight(FC6_NEURONS);
    for (int i = 0; i < FC7_NEURONS; ++i)
        fc7_biases[i] = 0.0f;

    for (size_t i = 0; i < fc8_weights_size; ++i)
        fc8_weights[i] = generate_weight(FC7_NEURONS);
    for (int i = 0; i < OUTPUT_NEURONS; ++i)
        fc8_biases[i] = 0.0f;


    // Allocate device memory and copy data
    float *d_image;
    cudaMalloc(&d_image, image_size * sizeof(float));
    cudaMemcpy(d_image, image, image_size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_conv1_weights, *d_conv1_biases;
    cudaMalloc(&d_conv1_weights, conv1_weights_size * sizeof(float));
    cudaMalloc(&d_conv1_biases, CONV1_FILTERS * sizeof(float));
    cudaMemcpy(d_conv1_weights, conv1_weights, conv1_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_biases, conv1_biases, CONV1_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_conv2_weights, *d_conv2_biases;
    cudaMalloc(&d_conv2_weights, conv2_weights_size * sizeof(float));
    cudaMalloc(&d_conv2_biases, CONV2_FILTERS * sizeof(float));
    cudaMemcpy(d_conv2_weights, conv2_weights, conv2_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_biases, conv2_biases, CONV2_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_conv3_weights, *d_conv3_biases;
    cudaMalloc(&d_conv3_weights, conv3_weights_size * sizeof(float));
    cudaMalloc(&d_conv3_biases, CONV3_FILTERS * sizeof(float));
    cudaMemcpy(d_conv3_weights, conv3_weights, conv3_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_biases, conv3_biases, CONV3_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_conv4_weights, *d_conv4_biases;
    cudaMalloc(&d_conv4_weights, conv4_weights_size * sizeof(float));
    cudaMalloc(&d_conv4_biases, CONV4_FILTERS * sizeof(float));
    cudaMemcpy(d_conv4_weights, conv4_weights, conv4_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_biases, conv4_biases, CONV4_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_conv5_weights, *d_conv5_biases;
    cudaMalloc(&d_conv5_weights, conv5_weights_size * sizeof(float));
    cudaMalloc(&d_conv5_biases, CONV5_FILTERS * sizeof(float));
    cudaMemcpy(d_conv5_weights, conv5_weights, conv5_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_biases, conv5_biases, CONV5_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fc6_weights, *d_fc6_biases;
    cudaMalloc(&d_fc6_weights, fc6_weights_size * sizeof(float));
    cudaMalloc(&d_fc6_biases, FC6_NEURONS * sizeof(float));
    cudaMemcpy(d_fc6_weights, fc6_weights, fc6_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc6_biases, fc6_biases, FC6_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fc7_weights, *d_fc7_biases;
    cudaMalloc(&d_fc7_weights, fc7_weights_size * sizeof(float));
    cudaMalloc(&d_fc7_biases, FC7_NEURONS * sizeof(float));
    cudaMemcpy(d_fc7_weights, fc7_weights, fc7_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc7_biases, fc7_biases, FC7_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fc8_weights, *d_fc8_biases;
    cudaMalloc(&d_fc8_weights, fc8_weights_size * sizeof(float));
    cudaMalloc(&d_fc8_biases, OUTPUT_NEURONS * sizeof(float));
    cudaMemcpy(d_fc8_weights, fc8_weights, fc8_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc8_biases, fc8_biases, OUTPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the Neural Network
    printf("Running AlexNet Neural Network...\n");
    neural_network(d_image,
                   d_conv1_weights, d_conv1_biases,
                   d_conv2_weights, d_conv2_biases,
                   d_conv3_weights, d_conv3_biases,
                   d_conv4_weights, d_conv4_biases,
                   d_conv5_weights, d_conv5_biases,
                   d_fc6_weights, d_fc6_biases,
                   d_fc7_weights, d_fc7_biases,
                   d_fc8_weights, d_fc8_biases,
                   final_output);

    // Display Output
    printf("Output Probabilities:\n");
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        printf("Class %d: %f\n", i, final_output[i]);
    }

    save_output("../Data/cuda_output.txt", final_output, OUTPUT_NEURONS);

    // Free Allocated Memory
    free(image);
    free(conv1_weights);
    free(conv1_biases);
    free(conv2_weights);
    free(conv2_biases);
    free(conv3_weights);
    free(conv3_biases);
    free(conv4_weights);
    free(conv4_biases);
    free(conv5_weights);
    free(conv5_biases);
    free(fc6_weights);
    free(fc6_biases);
    free(fc7_weights);
    free(fc7_biases);
    free(fc8_weights);
    free(fc8_biases);
    free(final_output);

    cudaFree(d_image);
    cudaFree(d_conv1_weights);
    cudaFree(d_conv1_biases);
    cudaFree(d_conv2_weights);
    cudaFree(d_conv2_biases);
    cudaFree(d_conv3_weights);
    cudaFree(d_conv3_biases);
    cudaFree(d_conv4_weights);
    cudaFree(d_conv4_biases);
    cudaFree(d_conv5_weights);
    cudaFree(d_conv5_biases);
    cudaFree(d_fc6_weights);
    cudaFree(d_fc6_biases);
    cudaFree(d_fc7_weights);
    cudaFree(d_fc7_biases);
    cudaFree(d_fc8_weights);
    cudaFree(d_fc8_biases);

    return 0;
}
