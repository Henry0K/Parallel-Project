#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
#define MAX_VAL(a, b) (((a) > (b)) ? (a) : (b))
#define INDEX3(c, h, w, H, W) ((c)*(H)*(W) + (h)*(W) + (w))
#define FILTER_INDEX(o, c, k_h, k_w, C, K_H, K_W) ((o)*(C)*(K_H)*(K_W) + (c)*(K_H)*(K_W) + (k_h)*(K_W) + (k_w))
#define OUTPUT_INDEX(o, h, w, H, W) ((o)*(H)*(W) + (h)*(W) + (w))

// Randomized weight initialization with Uniform distribution
float generate_weight(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return ((float)rand() / (float)RAND_MAX) * 2 * limit - limit;
}

// Activation Functions
void activate_relu(float *vector, int length) {
    for (int i = 0; i < length; ++i) {
        vector[i] = (vector[i] > 0) ? vector[i] : 0;
    }
}

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

// Convolution Operation
void convolution_operation(float *input, float *filters, float *output,
                           int input_height, int input_width, int input_channels,
                           int output_channels, int kernel_size, int stride, int padding) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    for (int o = 0; o < output_channels; ++o) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
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
    }
}

// Max Pooling Operation
void pooling_operation(float *input, float *output, int input_channels, int input_height, int input_width,
                       int pool_size, int stride) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    for (int c = 0; c < input_channels; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
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
    }
}

// Fully Connected Layer
void dense_layer(float *input, float *weights, float *biases, float *output, int input_length, int output_length) {
    for (int o = 0; o < output_length; ++o) {
        float sum = biases[o];
        for (int i = 0; i < input_length; ++i) {
            sum += input[i] * weights[o * input_length + i];
        }
        output[o] = sum;
    }
}

// Complete Model
void neural_network(float *input,
                    float *conv1_weights, float *conv1_biases,
                    float *conv2_weights, float *conv2_biases,
                    float *conv3_weights, float *conv3_biases,
                    float *conv4_weights, float *conv4_biases,
                    float *conv5_weights, float *conv5_biases,
                    float *fc6_weights, float *fc6_biases,
                    float *fc7_weights, float *fc7_biases,
                    float *fc8_weights, float *fc8_biases,
                    float *final_output) {

    clock_t total_start_time = clock();
    // Layer 1: Convolution + ReLU + Pooling
    clock_t conv1_start_time = clock();
    int conv1_output_height = (IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1;
    int conv1_output_width = (IMAGE_WIDTH + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1;
    int conv1_output_size = CONV1_FILTERS * conv1_output_height * conv1_output_width;
    float *conv1_output = (float *)malloc(conv1_output_size * sizeof(float));

    convolution_operation(input, conv1_weights, conv1_output,
                          IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                          CONV1_FILTERS, CONV1_KERNEL_SIZE, CONV1_STRIDE, CONV1_PADDING);
    clock_t conv1_end_time = clock();
    double conv1_time = (double)(conv1_end_time - conv1_start_time) / CLOCKS_PER_SEC;

    // Add biases and ReLU
    for (int i = 0; i < conv1_output_size; ++i) {
        conv1_output[i] += conv1_biases[i / (conv1_output_height * conv1_output_width)];
    }
    activate_relu(conv1_output, conv1_output_size);

    clock_t pool1_start_time = clock();
    int pool1_output_height = (conv1_output_height - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1;
    int pool1_output_width = (conv1_output_width - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1;
    int pool1_output_size = CONV1_FILTERS * pool1_output_height * pool1_output_width;
    float *pool1_output = (float *)malloc(pool1_output_size * sizeof(float));

    pooling_operation(conv1_output, pool1_output, CONV1_FILTERS, conv1_output_height, conv1_output_width,
                      POOL1_KERNEL_SIZE, POOL1_STRIDE);

    free(conv1_output);
    clock_t pool1_end_time = clock();
    double pool1_time = (double)(pool1_end_time - pool1_start_time) / CLOCKS_PER_SEC;

    // Layer 2: Convolution + ReLU + Pooling
    clock_t conv2_start_time = clock();
    int conv2_output_height = (pool1_output_height + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
    int conv2_output_width = (pool1_output_width + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
    int conv2_output_size = CONV2_FILTERS * conv2_output_height * conv2_output_width;
    float *conv2_output = (float *)malloc(conv2_output_size * sizeof(float));

    convolution_operation(pool1_output, conv2_weights, conv2_output,
                          pool1_output_height, pool1_output_width, CONV1_FILTERS,
                          CONV2_FILTERS, CONV2_KERNEL_SIZE, CONV2_STRIDE, CONV2_PADDING);
    clock_t conv2_end_time = clock();
    double conv2_time = (double)(conv2_end_time - conv2_start_time) / CLOCKS_PER_SEC;

    // Add biases and ReLU
    for (int i = 0; i < conv2_output_size; ++i) {
        conv2_output[i] += conv2_biases[i / (conv2_output_height * conv2_output_width)];
    }
    activate_relu(conv2_output, conv2_output_size);

    clock_t pool2_start_time = clock();
    int pool2_output_height = (conv2_output_height - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    int pool2_output_width = (conv2_output_width - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    int pool2_output_size = CONV2_FILTERS * pool2_output_height * pool2_output_width;
    float *pool2_output = (float *)malloc(pool2_output_size * sizeof(float));

    pooling_operation(conv2_output, pool2_output, CONV2_FILTERS, conv2_output_height, conv2_output_width,
                      POOL2_KERNEL_SIZE, POOL2_STRIDE);

    free(conv2_output);
    clock_t pool2_end_time = clock();
    double pool2_time = (double)(pool2_end_time - pool2_start_time) / CLOCKS_PER_SEC;

    // Layer 3: Convolution + ReLU
    clock_t conv3_start_time = clock();
    int conv3_output_height = (pool2_output_height + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1;
    int conv3_output_width = (pool2_output_width + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1;
    int conv3_output_size = CONV3_FILTERS * conv3_output_height * conv3_output_width;
    float *conv3_output = (float *)malloc(conv3_output_size * sizeof(float));

    convolution_operation(pool2_output, conv3_weights, conv3_output,
                          pool2_output_height, pool2_output_width, CONV2_FILTERS,
                          CONV3_FILTERS, CONV3_KERNEL_SIZE, CONV3_STRIDE, CONV3_PADDING);
    clock_t conv3_end_time = clock();
    double conv3_time = (double)(conv3_end_time - conv3_start_time) / CLOCKS_PER_SEC;

    // Add biases and ReLU
    for (int i = 0; i < conv3_output_size; ++i) {
        conv3_output[i] += conv3_biases[i / (conv3_output_height * conv3_output_width)];
    }
    activate_relu(conv3_output, conv3_output_size);

    free(pool2_output);

    // Layer 4: Convolution + ReLU
    clock_t conv4_start_time = clock();
    int conv4_output_height = (conv3_output_height + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1;
    int conv4_output_width = (conv3_output_width + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1;
    int conv4_output_size = CONV4_FILTERS * conv4_output_height * conv4_output_width;
    float *conv4_output = (float *)malloc(conv4_output_size * sizeof(float));

    convolution_operation(conv3_output, conv4_weights, conv4_output,
                          conv3_output_height, conv3_output_width, CONV3_FILTERS,
                          CONV4_FILTERS, CONV4_KERNEL_SIZE, CONV4_STRIDE, CONV4_PADDING);
    clock_t conv4_end_time = clock();
    double conv4_time = (double)(conv4_end_time - conv4_start_time) / CLOCKS_PER_SEC;

    // Add biases and ReLU
    for (int i = 0; i < conv4_output_size; ++i) {
        conv4_output[i] += conv4_biases[i / (conv4_output_height * conv4_output_width)];
    }
    activate_relu(conv4_output, conv4_output_size);

    free(conv3_output);

    // Layer 5: Convolution + ReLU + Pooling
    clock_t conv5_start_time = clock();
    int conv5_output_height = (conv4_output_height + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1;
    int conv5_output_width = (conv4_output_width + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1;
    int conv5_output_size = CONV5_FILTERS * conv5_output_height * conv5_output_width;
    float *conv5_output = (float *)malloc(conv5_output_size * sizeof(float));

    convolution_operation(conv4_output, conv5_weights, conv5_output,
                          conv4_output_height, conv4_output_width, CONV4_FILTERS,
                          CONV5_FILTERS, CONV5_KERNEL_SIZE, CONV5_STRIDE, CONV5_PADDING);
    clock_t conv5_end_time = clock();
    double conv5_time = (double)(conv5_end_time - conv5_start_time) / CLOCKS_PER_SEC;

    // Add biases and ReLU
    for (int i = 0; i < conv5_output_size; ++i) {
        conv5_output[i] += conv5_biases[i / (conv5_output_height * conv5_output_width)];
    }
    activate_relu(conv5_output, conv5_output_size);

    free(conv4_output);

    clock_t pool3_start_time = clock();
    int pool3_output_height = (conv5_output_height - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;
    int pool3_output_width = (conv5_output_width - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;
    int pool3_output_size = CONV5_FILTERS * pool3_output_height * pool3_output_width;
    float *pool3_output = (float *)malloc(pool3_output_size * sizeof(float));

    pooling_operation(conv5_output, pool3_output, CONV5_FILTERS, conv5_output_height, conv5_output_width,
                      POOL3_KERNEL_SIZE, POOL3_STRIDE);

    free(conv5_output);
    clock_t pool3_end_time = clock();
    double pool3_time = (double)(pool3_end_time - pool3_start_time) / CLOCKS_PER_SEC;

    // Flatten the output for Fully Connected Layers
    int flattened_length = CONV5_FILTERS * pool3_output_height * pool3_output_width;

    // Fully Connected Layer 6
    clock_t fc6_start_time = clock();
    float *fc6_output = (float *)malloc(FC6_NEURONS * sizeof(float));
    dense_layer(pool3_output, fc6_weights, fc6_biases, fc6_output, flattened_length, FC6_NEURONS);
    clock_t fc6_end_time = clock();
    double fc6_time = (double)(fc6_end_time - fc6_start_time) / CLOCKS_PER_SEC;
    activate_relu(fc6_output, FC6_NEURONS);

    free(pool3_output);

    // Fully Connected Layer 7
    clock_t fc7_start_time = clock();
    float *fc7_output = (float *)malloc(FC7_NEURONS * sizeof(float));
    dense_layer(fc6_output, fc7_weights, fc7_biases, fc7_output, FC6_NEURONS, FC7_NEURONS);
    clock_t fc7_end_time = clock();
    double fc7_time = (double)(fc7_end_time - fc7_start_time) / CLOCKS_PER_SEC;
    activate_relu(fc7_output, FC7_NEURONS);

    free(fc6_output);

    // Fully Connected Layer 8 (Output Layer)
    clock_t fc8_start_time = clock();
    float *fc8_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    dense_layer(fc7_output, fc8_weights, fc8_biases, fc8_output, FC7_NEURONS, OUTPUT_NEURONS);
    clock_t fc8_end_time = clock();
    double fc8_time = (double)(fc8_end_time - fc8_start_time) / CLOCKS_PER_SEC;
    apply_softmax_function(fc8_output, OUTPUT_NEURONS);

    free(fc7_output);

    // Copy final output
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        final_output[i] = fc8_output[i];
    }

    free(fc8_output);
    clock_t total_end_time = clock();
    double total_time = (double)(total_end_time - total_start_time) / CLOCKS_PER_SEC;

    // Print timing information
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
}

// Function to save data to a binary file
void save_data(const char *filename, float *data, size_t size) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }
    fwrite(data, sizeof(float), size, fp);
    fclose(fp);
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

    int pool3_output_height = ((((((IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE) / CONV1_STRIDE + 1) - POOL1_KERNEL_SIZE) / POOL1_STRIDE + 1) + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1 - POOL2_KERNEL_SIZE) / POOL2_STRIDE + 1;
    pool3_output_height = ((((pool3_output_height + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE) / CONV3_STRIDE + 1 + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE) / CONV4_STRIDE + 1 + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE) / CONV5_STRIDE + 1 - POOL3_KERNEL_SIZE) / POOL3_STRIDE + 1;
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

    save_data("../Data/input_image.bin", image, image_size);
    save_data("../Data/conv1_weights.bin", conv1_weights, conv1_weights_size);
    save_data("../Data/conv1_biases.bin", conv1_biases, CONV1_FILTERS);
    save_data("../Data/conv2_weights.bin", conv2_weights, conv2_weights_size);
    save_data("../Data/conv2_biases.bin", conv2_biases, CONV2_FILTERS);
    save_data("../Data/conv3_weights.bin", conv3_weights, conv3_weights_size);
    save_data("../Data/conv3_biases.bin", conv3_biases, CONV3_FILTERS);
    save_data("../Data/conv4_weights.bin", conv4_weights, conv4_weights_size);
    save_data("../Data/conv4_biases.bin", conv4_biases, CONV4_FILTERS);
    save_data("../Data/conv5_weights.bin", conv5_weights, conv5_weights_size);
    save_data("../Data/conv5_biases.bin", conv5_biases, CONV5_FILTERS);
    save_data("../Data/fc6_weights.bin", fc6_weights, fc6_weights_size);
    save_data("../Data/fc6_biases.bin", fc6_biases, FC6_NEURONS);
    save_data("../Data/fc7_weights.bin", fc7_weights, fc7_weights_size);
    save_data("../Data/fc8_weights.bin", fc8_weights, fc8_weights_size);
    save_data("../Data/fc6_biases.bin", fc6_biases, FC6_NEURONS);
    save_data("../Data/fc7_biases.bin", fc7_biases, FC7_NEURONS);
    save_data("../Data/fc8_biases.bin", fc8_biases, OUTPUT_NEURONS);

    // Execute the Neural Network
    printf("Running AlexNet Neural Network...\n");
    neural_network(image,
                   conv1_weights, conv1_biases,
                   conv2_weights, conv2_biases,
                   conv3_weights, conv3_biases,
                   conv4_weights, conv4_biases,
                   conv5_weights, conv5_biases,
                   fc6_weights, fc6_biases,
                   fc7_weights, fc7_biases,
                   fc8_weights, fc8_biases,
                   final_output);

    // Display Output
    printf("Output Probabilities:\n");
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        printf("Class %d: %f\n", i, final_output[i]);
    }


    save_output("../Data/sequential_output.txt", final_output, OUTPUT_NEURONS);

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

    return 0;
}
