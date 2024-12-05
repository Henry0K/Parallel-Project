/*
 Authors: Henry Hardane, Lorie Yahnian, Alex Kheir, and Alain Samaha
 Title: Reduced Model LeNet-5 Parallelized with OMP
 Description: OMP parallelized implementation of the LeNet-5 architecture.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

// Activation Functions
void activate_relu(float *vector, int length) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < length; ++i) {
        vector[i] = (vector[i] > 0) ? vector[i] : 0;
    }
}

void apply_softmax_function(float *input, int length) {
    float max_value = input[0];
    int i;
    for (i = 1; i < length; ++i) {
        if (input[i] > max_value) max_value = input[i];
    }

    float sum_exponentials = 0.0f;
    #pragma omp parallel for reduction(+:sum_exponentials)
    for (i = 0; i < length; ++i) {
        input[i] = expf(input[i] - max_value);
        sum_exponentials += input[i];
    }
    #pragma omp parallel for private(i)
    for (i = 0; i < length; ++i) {
        input[i] /= sum_exponentials;
    }
}

// Convolution Operation
void convolution_operation(float *input, float *filters, float *output, int input_dim, int num_filters, int kernel_dim) {
    int output_dim = input_dim - kernel_dim + 1;
    int f, row, col, i, j;
    #pragma omp parallel for private(f, row, col, i, j) collapse(3)
    for (f = 0; f < num_filters; ++f) {
        for (row = 0; row < output_dim; ++row) {
            for (col = 0; col < output_dim; ++col) {
                float acc = 0.0f;
                for (i = 0; i < kernel_dim; ++i) {
                    for (j = 0; j < kernel_dim; ++j) {
                        acc += input[COORD(row + i, col + j, input_dim)] *
                               filters[f * kernel_dim * kernel_dim + COORD(i, j, kernel_dim)];
                    }
                }
                output[f * output_dim * output_dim + COORD(row, col, output_dim)] = acc;
            }
        }
    }
}

// Max Pooling Operation
void pooling_operation(float *input, float *output, int input_dim, int pool_dim, int num_channels) {
    int output_dim = input_dim / pool_dim;
    int c, r, col, i, j;
    #pragma omp parallel for private(c, r, col, i, j) collapse(3)
    for (c = 0; c < num_channels; ++c) {
        for (r = 0; r < output_dim; ++r) {
            for (col = 0; col < output_dim; ++col) {
                float max_value = -INFINITY;
                for (i = 0; i < pool_dim; ++i) {
                    for (j = 0; j < pool_dim; ++j) {
                        float val = input[c * input_dim * input_dim +
                                          COORD(r * pool_dim + i, col * pool_dim + j, input_dim)];
                        max_value = MAX_VAL(max_value, val);
                    }
                }
                output[c * output_dim * output_dim + COORD(r, col, output_dim)] = max_value;
            }
        }
    }
}

// Fully Connected Layer
void dense_layer(float *input, float *weights, float *biases, float *output, int input_length, int output_length) {
    int o, i;
    #pragma omp parallel for private(o, i)
    for (o = 0; o < output_length; ++o) {
        float sum = biases[o];
        for (i = 0; i < input_length; ++i) {
            sum += input[i] * weights[o * input_length + i];
        }
        output[o] = sum;
    }
}

// Function to read data from a binary file
void read_data(const char *filename, float *data, size_t size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }
    fread(data, sizeof(float), size, fp);
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

// Complete Model
void neural_network(float *input, float *conv1_weights, float *conv2_weights, float *fc1_weights, float *fc1_biases,
                    float *fc2_weights, float *fc2_biases, float *final_output) {
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;
    float *conv1_output = (float *)malloc(conv1_output_size * sizeof(float));

    // Convolution Layer 1
    double conv1_start_time = omp_get_wtime();
    convolution_operation(input, conv1_weights, conv1_output, IMAGE_DIMENSION, LAYER1_CHANNELS, KERNEL_SIZE);
    double conv1_end_time = omp_get_wtime();
    double conv1_time = conv1_end_time - conv1_start_time;

    activate_relu(conv1_output, conv1_output_size);

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;
    float *pool1_output = (float *)malloc(pool1_output_size * sizeof(float));

    // Pooling Layer 1
    double pool1_start_time = omp_get_wtime();
    pooling_operation(conv1_output, pool1_output, conv1_output_dim, POOL_KERNEL, LAYER1_CHANNELS);
    double pool1_end_time = omp_get_wtime();
    double pool1_time = pool1_end_time - pool1_start_time;

    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;
    float *conv2_output = (float *)malloc(conv2_output_size * sizeof(float));

    // Convolution Layer 2
    double conv2_start_time = omp_get_wtime();
    convolution_operation(pool1_output, conv2_weights, conv2_output, pool1_output_dim, LAYER2_CHANNELS, KERNEL_SIZE);
    double conv2_end_time = omp_get_wtime();
    double conv2_time = conv2_end_time - conv2_start_time;

    activate_relu(conv2_output, conv2_output_size);

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    float *pool2_output = (float *)malloc(pool2_output_size * sizeof(float));

    // Pooling Layer 2
    double pool2_start_time = omp_get_wtime();
    pooling_operation(conv2_output, pool2_output, conv2_output_dim, POOL_KERNEL, LAYER2_CHANNELS);
    double pool2_end_time = omp_get_wtime();
    double pool2_time = pool2_end_time - pool2_start_time;

    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    float *fc1_output = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    // Fully Connected Layer 1
    double fc1_start_time = omp_get_wtime();
    dense_layer(pool2_output, fc1_weights, fc1_biases, fc1_output, flattened_length, FULLY_CONNECTED1_NEURONS);
    double fc1_end_time = omp_get_wtime();
    double fc1_time = fc1_end_time - fc1_start_time;

    activate_relu(fc1_output, FULLY_CONNECTED1_NEURONS);

    float *fc2_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Fully Connected Layer 2
    double fc2_start_time = omp_get_wtime();
    dense_layer(fc1_output, fc2_weights, fc2_biases, fc2_output, FULLY_CONNECTED1_NEURONS, OUTPUT_NEURONS);
    double fc2_end_time = omp_get_wtime();
    double fc2_time = fc2_end_time - fc2_start_time;

    apply_softmax_function(fc2_output, OUTPUT_NEURONS);

    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        final_output[i] = fc2_output[i];
    }

    // Print timing information
    printf("Layer Execution Times (seconds):\n");
    printf("Convolution Layer 1: %f\n", conv1_time);
    printf("Pooling Layer 1:     %f\n", pool1_time);
    printf("Convolution Layer 2: %f\n", conv2_time);
    printf("Pooling Layer 2:     %f\n", pool2_time);
    printf("Fully Connected 1:   %f\n", fc1_time);
    printf("Fully Connected 2:   %f\n", fc2_time);
    printf("Total Execution Time: %f\n", conv1_time + pool1_time + conv2_time + pool2_time + fc1_time + fc2_time);

    free(conv1_output);
    free(pool1_output);
    free(conv2_output);
    free(pool2_output);
    free(fc1_output);
    free(fc2_output);
}

// Main Function
int main() {
    // Input and Weights Allocation
    float *image = (float *)malloc(IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float));
    float *conv1_weights = (float *)malloc(LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *conv2_weights = (float *)malloc(LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    int fc1_input_size = LAYER2_CHANNELS * ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4) * ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4);
    float *fc1_weights = (float *)malloc(FULLY_CONNECTED1_NEURONS * fc1_input_size * sizeof(float));
    float *fc1_biases = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    float *fc2_weights = (float *)malloc(OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float));
    float *fc2_biases = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    float *final_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Read input and weights from files
    read_data("../Data/input_image.bin", image, IMAGE_DIMENSION * IMAGE_DIMENSION);
    read_data("../Data/conv1_weights.bin", conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    read_data("../Data/conv2_weights.bin", conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    read_data("../Data/fc1_weights.bin", fc1_weights, FULLY_CONNECTED1_NEURONS * fc1_input_size);
    read_data("../Data/fc1_biases.bin", fc1_biases, FULLY_CONNECTED1_NEURONS);
    read_data("../Data/fc2_weights.bin", fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
    read_data("../Data/fc2_biases.bin", fc2_biases, OUTPUT_NEURONS);

    // Execute the Neural Network
    printf("Running OpenMP Parallel Neural Network...\n");
    neural_network(image, conv1_weights, conv2_weights, fc1_weights, fc1_biases, fc2_weights, fc2_biases, final_output);

    // Display Output
    printf("Output Probabilities:\n");
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        printf("Class %d: %f\n", i, final_output[i]);
    }

    // Save output to file
    save_output("../Data/openmp_output.txt", final_output, OUTPUT_NEURONS);

    // Free Allocated Memory
    free(image);
    free(conv1_weights);
    free(conv2_weights);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    free(final_output);

    return 0;
}