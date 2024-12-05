/*
 Authors: Henry Hardane, Alex Kheir, Lori Yahnian, and Alain Samaha
 Title: Reduced Model LeNet-5 Sequential (No training, testing, or inference)
 Description: Sequential implementation of the LeNet-5 architecture with randomized weights and input. We based our parallelization on this sequential model instead of the full model as we wanted to focus on model-parallelization instead of data-parallelization.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
void convolution_operation(float *input, float *filters, float *output, int input_dim, int num_filters, int kernel_dim) {
    int output_dim = input_dim - kernel_dim + 1;
    for (int f = 0; f < num_filters; ++f) {
        for (int row = 0; row < output_dim; ++row) {
            for (int col = 0; col < output_dim; ++col) {
                float acc = 0.0f;
                for (int i = 0; i < kernel_dim; ++i) {
                    for (int j = 0; j < kernel_dim; ++j) {
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
    for (int c = 0; c < num_channels; ++c) {
        for (int r = 0; r < output_dim; ++r) {
            for (int col = 0; col < output_dim; ++col) {
                float max_value = -INFINITY;
                for (int i = 0; i < pool_dim; ++i) {
                    for (int j = 0; j < pool_dim; ++j) {
                        max_value = MAX_VAL(max_value, input[c * input_dim * input_dim +
                                                             COORD(r * pool_dim + i, col * pool_dim + j, input_dim)]);
                    }
                }
                output[c * output_dim * output_dim + COORD(r, col, output_dim)] = max_value;
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
void neural_network(float *input, float *conv1_weights, float *conv2_weights, float *fc1_weights, float *fc1_biases,
                    float *fc2_weights, float *fc2_biases, float *final_output) {
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;
    float *conv1_output = (float *)malloc(conv1_output_size * sizeof(float));

    // Timing for Convolution Layer 1
    clock_t conv1_start_time = clock();
    convolution_operation(input, conv1_weights, conv1_output, IMAGE_DIMENSION, LAYER1_CHANNELS, KERNEL_SIZE);
    clock_t conv1_end_time = clock();
    double conv1_time = (double)(conv1_end_time - conv1_start_time) / CLOCKS_PER_SEC;

    activate_relu(conv1_output, conv1_output_size);

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;
    float *pool1_output = (float *)malloc(pool1_output_size * sizeof(float));

    // Timing for Pooling Layer 1
    clock_t pool1_start_time = clock();
    pooling_operation(conv1_output, pool1_output, conv1_output_dim, POOL_KERNEL, LAYER1_CHANNELS);
    clock_t pool1_end_time = clock();
    double pool1_time = (double)(pool1_end_time - pool1_start_time) / CLOCKS_PER_SEC;


    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;
    float *conv2_output = (float *)malloc(conv2_output_size * sizeof(float));

    // Timing for Convolution Layer 2
    clock_t conv2_start_time = clock();
    convolution_operation(pool1_output, conv2_weights, conv2_output, pool1_output_dim, LAYER2_CHANNELS, KERNEL_SIZE);
    clock_t conv2_end_time = clock();
    double conv2_time = (double)(conv2_end_time - conv2_start_time) / CLOCKS_PER_SEC;

    activate_relu(conv2_output, conv2_output_size);

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    float *pool2_output = (float *)malloc(pool2_output_size * sizeof(float));

    // Timing for Pooling Layer 2
    clock_t pool2_start_time = clock();
    pooling_operation(conv2_output, pool2_output, conv2_output_dim, POOL_KERNEL, LAYER2_CHANNELS);
    clock_t pool2_end_time = clock();
    double pool2_time = (double)(pool2_end_time - pool2_start_time) / CLOCKS_PER_SEC;

    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    float *fc1_output = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    // Timing for Fully Connected Layer 1
    clock_t fc1_start_time = clock();
    dense_layer(pool2_output, fc1_weights, fc1_biases, fc1_output, flattened_length, FULLY_CONNECTED1_NEURONS);
    clock_t fc1_end_time = clock();
    double fc1_time = (double)(fc1_end_time - fc1_start_time) / CLOCKS_PER_SEC;

    activate_relu(fc1_output, FULLY_CONNECTED1_NEURONS);

    float *fc2_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Timing for Fully Connected Layer 2
    clock_t fc2_start_time = clock();
    dense_layer(fc1_output, fc2_weights, fc2_biases, fc2_output, FULLY_CONNECTED1_NEURONS, OUTPUT_NEURONS);
    clock_t fc2_end_time = clock();
    double fc2_time = (double)(fc2_end_time - fc2_start_time) / CLOCKS_PER_SEC;

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
    for (int i = 0; i < IMAGE_DIMENSION * IMAGE_DIMENSION; ++i)
        image[i] = ((float)rand() / RAND_MAX);
    for (int i = 0; i < LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        conv1_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    for (int i = 0; i < LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        conv2_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    for (int i = 0; i < FULLY_CONNECTED1_NEURONS * fc1_input_size; ++i)
        fc1_weights[i] = generate_weight(fc1_input_size);
    for (int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i)
        fc1_biases[i] = 0.05f;
    for (int i = 0; i < OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS; ++i)
        fc2_weights[i] = generate_weight(FULLY_CONNECTED1_NEURONS);
    for (int i = 0; i < OUTPUT_NEURONS; ++i)
        fc2_biases[i] = 0.05f;

    // Save input and weights to files
    save_data("../Data/input_image.bin", image, IMAGE_DIMENSION * IMAGE_DIMENSION);
    save_data("../Data/conv1_weights.bin", conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    save_data("../Data/conv2_weights.bin", conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    save_data("../Data/fc1_weights.bin", fc1_weights, FULLY_CONNECTED1_NEURONS * fc1_input_size);
    save_data("../Data/fc1_biases.bin", fc1_biases, FULLY_CONNECTED1_NEURONS);
    save_data("../Data/fc2_weights.bin", fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
    save_data("../Data/fc2_biases.bin", fc2_biases, OUTPUT_NEURONS);

    // Execute the Neural Network
    printf("Running Sequential Neural Network...\n");
    neural_network(image, conv1_weights, conv2_weights, fc1_weights, fc1_biases, fc2_weights, fc2_biases, final_output);

    // Display Output
    printf("Output Probabilities:\n");
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        printf("Class %d: %f\n", i, final_output[i]);
    }

    // Save output to file
    save_output("../Data/sequential_output.txt", final_output, OUTPUT_NEURONS);

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
