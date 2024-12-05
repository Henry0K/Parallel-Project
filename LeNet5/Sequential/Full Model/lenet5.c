/*
 Authors: Henry Hardane, Alex Kheir, Lori Yahnian, and Alain Samaha
 Title: Full Model LeNet-5 Sequential
 Description: Sequential implementation of the LeNet-5 architecture.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <stdint.h>

// Neural Network Parameters
#define IMAGE_DIMENSION 32
#define ORIGINAL_IMAGE_DIMENSION 28
#define KERNEL_SIZE 5
#define LAYER1_CHANNELS 6
#define LAYER2_CHANNELS 16
#define POOL_KERNEL 2
#define FULLY_CONNECTED1_NEURONS 120
#define OUTPUT_NEURONS 10

#define CONV1_WEIGHTS_FILE "conv1_weights.bin"
#define CONV1_BIASES_FILE  "conv1_biases.bin"
#define CONV2_WEIGHTS_FILE "conv2_weights.bin"
#define CONV2_BIASES_FILE  "conv2_biases.bin"
#define FC1_WEIGHTS_FILE   "fc1_weights.bin"
#define FC1_BIASES_FILE    "fc1_biases.bin"
#define FC2_WEIGHTS_FILE   "fc2_weights.bin"
#define FC2_BIASES_FILE    "fc2_biases.bin"

// Function to save a parameter array to a binary file
void save_parameters(const char *filename, float *data, size_t size) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for saving parameters");
        exit(1);
    }
    fwrite(data, sizeof(float), size, file);
    fclose(file);
}

// Function to load a parameter array from a binary file
void load_parameters(const char *filename, float *data, size_t size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for loading parameters");
        exit(1);
    }
    size_t read_count = fread(data, sizeof(float), size, file);
    if (read_count != size) {
        fprintf(stderr, "Failed to read all parameters from %s\n", filename);
        fclose(file);
        exit(1);
    }
    fclose(file);
}

void load_custom_image(const char *filename, float *image) {
    // Example: Load a raw 32x32 grayscale image
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open custom image file");
        exit(1);
    }

    unsigned char buffer[IMAGE_DIMENSION * IMAGE_DIMENSION];
    size_t read_count = fread(buffer, 1, IMAGE_DIMENSION * IMAGE_DIMENSION, file);
    if (read_count != IMAGE_DIMENSION * IMAGE_DIMENSION) {
        fprintf(stderr, "Failed to read image data from %s\n", filename);
        fclose(file);
        exit(1);
    }
    fclose(file);

    // Normalize pixel values to [0,1]
    for (int i = 0; i < IMAGE_DIMENSION * IMAGE_DIMENSION; ++i) {
        image[i] = buffer[i] / 255.0f;
    }
}

// Function to check if a file exists
int file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void load_all_parameters(float *conv1_weights, float *conv1_biases,
                        float *conv2_weights, float *conv2_biases,
                        float *fc1_weights, float *fc1_biases,
                        float *fc2_weights, float *fc2_biases) {
    load_parameters(CONV1_WEIGHTS_FILE, conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    load_parameters(CONV1_BIASES_FILE, conv1_biases, LAYER1_CHANNELS);

    load_parameters(CONV2_WEIGHTS_FILE, conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    load_parameters(CONV2_BIASES_FILE, conv2_biases, LAYER2_CHANNELS);

    // Calculate flattened_length as in main or pass it as a parameter
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;

    load_parameters(FC1_WEIGHTS_FILE, fc1_weights, FULLY_CONNECTED1_NEURONS * flattened_length);
    load_parameters(FC1_BIASES_FILE, fc1_biases, FULLY_CONNECTED1_NEURONS);

    load_parameters(FC2_WEIGHTS_FILE, fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
    load_parameters(FC2_BIASES_FILE, fc2_biases, OUTPUT_NEURONS);
}



// Macro Utilities
#define COORD(row, col, stride) ((row) * (stride) + (col))
#define MAX_VAL(a, b) (((a) > (b)) ? (a) : (b))

// Randomized weight initialization with Uniform distribution
float generate_weight(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return ((float)rand() / (float)RAND_MAX) * 2 * limit - limit;
}

// Activation Functions
void activate_relu(float *vector, int length, float *output) {
    for (int i = 0; i < length; ++i) {
        output[i] = (vector[i] > 0) ? vector[i] : 0;
    }
}

void relu_derivative(float *input, float *output, int length) {
    for (int i = 0; i < length; ++i) {
        output[i] = (input[i] > 0) ? 1.0f : 0.0f;
    }
}

void apply_softmax_function(float *input, int length) {
    float max_value = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_value)
            max_value = input[i];
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
void convolution_operation(float *input, float *filters, float *biases, float *output, int input_dim, int num_filters, int kernel_dim) {
    int output_dim = input_dim - kernel_dim + 1;
    for (int f = 0; f < num_filters; ++f) {
        float *filter = &filters[f * kernel_dim * kernel_dim];
        float bias = biases[f];
        for (int row = 0; row < output_dim; ++row) {
            for (int col = 0; col < output_dim; ++col) {
                float acc = 0.0f;
                for (int i = 0; i < kernel_dim; ++i) {
                    for (int j = 0; j < kernel_dim; ++j) {
                        acc += input[COORD(row + i, col + j, input_dim)] *
                               filter[COORD(i, j, kernel_dim)];
                    }
                }
                output[f * output_dim * output_dim + COORD(row, col, output_dim)] = acc + bias;
            }
        }
    }
}

// Convolution Gradient with respect to input
void convolution_input_gradient(float *output_grad, float *filters, float *input_grad, int input_dim, int num_filters, int kernel_dim) {
    int output_dim = input_dim - kernel_dim + 1;

    // Initialize input gradient to zero
    memset(input_grad, 0, input_dim * input_dim * sizeof(float));

    for (int f = 0; f < num_filters; ++f) {
        float *filter = &filters[f * kernel_dim * kernel_dim];
        for (int row = 0; row < output_dim; ++row) {
            for (int col = 0; col < output_dim; ++col) {
                float grad = output_grad[f * output_dim * output_dim + COORD(row, col, output_dim)];
                for (int i = 0; i < kernel_dim; ++i) {
                    for (int j = 0; j < kernel_dim; ++j) {
                        input_grad[COORD(row + i, col + j, input_dim)] +=
                            grad * filter[COORD(i, j, kernel_dim)];
                    }
                }
            }
        }
    }
}

// Convolution Gradient with respect to filters
void convolution_weight_gradient(float *input, float *output_grad, float *weight_grad, int input_dim, int num_filters, int kernel_dim) {
    int output_dim = input_dim - kernel_dim + 1;

    // Initialize weight gradient to zero
    memset(weight_grad, 0, num_filters * kernel_dim * kernel_dim * sizeof(float));

    for (int f = 0; f < num_filters; ++f) {
        float *filter_grad = &weight_grad[f * kernel_dim * kernel_dim];
        for (int row = 0; row < output_dim; ++row) {
            for (int col = 0; col < output_dim; ++col) {
                float grad = output_grad[f * output_dim * output_dim + COORD(row, col, output_dim)];
                for (int i = 0; i < kernel_dim; ++i) {
                    for (int j = 0; j < kernel_dim; ++j) {
                        filter_grad[COORD(i, j, kernel_dim)] +=
                            grad * input[COORD(row + i, col + j, input_dim)];
                    }
                }
            }
        }
    }
}

// Max Pooling Operation with Indices
void pooling_operation(float *input, float *output, int *indices, int input_dim, int pool_dim) {
    int output_dim = input_dim / pool_dim;
    for (int row = 0; row < output_dim; ++row) {
        for (int col = 0; col < output_dim; ++col) {
            float max_value = -INFINITY;
            int max_index = -1;
            for (int i = 0; i < pool_dim; ++i) {
                for (int j = 0; j < pool_dim; ++j) {
                    int input_row = row * pool_dim + i;
                    int input_col = col * pool_dim + j;
                    int input_index = COORD(input_row, input_col, input_dim);
                    if (input[input_index] > max_value) {
                        max_value = input[input_index];
                        max_index = input_index;
                    }
                }
            }
            output[COORD(row, col, output_dim)] = max_value;
            indices[COORD(row, col, output_dim)] = max_index;
        }
    }
}

// Max Pooling Gradient
void pooling_layer_backprop(float *output_grad, int *indices, float *input_grad, int output_size, int input_size) {
    // Initialize input gradient to zero
    memset(input_grad, 0, input_size * sizeof(float));

    for (int i = 0; i < output_size; ++i) {
        int idx = indices[i];
        input_grad[idx] = output_grad[i];
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

// Fully Connected Layer Gradient
void dense_layer_backprop(float *output_grad, float *weights, float *input_grad, int input_length, int output_length) {
    // Initialize input gradient to zero
    memset(input_grad, 0, input_length * sizeof(float));

    for (int i = 0; i < input_length; ++i) {
        for (int o = 0; o < output_length; ++o) {
            input_grad[i] += output_grad[o] * weights[o * input_length + i];
        }
    }
}

// Structure to hold network state
typedef struct {
    float *conv1_output;       // Pre-activation output of conv1
    float *conv1_output_relu;  // Post-activation output of conv1
    float *pool1_output;
    int *pool1_indices;

    float *conv2_output;
    float *conv2_output_relu;
    float *pool2_output;
    int *pool2_indices;

    float *fc1_output;
    float *fc1_output_relu;

    float *fc2_output;
    float *softmax_output;
} NetworkState;

// Structure to hold gradients
typedef struct {
    // Gradients for weights and biases
    float *fc2_weights_gradients; // [OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS]
    float *fc2_biases_gradients;  // [OUTPUT_NEURONS]

    float *fc1_weights_gradients; // [FULLY_CONNECTED1_NEURONS * flattened_length]
    float *fc1_biases_gradients;  // [FULLY_CONNECTED1_NEURONS]

    float *conv2_weights_gradients; // [LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]
    float *conv2_biases_gradients;  // [LAYER2_CHANNELS]

    float *conv1_weights_gradients; // [LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]
    float *conv1_biases_gradients;  // [LAYER1_CHANNELS]

    // Intermediate gradients
    float *fc1_output_relu_gradients; // [FULLY_CONNECTED1_NEURONS]
    float *fc1_output_gradients;      // [FULLY_CONNECTED1_NEURONS]

    float *pool2_output_gradients; // [flattened_length]

    float *conv2_output_relu_gradients; // [LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim]
    float *conv2_output_gradients;      // same size

    float *pool1_output_gradients; // [LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim]

    float *conv1_output_relu_gradients; // [LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim]
    float *conv1_output_gradients;      // same size

    float *input_gradients;             // [IMAGE_DIMENSION * IMAGE_DIMENSION]

    float *output_layer_gradients;      // [OUTPUT_NEURONS]
} NetworkGradients;

// Complete Model
void neural_network(float *input, float *conv1_weights, float *conv1_biases, float *conv2_weights, float *conv2_biases,
                    float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases,
                    NetworkState *state) {
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;

    convolution_operation(input, conv1_weights, conv1_biases, state->conv1_output, IMAGE_DIMENSION, LAYER1_CHANNELS, KERNEL_SIZE);
    activate_relu(state->conv1_output, conv1_output_size, state->conv1_output_relu);

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;

    pooling_operation(state->conv1_output_relu, state->pool1_output, state->pool1_indices, conv1_output_dim, POOL_KERNEL);

    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;

    convolution_operation(state->pool1_output, conv2_weights, conv2_biases, state->conv2_output, pool1_output_dim, LAYER2_CHANNELS, KERNEL_SIZE);
    activate_relu(state->conv2_output, conv2_output_size, state->conv2_output_relu);

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;

    pooling_operation(state->conv2_output_relu, state->pool2_output, state->pool2_indices, conv2_output_dim, POOL_KERNEL);

    int flattened_length = pool2_output_size;

    dense_layer(state->pool2_output, fc1_weights, fc1_biases, state->fc1_output, flattened_length, FULLY_CONNECTED1_NEURONS);
    activate_relu(state->fc1_output, FULLY_CONNECTED1_NEURONS, state->fc1_output_relu);

    dense_layer(state->fc1_output_relu, fc2_weights, fc2_biases, state->fc2_output, FULLY_CONNECTED1_NEURONS, OUTPUT_NEURONS);

    // Softmax output
    for (int i = 0; i < OUTPUT_NEURONS; ++i) {
        state->softmax_output[i] = state->fc2_output[i];
    }
    apply_softmax_function(state->softmax_output, OUTPUT_NEURONS);
}

// Read MNIST data
int read_int(FILE *file) {
    unsigned char buffer[4];
    fread(buffer, 1, 4, file);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

void load_mnist_images(const char *file_path, float **images, int *num_images) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        perror("Failed to open MNIST images file");
        exit(1);
    }

    int magic_number = read_int(file);
    *num_images = read_int(file);
    int rows = read_int(file);
    int cols = read_int(file);

    *images = (float *)malloc((*num_images) * IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float));
    unsigned char *buffer = (unsigned char *)malloc(rows * cols);

    for (int i = 0; i < *num_images; ++i) {
        fread(buffer, 1, rows * cols, file);
        float *image = &((*images)[i * IMAGE_DIMENSION * IMAGE_DIMENSION]);
        for (int r = 0; r < IMAGE_DIMENSION; ++r) {
            for (int c = 0; c < IMAGE_DIMENSION; ++c) {
                if (r < rows && c < cols) {
                    image[r * IMAGE_DIMENSION + c] = buffer[r * cols + c] / 255.0f;
                } else {
                    image[r * IMAGE_DIMENSION + c] = 0.0f;
                }
            }
        }
    }

    free(buffer);
    fclose(file);
}

void load_mnist_labels(const char *file_path, int **labels, int *num_labels) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        perror("Failed to open MNIST labels file");
        exit(1);
    }

    int magic_number = read_int(file);
    if (magic_number != 2049) { // Magic number for labels
        fprintf(stderr, "Invalid MNIST label file!\n");
        fclose(file);
        exit(1);
    }

    *num_labels = read_int(file);

    *labels = (int *)malloc((*num_labels) * sizeof(int));
    if (!*labels) {
        perror("Failed to allocate memory for labels");
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *num_labels; ++i) {
        unsigned char label;
        fread(&label, 1, 1, file);
        (*labels)[i] = label;
    }

    fclose(file);
}

float compute_loss(float *predictions, int label) {
    return -logf(predictions[label]);
}

void compute_output_layer_gradients(float *softmax_output, int label, float *gradients, int length) {
    for (int i = 0; i < length; ++i) {
        gradients[i] = softmax_output[i] - (i == label ? 1.0f : 0.0f);
    }
}

void update_weights(float *weights, float *gradients, float learning_rate, int length) {
    for (int i = 0; i < length; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}

void train_sample(float *image, int label,
                  float *conv1_weights, float *conv1_biases, float *conv2_weights, float *conv2_biases,
                  float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases,
                  float learning_rate, NetworkState *state, NetworkGradients *grads) {

    // Calculate dimensions
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;

    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;

    int flattened_length = pool2_output_size;

    // Forward pass
    neural_network(image, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                   fc1_weights, fc1_biases, fc2_weights, fc2_biases, state);

    // Compute loss and initial gradients
    float loss = compute_loss(state->softmax_output, label);

    compute_output_layer_gradients(state->softmax_output, label, grads->output_layer_gradients, OUTPUT_NEURONS);

    // Backpropagation
    // Gradients for fc2_weights and fc2_biases
    for (int o = 0; o < OUTPUT_NEURONS; ++o) {
        grads->fc2_biases_gradients[o] = grads->output_layer_gradients[o];
        for (int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i) {
            grads->fc2_weights_gradients[o * FULLY_CONNECTED1_NEURONS + i] +=
                grads->output_layer_gradients[o] * state->fc1_output_relu[i];
        }
    }

    // Backpropagation to fc1_output_relu
    memset(grads->fc1_output_relu_gradients, 0, FULLY_CONNECTED1_NEURONS * sizeof(float));
    for (int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i) {
        for (int o = 0; o < OUTPUT_NEURONS; ++o) {
            grads->fc1_output_relu_gradients[i] += grads->output_layer_gradients[o] * fc2_weights[o * FULLY_CONNECTED1_NEURONS + i];
        }
    }

    // Apply ReLU derivative
    relu_derivative(state->fc1_output, grads->fc1_output_gradients, FULLY_CONNECTED1_NEURONS);
    for (int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i) {
        grads->fc1_output_gradients[i] *= grads->fc1_output_relu_gradients[i];
    }

    // Gradients for fc1_weights and fc1_biases
    for (int o = 0; o < FULLY_CONNECTED1_NEURONS; ++o) {
        grads->fc1_biases_gradients[o] = grads->fc1_output_gradients[o];
        for (int i = 0; i < flattened_length; ++i) {
            grads->fc1_weights_gradients[o * flattened_length + i] +=
                grads->fc1_output_gradients[o] * state->pool2_output[i];
        }
    }

    // Backpropagation to pool2_output
    memset(grads->pool2_output_gradients, 0, flattened_length * sizeof(float));
    for (int i = 0; i < flattened_length; ++i) {
        for (int o = 0; o < FULLY_CONNECTED1_NEURONS; ++o) {
            grads->pool2_output_gradients[i] += grads->fc1_output_gradients[o] * fc1_weights[o * flattened_length + i];
        }
    }

    // Backpropagation through pooling layer 2
    memset(grads->conv2_output_relu_gradients, 0, conv2_output_size * sizeof(float));
    pooling_layer_backprop(grads->pool2_output_gradients, state->pool2_indices,
                           grads->conv2_output_relu_gradients, flattened_length, conv2_output_size);

    // Backpropagation through ReLU activation at conv2
    relu_derivative(state->conv2_output, grads->conv2_output_gradients, conv2_output_size);
    for (int i = 0; i < conv2_output_size; ++i) {
        grads->conv2_output_gradients[i] *= grads->conv2_output_relu_gradients[i];
    }

    // Gradients for conv2_weights and conv2_biases
    convolution_weight_gradient(state->pool1_output, grads->conv2_output_gradients,
                                grads->conv2_weights_gradients, pool1_output_dim, LAYER2_CHANNELS, KERNEL_SIZE);

    for (int f = 0; f < LAYER2_CHANNELS; ++f) {
        float sum = 0.0f;
        int offset = f * conv2_output_dim * conv2_output_dim;
        for (int i = 0; i < conv2_output_dim * conv2_output_dim; ++i) {
            sum += grads->conv2_output_gradients[offset + i];
        }
        grads->conv2_biases_gradients[f] += sum;
    }

    // Backpropagation to pool1_output
    convolution_input_gradient(grads->conv2_output_gradients, conv2_weights, grads->pool1_output_gradients,
                               pool1_output_dim, LAYER2_CHANNELS, KERNEL_SIZE);

    // Backpropagation through pooling layer 1
    memset(grads->conv1_output_relu_gradients, 0, conv1_output_size * sizeof(float));
    pooling_layer_backprop(grads->pool1_output_gradients, state->pool1_indices,
                           grads->conv1_output_relu_gradients, pool1_output_size, conv1_output_size);

    // Backpropagation through ReLU activation at conv1
    relu_derivative(state->conv1_output, grads->conv1_output_gradients, conv1_output_size);
    for (int i = 0; i < conv1_output_size; ++i) {
        grads->conv1_output_gradients[i] *= grads->conv1_output_relu_gradients[i];
    }

    // Gradients for conv1_weights and conv1_biases
    convolution_weight_gradient(image, grads->conv1_output_gradients,
                                grads->conv1_weights_gradients, IMAGE_DIMENSION, LAYER1_CHANNELS, KERNEL_SIZE);

    for (int f = 0; f < LAYER1_CHANNELS; ++f) {
        float sum = 0.0f;
        int offset = f * conv1_output_dim * conv1_output_dim;
        for (int i = 0; i < conv1_output_dim * conv1_output_dim; ++i) {
            sum += grads->conv1_output_gradients[offset + i];
        }
        grads->conv1_biases_gradients[f] += sum;
    }

    // Update weights and biases
    update_weights(conv1_weights, grads->conv1_weights_gradients, learning_rate, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    update_weights(conv1_biases, grads->conv1_biases_gradients, learning_rate, LAYER1_CHANNELS);

    update_weights(conv2_weights, grads->conv2_weights_gradients, learning_rate, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    update_weights(conv2_biases, grads->conv2_biases_gradients, learning_rate, LAYER2_CHANNELS);

    update_weights(fc1_weights, grads->fc1_weights_gradients, learning_rate, FULLY_CONNECTED1_NEURONS * flattened_length);
    update_weights(fc1_biases, grads->fc1_biases_gradients, learning_rate, FULLY_CONNECTED1_NEURONS);

    update_weights(fc2_weights, grads->fc2_weights_gradients, learning_rate, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
    update_weights(fc2_biases, grads->fc2_biases_gradients, learning_rate, OUTPUT_NEURONS);
}

void train_neural_network(float *images, int *labels, int num_images, int num_epochs, float learning_rate,
                          float *conv1_weights, float *conv1_biases, float *conv2_weights, float *conv2_biases,
                          float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases) {
    // Initialize weights and biases
    for (int i = 0; i < LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        conv1_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    for (int i = 0; i < LAYER1_CHANNELS; ++i)
        conv1_biases[i] = 0.0f;

    for (int i = 0; i < LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        conv2_weights[i] = generate_weight(KERNEL_SIZE * KERNEL_SIZE);
    for (int i = 0; i < LAYER2_CHANNELS; ++i)
        conv2_biases[i] = 0.0f;

    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;

    for (int i = 0; i < FULLY_CONNECTED1_NEURONS * flattened_length; ++i)
        fc1_weights[i] = generate_weight(flattened_length);
    for (int i = 0; i < FULLY_CONNECTED1_NEURONS; ++i)
        fc1_biases[i] = 0.0f;

    for (int i = 0; i < OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS; ++i)
        fc2_weights[i] = generate_weight(FULLY_CONNECTED1_NEURONS);
    for (int i = 0; i < OUTPUT_NEURONS; ++i)
        fc2_biases[i] = 0.0f;

    // Allocate memory for network state
    NetworkState state;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;
    state.conv1_output = (float *)malloc(conv1_output_size * sizeof(float));
    state.conv1_output_relu = (float *)malloc(conv1_output_size * sizeof(float));

    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;
    state.pool1_output = (float *)malloc(pool1_output_size * sizeof(float));
    state.pool1_indices = (int *)malloc(pool1_output_size * sizeof(int));

    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;
    state.conv2_output = (float *)malloc(conv2_output_size * sizeof(float));
    state.conv2_output_relu = (float *)malloc(conv2_output_size * sizeof(float));

    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    state.pool2_output = (float *)malloc(pool2_output_size * sizeof(float));
    state.pool2_indices = (int *)malloc(pool2_output_size * sizeof(int));

    state.fc1_output = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    state.fc1_output_relu = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    state.fc2_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    state.softmax_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Allocate memory for gradients
    NetworkGradients grads;
    grads.fc2_weights_gradients = (float *)calloc(OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS, sizeof(float));
    grads.fc2_biases_gradients = (float *)calloc(OUTPUT_NEURONS, sizeof(float));
    grads.fc1_output_relu_gradients = (float *)calloc(FULLY_CONNECTED1_NEURONS, sizeof(float));
    grads.fc1_output_gradients = (float *)calloc(FULLY_CONNECTED1_NEURONS, sizeof(float));
    grads.fc1_weights_gradients = (float *)calloc(FULLY_CONNECTED1_NEURONS * flattened_length, sizeof(float));
    grads.fc1_biases_gradients = (float *)calloc(FULLY_CONNECTED1_NEURONS, sizeof(float));
    grads.pool2_output_gradients = (float *)calloc(flattened_length, sizeof(float));
    grads.conv2_output_relu_gradients = (float *)calloc(LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim, sizeof(float));
    grads.conv2_output_gradients = (float *)calloc(LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim, sizeof(float));
    grads.conv2_weights_gradients = (float *)calloc(LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, sizeof(float));
    grads.conv2_biases_gradients = (float *)calloc(LAYER2_CHANNELS, sizeof(float));
    grads.pool1_output_gradients = (float *)calloc(LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim, sizeof(float));
    grads.conv1_output_relu_gradients = (float *)calloc(LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim, sizeof(float));
    grads.conv1_output_gradients = (float *)calloc(LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim, sizeof(float));
    grads.conv1_weights_gradients = (float *)calloc(LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, sizeof(float));
    grads.conv1_biases_gradients = (float *)calloc(LAYER1_CHANNELS, sizeof(float));
    grads.output_layer_gradients = (float *)calloc(OUTPUT_NEURONS, sizeof(float));
    grads.input_gradients = (float *)calloc(IMAGE_DIMENSION * IMAGE_DIMENSION, sizeof(float));

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < num_images; ++i) {
            float *image = &images[i * IMAGE_DIMENSION * IMAGE_DIMENSION];
            int label = labels[i];

            // Reset gradient arrays to zero
            memset(grads.fc2_weights_gradients, 0, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float));
            memset(grads.fc2_biases_gradients, 0, OUTPUT_NEURONS * sizeof(float));
            memset(grads.fc1_weights_gradients, 0, FULLY_CONNECTED1_NEURONS * flattened_length * sizeof(float));
            memset(grads.fc1_biases_gradients, 0, FULLY_CONNECTED1_NEURONS * sizeof(float));
            memset(grads.conv2_weights_gradients, 0, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
            memset(grads.conv2_biases_gradients, 0, LAYER2_CHANNELS * sizeof(float));
            memset(grads.conv1_weights_gradients, 0, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
            memset(grads.conv1_biases_gradients, 0, LAYER1_CHANNELS * sizeof(float));

            train_sample(image, label, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                         fc1_weights, fc1_biases, fc2_weights, fc2_biases, learning_rate, &state, &grads);

            total_loss += compute_loss(state.softmax_output, label);

            // Check prediction
            int predicted_class = 0;
            float max_probability = state.softmax_output[0];
            for (int j = 1; j < OUTPUT_NEURONS; ++j) {
                if (state.softmax_output[j] > max_probability) {
                    max_probability = state.softmax_output[j];
                    predicted_class = j;
                }
            }
            if (predicted_class == label) {
                correct++;
            }
        }

        float avg_loss = total_loss / num_images;
        float accuracy = (float)correct / num_images * 100;
        printf("Epoch %d completed - Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, avg_loss, accuracy);
    }

    printf("Training completed. Saving model parameters...\n");
    save_parameters(CONV1_WEIGHTS_FILE, conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    save_parameters(CONV1_BIASES_FILE, conv1_biases, LAYER1_CHANNELS);

    save_parameters(CONV2_WEIGHTS_FILE, conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    save_parameters(CONV2_BIASES_FILE, conv2_biases, LAYER2_CHANNELS);

    save_parameters(FC1_WEIGHTS_FILE, fc1_weights, FULLY_CONNECTED1_NEURONS * flattened_length);
    save_parameters(FC1_BIASES_FILE, fc1_biases, FULLY_CONNECTED1_NEURONS);

    save_parameters(FC2_WEIGHTS_FILE, fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
    save_parameters(FC2_BIASES_FILE, fc2_biases, OUTPUT_NEURONS);

    printf("Model parameters saved successfully.\n");

    // Free allocated memory
    free(state.conv1_output);
    free(state.conv1_output_relu);
    free(state.pool1_output);
    free(state.pool1_indices);
    free(state.conv2_output);
    free(state.conv2_output_relu);
    free(state.pool2_output);
    free(state.pool2_indices);
    free(state.fc1_output);
    free(state.fc1_output_relu);
    free(state.fc2_output);
    free(state.softmax_output);

    free(grads.fc2_weights_gradients);
    free(grads.fc2_biases_gradients);
    free(grads.fc1_output_relu_gradients);
    free(grads.fc1_output_gradients);
    free(grads.fc1_weights_gradients);
    free(grads.fc1_biases_gradients);
    free(grads.pool2_output_gradients);
    free(grads.conv2_output_relu_gradients);
    free(grads.conv2_output_gradients);
    free(grads.conv2_weights_gradients);
    free(grads.conv2_biases_gradients);
    free(grads.pool1_output_gradients);
    free(grads.conv1_output_relu_gradients);
    free(grads.conv1_output_gradients);
    free(grads.conv1_weights_gradients);
    free(grads.conv1_biases_gradients);
    free(grads.output_layer_gradients);
    free(grads.input_gradients);
}

void test_neural_network(float *images, int *labels, int num_images, float *conv1_weights, float *conv1_biases,
                         float *conv2_weights, float *conv2_biases, float *fc1_weights, float *fc1_biases,
                         float *fc2_weights, float *fc2_biases) {
    int correct_predictions = 0;

    // Allocate memory for network state
    NetworkState state;
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;
    state.conv1_output = (float *)malloc(conv1_output_size * sizeof(float));
    state.conv1_output_relu = (float *)malloc(conv1_output_size * sizeof(float));

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;
    state.pool1_output = (float *)malloc(pool1_output_size * sizeof(float));
    state.pool1_indices = (int *)malloc(pool1_output_size * sizeof(int));

    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;
    state.conv2_output = (float *)malloc(conv2_output_size * sizeof(float));
    state.conv2_output_relu = (float *)malloc(conv2_output_size * sizeof(float));

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    state.pool2_output = (float *)malloc(pool2_output_size * sizeof(float));
    state.pool2_indices = (int *)malloc(pool2_output_size * sizeof(int));

    state.fc1_output = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    state.fc1_output_relu = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    state.fc2_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    state.softmax_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    for (int i = 0; i < num_images; ++i) {
        float *image = &images[i * IMAGE_DIMENSION * IMAGE_DIMENSION];
        int label = labels[i];

        neural_network(image, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                       fc1_weights, fc1_biases, fc2_weights, fc2_biases, &state);

        // Determine predicted class
        int predicted_class = 0;
        float max_probability = state.softmax_output[0];
        for (int j = 1; j < OUTPUT_NEURONS; ++j) {
            if (state.softmax_output[j] > max_probability) {
                max_probability = state.softmax_output[j];
                predicted_class = j;
            }
        }

        if (predicted_class == label) {
            correct_predictions++;
        }
    }

    printf("Test Accuracy: %.2f%%\n", (float)correct_predictions / num_images * 100);

    // Free allocated memory
    free(state.conv1_output);
    free(state.conv1_output_relu);
    free(state.pool1_output);
    free(state.pool1_indices);
    free(state.conv2_output);
    free(state.conv2_output_relu);
    free(state.pool2_output);
    free(state.pool2_indices);
    free(state.fc1_output);
    free(state.fc1_output_relu);
    free(state.fc2_output);
    free(state.softmax_output);
}

int infer(float *image, float *conv1_weights, float *conv1_biases,
          float *conv2_weights, float *conv2_biases,
          float *fc1_weights, float *fc1_biases,
          float *fc2_weights, float *fc2_biases) {
    // Allocate memory for network state
    NetworkState state;
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;
    state.conv1_output = (float *)malloc(conv1_output_size * sizeof(float));
    state.conv1_output_relu = (float *)malloc(conv1_output_size * sizeof(float));

    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int pool1_output_size = LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim;
    state.pool1_output = (float *)malloc(pool1_output_size * sizeof(float));
    state.pool1_indices = (int *)malloc(pool1_output_size * sizeof(int));

    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;
    state.conv2_output = (float *)malloc(conv2_output_size * sizeof(float));
    state.conv2_output_relu = (float *)malloc(conv2_output_size * sizeof(float));

    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int pool2_output_size = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    state.pool2_output = (float *)malloc(pool2_output_size * sizeof(float));
    state.pool2_indices = (int *)malloc(pool2_output_size * sizeof(int));

    state.fc1_output = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    state.fc1_output_relu = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    state.fc2_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    state.softmax_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    // Perform forward pass
    neural_network(image, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                   fc1_weights, fc1_biases, fc2_weights, fc2_biases, &state);

    // Determine predicted class
    int predicted_class = 0;
    float max_probability = state.softmax_output[0];
    for (int j = 1; j < OUTPUT_NEURONS; ++j) {
        if (state.softmax_output[j] > max_probability) {
            max_probability = state.softmax_output[j];
            predicted_class = j;
        }
    }

    // Free allocated memory
    free(state.conv1_output);
    free(state.conv1_output_relu);
    free(state.pool1_output);
    free(state.pool1_indices);
    free(state.conv2_output);
    free(state.conv2_output_relu);
    free(state.pool2_output);
    free(state.pool2_indices);
    free(state.fc1_output);
    free(state.fc1_output_relu);
    free(state.fc2_output);
    free(state.softmax_output);

    return predicted_class;
}


int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));

    if (argc < 2) {
        printf("Usage: %s [train|test|infer] [optional: image_path]\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];

    // Declare weights and biases in main
    float *conv1_weights = (float *)malloc(LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *conv1_biases = (float *)malloc(LAYER1_CHANNELS * sizeof(float));

    float *conv2_weights = (float *)malloc(LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *conv2_biases = (float *)malloc(LAYER2_CHANNELS * sizeof(float));

    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;

    float *fc1_weights = (float *)malloc(FULLY_CONNECTED1_NEURONS * flattened_length * sizeof(float));
    float *fc1_biases = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));

    float *fc2_weights = (float *)malloc(OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float));
    float *fc2_biases = (float *)malloc(OUTPUT_NEURONS * sizeof(float));

    if (strcmp(mode, "train") == 0) {
        // Load training data
        float *train_images;
        int num_train_images, num_train_labels;
        int *train_labels;

        load_mnist_images("train-images.idx3-ubyte", &train_images, &num_train_images);
        load_mnist_labels("train-labels.idx1-ubyte", &train_labels, &num_train_labels);

        if (num_train_images != num_train_labels) {
            fprintf(stderr, "Mismatch between number of training images and labels!\n");
            return 1;
        }

        int num_epochs = 20;
        float learning_rate = 0.01f;

        train_neural_network(train_images, train_labels, num_train_images, num_epochs, learning_rate,
                             conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                             fc1_weights, fc1_biases, fc2_weights, fc2_biases);

        // Free training data
        free(train_images);
        free(train_labels);
    }
    else if (strcmp(mode, "test") == 0) {
        // Check if model parameters exist
        if (!file_exists(CONV1_WEIGHTS_FILE) || !file_exists(CONV1_BIASES_FILE) ||
            !file_exists(CONV2_WEIGHTS_FILE) || !file_exists(CONV2_BIASES_FILE) ||
            !file_exists(FC1_WEIGHTS_FILE) || !file_exists(FC1_BIASES_FILE) ||
            !file_exists(FC2_WEIGHTS_FILE) || !file_exists(FC2_BIASES_FILE)) {
            fprintf(stderr, "Model parameters not found. Please train the model first.\n");
            return 1;
        }

        // Load model parameters
        load_all_parameters(conv1_weights, conv1_biases,
                           conv2_weights, conv2_biases,
                           fc1_weights, fc1_biases,
                           fc2_weights, fc2_biases);

        // Load test data
        float *test_images;
        int num_test_images, num_test_labels;
        int *test_labels;

        load_mnist_images("t10k-images.idx3-ubyte", &test_images, &num_test_images);
        load_mnist_labels("t10k-labels.idx1-ubyte", &test_labels, &num_test_labels);

        if (num_test_images != num_test_labels) {
            fprintf(stderr, "Mismatch between number of test images and labels!\n");
            return 1;
        }

        // Test the model
        test_neural_network(test_images, test_labels, num_test_images, conv1_weights, conv1_biases,
                            conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases);

        // Free test data
        free(test_images);
        free(test_labels);
    }
   else if (strcmp(mode, "infer") == 0) {
        if (argc < 3) {
            printf("Usage for inference: %s infer <image_path>\n", argv[0]);
            return 1;
        }

        const char *image_path = argv[2];

        // Check if model parameters exist
        if (!file_exists(CONV1_WEIGHTS_FILE) || !file_exists(CONV1_BIASES_FILE) ||
            !file_exists(CONV2_WEIGHTS_FILE) || !file_exists(FC1_WEIGHTS_FILE) ||
            !file_exists(FC1_BIASES_FILE) || !file_exists(FC2_WEIGHTS_FILE) ||
            !file_exists(FC2_BIASES_FILE)) {
            fprintf(stderr, "Model parameters not found. Please train the model first.\n");
            return 1;
        }

        // Load model parameters
        load_all_parameters(conv1_weights, conv1_biases,
                        conv2_weights, conv2_biases,
                        fc1_weights, fc1_biases,
                        fc2_weights, fc2_biases);

        // Load and preprocess the custom image
        float custom_image[IMAGE_DIMENSION * IMAGE_DIMENSION];
        load_custom_image(image_path, custom_image);

        // Perform inference
        int predicted_class = infer(custom_image, conv1_weights, conv1_biases,
                                    conv2_weights, conv2_biases,
                                    fc1_weights, fc1_biases,
                                    fc2_weights, fc2_biases);

        printf("Predicted class: %d\n", predicted_class);
    }

    else {
        printf("Unknown mode: %s\n", mode);
        printf("Usage: %s [train|test|infer] [optional: image_path]\n", argv[0]);
        return 1;
    }

    // Free allocated memory for weights and biases
    free(conv1_weights);
    free(conv1_biases);
    free(conv2_weights);
    free(conv2_biases);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);

    return 0;
}

    
