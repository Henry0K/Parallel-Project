/*
 Authors: Henry Hardane, Alex Kheir, Lori Yahnian, and Alain Samaha
 Title: Reduced Model LeNet-5 Parallelized with MPI
 Description: MPI parallelized implementation of the LeNet-5 architecture.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
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

// Activation Functions
void activate_relu(float *vector, int length) {
    for (int i = 0; i < length; ++i) {
        vector[i] = (vector[i] > 0) ? vector[i] : 0;
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
void convolution_operation(float *input, float *filters, float *output, int input_dim, int num_filters,
                           int kernel_dim, int filter_start, int filter_end) {
    int output_dim = input_dim - kernel_dim + 1;
    for (int f = filter_start; f < filter_end; ++f) {
        for (int row = 0; row < output_dim; ++row) {
            for (int col = 0; col < output_dim; ++col) {
                float acc = 0.0f;
                for (int i = 0; i < kernel_dim; ++i) {
                    for (int j = 0; j < kernel_dim; ++j) {
                        acc += input[COORD(row + i, col + j, input_dim)] *
                               filters[f * kernel_dim * kernel_dim + COORD(i, j, kernel_dim)];
                    }
                }
                output[(f - filter_start) * output_dim * output_dim + COORD(row, col, output_dim)] = acc;
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
                        max_value = MAX_VAL(
                            max_value,
                            input[c * input_dim * input_dim +
                                  COORD(r * pool_dim + i, col * pool_dim + j, input_dim)]);
                    }
                }
                output[c * output_dim * output_dim + COORD(r, col, output_dim)] = max_value;
            }
        }
    }
}

// Fully Connected Layer
void dense_layer(float *input, float *weights, float *biases, float *output, int input_length,
                 int output_length, int neuron_start, int neuron_end) {
    for (int o = neuron_start; o < neuron_end; ++o) {
        float sum = biases[o];
        for (int i = 0; i < input_length; ++i) {
            sum += input[i] * weights[o * input_length + i];
        }
        output[o - neuron_start] = sum;
    }
}

// Function to load data from a binary file
void load_data(const char *filename, float *data, size_t size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for reading.\n", filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
void neural_network(float *input, float *conv1_weights, float *conv2_weights, float *fc1_weights,
                    float *fc1_biases, float *fc2_weights, float *fc2_biases, float *final_output,
                    int rank, int size) {
    double start_time = MPI_Wtime();
    // Convolution Layer 1
    int conv1_output_dim = IMAGE_DIMENSION - KERNEL_SIZE + 1;
    int conv1_output_size = LAYER1_CHANNELS * conv1_output_dim * conv1_output_dim;

    int base_filters_per_proc = LAYER1_CHANNELS / size;
    int extra_filters = LAYER1_CHANNELS % size;
    int conv1_filters_per_proc = base_filters_per_proc + (rank < extra_filters ? 1 : 0);
    int filter_start = rank * base_filters_per_proc + (rank < extra_filters ? rank : extra_filters);
    int filter_end = filter_start + conv1_filters_per_proc;

    int local_conv1_output_size = conv1_filters_per_proc * conv1_output_dim * conv1_output_dim;
    float *conv1_output_local = (float *)malloc(local_conv1_output_size * sizeof(float));

    // Timing for Convolution Layer 1
    double conv1_start_time = MPI_Wtime();
    convolution_operation(input, conv1_weights, conv1_output_local, IMAGE_DIMENSION, LAYER1_CHANNELS,
                          KERNEL_SIZE, filter_start, filter_end);
    activate_relu(conv1_output_local, local_conv1_output_size);
    double conv1_end_time = MPI_Wtime();
    double conv1_time = conv1_end_time - conv1_start_time;

    // Pooling Layer 1
    int pool1_output_dim = conv1_output_dim / POOL_KERNEL;
    int local_pool1_output_size = conv1_filters_per_proc * pool1_output_dim * pool1_output_dim;
    float *pool1_output_local = (float *)malloc(local_pool1_output_size * sizeof(float));

    // Timing for Pooling Layer 1
    double pool1_start_time = MPI_Wtime();
    pooling_operation(conv1_output_local, pool1_output_local, conv1_output_dim, POOL_KERNEL,
                      conv1_filters_per_proc);

    // Gather all pool1 outputs to all processes
    int *recvcounts_conv1 = (int *)malloc(size * sizeof(int));
    int *displs_conv1 = (int *)malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int filters_per_proc = base_filters_per_proc + (i < extra_filters ? 1 : 0);
        recvcounts_conv1[i] = filters_per_proc * pool1_output_dim * pool1_output_dim;
        displs_conv1[i] = offset;
        offset += recvcounts_conv1[i];
    }
    float *pool1_output_full = (float *)malloc(LAYER1_CHANNELS * pool1_output_dim * pool1_output_dim * sizeof(float));
    MPI_Allgatherv(pool1_output_local, local_pool1_output_size, MPI_FLOAT,
                   pool1_output_full, recvcounts_conv1, displs_conv1, MPI_FLOAT, MPI_COMM_WORLD);
    double pool1_end_time = MPI_Wtime();
    double pool1_time = pool1_end_time - pool1_start_time;

    // Convolution Layer 2
    int conv2_output_dim = pool1_output_dim - KERNEL_SIZE + 1;
    int conv2_output_size = LAYER2_CHANNELS * conv2_output_dim * conv2_output_dim;

    base_filters_per_proc = LAYER2_CHANNELS / size;
    extra_filters = LAYER2_CHANNELS % size;
    int conv2_filters_per_proc = base_filters_per_proc + (rank < extra_filters ? 1 : 0);
    filter_start = rank * base_filters_per_proc + (rank < extra_filters ? rank : extra_filters);
    filter_end = filter_start + conv2_filters_per_proc;

    int local_conv2_output_size = conv2_filters_per_proc * conv2_output_dim * conv2_output_dim;
    float *conv2_output_local = (float *)malloc(local_conv2_output_size * sizeof(float));

    // Timing for Convolution Layer 2
    double conv2_start_time = MPI_Wtime();
    convolution_operation(pool1_output_full, conv2_weights, conv2_output_local, pool1_output_dim,
                          LAYER2_CHANNELS, KERNEL_SIZE, filter_start, filter_end);
    activate_relu(conv2_output_local, local_conv2_output_size);
    double conv2_end_time = MPI_Wtime();
    double conv2_time = conv2_end_time - conv2_start_time;

    free(pool1_output_full);

    // Pooling Layer 2
    int pool2_output_dim = conv2_output_dim / POOL_KERNEL;
    int local_pool2_output_size = conv2_filters_per_proc * pool2_output_dim * pool2_output_dim;
    float *pool2_output_local = (float *)malloc(local_pool2_output_size * sizeof(float));

    // Timing for Pooling Layer 2
    double pool2_start_time = MPI_Wtime();
    pooling_operation(conv2_output_local, pool2_output_local, conv2_output_dim, POOL_KERNEL,
                      conv2_filters_per_proc);

    free(conv2_output_local);

    // Gather all pool2 outputs to all processes
    int *recvcounts_conv2 = (int *)malloc(size * sizeof(int));
    int *displs_conv2 = (int *)malloc(size * sizeof(int));
    offset = 0;
    for (int i = 0; i < size; ++i) {
        int filters_per_proc = base_filters_per_proc + (i < extra_filters ? 1 : 0);
        recvcounts_conv2[i] = filters_per_proc * pool2_output_dim * pool2_output_dim;
        displs_conv2[i] = offset;
        offset += recvcounts_conv2[i];
    }
    int flattened_length = LAYER2_CHANNELS * pool2_output_dim * pool2_output_dim;
    float *pool2_output_full = (float *)malloc(flattened_length * sizeof(float));
    MPI_Allgatherv(pool2_output_local, local_pool2_output_size, MPI_FLOAT,
                   pool2_output_full, recvcounts_conv2, displs_conv2, MPI_FLOAT, MPI_COMM_WORLD);
    double pool2_end_time = MPI_Wtime();
    double pool2_time = pool2_end_time - pool2_start_time;

    free(pool2_output_local);
    free(recvcounts_conv2);
    free(displs_conv2);

    // Fully Connected Layer 1
    int fc1_input_size = flattened_length;
    base_filters_per_proc = FULLY_CONNECTED1_NEURONS / size;
    extra_filters = FULLY_CONNECTED1_NEURONS % size;
    int fc1_neurons_per_proc = base_filters_per_proc + (rank < extra_filters ? 1 : 0);
    int neuron_start = rank * base_filters_per_proc + (rank < extra_filters ? rank : extra_filters);
    int neuron_end = neuron_start + fc1_neurons_per_proc;

    float *fc1_output_local = (float *)malloc(fc1_neurons_per_proc * sizeof(float));

    // Timing for Fully Connected Layer 1
    double fc1_start_time = MPI_Wtime();
    dense_layer(pool2_output_full, fc1_weights, fc1_biases, fc1_output_local, fc1_input_size,
                FULLY_CONNECTED1_NEURONS, neuron_start, neuron_end);
    activate_relu(fc1_output_local, fc1_neurons_per_proc);

    free(pool2_output_full);

    // Gather all fc1 outputs to all processes
    int *recvcounts_fc1 = (int *)malloc(size * sizeof(int));
    int *displs_fc1 = (int *)malloc(size * sizeof(int));
    offset = 0;
    for (int i = 0; i < size; ++i) {
        int neurons_per_proc = base_filters_per_proc + (i < extra_filters ? 1 : 0);
        recvcounts_fc1[i] = neurons_per_proc;
        displs_fc1[i] = offset;
        offset += recvcounts_fc1[i];
    }
    float *fc1_output_full = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    MPI_Allgatherv(fc1_output_local, fc1_neurons_per_proc, MPI_FLOAT,
                   fc1_output_full, recvcounts_fc1, displs_fc1, MPI_FLOAT, MPI_COMM_WORLD);
    double fc1_end_time = MPI_Wtime();
    double fc1_time = fc1_end_time - fc1_start_time;

    free(fc1_output_local);
    free(recvcounts_fc1);
    free(displs_fc1);

    // Fully Connected Layer 2
    base_filters_per_proc = OUTPUT_NEURONS / size;
    extra_filters = OUTPUT_NEURONS % size;
    int fc2_neurons_per_proc = base_filters_per_proc + (rank < extra_filters ? 1 : 0);
    neuron_start = rank * base_filters_per_proc + (rank < extra_filters ? rank : extra_filters);
    neuron_end = neuron_start + fc2_neurons_per_proc;

    float *fc2_output_local = (float *)malloc(fc2_neurons_per_proc * sizeof(float));

    // Timing for Fully Connected Layer 2
    double fc2_start_time = MPI_Wtime();
    dense_layer(fc1_output_full, fc2_weights, fc2_biases, fc2_output_local, FULLY_CONNECTED1_NEURONS,
                OUTPUT_NEURONS, neuron_start, neuron_end);

    free(fc1_output_full);

    // Gather all fc2 outputs to process 0
    int *recvcounts_fc2 = (int *)malloc(size * sizeof(int));
    int *displs_fc2 = (int *)malloc(size * sizeof(int));
    offset = 0;
    for (int i = 0; i < size; ++i) {
        int neurons_per_proc = base_filters_per_proc + (i < extra_filters ? 1 : 0);
        recvcounts_fc2[i] = neurons_per_proc;
        displs_fc2[i] = offset;
        offset += recvcounts_fc2[i];
    }
    float *fc2_output_full = NULL;
    if (rank == 0) {
        fc2_output_full = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    }
    MPI_Gatherv(fc2_output_local, fc2_neurons_per_proc, MPI_FLOAT,
                fc2_output_full, recvcounts_fc2, displs_fc2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double fc2_end_time = MPI_Wtime();
    double fc2_time = fc2_end_time - fc2_start_time;
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    free(fc2_output_local);
    free(recvcounts_fc2);
    free(displs_fc2);

    if (rank == 0) {
        apply_softmax_function(fc2_output_full, OUTPUT_NEURONS);
        for (int i = 0; i < OUTPUT_NEURONS; ++i) {
            final_output[i] = fc2_output_full[i];
        }

        // Print timing information
        printf("Layer Execution Times (seconds):\n");
        printf("Convolution Layer 1: %f\n", conv1_time);
        printf("Pooling Layer 1:     %f\n", pool1_time);
        printf("Convolution Layer 2: %f\n", conv2_time);
        printf("Pooling Layer 2:     %f\n", pool2_time);
        printf("Fully Connected 1:   %f\n", fc1_time);
        printf("Fully Connected 2:   %f\n", fc2_time);
        printf("Total Execution Time: %f\n", total_time);

        // Display Output
        printf("Output Probabilities:\n");
        for (int i = 0; i < OUTPUT_NEURONS; ++i) {
            printf("Class %d: %f\n", i, final_output[i]);
        }

        // Save output to file
        save_output("../Data/mpi_output.txt", final_output, OUTPUT_NEURONS);


        free(fc2_output_full); // Only process 0 allocated this
    }

    free(conv1_output_local);
    free(pool1_output_local);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42);

    // Input and Weights Allocation
    float *image = NULL;
    float *conv1_weights = NULL;
    float *conv2_weights = NULL;
    int fc1_input_size =
        LAYER2_CHANNELS * ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4) *
        ((IMAGE_DIMENSION - 2 * (KERNEL_SIZE - 1)) / 4);
    float *fc1_weights = NULL;
    float *fc1_biases = NULL;
    float *fc2_weights = NULL;
    float *fc2_biases = NULL;
    float *final_output = NULL;

    // Allocate memory on all processes
    image = (float *)malloc(IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(float));
    conv1_weights = (float *)malloc(LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    conv2_weights = (float *)malloc(LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    fc1_weights =
        (float *)malloc(FULLY_CONNECTED1_NEURONS * fc1_input_size * sizeof(float));
    fc1_biases = (float *)malloc(FULLY_CONNECTED1_NEURONS * sizeof(float));
    fc2_weights =
        (float *)malloc(OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS * sizeof(float));
    fc2_biases = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    if (rank == 0) {
        final_output = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    }

    // Load data from files on rank 0
    if (rank == 0) {
        load_data("../Data/input_image.bin", image,
                  IMAGE_DIMENSION * IMAGE_DIMENSION);
        load_data("../Data/conv1_weights.bin", conv1_weights,
                  LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
        load_data("../Data/conv2_weights.bin", conv2_weights,
                  LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
        load_data("../Data/fc1_weights.bin", fc1_weights,
                  FULLY_CONNECTED1_NEURONS * fc1_input_size);
        load_data("../Data/fc1_biases.bin", fc1_biases, FULLY_CONNECTED1_NEURONS);
        load_data("../Data/fc2_weights.bin", fc2_weights,
                  OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS);
        load_data("../Data/fc2_biases.bin", fc2_biases, OUTPUT_NEURONS);
    }

    // Broadcast data to all processes
    MPI_Bcast(image, IMAGE_DIMENSION * IMAGE_DIMENSION, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv1_weights, LAYER1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(conv2_weights, LAYER2_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(fc1_weights, FULLY_CONNECTED1_NEURONS * fc1_input_size, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(fc1_biases, FULLY_CONNECTED1_NEURONS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(fc2_weights, OUTPUT_NEURONS * FULLY_CONNECTED1_NEURONS, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(fc2_biases, OUTPUT_NEURONS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Execute the Neural Network
    if (rank == 0) {
        printf("Running MPI Parallel Neural Network...\n");
    }
    neural_network(image, conv1_weights, conv2_weights, fc1_weights, fc1_biases, fc2_weights,
                   fc2_biases, final_output, rank, size);

    // Free Allocated Memory
    free(image);
    free(conv1_weights);
    free(conv2_weights);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    if (rank == 0) {
        free(final_output);
    }

    MPI_Finalize();
    return 0;
}
