#ifndef TSNE_H
#define TSNE_H

typedef struct
{
    int iterations;

    double perplexity;
    double eta;
    
    double initial_momentum;
    double final_momentum;
} tsne_hyperparameters;

typedef struct
{
    // Number of data points
    int n; 
    
    // Input & Output dimensions
    int input_dim; 
    int output_dim; 

    // Input & Output vector
    double *input;
    double *output;
} tsne_data;

typedef void (*tsne_function)(tsne_data input, tsne_hyperparameters hyperparams);
typedef uint64_t (*tsne_flops_function)(tsne_data input, tsne_hyperparameters hyperparams);
#endif