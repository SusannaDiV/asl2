#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <stdint.h>

#include <tuple>
#include <map>

#include "papi.h"

#include "cli_arguments.h"
#include "input_output.h"

extern "C"
{
#include "tsc_x86.h"

#include "tsne/tsne.h"
#include "tsne/tsne_1.h"
#include "tsne/tsne_2.h"
#include "tsne/tsne_3.h"
#include "tsne/tsne_4.h"
#include "tsne/tsne_5.h"
// Add more implementations here!
}

using namespace std;

typedef struct
{
    tsne_function fn_ptr;
    
    uint64_t cycles;
    uint64_t flops;
    bool correct;
} tsne_funtion_data_t;


tsne_hyperparameters hyperparams;
cli_options options;

std::map<std::string, tsne_funtion_data_t> function_data;

void register_function(tsne_function fn_ptr, std::string name)
{
    tsne_funtion_data_t data;
    data.fn_ptr = fn_ptr;
    
    function_data[name] = data;
}

void register_functions()
{
    // Register more functions here!
    register_function(&tsne_1, "0_Baseline");
    //register_function(&tsne_2, "1_Online_Q");
    //register_function(&tsne_3, "2_Compress_P");
    register_function(&tsne_5, "4_AVX_Rest");
    register_function(&tsne_4, "3_Next");
}

tsne_data copy_input(tsne_data input)
{
    tsne_data copy;

    copy.n = input.n;

    copy.input_dim = input.input_dim;
    copy.output_dim = input.output_dim;

    copy.input = (double *)malloc(sizeof(double) * input.n * input.input_dim);
    memcpy(copy.input, input.input, sizeof(double) * input.n * input.input_dim);

    copy.output = (double *)malloc(sizeof(double) * input.n * input.output_dim);
    memcpy(copy.output, input.output, sizeof(double) * input.n * input.output_dim);

    return copy;
}

void free_input(tsne_data input)
{
    free(input.input);
    free(input.output);
}

tsne_data load_input(std::string input_file_path, std::string output_file_path)
{
    tsne_data input;

    int nx;
    if (load(input_file_path, &nx, &input.input_dim, &input.input) != 0)
    {
        cerr << "ERROR: Failed to load" << input_file_path << "!" << endl;
        exit(1);
    }

    int ny;
    if (load(output_file_path, &ny, &input.output_dim, &input.output) != 0)
    {
        cerr << "ERROR: Failed to load" << output_file_path << "!" << endl;
        exit(1);
    }

    if (nx != ny)
    {
        cerr << "ERROR: nx != ny!" << endl;
        exit(1);
    }

    input.n = nx;

    return input;
}

bool is_correct(tsne_data reference, tsne_data check)
{
    int n, d;
    n = reference.n;
    d = reference.output_dim;

    double max_difference = 0.0;
    double max_value = 0.0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            if (check.output[i * d + j] != check.output[i * d + j]) // Check for nan
                max_difference = INFINITY;
            max_value = fmax(max_value, fabs(reference.output[i * d + j]));
            max_difference = fmax(max_difference, fabs(reference.output[i * d + j] - check.output[i * d + j]));
        }
    }

    printf("Error = %lf\n", max_difference/max_value);

    return max_difference / max_value < 0.01;
}

int init_papi() {
    int retval = PAPI_OK;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT)
    {
        fprintf(stderr, "Error initializing PAPI! %s\n",
                PAPI_strerror(retval));
    }

    int event_set = PAPI_NULL;
    retval = PAPI_create_eventset(&event_set);
    if (retval != PAPI_OK)
    {
        fprintf(stderr, "Error creating eventset! %s\n",
                PAPI_strerror(retval));
    }

    retval = PAPI_add_event(event_set, PAPI_DP_OPS);
    if (retval != PAPI_OK)
    {
        fprintf(stderr, "Error adding PAPI_DP_OPS: %s\n",
                PAPI_strerror(retval));
    }

    return event_set;
}

void papi_start(int event_set) {
    PAPI_start(event_set);
}

uint64_t papi_stop(int event_set) {
    PAPI_stop(event_set, NULL);

    long long flops = 0;
    PAPI_read(event_set, &flops);
    return flops;
}

int main(int argc, char **argv)
{
    setvbuf(stdin, NULL, _IONBF, 0);

    // Hyperparameters
    hyperparams.iterations = 1000;
    hyperparams.eta = 500;
    hyperparams.initial_momentum = 0.5;
    hyperparams.final_momentum = 0.8;
    hyperparams.perplexity = 30.0;

    // Load options from cli arguments
    options.input_file_path = "./data/mnist500_x.data";
    options.seed_file_path = "./data/mnist500_seed.data";
    options.reference_file_path = "./data/mnist500_reference.data";
    options.output_directory = "";
    options.return_correctness = true;

    load_cli_options(&options, argc, argv);

    // Iterate all registered functions
    register_functions();

    // Loads data from files
    tsne_data base_input = load_input(options.input_file_path, options.seed_file_path);
    tsne_data reference = load_input(options.input_file_path, options.reference_file_path);

    bool are_all_correct = true;

    int papi_event_set = init_papi();

    for (auto &[name, tsne_function_data] : function_data)
    {
        tsne_data data = copy_input(base_input);

        std::cout << "Running " << name << "..." << std::endl;

        papi_start(papi_event_set);
        uint64_t cycles = start_tsc();

        tsne_function_data.fn_ptr(data, hyperparams);

        cycles = stop_tsc(cycles);
        tsne_function_data.flops = papi_stop(papi_event_set);
        tsne_function_data.cycles = cycles;

        if (options.output_directory != "") {
            save(options.output_directory + "/" + name, data.output, data.n, data.output_dim);
        }

        if (!is_correct(reference, data))
        {
            tsne_function_data.correct = false;
            std::cout << "Function " << name << " is not correct!" << std::endl;
        }
        else
        {
            tsne_function_data.correct = true;
            std::cout << "Function " << name << " passed and took " << cycles << " cycles" << std::endl;
        }

        free_input(data);
    }

    cout << endl;
    cout << "===================[ SUMMARY ]====================" << endl << endl;

    cout << "-----------------------------------------------------------------------------" << endl;
    cout << "| ";
    cout.width(15); cout << left << "Name" << " | ";
    cout.width(12); cout << left << "Cycles" << " | ";
    cout.width(7); cout << left << "Time" << " | ";
    cout.width(7); cout << left << "Speedup" << " | ";
    cout.width(11); cout << left << "Flops/Cycle" << " | ";
    cout.width(6); cout << left << "Passed" << " | ";
    cout << endl;

    for (auto &[name, tsne_function_data] : function_data)
    {
        double time = tsne_function_data.cycles / (double) CPU_FREQ;
        double speedup = (double) function_data["0_Baseline"].cycles / (double) tsne_function_data.cycles;
        double performance = (double) tsne_function_data.flops / (double) tsne_function_data.cycles;

        cout << "-----------------------------------------------------------------------------" << endl;
        cout << "| ";
        cout.width(15); cout << left << name << " | ";
        cout.width(12); cout << left << tsne_function_data.cycles << " | ";
        cout.width(7); cout << left << fixed << setprecision(2) << time << " | ";
        cout.width(7); cout << left << fixed << setprecision(2) << speedup << " | ";
        cout.width(11); cout << left << fixed << setprecision(2) << performance << " | ";
        if (tsne_function_data.correct) { 
            cout.width(6); cout << left << "Y" << " | ";
        } else {
            are_all_correct = false;
            cout.width(6); cout << left << "N" << " | ";
        }
        
        cout << endl;
    }
    cout << "-----------------------------------------------------------------------------" << endl;

    free_input(base_input);
    free_input(reference);

    if (options.return_correctness && !are_all_correct) {
        return 1;
    }
    return 0;
}