#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "cli_arguments.h"

void load_cli_options(cli_options *options, int argc, char **argv)
{
    int opt;

    while ((opt = getopt(argc, argv, "i:s:r:o:q")) != -1)
    {
        switch (opt)
        {
        case 'i':
            options->input_file_path = optarg;
            break;
        case 's':
            options->seed_file_path = optarg;
            break;
        case 'r':
            options->reference_file_path = optarg;
            break;
        case 'o':
            options->output_directory = optarg;
            break;
        case 'q':
            options->return_correctness = false;
            break;
        default:
            fprintf(stderr, "Usage: %s -i <input> -s <seed> -r <reference> -o <output_directory>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (options->input_file_path == "") {
        fprintf(stderr, "ERROR: -i is required!\n");
        exit(1);
    }

    if (options->seed_file_path == "") {
        fprintf(stderr, "ERROR: -s is required!\n");
        exit(1);
    }

    printf("Parsed arguments: \n");
    printf("- Input File: %s\n", options->input_file_path.data());
    printf("- Seed File: %s\n", options->seed_file_path.data());
    printf("- Reference File: %s\n", options->reference_file_path.data());
    printf("- Output Directory: %s\n", options->output_directory.data());

    printf("\n");
}