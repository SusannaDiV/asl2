#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>

#include "input_output.h"

// Loads in a vector from a file in our format. Outputs the number of elements n the dimensionality d.
int load(std::string file_path, int *n_out, int *d_out, double **buffer_out)
{
    FILE *fd;

    fd = fopen(file_path.data(), "rb");
    if (fd == NULL)
    {
        printf("Cannot open file %s.\n", file_path.data());
        return EXIT_FAILURE;
    }

    int64_t d, n;

    if (fread(&d, sizeof(d), 1, fd) != 1)
    {
        printf("Failed to read d from %s.\n", file_path.data());
        return EXIT_FAILURE;
    }

    if (fread(&n, sizeof(n), 1, fd) != 1)
    {
        printf("Failed to read n from %s.", file_path.data());
        return EXIT_FAILURE;
    }

    double *buffer = (double*) malloc(sizeof(double) * d * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double x;

            if (fread(&x, sizeof(x), 1, fd) != 1)
            {
                printf("Failed to read double at %d %d from %s.", i, j, file_path.data());
                return EXIT_FAILURE;
            }
            buffer[i * d + j] = x;
        }
    }

    fclose(fd);

    *n_out = n;
    *d_out = d;
    *buffer_out = buffer;

    return EXIT_SUCCESS;
}

int save(std::string file_path, double *X, uint64_t n, uint64_t d)
{
    FILE *fd;

    fd = fopen(file_path.data(), "wb");
    if (fd == NULL)
    {
        printf("Cannot open file %s.\n", file_path.data());
        return EXIT_FAILURE;
    }

    if (fwrite(&d, sizeof(d), 1, fd) != 1)
    {
        printf("Failed to write d to %s.\n", file_path.data());
        return EXIT_FAILURE;
    }

    if (fwrite(&n, sizeof(n), 1, fd) != 1)
    {
        printf("Failed to write n to %s.\n", file_path.data());
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            if (fwrite(&X[i*d+j], sizeof(double), 1, fd) != 1)
            {
                printf("Failed to read double at %d %d from %s.\n", i, j, file_path.data());
                return EXIT_FAILURE;
            }
        }
    }

    fclose(fd);

    return EXIT_SUCCESS;
}