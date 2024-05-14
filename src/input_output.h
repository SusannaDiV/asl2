#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int load(std::string file_path, int *n_out, int *d_out, double **buffer_out);

int save(std::string file_path, double *X, uint64_t n, uint64_t d);