#include "stdbool.h"
#include <string>

typedef struct
{
    std::string input_file_path;
    std::string seed_file_path;

    std::string reference_file_path;
    std::string output_directory;

    bool return_correctness;
} cli_options;

// We have the following options:
// -i   Path to the input data file
// -s   Path to the seed file
void load_cli_options(cli_options* options, int argc, char **argv);