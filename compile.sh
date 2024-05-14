#!/bin/bash
set -e
if [ "$DEBUG" = "1" ]; then
    CC_FLAGS="-g -gdwarf-4 -O3 -march=native -fno-inline -fsanitize=address"
else
    CC_FLAGS="-Ofast -march=native"
fi

mkdir -p build

C_FILES=("src/tsne/tsne_1.c" "src/tsne/tsne_2.c" "src/tsne/tsne_3.c" "src/tsne/tsne_4.c" "src/tsne/tsne_5.c")
CPP_FILES=("src/cli_arguments.cpp" "src/input_output.cpp")

for file in "${C_FILES[@]}"; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    clang -c $file $CC_FLAGS -o build/$filename.o
done

for file in "${CPP_FILES[@]}"; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    clang++ -std=c++17 -c $file $CC_FLAGS -o build/$filename.o
done

clang++ -std=c++17 build/tsne_1.o build/tsne_2.o build/tsne_3.o build/tsne_4.o build/tsne_5.o build/cli_arguments.o build/input_output.o src/driver.cpp -O3 -o build/tsne -lpapi -static
