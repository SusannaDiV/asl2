#!/bin/bash
set -e
if [ -d "tmp" ]; then
    rm -r tmp
fi

mkdir -p tmp
mkdir -p tmp/measurements

for n in 64 128 256 512 1024 2048; do
  echo "Measuring n=$n..."

  python3 tools/sample.py -i data/mnist2500_x.data -o tmp/input.data -n $n
  python3 tools/seed.py -o tmp/seed.data -n $n

  echo "$n" > tmp/measurements/$n.data
  ./build/tsne -q -i tmp/input.data -s tmp/seed.data -r tmp/seed.data >> tmp/measurements/$n.data
done

python3 tools/plot_measurements.py tmp/measurements runtime_plot.pdf performance_plot.pdf
