FAIL_MAX_DIFFERENCE=0.05
FAIL_AVG_DIFFERENCE=0.01

mkdir -p tmp
mkdir -p tmp/outputs
mkdir -p visuals

# Test on full mnist with fixed seed
./build/tsne -i data/mnist500_x.data -s data/mnist500_seed.data -r data/mnist500_reference.data -o tmp/outputs

# Plot outputs
python3 tools/plot.py data/mnist500_reference.data visuals/Reference.pdf --labels data/mnist500_labels.txt
for filename in tmp/outputs/*; do
    bname=$(basename -- "$filename")
    python3 tools/plot.py "$filename" "visuals/$bname.pdf" --labels data/mnist500_labels.txt
done