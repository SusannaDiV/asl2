# Helper to preprocess npy data files and output format expected by our implementation

import numpy as np
import click

from input_output import save, load

@click.command()
@click.option('-i', '--in_path')
@click.option('-l', '--labels_path')
@click.option('-o', '--out_path')
@click.option('-a', '--labels_out_path')
@click.option('-n', '--number', type=int)
def main(in_path, out_path, number, labels_path, labels_out_path):
    X = load(in_path)
    sampled_indices = np.random.choice(len(X), size=number, replace=False)
    X = X[sampled_indices]
    save(X, out_path)

    if labels_out_path:
        labels = np.loadtxt(labels_path)
        labels = labels[sampled_indices]
        np.savetxt(labels_out_path, labels)
    

if __name__ == '__main__':
    main()