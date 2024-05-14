# Helper to preprocess npy data files and output format expected by our implementation

import numpy as np
import click

from input_output import save

def pca(X=np.array([]), no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, _) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (_, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y.real

@click.command()
@click.option('-i', '--in_path')
@click.option('-o', '--out_path')
def main(in_path, out_path):
    X = np.loadtxt(in_path)
    X = pca(X)
    
    save(X, 'mnist2500_x.data')
    

if __name__ == '__main__':
    main()