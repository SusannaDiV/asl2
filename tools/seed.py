# Helper to create randomized initializations.

import numpy as np
import click
from input_output import save

@click.command()
@click.option('-n')
@click.option('-o', '--out_path')
def main(n, out_path):
    Y = np.random.randn(int(n), 2)
    save(Y, out_path)

if __name__ == '__main__':
    main()