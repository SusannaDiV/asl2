import pylab
import click
import numpy as np
from input_output import load

@click.command()
@click.argument('y_path')
@click.argument('output_path')
@click.option('--labels')
def main(y_path, output_path, labels):
    Y = load(y_path)

    if labels:
        pylab.scatter(Y[:, 0], Y[:, 1], 20, np.loadtxt(labels))
    else:
        pylab.scatter(Y[:, 0], Y[:, 1], 20)

    pylab.savefig(output_path)

if __name__ == '__main__':
    main()