import click
import numpy as np

from input_output import load

@click.command()
@click.argument('path1')
@click.argument('path2')
@click.option('--fail_max_difference', default=1.0)
@click.option('--fail_avg_difference', default=1.0)
def main(path1, path2, fail_max_difference, fail_avg_difference):
    X1 = load(path1)
    X2 = load(path2)

    print(f'{X1.shape=} {X2.shape}')

    normalize = max(np.abs(X1).max(), np.abs(X1).max())

    max_difference = np.abs(X1-X2).max() / normalize
    avg_difference = np.abs(X1-X2).mean() / normalize

    print(f'max(difference) = {100*max_difference}%')
    print(f'avg(difference) = {100*avg_difference}%')

    if max_difference >= fail_max_difference:
        print(f'Failed because max_difference={max_difference} > {fail_max_difference}=fail_max_difference')
        exit(1)

    if avg_difference >= fail_avg_difference:
        print(f'Failed because avg_difference={avg_difference} > {fail_avg_difference}=fail_avg_difference')
        exit(1)

    print("Success!")

if __name__ == '__main__':
    main()