import pylab
import os
import click
import numpy as np
from input_output import load

def parse_measurement_file(path):
    with open(path) as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        measurements = {}

        for l in lines:
            if not l.startswith('| ') or l.startswith("| Name"):
                continue
            
            splitted = [x.strip() for x in l.split("|")]
            measurements[splitted[1]] = (int(splitted[2]), float(splitted[5]))
    return n, measurements

def runtime_plot(data, output_path):
    pylab.cla()

    xs = sorted(list(data.keys()))

    for function_name in data[xs[0]].keys():
        ys = [data[n][function_name][0] for n in xs]

        pylab.plot(xs, ys, marker='o', label=function_name)

    pylab.legend()
    pylab.title("Runtime: MNIST")
    pylab.xlabel("n")
    pylab.ylabel("Cycles")    
    pylab.savefig(output_path)

def performance_plot(data, output_path):
    pylab.cla()

    xs = sorted(list(data.keys()))

    for function_name in data[xs[0]].keys():
        ys = [data[n][function_name][1] for n in xs]

        pylab.plot(xs, ys, marker='o', label=function_name)

    pylab.legend()
    pylab.title("Performance: MNIST")
    pylab.xlabel("n")
    pylab.ylabel("Flops/Cycles")    
    pylab.savefig(output_path)

@click.command()
@click.argument('measurements_directory')
@click.argument('runtime_output_path')
@click.argument('performance_output_path')
def main(measurements_directory, runtime_output_path, performance_output_path):
    data = {}
    
    for entry in os.listdir(measurements_directory):
        if not entry.endswith('.data'):
            continue
        full_entry_path = os.path.join(measurements_directory, entry)
        if os.path.isfile(full_entry_path):
            n, measurements = parse_measurement_file(full_entry_path)
            data[n] = measurements

    runtime_plot(data, runtime_output_path)
    performance_plot(data, performance_output_path)
            
if __name__ == '__main__':
    main()