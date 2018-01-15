import os
import subprocess
import re

TIMES = 20
jacobi = []
gauss = []
redblack = []
r = list(range(1, 13))

def get_mean_time(solver, times):
    avg = 0
    for _ in range(times):
        avg += get_time(solver) / times
    return avg

def get_time(solver):
    output = subprocess.Popen(["./heatomp", "test_{solver}.dat".format(solver=solver)],
                              stdout=subprocess.PIPE).communicate()[0]
    matched = re.search("Time:\s+(\d+\.\d+)", str(output))
    return float(matched.group(1))

for i in r:
    os.environ['OMP_NUM_THREADS'] = str(i)
    jacobi.append(get_mean_time('jacobi', TIMES))
    gauss.append(get_mean_time('gauss', TIMES))
    redblack.append(get_mean_time('redblack', TIMES))

print('jacobi\n', jacobi)
print('gauss\n', gauss)
print('red-black\n', redblack)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_to_file(jacobi, gauss, redblack, scale='linear'):
    plt.plot(r, jacobi, 'b', label='jacobi')
    plt.plot(r, gauss, 'g', label='gauss')
    plt.plot(r, redblack, 'r', label='red-black')
    plt.ylabel('Mean execution time')
    plt.xlabel('Number of threads')
    plt.legend()
    if scale == 'log':
        plt.yscale('log')
        plt.savefig('scalability-log')
    else:
        plt.savefig('scalability')
    plt.close()

save_to_file(jacobi, gauss, redblack)
save_to_file(jacobi, gauss, redblack, scale='log')