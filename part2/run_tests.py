import os
import subprocess
import re

TIMES = 5
jacobi = []
gauss = []
redblack = []
redblackalt = []
r = list(range(1, 13))

def get_mean_time(solver, times):
    avg = 0
    for _ in range(times):
        avg += get_time(solver) / times
    return avg

def get_time(solver):
    output = subprocess.Popen(["./heatomp", "../tests/test_{solver}.dat".format(solver=solver)],
                              stdout=subprocess.PIPE).communicate()[0]
    matched = re.search("Time:\s+(\d+\.\d+)", str(output))
    return float(matched.group(1))

for i in r:
    os.environ['OMP_NUM_THREADS'] = str(i)
    jacobi.append(get_mean_time('jacobi', TIMES))
    gauss.append(get_mean_time('gauss', TIMES))
    redblack.append(get_mean_time('redblack', TIMES))
    redblackalt.append(get_mean_time('redblack_alternative', TIMES))

print('jacobi\n', jacobi)
print('gauss\n', gauss)
print('red-black\n', redblack)
print('red-black-alternative\n', redblackalt)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_to_file(jacobi, gauss, redblack, redblackalt, scale='linear'):
    plt.plot(r, jacobi, 'b', label='jacobi')
    plt.plot(r, gauss, 'g', label='gauss')
    plt.plot(r, redblack, 'r', label='red-black')
    plt.plot(r, redblackalt, 'm', label='red-black-alternative')
    plt.ylabel('Mean execution time')
    plt.xlabel('Number of threads')
    plt.legend()
    if scale == 'log':
        plt.yscale('log')
        plt.savefig('py-scalability-log')
    else:
        plt.savefig('py-scalability')
    plt.close()

save_to_file(jacobi, gauss, redblack, redblackalt)
save_to_file(jacobi, gauss, redblack, redblackalt, scale='log')