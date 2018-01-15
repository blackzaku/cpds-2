import os
import subprocess
import re

TIMES = 5
SOLVER = 'jacobi'
jacobi = []
gauss = []
redblack = []
r = list(range(1, 13))

def get_mean_time(solver, times, n):
    avg = 0
    for _ in range(times):
        avg += get_time(solver, n) / times
    return avg

def get_time(solver, n):
    output = subprocess.Popen(["mpiexec", "-n {n} heatmpi test_{solver}.dat".format(n=n, solver=solver)],
                              stdout=subprocess.PIPE).communicate()[0]
    matched = re.search("Time:\s+(\d+\.\d+)", str(output))
    return float(matched.group(1))

for i in r:
    jacobi.append(get_mean_time('jacobi', TIMES, i))
    gauss.append(get_mean_time('gauss', TIMES, i))
    redblack.append(get_mean_time('redblack', TIMES, i))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(r, jacobi, 'b', label='jacobi')
plt.plot(r, gauss, 'g', r, redblack, label='gauss')
plt.plot(r, redblack, 'r', label='red-black')
plt.ylabel('Mean execution time')
plt.xlabel('Number of threads')
plt.legend()
plt.savefig('scalability')