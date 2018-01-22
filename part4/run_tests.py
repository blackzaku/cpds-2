import os
import json
import subprocess
import re

TIMES = 10
r = list(range(1, 20))
results = [ ["r", "cpu", "./heatCUDA", []], ["g", "gpu-diff", "./heatCUDA-gpu-diff", []],
            ["b", "gpu-reduce", "./heatCUDA-gpu-reduce", []],["c", "gpu-reduce-512", "./heatCUDA-gpu-reduce-512", []],
            ["y","gpu-atomic", "./heatCUDA-gpu-atomic", []]]
cpu_time = []

def get_mean_time(executable, times, n):
    avg = 0
    for _ in range(times):
        avg += get_time(executable, n) / times
    return avg

def get_time(executable, n):
    print(executable)
    output = str(subprocess.Popen([executable, "test.dat", "-t", "{n}".format(n=n)],stdout=subprocess.PIPE).communicate()[0])
    matched_cpu = re.search(r"Time on CPU in ms\.\s*=\s*(\d+\.\d+)", output)
    cpu_time.append(float(matched_cpu.group(1)))
    matched = re.search(r"Time on GPU in ms\.\s*=\s*(\d+\.\d+)", output)
    return float(matched.group(1))

for i in r:
    for res in results:
        res[3].append(get_mean_time(res[2], TIMES, i))

print(json.dumps(results, indent=4, separators=(',', ': ')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for res in results:
    plt.plot(r, res[3], res[0], label=res[1])
plt.axhline(y=sum(cpu_time) / len(cpu_time))
plt.ylabel('Mean execution time')
plt.xlabel('Threads/dimension')
plt.legend()
plt.savefig('scalability')