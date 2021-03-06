import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

gpu_data = pd.read_csv("phi_torch_gpu.csv")
cpu_data = pd.read_csv("phi_torch_cpu.csv")

resolutions = gpu_data.resolution.unique()

fig = plt.figure()
ax_rho = fig.add_subplot(1, 3, 1)
ax_vx = fig.add_subplot(1, 3, 2)
ax_vy = fig.add_subplot(1, 3, 3)

colors = itertools.cycle(('red', 'blue', 'orange', 'green', 'purple')) 

for res in resolutions:
  rho_diff = np.square(gpu_data[gpu_data.resolution == res].rho_means - cpu_data[cpu_data.resolution == res].rho_means)
  vx_diff = np.square(gpu_data[gpu_data.resolution == res].vx_means - cpu_data[cpu_data.resolution == res].vx_means)
  vy_diff = np.square(gpu_data[gpu_data.resolution == res].vy_means - cpu_data[cpu_data.resolution == res].vy_means)

  ax_rho.plot(np.linspace(1,len(rho_diff), len(rho_diff)), rho_diff)
  ax_vx.plot(np.linspace(1,len(rho_diff), len(rho_diff)), vx_diff)
  ax_vy.plot(np.linspace(1,len(rho_diff), len(rho_diff)), vy_diff)

#ax_rho.legend(resolutions)
ax_vx.legend(resolutions)
#ax_vy.legend(resolutions)
fig.savefig("diff.png")