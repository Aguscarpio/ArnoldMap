import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

mintau = 0.04
maxtau = 0.07
minK = 3
maxK = 10
dimtau = 128
dimK = 128

with open(f"periodsgrid_total.npy", "rb") as f:
    periods_grid_super = np.load(f)
#  with open(f"periodsgrid_total_flip.npy", "rb") as f:
    #  periods_grid_super_flip = np.load(f)
#  with open(f"periodsgrid_tau_total.npy", "rb") as f:
    #  periods_grid_tau_super = np.load(f)
#  with open(f"periodsgrid_tau_total_flip.npy", "rb") as f:
    #  periods_grid_tau_super_flip = np.load(f)

#  periods_grid_super_flip = np.flip(periods_grid_super_flip, 1)
#  periods_grid_tau_super_flip = np.flip(periods_grid_tau_super_flip, 0)

#  periods_grids_list = [periods_grid_super, periods_grid_super_flip, periods_grid_tau_super, periods_grid_tau_super_flip]

#  for i in range(1,17):
    #  if i not in [1,2,4,8,16]:
        #  #  periods_grid[periods_grid==i] = np.nan
        #  for periods_grid in periods_grids_list:
            #  periods_grid[periods_grid==i] = 2**np.ceil(np.log2(i))

def clean(arr):
    for i in range(dimK):
        for j in range(dimK):
            nb=arr[np.ix_(*((z-1, z, z+1-S) for z,S in zip((i,j), arr.shape)))].ravel()
            if np.count_nonzero(np.isnan(nb))>3:
                arr[i][j] = np.nan


taurange = np.linspace(mintau, maxtau, dimtau)*1000
Krange = np.linspace(minK, maxK, dimK)

kk, tt = np.meshgrid(Krange, taurange)

#  df_grid = pd.DataFrame(periods_grid,
            #  columns=[round(k, 3) for k in Krange],
            #  index=[round(t, 3) for t in taurange])
#  cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("tab20").colors[:16])
#  ax = sns.heatmap(df_grid, xticklabels=10, yticklabels=10, cmap=cmap)

colors = plt.cm.get_cmap('tab20',16)(np.arange(16))
cmap = matplotlib.colors.ListedColormap(colors, "")

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, sharex=True, sharey=True)
#  fig, ax1 = plt.subplots(1, 1)
cm1 = ax1.pcolormesh(kk, tt, periods_grid_super, cmap=cmap, vmin=0.5, vmax=16.5)
#  cm2 = ax2.pcolormesh(kk, tt, periods_grid_super_flip, cmap=cmap, vmin=0.5, vmax=16.5)
#  cm3 = ax3.pcolormesh(kk, tt, periods_grid_tau_super, cmap=cmap, vmin=0.5, vmax=16.5)
#  cm4 = ax4.pcolormesh(kk, tt, periods_grid_tau_super_flip, cmap=cmap, vmin=0.5, vmax=16.5)


for ax in [ax1, ax3]:
    #  ax.set_xlabel("K", size=24)
    ax.set_ylabel(r"$\tau$ (ms)", size=24)
for ax in [ax3, ax4]:
    ax.set_xlabel("K", size=24)
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
ax1.set_title("K ascendente", fontsize=24)
ax2.set_title("K descendente", fontsize=24)
ax3.set_title(r"$\tau$ ascendente", fontsize=24)
ax4.set_title(r"$\tau$ descendente", fontsize=24)
#  ax1.title.set_text("K ascendente")
#  ax2.title.set_text("K descendente")
#  ax3.title.set_text("Tau ascendente")
#  ax4.title.set_text("Tau descendente")
cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
cbar = fig.colorbar(cm1, ticks=range(1,17), cax=cbar_ax)
cbar.ax.tick_params(labelsize=24)
#  plt.gcf().clim(0.5, 16.5)
plt.show()

def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

eps = 0.1
n = 17
fig, ax = plt.subplots(figsize=(16,9))
from skimage import measure
for n in range(1,n):
    pgsectors = np.zeros_like(periods_grid_super)
    for periods_grid in periods_grids_list:
        pgcopy = np.copy(periods_grid)
        pgcopy[pgcopy!=n] = 0
        pgsectors += pgcopy
    pgsectors[pgsectors!=0] = n
    clusterized = measure.label(pgsectors>0)
    for region in measure.regionprops(clusterized):
        if region['area'] < 10:
            pgsectors[clusterized==region['label']] = 0
    ax.contourf(kk, tt, pgsectors, levels=[n-eps, n+eps], colors=[colors[n-1], "#000"], alpha=0.2)
    size = 30
    #  ax.set_xlabel("K", size=size)
    #  ax.set_ylabel(r"$\tau$)", size=size)
    ax.tick_params(axis='both', which='major', labelsize=size)
    ax.tick_params(axis='both', which='minor', labelsize=size)
    aaa = ax.contour(kk, tt, pgsectors, levels=[n-eps, n+eps], colors=[colors[n-1], "#000"], linewidths=[3,0], alpha=0.8)

#  for n in [17]:
    #  pgsectors = np.zeros_like(periods_grid_super)
    #  for periods_grid in periods_grids_list:
        #  pgcopy = np.copy(periods_grid)
        #  pgcopy[np.isnan(pgcopy)] = n
        #  pgcopy[pgcopy!=n] = 0
        #  pgsectors += pgcopy
    #  pgsectors[pgsectors!=0] = n
    #  ax.contourf(kk, tt, pgsectors, levels=[n-eps, n+eps], colors=["#999", "#000"], alpha=0.2)
    #  aaa = ax.contour(kk, tt, pgsectors, levels=[n-eps, n+eps], colors=["#999", "#000"], linewidths=[3,0], alpha=0.8)


#  for i in range(1):
    #  contour = aaa.collections[i]
    #  for path in contour.get_paths():
        #  if area(path.vertices)<1:
            #  path.remove()
            #  break

ax.set_xlabel(r"$K$ (a.u)", size=size)
ax.set_ylabel(r"$\tau$ (ms)", size=size)
#  plt.savefig("arnolds.pdf")
#  ax.set_xlabel("K")
#  ax.set_ylabel("Tau (ms)")
plt.show()





raise
ax.invert_yaxis()
colorbar = ax.collections[0].colorbar
n=16
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
colorbar.set_ticklabels([i for i in range(1,17)])
plt.show()
