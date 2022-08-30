import numpy as np
import h5py
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib.gridspec import GridSpec

plt.rcParams.update({"text.usetex": True})

fname = 'uz_snapshots_for_paper.h5'
# ['HB0.01_Pm1.0', 'HB0.0_Pm0.0', 'HB0.1_Pm0.01', 'HB0.1_Pm0.1', 'HB0.1_Pm1.0']
# ['R01.45', 'R03.0', 'R05.0', 'R07.0', 'R09.0']
# ['uz_z_x', 'x', 'z']
HB = 0.1
R0 = 7.0
Pms = [1.0, 0.1, 0.01, 0.0]
HBs = [0.1, 0.1, 0.1, 0.0]
uzs = []
zs = []
xs = []

umax = 0.0

with h5py.File(fname, 'r') as f:
    for i in range(len(Pms)):
        pm = Pms[i]
        hb = HBs[i]
        uz = np.array(f['HB{}_Pm{}/R0{}/uz_z_x'.format(hb, pm, R0)])
        z = np.array(f['HB{}_Pm{}/R0{}/z'.format(hb, pm, R0)])
        x = np.array(f['HB{}_Pm{}/R0{}/x'.format(hb, pm, R0)])
        uzs.append(uz)
        zs.append(z)
        xs.append(x)
        print(z[-1])
        if i>0:
            umax = max(umax, max(uz.max(), -uz.min()))

# uz1_max = max([uz1_max, -uz1_min])  # this was relevant for sharing one colorbar
scale = 1.0
# fig = plt.figure(figsize=(scale*4, scale*8), constrained_layout=True)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(scale*5, scale*7), constrained_layout=True, dpi=300)
for axi, ax in enumerate(axs.flat):
    ax.set_aspect(1)
for axi, ax in enumerate(axs.flat):
    #ax.set_aspect(1)
    uimax = max([uzs[axi].max(), -uzs[axi].min()])
    if axi == 0:
        pcm = ax.pcolormesh(xs[axi], zs[axi], uzs[axi], vmin=-uimax, vmax=uimax, cmap='RdBu', rasterized=True)  # , vmin=-uz1_max, vmax=uz1_max)
    else:
        pcm = ax.pcolormesh(xs[axi], zs[axi], uzs[axi], vmin=-umax, vmax=umax, cmap='RdBu',
                            rasterized=True)  # , vmin=-uz1_max, vmax=uz1_max)
    fig.colorbar(pcm, ax=ax, use_gridspec=False)#, aspect=5*2, shrink=1.0/2.0)
    if axi in [0, 2]:
        ax.set_ylabel(r'$z$')
    if axi in [2, 3]:
        ax.set_xlabel(r'$x$')
    if axi in [0, 1, 2]:
        ax.set_title(r'$\mathrm{{Pm}} = {}$'.format(Pms[axi]))
    else:
        ax.set_title(r'Hydro')

sublabels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$']
for axi, ax in enumerate(axs.flat):
    ax.text(-0.2, 1.07, sublabels[axi], horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)

# plt.show()
plt.savefig('figures/uz_snapshots_Pmscan-R07_fixed-cbar.pdf')
