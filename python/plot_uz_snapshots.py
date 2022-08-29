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
HB1 = 0.1
HB2 = 0.01
Pm = 1.0
R0s = np.array([1.45, 3.0, 5.0, 7.0, 9.0])
uzs1 = []
zs1 = []
xs1 = []
uzs2 = []
zs2 = []
xs2 = []

uz1_min = 0.0
uz1_max = 0.0
uz2_min = 0.0
uz2_max = 0.0

with h5py.File(fname, 'r') as f:
    for r0 in R0s:
        uz = np.array(f['HB{}_Pm{}/R0{}/uz_z_x'.format(HB1, Pm, r0)])
        z = np.array(f['HB{}_Pm{}/R0{}/z'.format(HB1, Pm, r0)])
        x = np.array(f['HB{}_Pm{}/R0{}/x'.format(HB1, Pm, r0)])
        uzs1.append(uz)
        zs1.append(z)
        xs1.append(x)
        uz1_max = max([uz1_max, uz.max()])
        uz1_min = min([uz1_min, uz.min()])

        uz = np.array(f['HB{}_Pm{}/R0{}/uz_z_x'.format(HB2, Pm, r0)])
        z = np.array(f['HB{}_Pm{}/R0{}/z'.format(HB2, Pm, r0)])
        x = np.array(f['HB{}_Pm{}/R0{}/x'.format(HB2, Pm, r0)])
        uzs2.append(uz)
        zs2.append(z)
        xs2.append(x)
        uz2_max = max([uz2_max, uz.max()])
        uz2_min = min([uz2_min, uz.min()])

# uz1_max = max([uz1_max, -uz1_min])  # this was relevant for sharing one colorbar
scale = 1.2
fig = plt.figure(figsize=(scale*4, scale*8), constrained_layout=True)
gs0 = gridspec.GridSpec(1, 2, figure=fig)
gs00 = gs0[0].subgridspec(6, 1)  # gridspec.GridSpec(6, 1)
ax1 = fig.add_subplot(gs00[0, :])
ax2 = fig.add_subplot(gs00[1, :])
ax3 = fig.add_subplot(gs00[2:4, :])
ax4 = fig.add_subplot(gs00[4:6, :])
ax5 = fig.add_subplot(gs0[1])
aspects = [1, 1, 2, 2, 8]
for axi, ax in enumerate(fig.axes):
    # ax.set_aspect(1)
    ax.set_box_aspect(aspects[axi])
for axi, ax in enumerate(fig.axes):
    umax = max([uzs1[axi].max(), -uzs1[axi].min()])
    pcm = ax.pcolormesh(xs1[axi], zs1[axi], uzs1[axi], vmin=-umax, vmax=umax, cmap='RdBu')  # , vmin=-uz1_max, vmax=uz1_max)
    fig.colorbar(pcm, ax=ax, use_gridspec=True)#, aspect=2.5*aspects[axi])#, shrink=1.0/aspects[axi])
    ax.set_title(r'$R_0 = {}$'.format(R0s[axi]))
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylabel(r'$z$')
ax4.set_xlabel(r'$x$')
ax5.set_xlabel(r'$x$')

plt.show()
