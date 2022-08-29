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
R0s2 = [1.45, 3, 5, 7, 9]
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
# fig = plt.figure(figsize=(scale*4, scale*8), constrained_layout=True)
fig = plt.figure(figsize=(scale*8, scale*8), constrained_layout=True, dpi=300)
gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs00 = gs0[1].subgridspec(1, 2)  # switching these two so that HB=0.01 is on the left, HB=0.1 is on the right
gs01 = gs0[0].subgridspec(1, 2)  # in hindsight I should have just switched the values of HB1 and HB2 at the very top...

gs000 = gs00[0].subgridspec(6, 1)
gs010 = gs01[0].subgridspec(6, 1)
ax1 = fig.add_subplot(gs000[0, :])
ax2 = fig.add_subplot(gs000[1, :])
ax3 = fig.add_subplot(gs000[2:4, :])
ax4 = fig.add_subplot(gs000[4:6, :])
ax5 = fig.add_subplot(gs00[1])
aspects = [1, 1, 2, 2, 8]
for axi, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    # ax.set_aspect(1)
    ax.set_box_aspect(aspects[axi])
for axi, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    umax = max([uzs1[axi].max(), -uzs1[axi].min()])
    pcm = ax.pcolormesh(xs1[axi], zs1[axi], uzs1[axi], vmin=-umax, vmax=umax, cmap='RdBu', rasterized=True)  # , vmin=-uz1_max, vmax=uz1_max)
    if axi == 4:
        fig.colorbar(pcm, ax=ax, use_gridspec=True, aspect=32)#, aspect=2.5*aspects[axi])#, shrink=1.0/aspects[axi])
    else:
        fig.colorbar(pcm, ax=ax, use_gridspec=True)  # , aspect=2.5*aspects[axi])#, shrink=1.0/aspects[axi])
    if axi in [0, 4]:
        ax.set_title(r'$R_0 = {}$, $H_B = {}$'.format(R0s2[axi], HB1))
    else:
        ax.set_title(r'$R_0 = {}$'.format(R0s2[axi]))
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylabel(r'$z$')
ax4.set_xlabel(r'$x$')
ax5.set_xlabel(r'$x$')

ax6 = fig.add_subplot(gs010[0, :])
ax7 = fig.add_subplot(gs010[1, :])
ax8 = fig.add_subplot(gs010[2:4, :])
ax9 = fig.add_subplot(gs010[4:6, :])
ax10 = fig.add_subplot(gs01[1])
aspects = [1, 1, 2, 2, 8]
for axi, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    # ax.set_aspect(1)
    ax.set_box_aspect(aspects[axi])
for axi, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    umax = max([uzs2[axi].max(), -uzs2[axi].min()])
    pcm = ax.pcolormesh(xs2[axi], zs2[axi], uzs2[axi], vmin=-umax, vmax=umax, cmap='RdBu', rasterized=True)  # , vmin=-uz1_max, vmax=uz1_max)
    if axi == 4:
        fig.colorbar(pcm, ax=ax, use_gridspec=True, aspect=32)#, aspect=2.5*aspects[axi])#, shrink=1.0/aspects[axi])
    else:
        fig.colorbar(pcm, ax=ax, use_gridspec=True)  # , aspect=2.5*aspects[axi])#, shrink=1.0/aspects[axi])
    if axi in [0, 4]:
        ax.set_title(r'$R_0 = {}$, $H_B = {}$'.format(R0s2[axi], HB2))
    else:
        ax.set_title(r'$R_0 = {}$'.format(R0s2[axi]))
for ax in [ax6, ax7, ax8, ax9]:
    ax.set_ylabel(r'$z$')
ax9.set_xlabel(r'$x$')
ax10.set_xlabel(r'$x$')

sublabels = [r'$\mathrm{(f)}$', r'$\mathrm{(g)}$', r'$\mathrm{(h)}$', r'$\mathrm{(i)}$', r'$\mathrm{(j)}$',
             r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$', r'$\mathrm{(e)}$']
for axi, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]):
    if axi in [4, 9]:
        ax.text(-0.23, 1.01, sublabels[axi], horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    else:
        ax.text(-0.23, 1.07, sublabels[axi], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

# ax_left = fig.add_subplot(gs00[:])
# ax_left.axis('off')
# ax_left.set_title('test')

# plt.show()
plt.savefig('figures/uz_snapshots_Pm1.pdf')
