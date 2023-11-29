import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
plt.style.use('style_file.mplstyle')

Prs = np.geomspace(1e-7, 1e-5, num=55)  # num=55 just to make sure I didn't make dumb mistakes with array shapes
taus = 0.1*Prs
# taus = np.ones_like(Prs) * 1e-7
Pms = np.geomspace(1e-2, 1e0, num=50)
R0max = np.zeros((len(Prs), len(Pms)), np.float64)
rmax = np.zeros_like(R0max)
# R0max2 = np.zeros_like(R0max)

for pri, pr in enumerate(Prs):
    for pmi, pm in enumerate(Pms):
        R0max[pri, pmi] = min(taus[pri]**-1, 4*np.pi**2*pm**2/pr)
        rmax[pri, pmi] = (R0max[pri, pmi] - 1) / (taus[pri]**-1 - 1)
        # R0max[pri, pmi] = taus[pri]**-1
        # R0max[pri, pmi] = 4 * np.pi ** 2 * pm ** 2 / pr
        # R0max2[pri, pmi] = 4 * np.pi ** 2 * pm ** 2 / pr

plt.contourf(Pms, Prs, R0max, levels=np.geomspace(1e2,1e9,15), norm=colors.LogNorm(), cmap='viridis')
# plt.contourf(Pms, Prs, rmax, norm=colors.LogNorm(), cmap='viridis')

# plt.pcolormesh(Pms, Prs, R0max2 - R0max, norm=colors.LogNorm())
plt.grid(visible=False)
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label=r'$R_{0,\mathrm{max}}$')
plt.xlabel(r'$\mathrm{Pm}$')
plt.ylabel(r'$\mathrm{Pr}$')
# plt.show()
plt.savefig('figures/Fig5.pdf')
