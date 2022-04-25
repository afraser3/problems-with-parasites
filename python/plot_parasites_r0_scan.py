import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import kolmogorov_EVP
import glob
plt.style.use('style_file.mplstyle')

plot_quantity = "FC"
log_x = False
log_y = True
# choose from ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re", "M2"]
compare_eq32 = True
compare_hydro = True

N = 17  # need to check this often. 33 is plenty for Pr=0.01, but not for 0.001 w/ HB=10
Pr = 1e-1
tau = 1e-1
HB = 0.01
delta = 0.0  # from KH analysis -- probably leave at 0
Pm = 1.0  # magnetic Prandtl number
DB = Pr / Pm

ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))

# Set up the array of R0s or rs to solve for
R0s = np.linspace(1.45, 9.8, num=22, endpoint=True)

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)

# The following horrendous code is for going from a list of dicts to a dict of lists
names = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re", "M2", "kmax"]
results = {name: np.zeros_like(R0s) for name in names}
if compare_eq32:
    names_eq32 = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re", "M2"]
    results_eq32 = {name: np.zeros_like(R0s) for name in names_eq32}
if compare_hydro:
    names_hydro = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re", "M2"]
    results_hydro = {name: np.zeros_like(R0s) for name in names_hydro}

for ri, R0 in enumerate(R0s):
    print('solving for R0 = ', R0)
    result_ri = parasite_model.results_vs_R0(R0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], CH=1.66)
    for name in names:
        results[name][ri] = result_ri[name]
    if compare_eq32:
        result_ri = parasite_model.results_vs_R0(R0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=True)
        for name in names_eq32:
            results_eq32[name][ri] = result_ri[name]
    if compare_hydro:
        result_ri = parasite_model.results_vs_R0(R0, 0.0, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=True)
        for name in names_hydro:
            results_hydro[name][ri] = result_ri[name]
# end of horrendous dict-vs-lists code

scale = 0.8
plt.figure(figsize=(6.4 * scale, 4.8 * scale))
# plt.plot(R0s, NuCs_hydro, '-', c='k', label=r'$H_B = 0$ (hydro)')
plt.plot(R0s, results[plot_quantity], '-', c='C0', label=r'$H_B = {}$'.format(HB))
if compare_eq32:
    plt.plot(R0s, results_eq32[plot_quantity], '--', c='C0', label=r'Ideal MHD parasite model')
if compare_hydro:
    plt.plot(R0s, results_hydro[plot_quantity], '-', c='k', label='hydro')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
plt.ylabel(plot_quantity)
plt.legend()

plt.tight_layout()
plt.show()
