import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import kolmogorov_EVP
import glob
plt.style.use('style_file.mplstyle')


compare_eq32 = True
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
ws_hydro = 2.0 * np.pi * lamhats / lhats
NuCs_hydro = np.array([parasite_model.NuC(tau, ws_hydro[i], lamhats[i], l2hats[i]) for i in range(len(R0s))])
# various quantities to solve for and plot
names = ["NuC", "NuT", "gammatot", "wf", "Re", "M2", "kmax"]
results = {name: np.zeros_like(R0s) for name in names}
results['FC'] = np.zeros_like(R0s)
results['FT'] = np.zeros_like(R0s)
if compare_eq32:
    names_eq32 = ["NuC", "NuT", "gammatot", "wf", "Re", "M2"]
    results_eq32 = {name: np.zeros_like(R0s) for name in names_eq32}
    results_eq32['FC'] = np.zeros_like(R0s)
    results_eq32['FT'] = np.zeros_like(R0s)

for ri, R0 in enumerate(R0s):
    print('solving for R0 = ', R0)
    result_ri = parasite_model.results_vs_R0(R0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], CH=1.66)
    for i, name in enumerate(names):
        results[name][ri] = result_ri[i]
    results['FC'][ri] = results['wf'][ri]**2.0/(R0*(lamhats[ri] + tau * l2hats[ri]))
    results['FT'][ri] = results['wf'][ri]**2.0/(lamhats[ri] + l2hats[ri])

    results['NuC'][ri] = parasite_model.NuC(tau, results['wf'][ri], lamhats[ri], l2hats[ri])
    if compare_eq32:
        result_ri = parasite_model.results_vs_R0(R0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=True)
        for i, name in enumerate(names_eq32):
            results_eq32[name][ri] = result_ri[i]
        results_eq32['FC'][ri] = results_eq32['wf'][ri]**2.0/(R0*(lamhats[ri] + tau * l2hats[ri]))
        results_eq32['FT'][ri] = results_eq32['wf'][ri]**2.0/(lamhats[ri] + l2hats[ri])

scale = 0.8
plt.figure(figsize=(6.4 * scale, 4.8 * scale))
plt.semilogy(R0s, NuCs_hydro, '-', c='k', label=r'$H_B = 0$ (hydro)')
plt.semilogy(R0s, results['NuC'], '-', c='C0', label=r'$H_B = {}$'.format(HB))
if compare_eq32:
    plt.semilogy(R0s, results_eq32['NuC'], '--', c='C0', label=r'Ideal MHD parasite model')
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\mathrm{Nu}_C$')
plt.legend()

plt.tight_layout()
plt.show()
