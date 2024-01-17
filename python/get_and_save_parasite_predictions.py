"""
Computes the predictions of parasite models via parasite_model.py, then saves the results to a .txt file where each row
is a different R0, and each column is a different quantity.
"""
import numpy as np
import fingering_modes
import parasite_model

Pr = 1e-1
tau = Pr
HB = 0.1
Pm = 0.01
DB = Pr / Pm
R0s = np.array([1.45, 1.5, 3.0, 5.0, 7.0, 9.0])
# rs = np.linspace(1e-5, 1, endpoint=False)
# R0s = rs*((1/tau) - 1) + 1

N = 21
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
C1 = 3.29
C2 = 0.8
delta = 0.0
use_eq32 = False
with_TC = True

# plans out which quantities will be saved and where
if use_eq32:
    names = ["R0", "FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star"]
    fname = 'parasite_predictions/Pr{}_HB{}_HG19eq32_parasite.txt'.format(Pr, HB)
else:
    if with_TC:
        names = ["R0", "FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star", "kmax-star"]
        fname = 'parasite_predictions/Pr{}_HB{}_Pm{}_N{}_withTC_parasite.txt'.format(Pr, HB, Pm, N)
        if C1 != 0.62 or C2 != 0.33:
            fname = 'parasite_predictions/Pr{}_HB{}_Pm{}_N{}_C1-{}_C2-{}_withTC_parasite.txt'.format(Pr, HB, Pm, N, C1, C2)
    else:
        names = ["R0", "FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star", "kmax-star"]
        fname = 'parasite_predictions/Pr{}_HB{}_Pm{}_N{}_parasite.txt'.format(Pr, HB, Pm, N)

# array that will eventually be saved to txt file
results_array = np.zeros((len(R0s), len(names)), dtype=np.float64)

# actually runs the parasite models
lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
# results_vs_r0(r0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=False, double_N=False, delta=0.0, ideal=False,
#                   badks_exception=True, withTC=False, Sam=False, C1=1.24, C2=1/1.66):
results_dict = parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=use_eq32, delta=delta, withTC=with_TC, Sam=with_TC, C1=C1, C2=C2)

# parses the output of results_vs_r0 (the parasite model) into results_array so it can be saved more easily
for ri, r0 in enumerate(R0s):
    results_array[ri, 0] = r0
for ni, name in enumerate(names[1:]):
    results_array[:, ni+1] = results_dict[name]
# saves result to txt file
np.savetxt(fname, results_array, header=' '.join(names))

print('done')
