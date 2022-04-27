"""
Computes the predictions of parasite models via parasite_model.py, then saves the results to a .txt file where each row
is a different R0, and each column is a different quantity.
"""
import numpy as np
import fingering_modes
import parasite_model

Pr = 1e-1
tau = Pr
HB = 0.01
Pm = 1.0
DB = Pr / Pm
R0s = np.array([1.45, 1.5, 3.0, 5.0, 7.0, 9.0])

N = 17
ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))
delta = 0.0
use_eq32 = False

# plans out which quantities will be saved and where
if use_eq32:
    names = ["R0", "FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star"]
    fname = 'parasite_predictions/Pr{}_HB{}_HG19eq32_parasite.txt'.format(Pr, HB)
else:
    names = ["R0", "FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star", "kmax-star"]
    fname = 'parasite_predictions/Pr{}_HB{}_Pm{}_N{}_parasite.txt'.format(Pr, HB, Pm, N)

# array that will eventually be saved to txt file
results_array = np.zeros((len(R0s), len(names)), dtype=np.float64)

# actually runs the parasite models
lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
results_dict = parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=use_eq32, delta=delta)

# parses the output of results_vs_r0 (the parasite model) into results_array so it can be saved more easily
for ri, r0 in enumerate(R0s):
    results_array[ri, 0] = r0
for ni, name in enumerate(names[1:]):
    results_array[:, ni+1] = results_dict[name]
# saves results to txt file
np.savetxt(fname, results_array, header=' '.join(names))

print('done')
