"""
Compares FC vs wrms via equation 27 of HG19, each saved to a .txt file using extract_and_save_data_from_DNS.py
"""
import numpy as np
import fingering_modes
from matplotlib import pyplot as plt

Pr = 1e-1
tau = 1e-1
symbols = ['x', '+']
colors = ['C0', 'C1']

for hbi, HB in enumerate([0.01, 0.1]):
    for pmi, Pm in enumerate([0.1, 1.0]):
        fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data.txt'.format(Pr, HB, Pm)

        data = np.loadtxt(fname)
        R0s = data[:, 0]
        FC = data[:, 6]
        wrms = data[:, 10]

        lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])

        Eq27_coeff = 1.24 / (R0s * (lamhats + tau*l2hats))

        plt.loglog(Eq27_coeff * wrms**2.0, FC, symbols[pmi], c=colors[hbi], label=r'$HB = {}, Pm = {}$'.format(HB, Pm))
plt.loglog([1e-2, 10], [1e-2, 10], '--', c='k')
# plt.xlim(xmin=0)
# plt.ylim(ymin=0)
plt.legend()
plt.xlabel(r'RHS of Eq 27 in HG19')
plt.ylabel(r'$\langle u_z C \rangle$')
# plt.show()
plt.savefig('HG19_Eq27_verification.pdf')
