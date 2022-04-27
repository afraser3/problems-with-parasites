"""
Compares FC vs wrms via equation 27 of HG19, each saved to a .txt file using extract_and_save_data_from_DNS.py
"""
import numpy as np
import fingering_modes
from matplotlib import pyplot as plt

Pr = 1e-1
tau = 1e-1
HB = 0.1
Pm = 0.1

fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data.txt'.format(Pr, HB, Pm)

data = np.loadtxt(fname)
R0s = data[:, 0]
FC = data[:, 6]
wrms = data[:, 10]

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])

Eq27_coeff = 1.24 / (R0s * (lamhats + tau*l2hats))

plt.plot(wrms*Eq27_coeff, FC, '.')
plt.plot([0.0, np.max(FC)], [0.0, np.max(FC)], '--', c='k')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show()
