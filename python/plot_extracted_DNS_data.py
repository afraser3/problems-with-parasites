import numpy as np
from matplotlib import pyplot as plt
import OUTfile_reader
plt.style.use('style_file.mplstyle')

# column order is below for MHD runs, so, e.g., DNS_column = 1 extracts urms
# R0 urms TEMPrms CHEMrms Brms flux_Temp flux_Chem u_max(3) uxrms uyrms uzrms Bxrms Byrms Bzrms diss_Temp diss_Chem
# and here's the column order for hydro:
# R0 urms TEMPrms CHEMrms flux_Temp flux_Chem u_max(3) uxrms uyrms uzrms diss_Temp diss_Chem
# surely there's a cleaner way to do this
DNS_column = 6
compare_hydro = True
hydro_column = 5
log_x = False
log_y = True


Pr = 1e-1
tau = 1e-1
HB = 0.1
Pm = 1.0  # magnetic Prandtl number

params = [[0.1, 0.01, 1.0], [0.1, 0.1, 1.0]]

scale = 0.8
plt.figure(figsize=(6.4 * scale, 4.8 * scale))
if compare_hydro:
    fname_hydro = 'extracted_data/Pr{}_HB{}_R0scan_data.txt'.format(Pr, 0.0)
    R0s_hydro, results_DNS_hydro = np.loadtxt(fname_hydro, usecols=(0, hydro_column)).T
    plt.plot(R0s_hydro, results_DNS_hydro, '.', c='k')
for i, param in enumerate(params):
    pr = param[0]
    hb = param[1]
    pm = param[2]
    fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data.txt'.format(pr, hb, pm)
    R0s, results_DNS = np.loadtxt(fname, usecols=(0, DNS_column)).T
    plt.plot(R0s, results_DNS, 'x', c='C{}'.format(i))
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
#plt.legend()

plt.tight_layout()
plt.show()
# plt.savefig('test_CH1.1.pdf')
