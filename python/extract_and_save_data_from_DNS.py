"""
Extracts and does time averages of various quantities from OUT files generated by PADDIM (using OUTfile_reader.py), then
saves the results to a .txt file where each row is a different R0, and each column is a different averaged quantity.
"""
import numpy as np
import OUTfile_reader

Pr = 1e-1
# R0s = np.array([1.45, 3.0, 5.0, 7.0, 9.0])
R0s = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
HB = 0.0
Pm = 0.01

if HB > 0:
    names = ["R0", "urms", "TEMPrms", "CHEMrms", "Brms", "flux_Temp", "flux_Chem", "u_max(3)",
             "uxrms", "uyrms", "uzrms", "Bxrms", "Byrms", "Bzrms", "diss_Temp", "diss_Chem"]
else:
    names = ["R0", "urms", "TEMPrms", "CHEMrms", "flux_Temp", "flux_Chem", "u_max(3)",
             "uxrms", "uyrms", "uzrms", "diss_Temp", "diss_Chem"]

data_array = np.zeros((len(R0s), len(names)), dtype=np.float64)
variance_array = np.zeros_like(data_array)
for ri, r0 in enumerate(R0s):
    data_array[ri, 0] = r0
    variance_array[ri, 0] = r0
    for ni, name in enumerate(names[1:]):
        if HB > 0:
            out = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, Pm, name, True)
            data_array[ri, ni + 1] = out[0]
            variance_array[ri, ni + 1] = out[1]
        else:
            out = OUTfile_reader.get_avg_from_hydr_DNS(r0, name, True)
            data_array[ri, ni + 1] = out[0]
            variance_array[ri, ni + 1] = out[1]

if HB > 0:
    fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data.txt'.format(Pr, HB, Pm)
    variance_fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data_variance.txt'.format(Pr, HB, Pm)
else:
    fname = 'extracted_data/Pr{}_HB{}_R0scan_data.txt'.format(Pr, HB)
    variance_fname = 'extracted_data/Pr{}_HB{}_R0scan_data_variance.txt'.format(Pr, HB)

np.savetxt(fname, data_array, header=' '.join(names))
np.savetxt(variance_fname, variance_array, header=' '.join(names))

print('done')
