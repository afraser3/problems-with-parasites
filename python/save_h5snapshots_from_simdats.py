import numpy as np
from scipy.io import netcdf_file
import h5py

HBs = [0.1, 0.01, 0.0]
Pms = [1.0, 0.1, 0.01, 0.0]
R0s = [1.45, 1.5, 3.0, 5.0, 7.0, 9.0]
# (R0, HB, Pm)
# data_dir points to subdirectory of PADDIM/data/
simdat_files = {(1.45, 0.1, 1.0): 'peter_data/Pr0.1_t0.1_R1.45_DB0.1_CL0.1_Bz/simdat01.cdf',
                (3.0, 0.1, 1.0): 'Pr0.1_R3_HB0.1_Pm1.0/simdat1.cdf',
                (5.0, 0.1, 1.0): 'Pr0.1_R5_HB0.1_Pm1.0/simdat1.cdf',
                (7.0, 0.1, 1.0): 'Pr0.1_R7_HB0.1_Pm1.0/simdat1.cdf',
                (9.0, 0.1, 1.0): 'Pr0.1_R9_HB0.1_Pm1.0/boxsize_100_100_800/simdat5.cdf',
                (1.45, 0.01, 1.0): 'peter_data/Pr0.1_t0.1_R1.45_DB0.1_CL0.01_Bz/simdat03.cdf',
                (3.0, 0.01, 1.0): 'Pr0.1_R3_HB0.01_Pm1.0/simdat1.cdf',
                (5.0, 0.01, 1.0): 'Pr0.1_R5_HB0.01_Pm1.0/simdat1.cdf',
                (7.0, 0.01, 1.0): 'Pr0.1_R7_HB0.01_Pm1.0/redo/simdat1.cdf',
                (9.0, 0.01, 1.0): 'Pr0.1_R9_HB0.01_Pm1.0/boxsize_100_100_800/simdat3.cdf',
                (1.5, 0.1, 0.1): 'Pr0.1_R1.5_HB0.1_Pm0.1/boxsize_100_100_100/simdat1.cdf',
                (3.0, 0.1, 0.1): 'Pr0.1_R3_HB0.1_Pm0.1/boxsize_100_100_100/simdat1.cdf',
                (5.0, 0.1, 0.1): 'Pr0.1_R5_HB0.1_Pm0.1/boxsize_100_100_200/simdat1.cdf',
                (7.0, 0.1, 0.1): 'Pr0.1_R7_HB0.1_Pm0.1/boxsize_100_100_200/simdat1.cdf',  # my model/DNS comparisons have been using the Lz=100 one ...
                (9.0, 0.1, 0.1): 'Pr0.1_R9_HB0.1_Pm0.1/boxsize_100_100_800/simdat12.cdf',
                (1.5, 0.01, 0.1): 'Pr0.1_R1.5_HB0.01_Pm0.1/boxsize_100_100_100/simdat1.cdf',
                (3.0, 0.01, 0.1): 'Pr0.1_R3_HB0.01_Pm0.1/boxsize_100_100_100/simdat1.cdf',
                (5.0, 0.01, 0.1): 'Pr0.1_R5_HB0.01_Pm0.1/boxsize_100_100_200/simdat1.cdf',
                (7.0, 0.01, 0.1): 'Pr0.1_R7_HB0.01_Pm0.1/boxsize_100_100_200/simdat1.cdf',
                (9.0, 0.01, 0.1): 'Pr0.1_R9_HB0.01_Pm0.1/boxsize_100_100_400/simdat4.cdf',
                (1.5, 0.1, 0.01): 'Pr0.1_R1.5_HB0.1_Pm0.01/simdat1.cdf',
                (3.0, 0.1, 0.01): 'Pr0.1_R3_HB0.1_Pm0.01/simdat1.cdf',
                (5.0, 0.1, 0.01): 'Pr0.1_R5_HB0.1_Pm0.01/simdat1.cdf',
                (7.0, 0.1, 0.01): 'Pr0.1_R7_HB0.1_Pm0.01/simdat1.cdf',
                (9.0, 0.1, 0.01): 'Pr0.1_R9_HB0.1_Pm0.01/boxsize_100_100_800/simdat4.cdf',
                (1.45, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R1.45/simdat2.cdf',
                (1.5, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R1.5/simdat1.cdf',
                (3, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R3/simdat1.cdf',
                (5, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R5/simdat1.cdf',
                (7, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R7/simdat1.cdf',
                (9, 0.0, 0.0): 'PADDI/Pr0.1_tau0.1/R9/simdat2.cdf'
                }

with h5py.File('uz_snapshots_for_paper.h5', mode='w') as h5file:
    for hb in HBs:
        for pm in Pms:
            print(hb, pm)
            if (9, hb, pm) in simdat_files:
                hbpm_group = h5file.create_group('HB{}_Pm{}'.format(hb, pm))
                for r0 in R0s:
                    if (r0, hb, pm) in simdat_files:
                        r0_group = hbpm_group.create_group("R0{}".format(r0))
                        fname = '../../PADDIM/data/'+simdat_files[(r0, hb, pm)]
                        with netcdf_file(fname, 'r') as f:
                            uz = f.variables['uz'][-1, :, 0, :].copy()  # picks out the last timestep, and the 0th slice in y
                            xs = f.variables['x'][:].copy()
                            zs = f.variables['z'][:].copy()
                        uz_dset = r0_group.create_dataset('uz_z_x', data=uz)  # name is to emphasize the first axis is z, second is x
                        xs_dset = r0_group.create_dataset('x', data=xs)
                        zs_dset = r0_group.create_dataset('z', data=zs)
