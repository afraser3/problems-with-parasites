"""
My attempt at testing the speed of different ways to calculate w_f.

It's set up to calculate w_f vs R_0 for parameters relevant to RGB stars. Note this is equivalent to Fig 7a of FRG24
(or fig 8a of the arXiv v2 preprint) since D_C/kappa_C is related to w_f by a simple formula.
"""
import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import time
plt.style.use('style_file.mplstyle')


N = 17
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
# ks = np.append(np.geomspace(1e-8, 0.1, num=100, endpoint=False), np.linspace(0.1, 4.0, num=100))

# RGB parameters
tau = 1e-7
Pr = 1e-6
Pm = 0.1
DB = Pr / Pm
HB = 1e-7  # [1e-7, 1e-5]

delta = 0.0  # from KH analysis -- leave at 0, corresponds to finding parasites with x-periodicity matching lambda_f
C1 = 0.62  # for with_TC model
C2 = 0.33  # for with_TC model
kb = 1.24  # value of C1 to use whenever using eq32 (i.e. the HG19 model and/or the Brown model)

# Set up the array of rs (reduced density ratios) to solve for, and thus the R0s
rs = np.linspace(1/49, 1, num=10, endpoint=False)
# rs = np.linspace(1/49, 1, num=20, endpoint=False)
# r = (R0 - 1.0) / ((1.0 / tau) - 1)
R0s = rs*((1/tau) - 1) + 1

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)

start_time_oldway = time.time()
# w_f_withTC(Pr, tau, R0, HB, DB, ks, N, badks_exception=True, get_kmax=False, C2=0.33, lamhat=0.0, l2hat=0.0, Sam=False, test_Sam_no_TC=False, full_solver_object=False, wbounds=None, sparse=False)
results_oldway = [parasite_model.w_f_withTC(Pr, tau, R0s[ri], HB, DB, ks, N, badks_exception=True, get_kmax=False, C2=C2, lamhat=lamhats[ri], l2hat=l2hats[ri], Sam=True, full_solver_object=True, sparse=False) for ri in range(len(R0s))]
# results_oldway = results_vs_r0(R0s1, HB, Pr, tau, DB, ks, N, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2)
end_time_oldway = time.time()
time_oldway = end_time_oldway - start_time_oldway

print('test')

start_time_newway = time.time()
results_newway = [parasite_model.w_f_withTC(Pr, tau, R0s[ri], HB, DB, ks, N, badks_exception=True, get_kmax=False, C2=C2, lamhat=lamhats[ri], l2hat=l2hats[ri], Sam=True, full_solver_object=True, sparse=True) for ri in range(len(R0s))]
# results_newway = results_vs_r0(R0s1, HB, Pr, tau, DB, ks, N, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2, sparse=True)
end_time_newway = time.time()
time_newway = end_time_newway - start_time_newway

wfs_old = np.array([result.root for result in results_oldway])
wfs_new = np.array([result.root for result in results_newway])
wf_diff = np.abs(wfs_old - wfs_new)/wfs_old
print("Relative speedup between the two methods (time1/time2): ", time_oldway/time_newway)
print("Array of relative errors (out of 1, i.e., (w1 - w2)/w1): ", wf_diff)
print("Max error: ", np.max(wf_diff))
