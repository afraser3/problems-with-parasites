"""
For reading OUTfiles from PADDIM
"""

import numpy as np
import glob


def get_vars(file_names, vars, flat=False):
    """
    Reads OUT files, returns data.
    """

    # 1 2 3 4 5 6 7 8 9 10
    # istep, t, dt, urms, VORTrms, TEMPrms, CHEMrms, Brms, flux_Temp, flux_Chem, &
    # 11 12 13 14
    # Temp_min, Temp_max, Chem_min, Chem_max, &
    # 15 16 17 18 19 20
    # u_min(1), u_max(1), u_min(2), u_max(2), u_min(3), u_max(3), &
    # 21 22 23 24 25 26
    # VORT_min(1), VORT_max(1), VORT_min(2), VORT_max(2), VORT_min(3), VORT_max(3), &
    # 27 28 29 30 31 32
    # B_min(1), B_max(1), B_min(2), B_max(2), B_min(3), B_max(3), &
    # 33 34 35
    # u_max_abs, VORT_max_abs, B_max_abs, &
    # 36 37 38 39 40 41 42 43 44
    # uxrms, uyrms, uzrms, VORTXrms, VORTYrms, VORTZrms, Bxrms, Byrms, Bzrms, &
    # 45 46
    # diss_Temp, diss_Chem
    var_names = ["istep", "t", "dt", "urms", "VORTrms", "TEMPrms", "CHEMrms", "Brms", "flux_Temp", "flux_Chem",
                 "Temp_min", "Temp_max", "Chem_min", "Chem_max",
                 "u_min(1)", "u_max(1)", "u_min(2)", "u_max(2)", "u_min(3)", "u_max(3)",
                 "VORT_min(1)", "VORT_max(1)", "VORT_min(2)", "VORT_max(2)", "VORT_min(3)", "VORT_max(3)",
                 "B_min(1)", "B_max(1)", "B_min(2)", "B_max(2)", "B_min(3)", "B_max(3)",
                 "u_max_abs", "VORT_max_abs", "B_max_abs",
                 "uxrms", "uyrms", "uzrms", "VORTXrms", "VORTYrms", "VORTZrms", "Bxrms", "Byrms", "Bzrms",
                 "diss_Temp", "diss_Chem"]
    var_inds = dict(zip(var_names, range(len(var_names))))
    # if you do var_inds['t'], you should get back 1

    for fi, fname in enumerate(file_names):
        if fi == 0:
            if flat:
                vars_out = np.loadtxt(fname, usecols=[var_inds[name] for name in vars])
            else:
                vars_out = [np.loadtxt(fname, usecols=[var_inds[name] for name in
                                                       vars])]  # data to return, len(file_names) list of arrays
        else:
            if flat:
                vars_out = np.append(vars_out, np.loadtxt(fname, usecols=[var_inds[name] for name in vars]), axis=0)
            else:
                vars_out.append(np.loadtxt(fname, usecols=[var_inds[name] for name in vars]))
    return vars_out


def get_vars_hydro(file_names, vars, flat=False):
    """
    Reads OUT files, returns data.
    """
    var_names = ["istep", "t", "dt", "urms", "VORTrms", "TEMPrms", "CHEMrms", "flux_Temp", "flux_Chem",
                 "Temp_min", "Temp_max", "Chem_min", "Chem_max",
                 "u_min(1)", "u_max(1)", "u_min(2)", "u_max(2)", "u_min(3)", "u_max(3)",
                 "VORT_min(1)", "VORT_max(1)", "VORT_min(2)", "VORT_max(2)", "VORT_min(3)", "VORT_max(3)",
                 "u_max_abs", "VORT_max_abs",
                 "uxrms", "uyrms", "uzrms", "VORTXrms", "VORTYrms", "VORTZrms",
                 "diss_Temp", "diss_Chem"]
    var_inds = dict(zip(var_names, range(len(var_names))))
    # if you do var_inds['t'], you should get back 1

    for fi, fname in enumerate(file_names):
        if fi == 0:
            if flat:
                vars_out = np.loadtxt(fname, usecols=[var_inds[name] for name in vars])
            else:
                vars_out = [np.loadtxt(fname, usecols=[var_inds[name] for name in
                                                       vars])]  # data to return, len(file_names) list of arrays
        else:
            if flat:
                vars_out = np.append(vars_out, np.loadtxt(fname, usecols=[var_inds[name] for name in vars]), axis=0)
            else:
                vars_out.append(np.loadtxt(fname, usecols=[var_inds[name] for name in vars]))
    return vars_out


# For my PADDIM runs, the following dict notes for each (R0, HB, Pm) if a particular subdirectory is needed.
# If no entry in this dict, assume the default directory is fine.
subdirs = {(1.5, 0.01, 0.1): 'boxsize_100_100_100/',
           (1.5, 0.1, 0.1): 'boxsize_100_100_100/',
           (3.0, 0.01, 0.1): 'boxsize_100_100_100/',
           (3.0, 0.1, 0.1): 'boxsize_100_100_100/',
           (7.0, 0.01, 0.1): 'boxsize_100_100_100/',
           (7.0, 0.1, 0.1): 'boxsize_100_100_100/',
           (9.0, 0.01, 0.1): 'boxsize_100_100_400/',
           (9.0, 0.1, 0.1): 'boxsize_100_100_800/',
           (9.0, 0.01, 1.0): 'boxsize_100_100_800/',
           (9.0, 0.1, 1.0): 'boxsize_100_100_800/',
           (1.45, 0.01, 1.0): 'OUT-files/',
           (1.45, 0.1, 1.0): 'OUT-files/',
           (9.0, 0.1, 0.01): 'boxsize_100_100_800/'}
# For my PADDIM runs, the following dict notes for each (R0, HB, Pm) which entry in the OUT files
# (concatenated together) is more or less safe to begin time-averaging over.
# At the end, I've included some (Pr, R0, HB, Pm) entries.
avg_starts = {(1.45, 0.01, 1.0): 1000,
              (1.45, 0.1, 1.0): 100,
              (1.5, 0.01, 0.1): 500,
              (1.5, 0.1, 0.1): 500,
              (3.0, 0.01, 0.1): 300,
              (3.0, 0.1, 0.1): 300,
              (3.0, 0.01, 1.0): 300,
              (3.0, 0.1, 1.0): 400,
              (5.0, 0.01, 0.1): 400,
              (5.0, 0.1, 0.1): 400,
              (5.0, 0.01, 1.0): 400,
              (5.0, 0.1, 1.0): 400,
              (7.0, 0.01, 0.1): 500,
              (7.0, 0.1, 0.1): 1000,
              (7.0, 0.01, 1.0): 500,
              (7.0, 0.1, 1.0): 750,
              (9.0, 0.01, 0.1): 2000,
              (9.0, 0.1, 0.1): 2500,
              (9.0, 0.01, 1.0): 2000,
              (9.0, 0.1, 1.0): 2250,
              (1.5, 0.1, 0.01): 750,
              (3.0, 0.1, 0.01): 1000,
              (5.0, 0.1, 0.01): 1500,
              (7.0, 0.1, 0.01): 1500,
              (9.0, 0.1, 0.01): 7500,
              (0.04, 5.0, 0.01, 0.2): 1500,
              (0.04, 5.0, 0.1, 0.2): 1500,
              (0.04, 2.5, 0.01, 0.2): 750,
              (0.04, 2.5, 0.1, 0.2): 750,
              (0.04, 7.5, 0.01, 0.2): 1000,
              (0.04, 7.5, 0.1, 0.2): 1000}

# For my Pr = tau = 0.1 hydro runs, the following notes which entry in the OUT file is sufficiently
# within the saturated state to safely time-average
R0s_hydro = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
R0strings_hydro = ['1.5', '3', '5', '7', '9']
avg_starts_hydro_DNS = [300, 250, 150, 150, 750]
avg_starts_hydro_DNS_dict = dict(zip(R0strings_hydro, avg_starts_hydro_DNS))


def get_avg_from_DNS(pr, r0, hb, pm, var, with_variance=False):
    if int(r0) == r0:
        r0string = str(int(r0))
    else:
        r0string = str(r0)
    base_dir = '/Users/adfraser/PADDIM/Pr{}_R'.format(pr) + r0string + '_HB{}_Pm{}/'.format(hb, pm)
    if (r0, hb, pm) in subdirs.keys():
        outdir = base_dir + subdirs[(r0, hb, pm)]
    else:
        outdir = base_dir
    last_out = max([int(OUTpath.split('OUT')[-1]) for OUTpath in glob.glob(outdir + 'OUT*')])  # count OUTfiles
    names = [outdir + 'OUT{}'.format(i) for i in range(1, last_out + 1)]  # create list of OUTfiles
    vars_in = ['t', var]
    if pr == 0.1:
        avg_start = avg_starts[(r0, hb, pm)]
    else:
        avg_start = avg_starts[(pr, r0, hb, pm)]
    out_full = get_vars(names, vars_in, flat=True)[avg_start:]
    if with_variance:
        out_avg = np.mean(out_full[:, 1])
        out_variance = np.sqrt(np.var(out_full[:, 1]))
        return out_avg, out_variance
    else:
        out_avg = np.trapz(out_full[:, 1], x=out_full[:, 0])/(out_full[-1, 0] - out_full[0, 0])  # a more accurate mean
        return out_avg


def get_avg_from_hydr_DNS(r0, var, with_variance=False):
    if int(r0) == r0:
        r0string = str(int(r0))
    else:
        r0string = str(r0)
    base_dir = '/Users/adfraser/PADDIM/Pr{}_R'.format(0.1) + r0string + '_HB0/'
    outdir = base_dir
    last_out = max([int(OUTpath.split('OUT')[-1]) for OUTpath in glob.glob(outdir + 'OUT*')])  # count OUTfiles
    names = [outdir + 'OUT{}'.format(i) for i in range(1, last_out + 1)]  # create list of OUTfiles
    vars_in = ['t', var]
    avg_start = avg_starts_hydro_DNS_dict[r0string]
    out_full = get_vars_hydro(names, vars_in, flat=True)[avg_start:]
    if with_variance:
        out_avg = np.mean(out_full[:, 1])
        out_variance = np.sqrt(np.var(out_full[:, 1]))
        return out_avg, out_variance
    else:
        out_avg = np.trapz(out_full[:, 1], x=out_full[:, 0])/(out_full[-1, 0] - out_full[0, 0])
        return out_avg


def fluxes_nusselts_wrms_hydr_DNS():
    """
    Now that I've added get_vars_hydro and get_avg_from_hydro_DNS, this function is outdated.
    Keeping it for now because some of my scripts still call it. TODO: fix that
    """
    tau = 0.1
    FCs = np.zeros_like(R0s_hydro)
    FTs = np.zeros_like(R0s_hydro)
    NuCs = np.zeros_like(R0s_hydro)
    NuTs = np.zeros_like(R0s_hydro)
    wrmss = np.zeros_like(R0s_hydro)
    for ri, r0 in enumerate(R0s_hydro):
        if int(r0) == r0:
            r0string = str(int(r0))
        else:
            r0string = str(r0)
        outdir = '/Users/adfraser/PADDIM/Pr0.1_R' + r0string + '_HB0/'
        last_out = max([int(OUTpath.split('OUT')[-1]) for OUTpath in glob.glob(outdir + 'OUT*')])  # count OUTfiles
        names = [outdir + 'OUT{}'.format(i) for i in range(1, last_out + 1)]  # create list of OUTfiles
        # vars_in = ['t', 'flux_Temp', 'flux_Chem', 'uzrms']
        # dumb hack: using a PADDIM OUT reader for PADDI OUT files
        # (basically, what I'm calling 'Brms' here is actually 'flux_Temp', because the columns are different in PADDI
        # than they are in PADDIM because you don't have all the B-related entries)
        vars_in = ['t', 'Brms', 'flux_Temp', 'B_max(2)']
        out = get_vars(names, vars_in, flat=True)[avg_starts_hydro_DNS[ri]:]
        FTs[ri] = np.trapz(out[:, 1], x=out[:, 0]) / (out[-1, 0] - out[0, 0])
        NuTs[ri] = 1.0 + FTs[ri]
        FCs[ri] = np.trapz(out[:, 2], x=out[:, 0]) / (out[-1, 0] - out[0, 0])
        NuCs[ri] = 1.0 + r0 * FCs[ri] / tau
        # gamtot_hydro_DNS[ri] = (FTs[ri] + 1.0) / (FCs[ri] + tau / r0)
        wrmss[ri] = np.trapz(out[:, 3], x=out[:, 0]) / (out[-1, 0] - out[0, 0])  # * np.sqrt(2.0)
    return [FCs, FTs, NuCs, NuTs, wrmss]
