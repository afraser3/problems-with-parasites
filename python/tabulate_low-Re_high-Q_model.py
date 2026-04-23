"""
This is meant to demonstrate to Rich how to calculate the low-Re, high-Q limit of the FRG24 model in a way that can be
readily tabulated so that MESA doesn't need to calculate it on-the-fly.

In the docstring and code below, I refer heavily to the "supercriticality" epsilon = -1 + 1/(R_0 * tau) (called "eps" in
the code). This basically measures how far away from the right end of FRG24 fig 7 one is.

I also refer to the aspect ratio or anisotropy, alpha = epsilon^1/4 / Q^1/2. Here,
Q = DB * Pr / HB = nu * eta / (d * v_A)^2 is the Chandrasekhar number (nu is kinematic viscosity, eta is resistivity,
d is the characteristic finger width, v_A is the Alfven speed of the background magnetic field).

Assumptions and regimes of validity:
- First, we take the low-Peclet-number limit. We assume that tau (really, tau * epsilon^a for some positive a) is
  so small that the T equation reduces to a balance between the psi_m part and the diffusive k_m^2 T_m part.
- We also take the low-magnetic-Reynolds-number, or quasistatic, limit. We assume that the magnetic Reynolds number
  (which is something like tau * epsilon^a / DB for some positive a) so that the A equation reduces to a balance
  between the psi_m part and the diffusive k_m^2 A_m part.
- Next, we assume that tau * epsilon / Pr is small. This is basically a low-Reynolds-number limit, since it makes
  inertia subdominant in the momentum equation and leaves just viscosity and other forces (specifically buoyancy and
  Lorentz). Thus, the momentum equation reduces to a balance between the viscous, buoyancy, and Lorentz forces.
- The above three are the most important aspects of this regime physically. But there's a fourth bit that is crucial for
  reducing things in a way that is conducive towards tabulation. If we further assume that the aspect ratio alpha is
  very small (note that Q is usually quite large -- it's 10^6 for the B_0 = 1000 G curve in Fig 7a of FRG24, so it's
  10^5 for the B_0 = 100 G curve), then various terms in the model reduce in a way that the resulting model depends on
  fewer physical parameters. Thus, the tabulation only needs to be in terms of epsilon. If we didn't take this step, the
  tabulation would need to be in terms of more input parameters.

After making the above assumptions, doing the parity-splitting (where one finds that only one of the parity-split blocks
actually needs to be retained/evaluated, the other can be discarded) *and* doing various unit conversions, the finger
velocity w_f, in funny units, can be calculated as a simple function of epsilon. Let's call that w_f_breve. Then the w_f
in the units adopted by the original FRG24 model -- let's call that w_f_hat -- can be calculated according to:
  w_f_hat = w_f_breve * tau * epsilon^(5/4) / alpha

Oh, another nice thing about this model is that I have a pretty good idea for where the peak k_z is. So the scan over
k_z is a little easier. Also,
"""


import numpy as np
from scipy import optimize as opt


def eval_finger_IFSC(eps):
    # Like Rich's `eval_finger()` code, but in the low-Peclet-number and low-Reynolds-number limit, which is
    # called the "inertia-free salt convection" (IFSC) limit by some. Evaluates eqns 24-25 of Xie et al (2017) Fluids;
    # returns the growth rate and wavenumber in this limit, which is just a function of supercriticality (my eps and
    # Xie's Ra are related by Ra = eps + 1). Note this returns the quantities in the tilde'd units of my notes.
    # Unit conversions (lam_f_ACCENT and l_f_ACCENT are the growth rate \lambda_f and wavenumber l_f in ACCENT units):
    # lam_f_bar = lam_f_tilde / eps**1.5
    # l_f_bar = l_f_tilde / eps**0.25
    # lam_f_hat = lam_f_tilde * tau
    # l_f_hat = l_f_tilde
    # l_f_breve = l_f_bar
    # lam_f_breve = lam_f_bar
    Ra = eps + 1
    lam_f_tilde = np.sqrt(0.5*(-2 - Ra + np.sqrt(Ra**2 + 8*Ra))) * (3*Ra - np.sqrt(Ra**2 + 8*Ra))/(np.sqrt(Ra**2 + 8*Ra) - Ra)
    l4_f_tilde = 0.5*(-2 - Ra + np.sqrt(Ra**2 + 8*Ra))
    return lam_f_tilde, l4_f_tilde**0.25


def NuC(tau, w, lamhat, l2hat, KB=1.24):
    # Calculates the compositional Nusselt (or "Sherwood"; because fluid dynamicists can't name enough things after
    # enough people) number given w_f_hat.
    # Note "KB" here and in HG19 is equivalent to "C_1" in FRG24, and 1.24 is the value advocated by HG19 (and BGS13)
    # but 0.62 by FRG24 when using the full FRG24 ("with_TC") model
    return 1 + KB * w ** 2.0 / (tau * (lamhat + tau * l2hat))


def max_eval_from_matrix(L):
    ws = np.linalg.eigvals(L)
    ind = np.argmax(np.real(ws))
    return np.real(ws[ind])


def build_matrix_breve_LPN_QS_LRN_LA_b(N, kz_breve, eps, l_f_bar, E_psi_breve, E_C_breve):
    # Same as the build_matrix functions written by Rich, but assumes:
    # - Low-Peclet-number balance ("LPN", reduces T equation)
    # - Quasistatic balance ("QS", reduces A equation)
    # - Low-Reynolds-number balance ("LS", reduces psi equation)
    # - Low aspect ratio, alpha ("LA", takes k_m^2 \approx m^2 l_f^2 for m>0)
    # This is the "b" submatrix, which includes the C_0, C_1+, ... elements. I haven't tried working this model out for
    # either the "a" submatrix or for the full, non-parity-split system, but my notes suggest that this is the only part
    # we care about in this limit.
    #
    # "breve" because this is in the breve'd units of my notes

    # Set up the matrix elements
    n = N + 1

    Q = np.zeros((n, n), dtype=complex)

    # The basis ordering is: C_0, C_1+, ...

    # start with the m=0 part

    # first, set up the 0th row, corresponding to C_0
    # no Q[0, 0] part for low-aspect-ratio limit
    Q[0, 1] = - l_f_bar * kz_breve * E_psi_breve
    Q[0, 1] += l_f_bar**2 * (kz_breve/eps) * E_C_breve * 2 / (2**4 * l_f_bar**4 + kz_breve**2 + 1/eps)

    for m in range(1, N + 1):

        # Set up block indices

        i_c = m
        i_c_p = i_c + 1
        i_c_m = i_c - 1

        # Evaluate wavenumbers (see the line immediately following eq 44 in FRG24)
        # note these expressions are reduced in the low-alpha limit -- the kz^2 part drops out

        k2_m = m ** 2 * l_f_bar ** 2
        k2_m_p = (m + 1) ** 2 * l_f_bar ** 2
        k2_m_m = (m - 1) ** 2 * l_f_bar ** 2

        # Set matrix elements

        # C eqn

        Q[i_c, i_c] = -k2_m / eps
        Q[i_c, i_c] += (1 + 1/eps) * (k2_m / eps) / (k2_m ** 2 + kz_breve ** 2 + 1/eps)

        Q[i_c, i_c_m] = l_f_bar * kz_breve * E_psi_breve
        if m > 1:
            Q[i_c, i_c_m] += l_f_bar**2 * kz_breve * (E_C_breve/eps) * (m-1) / (k2_m_m**2 + kz_breve ** 2 + 1/eps)
        if m < N:
            Q[i_c, i_c_p] = -l_f_bar * kz_breve * E_psi_breve
            Q[i_c, i_c_p] += l_f_bar**2 * kz_breve * (E_C_breve/eps) * (m+1) / (k2_m_p**2 + kz_breve ** 2 + 1/eps)

    return Q


def gammas_minus_lambda_breve_LPN_QS_LRN_LA(wbreve, N, kz_breves, eps, l_f_bar, lam_f_bar, C2=0.33):
    # Helper function that, for a given w_f, calculates the parasite growth rate (sigma) vs k, finds the max, then
    # multiplies by C2 and subtracts lam_f, see FRG24 eq 28. This helper function can then be passed to
    # scipy.optimize.root_scalar()

    # l_f_breve0 is the value of lim_{eps -> 0} l_f_breve (see Eq 26a of Xie et al 2017; recall l_f_breve = l_f_bar). 
    # For some reason, in the E_C expression below, one instance of l_f_bar is replaced with l_f_breve0. I think this 
    # was a relic from when I was previously testing the eps -> 0 limit, so strictly speaking it's a typo. But for some 
    # reason it makes this model better! Go figure. Gotta look into this more later.
    l_f_breve0 = 3 ** -0.25
    E_psi_breve = wbreve / (2 * l_f_bar)
    E_C_breve = - l_f_breve0 * E_psi_breve * (eps + 1) / (l_f_bar**2 + eps*lam_f_bar)
    gammas = np.array([max_eval_from_matrix(build_matrix_breve_LPN_QS_LRN_LA_b(N, kzb, eps, l_f_bar, E_psi_breve, E_C_breve)) for kzb in kz_breves])
    gamma_max = np.max(gammas)
    # if gamma_max == gammas[0]:
    #     raise ValueError("Need to scan lower k's")
    if gamma_max == gammas[-1]:
        raise ValueError("Need to scan higher k's")
    return gamma_max * C2 - lam_f_bar


def w_f_breve_LPN_QS_LRN_LA(N, kz_breves, eps, l_f_bar, lam_f_bar, C2=0.33):
    w1 = 22
    w2 = 28
    wbounds = [w1, w2]
    args = (N, kz_breves, eps, l_f_bar, lam_f_bar, C2)

    rbound_eval = gammas_minus_lambda_breve_LPN_QS_LRN_LA(wbounds[1], N, kz_breves, eps, l_f_bar, lam_f_bar, C2)
    lbound_eval = gammas_minus_lambda_breve_LPN_QS_LRN_LA(wbounds[0], N, kz_breves, eps, l_f_bar, lam_f_bar, C2)
    # print(eps, lbound_eval, rbound_eval)
    count = 0
    while rbound_eval < 0:
        print('*adjusting bounds, count=', count)  # keeps track of how many times w1 and w2 need adjustment
        count += 1
        wbounds = [wbounds[1], 2*wbounds[1]]
        rbound_eval = gammas_minus_lambda_breve_LPN_QS_LRN_LA(wbounds[1], N, kz_breves, eps, l_f_bar, lam_f_bar, C2)
    # if rbound_eval < 0:
    #     raise ValueError("w2 is too small")
    sol = opt.root_scalar(gammas_minus_lambda_breve_LPN_QS_LRN_LA, args=args, bracket=wbounds)
    return sol.root


epsilons = np.linspace(1e-2, 0.227, num=20, endpoint=True)
Pr = 1e-6
tau = 1e-7
Pm = 1e-3
DB = Pr / Pm
HB = 1e-5
alphas = epsilons**0.25 * np.sqrt(DB * Pr / HB)

N = 17
# wavenumbers k_z to scan over when maximizing parasite growth rate sigma(k_z)
k_stars = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.append(np.linspace(0.1, 4, endpoint=False), np.geomspace(4, 1e3, num=50, endpoint=False)))

C1 = 0.62
C2 = 0.33
lam_f_tildes_IFSC, l_f_tildes_IFSC = np.transpose([eval_finger_IFSC(eps) for eps in epsilons])
lam_f_bars_IFSC = lam_f_tildes_IFSC / (epsilons**1.5)
l_f_bars_IFSC = l_f_tildes_IFSC / (epsilons**0.25)
lam_f_hats_IFSC = lam_f_tildes_IFSC * tau
l_f_hats_IFSC = l_f_tildes_IFSC

wf_breves_LPN_QS_LRN_LA = np.zeros_like(epsilons)
# at each epsilon, calculate w_f_breve. Note from the call to w_f_breve_LPN_QS_LRN_LA() that we're just passing epsilon,
# not Pr, tau, HB, etc. So this quantity ONLY depends on epsilon
for wi, eps in enumerate(epsilons):
    k_breves = k_stars * l_f_bars_IFSC[wi] / alphas[wi]
    wf_breves_LPN_QS_LRN_LA[wi] = w_f_breve_LPN_QS_LRN_LA(N, k_breves, eps, l_f_bars_IFSC[wi], lam_f_bars_IFSC[wi], C2=C2)

# unit conversion: calculate w_f_hat from w_f_breve. Here's where those other physical parameters come in as power laws
w_f_hats = wf_breves_LPN_QS_LRN_LA * tau * epsilons**(5/4) / alphas
# then calculate D_C/kappa_C from w_f_hat the same old way as before
DCs_LPN_QS_LRN_LA = NuC(tau, w_f_hats, lam_f_hats_IFSC, l_f_hats_IFSC**2, KB=C1) - 1
