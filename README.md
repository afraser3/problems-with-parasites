# Problems with parasites
This repo contains tools for calculating parasitic saturation models of the fingering instability in MHD, and contains 
scripts for making several figures that appear in the paper "Magnetized fingering convection in stars" by A.E. Fraser, 
S.A. Reifenstein, and P. Garaud (submitted to ApJ, arXiv:2302.11610)

## The three main scripts that everything else is built on:
### fingering_modes.py
This script contains various functions for calculating the growth rate, lambda (sometimes "gamma" or "gam"), and the 
fastest-growing wavenumber, l_f (or its square, sometimes called "l2" or "k2" in the code), of the fingering 
instability. It's just solving equations 19 and 20 of Brown et al 2013 using a Newton method.

### kolmogorov_EVP.py
This script contains several functions used to solve the linear stability problem of a sinusoidal shear flow with a 
uniform, streamwise magnetic field (the system described in Fraser et al JFM 2022). This is needed for calculating
sigma_KH in certain parasitic saturation models of the fingering instability. Note that this hasn't been updated yet 
with the "with T & C" model in the paper (i.e. the "full" model that correctly predicts the fluxes in simulations).

### parasite_model.py
This script contains functions for calculating w_f, the vertical velocity of the fingers. The first function, w_f, uses
kolmogorov_EVP.py to do so in the "full" model (i.e., with sigma_KH calculated explicitly rather than using the fit from
HG19). The functions HG19_eq32, dEQ32dw, and w_f_HG19 calculate w_f using HG19's approximate equation for sigma_KH, 
rather than manually calculating sigma_KH. All of the remaining functions are just convenience functions for, e.g., 
calculating Nusselt numbers from w_f.