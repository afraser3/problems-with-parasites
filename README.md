# Problems with parasites
This repo contains tools for calculating parasitic saturation models of the fingering instability in MHD, and contains 
scripts for making several figures that appear in the paper "Magnetized fingering convection in stars" (previously 
titled "Problems with Parasites") by A.E. Fraser, S.A. Reifenstein, and P. Garaud (submitted to ApJ, arXiv:2302.11610).

## The three main scripts that everything else is built on:
### python/fingering_modes.py
This script contains various functions for calculating the growth rate, lambda (sometimes "gamma" or "gam"), and the 
fastest-growing wavenumber, l_f (or its square, sometimes called "l2" or "k2" in the code), of the fingering 
instability. It's just solving equations 19 and 20 of Brown et al. 2013 using a Newton method. (Note that a faster & 
more reliable method exists, as pointed out by Rich Townsend, and I still need to implement it.)

### python/kolmogorov_EVP.py
This script contains several functions used to solve the linear stability problem of a sinusoidal shear flow with a 
uniform, streamwise magnetic field (the system described in Fraser et al JFM 2022). This is needed for calculating
sigma_KH in certain parasitic saturation models of the fingering instability. Recently updated to include the 
"with T & C" model described in the paper (i.e. the "full" model that correctly predicts the fluxes in simulations).

### python/parasite_model.py
This script contains functions for calculating w_f, the vertical velocity of the fingers. The first function, w_f, uses
kolmogorov_EVP.py to do so in the "full" model (i.e., with sigma_KH calculated explicitly rather than using the fit from
HG19). The functions HG19_eq32, dEQ32dw, and w_f_HG19 calculate w_f using HG19's approximate equation for sigma_KH, 
rather than manually calculating sigma_KH. All of the remaining functions are just convenience functions for, e.g., 
calculating Nusselt numbers from w_f.

## Some general guidance / words of caution
- I provide scripts for generating a variety of plots (including plots in the paper) in hopes that they are useful. 
Many can be readily modified to plot a variety of different quantities besides just compositional flux. If a plot is
passing a string along to parasite_model.results_vs_r0, for instance, you can take a look at the source code for that
function to see a variety of other strings to pass if you want to plot other quantities.
  - You can also play with get_and_save_parasite_predictions.py if you want to just save parasite model predictions to a txt file and then write your own plot script 
  - As currently written, the scripts that compare to simulation data just directly access the relevant simulation output files I have on my laptop and do time averages on the fly. In case others want to play with this data, I put time-averaged DNS data in the extracted_data directory, and it shouldn't be too hard to modify plotting scripts to extract that data instead of the way I have it set up on my local machine (I hope to fix this soon!) 
- In most of these plot scripts you should find various physical parameters specified near the top. You should be able
to change them if you want to quickly see how a figure changes for some other star / set of physical parameters.
- As described in the paper, you need to select a number of Fourier modes to include in the parasite model. *You should
always do convergence checks on this quantity.*
- The parasite model involves a search over kz (vertical wavenumber of the parasitic mode). The kz's to search over is
generally specified at the top of each plotting script. Generally (I think in all cases!) they are kz-stars 
(i.e., kz in the units presented in arXiv v1 of this paper, or in Fraser, Cresswell, Garaud 2022 JFM). *Always* 
scrutinize the choice of kz's to search over. Make sure your answer doesn't change if you pass a denser sampling of 
kz's, and make sure your smallest kz is small enough, otherwise you risk missing different branches of the instability 
(e.g., the varicose and DF modes from our 2022 JFM, or equivalent modes for the withTC case).