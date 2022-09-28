# ArnoldMap
harmonic period bifurcation map computing and simulation

usage: run.py [-h] [-ncpus NCPUS] [-dimK DIMK] [-dimtau DIMTAU] [-N_steps N_STEPS] [-dt DT]
              [--progressbar]

Computes n-period for solutions generated by a model for a (dimK x dimtau) parameter space grid

options:
  -h, --help        show this help message and exit
  -ncpus NCPUS      Number of child processes It is recommended to use at most the number of available
                    CPU cores
  -dimK DIMK        K parameter dimension, corresponding to K-tau plane
  -dimtau DIMTAU    tau parameter dimension, corresponding to K-tau plane
  -N_steps N_STEPS  Number of steps performed on each simulation Consider the program runs a whole
                    simulation for each cell of the grid
  -dt DT            dt
  --progressbar     display progressbar

------------------------------
Full documentation coming soon
