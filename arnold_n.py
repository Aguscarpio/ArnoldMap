#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022

@author: aguscarpio99
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from numba import njit
from delayed_wilsoncowan_rk4 import rk4_numba
from save import save_return, save_data
from period_funcs import period

# TODO: Let users pick the model (+ options than just delayed_wilsoncowan_rk4)

# Fixed parameters
rox = -2.3
roy = -9
mu = 150

# Variable parameters' ranges
mintau = 0.04
maxtau = 0.07
minK = 3
maxK = 10

# Parameters passed with run.py
arn_number, ncpus, dimK, dimtau, N_steps = [int(n) for n in sys.argv[1:6]]
dt = float(sys.argv[6])

# main function
def run_arnold(arn_number, ncpus, dimK, dimtau, N_steps, dt):
    tau_steps = int(maxtau/dt)

    # Precompilation (not sure if needed)
    Z_0_vec = np.zeros(tau_steps+1, dtype="complex")
    rk4_numba(Z_0_vec, dt, 1, rox, roy, mu, 0, tau_steps)

    # We need some initial condition vector (here taken from last simulation)
    with open("last_tau_vector.npy", "rb") as f:
        frames_base = np.load(f)

    # empty grid to fill with period calculations
    periods_grid = np.empty(shape=(dimtau,dimK))*np.nan

    # Integer needed
    splitted_size = dimK//ncpus

    # loop through parameters grid (splitted in ncpus parts)
    tau_index = arn_number*splitted_size
    for tau in np.split(np.linspace(mintau, maxtau, dimtau),ncpus)[arn_number]:
        frames = frames_base
        tau_steps = int(tau/dt)
        K_index = 0
        for K in np.linspace(minK, maxK, dimK):
            Z_0_vec = frames[-tau_steps:]

            # Numerical integration of the model
            frames = rk4_numba(Z_0_vec, dt, N_steps, rox, roy, mu, K, tau_steps)

            # Period calculation with imported function
            periods_grid[tau_index][K_index] = period(frames[-150000:])
            K_index += 1
        tau_index += 1

        # Save values (a file per partition)
        with open(f"results/periodsgrid_{arn_number}.npy", "wb") as f:
            np.save(f, periods_grid)
    with open(f"periodsgrid_{arn_number}.npy", "wb") as f:
        np.save(f, periods_grid)

run_arnold(arn_number, ncpus, dimK, dimtau, N_steps, dt)
