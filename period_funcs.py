#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022

@author: aguscarpio99
"""
import numpy as np

def true_neighbours(frames, i, j, order, T, Rtol):
    R = 0
    for o in range(order):
        R += abs(frames[i+o*T]-frames[j+o*T])
    #  print(R)
    if R<Rtol:
        return True
    return False

def period(frames):
    X = frames.real
    Y = frames.imag
    dXdt = np.gradient(X)
    dYdt = np.gradient(Y)
    TH = np.arctan2(dYdt, dXdt)

    dTH = np.diff(TH)
    dTH[dTH < -6] += 2*np.pi
    cumulative_dTH = np.cumsum(dTH)
    # 2000 arbitrario
    idx = np.argmax(np.gradient(cumulative_dTH[:5000]))

    vueltas = (cumulative_dTH-cumulative_dTH[idx])/(2*np.pi)
    land = np.logical_and
    candidate_idxs = []
    for i in range(1,17):
                candidate_idxs += list(np.where(land(vueltas[:-1]<i, vueltas[1:]>i))[0])
    for i_cand in candidate_idxs:
        if true_neighbours(frames, idx, i_cand, 30, 300, 0.1):
            return round(vueltas[i_cand])
    return np.nan
