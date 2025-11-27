# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 01:43:36 2025

@author: railt
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv
from functions import (generate_circle_100grid, hausdorff, boundary_dilation, 
                       compute_boundary, rademacher, generate_gradient
                       , numerical_test)
from scipy.spatial.distance import directed_hausdorff
import os
import pandas as pd


FWHMS = [1,2,3]
ALPHAS = [0.1, 0.05]
SAMPLES = [50,100,250,400]
SIGNAL_TYPES = ['circle','grad']
BOOT_REP = 1000
THR = 2
REPEAT = 500

results = []
for signal in SIGNAL_TYPES:
    for fwhm in FWHMS:
        for alpha in ALPHAS:
            for n in SAMPLES:
                
                print(f"Running: signal={signal}, fwhm={fwhm}, alpha={alpha}, samples={n}")
                
                success_rate = numerical_test(repeat= REPEAT, alpha = alpha,
                                              signal_type = signal, threshold = THR,
                                              fwhm = fwhm, samples = n,
                                              boot_rep = BOOT_REP)
                row = {"signal_type" : signal,
                       "fwhm": fwhm,
                       "alpha":alpha,
                       "sample":n,
                       "threshold": THR,
                       "boot_rep": BOOT_REP,
                       "repeat": REPEAT,
                       "success_rate":success_rate
                       } 
                results.append(row)

df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
out_csv = "results/numerical_alex.csv"
df.to_csv(out_csv, index=False)

