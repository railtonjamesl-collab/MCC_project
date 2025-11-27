# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:24:35 2025

@author: railt
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv(r"C:\Users\railt\MCC_project\results\numerical_alex.csv")

# Unique alphas
alphas = df["alpha"].unique()

for alpha in alphas:
    df_alpha = df[df["alpha"] == alpha]

    for signal in df_alpha["signal_type"].unique():
        subset = df_alpha[df_alpha["signal_type"] == signal]
        
        pivot = subset.pivot_table(
            index="sample",
            columns="fwhm",
            values="success_rate"
        )

        plt.figure()

        for fwhm in pivot.columns:
            plt.plot(
                pivot.index,
                pivot[fwhm],
                marker='o',
                label=f"FWHM = {fwhm}"
            )

        plt.title(f"Success Rate vs Sample Size, {signal}, α = {alpha}")
        plt.xlabel("Sample Size")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.axhline(y=1-alpha, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.show()