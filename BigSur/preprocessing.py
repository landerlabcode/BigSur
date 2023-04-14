#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:55:15 2022

@author: emmanueldollinger
"""
from typing import Optional, Union
import time
import pandas as pd
import numpy as np
from anndata import AnnData
from scipy.optimize import curve_fit, brentq
from sklearn.utils.sparsefuncs import mean_variance_axis
from mpmath import ncdf, exp
from statsmodels.stats.multitest import fdrcorrection
import numexpr as ne
import warnings

warnings.simplefilter('always', UserWarning)

def make_vars_and_QC(adata, layer):
    """Extracts raw_count_mat, means, variances, g_counts and does checks for QC fail and raw counts"""
    if layer == "X":
        raw_count_mat = adata.X.copy()
    else:
        raw_count_mat = adata.layers[layer].copy()

    means, variances = mean_variance_axis(raw_count_mat, axis=0)
    g_counts = np.asarray(raw_count_mat.sum(axis=0)).flatten()

    ## Checks
    ### Check if any means == 0, probably means QC was not done
    if np.any(means == 0):
        raise Exception("Zero means were found, run QC steps before calculating mFF.")
    ### Check if integer data was passed, if not probably passed normalized data

    if any(g_counts.astype(int) != g_counts):
        raise Exception("This function takes raw counts, normalized data was passed.")
    return raw_count_mat, means, variances, g_counts


def calculate_residuals(cv, verbose, raw_count_mat, means, variances, g_counts):
    """This function calculates the corrected fano factors"""
    # Estimate the coefficient of variation if one is not supplied
    if not isinstance(cv, float):
        unmod_fanos = variances / means
        cv = fit_cv(xdata=means, ydata=unmod_fanos)
    if verbose > 1:
        print(f"Using a coefficient of variation of {cv:.4}.")

    # Correcting for differential read depth among cells (calculating cell-specific expected gene means)
    total_umi = np.array(raw_count_mat.sum(axis=1)).flatten()
    normlist = total_umi / raw_count_mat.sum()
    # Modify Fano factors by accounting for differential read depth and dividing 1+c^2*mu
    n_cells = normlist.shape[0]

    outerproduct = np.outer(normlist, g_counts)

    dense = raw_count_mat.toarray()
    residuals = ne.evaluate(
        "(dense-outerproduct)/(outerproduct*(1+outerproduct*cv**2))**(1/2)",
        {"dense": dense, "outerproduct": outerproduct, "cv": cv},
    )
    return cv, normlist, residuals, n_cells


def fit_cv(xdata, ydata, p0=0.5):
    def expected_fano(x, c):
        return 1 + x * c * c

    init_fit, _ = curve_fit(f=expected_fano, xdata=xdata, ydata=ydata, p0=p0)
    cv = init_fit[0]
    return cv