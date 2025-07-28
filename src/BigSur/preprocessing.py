#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:55:15 2022

@author: emmanueldollinger
"""
import numpy as np
from scipy.stats import linregress
from sklearn.utils.sparsefuncs import mean_variance_axis
import numexpr as ne
import warnings

warnings.simplefilter('always', UserWarning)

def make_vars_and_qc(adata, layer):
    """Extracts raw_count_mat, means, variances, g_counts and does checks for QC fail and raw counts"""
    if layer == "X":
        raw_count_mat = adata.X.copy()
    else:
        raw_count_mat = adata.layers[layer].copy()

    # We've had issues with numerical precision, so we cast everything to float64
    raw_count_mat = raw_count_mat.astype(np.float64)

    means, variances = mean_variance_axis(raw_count_mat, axis=0)
    g_counts = np.asarray(raw_count_mat.sum(axis=0)).flatten()

    ## Checks
    ### Check if any means == 0, probably means QC was not done
    if np.any(means == 0):
        raise Exception("Zero gene means were found, remove genes with no counts before calculating mcFanos.")
    
    ### Check if integer data was passed, if not probably passed normalized data
    if any(g_counts.astype(int) != g_counts):
        raise Exception("This function takes raw counts, normalized data was passed.")
    return raw_count_mat, means, variances, g_counts


def calculate_residuals(cv, raw_count_mat, g_counts):
    """Calculate the corrected modified residuals ."""
    # Correct for differential read depth among cells (calculating cell-specific expected gene means)
    normlist, n_cells, e_mat = calculate_emat(raw_count_mat, g_counts)
    dense = raw_count_mat.toarray()
    residuals = ne.evaluate(
        "(dense-e_mat)/(e_mat*(1+e_mat*cv**2))**(1/2)",
        {"dense": dense, "e_mat": e_mat, "cv": cv},
    )
    return cv, normlist, residuals, n_cells

def calculate_emat(raw_count_mat, g_counts):
    '''Calculate the expectation matrix (e_mat)'''
    total_umi = np.array(raw_count_mat.sum(axis=1)).flatten()
    normlist = total_umi / raw_count_mat.sum()
    # Modify Fano factors by accounting for differential read depth and dividing 1+c^2*mu
    n_cells = normlist.shape[0]
    e_mat = np.outer(normlist, g_counts)
    return normlist,n_cells,e_mat


def fit_cv(raw_count_mat, means, g_counts, verbose, min_mean = 0.1, max_mean = 100):
    '''Fits CV to genes with means > min_mean and means < max_min. Slope of linear fit in mcFano vs mean should be 0, so try different CVs and pick the CV with slope closest to zero.'''

    log_vec = np.logical_and(means > min_mean , means < max_mean)
    subset_means = means[log_vec]
    subset_g_counts = g_counts[log_vec]
    subset_raw_count_mat = raw_count_mat[:, log_vec]

    cv_store = 1.0
    slope_store = 1.0
    for cv_try in np.arange(0.05, 1.05, 0.05):
        cv_try = np.round(cv_try, 3)
        cv_try, normlist, residuals, n_cells = calculate_residuals(
            cv_try, subset_raw_count_mat, subset_g_counts
        )

        corrected_fanos = calculate_mcfano(residuals, n_cells)

        fit_object = linregress(np.log10(subset_means), np.log10(corrected_fanos))

        slope = fit_object[0]

        if slope < 0:
            if slope_store < np.abs(slope):
                cv = cv_store
            else:
                cv = cv_try
            if verbose > 1:
                print(f"Using a coefficient of variation of {cv:.4}.")
            return cv
        else:
            cv_store = cv_try
            slope_store = slope
    warnings.warn(
                'CV cannot be fit in biological range -- this probably means that the dataset is composed of multiple celltypes. We recommend subsetting the celltypes and redoing CV fit. Setting CV = 0.5.')
    cv = 0.5
    return cv

def calculate_mcfano(residuals, n_cells):
    '''Calculate modified corrected Fano factors.'''
    squared_residuals = residuals**2
    corrected_fanos = 1 / (n_cells - 1) * np.sum(squared_residuals, axis=0)
    corrected_fanos = np.array(corrected_fanos).flatten()

    return corrected_fanos