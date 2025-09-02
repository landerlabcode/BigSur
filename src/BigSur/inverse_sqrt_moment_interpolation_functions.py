from typing import Union, Iterable
import time
import numpy as np
import mpmath
import numexpr as ne
import warnings
import pandas as pd
import os

## Numpy
from numpy.polynomial import Polynomial

## Anndata
from anndata import AnnData
from mpmath import ncdf, exp

## statsmodels
from statsmodels.stats.multitest import fdrcorrection

## Joblib
from joblib import Parallel, delayed

### Scipy
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from scipy.stats import norm
from scipy.sparse import csr_matrix, save_npz

# Functions for inverse square moment interpolation
def inv_sqrt_moment_interpolation(sample_moments, gene_totals, points):
    int_moments = np.empty((4, gene_totals.shape[0]))
    for j in range(4):
        approx_func = interp1d(
            np.log10(points),
            np.log10(sample_moments[:, j]),
            kind='linear',
            fill_value='extrapolate'
        )
        interpolated = np.power(10, approx_func(np.log10(gene_totals)))
        int_moments[j, :] = interpolated
    e_moments = np.array([np.outer(m, m) for m in int_moments]) # 4 x n_genes x n_genes
    return e_moments
def inverse_sqrt_mcfano_correction(n_cells, g_counts, c, normlist):
    a = max(2, min(g_counts))
    e = n_cells / 50
    h = max(g_counts)
    points = np.array([a, a * (e / a) ** (1 / 4), a * (e / a) ** (1 / 2), a * (e / a) ** (3 / 4), e, e * (h / e) ** (1 / 3), e * (h / e) ** (2 / 3), h], dtype=int) # 8 points
    trials = 4*10**7/(n_cells*(np.log10(points)**(1/5)+0.5*np.log10(points)**3)) # should be ints
    trials = trials.astype(int) # the same

    sim_emat = np.outer(points, normlist) # 8 x n_cells

    sample_moments = np.array([simulate_inverse_sqrt_mcfano_moments(sim_emat[i,:], c, n_cells, trials[i]) for i in range(points.shape[0])])

    e_moments = inv_sqrt_moment_interpolation(sample_moments, g_counts, points)
    return e_moments
def simulate_inverse_sqrt_mcfano_moments(sim_emat_subset, c, n_cells, trial, starting_seed = 0):
    mu = np.log(sim_emat_subset / np.sqrt(1 + c**2))
    sigma = np.sqrt(np.log(1 + c**2))

    rng = np.random.default_rng(starting_seed)

    PLN_samples = rng.poisson(rng.lognormal(mean=mu.reshape(1,-1), sigma=sigma), size = (trial, n_cells))
    samples = 1/np.sqrt(np.sum((PLN_samples-sim_emat_subset)**2/(sim_emat_subset+c**2*sim_emat_subset**2), axis = 1)/(n_cells-1))

    results = [np.mean(samples**n) for n in range(2, 6)] # Return the second through 5th moments

    return(results)