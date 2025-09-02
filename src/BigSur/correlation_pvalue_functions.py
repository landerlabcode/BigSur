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


def calculate_pvalues(correlation_roots):
    print("Estimating p-values.")
    # Calculate log p-values for column 3, two-sided
    p = norm.logcdf(correlation_roots)
    p_mpfr = -np.log10(-p / np.log(10))
    p_mpfr[correlation_roots < 8.2] = -np.log10(1 - np.exp(p[correlation_roots < 8.2]))
    p_mpfr[correlation_roots >= 38.4] = [float(mpmath.nstr(-mpmath.log10(0.5 * mpmath.exp(- (mpmath.power(mpmath.mpf(correlation_roots[loc]), 2)) / 2)), 15)) for loc in np.where(correlation_roots >= 38.4)[0]]

    # tic = time.perf_counter()
    # for row in range(correlation_roots.shape[0]):
    #     abs_root = correlation_roots[row]
    #     log_p = p[row]
    #     if abs_root < 8.2:
    #         val = -np.log10(1 - np.exp(log_p))
    #     elif abs_root >= 38.4:
    #         mp_val = mpmath.nstr(-mpmath.log10(0.5 * mpmath.exp(- (abs_root ** 2) / 2)), 15)
    #         val = float(mp_val)
    #     else:
    #         val = -np.log10(-log_p / np.log(10))
    #     p_mpfr[row] = val
    # toc = time.perf_counter()
    # print(f"Finished estimating p-values in {(toc-tic):04f} seconds.")
    p_values = np.array(10**-p_mpfr)
    return p_values

def BH_correction(p_values, num_genes):
    '''Perform Benjamini-Hochberg correction on p-values.'''
    indices_of_smallest_to_greatest_p_values = p_values.argsort()
    recovery_index = np.argsort(indices_of_smallest_to_greatest_p_values)
    sorted_pvals = p_values[indices_of_smallest_to_greatest_p_values]
    BH_corrected_pvalues = sorted_pvals * ((num_genes * (num_genes - 1)) / 2) / np.arange(1, len(sorted_pvals) + 1)
    BH_corrected_pvalues[BH_corrected_pvalues > 1] = 1
    BH_corrected_pvalues_reordered = BH_corrected_pvalues[recovery_index]
    return BH_corrected_pvalues_reordered