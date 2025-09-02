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

def calculate_mcPCCs_coefficients(k2, k3, k4, k5, mcPCCs):
    # Convert to numexpr
    c1 = -mcPCCs - k3/(6*k2) + 17*k3**3/(324*k2**4) - k3*k4/(12*k2**3) + k5/(40*k2**2)
    c2 = np.sqrt(k2) + 5*k3**2/(36*k2**(5/2)) - k4/(8*k2**(3/2))
    c3 = k3/(6*k2) - 53*k3**3/(324*k2**4) + 5*k3*k4/(24*k2**3) - k5/(20*k2**2)
    c4 = -k3**2/(18*k2**(5/2)) + k4/(24*k2**(3/2))
    c5 = k3**3/(27*k2**4) - k3*k4/(24*k2**3) + k5/(120*k2**2)

    #clist = np.hstack([c1, c2, c3, c4, c5])
    mcPCCs_length = c1.shape[0]
    z = np.arange(mcPCCs_length)

    # Generate row/col indices for lower triangle
    # Currently, c1[0, 10] != c1[10, 0]
    rows, cols = np.tril_indices(mcPCCs_length, -1)

    c1_lower_flat = np.tril(c1, -1)[rows, cols]
    c2_lower_flat = np.tril(c2, -1)[rows, cols]
    c3_lower_flat = np.tril(c3, -1)[rows, cols]
    c4_lower_flat = np.tril(c4, -1)[rows, cols]
    c5_lower_flat = np.tril(c5, -1)[rows, cols]
    return rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat