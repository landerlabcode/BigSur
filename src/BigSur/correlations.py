#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Iterable
import time
import numpy as np
from numpy.polynomial import Polynomial
from anndata import AnnData
from mpmath import ncdf, exp
import numexpr as ne
import warnings
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
import scanpy as sc
import pandas as pd
from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)


def calculate_mcPCCs(adata, layer, cv = None):
    '''Calculate modified Pearson correlation coefficients (mcPCCs) for each gene pair in the dataset.'''
    # Get the expression data for the specified layer

    # Setup        
    # Create variables
    raw_count_mat, means, variances, g_counts = make_vars_and_qc(adata, layer)

    tic = time.perf_counter()
    # Fit cv if not provided
    if cv is None:
        cv = fit_cv(raw_count_mat, means, g_counts, verbose = 0)
    # Calculate residuals
    cv, normlist, residuals, n_cells = calculate_residuals(cv, raw_count_mat, g_counts)
    # Calculate mcfanos from residuals, convert to 2D array
    corrected_fanos = calculate_mcfano(residuals, n_cells)
    corrected_fanos = corrected_fanos.reshape(-1, 1)

    # Calculate mcPCCs
    mcPCCs = 1/((n_cells - 1) * np.sqrt(corrected_fanos * corrected_fanos.T)) * (residuals.T @ residuals)

    adata.varm['mcPCCs'] = mcPCCs