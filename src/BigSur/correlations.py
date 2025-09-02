#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano, calculate_emat
from .helper_functions import load_or_calculate_coefficients, load_or_calculate_mc_fanos, load_or_calculate_residuals, load_or_calculate_mcpccs, load_or_calculate_cumulants, calculate_mcPCCs
from .inverse_sqrt_moment_interpolation_functions import inverse_sqrt_mcfano_correction
from .correlation_root_finding_functions import calculate_mcPCCs_CF_roots
from .correlation_pvalue_functions import calculate_pvalues, BH_correction

warnings.simplefilter('always', UserWarning)

def calculate_correlations(
        adata: AnnData, 
        layer: str, 
        cv: Union[float, None] = None, 
        write_out: Union[str, None] = None, 
        previously_run: bool = False, 
        n_jobs: int = -2,
        verbose: int = 1):
    '''Calculate modified Pearson correlation coefficients (mcPCCs) and Benjamini-Hochberg (BH) corrected p-values for each gene pair in the dataset.

    Parameters
    ----------
    adata - adata object containing information about the raw counts and gene names.
    layer - String, describing the layer of adata object containing raw counts (pass "X" if raw counts are in adata.X).
    cv - Float, coefficient of variation for the given dataset. If None, the CV will be estimated.
    write_out - String, path to directory where intermediate results will be written. If calculating correlations on a large dataset, this is highly recommended to conserve memory.
    previously_run - Bool, if True, the function will attempt to load previously calculated results from the write_out directory.
    n_jobs - Int, how many cores to use for p-value parallelization. Default is -2 (all but 1 core).
    verbose - Int, whether to print computations and top 100 genes. 0 is no verbose, 1 is a little (what the function is doing) and 2 is full verbose.
    '''

    # Setup
    if write_out is not None:
        os.makedirs(write_out, exist_ok=True)
    # Create variables
    raw_count_mat, means, _, g_counts = make_vars_and_qc(adata, layer)

    tic = time.perf_counter()
    if verbose > 1:
        print("Calculating modified corrected Fano factors.")
    # Fit cv if not provided
    if cv is None:
        if verbose > 1:
            print('Fitting cv.')
        cv = fit_cv(raw_count_mat, means, g_counts, verbose)
        if verbose >= 1:
            print(f'After fitting, cv = {cv}')
    # Calculate residuals
    cv, save_residuals, residuals, n_cells, normlist, e_mat = load_or_calculate_residuals(cv, write_out, previously_run, raw_count_mat, g_counts)
    # Calculate mcfanos from residuals
    save_mc_fanos, mc_fanos = load_or_calculate_mc_fanos(write_out, previously_run, residuals, n_cells)

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Fano factors for {mc_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    if write_out is not None:
        if previously_run & (not save_mc_fanos):
            pass
        else:
            if verbose > 1:
                print('Writing mcFanos to disk.', flush = True)
            adata_var = adata.var.copy()
            adata_var['mc_Fano'] = np.array(mc_fanos).flatten()
            adata_var.to_csv(write_out + 'mc_Fano.csv')
    else:
        adata.var["mc_Fano"] = np.array(mc_fanos).flatten()
    adata.uns['CV_for_mc_Fano_fit'] = cv

    # Calculate mcPCCs
    tic = time.perf_counter()
    save_mcPCCs, mcPCCs = load_or_calculate_mcpccs(verbose, write_out, previously_run, tic, residuals, n_cells, mc_fanos)

    del mc_fanos

    # Calculate inverse mcFano moments
    tic = time.perf_counter()
    e_moments = inverse_sqrt_mcfano_correction(n_cells, g_counts, cv, normlist) # These functions are correct

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    save_kappas, kappa2, kappa3, kappa4, kappa5 = load_or_calculate_cumulants(verbose, cv, write_out, previously_run, g_counts, residuals, e_mat, e_moments)

    # Store
    if write_out is not None:
        if previously_run:
            if save_residuals:
                if verbose > 1:
                    print('Writing residuals and expectation matrix to disk.', flush = True)
                np.savez_compressed(write_out + 'residuals.npz', residuals=residuals)
                np.savez_compressed(write_out + 'e_mat.npz', e_mat=e_mat)
            if save_kappas:
                if verbose > 1:
                    print('Writing cumulants to disk.', flush = True)
                np.savez_compressed(write_out + 'cumulants.npz', kappa2=kappa2, kappa3=kappa3, kappa4=kappa4, kappa5=kappa5)
            else:
                pass
        else:
            if verbose > 1:
                    print('Writing cumulants, residuals and expectation matrix to disk.', flush = True)
            np.savez_compressed(write_out + 'residuals.npz', residuals=residuals)
            np.savez_compressed(write_out + 'cumulants.npz', kappa2=kappa2, kappa3=kappa3, kappa4=kappa4, kappa5=kappa5)
            np.savez_compressed(write_out + 'e_mat.npz', e_mat=e_mat)
    else:
        adata.layers["residuals"] = residuals

    del residuals, e_moments, e_mat

    save_coefficients, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat = load_or_calculate_coefficients(verbose, write_out, previously_run, g_counts, mcPCCs, kappa2, kappa3, kappa4, kappa5)

    if write_out is not None:
        if previously_run:
            if save_mcPCCs:
                if verbose > 1:
                    print('Writing mcPCCs to disk.', flush = True)
                np.savez_compressed(write_out + 'mcPCCs.npz', mcPCCs=mcPCCs)
            if save_coefficients:
                if verbose > 1:
                    print('Writing coefficients to disk.', flush = True)
                np.savez_compressed(write_out + 'coefficients.npz', rows=rows, cols=cols, c1_lower_flat=c1_lower_flat, c2_lower_flat=c2_lower_flat, c3_lower_flat=c3_lower_flat, c4_lower_flat=c4_lower_flat, c5_lower_flat=c5_lower_flat)
            else:
                pass
        else:
            if verbose > 1:
                    print('Writing mcPCCs and coefficients to disk.', flush = True)
            np.savez_compressed(write_out + 'mcPCCs.npz', mcPCCs=mcPCCs)
            np.savez_compressed(write_out + 'coefficients.npz', rows=rows, cols=cols, c1_lower_flat=c1_lower_flat, c2_lower_flat=c2_lower_flat, c3_lower_flat=c3_lower_flat, c4_lower_flat=c4_lower_flat, c5_lower_flat=c5_lower_flat)
    else:
        adata.varm["mcPCCs"] = mcPCCs
    del mcPCCs

    rows_to_keep, cols_to_keep, correlation_roots = calculate_mcPCCs_CF_roots(adata, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, 2, g_counts, n_jobs=n_jobs, verbose=verbose)

    # For memory purposes, delete all the cumulants that we don't need. This may not have a large impact on memory because the cumulants are simply vectors.
    del c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat,

    tic = time.perf_counter()
    correlation_pvalues = calculate_pvalues(correlation_roots)
    toc = time.perf_counter()
    if verbose > 1:
        print(f"Finished calculating p-values for {correlation_roots.shape[0]} correlations in {(toc-tic):04f} seconds.")
    BH_corrected_pvalues = BH_correction(correlation_pvalues, adata.shape[1])

    # Reshape everything into sparse matrix
    #BH_corrected_pvalues_matrix = csr_matrix((BH_corrected_pvalues, (rows_to_keep, cols_to_keep)), shape=(g_counts.shape[0], g_counts.shape[0]))

    # Make new empty matrix
    matrix_reconstructed = np.ones((g_counts.shape[0], g_counts.shape[0]))
    matrix_reconstructed[rows_to_keep, cols_to_keep] = BH_corrected_pvalues
    matrix_reconstructed_lower_triangular = np.tril(matrix_reconstructed, -1)
    matrix_reconstructed_lower_triangular_sparse = csr_matrix(matrix_reconstructed_lower_triangular)

    if write_out is not None:
        if verbose > 1:
            print('Writing BH corrected p-values to disk.', flush = True)
        save_npz(write_out + 'BH_corrected_pvalues.npz', matrix_reconstructed_lower_triangular_sparse)
    else:
        adata.varm["BH-corrected p-values of mcPCCs"] = matrix_reconstructed_lower_triangular_sparse

