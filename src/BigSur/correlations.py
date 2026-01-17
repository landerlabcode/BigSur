#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import time
import numpy as np
import warnings
import os

## Anndata
from anndata import AnnData

### Scipy
from scipy.sparse import csr_matrix, save_npz
from scipy.io import mmwrite

# Load BigSur functions
from .preprocessing import make_vars_and_qc, fit_cv
from .correlation_coefficient_functions import load_or_calculate_residuals, load_or_calculate_mc_fanos, load_or_calculate_mcpccs, load_or_calculate_coefficients
from .correlation_cumulant_functions import load_or_calculate_cumulants
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
        store_intermediate_results: bool = False,
        n_jobs: int = -2,
        starting_seed: int = 0,
        verbose: int = 1):
    '''Calculate modified corrected Pearson correlation coefficients (mcPCCs) and Benjamini-Hochberg (BH) corrected p-values for each gene pair in the dataset.

    Parameters
    ----------
    adata - adata object containing information about the raw counts and gene names.
    layer - String, describing the layer of adata object containing raw counts (pass "X" if raw counts are in adata.X).
    cv - Float, coefficient of variation for the given dataset. If None, the CV will be estimated.
    write_out - String, path to directory where intermediate results will be written. 
    previously_run - Bool, if True, the function will attempt to load previously calculated results from the write_out directory.
    store_intermediate_results - Bool, if True, intermediate results will be saved to disk.
    n_jobs - Int, how many cores to use for p-value parallelization. Default is -2 (all but 1 core).
    starting_seed - Int, random seed for moment interpolation. Default is 0.
    verbose - Int, whether to print computations. 0 is no verbose, 1 is a little (what the function is doing) and 2 is full verbose.

    Returns
    -------
    If write_out is a string, the mcPCCs, BH-corrected p-values and gene names will be saved to the specified directory (write_out). Both the mcPCC and BH-corrected p-value matrices are lower triangular with zeros on the diagonal. Since the pipeline just calculates p-values of correlations that are likely to be significant, the BH p-values of the correlations that are judged to be likely non-significant (and therefore not calculated) are stored as 1's. 
    If write_out is None, the mcPCCs and BH-corrected p-values will be stored in the AnnData object as `adata.varm['mcPCCs']` and `adata.varm['BH_corrected_pvalues']`, respectively.

    Example
    -------
    calculate_correlations(adata, layer = 'counts', verbose = 2, write_out=write_out_folder, previously_run=False)

    from scipy.sparse import load_npz
    
    mcPCCs = load_npz(f'{write_out_folder}/mcPCCs.npz')
    BH_corrected_pvalues = load_npz(f'{write_out_folder}/BH_corrected_pvalues.npz')
    mcPCCs_significant = mcPCCs.copy()
    mcPCCs_significant[BH_corrected_pvalues > 0.05] = 0 # Threshold mcPCCs to those that have BH-corrected p-values <= 0.05
    mcPCCs_significant_symmetrical = mcPCCs_significant + mcPCCs_significant.T # If necessary, calculate the mcPCCs symmetrical matrix by adding the lower triangular matrix to its transpose.
    -------
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

    timing_print_statement(verbose, 'modified corrected Fano factors', g_counts.shape[0], tic, toc)

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

    # Calculate inverse mcFano moments
    tic = time.perf_counter()
    e_moments = inverse_sqrt_mcfano_correction(n_cells, g_counts, cv, normlist, starting_seed = starting_seed)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'interpolated moments', g_counts.shape[0], tic, toc)

    tic = time.perf_counter()
    save_kappas, kappa2, kappa3, kappa4, kappa5 = load_or_calculate_cumulants(verbose, cv, write_out, previously_run, g_counts, residuals, e_mat, e_moments)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'cumulants', int((kappa2.shape[0]**2 - kappa2.shape[0])/2), tic, toc, 'correlations')

    # Store
    if (write_out is not None) and (store_intermediate_results):
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

    # Calculate mcPCCs
    tic = time.perf_counter()
    save_mcPCCs, mcPCCs = load_or_calculate_mcpccs(write_out, previously_run, residuals, n_cells, mc_fanos)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'modified corrected Pearson correlation coefficients', int((mcPCCs.shape[0]**2 - mcPCCs.shape[0])/2), tic, toc)

    del mc_fanos, residuals, e_moments, e_mat

    tic = time.perf_counter()
    save_coefficients, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat = load_or_calculate_coefficients(verbose, write_out, previously_run, g_counts, mcPCCs, kappa2, kappa3, kappa4, kappa5)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'coefficients', c1_lower_flat.shape[0], tic, toc, 'correlations')

    if write_out is not None:
        if previously_run:
            if save_mcPCCs:
                if verbose > 1:
                    print('Writing mcPCCs to disk.', flush = True)
                # Convert mcPCCs to lower triangular
                mcPCCs_lower = np.tril(mcPCCs, -1)
                mcPCCs_lower_sparse = csr_matrix(mcPCCs_lower)
            if save_coefficients and (store_intermediate_results):
                if verbose > 1:
                    print('Writing coefficients to disk.', flush = True)
                np.savez_compressed(write_out + 'coefficients.npz', rows=rows, cols=cols, c1_lower_flat=c1_lower_flat, c2_lower_flat=c2_lower_flat, c3_lower_flat=c3_lower_flat, c4_lower_flat=c4_lower_flat, c5_lower_flat=c5_lower_flat)
            else:
                pass
        else:
            if verbose > 1:
                    print('Writing mcPCCs to disk.', flush = True)
            # Convert mcPCCs to lower triangular
            mcPCCs_lower = np.tril(mcPCCs, -1)
            mcPCCs_lower_sparse = csr_matrix(mcPCCs_lower)
            save_npz(write_out + 'mcPCCs.npz', mcPCCs_lower_sparse)
            if store_intermediate_results:
                if verbose > 1:
                    print('Writing coefficients to disk.', flush = True)
                np.savez_compressed(write_out + 'coefficients.npz', rows=rows, cols=cols, c1_lower_flat=c1_lower_flat, c2_lower_flat=c2_lower_flat, c3_lower_flat=c3_lower_flat, c4_lower_flat=c4_lower_flat, c5_lower_flat=c5_lower_flat)
    else:
        # Convert mcPCCs to lower triangular
        mcPCCs_lower = np.tril(mcPCCs, -1)
        mcPCCs_lower_sparse = csr_matrix(mcPCCs_lower)
        adata.varm["mcPCCs"] = mcPCCs_lower_sparse
    
    del mcPCCs

    tic = time.perf_counter()
    rows_to_keep, cols_to_keep, abs_correlation_roots = calculate_mcPCCs_CF_roots(adata, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, 2, g_counts, n_jobs=n_jobs, verbose=verbose)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'roots', abs_correlation_roots.shape[0], tic, toc, 'correlations')

    # For memory purposes, delete all the cumulants that we don't need. This may not have a large impact on memory because the cumulants are simply vectors.
    del c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat,

    tic = time.perf_counter()
    correlation_pvalues = calculate_pvalues(abs_correlation_roots, n_jobs=n_jobs)
    toc = time.perf_counter()
    timing_print_statement(verbose, 'p-values', correlation_pvalues.shape[0], tic, toc, 'correlations')
    
    BH_corrected_pvalues = BH_correction(correlation_pvalues, adata.shape[1])

    # Make new empty matrix
    matrix_reconstructed = np.ones((g_counts.shape[0], g_counts.shape[0]))
    matrix_reconstructed[rows_to_keep, cols_to_keep] = BH_corrected_pvalues
    matrix_reconstructed[cols_to_keep, rows_to_keep] = BH_corrected_pvalues

    # convert to lower triangular
    matrix_reconstructed_lower_triangular = np.tril(matrix_reconstructed, -1)
    matrix_reconstructed_lower_triangular = csr_matrix(matrix_reconstructed_lower_triangular)

    if write_out is not None:
        if verbose > 1:
            print('Writing BH corrected p-values to disk.', flush = True)
        save_npz(write_out + 'BH_corrected_pvalues.npz', matrix_reconstructed_lower_triangular)
        adata.var.index.to_csv(write_out + 'gene_names.csv')
    else:
        adata.varm["BH-corrected p-values of mcPCCs"] = matrix_reconstructed_lower_triangular

def timing_print_statement(verbose, calculated_variable, counts_of_calculated_variable, tic, toc, variable_type = 'genes'):
    if verbose > 1:
        time_diff = toc-tic
        if time_diff < 60:
            print(
                f"Finished calculating {calculated_variable} for {counts_of_calculated_variable} {variable_type} in {(time_diff):04f} seconds."
            )
        else:
            print(
                f"Finished calculating {calculated_variable} for {counts_of_calculated_variable} {variable_type} in {(time_diff/60):04f} minutes."
            )

