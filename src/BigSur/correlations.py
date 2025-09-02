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

from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)

def calculate_mcPCCs(n_cells, mc_fanos, residuals):
    mcPCCs = 1/((n_cells - 1) * np.sqrt(np.outer(mc_fanos, mc_fanos.T))) * (residuals.T @ residuals)
    return mcPCCs

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

# Functions for p-value calculations
def calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv):
    n_cells = residuals.shape[0]
    f2 = e_moments[0,:]
    f3 = e_moments[1,:]
    f4 = e_moments[2,:]
    f5 = e_moments[3,:]

    one_plus_cv_squared_times_emat = 1+cv**2*e_mat
    k3_matrix = (one_plus_cv_squared_times_emat*(3+cv**2*(3+cv**2)*e_mat))/(np.sqrt(e_mat)*(one_plus_cv_squared_times_emat)**(3/2))
    k4_matrix = (1+e_mat*(3+cv**2*(7+e_mat*(6+3*cv**2*(6+e_mat)+cv**4*(6+(16+15*cv**2+6*cv**4+cv**6)*e_mat)))))/(e_mat*(one_plus_cv_squared_times_emat)**2)
    k5_matrix_2 = 1/(e_mat**(3/2)*(one_plus_cv_squared_times_emat)**(5/2)) * (1 + 5*(2+3*cv**2)*e_mat + 5*cv**2*(8+15*cv**2+5*cv**4)*e_mat**2+10*cv**4*(6+17*cv**2+15*cv**4+6*cv**6+cv**8)*e_mat**3+cv**6*(30+135*cv**2+222*cv**4+205*cv**6+120*cv**8+45*cv**10+10*cv**12+cv**14)*e_mat**4)
    
    k3_crossprod = np.matmul(k3_matrix.T, k3_matrix)
    k4_crossprod = np.matmul(k4_matrix.T, k4_matrix)
    k5_crossprod_2 = np.matmul(k5_matrix_2.T, k5_matrix_2)

    kappa2 = (1/(n_cells-1)**2) * f2 * n_cells
    kappa3 = (1/(n_cells-1)**3) * f3 * k3_crossprod
    kappa4 = (1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)
    kappa5 = (1/(n_cells-1)**5) * (-10*f2*f3*k3_crossprod + f5*k5_crossprod_2)

    return kappa2, kappa3, kappa4, kappa5

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

def QuickTest6CF(rows, cols,c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff):
    cut = np.sqrt(2) * erfcinv(2 * 10 ** -first_pass_cutoff)

    def testfunc_1(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat):
        return c1_lower_flat + c2_lower_flat * x + c3_lower_flat * x**2 + c4_lower_flat * x**3 + c5_lower_flat * x**4

    def testfunc_2(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat):
        return c1_lower_flat * testfunc_1(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)

    # Compute cut.vec for both pos and neg cut
    cut_vec_1 = testfunc_1(cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_vec_2 = testfunc_1(-cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_bool = ~(cut_vec_1 * cut_vec_2 < 0)#~np.logical_and(cut_vec_1 < 0, cut_vec_2 < 0) # return False if both are negative
    
    cut_vec2 = testfunc_2(cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_bool2 = ~(cut_vec2 < 0) # return False if cut_vec2 is negative

    correlations_to_keep = np.logical_and(cut_bool, cut_bool2) # keep correlations if both cut_bool 1 and cut_bool 2 are True

    return correlations_to_keep
def SecondTestCF(c2, c3, c4, c5, first_pass_cutoff, n_jobs):
    
    cut = np.sqrt(2) * erfcinv(2 * 10**(-first_pass_cutoff))

    def derivative_function(x, c2, c3, c4, c5):
        derivative = c2 + 2*c3*x + 3*c4*x**2 + 4*c5*x**3
        return derivative

    def test_conditions(c2, c3, c4, c5):
        if derivative_function(-cut, c2, c3, c4, c5) < 0:
            return True
        elif derivative_function(cut, c2, c3, c4, c5) < 0:
            return False
        elif 3*c4**2 < 8*c3*c5:
            return True
        else: 
            sqrt_inner = 9*c4**2-24*c3*c5
            sqrt_val = np.sqrt(sqrt_inner)
            expr1 = (3*c4 - sqrt_val) / (12*c5)
            expr2 = (3*c4 + sqrt_val) / (12*c5)

            term1 = (
                -cut < expr1 < cut and
                ((45*c4**3-36*c3*c4*c5-15*c4**2*sqrt_val+8*c5*(9*c2*c5-c3*sqrt_val)) < 0
            ) )
            term2 = (
                -cut < expr2 < cut and
                ((45*c4**3-36*c3*c4*c5+15*c4**2*sqrt_val+8*c5*(9*c2*c5+c4*sqrt_val))<0) )

            if term1:
                return False
            elif term2:
                return False
            else:
                return True

    # Apply test_conditions per correlation
    correlations_passing = np.array(Parallel(n_jobs=n_jobs)(delayed(test_conditions)(c2[correlation_row], c3[correlation_row], c4[correlation_row], c5[correlation_row]) for correlation_row in range(c2.shape[0])))
    toc = time.perf_counter()

    # index_tracker = np.array(list(range(c2.shape[0])))

    # c2_to_subset = c2.copy()
    # c3_to_subset = c3.copy()
    # c4_to_subset = c4.copy()
    # c5_to_subset = c5.copy()

    # first_test = derivative_function(-cut, c2_to_subset, c3_to_subset, c4_to_subset, c5_to_subset) < 0 # If first test is True, keep correlations
    # indices_to_keep = np.where(first_test)[0]

    # c2_to_subset = c2_to_subset[~first_test]
    # c3_to_subset = c3_to_subset[~first_test]
    # c4_to_subset = c4_to_subset[~first_test]
    # c5_to_subset = c5_to_subset[~first_test]
    # index_tracker = index_tracker[~first_test]

    # second_test = (derivative_function(cut, c2_to_subset, c3_to_subset, c4_to_subset, c5_to_subset) < 0) # If second test is True, do not keep correlations

    # c2_to_subset = c2_to_subset[~second_test]
    # c3_to_subset = c3_to_subset[~second_test]
    # c4_to_subset = c4_to_subset[~second_test]
    # c5_to_subset = c5_to_subset[~second_test]
    # index_tracker = index_tracker[~second_test]

    # third_test = 3*c4_to_subset**2 < 8*c3_to_subset*c5_to_subset # If third test is True, keep correlations
    # indices_to_keep = np.append(indices_to_keep, index_tracker[np.where(third_test)[0]])

    # c2_to_subset = c2_to_subset[~third_test]
    # c3_to_subset = c3_to_subset[~third_test]
    # c4_to_subset = c4_to_subset[~third_test]
    # c5_to_subset = c5_to_subset[~third_test]
    # index_tracker = index_tracker[~third_test]

    # sqrt_inner = 9*c4_to_subset**2-24*c3_to_subset*c5_to_subset
    # sqrt_val = np.sqrt(sqrt_inner)
    # expr1 = (3*c4_to_subset - sqrt_val) / (12*c5_to_subset)
    # expr2 = (3*c4_to_subset + sqrt_val) / (12*c5_to_subset)

    # fourth_test = (
    #     np.logical_and(np.logical_and(-cut < expr1, expr1 < cut) ,
    #     ((45*c4_to_subset**3-36*c3_to_subset*c4_to_subset*c5_to_subset-15*c4_to_subset**2*sqrt_val+8*c5_to_subset*(9*c2_to_subset*c5_to_subset-c3_to_subset*sqrt_val)) < 0
    # ) ) ) # If True, do not keep correlations

    # c2_to_subset = c2_to_subset[~fourth_test]
    # c3_to_subset = c3_to_subset[~fourth_test]
    # c4_to_subset = c4_to_subset[~fourth_test]
    # c5_to_subset = c5_to_subset[~fourth_test]
    # index_tracker = index_tracker[~fourth_test]

    # fifth_test = np.logical_and(
    #     np.logical_and(-cut < expr2,  expr2 < cut), 
    #     ((45*c4_to_subset**3-36*c3_to_subset*c4_to_subset*c5_to_subset+15*c4_to_subset**2*sqrt_val+8*c5_to_subset*(9*c2_to_subset*c5_to_subset+c4_to_subset*sqrt_val))<0) ) # If True, do not keep correlations
    
    # c2_to_subset = c2_to_subset[~fifth_test]
    # c3_to_subset = c3_to_subset[~fifth_test]
    # c4_to_subset = c4_to_subset[~fifth_test]
    # c5_to_subset = c5_to_subset[~fifth_test]
    # index_tracker = index_tracker[~fifth_test]

    # # If any genes still exist in index_tracker, keep

    # indices_to_keep = np.append(indices_to_keep, index_tracker) # If fourth_test is True, do not keep correlations

    indices_passing = np.where(correlations_passing)[0]

    return indices_passing

# Find roots of polynomials for each row
def find_real_root(*coefs):
    '''Find the real root of a polynomial with given coefficients. Considers a root "real" if the imaginary part is smaller than 0.00001. Calculates the absolute value of each root and returns the smallest of these. If there are no real roots, returns NaN.'''
    p = Polynomial([*coefs], domain=[-100, 100])
    complex_roots = p.roots()
    real_roots = complex_roots[np.abs(complex_roots.imag) < 0.00001].real
    # Why return min(abs(root))?
    root = np.min(np.abs(real_roots)) if real_roots.size > 0 else np.nan
    return root

def calculate_mcPCCs_CF_roots(adata, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff, gene_totals, n_jobs = -2, verbose = 1):
    '''This function calculates the roots of the Cornish-Fisher expansion for the given correlations. It first limits the calculation of the roots to correlations that pass multiple tests. It then calculates the roots for each correlation that passed using its cumulants.'''
    if verbose > 1:
        print("Beginning root finding process for Cornish Fisher.")
    # First passing test
    correlations_to_keep = QuickTest6CF(rows, cols,c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff)

    if verbose > 1:
        print(f"First pruning complete.", flush = True)

    # Second passing test.
    # Test gene totals for threshold. If the total UMIs of a gene > 84, we keep them in all cases. If the total UMIs of a gene ≤ 84, we need to test further.
    ## I am testing each row and column of the correlations (remember that we flattened the correlation matrix to 1D). It's very redundant but it's still really fast.
    test_array = np.array([gene_totals[row] for row in rows])
    to_test_bool_rows = test_array <= 84

    test_array = np.array([gene_totals[col] for col in cols])
    to_test_bool_cols = test_array <= 84
    
    to_test_further = np.logical_or(to_test_bool_rows, to_test_bool_cols)

    # If gene total > 84, we keep for root finding
    indices_to_keep = np.where(~to_test_further & correlations_to_keep)[0]

    # If gene total ≤ 84, we need to test further
    indices_to_test_further = np.where(to_test_further & correlations_to_keep)[0]
    c2_lower_flat_pruned_1_more_testing = c2_lower_flat[indices_to_test_further]
    c3_lower_flat_pruned_1_more_testing = c3_lower_flat[indices_to_test_further]
    c4_lower_flat_pruned_1_more_testing = c4_lower_flat[indices_to_test_further]
    c5_lower_flat_pruned_1_more_testing = c5_lower_flat[indices_to_test_further]

    # Third passing test
    # Function is correct
    indices_passing = SecondTestCF(c2_lower_flat_pruned_1_more_testing, c3_lower_flat_pruned_1_more_testing, c4_lower_flat_pruned_1_more_testing, c5_lower_flat_pruned_1_more_testing, first_pass_cutoff, n_jobs=n_jobs)

    indices_to_keep = np.unique(np.append(indices_to_keep, indices_passing))

    if verbose > 1:
        n_correlations_removed = rows.shape[0] - indices_to_keep.shape[0]
        print(f"Second pruning complete. In total, removed {n_correlations_removed} ({np.round(n_correlations_removed/rows.shape[0],3)*100}%) correlations. {indices_to_keep.shape[0]} correlations remain.", flush=True)

    c1_lower_flat_to_keep = c1_lower_flat[indices_to_keep]
    c2_lower_flat_to_keep = c2_lower_flat[indices_to_keep]
    c3_lower_flat_to_keep = c3_lower_flat[indices_to_keep]
    c4_lower_flat_to_keep = c4_lower_flat[indices_to_keep]
    c5_lower_flat_to_keep = c5_lower_flat[indices_to_keep]

    rows_to_keep = rows[indices_to_keep]
    cols_to_keep = cols[indices_to_keep]

    if verbose > 1:
        print("Beginning root finding.", flush=True)
    tic = time.perf_counter()
    # Find roots is correct
    correlation_roots = np.array(Parallel(n_jobs=n_jobs)(delayed(find_real_root)(c1_lower_flat_to_keep[correlation_row], c2_lower_flat_to_keep[correlation_row], c3_lower_flat_to_keep[correlation_row], c4_lower_flat_to_keep[correlation_row], c5_lower_flat_to_keep[correlation_row]) for correlation_row in range(c1_lower_flat_to_keep.shape[0])))

    indices_of_not_found_roots = np.where(np.isnan(correlation_roots))[0]

    # If no real roots are found, find the roots of the derivatives and use those.
    if indices_of_not_found_roots.shape[0] != 0:
        derivative_roots_of_not_initially_found_roots = np.array(Parallel(n_jobs=n_jobs)(delayed(find_real_root)(2*c2_lower_flat_to_keep[correlation_row], 3*c3_lower_flat_to_keep[correlation_row], 4*c4_lower_flat_to_keep[correlation_row], 5*c5_lower_flat_to_keep[correlation_row]) for correlation_row in indices_of_not_found_roots))
        correlation_roots[indices_of_not_found_roots] = derivative_roots_of_not_initially_found_roots
    toc = time.perf_counter()
    if verbose > 1:
        print(f"Root finding complete, took {toc - tic:0.4f} seconds.")
    return rows_to_keep, cols_to_keep, correlation_roots
def calculate_pvalues(correlation_roots):
    print("Estimating p-values.")
    # Calculate log p-values for column 3, two-sided
    p = norm.logcdf(correlation_roots)
    p_mpfr = np.empty(correlation_roots.shape[0])
    for row in range(correlation_roots.shape[0]):
        abs_root = correlation_roots[row]
        log_p = p[row]
        if abs_root < 8.2:
            val = -np.log10(1 - np.exp(log_p))
        elif abs_root >= 38.4:
            mp_val = mpmath.nstr(-mpmath.log10(0.5 * mpmath.exp(- (abs_root ** 2) / 2)), 15)
            val = float(mp_val)
        else:
            val = -np.log10(-log_p / np.log(10))
        p_mpfr[row] = val
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

def calculate_correlations(adata, layer, verbose = 1, cv = None, write_out = None, previously_run = False, return_residuals = False, n_jobs = -2):
    '''Calculate modified Pearson correlation coefficients (mcPCCs) for each gene pair in the dataset.'''

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
    cv, normlist, residuals, n_cells, e_mat = calculate_residuals(cv, raw_count_mat, g_counts)
    # Calculate mcfanos from residuals
    mc_fanos = calculate_mcfano(residuals, n_cells)

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Fano factors for {mc_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    if write_out is not None:
        adata_var = adata.var.copy()
        adata_var['mc_Fano'] = np.array(mc_fanos).flatten()
        adata_var.to_csv(write_out + 'mc_Fano.csv')
    else:
        adata.var["mc_Fano"] = np.array(mc_fanos).flatten()
    adata.uns['CV_for_mc_Fano_fit'] = cv

    # Calculate mcPCCs
    tic = time.perf_counter()
    mcPCCs = calculate_mcPCCs(n_cells, mc_fanos, residuals)
    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Pearson correlation coefficients for {mc_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    del mc_fanos

    # Calculate inverse mcFano moments
    tic = time.perf_counter()
    e_moments = inverse_sqrt_mcfano_correction(n_cells, g_counts, cv, normlist) # These functions are correct

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    tic = time.perf_counter()
    kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv) # These functions are correct
    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating cumulants for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store
    if write_out is not None:
        np.savez_compressed(write_out + 'residuals.npz', residuals)
        np.savez_compressed(write_out + 'cumulants.npz', kappa2=kappa2, kappa3=kappa3, kappa4=kappa4, kappa5=kappa5)
        np.savez_compressed(write_out + 'e_mat.npz', e_mat)
    else:
        if return_residuals == True:
            adata.layers["residuals"] = residuals

    del residuals, e_moments, e_mat

    tic = time.perf_counter()
    rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat = calculate_mcPCCs_coefficients(kappa2, kappa3, kappa4, kappa5, mcPCCs) # These functions are correct
    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating coefficients for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    if write_out is not None:
        np.savez_compressed(write_out + 'mcPCCs.npz', mcPCCs)
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
        print(f"Finished calculating p-values for {correlation_roots.shape[0]} genes in {(toc-tic):04f} seconds.")
    BH_corrected_pvalues = BH_correction(correlation_pvalues, adata.shape[1])

    # Reshape everything into sparse matrix
    BH_corrected_pvalues_matrix = csr_matrix((BH_corrected_pvalues, (rows_to_keep, cols_to_keep)), shape=(g_counts.shape[0], g_counts.shape[0]))

    # Make new empty matrix
    matrix_reconstructed = np.ones((g_counts.shape[0], g_counts.shape[0]))
    matrix_reconstructed[rows_to_keep, cols_to_keep] = BH_corrected_pvalues
    matrix_reconstructed_lower_triangular = np.tril(matrix_reconstructed, -1)
    matrix_reconstructed_lower_triangular_sparse = csr_matrix(matrix_reconstructed_lower_triangular)

    if write_out is not None:
        save_npz(write_out + 'BH_corrected_pvalues.npz', matrix_reconstructed_lower_triangular_sparse)
    else:
        adata.varm["BH-corrected p-values of mcPCCs"] = BH_corrected_pvalues_matrix