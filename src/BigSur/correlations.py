#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Iterable
import time
import numpy as np
import mpmath
import numexpr as ne
import warnings
import pandas as pd

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

from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)

def calculate_mcPCCs(n_cells, mc_fanos, residuals):
    mcPCCs = 1/((n_cells - 1) * np.sqrt(mc_fanos * mc_fanos.T)) * (residuals.T @ residuals)
    return mcPCCs

# Functions for inverse square moment interpolation
def inv_sqrt_moment_interpolation(sample_moments, gene_totals, points):
    moments_mat = np.array(sample_moments).reshape(-1, 4) # I may not need sample_moments to be reshaped
    int_moments = []
    for j in range(4):
        approx_func = interp1d(
            np.log10(points),
            np.log10(moments_mat[:, j]),
            kind='linear',
            fill_value='extrapolate'
        )
        interpolated = np.power(10, approx_func(np.log10(gene_totals)))
        int_moments.append(interpolated)
    e_moments = [np.outer(m, m) for m in int_moments]
    return e_moments
def inverse_sqrt_mcfano_correction(n_cells, g_counts, c, normlist):
    a = max(2, min(g_counts))
    e = n_cells / 50
    h = max(g_counts)
    points = np.array([a, a * (e / a) ** (1 / 4), a * (e / a) ** (1 / 2), a * (e / a) ** (3 / 4), e, e * (h / e) ** (1 / 3), e * (h / e) ** (2 / 3), h], dtype=int) # 8 points
    trials = 4*10**7/(n_cells*(np.log10(points)**(1/5)+0.5*np.log10(points)**3)) # should be ints
    trials = trials.astype(int)

    simemat = np.outer(points, normlist)

    sample_moments = []

    for i in range(points.shape[0]):
        sample_moments.append(simulate_inverse_sqrt_mcfano_moments(simemat[i,:], c, n_cells, trials[i]))

    e_moments = inv_sqrt_moment_interpolation(sample_moments, g_counts, points)
    return e_moments
def simulate_inverse_sqrt_mcfano_moments(simemat_subset, c, n_cells, trial, starting_seed = 0):
    samples = np.repeat(0, trial)
    x = simemat_subset
    mu = np.log(x / np.sqrt(1 + c**2))
    sigma = np.sqrt(np.log(1 + c**2))

    rng = np.random.default_rng(starting_seed)

    for i in range(trial):
        PLN_samples = rng.poisson(rng.lognormal(mean=mu, sigma=sigma))
        sample = 1/np.sqrt(np.sum((PLN_samples-x)**2/(x+c**2*x**2))/(n_cells-1))
        samples[i] = sample

    # Above for loop should be equivalent to:
    PLN_samples = rng.poisson(rng.lognormal(mean=mu, sigma=sigma), size = (n_cells, trial))
    samples = 1/np.sqrt(np.sum((PLN_samples-x)**2/(x+c**2*x**2))/(n_cells-1))

    results = [np.mean(samples**n) for n in range(2, 6)] # Return the first 5 moments

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
    k5_matrix_1 = (one_plus_cv_squared_times_emat*(3+cv**2*(3+cv**2)*e_mat))/(np.sqrt(e_mat)*(one_plus_cv_squared_times_emat)**(3/2))
    k5_matrix_2 = (
        1/(e_mat**(3/2)*(one_plus_cv_squared_times_emat)**(5/2)) *
        (1 +
         5*(2+3*cv**2)*e_mat +
         5*cv**2*(8+15*cv**2+5*cv**4)*e_mat**2 +
         10*cv**4*(6+17*cv**2+15*cv**4+6*cv**6+cv**8)*e_mat**3 +
         cv**6*(30+135*cv**2+222*cv**4+205*cv**6+
               120*cv**8+45*cv**10+10*cv**12+cv**14)*e_mat**4)
    )

    k3_crossprod = np.matmul(k3_matrix, k3_matrix.T)
    k4_crossprod = np.matmul(k4_matrix, k4_matrix.T)
    k5_crossprod_1 = np.matmul(k5_matrix_1, k5_matrix_1.T)
    k5_crossprod_2 = np.matmul(k5_matrix_2, k5_matrix_2.T)

    kappa2 = (1/(n_cells-1)**2) * f2 * n_cells
    kappa3 = (1/(n_cells-1)**3) * f3 * k3_crossprod
    kappa4 = (1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)
    kappa5 = (1/(n_cells-1)**5) * (-10*f2*f3*k5_crossprod_1 + f5*k5_crossprod_2)

    k_list = np.hstack([kappa2, kappa3, kappa4, kappa5]) # Maybe needs to be vstack
    return k_list

def calculate_mcPCCs_coefficients(k_list, mcPCCs):
    k2 = k_list[0]
    k3 = k_list[1]
    k4 = k_list[2]
    k5 = k_list[3]

    c1 = -mcPCCs - k3/(6*k2) + 17*k3**3/(324*k2**4) - k3*k4/(12*k2**3) + k5/(40*k2**2)
    c2 = np.sqrt(k2) + 5*k3**2/(36*k2**(5/2)) - k4/(8*k2**(3/2))
    c3 = k3/(6*k2) - 53*k3**3/(324*k2**4) + 5*k3*k4/(24*k2**3) - k5/(20*k2**2)
    c4 = -k3**2/(18*k2**(5/2)) + k4/(24*k2**(3/2))
    c5 = k3**3/(27*k2**4) - k3*k4/(24*k2**3) + k5/(120*k2**2)

    clist = np.hstack([c1, c2, c3, c4, c5])
    mcPCCs_length = clist[0].shape[0]
    z = np.arange(1, mcPCCs_length+1)

    # Generate row/col indices for lower triangle
    row = np.concatenate([np.arange(x, mcPCCs_length+1) for x in range(2, mcPCCs_length+1)])
    col = np.repeat(z[:-1], np.arange(mcPCCs_length, 1, -1) - 1)

    def lower_tri_to_1d(matrix):
        return matrix[np.tril_indices(mcPCCs_length, -1)]

    c1v = lower_tri_to_1d(clist[0])
    c2v = lower_tri_to_1d(clist[1])
    c3v = lower_tri_to_1d(clist[2])
    c4v = lower_tri_to_1d(clist[3])
    c5v = lower_tri_to_1d(clist[4])

    cmatrix = np.column_stack((row, col, c1v, c2v, c3v, c4v, c5v)) 
    return cmatrix

def QuickTest6CF(cmatrix, first_pass_cutoff):
    cut = np.sqrt(2) * erfcinv(2 * 10 ** -first_pass_cutoff)
    
    def testfunc_1(x, cmatrix):
        return cmatrix[:, 2] + cmatrix[:, 3] * x + cmatrix[:, 4] * x**2 + cmatrix[:, 5] * x**3 + cmatrix[:, 6] * x**4

    def testfunc_2(x, cmatrix):
        return cmatrix[:, 2] * testfunc_1(x, cmatrix)

    # Compute cut.vec for both pos and neg cut
    cut_vec_1 = testfunc_1(cut, cmatrix)
    cut_vec_2 = testfunc_1(-cut, cmatrix)
    cut_bool = ~(cut_vec_1 * cut_vec_2  < 0) # return False if both are negative
    cmatrix = cmatrix[cut_bool]

    cut_vec2 = testfunc_2(cut, cmatrix)
    cut_bool2 = cut_vec2 >= 0
    cmatrix = cmatrix[cut_bool2]

    return cmatrix
def SecondTestCF(cmatrix_more_testing, first_pass_cutoff):
    cut = np.sqrt(2) * erfcinv(2 * 10**(-first_pass_cutoff))

    def derivative(x, cmatrix_subset):
        return cmatrix_subset[3] + 2*cmatrix_subset[4]*x + 3*cmatrix_subset[5]*x**2 + 4*cmatrix_subset[6]*x**3

    def test_conditions(cmatrix_subset):
        if derivative(-cut, cmatrix_subset) < 0:
            return True
        elif derivative(cut, cmatrix_subset) < 0:
            return False
        elif 3*cmatrix_subset[5]**2 < 8*cmatrix_subset[4]*cmatrix_subset[6]:
            return True
        else:
            sqrt_inner = 9*cmatrix_subset[5]**2-24*cmatrix_subset[4]*cmatrix_subset[6]
            sqrt_val = np.sqrt(sqrt_inner)
            expr1 = (3*cmatrix_subset[5] - sqrt_val) / (12*cmatrix_subset[6])
            expr2 = (3*cmatrix_subset[5] + sqrt_val) / (12*cmatrix_subset[6])

            term1 = (
                -cut < expr1 < cut and
                ((45*cmatrix_subset[5]**3-36*cmatrix_subset[4]*cmatrix_subset[5]*cmatrix_subset[6]-15*cmatrix_subset[5]**2*sqrt_val+8*cmatrix_subset[6]*(9*cmatrix_subset[3]*cmatrix_subset[6]-cmatrix_subset[4]*sqrt_val)) < 0
            ) )
            term2 = (
                -cut < expr2 < cut and
                ((45*cmatrix_subset[5]**3-36*cmatrix_subset[4]*cmatrix_subset[5]*cmatrix_subset[6]+15*cmatrix_subset[5]**2*sqrt_val+8*cmatrix_subset[6]*(9*cmatrix_subset[3]*cmatrix_subset[6]+cmatrix_subset[4]*sqrt_val))<0) )

            if term1:
                return False
            elif term2:
                return False
            else:
                return True

    # Apply test_conditions row-wise
    mask = np.apply_along_axis(test_conditions, 0, cmatrix_more_testing) # Should be row-wise; needs to be vectorized
    cmatrix_passed = cmatrix_more_testing[mask]
    return cmatrix_passed

# Find roots of polynomials for each row
def find_real_root(coefs):
    p = Polynomial(coefs)
    complex_roots = p.roots()
    real_roots = complex_roots[np.isreal(complex_roots)].real
    return np.min(np.abs(real_roots)) if real_roots.size > 0 else np.nan

def calculate_mcPCCs_CF_roots(correlation_coefficients, first_pass_cutoff, gene_totals):
    print("Beginning root finding process for Cornish Fisher.")
    cmatrix_pruned_1 = QuickTest6CF(correlation_coefficients, first_pass_cutoff)
    print(f"First pruning complete. Removed {correlation_coefficients.shape[0] - cmatrix_pruned_1.shape[0]} insignificant correlations.")

    # Test gene totals for threshold
    to_test_bool = np.array([
        True if (gene_totals[int(row[0])] <= 84 or gene_totals[int(row[1])] <= 84) else False 
        for row in cmatrix_pruned_1
    ]) # Remove gene_totals that smaller than 84?

    cmatrix_to_reject = np.column_stack([cmatrix_pruned_1, to_test_bool])

    cmatrix_to_keep = cmatrix_pruned_1[~cmatrix_to_reject[:, -1].astype(bool)]
    cmatrix_more_testing = cmatrix_pruned_1[cmatrix_to_reject[:, -1].astype(bool)]
    
    cmatrix_passed = SecondTestCF(cmatrix_more_testing, first_pass_cutoff)
    cmatrix_to_keep = np.vstack([cmatrix_to_keep, cmatrix_passed])
    
    print(f"Second pruning complete. {cmatrix_to_keep.shape} correlations remain.")
    print("Beginning root finding.")
    
    roots = np.array([find_real_root(row[2:7]) for row in cmatrix_to_keep])
    cmatrix_to_keep = np.column_stack([cmatrix_to_keep, roots])
    found_roots = cmatrix_to_keep[~np.isnan(cmatrix_to_keep[:, 9])]
    unfound_roots = cmatrix_to_keep[np.isnan(cmatrix_to_keep[:, 9])]

    if unfound_roots.shape == 0:
        roots_matrix = found_roots[:, [0,1,9]]
    elif unfound_roots.shape == 1:
        single_d_coefs = [
            unfound_roots[0,3],
            2*unfound_roots[0,4],
            3*unfound_roots[0,5],
            4*unfound_roots[0,6]
        ]
        single_d_root = find_real_root(single_d_coefs)
        roots_matrix = np.vstack([found_roots[:, [0,1,9]], [unfound_roots[0,0], unfound_roots[0,1], single_d_root]])
    else:
        d_coefs = np.column_stack([
            unfound_roots[:,3],
            2*unfound_roots[:,4],
            3*unfound_roots[:,5],
            4*unfound_roots[:,6]
        ])
        d_roots = np.array([find_real_root(row) for row in d_coefs])
        unfound_roots = np.column_stack([unfound_roots, d_roots])
        roots_matrix = np.vstack([found_roots[:, [0,1,9]], unfound_roots[:, [0,1,10]]])

    print("Root finding complete.")
    return roots_matrix
def calculate_pvalues(correlation_roots):
    print("Estimating p-values.")
    # Calculate log p-values for column 3, two-sided
    p = norm.logcdf(np.abs(correlation_roots[:, 2]))
    # Combine roots_matrix and p
    p_matrix = np.hstack((correlation_roots, p.reshape(-1, 1)))
    p_mpfr = []
    for x in p_matrix:
        abs_root = abs(x[2])
        log_p = x[3]
        if abs_root < 8.2:
            val = -np.log10(1 - np.exp(log_p))
        elif abs_root >= 38.4:
            mp_val = mpmath.nstr(-np.log10(0.5 * mpmath.exp(- (abs_root ** 2) / 2)), 15)
            val = float(mp_val)
        else:
            val = -np.log10(-log_p / np.log(10))
        p_mpfr.append(val)
    p_matrix = np.hstack((p_matrix, np.array(p_mpfr).reshape(-1, 1)))
    print("P-value estimation complete.")
    return p_mpfr#p_matrix
def BH_correction(p_mpfr, num_genes):
    sorted_pvals = np.sort(p_mpfr) # greatest to smallest?
    BH_corrected_pvalues = (10**-sorted_pvals) * ((num_genes * (num_genes - 1)) / 2) / np.arange(1, len(sorted_pvals) + 1)
    return BH_corrected_pvalues

def calculate_correlations(adata, layer, verbose = 1, cv = None):
    '''Calculate modified Pearson correlation coefficients (mcPCCs) for each gene pair in the dataset.'''

    # Setup        
    # Create variables
    raw_count_mat, means, variances, g_counts = make_vars_and_qc(adata, layer)

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
    adata.var["mc_Fano"] = np.array(mc_fanos).flatten()
    adata.uns['CV_for_mc_Fano_fit'] = cv

    # Calculate mcPCCs
    mcPCCs = calculate_mcPCCs(n_cells, mc_fanos, residuals)
    adata.varm['mcPCCs'] = mcPCCs

    # Calculate inverse mcFano moments
    e_moments = inverse_sqrt_mcfano_correction(n_cells, g_counts, cv, normlist)

    correlation_cumulants = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
    correlation_coefficients = calculate_mcPCCs_coefficients(correlation_cumulants, mcPCCs)
    correlation_roots = calculate_mcPCCs_CF_roots(correlation_coefficients, 2, g_counts)

    correlation_pvalues = calculate_pvalues(correlation_roots)

    BH_corrected_pvalues = BH_correction(correlation_pvalues, adata.shape[1])

    return adata