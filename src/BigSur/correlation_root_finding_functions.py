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

def find_passing_correlations_1(rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff):
    '''This function tests correlations to see if they pass a series of tests. It returns a boolean array of the correlations that pass all tests.'''
    cut = np.sqrt(2) * erfcinv(2 * 10 ** -first_pass_cutoff)

    def testfunc_1(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat):
        return c1_lower_flat + c2_lower_flat * x + c3_lower_flat * x**2 + c4_lower_flat * x**3 + c5_lower_flat * x**4

    def testfunc_2(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat):
        return c1_lower_flat * testfunc_1(x, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)

    # Compute cut.vec for both pos and neg cut
    cut_vec_1 = testfunc_1(cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_vec_2 = testfunc_1(-cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_bool = ~(cut_vec_1 * cut_vec_2 < 0)# return False if both are negative
    
    cut_vec2 = testfunc_2(cut, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat)
    cut_bool2 = ~(cut_vec2 < 0) # return False if cut_vec2 is negative

    indices_to_keep = np.logical_and(cut_bool, cut_bool2) # keep correlations if both cut_bool 1 and cut_bool 2 are True
    return indices_to_keep

def find_passing_correlations_2(c2, c3, c4, c5, first_pass_cutoff):
    '''This function tests correlations of genes that have less than or equal to 84 total UMIs (for either gene). Certain of these correlations will not be significant. This function identifies these correlations and removes them before root finding.'''
    
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
    # correlations_passing = np.array(Parallel(n_jobs=n_jobs)(delayed(test_conditions)(c2[correlation_row], c3[correlation_row], c4[correlation_row], c5[correlation_row]) for correlation_row in range(c2.shape[0])))
    # toc = time.perf_counter()

    # indices_passing = np.where(correlations_passing)[0]

    # index_tracker is the elements of the flattened coefficients
    index_tracker = np.array(list(range(c2.shape[0])))

    # c2, c3, c4, c5 =
    #x[4], x[5], x[6], x[7]

    c2_to_subset = c2.copy()
    c3_to_subset = c3.copy()
    c4_to_subset = c4.copy()
    c5_to_subset = c5.copy()

    first_test = derivative_function(-cut, c2_to_subset, c3_to_subset, c4_to_subset, c5_to_subset) < 0 # If first test is True, keep correlations
    indices_to_keep = np.where(first_test)[0]

    c2_to_subset = c2_to_subset[~first_test]
    c3_to_subset = c3_to_subset[~first_test]
    c4_to_subset = c4_to_subset[~first_test]
    c5_to_subset = c5_to_subset[~first_test]
    index_tracker = index_tracker[~first_test]

    second_test = derivative_function(cut, c2_to_subset, c3_to_subset, c4_to_subset, c5_to_subset) < 0 # If second test is True, do not keep correlations

    c2_to_subset = c2_to_subset[~second_test]
    c3_to_subset = c3_to_subset[~second_test]
    c4_to_subset = c4_to_subset[~second_test]
    c5_to_subset = c5_to_subset[~second_test]
    index_tracker = index_tracker[~second_test]

    third_test = 3*c4_to_subset**2 < 8*c3_to_subset*c5_to_subset # If third test is True, keep correlations
    indices_to_keep = np.append(indices_to_keep, index_tracker[np.where(third_test)[0]])

    c2_to_subset = c2_to_subset[~third_test]
    c3_to_subset = c3_to_subset[~third_test]
    c4_to_subset = c4_to_subset[~third_test]
    c5_to_subset = c5_to_subset[~third_test]
    index_tracker = index_tracker[~third_test]

    sqrt_inner = 9*c4_to_subset**2-24*c3_to_subset*c5_to_subset
    sqrt_val = np.sqrt(sqrt_inner)
    expr1 = (3*c4_to_subset - sqrt_val) / (12*c5_to_subset)

    # First expression is correct
    fourth_test = (
        np.logical_and(np.logical_and(-cut < expr1, expr1 < cut) ,
        ((45*c4_to_subset**3-36*c3_to_subset*c4_to_subset*c5_to_subset-15*c4_to_subset**2*sqrt_val+8*c5_to_subset*(9*c2_to_subset*c5_to_subset-c3_to_subset*sqrt_val)) < 0
    ) ) ) # If True, do not keep correlations

    c2_to_subset = c2_to_subset[~fourth_test]
    c3_to_subset = c3_to_subset[~fourth_test]
    c4_to_subset = c4_to_subset[~fourth_test]
    c5_to_subset = c5_to_subset[~fourth_test]
    index_tracker = index_tracker[~fourth_test]

    

    sqrt_inner = 9*c4_to_subset**2-24*c3_to_subset*c5_to_subset
    sqrt_val = np.sqrt(sqrt_inner)
    expr2 = (3*c4_to_subset+sqrt_val)/(12*c5_to_subset)

    fifth_test = np.logical_and(
        np.logical_and(-cut < expr2,  expr2 < cut), 
        ((45*c4_to_subset**3-36*c3_to_subset*c4_to_subset*c5_to_subset+15*c4_to_subset**2*sqrt_val+8*c5_to_subset*(9*c2_to_subset*c5_to_subset+c3_to_subset*sqrt_val))<0) ) # If True, do not keep correlations
    
    c2_to_subset = c2_to_subset[~fifth_test]
    c3_to_subset = c3_to_subset[~fifth_test]
    c4_to_subset = c4_to_subset[~fifth_test]
    c5_to_subset = c5_to_subset[~fifth_test]
    index_tracker = index_tracker[~fifth_test]

    # If any genes still exist in index_tracker, keep

    indices_to_keep = np.append(indices_to_keep, index_tracker) # If fourth_test is True, do not keep correlations

    return indices_to_keep

# Find roots of polynomials for each row
def find_real_root(*coefs):
    '''Find the real root of a polynomial with given coefficients. Considers a root "real" if the imaginary part is smaller than 0.00001. Calculates the absolute value of each root and returns the smallest of these. If there are no real roots, returns NaN.'''
    p = Polynomial([*coefs], domain=[-3, 3])
    complex_roots = p.roots()
    real_roots = complex_roots[np.abs(complex_roots.imag) < 0.00001].real
    root = np.min(np.abs(real_roots)) if real_roots.size > 0 else np.nan
    return root

def calculate_mcPCCs_CF_roots(adata, rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff, gene_totals, n_jobs = -2, verbose = 1):
    '''This function calculates the roots of the Cornish-Fisher expansion for the given correlations. It first limits the calculation of the roots to correlations that pass multiple tests. It then calculates the roots for each correlation that passed using its cumulants.'''
    if verbose > 1:
        print("Beginning root finding process for Cornish Fisher.")
    # First passing test
    indices_to_keep = find_passing_correlations_1(rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat, first_pass_cutoff)

    # # Testing block, delete me
    # np.savez_compressed('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/first_pass_sparse.npz', rows=rows, cols=cols, indices_to_keep=indices_to_keep)
    # # Testing block, delete me

    if verbose > 1:
        print(f"First pruning complete.", flush = True)

    # Second passing test.
    # Test gene totals for threshold. If the total UMIs of a gene > 84, we keep them for root finding. If the total UMIs of a gene ≤ 84, we need to test further.
    ## I am testing each row and column of the correlations (remember that we flattened the correlation matrix to 1D). It's very redundant but it's still really fast.
    indices_to_keep, indices_to_test_further = find_total_umis_of_genes(rows, cols, gene_totals, indices_to_keep)

    c2_lower_flat_pruned_1_more_testing = c2_lower_flat[indices_to_test_further]
    c3_lower_flat_pruned_1_more_testing = c3_lower_flat[indices_to_test_further]
    c4_lower_flat_pruned_1_more_testing = c4_lower_flat[indices_to_test_further]
    c5_lower_flat_pruned_1_more_testing = c5_lower_flat[indices_to_test_further]

    # Third passing test
    # Function is correct
    indices_passing = find_passing_correlations_2(c2_lower_flat_pruned_1_more_testing, c3_lower_flat_pruned_1_more_testing, c4_lower_flat_pruned_1_more_testing, c5_lower_flat_pruned_1_more_testing, first_pass_cutoff)

    # # Testing block, delete me
    # np.savez_compressed('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/indices_passing_SecondTestCF.npz', rows=rows[indices_passing], cols=cols[indices_passing], indices_to_keep=indices_passing)
    # # Testing block, delete me

    indices_to_keep = np.unique(np.append(indices_to_keep, indices_passing))

    if verbose > 1:
        n_correlations_removed = rows.shape[0] - indices_to_keep.shape[0]
        print(f"Second pruning complete. In total, removed {n_correlations_removed} ({np.round(n_correlations_removed/rows.shape[0], 3)*100:.1f}%) correlations. {indices_to_keep.shape[0]} correlations remain.", flush=True)

    c1_lower_flat_to_keep = c1_lower_flat[indices_to_keep]
    c2_lower_flat_to_keep = c2_lower_flat[indices_to_keep]
    c3_lower_flat_to_keep = c3_lower_flat[indices_to_keep]
    c4_lower_flat_to_keep = c4_lower_flat[indices_to_keep]
    c5_lower_flat_to_keep = c5_lower_flat[indices_to_keep]

    rows_to_keep = rows[indices_to_keep]
    cols_to_keep = cols[indices_to_keep]

    # # Testing block, delete me
    # np.savez_compressed('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/indices_passing_final_sparse.npz', rows=rows_to_keep, cols=cols_to_keep, indices_to_keep=indices_to_keep)
    # # Testing block, delete me

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
        print(f"Root finding complete, took {toc - tic:0.4f} seconds.") # old domain, took 27.92s

    # # Testing block, delete me
    # roots_matrix  = np.empty((gene_totals.shape[0], gene_totals.shape[0]))
    # roots_matrix[rows_to_keep, cols_to_keep] = correlation_roots
    # roots_matrix_sparse = csr_matrix(roots_matrix)
    # save_npz('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/roots_matrix_sparse.npz', roots_matrix_sparse)
    # # Testing block, delete me
    
    return rows_to_keep, cols_to_keep, correlation_roots

def find_total_umis_of_genes(rows, cols, gene_totals, correlations_to_keep):
    '''This function tests the total UMIs of genes in correlations that passed the first test. If either gene in a correlation has total UMIs ≤ 84, that correlation may not be significant. We test these correlations further in find_passing_correlations_2. If both genes have total UMIs > 84, the correlation is kept for root finding.'''
    test_array_rows = gene_totals[rows]
    test_array_cols = gene_totals[cols]

    to_test_bool_rows = test_array_rows <= 84
    to_test_bool_cols = test_array_cols <= 84
    
    to_test_further = np.logical_or(to_test_bool_rows, to_test_bool_cols)

    # # Testing block, delete me
    # np.savez_compressed('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/smaller_or_equal_to_84_sparse.npz', rows=rows, cols=cols, to_test_further=to_test_further)
    # # Testing block, delete me

    # If gene total > 84, we keep for root finding
    indices_to_keep = np.where(~to_test_further & correlations_to_keep)[0]

    # If gene total ≤ 84, we need to test further
    indices_to_test_further = np.where(to_test_further & correlations_to_keep)[0]
    return indices_to_keep, indices_to_test_further