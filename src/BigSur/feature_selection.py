#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:55:15 2022

@author: emmanueldollinger
"""
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


def mcfano_feature_selection(
    adata: AnnData,
    layer: str,
    cv: Union[bool, float, dict] = None,
    n_genes_for_PCA: Union[bool, int] = False,
    min_mcfano_cutoff: Union[bool, float] = 0.9,
    p_val_cutoff: Union[bool, float] = 0.05,
    return_residuals: bool = False,
    batch_key: str = None,
    n_jobs: int = -2,
    verbose: int = 1,
):
    """
    Calculate the modified corrected Fano factor for all genes in the dataset. mc_Fano column will be added to .var. Highly_variable column will be added to .var based on the n_genes_for_pca, min_mcfano_cutoff and p_val_cutoff parameters.

    Parameters
    ----------
    adata - adata object containing information about the raw counts and gene names.
    layer - String, describing the layer of adata object containing raw counts (pass "X" if raw counts are in adata.X).
    cv - Float or dict, coefficient of variation for the dataset or batches. If dict, keys should be batch names and values should be the corresponding CVs. If None, the CV(s) will be estimated.
    n_genes_for_PCA - [Int, Bool], top number of genes to use for highly_variable slot, ranked by modified corrected Fano factor and filtered by p-value cutoff. mcFano factor cutoff and p-value cutoff will first be calculated. If n_genes_for_PCA is greater than the genes passing both mcFano and p-value cutoff, the function will throw a warning and only use the genes meeting the cutoff. If False, default to combination of min_mcfano_cutoff and/or p_val_cutoff.
    min_mcfano_cutoff - Union[bool, float], Only include modified corrected Fano factors greater than min_mcfano_cutoff quantile and that meet the p-value cutoff in the highly_variable column. If verbose = 2, we suggest a quantile cutoff based on the statistics of the dataset.
    p_val_cutoff - [Bool, Float], if a float value is provided, that p-value cutoff will be used to select genes. If False, default to combination of min_mcfano_cutoff and/or n_genes_for_PCA.
    return_residuals - Bool, if True, the function will return a matrix containing the calculated mean-centered corrected Pearson residuals matrix stored in adata.layers['residuals'].
    batch_key - String or None. Specifies the batch key to use for batch-specific expectation matrix calculating and CV fitting.
    n_jobs - Int, how many cores to use for p-value parallelization. Default is -2 (all but 1 core).
    verbose - Int, whether to print computations and top 100 genes. 0 is no verbose, 1 is a little (what the function is doing) and 2 is full verbose.
    
    """
    # TO DO:
    # An assumption made throughout is each batch has the same genes. Create a function that checks this.

    # Create variables
    batch_dict = make_vars_and_qc(adata, layer, batch_key = batch_key)

    # Print cutoff parameters
    determine_cutoff_parameters(n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose)
        
    tic = time.perf_counter()
    # Fit cv if not provided
    if cv is None:
        if verbose > 1:
            print('Fitting cv.')
            fit_cv(batch_dict, verbose)
    elif isinstance(cv, float):
        if batch_key is not None:
            raise Exception("Batch key was provided, but cv only has one value. Please pass cv as a dict of {batch:cv} pairs.")
        batch_dict['All']['CV'] = cv
    elif isinstance(cv, dict):
        if not cv.keys() <= batch_dict.keys():
            raise Exception("If providing CV as a dict, keys must match batch names.")
        for batch in cv:
            batch_dict[batch]['CV'] = cv[batch]
            
    if verbose > 1:
        for batch in batch_dict:
            # If batch is 'All' and there's more than one batch, skip it
            if (len(batch_dict) > 1) and (batch == 'All'):
                continue
            # If batch is 'All' and there's only one batch, print its CV
            elif (batch == 'All') and (len(batch_dict) == 1):
                print(f"Using a coefficient of variation of {batch_dict[batch]['CV']:.4}.")
            # If batch is not 'All', print its CV
            else:
                print(f"Using a coefficient of variation of {batch_dict[batch]['CV']:.4} for batch {batch}.")
    # Calculate residuals
    calculate_residuals(batch_dict)

    if len(batch_dict.keys()) > 1:
        batch_dict['All']['Residuals'] = np.concatenate([batch_dict[batch]['Residuals'] for batch in batch_dict if batch != 'All'])
    
    # Calculate mcfanos from residuals
    if verbose > 1:
        print("Calculating modified corrected Fano factors.")
    
    calculate_mcfano({'All':batch_dict['All']})

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Fano factors for {batch_dict['All']['mcFanos'].shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    adata.var["mc_Fano"] = np.array(batch_dict['All']['mcFanos']).flatten()

    for batch in batch_dict:
        # If batch is 'All' and there's more than one batch, skip it
        if (len(batch_dict) > 1) and (batch == 'All'):
            continue
        # If batch is 'All' and there's only one batch, print its CV
        elif (batch == 'All') and (len(batch_dict) == 1):
            adata.uns['CV_for_mc_Fano_fit'] = batch_dict['All']['CV']
        else:
            adata.uns[f'CV_for_mc_Fano_fit_{batch}'] = batch_dict[batch]['CV']

    # Calculate p-values
    if isinstance(p_val_cutoff, float):
        tic = time.perf_counter()
        if verbose > 1:
            print("Calculating p-values.")
        p_vals_corrected, p_vals = calculate_p_value(
            batch_dict, p_val_cutoff, n_jobs
        )
        toc = time.perf_counter()
        if verbose > 1:
            print(
                f"Finished calculating p-values in {(toc-tic):04f} seconds."
            )

        # Store p-values
        adata.var["FDR_adj_pvalue"] = p_vals_corrected
        adata.var["p_value"] = p_vals
    else:
        if verbose > 1:
            print("Skipping p-value calculation.")
    
    if verbose == 2:
        suggest_cutoffs(adata, layer)

    genes = determine_HVGs(adata, n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, n_jobs, verbose)
    adata.var["highly_variable"] = False
    adata.var.loc[genes, "highly_variable"] = True
    if verbose > 1:
        if genes.shape[0] <= 100:
            n_genes_to_print = genes.shape[0]
        else:
            n_genes_to_print = 100
        print(
            f"Top {n_genes_to_print} selected genes: \n {np.sort(genes[:n_genes_to_print])}")

    # Returning data
    if return_residuals:
        index_map = {value: idx for idx, value in enumerate(batch_dict['All']['Barcodes'])}
        indices = np.array([index_map[barcode] for barcode in adata.obs.index])
        adata.layers['residuals'] = batch_dict['All']['Residuals'][indices, :]
        

# UX functions
def determine_cutoff_parameters(n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose):
    '''Determine whether using pvals or n top genes or min_fano or combo thereof'''
    is_n_genes = type(n_genes_for_PCA) == int
    is_p_val_cutoff = isinstance(p_val_cutoff, float)
    is_min_fano_cutoff = isinstance(min_mcfano_cutoff, float)

    if not is_n_genes and not is_p_val_cutoff and not is_min_fano_cutoff:
        raise Exception(
            "Please specify either number of top genes or pvalue cutoff or min fano quantile cutoff."
        )

    if verbose >= 1:
        print_string = "Using"
        n_of_ands = 0
        if is_n_genes:
            print_string += f" {n_genes_for_PCA} top genes"
            n_of_ands += 1
        if is_p_val_cutoff:
            if n_of_ands == 1:
                print_string += f" and {p_val_cutoff} for pvalue cutoff"
            else:
                print_string += f" {p_val_cutoff} for pvalue cutoff"
            n_of_ands += 1
        if is_min_fano_cutoff:
            if n_of_ands > 0:
                print_string += f" and {min_mcfano_cutoff} for mcFano quantile cutoff"
            else:
                print_string += f" {min_mcfano_cutoff} for mcFano quantile cutoff"
        print_string += " for highly variable genes."
        print(print_string)

def suggest_cutoffs(adata, layer):
    '''Suggests cutoffs, based on statistics of the dataset. These cutoffs are merely starting points; the optimal cutoff will depend on unknowables.'''
    # First, calculate median UMI/cell and number of cells
    number_of_cells = adata.shape[0]
    median_umi_per_cell = np.median(np.array(adata.layers[layer].sum(axis = 1)).flatten())

    # Now, calculate the percent of mcFanos that are significant
    percent_of_significant_mcfanos = int(np.round(np.sum(adata.var['FDR_adj_pvalue'] < 0.05) / adata.shape[1] * 100, 0))

    # Determine suggested cutoff
    ## If the dataset is shallowly sequenced or if there aren't enough cells, suggest 10%
    if (median_umi_per_cell < 3000) or (number_of_cells < 150):
        suggested_cutoff = ['10%', '0.9']
        dataset_quality = 'poor'
    
    ## If the dataset is sequenced enough, and there are enough cells, consider the percent of mcFanos that are significant
    else:
        dataset_quality = 'high'
        if percent_of_significant_mcfanos > 5:
            suggested_cutoff = ['10%', '0.9']
            
        else:
            suggested_cutoff = ['1%', '0.99']
            
    # Print
    if dataset_quality == 'poor':
        print(f'There are {number_of_cells} cells with a median sequencing depth of {median_umi_per_cell} UMI/cell. These numbers are relatively low; we therefore suggest only selecting the top {suggested_cutoff[0]} of mcFanos that have p-values lower than 0.05. To do so, set min_mcfano_cutoff = {suggested_cutoff[1]}.')
    else:
        print(f'There are {number_of_cells} cells with a median sequencing depth of {median_umi_per_cell} UMI/cell. Since {np.round(percent_of_significant_mcfanos, 2)}% of mcFanos are significant, we suggest selecting the top {suggested_cutoff[0]} of mcFanos that have p-values lower than 0.05. To do so, set min_mcfano_cutoff = {suggested_cutoff[1]}.')


def determine_HVGs(adata, n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, n_jobs, verbose):
    is_n_genes = isinstance(n_genes_for_PCA, bool)
    is_p_val_cutoff = isinstance(p_val_cutoff, float)
    is_min_fano_cutoff = isinstance(min_mcfano_cutoff, float)

    # Store adata for easy filtering
    adata_var_df = adata.var.sort_values('mc_Fano', ascending = False).copy() # Sort from greatest to smallest mcFano; without copy there's a bug

    if is_min_fano_cutoff:
        min_fano = np.quantile(adata_var_df['mc_Fano'], min_mcfano_cutoff)
    else:
        print('Optimization of mcFano using silhouette score is now disabled. Set verbose = 2 and use the suggested cutoffs. Setting the quantile cutoff to 0.9.')
        min_mcfano_cutoff = 0.9
        min_fano = np.quantile(adata_var_df['mc_Fano'], min_mcfano_cutoff)
    
    genes = adata_var_df[adata_var_df['mc_Fano'] > min_fano].index
    
    if is_p_val_cutoff:
        genes = np.intersect1d(genes, adata_var_df[adata_var_df['FDR_adj_pvalue'] < p_val_cutoff].index)
        genes = adata_var_df.loc[genes].sort_values('mc_Fano', ascending = False).index # Reorder

    if not is_n_genes:
        if n_genes_for_PCA > genes.shape[0]:
            warnings.warn(
            f'Number of genes meeting cutoffs ({genes.shape[0]}) is lower than user requested genes ({n_genes_for_PCA}). Only including genes meeting cutoff in "highly_variable" slot. Please increase the min_mcfano_cutoff, decrease the p_val_cutoff, or set either or both to False.')
        else:
            genes = genes[:n_genes_for_PCA] # Should be already sorted

    if verbose > 1:
        print(f'Setting {genes.shape[0]} genes as highly variable.')

    return genes

# p-value functions
## Main p-value function
def calculate_p_value(
    batch_dict, cutoff, n_jobs
):
    """Calculate the p-value for corrected fanos. First calculate cumulants of gene distributions, then calculate the 4th order Cornish Fisher polynomial, then solve for the root of the CF."""

    k2, k3, k4, k5 = find_cumulants(
        batch_dict, n_jobs
    )

    c1, c2, c3, c4, c5 = cf_coefficients(batch_dict, k2, k3, k4, k5)

    p_vals = solve_CF(c1, c2, c3, c4, c5)

    # FDR correct:
    _, p_vals_corrected = fdrcorrection(p_vals, alpha=cutoff)

    return p_vals_corrected, p_vals

## Cumulants main function
def find_cumulants(batch_dict, n_jobs):
    """Find cumulants for each gene distribution."""
    # Calculating cumulants from expected distribution of modified corrected Fano factors per cell
    # This is where I stopped. The equations in do_loop use cv. Each equation is per gene, so I could calculate the k's using the batch-specific cv. For example, for batch_key [0, 0, 0, 1, 1, 1], I could do k1[batch == batch_key] = equation(cv[batch]) for batch in batch_key.
    for batch in batch_dict:
        # If batch is 'All' and there's more than one batch, skip it
        if (len(batch_dict) > 1) and (batch == 'All'):
            continue
        batch_dict[batch]['Chi'] = 1 + batch_dict[batch]['CV']**2
        
    outs = np.array(Parallel(n_jobs=n_jobs)(delayed(do_loop_ks_calculation)(batch_dict, gene_row) for gene_row in range(batch_dict['All']['mcFanos'].shape[0])))
    k2, k3, k4, k5 = np.split(outs, 4, axis=1)

    # Fix k shape
    k2 = k2.flatten()
    k3 = k3.flatten()
    k4 = k4.flatten()
    k5 = k5.flatten()

    return k2, k3, k4, k5

## Solve CF polynomial
def solve_CF(c1, c2, c3, c4, c5):
    """Solve CF polynomial and find p values for each corrected fano."""
    coefficients = np.stack((c1, c2, c3, c4, c5), axis=1)
    p_vals = np.empty(coefficients.shape[0])
    for gene_row in range(coefficients.shape[0]):
        p = Polynomial(coefficients[gene_row,:])
        complex_roots = p.roots()
        real_roots = complex_roots[np.isreal(complex_roots)].real   
        if real_roots.shape[0] == 0: ## If there are no real roots, set pvalue to 0.5
            cdf = 0.5
        else:
            abs_roots = [abs(i) for i in real_roots]
            min_root = min(abs_roots)
            desired_root = [i for i in real_roots][abs_roots.index(min_root)]
            if desired_root >=8:
                cdf = 0.5 * exp(-(np.longdouble(desired_root)**2) / 2)
            else:
                cdf = 1 - ncdf(desired_root)

        p_vals[gene_row] = cdf

    return p_vals

## Calculate cumulants
def do_loop_ks_calculation(batch_dict, gene_row):
    '''Calculate individual cumulants for a gene. The function calculates the second to fifth cumulants (k2 to k5) for a given gene. The expectation matrices (e_mat) and coefficient of variations (CV) were calculated per batch, so the final cumulants are functions of batches. I.e., k2[batch == batch_key] = equation(cv[batch]) for batch in batch_key.'''

    total_n_cells_in_dataset = batch_dict['All']['Batch_vector'].shape[0]

    if len(batch_dict.keys()) > 1:
        vars_dict = {batch:{} for batch in batch_dict if batch != 'All'}
    else:
        vars_dict = {'All':{}}

    for batch in vars_dict:
        vars_dict[batch]['subsetmat'] = batch_dict[batch]['e_mat'][:,gene_row]
        vars_dict[batch]["subsetmat2"] = ne.evaluate(
            "subsetmat**2", vars_dict[batch])
        vars_dict[batch]["subsetmat3"] = ne.evaluate(
            "subsetmat**3", vars_dict[batch])
        vars_dict[batch]["subsetmat4"] = ne.evaluate(
            "subsetmat**4", vars_dict[batch])
        vars_dict[batch]["subsetmat5"] = ne.evaluate(
            "subsetmat**5", vars_dict[batch])
        vars_dict[batch]["subsetmat6"] = ne.evaluate(
            "subsetmat**6", vars_dict[batch])
        vars_dict[batch]["subsetmat7"] = ne.evaluate(
            "subsetmat**7", vars_dict[batch])
        vars_dict[batch]['chi'] = batch_dict[batch]['Chi']
        vars_dict[batch]["n_cells"] = batch_dict[batch]['e_mat'].shape[0]

    if len(vars_dict.keys()) == 1:
        k2_list = k2_per_batch(vars_dict, 'All')
        k3_list = k3_per_batch(vars_dict, 'All')
        k4_list = k4_per_batch(vars_dict, 'All')
        k5_list = k5_per_batch(vars_dict, 'All')
    else:
        k2_list = [k2_per_batch(vars_dict, batch) for batch in vars_dict if (batch != 'All')]
        k3_list = [k3_per_batch(vars_dict, batch) for batch in vars_dict if (batch != 'All')]
        k4_list = [k4_per_batch(vars_dict, batch) for batch in vars_dict if (batch != 'All')]
        k5_list = [k5_per_batch(vars_dict, batch) for batch in vars_dict if (batch != 'All')]

    k2 = -(1/total_n_cells_in_dataset) + np.hstack(k2_list).sum()
    k3 = np.hstack(k3_list).sum()
    k4 = np.hstack(k4_list).sum()
    k5 = np.hstack(k5_list).sum()
    return k2, k3, k4, k5

def k2_per_batch(vars_dict, batch):
    k2 = ne.evaluate('(1+subsetmat*(-4+7*chi+6*subsetmat*(1-2*chi+chi**3)+subsetmat2*(-3+6*chi-4*chi**3+chi**6)))/(subsetmat*(n_cells+subsetmat*n_cells*(-1+chi))**2)', vars_dict[batch])

    return k2

def k3_per_batch(vars_dict, batch):
    k3 = ne.evaluate('1/(subsetmat2 * (n_cells + subsetmat * n_cells * (-1 + chi))**3) * (1 + subsetmat * (-9 + 31 * chi) + 2 * subsetmat2 * (16 - 57 * chi + 45 * chi**3) + subsetmat3 * (-56 + 180 * chi - 21 * chi**2 - 168 * chi**3 + 65 * chi**6) + 3 * subsetmat4 * (16 - 48 * chi + 14 * chi**2 + 40 * chi**3 - 6 * chi**4 - 21 * chi**6 + 5 * chi**10) + subsetmat5 * (-16 + 48  * chi - 24 * chi**2 - 30 * chi**3 + 12 * chi**4 + 18 * chi**6 - 3 * chi**7 - 6 * chi**10 + chi**15))', vars_dict[batch])

    return k3

def k4_per_batch(vars_dict, batch):
    k4 = ne.evaluate('1/(subsetmat3 * (n_cells + subsetmat * n_cells * (-1 + chi))**4) * (1 + subsetmat * (-15 + 127 * chi) + subsetmat2 * (92 - 674 * chi + 966 * chi**3) + subsetmat3 * (-302 + 1724 * chi - 271 * chi**2 - 2804 * chi**3 + 1701 * chi**6) + 6 * subsetmat4 * (96 - 452 * chi + 174 * chi**2 + 620 * chi**3 - 102 * chi**4 - 511 * chi**6 + 175 * chi**10) + 2 * subsetmat5 * (-320 + 1344 * chi - 822 * chi**2 - 1390 * chi**3 + 672 * chi**4 + 1124 * chi**6 - 151 * chi**7 - 590 * chi**10 + 133 * chi**15) + 4 * subsetmat6 * (96 - 384 * chi + 312 * chi**2 + 278 * chi**3 - 276 * chi**4 + 18 * chi**5 - 194 * chi**6 + 84 * chi**7 - 9 * chi**9 + 126 * chi**10 - 15 * chi**11 - 43 * chi**15 + 7 * chi**21) + subsetmat7 * (-96 + 384 * chi - 384 * chi**2 - 160 * chi**3 + 314 * chi**4 - 48 * chi**5 + 112 * chi**6 - 120 * chi**7 + 12 * chi**8 + 24 * chi**9 - 80 * chi**10 + 24 * chi**11 - 3 * chi**12 + 32 * chi**15 - 4 * chi**16 - 8 * chi**21 + chi**28))', vars_dict[batch])

    return k4

def k5_per_batch(vars_dict, batch):
    k5 = ne.evaluate('1/(subsetmat4 * (n_cells + subsetmat * n_cells * (-1 + chi))**5) * (1 + subsetmat * (-25 + 511 * chi) + 30 * subsetmat2 * (8 - 119 * chi + 311 * chi**3) + 5 * subsetmat3 * (-248 + 2540 * chi - 561 * chi**2 - 7208 * chi**3 + 6821 * chi**6) + subsetmat4 * (3904 - 29880 * chi + 15690 * chi**2 + 68000 * chi**3 - 12990 * chi**4 - 86865 * chi**6 + 42525 * chi**10) + subsetmat5 * (-7872 + 49360 * chi - 39660 * chi**2 - 81110 * chi**3 + 46460 * chi**4 + 98270 * chi**6 - 13365 * chi**7 - 74910 * chi**10 + 22827 * chi**15) + 20 * subsetmat6 * (512 - 2832 * chi + 2898 * chi**2 + 3168 * chi**3 - 3654 * chi**4 + 216 * chi**5 - 3109 * chi**6 + 1499 * chi**7 - 240 * chi**9 + 2953 * chi**10 - 315 * chi**11 - 1390 * chi**15 + 294 * chi**21) + 10 * subsetmat7 * (-832 + 4288 * chi - 5136 * chi**2 - 2940 * chi**3 + 6222 * chi**4 - 1008 * chi**5 + 2276 * chi**6 - 2778 * chi**7 + 172 * chi**8 + 806 * chi**9 - 2636 * chi**10 + 842 * chi**11 - 65 * chi**12 - 90 * chi**13 + 1420 * chi**15 - 140 * chi**16 - 476 * chi**21 + 75 * chi**28) + 5 * subsetmat**8 * (768 - 3840 * chi + 5184 * chi**2 + 1152 * chi**3 - 5656 * chi**4 + 1728 * chi**5 - 936 * chi**6 + 2432 * chi**7 - 420 * chi**8 - 960 * chi**9 + 1448 * chi**10 - 912 * chi**11 + 186 * chi**12 + 192 * chi**13 - 720 * chi**15 + 170 * chi**16 - 12 * chi**18 + 288 * chi**21 - 28 * chi**22 - 73 * chi**28 + 9 * chi**36) + subsetmat**9 * (-768 + 3840 * chi - 5760 * chi**2 + 320 * chi**3 + 5280 * chi**4 - 2536 * chi**5 + 560 * chi**6 - 2240 * chi**7 + 840 * chi**8 + 980 * chi**9 - 1072 * chi**10 + 880 * chi**11 - 300 * chi**12 - 210 * chi**13 + 400 * chi**15 - 180 * chi**16 + 20 * chi**17 + 40 * chi**18 - 170 * chi**21 + 40 * chi**22 + 50 * chi**28 - 5 * chi**29 - 10 * chi**36 + chi**45))', vars_dict[batch])

    return k5

## Calculate CF coefficients
def cf_coefficients(batch_dict, k2, k3, k4, k5):
    """ Calculate coefficients for Cornish Fisher polynomial"""

    modified_corrected_fanos = batch_dict['All']['mcFanos']
    c1 = 1-modified_corrected_fanos-k3/(6*k2)+17*k3**3/(324*k2**4)-k3*k4/(12*k2**3)+k5/(40*k2**2)
    c2 = k2**(1/2)+5*k3**2/(36*k2**(5/2))-k4/(8*k2**(3/2))
    c3 = k3/(6*k2)-53*k3**3/(324*k2**4)+5*k3*k4/(24*k2**3)-k5/(20*k2**2)
    c4 = -k3**2/(18*k2**(5/2))+k4/(24*k2**(3/2))
    c5 = k3**3/(27*k2**4)-k3*k4/(24*k2**3)+k5/(120*k2**2)

    return c1, c2, c3, c4, c5