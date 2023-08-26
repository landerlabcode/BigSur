#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:55:15 2022

@author: emmanueldollinger
"""
from typing import Union
import time
import numpy as np
from scipy.optimize import brentq
from anndata import AnnData
from mpmath import ncdf, exp
import numexpr as ne
import warnings
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)


def mcfano_feature_selection(
    adata: AnnData,
    layer: str,
    cv: Union[bool, float] = 0.5,
    n_genes_for_PCA: Union[bool, int] = False,
    min_mcfano_cutoff: Union[bool, float] = 0.95,
    p_val_cutoff: Union[bool, float] = 0.2,
    return_residuals: bool = False,
    n_jobs: int = -2,
    verbose: int = 1,
):
    """
    Calculate the corrected Fano factor for all genes in the dataset. mc_Fano column will be added to .var. Highly_variable column will be added to .var based on the n_genes_for_pca, min_mcfano_cutoff and p_val_cutoff parameters.

    Parameters
    ----------
    adata - adata object containing information about the raw counts and gene names.
    layer - String, describing the layer of adata object containing raw counts (pass "X" if raw counts are in adata.X).
    cv - Float, coefficient of variation for the given dataset. If None, the CV will be estimated.
    n_genes_for_PCA - [Int, Bool], top number of genes to use for PCA, ranked by corrected modified Fano factor. If False, default to combination of min_mcfano_cutoff and/or p_val_cutoff.
    min_mcfano_cutoff - Union[bool, float], calculate p-values for corrected modified Fano factors greater than min_mcfano_cutoff quantile and only include these genes in highly_variable column. If False default to combination of n_genes_for_PCA and/or p_val_cutoff.
    p_val_cutoff - [Bool, Float], if a float value is provided, that p-value cutoff will be used to select genes. If False, default to combination of min_mcfano_cutoff and/or n_genes_for_PCA.
    return_residuals - Bool, if True, the function will return a matrix containing the calculated mean-centered corrected Pearson residuals matrix stored in adata.layers['residuals'].
    n_jobs - Int, how many cores to use for p-value parallelization. Default is -2 (all but 1 core).
    verbose - Int, whether to print computations and top 100 genes. 0 is no verbose, 1 is a little (what the function is doing) and 2 is full verbose.
    
    """

    # Setup
    determine_cutoff_parameters(n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose)
        
    # Create variables
    raw_count_mat, means, variances, g_counts = make_vars_and_qc(adata, layer)

    tic = time.perf_counter()
    if verbose > 1:
        print("Calculating corrected Fano factors.")
    # Fit cv if not provided
    if cv is None:
        if verbose > 1:
            print('Fitting cv.')
        cv = fit_cv(raw_count_mat, means, variances, g_counts, verbose)
        if verbose >= 1:
            print(f'After fitting, cv = {cv}')
    # Calculate residuals
    cv, normlist, residuals, n_cells = calculate_residuals(cv, verbose, raw_count_mat, means, variances, g_counts)
    # Calculate mcfanos from residuals
    corrected_fanos = calculate_mcfano(residuals, n_cells)

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating corrected Fano factors for {corrected_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    adata.var["mc_Fano"] = np.array(corrected_fanos).flatten()
    adata.uns['CV_for_mc_Fano_fit'] = cv

    # Calculate p-values
    if isinstance(p_val_cutoff, float):
        tic = time.perf_counter()
        if verbose > 1:
            print("Calculating p-values.")
        meets_cutoff, p_vals_corrected, p_vals = calculate_p_value(
            raw_count_mat, cv, means, normlist, corrected_fanos, p_val_cutoff, n_jobs
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
            
    genes = determine_HVGs(adata, n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose)
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
        adata.layers['residuals'] = residuals

def calculate_p_value(
    raw_count_mat, cv, means, normlist, corrected_fanos, cutoff, n_jobs
):
    """Calculate the p value for corrected fanos"""

    p_vals, k2, k3, k4 = find_moments(
        raw_count_mat, cv, means, normlist, corrected_fanos, n_jobs
    )

    p_vals = find_pvals(corrected_fanos, p_vals, k2, k3, k4)

    # FDR correct:
    meets_cutoff, p_vals_corrected = fdrcorrection(p_vals, alpha=cutoff)

    return meets_cutoff, p_vals_corrected, p_vals

def find_moments(raw_count_mat, cv, means, normlist, corrected_fanos, n_jobs):
    """Find moments for each gene distribution"""
    wlist = len(normlist) * normlist

    # Calculating expectation matrix for per-cell gene means
    emat = np.outer(means, wlist)  # genesxcells

    # Calculating moments from expected distribution of Fano factors per cell
    chi = 1 + cv**2
    n_cells = raw_count_mat.shape[0]
    p_vals = np.empty(corrected_fanos.shape[0])
    # Don't calculate p_vals for genes below cutoff

    dict_for_vars = {"chi": chi, "n_cells": n_cells, "emat": emat}
        
    outs = np.array(Parallel(n_jobs=n_jobs)(delayed(do_loop_ks_calculation)(dict_for_vars, gene_row) for gene_row in range(dict_for_vars['emat'].shape[0])))
    k2, k3, k4 = np.split(outs, 3, axis=1)

    return p_vals, k2, k3, k4

def find_pvals(corrected_fanos, p_vals, k2, k3, k4):
    """Take moments and find p values for each corrected fano"""
    for gene_row in range(corrected_fanos.shape[0]):
        fano = corrected_fanos[gene_row]
        subk2 = k2[gene_row]
        subk3 = k3[gene_row]
        subk4 = k4[gene_row]

        def f(x):
            out = (
                (1 - fano - subk3 / (6 * subk2))
                + (
                    subk2**0.5
                    + (
                        5 * subk3**2 / (36 * subk2 ** (5 / 2))
                        - subk4 / (8 * subk2 ** (3 / 2))
                    )
                )
                * x
                + (subk3 / (6 * subk2)) * x**2
                + (
                    -(subk3**2) / (18 * subk2 ** (5 / 2))
                    + subk4 / (24 * subk2 ** (3 / 2))
                )
                * x**3
            )
            return out

        # Check ranges for brent's method; sign(f(x0)) != sign(f(x1))
        x0 = 0
        x1 = 500
        fx0 = f(x0)
        fx1 = f(x1)

        if fx0 * fx1 >= 0:
            x0 = 500
            x1 = 20000
            fx0 = f(x0)
            fx1 = f(x1)
            if fx0 * fx1 >= 0:
                x0 = -20000
                fx0 = f(x0)
                fx1 = f(x1)
                if fx0 * fx1 >= 0: # kitchen sink approach
                    range_for_fx = np.arange(-10000, 10000, 0.1)
                    fx_out = np.array([f(x) for x in range_for_fx]).flatten()
                    x0 = range_for_fx[np.where(fx_out == fx_out.min())[0]]
                    sorted_fx_out = np.sort(fx_out)
                    x1 = sorted_fx_out[np.where(sorted_fx_out > 0)[0][0]]

        ge_brent = brentq(f, x0, x1)
        if ge_brent >= 8:
            cdf_brent = 0.5 * exp(-(ge_brent**2) / 2)
        else:
            cdf_brent = 1 - ncdf(ge_brent)

        p_vals[gene_row] = cdf_brent

    return p_vals

def determine_cutoff_parameters(n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose):
        # Determine whether using pvals or n top genes or min_fano or combo thereof
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
                    print_string += f" and {min_mcfano_cutoff} for mcfano quantile cutoff"
                else:
                    print_string += f" {min_mcfano_cutoff} for mcfano quantile cutoff"
            print_string += " for highly variable genes."        
            
            print(print_string)

def determine_HVGs(adata, n_genes_for_PCA, p_val_cutoff, min_mcfano_cutoff, verbose):
    is_n_genes = type(n_genes_for_PCA) == int
    is_p_val_cutoff = isinstance(p_val_cutoff, float)
    is_min_fano_cutoff = isinstance(min_mcfano_cutoff, float)

    # Store adata for easy filtering
    adata_var_df = adata.var.sort_values('mc_Fano', ascending = False).copy() # Sort from greatest to smallest mcFano; without copy there's a bug

    if is_min_fano_cutoff:
        min_fano = np.quantile(adata_var_df['mc_Fano'], min_mcfano_cutoff)
        if verbose > 1:
            print(f'Setting min_fano to {min_fano:.4f}.')
    else:
        min_fano = 0
    genes = adata_var_df[adata_var_df['mc_Fano'] > min_fano].index
    
    if is_p_val_cutoff:
        genes = np.intersect1d(genes, adata_var_df[adata_var_df['FDR_adj_pvalue'] < p_val_cutoff].index)
        genes = adata_var_df.loc[genes].sort_values('mc_Fano', ascending = False).index # Reorder

    if is_n_genes:
        if n_genes_for_PCA > genes.shape[0]:
            warnings.warn(
            f'Number of genes meeting cutoffs ({genes.shape[0]}) is lower than user requested genes ({n_genes_for_PCA}). Only including genes meeting cutoff in "highly_variable" slot. Please increase the min_mcfano_cutoff, decrease the p_val_cutoff, or set either or both to False.')
        else:
            genes = genes[:n_genes_for_PCA] # Should be already sorted

    if verbose > 1:
        print(f'Setting {genes.shape[0]} genes as highly variable.')

    return genes

def do_loop_ks_calculation(dict_for_vars, gene_row):
        dict_for_vars['subsetemat'] = dict_for_vars['emat'][gene_row,:]

        dict_for_vars["subsetematsquare"] = ne.evaluate(
        "subsetemat**2", dict_for_vars)
        dict_for_vars["subsetematcube"] = ne.evaluate(
            "subsetemat**3", dict_for_vars)
        dict_for_vars["subsetmatfourth"] = ne.evaluate(
            "subsetemat**4", dict_for_vars)

        dict_for_vars["p1"] = ne.evaluate(
            "1+subsetemat*(-4+7*chi+6*(1-2*chi+chi**3)*subsetemat+(-3+6*chi-4*chi**3+chi**6)*subsetematsquare)",
            dict_for_vars,
        )
        dict_for_vars["p2"] = ne.evaluate(
            "subsetemat*(n_cells+n_cells*(-1+chi)*subsetemat)**2", dict_for_vars
        )
        dict_for_vars["p3"] = ne.evaluate("n_cells*p2", dict_for_vars)
        dict_for_vars["p4"] = ne.evaluate(
            "(1+subsetemat*(-6+31*chi+15*(1-6*chi+6*chi**3)*subsetemat+5*(-4+21*chi-30*chi**3+13*chi**6)*subsetematsquare+15*(1-4*chi+6*chi**3-4*chi**6+chi**10)*subsetematcube+(-5+15*chi-20*chi**3+15*chi**6-6*chi**10+chi**15)*subsetmatfourth))/(subsetematsquare*(n_cells+n_cells*(-1+chi)*subsetemat)**3)",
            dict_for_vars,
        )

        dict_for_vars["sump1p2"] = ne.evaluate("sum(p1/p2)", dict_for_vars)
        dict_for_vars["sump1p3"] = ne.evaluate("sum(p1/p3)", dict_for_vars)
        dict_for_vars["sump4"] = ne.evaluate("sum(p4)", dict_for_vars)

        g1 = 1

        g2 = ne.evaluate("1-1/n_cells+sump1p2", dict_for_vars)

        g3 = ne.evaluate(
            "1-2/(n_cells**2)-3/n_cells+3*sump1p2-3*sump1p3 + sump4", dict_for_vars
        )

        dict_for_vars["sum1"] = ne.evaluate(
            "sum(p1/(n_cells**2*p2))", dict_for_vars
        )
        dict_for_vars["sum2"] = ne.evaluate(
            "sum(p1**2/(subsetematsquare*(n_cells+n_cells*(-1+chi)*subsetemat)**4))",
            dict_for_vars,
        )
        dict_for_vars["sum3"] = ne.evaluate(
            "sum(p4/n_cells)", dict_for_vars)
        dict_for_vars["sum4"] = ne.evaluate(
            "sum((1+subsetemat*(-8+127*chi+14*(2-36*chi+69*chi**3)*subsetemat+7*(-8+124*chi-344*chi**3+243*chi**6)*subsetematsquare+70*(1-12*chi+36*chi**3-40*chi**6+15*chi**10)*subsetematcube+14*(-4+35*chi-100*chi**3+130*chi**6-80*chi**10+19*chi**15)*subsetmatfourth+28*(1-6*chi+15*chi**3-20*chi**6+15*chi**10-6*chi**15+chi**21)*subsetemat**5+(-7+28*chi-56*chi**3+70*chi**6-56*chi**10+28*chi**15-8*chi**21+chi**28)*subsetemat**6))/(subsetematcube*(n_cells+n_cells*(-1+chi)*subsetemat)**4))",
            dict_for_vars,
        )

        g4 = ne.evaluate(
            "1-6/(n_cells**3)+11/(n_cells**2)-6/n_cells+(6*(-1+n_cells)*sump1p2)/n_cells+3*(sump1p2)**2+12*sum1-12*sump1p3-3*sum2+4*sump4-4*sum3+sum4",
            dict_for_vars,
        )

        k2 = -(g1**2) + g2

        k3 = 2 * g1**3 - 3 * g1 * g2 + g3

        k4 = -6 * g1**4 + 12 * g1**2 * g2 - 3 * g2**2 - 4 * g1 * g3 + g4
        return k2, k3, k4

def cf_coefficients(corrected_fanos,k2,k3,k4,k5 = None):
    """ Calculate coefficients for Cornish Fisher"""
    
    k2 = np.reshape(k2, (k2.shape[0],))
    k3 = np.reshape(k3, (k3.shape[0],))
    k4 = np.reshape(k4, (k4.shape[0],))
    #k5 = np.reshape(k5, (k5.shape[0],))
    
    
    c1 = 1-corrected_fanos-k3/(6*k2)+17*k3**3/(324*k2**4)-k3*k4/(12*k2**3)+k5/(40*k2**2)
    c2 = np.sqrt(k2)+5*k3**2/(36*k2**(5/2))-k4/(8*k2**(3/2))
    c3 = k3/(6*k2)-53*k3**3/(324*k2**4)+5*k3*k4/(24*k2**3)-k5/(20*k2**2)
    c4 = -k3**2/(18*k2**(5/2))+k4/(24*k2**(3/2))
    #c5 = k3**3/(27*k2**4)-k3*k4/(24*k2**3)+k5/(120*k2**2)
    
    return c1, c2, c3, c4, #c5
