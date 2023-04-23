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
from statsmodels.stats.multitest import fdrcorrection
import numexpr as ne
import warnings
from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)


def mcfano_feature_selection(
    adata: AnnData,
    layer: str,
    cv: Union[bool, float] = 0.5,
    n_genes_for_PCA: Union[bool, int] = False,
    min_mcfano_cutoff: float = 0.95,
    p_val_cutoff: Union[bool, float] = 0.05,
    return_residuals: bool = False,
    verbose: int = 1,
):
    """
    Calculate the corrected Fano factor for all genes in the dataset. mc_Fano column will be added to .var. Highly_variable column will be added to .var based on the n_genes_for_pca and cutoff parameters.

    Parameters
    ----------
    adata - adata object containing information about the raw counts and gene names.
    layer - String, describing the layer of adata object containing raw counts (pass "X" if raw counts are in adata.X).
    cv - Float, coefficient of variation for the given dataset. If None, the CV will be estimated.
    n_genes_for_PCA - [Int, Bool], top number of genes to use for PCA, ranked by corrected modified Fano factor. If False, use p_val_cutoff and min_mcfano_cutoff for cutoffs.
    min_mcfano_cutoff - [Float], calculate p-values for corrected modified Fano factors greater than min_mcfano_cutoff quantile and only include these genes in highly_variable column.
    p_val_cutoff - [Bool, Float], if a float value is provided, that p-value cutoff will be used to select genes. If False, only use top genes cutoff method.
    verbose - Int, whether to print computations and top 100 genes. 0 is no verbose, 1 is a little (what the function is doing) and 2 is full verbose.
    return_residuals - Bool, if True, the function will return a matrix containing the calculated mean-centered corrected Pearson residuals matrix stored in adata.layers['residuals'].
    """

    # Setup
    # Determine whether using pvals or n top genes or both
    is_n_genes = isinstance(n_genes_for_PCA, bool)
    is_cutoff = isinstance(p_val_cutoff, float)
    # Need to add min_fano only as an option
    if not is_n_genes and is_cutoff:
        if verbose >= 1:
            print(
                f"Using pvalue cutoff of {p_val_cutoff} and calculating pvalues for genes with mcFano factor that are {min_mcfano_cutoff} quantile and top {n_genes_for_PCA} genes for highly variable genes"
            )
        pval_or_ntop_genes = "Both"
    elif is_n_genes and is_cutoff:
        if verbose >= 1:
            print(
                f"Only using pvalue cutoff {p_val_cutoff} and calculating pvalues for genes with mcFano factor that are {min_mcfano_cutoff} quantile for highly variable genes"
            )
        pval_or_ntop_genes = "pvalue"
    elif not is_n_genes and not is_cutoff:
        if verbose >= 1:
            print(
                f"Only using top {n_genes_for_PCA} genes for highly variable genes")
        pval_or_ntop_genes = "nTop"
    elif not is_cutoff and is_n_genes:
        if verbose >= 1:
            print(
                f"Only using min fano quantile cutoff of {min_mcfano_cutoff} for highly variable genes")
        pval_or_ntop_genes = "min_fano"
    else:
        raise Exception(
            "Please specify either number of top genes or pvalue cutoff or min fano quantile cutoff."
        )

    # Create variables
    raw_count_mat, means, variances, g_counts = make_vars_and_qc(adata, layer)

    tic = time.perf_counter()
    if verbose > 1:
        print("Calculating corrected Fano factors.")

    if cv is None:
        cv = fit_cv(raw_count_mat, means, variances, g_counts, verbose)

    cv, normlist, residuals, n_cells = calculate_residuals(cv, verbose, raw_count_mat, means, variances, g_counts)

    corrected_fanos = calculate_mcfano(residuals, n_cells)

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating corrected Fano factors for {corrected_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    adata.var["mc_Fano"] = np.array(corrected_fanos).flatten()
    adata.uns['CV_for_mc_Fano_fit'] = cv

    min_fano = np.quantile(corrected_fanos, min_mcfano_cutoff)
    if verbose > 1:
        print(f'Setting min_fano to {min_fano}')

    # Calculate p-values
    if pval_or_ntop_genes != "nTop" and pval_or_ntop_genes != "min_fano":
        tic = time.perf_counter()
        if verbose > 1:
            print("Calculating p-values.")
        meets_cutoff, p_vals_corrected = calculate_p_value(
            raw_count_mat, cv, means, normlist, corrected_fanos, min_fano, p_val_cutoff
        )
        toc = time.perf_counter()
        if verbose > 1:
            print(
                f"Finished calculating p-values for {p_vals_corrected[~np.isnan(p_vals_corrected)].shape[0]} corrected Fano factors in {(toc-tic):04f} seconds."
            )

        # Store p-values
        adata.var["p-value"] = p_vals_corrected

    # Store HVGs
    if pval_or_ntop_genes == "pvalue":
        genes = (
            adata.var.loc[meets_cutoff, "mc_Fano"].sort_values(
                ascending=False).index
        )  # Sorted for easy printing later
        if verbose > 1:
            print(
                f"Setting {len(genes)} genes with p-values below {p_val_cutoff} and Fano factors above {min_fano} as highly variable."
            )
    elif pval_or_ntop_genes == "nTop":
        genes = adata.var.sort_values("mc_Fano", ascending=False)[
            :n_genes_for_PCA
        ].index
        if verbose > 1:
            print(f"Setting top {n_genes_for_PCA} genes as highly variable.")
    elif pval_or_ntop_genes == "min_fano":
        genes = adata.var.loc[adata.var['mc_Fano'] >
                              min_fano, 'mc_Fano'].sort_values().index
        if verbose > 1:
            print(
                f"Setting all genes with mcFano > {min_fano} as highly variable.")
    elif pval_or_ntop_genes == "Both":
        genesdf = adata.var.loc[meets_cutoff, "mc_Fano"]
        genes = genesdf.sort_values(ascending=False)[:n_genes_for_PCA].index
        # Check if pvalue cuttoff is lower than requested number of genes
        if meets_cutoff.sum() < n_genes_for_PCA:
            warnings.warn(
                f'Number of genes meeting cutoffs ({meets_cutoff.sum()}) is lower than user requested genes ({n_genes_for_PCA}). Only including genes meeting cutoff in "highly_variable" slot.')
        if verbose > 1:
            print(
                f"Setting {len(genes)} genes with p-values below {p_val_cutoff} and Fano factors above {min_fano} as highly variable."
            )
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
    raw_count_mat, cv, means, normlist, corrected_fanos, min_fano, cutoff
):
    """Calculate the p value for corrected fanos"""

    p_vals, indices, subsetemat, k2, k3, k4 = find_moments(
        raw_count_mat, cv, means, normlist, corrected_fanos, min_fano
    )

    p_vals = find_pvals(corrected_fanos, p_vals,
                        indices, subsetemat, k2, k3, k4)

    # FDR correct:
    _, p_vals_corrected_sub = fdrcorrection(p_vals[indices], alpha=cutoff)
    p_vals[indices] = p_vals_corrected_sub
    meets_cutoff = np.repeat(False, p_vals.shape[0])
    # has nan for pvalues that weren't calculated
    meets_cutoff[p_vals <= cutoff] = True

    return meets_cutoff, p_vals


def find_moments(raw_count_mat, cv, means, normlist, corrected_fanos, min_fano):
    """Find moments for each gene distribution"""
    wlist = len(normlist) * normlist

    # Calculating expectation matrix for per-cell gene means
    emat = np.outer(means, wlist)  # genesxcells

    # Calculating moments from expected distribution of Fano factors per cell
    chi = 1 + cv**2
    n_cells = raw_count_mat.shape[0]
    p_vals = np.empty(corrected_fanos.shape[0])
    # Don't calculate p_vals for genes below cutoff
    p_vals[corrected_fanos <= min_fano] = np.nan
    indices = np.where(corrected_fanos > min_fano)[0]
    subsetemat = emat[indices, :].astype(float)

    dict_for_vars = {"chi": chi, "n_cells": n_cells, "subsetemat": subsetemat}

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

    dict_for_vars["sump1p2"] = ne.evaluate("sum(p1/p2, axis=1)", dict_for_vars)
    dict_for_vars["sump1p3"] = ne.evaluate("sum(p1/p3, axis=1)", dict_for_vars)
    dict_for_vars["sump4"] = ne.evaluate("sum(p4, axis=1)", dict_for_vars)

    g1 = np.ones(subsetemat.shape[0])

    g2 = ne.evaluate("1-1/n_cells+sump1p2", dict_for_vars)

    g3 = ne.evaluate(
        "1-2/(n_cells**2)-3/n_cells+3*sump1p2-3*sump1p3 + sump4", dict_for_vars
    )

    dict_for_vars["sum1"] = ne.evaluate(
        "sum(p1/(n_cells**2*p2), axis=1)", dict_for_vars
    )
    dict_for_vars["sum2"] = ne.evaluate(
        "sum(p1**2/(subsetematsquare*(n_cells+n_cells*(-1+chi)*subsetemat)**4), axis=1)",
        dict_for_vars,
    )
    dict_for_vars["sum3"] = ne.evaluate(
        "sum(p4/n_cells, axis=1)", dict_for_vars)
    dict_for_vars["sum4"] = ne.evaluate(
        "sum((1+subsetemat*(-8+127*chi+14*(2-36*chi+69*chi**3)*subsetemat+7*(-8+124*chi-344*chi**3+243*chi**6)*subsetematsquare+70*(1-12*chi+36*chi**3-40*chi**6+15*chi**10)*subsetematcube+14*(-4+35*chi-100*chi**3+130*chi**6-80*chi**10+19*chi**15)*subsetmatfourth+28*(1-6*chi+15*chi**3-20*chi**6+15*chi**10-6*chi**15+chi**21)*subsetemat**5+(-7+28*chi-56*chi**3+70*chi**6-56*chi**10+28*chi**15-8*chi**21+chi**28)*subsetemat**6))/(subsetematcube*(n_cells+n_cells*(-1+chi)*subsetemat)**4), axis=1)",
        dict_for_vars,
    )

    g4 = ne.evaluate(
        "1-6/(n_cells**3)+11/(n_cells**2)-6/n_cells+(6*(-1+n_cells)*sump1p2)/n_cells+3*(sump1p2)**2+12*sum1-12*sump1p3-3*sum2+4*sump4-4*sum3+sum4",
        dict_for_vars,
    )

    k2 = -(g1**2) + g2

    k3 = 2 * g1**3 - 3 * g1 * g2 + g3

    k4 = -6 * g1**4 + 12 * g1**2 * g2 - 3 * g2**2 - 4 * g1 * g3 + g4
    return p_vals, indices, subsetemat, k2, k3, k4


def find_pvals(corrected_fanos, p_vals, indices, subsetemat, k2, k3, k4):
    """Take moments and find p values for each corrected fano"""
    for i in range(subsetemat.shape[0]):
        fano = corrected_fanos[indices[i]]
        subk2 = k2[i]
        subk3 = k3[i]
        subk4 = k4[i]

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

        ge_brent = brentq(f, x0, x1)
        if ge_brent >= 8:
            cdf_brent = 0.5 * exp(-(ge_brent**2) / 2)
        else:
            cdf_brent = 1 - ncdf(ge_brent)

        p_vals[indices[i]] = cdf_brent

    return p_vals
