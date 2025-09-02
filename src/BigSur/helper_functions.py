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
from .correlation_coefficient_functions import calculate_mcPCCs_coefficients
from .correlation_cumulant_functions import calculate_mcPCCs_cumulants

def calculate_mcPCCs(n_cells, mc_fanos, residuals):
    '''Calculate mcPCCs.'''
    mcPCCs = 1/((n_cells - 1) * np.sqrt(np.outer(mc_fanos, mc_fanos.T))) * (residuals.T @ residuals)
    return mcPCCs

def load_or_calculate_coefficients(verbose, write_out, previously_run, g_counts, mcPCCs, kappa2, kappa3, kappa4, kappa5):
    save_coefficients = False
    if previously_run:
        if os.path.isfile(write_out + 'coefficients.npz'):
            loader = np.load(write_out + 'coefficients.npz', allow_pickle=True)
            rows = loader['rows']
            cols = loader['cols']
            c1_lower_flat = loader['c1_lower_flat']
            c2_lower_flat = loader['c2_lower_flat']
            c3_lower_flat = loader['c3_lower_flat']
            c4_lower_flat = loader['c4_lower_flat']
            c5_lower_flat = loader['c5_lower_flat']
        else:
            print('Coefficients file not found, recalculating.')
            save_coefficients = True
            tic = time.perf_counter()
            rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat = calculate_mcPCCs_coefficients(kappa2, kappa3, kappa4, kappa5, mcPCCs) # These functions are correct
            toc = time.perf_counter()
            if verbose > 1:
                print(
                    f"Finished calculating coefficients for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
                )
    else:
        tic = time.perf_counter()
        rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat = calculate_mcPCCs_coefficients(kappa2, kappa3, kappa4, kappa5, mcPCCs) # These functions are correct
        toc = time.perf_counter()
        if verbose > 1:
            print(
                f"Finished calculating coefficients for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
            )
            
    return save_coefficients,rows,cols,c1_lower_flat,c2_lower_flat,c3_lower_flat,c4_lower_flat,c5_lower_flat

def load_or_calculate_cumulants(verbose, cv, write_out, previously_run, g_counts, residuals, e_mat, e_moments):
    save_kappas = False
    if previously_run:
        if os.path.isfile(write_out + 'cumulants.npz'):
            loader = np.load(write_out + 'cumulants.npz', allow_pickle=True)
            kappa2 = loader['kappa2']
            kappa3 = loader['kappa3']
            kappa4 = loader['kappa4']
            kappa5 = loader['kappa5']
        else:
            print("Cumulants file not found, recalculating.")
            save_kappas = True
            tic = time.perf_counter()
            kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
            toc = time.perf_counter()
            if verbose > 1:
                print(
                    f"Finished calculating cumulants for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
                )
    else:
        tic = time.perf_counter()
        kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
        toc = time.perf_counter()
        if verbose > 1:
            print(
                f"Finished calculating cumulants for {g_counts.shape[0]} genes in {(toc-tic):04f} seconds."
            )
            
    return save_kappas,kappa2,kappa3,kappa4,kappa5

def load_or_calculate_mcpccs(verbose, write_out, previously_run, tic, residuals, n_cells, mc_fanos):
    save_mcPCCs = False
    if previously_run:
        try:
            mcPCCs = np.load(write_out + 'mcPCCs.npz', allow_pickle=True)['mcPCCs']
        except FileNotFoundError:
            save_mcPCCs = True
            print("mcPCCs file not found, recalculating.")
            mcPCCs = calculate_mcPCCs(n_cells, mc_fanos, residuals)
    else:
        mcPCCs = calculate_mcPCCs(n_cells, mc_fanos, residuals)
    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Pearson correlation coefficients for {mc_fanos.shape[0]} genes in {(toc-tic):04f} seconds."
        )
        
    return save_mcPCCs,mcPCCs

def load_or_calculate_mc_fanos(write_out, previously_run, residuals, n_cells):
    save_mc_fanos = False
    if previously_run:
        try:
            mc_fanos = np.load(write_out + 'mc_fanos.npz', allow_pickle=True)['mc_fanos']
        except FileNotFoundError:
            print("mcFano file not found, recalculating.")
            save_mc_fanos = True
            mc_fanos = calculate_mcfano(residuals, n_cells)
    else:
        mc_fanos = calculate_mcfano(residuals, n_cells)
    return save_mc_fanos,mc_fanos

def load_or_calculate_residuals(cv, write_out, previously_run, raw_count_mat, g_counts):
    save_residuals = False
    if previously_run:
        try:
            residuals = np.load(write_out + 'residuals.npz', allow_pickle=True)['residuals']
            n_cells = residuals.shape[0]
            normlist, n_cells, e_mat = calculate_emat(raw_count_mat, g_counts)
        except FileNotFoundError:
            print("Residuals file not found, recalculating.")
            save_residuals = True
            cv, normlist, residuals, n_cells, e_mat = calculate_residuals(cv, raw_count_mat, g_counts)
    else:
        cv, normlist, residuals, n_cells, e_mat = calculate_residuals(cv, raw_count_mat, g_counts)
    return cv,save_residuals,residuals,n_cells,normlist,e_mat