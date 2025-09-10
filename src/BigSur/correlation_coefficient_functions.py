import time
import numpy as np
import numexpr as ne
import os
import pandas as pd

from scipy.sparse import load_npz

## BigSur
from .preprocessing import calculate_residuals, calculate_mcfano, calculate_emat

def calculate_mcPCCs_coefficients(k2, k3, k4, k5, mcPCCs):
    # Convert to numexpr

    dict_for_calculations = {'k2':k2, 'k3':k3, 'k4':k4, 'k5':k5, 'mcPCCs':mcPCCs}

    c1 = ne.evaluate('-mcPCCs-k3/(6*k2)+17*k3**3/(324*k2**4)-k3*k4/(12*k2**3)+k5/(40*k2**2)', dict_for_calculations)
    c2 = ne.evaluate('sqrt(k2)+5*k3**2/(36*k2**(5/2))-k4/(8*k2**(3/2))', dict_for_calculations)
    c3 = ne.evaluate('k3/(6*k2)-53*k3**3/(324*k2**4)+5*k3*k4/(24*k2**3)-k5/(20*k2**2)', dict_for_calculations)
    c4 = ne.evaluate('-k3**2/(18*k2**(5/2))+k4/(24*k2**(3/2))', dict_for_calculations)
    c5 = ne.evaluate('k3**3/(27*k2**4)-k3*k4/(24*k2**3)+k5/(120*k2**2)', dict_for_calculations)

    mcPCCs_length = c1.shape[0]

    # Generate row/col indices for lower triangle
    # Currently, c1[0, 10] != c1[10, 0]
    rows, cols = np.tril_indices(mcPCCs_length, -1)

    c1_lower_flat = np.tril(c1, -1)[rows, cols]
    c2_lower_flat = np.tril(c2, -1)[rows, cols]
    c3_lower_flat = np.tril(c3, -1)[rows, cols]
    c4_lower_flat = np.tril(c4, -1)[rows, cols]
    c5_lower_flat = np.tril(c5, -1)[rows, cols]
    return rows, cols, c1_lower_flat, c2_lower_flat, c3_lower_flat, c4_lower_flat, c5_lower_flat

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

def calculate_mcPCCs(n_cells, mc_fanos, residuals):
    '''Calculate mcPCCs.'''
    mcPCCs = 1/((n_cells - 1) * np.sqrt(np.outer(mc_fanos, mc_fanos.T))) * (residuals.T @ residuals)
    return mcPCCs

def load_or_calculate_mcpccs(verbose, write_out, previously_run, tic, residuals, n_cells, mc_fanos):
    save_mcPCCs = False
    if previously_run:
        try:
            mcPCCs = load_npz(write_out + 'mcPCCs.npz')
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
            mc_fanos = pd.read_csv(write_out + 'mc_Fano.csv', index_col = 0).to_numpy().flatten()
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