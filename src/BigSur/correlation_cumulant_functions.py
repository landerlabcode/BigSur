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