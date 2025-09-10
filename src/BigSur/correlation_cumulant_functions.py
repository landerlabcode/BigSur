from typing import Union, Iterable
import time
import numpy as np
import mpmath
import numexpr as ne
import warnings
import pandas as pd
import os
import numexpr as ne

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

    tic = time.perf_counter()

    k3_matrix = (1+cv**2*e_mat*(3+cv**2*(3+cv**2)*e_mat))/(np.sqrt(e_mat)*(1+cv**2*e_mat)**(3/2))
    k4_matrix = (1+e_mat*(3+cv**2*(7+e_mat*(6+3*cv**2*(6+e_mat)+cv**4*(6+(16+15*cv**2+6*cv**4+cv**6)*e_mat)))))/(e_mat*(1+cv**2*e_mat)**2)
    k5_matrix_2 = 1/(e_mat**(3/2)*(1+cv**2*e_mat)**(5/2)) * (1 + 5*(2+3*cv**2)*e_mat + 5*cv**2*(8+15*cv**2+5*cv**4)*e_mat**2+10*cv**4*(6+17*cv**2+15*cv**4+6*cv**6+cv**8)*e_mat**3+cv**6*(30+135*cv**2+222*cv**4+205*cv**6+120*cv**8+45*cv**10+10*cv**12+cv**14)*e_mat**4)

    # Calculate kappa 2
    kappa2 = (1/(n_cells-1)**2) * f2 * n_cells

    k3_crossprod = k3_matrix.T @ k3_matrix 

    kappa3 = (1/(n_cells-1)**3) * f3 * k3_crossprod
    
    # Calculate kappa 4
    k4_crossprod = k4_matrix.T @ k4_matrix 
    kappa4 = (1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)
    del k4_crossprod, f4

    # Calculate kappa 5
    k5_crossprod_2 = k5_matrix_2.T @ k5_matrix_2 
    kappa5 = (1/(n_cells-1)**5) * (-10*f2*f3*k3_crossprod + f5*k5_crossprod_2)

    non_numexpr = time.perf_counter() - tic

    # Right now, numexpr provides a ~3x speedup, but k4 and k5 have large (10^-5 and 10^-2) differences.

    # tic = time.perf_counter()

    # dict_for_calculations = {'n_cells': residuals.shape[0], 'f2': e_moments[0,:], 'f3': e_moments[1,:], 'f4': e_moments[2,:], 'f5': e_moments[3,:], 'one_plus_cv_squared_times_emat' : 1+cv**2*e_mat, 'cv':cv, 'e_mat':e_mat}

    # dict_for_calculations['k3_matrix_new'] = ne.evaluate('(one_plus_cv_squared_times_emat*(3+cv**2*(3+cv**2)*e_mat))/(sqrt(e_mat)*(one_plus_cv_squared_times_emat)**(3/2))', dict_for_calculations)

    # # k4_matrix is 10^-14 different from k4_matrix_new. The matrix multiplication is 10^-5 different. 

    # dict_for_calculations['k4_matrix_new'] = ne.evaluate('(1+e_mat*(3+cv**2*(7+e_mat*(6+3*cv**2*(6+e_mat)+cv**4*(6+(16+15*cv**2+6*cv**4+cv**6)*e_mat)))))/(e_mat*(one_plus_cv_squared_times_emat)**2)', dict_for_calculations)

    # dict_for_calculations['k5_matrix_2_new'] = ne.evaluate('1/(e_mat**(3/2)*(one_plus_cv_squared_times_emat)**(5/2)) * (1 + 5*(2+3*cv**2)*e_mat + 5*cv**2*(8+15*cv**2+5*cv**4)*e_mat**2+10*cv**4*(6+17*cv**2+15*cv**4+6*cv**6+cv**8)*e_mat**3+cv**6*(30+135*cv**2+222*cv**4+205*cv**6+120*cv**8+45*cv**10+10*cv**12+cv**14)*e_mat**4)', dict_for_calculations)

    # del dict_for_calculations['e_mat'], dict_for_calculations['one_plus_cv_squared_times_emat']

    # # Calculate kappa 2
    # dict_for_calculations['kappa2'] = ne.evaluate('(1/(n_cells-1)**2) * f2 * n_cells', dict_for_calculations)

    # # Calculate kappa 3
    # #_, R = np.linalg.qr(dict_for_calculations['k3_matrix_new'])
    # dict_for_calculations['k3_crossprod'] = dict_for_calculations['k3_matrix_new'].T @ dict_for_calculations['k3_matrix_new'] #R.T @ R
    # #del R
    # dict_for_calculations['kappa3'] = ne.evaluate('(1/(n_cells-1)**3) * f3 * k3_crossprod', dict_for_calculations)
    
    # # Calculate kappa 4
    # #_, R = np.linalg.qr(dict_for_calculations['k4_matrix_new'])
    # dict_for_calculations['k4_crossprod'] = dict_for_calculations['k4_matrix_new'].T @ dict_for_calculations['k4_matrix_new'] #R.T @ R
    # #del R
    # dict_for_calculations['kappa4'] = ne.evaluate('(1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)', dict_for_calculations)
    # del dict_for_calculations['k4_crossprod'], dict_for_calculations['f4']

    # # Calculate kappa 5
    # #_, R = np.linalg.qr(dict_for_calculations['k5_matrix_2_new'])
    # dict_for_calculations['k5_crossprod_2'] = dict_for_calculations['k5_matrix_2_new'].T @ dict_for_calculations['k5_matrix_2_new']
    # #del R
    # dict_for_calculations['kappa5'] = ne.evaluate('(1/(n_cells-1)**5) * (-10*f2*f3*k3_crossprod + f5*k5_crossprod_2)', dict_for_calculations)

    # dict_for_calculations['k3_matrix_new_transpose'] = dict_for_calculations['k3_matrix_new'].T
    # dict_for_calculations['k4_matrix_new_transpose'] = dict_for_calculations['k4_matrix_new'].T
    # dict_for_calculations['k5_matrix_2_new_transpose'] = dict_for_calculations['k5_matrix_2_new'].T

    # dict_for_calculations['k3_crossprod'] = np.matmul(dict_for_calculations['k3_matrix_new_transpose'], dict_for_calculations['k3_matrix_new'])
    # dict_for_calculations['k4_crossprod'] = np.matmul(dict_for_calculations['k4_matrix_new'].T, dict_for_calculations['k4_matrix_new'])
    # dict_for_calculations['k5_crossprod_2'] = np.matmul(dict_for_calculations['k5_matrix_2_new_transpose'], dict_for_calculations['k5_matrix_2_new'])

    # kappa2_new = ne.evaluate('(1/(n_cells-1)**2) * f2 * n_cells', dict_for_calculations)
    # kappa3_new = ne.evaluate('(1/(n_cells-1)**3) * f3 * k3_crossprod', dict_for_calculations)
    # kappa4_new = ne.evaluate('(1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)', dict_for_calculations)
    # kappa5_new = ne.evaluate('(1/(n_cells-1)**5) * (-10*f2*f3*k3_crossprod + f5*k5_crossprod_2)', dict_for_calculations)

    numexpr = time.perf_counter() - tic

    # kappa2 = pd.read_csv('../../Data/results/lymph_nodes/correlations/correlations_R_testing/cumulants_1.csv', index_col=0)
    # kappa3 = pd.read_csv('../../Data/results/lymph_nodes/correlations/correlations_R_testing/cumulants_2.csv', index_col=0)
    # kappa4 = pd.read_csv('../../Data/results/lymph_nodes/correlations/correlations_R_testing/cumulants_3.csv', index_col=0)
    # kappa5 = pd.read_csv('../../Data/results/lymph_nodes/correlations/correlations_R_testing/cumulants_4.csv', index_col=0)

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
            time_diff = toc-tic
            if time_diff < 60:
                print(
                    f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(time_diff):04f} seconds."
                )
            else:
                print(
                    f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(time_diff/60):04f} minutes."
                )
    else:
        tic = time.perf_counter()
        kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
        toc = time.perf_counter()
        time_diff = toc-tic
        if time_diff < 60:
            print(
                f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(time_diff):04f} seconds."
            )
        else:
            print(
                f"Finished calculating interpolated moments for {g_counts.shape[0]} genes in {(time_diff/60):04f} minutes."
            )
            
    return save_kappas,kappa2,kappa3,kappa4,kappa5