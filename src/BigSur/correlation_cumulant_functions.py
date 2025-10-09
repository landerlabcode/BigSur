import numpy as np
import numexpr as ne
import os
import numexpr as ne

# Functions for p-value calculations
def calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv):
    '''Calculate the cumulants (kappa's) from the interpolated moments (for p-value calculations).'''

    dict_for_calculations = {'n_cells': np.array([residuals.shape[0]], dtype=float), 'f2': e_moments[0,:], 'f3': e_moments[1,:], 'f4': e_moments[2,:], 'f5': e_moments[3,:], 'cv':cv, 'e_mat':e_mat}

    dict_for_calculations['k3_matrix_new'] = ne.evaluate('(1+cv**2*e_mat*(3+cv**2*(3+cv**2)*e_mat))/(sqrt(e_mat)*(1+cv**2*e_mat)**(3/2))', dict_for_calculations)

    dict_for_calculations['k4_matrix_new'] = ne.evaluate('(1+e_mat*(3+cv**2*(7+e_mat*(6+3*cv**2*(6+e_mat)+cv**4*(6+(16+15*cv**2+6*cv**4+cv**6)*e_mat)))))/(e_mat*(1+cv**2*e_mat)**2)', dict_for_calculations)

    dict_for_calculations['k5_matrix_2_new'] = ne.evaluate('1/(e_mat**(3/2)*(1+cv**2*e_mat)**(5/2)) * (1 + 5*(2+3*cv**2)*e_mat + 5*cv**2*(8+15*cv**2+5*cv**4)*e_mat**2+10*cv**4*(6+17*cv**2+15*cv**4+6*cv**6+cv**8)*e_mat**3+cv**6*(30+135*cv**2+222*cv**4+205*cv**6+120*cv**8+45*cv**10+10*cv**12+cv**14)*e_mat**4)', dict_for_calculations)

    del dict_for_calculations['e_mat'], e_mat # Free up memory

    # Calculate kappa 2
    kappa2 = ne.evaluate('(1/(n_cells-1)**2) * f2 * n_cells', dict_for_calculations)

    # Calculate kappa 3
    dict_for_calculations['k3_crossprod'] = dict_for_calculations['k3_matrix_new'].T @ dict_for_calculations['k3_matrix_new']
    kappa3 = ne.evaluate('(1/(n_cells-1)**3) * f3 * k3_crossprod', dict_for_calculations)

    # Calculate kappa 4
    dict_for_calculations['k4_crossprod'] = dict_for_calculations['k4_matrix_new'].T @ dict_for_calculations['k4_matrix_new'] 
    kappa4 = ne.evaluate('(1/(n_cells-1)**4) * (-3*n_cells*f2**2 + f4 * k4_crossprod)', local_dict = dict_for_calculations)

    # Calculate kappa 5
    dict_for_calculations['k5_crossprod_2'] = dict_for_calculations['k5_matrix_2_new'].T @ dict_for_calculations['k5_matrix_2_new']
    kappa5 = ne.evaluate('(1/(n_cells-1)**5) * (-10*f2*f3*k3_crossprod + f5*k5_crossprod_2)', dict_for_calculations)

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
            kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
    else:
        kappa2, kappa3, kappa4, kappa5 = calculate_mcPCCs_cumulants(residuals, e_moments, e_mat, cv)
            
    return save_kappas,kappa2,kappa3,kappa4,kappa5