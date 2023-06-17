import numpy as np
from scipy.special import erf

from .feature_selection import make_vars_and_qc, fit_cv, calculate_residuals, calculate_mcfano

# check if mc_fanos are in adata.var
## If not, do mc_fanos; for right now don't do pvalues
## corrected_modified_correlations = DotProductCF[Evaluate[pmatrix/Sqrt[(n - 1) mflist]], n] # 
## For now: fanocorrectionmatrices = Table[ ConstantArray[1, {m, m}], 4] 
## ccmats = CorrelationCumulantsNewCF[ematrix, fanocorrectionmatrices[[1]], fanocorrectionmatrices[[2]], fanocorrectionmatrices[[3]], fanocorrectionmatrices[[4]], c, n]
## polymatrix = CornishFisherPolynomialCoefficients5CF[ccmats[[1]], ccmats[[2]], ccmats[[3]], ccmats[[4]] , pccmatrix] # cornish fisher polynomial coefficients
## Find roots function, skip for now
## logpmatrix = function(roots), finds p values given roots. THen take log? Why? Might be machine precision issues
## EquivalentPPCs = function(logpmat, pccmatrix)

class correlations_object:
    def __init__(self, residuals, corrected_fanos, n_cells, emat, cv):
        self.residuals = residuals
        self.corrected_fanos = corrected_fanos
        self.n_cells = n_cells
        self.n_genes = self.corrected_fanos.shape[0]
        self.emat = emat
        self.cv = cv

def calculate_correlations(adata, layer, cv = 0.5, inverse_moments = False, verbose = 1):
    '''Main function for calculating correlations'''
    correlations_store = prep_correlations_object(adata, layer, cv, verbose)
    mc_correlations(correlations_store) # Maybe optimize, make lower triangular?
    inverse_sqrt_moments(correlations_store, inverse_moments) 
    cornish_fisher_polynomial_coefficient(correlations_store)
    calculate_log_p_values(correlations_store)
    calculate_equivalent_pccs(correlations_store)


def prep_correlations_object(adata, layer, cv, verbose):
    if ['residuals', 'corrected_fanos'] in adata.var.columns:
        print('mc_fanos already calculated, continue')
        correlations_store = correlations_object(adata.var['residuals'], adata.var['corrected_fanos'], adata.shape[0])
    else:
        print('Need to calculate mc_fanos before correlations:')
        raw_count_mat, means, variances, g_counts = make_vars_and_qc(adata, layer)
        if verbose > 1:
            print("Calculating corrected Fano factors.")

        if cv is None:
            cv = fit_cv(raw_count_mat, means, variances, g_counts, verbose)

        cv, normlist, residuals, n_cells = calculate_residuals(cv, verbose, raw_count_mat, means, variances, g_counts)
        wlist = len(normlist) * normlist
        emat = np.outer(means, wlist)
        corrected_fanos = calculate_mcfano(residuals, n_cells)
        correlations_store = correlations_object(residuals, corrected_fanos, n_cells, emat, cv)
    return correlations_store

def mc_correlations(correlations_store):
    # Prep variables
    residuals = correlations_store.residuals
    n_cells = correlations_store.n_cells
    corrected_fanos = correlations_store.corrected_fanos
    matrix_for_dotproduct = (residuals/((n_cells-1)*corrected_fanos))**(-1/2)
    mc_correlations_matrix = np.dot(matrix_for_dotproduct, n_cells)
    correlations_store.mc_correlations_matrix
    # Don't need residuals after this, consider deleting if space is an issue

def inverse_sqrt_moments(correlations_store, inverse_moments):
    if inverse_moments:
        # do stuff here
        print('Skipping for right now')
    else:
        correlations_store.f2 = np.ones((correlations_store.n_genes, correlations_store.n_genes))
        correlations_store.f3 = np.ones((correlations_store.n_genes, correlations_store.n_genes))
        correlations_store.f4 = np.ones((correlations_store.n_genes, correlations_store.n_genes))
        correlations_store.f5 = np.ones((correlations_store.n_genes, correlations_store.n_genes))
    

def cumulants_of_correlations(correlations_store):
    # Prep variables
    f2 = correlations_store.f2
    f3 = correlations_store.f3
    f4 = correlations_store.f4
    f5 = correlations_store.f5
    emat = correlations_store.emat
    n_cells = correlations_store.n_cells
    cv = correlations_store.cv

    # Start calculations
    k2 = 1/(-1+n_cells)**2 * f2 * n_cells
    
    numerator_for_dotproduct_k3 = (1 + cv**2 * emat * (3 + cv**2 * (3 + cv**2) * emat))
    denominator_for_dotproduct_k3 = (emat**(-1/2) * (1 + cv**2 * emat)**(3/2))
    dotproduct_for_k3 = np.dot(numerator_for_dotproduct_k3/denominator_for_dotproduct_k3, n_cells)
    k3 = 1/(-1+n_cells)**3 * f3 * dotproduct_for_k3
    
    numerator_for_dotproduct_k4 = (1 + emat * (3 + cv**2 * (7 + emat * (6 + 3 * cv**2 * (6 + emat) + cv**4 * (6 + (16 + 15 * cv**2 + 6 * cv**4 + cv**6) * emat)))))
    denominator_for_dotproduct_k4 = (emat * (1+cv**2 * emat)**2)
    k4 = 1/(-1 + n_cells)**4 * (-3 * n_cells * f2**2  + f4 * np.dot(numerator_for_dotproduct_k4 / denominator_for_dotproduct_k4, n_cells))
    
    numerator_for_dotproduct_k5 = (1 + 5 * (2 + 3 * cv**2) * emat + 5 * cv**2 * (8 + 15 * cv**2 + 5 * cv**4) * emat**2 + 10 * cv**4 * (6 + 17 * cv**2 + 15 * cv**4 + 6 * cv**6 + cv**8) * emat**3 + cv**6 * (30 + 135 * cv**2 + 222 * cv**4 + 205 * cv**6 + 120 * cv**8 + 45 * cv**10 + 10 * cv**12 + cv**14) * emat**4)
    denominator_for_dotproduct_k5 = emat**(3/2) * (1 + cv**2 * emat)**(5/2)
    k5 = 1/((-1 + n_cells)**5) * (-10 * f2 * f3 * dotproduct_for_k3 + f5 * np.dot(numerator_for_dotproduct_k5/denominator_for_dotproduct_k5))

    # Store
    correlations_store.k2 = k2
    correlations_store.k3 = k3
    correlations_store.k4 = k4
    correlations_store.k5 = k5

def cornish_fisher_polynomial_coefficient(correlations_store):
    # Vars: the 4 ks and mc_correlations_matrix
    # Returns correlation coffeicients for each of the moments I think?
    
    # Prep variables
    k2 = correlations_store.k2
    k3 = correlations_store.k3
    k4 = correlations_store.k4
    k5 = correlations_store.k5
    mc_correlations_matrix = correlations_store.mc_correlations_matrix

    # Prep math
    k2_cubed = k2**3
    k2_to_4th = k2**4
    k3_cubed = k3**3
    k2_squared = k2**2
    k2_to_3_over_2 = k2**(3/2)
    k3_squared = k3**2
    k2_to_5_over_2 = k2**(5/2)
    k2_cubed = k2**3
    k2_squared = k2**2

    # Start calculations
    coefficents_1 = - mc_correlations_matrix - k3/(6*k2) + (17*k3_cubed)/(324*k2_to_4th) - (k3*k4)/(12*k2_cubed) + k5/(40*k2_squared)
    coefficents_2 = k2**(-1/2) + (5*k3_squared)/(36*k2_to_5_over_2) - k4/(8*k2_to_3_over_2)
    coefficents_3 = k3/(6*k2) - 53*k3_cubed/(324*k2_to_4th) + 5*k3*k4/(24*k2_cubed) - k5/(20*k2_squared)
    coefficents_4 = - k3_squared/(18*k2_to_5_over_2) + k4/(24*k2_to_3_over_2)
    coefficents_5 = k3_cubed/(27*k2_to_4th) - (k3 * k4)/(24 * k2_cubed) + k5/(120*k2_squared)

    # Store calculations
    correlations_store.coefficents_1 = coefficents_1
    correlations_store.coefficents_2 = coefficents_2
    correlations_store.coefficents_3 = coefficents_3
    correlations_store.coefficents_4 = coefficents_4
    correlations_store.coefficents_5 = coefficents_5

    # Here consider deleting ks if space becomes an issue

def calculate_log_p_values(correlations_store):
    x_list = correlations_store.x_list

    log_p_values = np.empty(x_list.shape[0])
    log_vec = np.abs(x_list) > 8.3
    log_p_values[log_vec] = 1 + (x_list[log_vec]**2/2 + np.log(2))/np.log(10)
    log_p_values[~log_vec] = -np.log10(1/2 * (1 - erf(np.abs(x_list)/np.sqrt(2))))
    correlations_store.log_p_values

def calculate_equivalent_pccs(correlations_store):
    # No idea what's going on here







