import numpy as np
import mpmath

## Scipy
from scipy.stats import norm

## Joblib
from joblib import Parallel, delayed


def calculate_pvalues(correlation_roots, n_jobs = -2):
    '''Calculate p-values from correlation roots.'''
    print("Estimating p-values.")
    p = norm.logcdf(correlation_roots)
    p_mpfr = -np.log10(-p / np.log(10))
    p_mpfr[correlation_roots < 8.2] = -np.log10(1 - np.exp(p[correlation_roots < 8.2]))

    def robust_pvalue(correlation_root):
        p_value = float(mpmath.nstr(-mpmath.log10(0.5 * mpmath.exp(- (mpmath.power(mpmath.mpf(correlation_root), 2)) / 2)), 15))
        return p_value
    
    p_mpfr[correlation_roots >= 38.4] = np.array(Parallel(n_jobs=n_jobs)(delayed(robust_pvalue)(correlation_roots[correlation_row]) for correlation_row in np.where(correlation_roots >= 38.4)[0]))

    p_values = np.array(10**-p_mpfr)
    return p_values

def BH_correction(p_values, num_genes):
    '''Do Benjamini-Hochberg correction on p-values.'''
    indices_of_smallest_to_greatest_p_values = p_values.argsort()
    recovery_index = np.argsort(indices_of_smallest_to_greatest_p_values)
    sorted_pvals = p_values[indices_of_smallest_to_greatest_p_values]
    BH_corrected_pvalues = sorted_pvals * ((num_genes * (num_genes - 1)) / 2) / np.arange(1, len(sorted_pvals) + 1)
    BH_corrected_pvalues[BH_corrected_pvalues > 1] = 1
    BH_corrected_pvalues_reordered = BH_corrected_pvalues[recovery_index]
    return BH_corrected_pvalues_reordered