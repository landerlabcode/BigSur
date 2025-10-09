from turtle import pd
import numpy as np

### Scipy
from scipy.interpolate import interp1d

# Functions for inverse square moment interpolation
def inv_sqrt_moment_interpolation(sample_moments, gene_totals, points):
    '''Interpolate the moments of the inverse square root mcFano factors.'''
    int_moments = np.empty((4, gene_totals.shape[0]))
    for j in range(4):
        approx_func = interp1d(
            np.log10(points),
            np.log10(sample_moments[:, j]),
            kind='linear',
            fill_value='extrapolate'
        )
        interpolated = np.power(10, approx_func(np.log10(gene_totals)))
        int_moments[j, :] = interpolated
    e_moments = np.array([np.outer(m, m) for m in int_moments]) # 4 x n_genes x n_genes
    return e_moments
def inverse_sqrt_mcfano_correction(n_cells, g_counts, c, normlist):
    '''Calculate the interpolated moments of the product of inverse square root mcFano factors.'''
    a = max(2, min(g_counts))
    e = n_cells / 50
    h = max(g_counts)
    points = np.array([a, a * (e / a) ** (1 / 4), a * (e / a) ** (1 / 2), a * (e / a) ** (3 / 4), e, e * (h / e) ** (1 / 3), e * (h / e) ** (2 / 3), h], dtype=int) # 8 points
    trials = 4*10**7/(n_cells*(np.log10(points)**(1/5)+0.5*np.log10(points)**3))
    trials = trials.astype(int) # convert to ints

    sim_emat = np.outer(points, normlist) # 8 x n_cells

    sample_moments = np.array([simulate_inverse_sqrt_mcfano_moments(sim_emat[i,:], c, n_cells, trials[i]) for i in range(points.shape[0])])

    #import pandas as pd

    #sample_moments_R = pd.read_csv("/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_R_testing/inv_correction.csv", index_col = 0).to_numpy()

    #sample_moments_R = sample_moments_R.T # If I load in moments from R, the interpolated moments in R and python are equal

    e_moments = inv_sqrt_moment_interpolation(sample_moments, g_counts, points)

    # import pandas as pd
    
    # e_moments_1 = pd.read_csv( "/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_R_testing/moment_interp_1.csv", index_col = 0)
    # e_moments_2 = pd.read_csv( "/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_R_testing/moment_interp_2.csv", index_col = 0)
    # e_moments_3 = pd.read_csv( "/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_R_testing/moment_interp_3.csv", index_col = 0)
    # e_moments_4 = pd.read_csv( "/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_R_testing/moment_interp_4.csv", index_col = 0)

    # e_moments_R = np.array([e_moments_1.to_numpy(), e_moments_2.to_numpy(), e_moments_3.to_numpy(), e_moments_4.to_numpy()])

    # np.savetxt('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/moment_interp_1.csv', e_moments[0,:,:], delimiter=',')
    # np.savetxt('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/moment_interp_2.csv', e_moments[1,:,:], delimiter=',')
    # np.savetxt('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/moment_interp_3.csv', e_moments[2,:,:], delimiter=',')
    # np.savetxt('/Users/emmanueldollinger/Documents/Projects/Pipeline_development/Data/results/lymph_nodes/correlations/correlations_python_testing/moment_interp_4.csv', e_moments[3,:,:], delimiter=',')

    return e_moments#e_moments#e_moments_R#e_moments#e_moments#e_moments#e_moments_R#e_moments#e_moments_R#e_moments# e_moments_R#e_moments_R #
def simulate_inverse_sqrt_mcfano_moments(sim_emat_subset, c, n_cells, trial, starting_seed = 0):
    '''Simulate inverse square root mcFano factors and calculate their moments.'''
    mu = np.log(sim_emat_subset / np.sqrt(1 + c**2))
    sigma = np.sqrt(np.log(1 + c**2)) # This is std

    rng = np.random.default_rng(starting_seed)

    PLN_samples = rng.poisson(rng.lognormal(mean=mu.reshape(1,-1), sigma=sigma), size = (trial, n_cells)) # gene x cells?
    samples = 1/np.sqrt(np.sum((PLN_samples-sim_emat_subset)**2/(sim_emat_subset+c**2*sim_emat_subset**2), axis = 1)/(n_cells-1))

    results = [np.mean(samples**n) for n in range(1, 5)] # Return the first through 4th moments

    return(results)