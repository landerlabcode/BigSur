#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Iterable
import time
import numpy as np
from numpy.polynomial import Polynomial
from anndata import AnnData
from mpmath import ncdf, exp
import numexpr as ne
import warnings
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
import scanpy as sc
import pandas as pd
from .preprocessing import make_vars_and_qc, calculate_residuals, fit_cv, calculate_mcfano

warnings.simplefilter('always', UserWarning)

def calculate_correlations(batch_dict):
        n_cells = batch_dict['All']['Barcodes'].shape[0]
        mc_fanos = batch_dict['All']['mcFanos']
        mcPCCs = 1/((n_cells - 1) * np.sqrt(mc_fanos * mc_fanos.T)) * (batch_dict['All']['Residuals'].T @ batch_dict['All']['Residuals'])

        batch_dict['mcPCCs'] = mcPCCs


def calculate_mcPCCs(adata, layer, verbose = 1, batch_key: str = None, cv = None):
    '''Calculate modified Pearson correlation coefficients (mcPCCs) for each gene pair in the dataset.'''

    # Setup        
    # Create variables
    batch_dict = make_vars_and_qc(adata, layer, batch_key = batch_key)
        
    tic = time.perf_counter()
    # Fit cv if not provided
    if cv is None:
        if verbose > 1:
            print('Fitting cv.')
        fit_cv(batch_dict, verbose)
    elif isinstance(cv, float):
        if batch_key is not None:
            raise Exception("Batch key was provided, but cv only has one value. Please pass cv as a dict of {batch:cv} pairs.")
        batch_dict['All']['CV'] = cv
    elif isinstance(cv, dict):
        if not cv.keys() <= batch_dict.keys():
            raise Exception("If providing CV as a dict, keys must match batch names.")
        for batch in cv:
            batch_dict[batch]['CV'] = cv[batch]
            
    if verbose > 1:
        for batch in batch_dict:
            # If batch is 'All' and there's more than one batch, skip it
            if (len(batch_dict) > 1) and (batch == 'All'):
                continue
            # If batch is 'All' and there's only one batch, print its CV
            elif (batch == 'All') and (len(batch_dict) == 1):
                print(f"Using a coefficient of variation of {batch_dict[batch]['CV']:.4}.")
            # If batch is not 'All', print its CV
            else:
                print(f"Using a coefficient of variation of {batch_dict[batch]['CV']:.4} for batch {batch}.")
    # Calculate residuals
    calculate_residuals(batch_dict)
    if len(batch_dict.keys()) > 1:
        batch_dict['All']['Residuals'] = np.concatenate([batch_dict[batch]['Residuals'] for batch in batch_dict if batch != 'All'])
    
    # Calculate mcfanos from residuals
    if verbose > 1:
        print("Calculating modified corrected Fano factors.")
    
    calculate_mcfano({'All':batch_dict['All']})

    toc = time.perf_counter()
    if verbose > 1:
        print(
            f"Finished calculating modified corrected Fano factors for {batch_dict['All']['mcFanos'].shape[0]} genes in {(toc-tic):04f} seconds."
        )

    # Store mc_Fano and cv
    adata.var["mc_Fano"] = np.array(batch_dict['All']['mcFanos']).flatten()

    # Calculate mcPCCs
    calculate_correlations(batch_dict)


    adata.varm['mcPCCs'] = batch_dict['mcPCCs']

    return adata

    
