#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:55:15 2022

@author: emmanueldollinger
"""
import numpy as np
from scipy.stats import linregress
from sklearn.utils.sparsefuncs import mean_variance_axis
import numexpr as ne
import warnings

warnings.simplefilter('always', UserWarning)

def check_means_and_counts(means, g_counts):
    """Check means and gene counts for QC."""
    ### Check if any means == 0, probably means QC was not done
    if np.any(means == 0):
        raise Exception("Zero gene means were found, remove genes with no counts before calculating mcFanos.")

    ### Check if integer data was passed, if not probably passed normalized data
    if any(g_counts.astype(int) != g_counts):
        raise Exception("This function takes raw counts, normalized data was passed.")

def make_vars_and_qc(adata, layer, batch_key = None):
    """Extracts raw_count_mat, means, variances, g_counts and does checks for QC fail and raw counts."""
    if layer == "X":
        raw_count_mat = adata.X.copy()
    else:
        raw_count_mat = adata.layers[layer].copy()

    # We've had issues with numerical precision, so we cast everything to float64
    raw_count_mat = raw_count_mat.astype(np.float64)

    batch_dict = {}

    batch_dict['All'] = {}

    barcodes_list = np.empty(0)

    if batch_key is not None:
        batch_list = adata.obs[batch_key].values
    else:
        batch_list = np.repeat('All', adata.shape[0])
    
    for batch in np.unique(batch_list):
        batch_mask = batch_list == batch
        raw_count_mat_batch = raw_count_mat[batch_mask, :]
        means, variances = mean_variance_axis(raw_count_mat_batch, axis=0)
        g_counts = np.asarray(raw_count_mat_batch.sum(axis=0)).flatten()

        check_means_and_counts(means, g_counts)

        temp_dict = {'Batch':batch,'Means':means, 'Variances':variances, 'Gene counts':g_counts, 'Raw counts':raw_count_mat_batch}
        barcodes_list = np.append(barcodes_list, adata.obs.index[batch_mask].values)
        batch_dict[batch] = temp_dict

    batch_dict['All']['Batch_vector'] = batch_list
    batch_dict['All']['Barcodes'] = barcodes_list

    return batch_dict


def calculate_residuals(batch_dict):
    """Calculate the corrected modified residuals ."""
    # Correct for differential read depth among cells (calculating cell-specific expected gene means)
    for batch in batch_dict:
        # If batch is 'All' and there's more than one batch, skip it
        if batch == 'All' and len(batch_dict.keys()) > 1:
            continue
        calculate_emat(batch_dict[batch])
        dense = batch_dict[batch]['Raw counts'].toarray()

        cv = batch_dict[batch]['CV']
        e_mat = batch_dict[batch]['e_mat']
        residuals = ne.evaluate(
            "(dense-e_mat)/(e_mat*(1+e_mat*cv**2))**(1/2)",
            {"dense": dense, "e_mat": e_mat, "cv": cv},
    )
        batch_dict[batch]['Residuals'] = residuals

def calculate_emat(batch_dict_subset):
    '''Calculate the expectation matrix (e_mat). batch_dict here is a subset of the main batch_dict for a specific batch.'''
    raw_count_mat = batch_dict_subset['Raw counts']
    g_counts = batch_dict_subset['Gene counts']
    total_umi = np.array(raw_count_mat.sum(axis=1)).flatten()
    normlist = total_umi / raw_count_mat.sum()
    # Modify Fano factors by accounting for differential read depth and dividing 1+c^2*mu
    n_cells = normlist.shape[0]
    e_mat = np.outer(normlist, g_counts)
    batch_dict_subset['e_mat'] = e_mat


def fit_cv(batch_dict, verbose, min_mean = 0.1, max_mean = 100):
    '''Fits CV to genes with means > min_mean and means < max_min. Slope of linear fit in mcFano vs mean should be 0, so try different CVs and pick the CV with slope closest to zero.'''

    for batch in batch_dict:
        # If batch is 'All' and there's more than one batch, skip it
        if batch == 'All':
            if len(batch_dict.keys()) > 1:
                continue

        means = batch_dict[batch]['Means']

        log_vec = np.logical_and(means > min_mean , means < max_mean)

        subset_batch_dict = {batch: batch_dict[batch].copy()}

        for key in ['Means', 'Gene counts', 'Variances']:
            subset_batch_dict[batch][key] = subset_batch_dict[batch][key][log_vec]
        subset_batch_dict[batch]['Raw counts'] = subset_batch_dict[batch]['Raw counts'][:, log_vec]
        cv_store = 1.0
        slope_store = 1.0
        for cv_try in np.arange(0.05, 1.05, 0.05):
            if 'CV' in batch_dict[batch]:
                break
            elif cv_try == 1.05:
                warnings.warn(
                    'CV cannot be fit in biological range -- this probably means that the dataset is composed of multiple celltypes. We recommend subsetting the celltypes and redoing CV fit. Setting CV = 0.5.')
                cv = 0.5
                batch_dict[batch]['CV'] = cv
                break
            cv_try = np.round(cv_try, 3)
            subset_batch_dict[batch]['CV'] = cv_try
            
            calculate_residuals(subset_batch_dict)

            calculate_mcfano(subset_batch_dict)

            fit_object = linregress(np.log10(subset_batch_dict[batch]['Means']), np.log10(subset_batch_dict[batch]['mcFanos']))

            slope = fit_object[0]

            if slope < 0:
                if slope_store < np.abs(slope):
                    cv = cv_store
                else:
                    cv = cv_try
                batch_dict[batch]['CV'] = cv
            else:
                cv_store = cv_try
                slope_store = slope

def calculate_mcfano(batch_dict):
    '''Calculate modified corrected Fano factors.'''
    for batch in batch_dict:
        squared_residuals = batch_dict[batch]['Residuals']**2
        modified_corrected_fanos = 1 / (batch_dict[batch]['Residuals'].shape[0] - 1) * np.sum(squared_residuals, axis=0)
        batch_dict[batch]['mcFanos'] = np.array(modified_corrected_fanos).flatten()