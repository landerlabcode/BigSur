# BigSur
BigSur is a package for principled, robust single-cell transcriptomics normalization, feature selection and correlations calculation. This ReadMe file includes a quick summary of what BigSur can be used for, along with small code examples to get started. 

# What is BigSur?
Basic Informatics and Gene Statistics from Unnormalized Reads (BigSur) is an analytical model of single-cell transcriptomics (scRNA-seq) data. This model can be used to select features and calculate correlation, taking into account the biological and technical noise inherent in scRNA-seq.
* The importance of feature selection, along with results showing BigSur performs equivalently to, if not better than, Seurat and scanpy feature selection, are shown in [Dollinger et al. 2025](https://doi.org/10.1186/s12859-025-06240-y).
* The pitfalls of using Pearson's Correlation Coefficients (PCCs) to calculate correlations in scRNA-seq data and the corrections made to PCCs to account for the noise and sparsity in these data are shown in [Silkwood et al. 2023](https://doi.org/10.1186/s12859-024-05926-z).

# Updates 
## 10/15/25
The GitHub repository now includes the code to calculate correlations. See below for the quickstart. The tutorial for the correlations will be uploaded soon. The pip package does not currently include the correlations code.

# Installation
The easiest way to install bigsur is via pip:

    conda create -n bigsur_env python pip
    conda activate bigsur_env
    pip install bigsur

Alternatively, you can clone the GitHub repo. We've included a environment file for [conda environment installation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments); the only package we require that isn't installed with scanpy is [mpmath](https://github.com/mpmath/mpmath) and [numexpr](https://github.com/pydata/numexpr). For example:

In terminal:

    cd bigsur_dir #directory to clone to

    git clone https://github.com/landerlabcode/BigSur.git

    conda create -f environment.yml -n bigsur

## A note about the virtual environment
This environment contains all packages that are required to reproduce any result of the paper. If you want a lightweight conda enviroment (or alternatively, if the environment file is causing issues), you can create a sufficient conda environment as follows:

In terminal:

    conda create -n bigsur -c conda-forge scanpy mpmath numexpr ipykernel python-igraph leidenalg

# Usage
## Feature selection
Usage for feature selection is detailed in the [example notebook](https://github.com/landerlabcode/BigSur/blob/main/feature_selection_example_usage.ipynb). 

TL;DR:

    import sys
    
    sys.path.append(bigsur_dir) # directory where git repo was cloned, not necessary if BigSur was installed using pip
    
    from BigSur.feature_selection import mcfano_feature_selection as mcfano

Replace <code>sc.pp.highly_variable_genes(adata)</code> in your pipeline with <code>mcfano(adata, layer='counts')</code>, where the UMIs are in <code>adata.layers['counts']</code>.

And that's it! You can read more about how to use BigSur for feature selection, and in particular how to optimize cutoffs for a given dataset, in the [example notebook](https://github.com/landerlabcode/BigSur/blob/main/feature_selection_example_usage.ipynb). 

## Correlations
To calculate correlations on data contained within an adata, where the UMIs are stored in <code>adata.layers['counts']</code>, run the following commands:

    import sys
    
    sys.path.append(bigsur_dir) # directory where git repo was cloned

    from BigSur.correlations import calculate_correlations

    calculate_correlations(adata, layer = 'counts', cv = None, verbose = 2, write_out=write_out_folder, previously_run=False, store_intermediate_results=True)

By default, the function stores the mcPCCs and the BH-corrected $p$-values in adata.varm. Both these matrices are lower-triangular and sparse. Given the potential size of these files, we recommend saving the mcPCCs and BH-corrected $p$-values to disk, by specifying a folder to write to, using the <code>write_out</code> parameter. See the docstring for more details. 

Since the correlations $p$-value calculation can take a long time to run and can require a lot of memory, we've included optional parameters to ensure that intermediate results are saved to disk if the application runs out of memory. The <code>store_intermediate_results</code> parameter tells the function whether to store intermediate results, such as cumulants or coefficients, in the <code>write_out</code> folder. The <conde>previously_run</code> parameter tells the function to look in that folder for any intermediate results that were previously generated.  If it is likely that the application will run out of memory, we suggest storing the intermediate results; however, some of these files are not sparse matrices and therefore can take a lot of storage space. 
