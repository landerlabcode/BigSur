# BigSur
BigSur is a package for principled, robust scRNAseq normalization. Currently we can perform feature selection, see [BigSurR](https://github.com/landerlabcode/BigSurR) for correlations.

# What is BigSur?
Basic Informatics and Gene Statistics from Unnormalized Reads (BigSur) is a principled pipeline allowing for feature selection, correlation and clustering in scRNAseq.
* The correlation are detailed in [Silkwood et al. 2023](https://doi.org/10.1101/2023.03.14.532643).
* The feature selection derivations are detailed in [the BioRxiv paper Dollinger et al. 2023](https://www.biorxiv.org/content/10.1101/2024.10.11.617709v1).

# Installation
The only way to install BigSur currently is to clone the GitHub repo. We've included an environment.yml file for [conda environment installation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments); the only package we require that isn't installed with scanpy is [mpmath](https://github.com/mpmath/mpmath) and [numexpr](https://github.com/pydata/numexpr).
For example:

In terminal:

    cd bigsur_dir #directory to clone to

    git clone https://github.com/landerlabcode/BigSur.git

# Usage
Usage for feature selection is detailed in the [example notebook](https://github.com/landerlabcode/BigSur/blob/main/feature_selection_example_usage.ipynb). 

TL;DR:

    import sys
    
    sys.path.append(bigsur_dir) # directory where git repo was cloned
    
    from BigSur.feature_selection import mcfano_feature_selection as mcfano

Replace <code>sc.pp.highly_variable_genes(adata)</code> in your pipeline with <code>mcfano(adata, layer='counts')</code>, where the UMI counts are in <code>adata.layers['counts']</code>.

And that's it! You can read more about how to use BigSur for feature selection, and in particular how to optimize cutoffs for a given dataset, in the [example notebook](https://github.com/landerlabcode/BigSur/blob/main/feature_selection_example_usage.ipynb). 
