# BigSur
BigSur is a package for principled, robust scRNAseq normalization. Currently we can perform feature selection and correlations.

# What is BigSur?
Basic Informatics and Gene Statistics from Unnormalized Reads (BigSur) is a principled pipeline allowing for feature selection, correlation and clustering in scRNAseq.
* The correlation derivations are detailed in [Silkwood et al. 2023](https://doi.org/10.1101/2023.03.14.532643).
* The feature selection derivations are detailed in Dollinger and Silkwood et al. 2023 (on bioRxiv soon!).

# Installation
The only way to install BigSur currently is to clone the GitHub repo. We've included an environment.yml file for [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments); the only package we require that isn't installed with scanpy is [mpmath](https://github.com/mpmath/mpmath). 

# Usage
Usage for feature selection is detailed in the [example notebook](https://github.com/landerlabcode/BigSur/blob/main/feature_selection_example_usage.ipynb). 

TL;DR:

    from BigSur.feature_selection import mcfano_feature_selection as mcfano
Replace <code>sc.pp.highly_variable_genes(adata)</code> in your pipeline with <code>mcfano(adata, layer='counts')</code>