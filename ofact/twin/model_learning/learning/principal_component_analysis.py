"""
Contains the staff needed for the Principal Component Analysis (PCA).
Use Case:
    - In general the PCA is used for dimensionality reduction.
    - The PCA is used as predecessor step before the learning.
Conditions:
    - Normalized data
@Last update: 17.10.2023
"""

from typing import Optional

import torch
from sklearn.decomposition import PCA


def get_pca(input_data: torch.tensor, n_components: int, pca: Optional[PCA] = None, print_per_component=False) -> \
        [PCA, torch.tensor]:
    """
    The Principal Component Analysis (PCA) is used to raise the variance in the dimensions of the data.
    This is done to simplify the procedure of learning the model.
    :param pca: PCA
    :param input_data: data used as input
    :param n_components: number of dimensions as output
    :param print_per_component: determine if the print is done
    :return pca: fitted pca object
    :return fitted_pca: data relocated by pca
    """

    if pca is None:
        pca = PCA(n_components=n_components)
    fitted_pca = pca.fit_transform(input_data)
    fitted_pca = torch.tensor(fitted_pca)

    exp_variance = torch.tensor(pca.explained_variance_ratio_)
    print(f"Total variance explained: {torch.sum(exp_variance):.5} ({n_components} dimensions)")
    if print_per_component:
        print(f"\nPer component: {exp_variance}")

    return pca, fitted_pca
