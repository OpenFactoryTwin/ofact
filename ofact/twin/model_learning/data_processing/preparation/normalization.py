"""Contains the tools used to normalize the data."""

from __future__ import annotations

from typing import Union, Optional, Dict

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd


def get_normalization_parameters(data_df, approach='min_max', **kwargs) -> pd.DataFrame:
    """
    Get the parameter for the
    :param data_df:
    :param approach: outlier detection approach
    :param kwargs:
        - k: inter quantile range multiplier. The coefficient used to determine the threshold for outliers.
    :return: outlier_detection_parameter
    """

    if approach == 'min_max':
        normalization_parameters_df = get_min_max_parameters_df(data_to_normalize=data_df)
    else:
        raise Exception(f"Approach {approach} not supported.")  # For more approaches use a dict of functions ...

    return normalization_parameters_df


def get_min_max_parameters_df(data_to_normalize: pd.DataFrame):
    """Determine the min and max value based on a df"""

    data_min = data_to_normalize.min()
    data_max = data_to_normalize.max()

    normalization_parameters_df = pd.DataFrame({"min": data_min,
                                                "max": data_max})

    return normalization_parameters_df


def normalize_min_max(data_to_normalize: Union[np.array, pd.Series]) -> Union[np.array, pd.Series]:
    """
    Normalize the given data using the min-max scaling method.
    :param data_to_normalize: The data to be normalized.
    :return: The normalized data.
    """
    data_min = np.min(data_to_normalize)  # ToDo
    data_max = np.max(data_to_normalize)

    data_normalized = (data_to_normalize - data_min + 1e-12) / (data_max - data_min + 1e-12)
    return data_normalized


def normalize_min_max_df(data_to_normalize: pd.DataFrame,
                         normalization_parameters_df: Optional[pd.DataFrame] = None) -> (
        Union[np.array, pd.Series]):
    """
    Normalize the values of a given dataframe between 0 and 1.
    :param data_to_normalize: The dataframe containing the data to be normalized.
    :param normalization_parameters_df: provide the min and max values for each column that need normalization
    :returns The normalized data.
    """

    data_normalized = ((data_to_normalize - normalization_parameters_df["min"] + 1e-12) /
                       (normalization_parameters_df["max"] - normalization_parameters_df["min"] + 1e-12))
    # set values zero that are one because they have no variance
    data_normalized[normalization_parameters_df["max"][(normalization_parameters_df["max"] ==
                                                        normalization_parameters_df["min"])].index] = 0

    return data_normalized


def normalize_min_max_a(data_to_normalize: pd.DataFrame,
                        normalization_parameters_dict: Optional[Dict] = None) -> np.array:
    """With array as input"""
    data_normalized = ((data_to_normalize - normalization_parameters_dict["min"] + 1e-12) /
                       (normalization_parameters_dict["max"] - normalization_parameters_dict["min"] + 1e-12))
    # set values zero that are one because they have no variance
    data_normalized[normalization_parameters_dict["max"] == normalization_parameters_dict["min"]] = 0
    return data_normalized


def enforce_values_between_zero_and_one_df(data_df: pd.DataFrame):
    # condition is False values are replaced

    data_df = data_df.where(data_df <= 1, 1)
    data_df = data_df.where(data_df >= 0, 0)

    return data_df


def enforce_values_between_zero_and_one_a(data_a: pd.Series):
    # condition is False values are replaced

    data_a[1 < data_a] = 1
    data_a[data_a < 0] = 0

    return data_a
