"""Contains the tools used to handle numeric outliers in the data."""
import numpy as np
import pandas as pd


def get_outlier_detection_parameters(data_df, target_column_name, approach='iqr', **kwargs) -> pd.Series:
    """
    Get the parameter for the
    :param data_df:
    :param target_column_name:
    :param approach: outlier detection approach
    :param kwargs:
        - k: inter quantile range multiplier. The coefficient used to determine the threshold for outliers.
    :return: outlier_detection_parameter
    """

    if approach == 'iqr':
        lower, upper = _get_lower_upper_bound_iqr(data_df, target_column_name, k=kwargs["k"])
    else:
        raise Exception(f"Approach {approach} not supported.")  # For more approaches use a dict of functions ...

    if lower == upper:
        lower = None
        upper = None

    outlier_detection_parameters_s = pd.Series({"Lower Bound": lower,
                                                "Upper Bound": upper})

    return outlier_detection_parameters_s


def _get_lower_upper_bound_iqr(data_df, target_column_name, k=1.5):
    """
    Determine the lower and upper bounds of the outliers in the given DataFrame
    based on the inter-quartile range procedure.
    :param data_df: The DataFrame containing the data.
    :param target_column_name: The name of the column to handle outliers for.
    :param k: inter quantile range multiplier. The coefficient used to determine the threshold for outliers.
    :returns lower, upper: The lower and upper bounds of the outliers.
    """

    # Calculate the upper and lower limits
    data_df.reset_index(drop=True, inplace=True)
    Q1 = data_df[target_column_name].quantile(0.25)
    Q3 = data_df[target_column_name].quantile(0.75)
    IQR = Q3 - Q1
    kIQR = k * IQR
    lower = Q1 - kIQR
    upper = Q3 + kIQR

    # print(f"Lower: {lower}, Upper: {upper}")

    return lower, upper


def handle_numeric_outliers(data_df, target_column_name, outlier_detection_parameters):
    """
    Handle numeric outliers in the given DataFrame.
    :param data_df: The DataFrame containing the data.
    :param target_column_name: The name of the column to handle outliers for.
    :returns data_df: The DataFrame with the outliers removed.
    """

    # Create arrays of Boolean values indicating the outlier rows
    if outlier_detection_parameters["Lower Bound"] is not None:
        lower_array = np.where(data_df[target_column_name] <= outlier_detection_parameters["Lower Bound"])[0]
    else:
        lower_array = np.array([])

    if outlier_detection_parameters["Upper Bound"] is not None:
        upper_array = np.where(data_df[target_column_name] >= outlier_detection_parameters["Upper Bound"])[0]
    else:
        upper_array = np.array([])

    # Removing the outliers
    data_df.reset_index(drop=True, inplace=True)
    data_df = data_df.drop(index=upper_array)
    if outlier_detection_parameters["Lower Bound"] == outlier_detection_parameters["Upper Bound"]:
        lower_array = np.setdiff1d(lower_array, upper_array)
    data_df = data_df.drop(index=lower_array)

    return data_df
