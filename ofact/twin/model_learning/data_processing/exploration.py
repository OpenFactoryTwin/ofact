"""
Used to explore the data given ...

"""
from pathlib import Path

# Import 2: ...
import pandas as pd
import pygwalker as pyg

from ofact.settings import ROOT_PATH


# different levels
# - raw data
# - preprocessed data
# - processed data


def explore_data_with_pyg(df: pd.DataFrame, use_kernel_calc=True,
                          path=None,
                          result_file_name="data_exploration"):
    """
    :param df: contains the data to explore
    :param result_file_name: this json file will save your chart state,
    you need to click save button in ui manual when you finish a chart,
    'autosave' will be supported in the future.
    :param path: also used for the saving of the file ...
    :param use_kernel_calc: set `use_kernel_calc=True`, pygwalker will use duckdb as computing engine,
    it supports you explore bigger dataset(<=100GB).
    """

    if path is None:
        path = str(Path(ROOT_PATH + f"/DigitalTwin/core/model_learning/data_processing/{result_file_name}.json"))
    else:
        path = path + f"{result_file_name}.json"

    walker = pyg.walk(df,
                      # spec=path,
                      use_kernel_calc=use_kernel_calc)

    return walker
