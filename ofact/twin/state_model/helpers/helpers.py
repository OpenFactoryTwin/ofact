"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################

Provides general functions that can be used for different issues

@contact persons: Adrian Freiter
@last update: 08.11.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Union
from ast import literal_eval
import pathlib
import platform


# Imports Part 2: PIP Imports
import dill as pickle
import numpy as np
import pandas as pd

available_pd_file_readers = {"csv": pd.read_csv,
                             "xlsx": pd.read_excel}


def get_file_type(file_path):
    if isinstance(file_path, pathlib.Path):
        suffixes_file_path = file_path.suffixes

        if len(suffixes_file_path) != 1:
            raise NotImplementedError(f"For the suffixes '{suffixes_file_path}' too many suffixes are given ...")

        suffix = suffixes_file_path[0][1:]

    else:
        suffix = file_path.split(".")[-1]

    return suffix


def get_pd_file_reader(file_path) -> Union[pd.read_csv, pd.read_excel]:
    suffix = get_file_type(file_path)
    if suffix not in available_pd_file_readers:
        raise NotImplementedError(f"For the suffix '{suffix}' no reader is implemented ... \n"
                                  f"Please choose one of the following file types: "
                                  f"{list(available_pd_file_readers.keys())}")

    file_reader = available_pd_file_readers[suffix]
    return file_reader


def load_from_pickle(pickle_path):
    """Assumed the pickle object is created on windows ..."""

    system_name = platform.system()
    print("Platform name:", system_name)

    if system_name == "Windows":
        with open(pickle_path, 'rb') as inp:
            loaded_object = pickle.load(inp)

    elif system_name == "Linux":
        windows_backup = pathlib.WindowsPath
        try:
            pathlib.WindowsPath = pathlib.PosixPath
            with open(pickle_path, 'rb') as inp:
                loaded_object = pickle.load(inp)
        finally:
            pathlib.WindowsPath = windows_backup

    return loaded_object


def convert_lst_of_lst_to_lst(lst_of_lst):
    """conversion from list of lists to list of elements"""
    if lst_of_lst:
        if isinstance(lst_of_lst[0], list):
            lst = [object_ for lst in lst_of_lst for object_ in lst if object_]
            return lst
        else:
            return lst_of_lst
    else:
        return lst_of_lst


def convert_to_datetime(input_: Union[datetime, np.datetime64, int, float, str]) -> datetime:
    """Conversion to np.datetime64"""

    if isinstance(input_, datetime):
        return input_
    elif isinstance(input_, np.datetime64):
        biased_time = (input_ - np.datetime64('1970-01-01T00:00:00')) \
                if input_ != np.datetime64('0001-01-01T00:00:00') else np.timedelta64(0, "s")
        datetime_object = datetime.utcfromtimestamp(biased_time / np.timedelta64(1, 's'))

        return datetime_object
    elif isinstance(input_, int) or isinstance(input_, float):
        raise NotImplementedError("More information like unit needed!", input_)
        # return np.datetime64(input_)
    elif isinstance(input_, str):
        # for more than one time format see:
        # https://stackoverflow.com/questions/23581128/how-to-format-date-string-via-multiple-formats-in-python
        return datetime.strptime(input_, '%Y-%m-%d %H:%M:%S.%f')
    elif input_ is None:
        ValueError("None cannot be converted!")

    raise ValueError("Format is not compatible!")


def convert_to_np_datetime(input_: np.datetime64 | int | float | datetime) -> np.datetime64:
    """Conversion to np.datetime64"""
    if isinstance(input_, np.datetime64):
        return input_
    elif isinstance(input_, int) or isinstance(input_, float):
        raise NotImplementedError("More information like unit needed!")
        # return np.datetime64(input_)
    elif isinstance(input_, datetime):
        return np.datetime64(input_)

    raise ValueError("Format is not compatible!")


def convert_to_np_timedelta(input_: Union[np.timedelta64, int, float, timedelta]) -> np.timedelta64:
    """Conversion to np.timedelta64"""
    if isinstance(input_, np.timedelta64):
        return input_
    elif isinstance(input_, int) or isinstance(input_, timedelta):
        return np.timedelta64(input_, "s")
    elif isinstance(input_, float):
        return np.timedelta64(int(input_), "s")

    raise ValueError("Format is not compatible!")


def handle_bool(value):
    try:
        bool_value = literal_eval(value)
    except:
        bool_value = None

    return bool_value


def handle_str(value):
    try:
        str_value = str(value)
    except:
        str_value = None

    return str_value


def handle_numerical_value(str_element: Union[str, float, int]):
    """Convert a possible string element to a float or integer"""

    if str_element is None:
        # print(f"{str_element} is set to 0 because it cannot be evaluated as a float")
        float_element = 0
        return float_element

    if isinstance(str_element, float) or isinstance(str_element, int):  # note int is also allowed ...
        return str_element

    float_element = None

    if "/" in str_element:
        str_elements = str_element.split("/")
        float_element = handle_numerical_value(str_elements[0]) / handle_numerical_value(str_elements[1])

    elif isinstance(str_element, str):
        str_element = str_element.replace(",", ".")
        if str_element != "0" and "." not in str_element:
            str_element = str_element.lstrip("0")
        try:
            float_element = literal_eval(str_element)
        except:
            # print(f"{str_element} is set to 0 because it cannot be evaluated as a float")
            float_element = 0

    if float_element is None:
        raise Exception(float_element)

    return float_element


def get_clean_attribute_name(attribute_name: str) -> str:
    """
    Create a clean attribute name means that the '_'-string character at the beginning of the attribute name is removed.
    Needed to ensure a common language because some attributes are private and other public etc.

    Parameters
    ----------
    attribute_name: name of the attribute that can be a private attribute

    Returns
    -------
    attribute_name: cleaned attribute name
    """

    if attribute_name[0] != '_':
        return attribute_name

    while True:
        attribute_name = attribute_name[1:]
        if attribute_name[0] != '_':
            return attribute_name
