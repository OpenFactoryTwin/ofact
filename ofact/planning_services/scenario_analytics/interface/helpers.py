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

Helpers for the frontend api.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations
from datetime import datetime, timedelta
from functools import wraps

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd
from cerberus import Validator


# ====Helper Methods====================================================================================================
def validate_request(validation_scheme: dict, input_document: dict):
    """
    The input_document dictionary will be matched with a given validation_scheme.
    :param validation_scheme: A scheme that describes the structure of a dictionary
    :param input_document: A dictionary that should be matched against the validation_scheme
    :raise TypeError: The input_document did not match the given validation_scheme.
    """
    v = Validator(validation_scheme)
    if v.validate(input_document):
        print("[API]\t>input valid")
        return input_document
    else:
        print("[API]\t>input invalid " + str(v.errors))
        raise TypeError("Input could not be matched: " + str(v.errors))


# ==== time conversion
def get_timestamp(datetime_):
    """
    Convert datetime to timestamp
    :return a timestamp
    """
    return datetime.timestamp(datetime_)


def get_datetime(timestamp_):
    """
    Convert timestamp to datetime
    :return a datetime
    """
    return datetime.fromtimestamp(timestamp_)


def argument_preparation(func):
    @wraps(func)
    def numbers_to_int(*args, **kwargs):
        """query input arguments for several number formats and replace values with an integer representative"""
        new_kwargs_dict = {key: prepare_dict_entry(value)
                           for key, value in kwargs.items()}

        return func(*args, **new_kwargs_dict)

    return numbers_to_int


def prepare_dict_entry(value):
    if isinstance(value, timedelta):
        new_value = int(value.total_seconds())
    elif value != value:
        # check NaN
        new_value = 0
    elif isinstance(value, float):
        new_value = round(value, 1)
        # round to 3 decimals after point - notice: not the normal round
    elif isinstance(value, np.integer):
        new_value = int(value)
    elif isinstance(value, np.floating):
        new_value = int(value)
    elif isinstance(value, np.ndarray):
        new_value = int(value)
    elif isinstance(value, list):
        new_value = value
    elif pd.isnull(value):
        new_value = 0
    else:
        new_value = value

    return new_value
