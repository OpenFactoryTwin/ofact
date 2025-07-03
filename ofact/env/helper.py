from datetime import datetime

import numpy as np
import pandas as pd
from math import ceil


def np_datetime64_to_datetime(np_datetime64_ns, round_ceil_microseconds=True) -> datetime:
    """Conversion of np.datetime64 to datetime object"""

    pandas_timestamp = pd.to_datetime(np_datetime64_ns, unit="ns")
    nanoseconds = np_datetime64_ns.astype(np.int64) % 1000000000

    datetime_object = pandas_timestamp.to_pydatetime()
    # if round_ceil_microseconds:
    #     microseconds = int(ceil(nanoseconds * 1e-3))
    # else:
    microseconds = round(nanoseconds * 1e-3)
    datetime_object = datetime_object.replace(microsecond=microseconds)

    return datetime_object
