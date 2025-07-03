"""
The module comprises helper classes and functions. These can be used in arbitrary contexts.
Classes:
    Singleton: each class which inherit from the singleton class is guaranteed to have at most one instance.
Functions:
    colored_print: colored print in the console
@last update: ?.?.2022
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import Union
from weakref import WeakValueDictionary
import cProfile, pstats, io
from datetime import datetime, timedelta
import pytz

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports


# code used for debugging
# Used to identify the caller method/ function
# import inspect
# curframe = inspect.currentframe()
# calframe = inspect.getouterframes(curframe, 2)
# print('caller name:', calframe[1][3])


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        max_characters = 5000  # None

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue()[:max_characters])
        return retval

    return inner


class Singleton(type):
    """
    Singleton can be used as metaclass, if you want to ensure that only one object is created and used for reference.

    example Usage:
    class MyClass(metaclass=Singleton):
        __metaclass__ = Singleton

        def __init__(self, arg):
            self.arg = arg

    x = MyClass(1)
    y = MyClass(2)
    print(x)  # print "1" as it was first time initialized
    print(y)  # will print also "1" because MyClass was already instantiated so x will be referenced
    """
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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


def conversion_to_timedelta(input_: timedelta | np.datetime64 | int | float) -> timedelta:
    """Conversion to np.timedelta64"""
    if isinstance(input_, timedelta):
        return input_
    elif isinstance(input_, int):
        return timedelta(seconds=input_)
    elif isinstance(input_, float):
        return timedelta(seconds=int(input_))
    elif isinstance(input_, np.timedelta64):
        return input_.item()

    raise ValueError("Format is not compatible!")


def colored_print(print_string):
    # TODO make style, foreground and background somehow auto-set or accessible?
    #  Maybe alternatively it can automatically generate a color-scheme for each class that uses the function
    """
    styles (style, foreground-color, background-color) and prints a given string to console
    :param print_string: the string that should be formatted
    """
    style = 0
    foreground = 37
    background = 40
    start = f'\x1b[{style};{foreground};{background}m'
    end = '\x1b[0m'
    print(start + print_string + end)


def datetime_to_timestamp(dt: datetime) -> int:
    dt = dt.replace(tzinfo=pytz.utc)
    ts = int(dt.timestamp() * 1e9)
    return ts


def timestamp_to_datetime(ts: int | float) -> datetime:
    dt = datetime.fromtimestamp(ts / 1e9, pytz.utc)
    dt = dt.replace(tzinfo=None)
    return dt


def root_path_without_lib_name(root_path):
    return str(root_path).rsplit("ofact", 1)[0]