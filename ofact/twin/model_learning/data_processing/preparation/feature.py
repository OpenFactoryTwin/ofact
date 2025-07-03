"""Contains the tools used to engineer/ adapt features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
from ast import literal_eval

import numpy as np


if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Resource


def letter_to_int(letter):
    """
    Given a letter, this function returns the corresponding integer value based on its position in the alphabet.
    :param letter: The letter for which the integer value needs to be determined.
    :return: The integer value associated with the given letter.
    """
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    integer_associated_with_letter = alphabet.index(letter) + 1

    return integer_associated_with_letter


def replace_comma_by_dot(str_to_adapt: str):
    """
    Replaces commas with dots in a given string and returns the adapted string as a float.
    """
    adapted_str = literal_eval(str_to_adapt.replace(",", "."))
    return adapted_str


def get_elements_of_string(str_to_adapt: str, start: int = 0, end: int = -1) -> Union[str, np.nan]:
    """
    Get a portion of a given string.
    :param str_to_adapt: The string to adapt.
    :param start: The starting index of the substring. Defaults to 0.
    :param end: The ending index of the substring. Defaults to -1.
    :return: adapted string
    """
    if str_to_adapt != str_to_adapt:
        return np.nan
    if not isinstance(str_to_adapt, str):
        return np.nan

    adapted_str = str_to_adapt[start:end]
    return adapted_str


def get_hash_of_string(str_to_hash, amount_of_categories):
    # ToDo: ...
    pass


def get_storage_place_containing_name_from_resource(resource: Resource, storage_place_name: str):
    """
    Take the storage places from resource and check if one of the storage places
    has the string 'storage_place_name' as substring in the name.
    :param resource: provides the storage_places
    :param storage_place_name: the string searched in the storage place name
    :returns a storage place found
    """

    storages = resource.get_storages_without_entity_types()
    storages = [storage
                for storage in storages
                if storage_place_name in storage.name]

    if len(storages) != 1:
        raise Exception([storage.name for storage in storages])

    storage_place_containing_name = storages[0]
    return storage_place_containing_name


def get_resource_of_type(resources, resource_type):
    resources_filtered = [resource
                          for resource in resources
                          if isinstance(resource, resource_type)]
    if len(resources_filtered) == 1:
        resource = resources_filtered[0]
    else:
        resource = None

    return resource


def get_one_hot_encoding_mapping(possible_entries, additional_elements_possible=True):
    """
    Generates a one-hot encoding mapping for a given set of possible entries.
    This function takes a list of possible entries and generates a mapping that assigns a one-hot encoding vector
    to each possible entry. The one-hot encoding vector is a binary vector of length `feature_length`,
    where `feature_length` is the number of possible entries. Each entry in the mapping is a key-value pair,
    where the key is a possible entry and the value is its corresponding one-hot encoding vector.
    :param possible_entries: A list of possible entries.
    :param additional_elements_possible: Indicates whether additional elements are possible in the mapping. If `True`,
    an additional element is added to the mapping with a one-hot encoding vector of all zeros except the last element,
    which is set to 1. Default is `True`.
    :return: A dictionary that maps each possible entry to its corresponding one-hot encoding vector.
    """
    feature_length = len(possible_entries)
    if additional_elements_possible:
        feature_length += 1

    one_hot_encoding_mapping = {possible_entry:
                                    np.array([0 if i != idx else 1
                                              for i in range(feature_length)])
                                for idx, possible_entry in enumerate(possible_entries)}

    if additional_elements_possible:
        additional_element_array = np.zeros(feature_length)
        additional_element_array[-1] = 1
        one_hot_encoding_mapping[None] = additional_element_array

    return one_hot_encoding_mapping
