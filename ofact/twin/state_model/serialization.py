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

Serialization and instantiation classes of the twin model that are needed across the whole model

Classes:
    SerializableWarning: used for the serialization of the digital twin model
    Serializable: use as base class for the serialization of the digital twin model
    ---
    InstantiationFromDict: Provide the ability to instantiate an object without using the __init__ method

@contact persons: Christian Schwede & Adrian Freiter
@last update: 22.04.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import warnings
import inspect
import json
from typing import Any, Union, TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np


# Imports Part 3: Project Imports
from ofact.twin.state_model.helpers.helpers import get_clean_attribute_name

if TYPE_CHECKING:
    from ofact.twin.state_model.basic_elements import DigitalTwinObject


class SerializableWarning(Warning):
    pass


class Serializable:
    version = 1.0

    already_serialized = set()

    @staticmethod
    def warn_if_attributes_are_missing(attributes: list[str], ignore: list[str], dictionary: dict[Any, Any],
                                       use_ignore: bool = False, ignore_private: bool = False) -> None:
        """
        Checks whether all attributes in a list are keys in a dictionary.
        If some are missing, a warning message is printed.

        Parameters
        ----------
        attributes: The list of attributes to check.
        ignore: The list of attributes to ignore.
        dictionary: The dictionary to check against.
        use_ignore: Flag indicating whether to use the ignore list. Defaults to False.
        ignore_private: Flag indicating whether to ignore private attributes. Defaults to False.
        """

        if use_ignore:
            attributes = set(attributes) - set(ignore)
        missing_attributes = (Serializable.__are_values_keys(attributes, dictionary, ignore_private) -
                              {'object_type', 'version', 'label'})

        # Generate warning messages for each missing attribute
        for attr in missing_attributes:
            object_type = dictionary.get('object_type', '')
            warning_message = (f"Missing Object Attribute in Json: {attr} is not a key "
                               f"in the dictionary of {object_type}.")
            warnings.warn(warning_message, category=SerializableWarning)

    @staticmethod
    def serialize_list(list_of_objects: list[Serializable],
                       serialize_private: bool = True,
                       deactivate_id_filter=False,
                       use_label_for_situated_in: bool = True,
                       use_label=False) -> list[dict]:
        """
        Serializes a list of objects.

        Parameters
        ----------
        list_of_objects: The list of objects to serialize.
        serialize_private: Whether to ignore private attributes. Defaults to True.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_label_for_situated_in: Given down to the objects in the list. Whether to use the static model id
            for the situated_in attribute. Defaults to True.
        use_label: Whether to represent the objects in the list by their static model id.

        Returns
        -------
        A list of dictionaries containing the serialized objects.
        """
        serialized_list = \
            [item.dict_serialize(serialize_private=serialize_private,
                                 deactivate_id_filter=deactivate_id_filter,
                                 use_label_for_situated_in=use_label_for_situated_in,
                                 use_label=use_label, )
             if hasattr(item, 'dict_serialize') else item
             for item in list_of_objects]

        return serialized_list

    @staticmethod
    def serialize_2d_numpy_array(np_array_of_objects: np.ndarray,
                                 serialize_private: bool = True,
                                 deactivate_id_filter=False):
        """
        Serializes a list of objects.

        Parameters
        ----------
        np_array_of_objects: The list of objects to serialize.
        serialize_private: Whether to ignore private attributes. Defaults to True.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.

        Returns
        -------
        np_array_as_list: A list of dictionaries containing the serialized objects.
        """
        np_array_as_list = np_array_of_objects.tolist()
        for index, item in enumerate(np_array_as_list):
            new_tuple_list = []
            for tuple_item in item:
                if hasattr(tuple_item, "dict_serialize"):
                    tuple_item = tuple_item.dict_serialize(serialize_private=serialize_private,
                                                           deactivate_id_filter=deactivate_id_filter)
                new_tuple_list.append(tuple_item)
            np_array_as_list[index] = tuple(new_tuple_list)

        return np_array_as_list

    @staticmethod
    def serialize_dict(dictionary: dict[Any, Any],
                       serialize_private: bool = True,
                       deactivate_id_filter: bool = False,
                       use_label_for_situated_in: bool = True,
                       use_label: bool = False
                       ) -> dict[Any, Any]:
        """
        Serializes a dictionary.

        Parameters
        ----------
        dictionary: The dictionary to serialize.
        serialize_private: Whether to ignore private attributes. Defaults to True.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_label_for_situated_in: Given down to the objects in the dict. Whether to use the static model id
            for the situated_in attribute. Defaults to True.
        use_label: Whether to represent the objects in the dict by their static model id.

        Returns
        -------
        new_attribute: A dictionary containing the serialized objects.
        """

        new_attribute = {}
        for key, value in dictionary.items():
            if hasattr(key, 'dict_serialize'):
                # Check if key is an instance or a class.
                # If it is a class return the name, otherwise serialize it.
                if not inspect.isclass(key):
                    new_key = key.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                                 serialize_private=serialize_private,
                                                 use_label_for_situated_in=use_label_for_situated_in,
                                                 use_label=use_label)
                    if not isinstance(new_key, str):
                        new_key = new_key['label']
                else:
                    new_key = key.__name__
            else:
                new_key = key
            if isinstance(value, list):
                new_value = [obj.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                                serialize_private=serialize_private,
                                                use_label_for_situated_in=use_label_for_situated_in,
                                                use_label=use_label
                                                ) if hasattr(obj, "dict_serialize") else obj
                             for obj in value]
            elif hasattr(value, 'dict_serialize'):
                if not inspect.isclass(value):
                    new_value = value.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                                     serialize_private=serialize_private,
                                                     use_label_for_situated_in=use_label_for_situated_in,
                                                     use_label=use_label
                                                     )
                else:
                    new_value = value.__name__
            else:
                new_value = value

            new_attribute.update({new_key: new_value})

        return new_attribute

    @staticmethod
    def serialize_list_of_tuple(list_of_tuple: list[tuple],
                                serialize_private: bool = True,
                                deactivate_id_filter: bool = False,
                                use_label_for_situated_in: bool = True,
                                use_label: bool = False
                                ) -> list[tuple]:
        """
        Serializes a list of tuples.

        Parameters
        ----------
        list_of_tuple: The list of tuples to serialize.
        serialize_private: Whether to serialize private attributes. Defaults to True.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_label_for_situated_in: Given down to the objects in the list. Whether to use the static model id
            for the situated_in attribute. Defaults to True.
        use_label: Whether to represent the objects in the list by their static model id.

        Returns
        -------
        list_of_tuple: A list of tuples containing the serialized objects.
        """
        for index, item in enumerate(list_of_tuple):
            new_tuple_items = []
            for index_tuple, item_tuple in enumerate(item):
                if isinstance(item_tuple, dict):
                    serializable_object = (
                        Serializable.serialize_dict(item_tuple,
                                                    serialize_private=serialize_private,
                                                    deactivate_id_filter=deactivate_id_filter,
                                                    use_label=use_label,
                                                    use_label_for_situated_in=use_label_for_situated_in))
                    new_tuple_items.append(serializable_object)

                elif isinstance(item_tuple, list):
                    serializable_object = (
                        Serializable.serialize_list_of_tuple(item_tuple,
                                                             serialize_private=serialize_private,
                                                             deactivate_id_filter=deactivate_id_filter,
                                                             use_label=use_label,
                                                             use_label_for_situated_in=use_label_for_situated_in))
                    new_tuple_items.append(serializable_object)

                elif hasattr(item_tuple, 'dict_serialize'):
                    new_tuple_items.append(item_tuple.dict_serialize(use_label=True))

                else:
                    new_tuple_items.append(item_tuple)

            list_of_tuple[index] = tuple(new_tuple_items)

        return list_of_tuple

    @staticmethod
    def __are_values_keys(values: Union[list[str] | set[str]], dictionary: dict[Any, Any],
                          ignore_private: bool = False) -> set:
        """
        Checks whether all values in a list are keys in a dictionary.

        Parameters
        ----------
        values: The list of values to check.
        dictionary: The dictionary to check against.
        ignore_private: Flag indicating whether to ignore private attributes. Defaults to False.

        Returns
        -------
        True if all values in the list are keys in the dictionary, False otherwise.
        """
        if ignore_private:
            dictionary = {key: value
                          for key, value in dictionary.items()
                          if not key.startswith('_')}

        difference = set(values) - set(dictionary.keys()) - {'object_type'} - {'version'}
        return difference

    def dict_serialize(self, serialize_private: bool = True,
                       deactivate_id_filter: bool = False,
                       use_label: bool = False,
                       use_label_for_situated_in: bool = True) -> Union[dict | str]:
        """
        Creates a dict representation of the object.
        Also serialize an object in attributes by calling the function on them.

        Parameters
        ----------
        serialize_private: Whether to serialize the private attributes.
            Defaults to True.
        deactivate_id_filter: Whether to check if an obj has already been serialized.
            Then the static model id is used. Defaults to False.
        use_label_for_situated_in: No functionality.
            Only to support usage in standard functions (e.g., serialize_list)
        use_label: Whether to represent this object by its label.
            Defaults to False.

        Returns
        -------
        object_dict: The dict representation of the entity type or the static model id.
        """
        is_digital_twin_object = hasattr(self, "external_identifications")
        already_serialized = self in Serializable.already_serialized
        if ((not deactivate_id_filter and already_serialized) or use_label) and is_digital_twin_object:
            self: DigitalTwinObject
            return self.get_static_model_id()

        # Should we change this to be self.__class__.__name__ instead?
        object_dict = {'version': self.version,
                       'object_type': str(type(self))}
        object_dict.update(self.__dict__)
        Serializable.already_serialized.add(id(self))
        if hasattr(self, 'get_static_model_id'):
            # This is hideous but needed for import
            object_dict["label"] = self.get_static_model_id()[1:]
        if 'dynamic_attributes' in object_dict:
            del object_dict['dynamic_attributes']

        if hasattr(self, 'drop_before_serialization'):
            for delete in self.drop_before_serialization:
                if delete in object_dict:
                    del object_dict[delete]

        if not serialize_private:
            object_dict = Serializable.remove_private(object_dict)

        return object_dict

    @staticmethod
    def remove_private(object_dict):
        """
        Removes all private attributes from a dict.

        Returns
        -------
        object_dict:  The dict to remove the private attributes from.
        """
        keys_to_remove = set()
        for key in object_dict:
            if key.startswith("_"):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            object_dict.pop(key, None)

        return object_dict

    def to_json(self,
                human_readable=False,
                serialize_private=False,
                file_path=None) -> str | None:
        """
        Converts the dict representation of a class instance to the json representation.

        Parameters
        ----------
        human_readable: If set to true, the json string is returned with indentation.
        serialize_private: If sets to true private attributes are serialized.
        file_path: If set to a path, the json string is written to the file.

        Returns
        -------
        json_string: The json representation as string.
        """
        indent = 4 if human_readable else None
        print("Create serializable Representation")
        object_dict = self.dict_serialize(serialize_private=serialize_private)
        if file_path is None:
            json_string = json.dumps(object_dict, indent=indent)
            return json_string
        else:
            print("Dump representation to File")
            with open(file_path, 'w') as json_stream:
                json.dump(object_dict, json_stream, indent=indent)

    @staticmethod
    def check_key_value_in_dict(dictionary: dict, key_value: tuple) -> bool:
        for key, value in dictionary.items():
            if key == key_value[0] and value == key_value[1]:
                return True
            if isinstance(value, dict):
                Serializable.check_key_value_in_dict(value, key_value)
        return False


def _get_instantiation_dict(input_parameters, data_dict):
    """
    Used to create a parameter dict for a class instance without "_"-elements before the attribute/parameter name.

    Parameters
    ----------
    input_parameters: dict with input parameters of the class
    data_dict: dict with parameters

    Returns
    -------
    instantiation_dict: dict with cleaned attribute names (keys)
    """
    instantiation_dict = {get_clean_attribute_name(key): value
                          for key, value in data_dict.items()
                          if get_clean_attribute_name(key) in input_parameters}

    return instantiation_dict


class InstantiationFromDict:
    """
    If an object only exists by a dict (e.g., the object was persisted or communicated before and should be translated
    to a DigitalTwinObject again), the instantiation through the init is not possible in all cases.
    Some attributes are not settable or newly instantiated in the init.
    Therefore, this class provides the possibility to create the object from dict
    Example:
    deserialized_dict = {
        'name': 'John Doe',
        'age': 30,
        'email': 'johndoe@example.com'
    }
    instantiated_object = InstantiationFromDict.from_dict(deserialized_dict)
    """

    @classmethod
    def from_dict(cls, data_dict, instantiation_dict=None):
        input_parameters = (cls.__init__.__code__.co_varnames[1:])
        if instantiation_dict is None:
            instantiation_dict = data_dict
        instantiation_dict_updated: dict = _get_instantiation_dict(input_parameters, instantiation_dict)
        instantiated_object = cls(**instantiation_dict_updated)
        instantiated_object.__dict__.update(data_dict)

        return instantiated_object
