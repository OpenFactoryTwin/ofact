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

@contact persons: Christian Schwede & Adrian Freiter
@last update: 16.10.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import inspect
import json
import warnings
from copy import copy
from datetime import datetime
from typing import Any, Union, TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

from ofact.helpers import datetime_to_timestamp
# Imports Part 3: Project Imports
from ofact.twin.state_model.helpers.helpers import get_clean_attribute_name

if TYPE_CHECKING:
    from ofact.twin.state_model.basic_elements import DigitalTwinObject


class SerializableWarning(Warning):
    pass


further_serialization = {  # ToDo: maybe derivable from the objects itself
    "DigitalTwinObject": {"object": ["_domain_specific_attributes"]},
    "DomainSpecificAttributes": {"dict": ["type_", "cross_domain_attributes_definition", "attributes"]},
    "Plant": {"datetime": ['current_time']},
    "EntityType": {"object": ['entity_type', 'super_entity_type']},
    "Entity": {"object": ['_situated_in', '_entity_type', '_process_execution', 'dynamic_attributes']},
    "Part": {"object": ['part_of'],
             "list": ['parts']},
    "Resource": {"object": ['plant', '_process_execution_plan', '_physical_body', 'equipment_of'],
                 "list": ['equipments']},
    "StationaryResource": {"object": ['_efficiency']},
    "Storage": {"list": ['stored_entities'],
                "object": ['allowed_entity_type']},
    "WorkStation": {"storage_places": ['_buffer_stations']},
    "Warehouse": {"storage_places": ['_storage_places']},
    "ConveyorBelt": {"object": ['origin', 'destination'],
                     "list": ['entities_on_transport', 'allowed_entity_types']},
    "NonStationaryResource": {"storage_places": ['_storage_places']},
    "ActiveMovingResource": {"object": ['speed']},
    "ResourceController": {"object": ['_resource_model']},
    "ProcessTimeController": {"object": ['_process_time_model']},
    "TransitionController": {"object": ['_transition_model']},
    "QualityController": {"object": ['_quality_model']},
    "TransformationController": {"object": ['_transformation_model']},
    "Process": {"object": ['group', '_lead_time_controller', '_quality_controller', '_resource_controller',
                           '_transition_controller', '_transformation_controller']},
    "ValueAddedProcess": {"object": ['feature'],
                          "list": ['successors'],
                          "list_of_tuples": ['predecessors']},
    "ResourceGroup": {"list": ['resources', 'main_resources']},
    "ResourceModel": {"list": ["_resource_groups"]},
    "TransitionModel": {"list": ['_possible_origins', '_possible_destinations']},
    "EntityTransformationNode": {"object": ['entity_type'],
                                 "list": ['parents', 'children'],
                                 "enum": ['transformation_type', 'io_behaviour']},
    "TransformationModel": {"list": ['_root_nodes']},
    "FeatureCluster": {"object": ['product_class']},
    "Feature": {"object": ['feature_cluster', 'selection_probability_distribution']},
    "Order": {"object": ['feature', 'customer', 'dynamic_attributes'],
              "dict": ['feature_process_execution_match'],
              "list": ['products',
                       'product_classes',
                       'features_completed',
                       'features_requested',
                       '_process_executions'],
              "datetime": ['order_date',
                           'release_date_planned',
                           'release_date_actual',
                           'start_time_planned',
                           'start_time_actual',
                           'end_time_planned',
                           'end_time_actual',
                           'delivery_date_requested',
                           'delivery_date_planned',
                           'delivery_date_actual']},
    "StateModel": {},
    "ProcessExecution": {"object": ["process", "_main_resource", "_origin", "_destination", "_order",
                                    "_connected_process_execution"],
                         "list_of_tuples": ["_parts_involved", "_resources_used"],
                         "datetime": ["_executed_start_time", "_executed_end_time"],
                         "enum": ["_event_type"], }
}
use_ignore = True

input_parameters_classes = {}


class Serializable:
    version = '1.0.0'
    already_serialized = set()

    @classmethod
    def from_dict(cls, data_dict: dict, instantiation_dict: dict | None = None):
        """
        Towards the from_dict method
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

        Parameters
        ----------
        instantiation_dict: The dict used for the instantiation of the object
        data_dict: The dict of the object
        """

        if cls in input_parameters_classes:  # memoization
            input_parameters = input_parameters_classes[cls]
        else:
            input_parameters = (cls.__init__.__code__.co_varnames[1:])
            input_parameters_classes[cls] = input_parameters

        if instantiation_dict is None:
            instantiation_dict = data_dict
        instantiation_dict_updated: dict = _get_instantiation_dict(input_parameters, instantiation_dict)

        # delete the version (since the version is a class attribute)
        if "version" in instantiation_dict_updated:
            version = instantiation_dict_updated.pop("version")
            if version != cls.version:
                raise ValueError(f"Object version {version} does not match class version {cls.version}."
                                 f"Recheck the compatibility of the classes")
        if "version" in data_dict:
            del data_dict["version"]

        instantiated_object = cls(**instantiation_dict_updated)
        instantiated_object.__dict__.update(data_dict)  # ToDo: make them obsolete!

        return instantiated_object

    @staticmethod
    def warn_if_attributes_are_missing(attributes: list[str], ignore: list[str], dictionary: dict[Any, Any]) -> None:
        """
        Checks whether all attributes in a list are keys in a dictionary.
        If some are missing, a warning message is printed.

        Parameters
        ----------
        attributes: The list of attributes to check.
        ignore: The list of attributes to ignore.
        dictionary: The dictionary to check against.
        """

        if use_ignore:
            attributes = set(attributes) - set(ignore)
        missing_attributes = (Serializable.__are_values_keys(attributes, dictionary) -
                              {'object_type', 'version', 'label'})

        # Generate warning messages for each missing attribute
        for attr in missing_attributes:
            object_type = dictionary.get('object_type', '')
            warning_message = (f"Missing Object Attribute in Json: {attr} is not a key "
                               f"in the dictionary of {object_type}.")
            warnings.warn(warning_message, category=SerializableWarning)

    @staticmethod
    def serialize_list(list_of_objects: list[Serializable] | tuple,
                       deactivate_id_filter=False,
                       use_reference=False,
                       drop_before_serialization: dict[str, list[str]] = None,
                       further_serializable: dict[str, list[str]] = None,
                       reference_type: str = "identification") -> list[dict]:
        """
        Serializes a list of objects.

        Parameters
        ----------
        list_of_objects: The list of objects to serialize.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_reference: Whether to represent the objects in the list by their static model id.

        Returns
        -------
        A list of dictionaries containing the serialized objects.
        """

        serialized_list = \
            [item.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                 use_reference=use_reference,
                                 reference_type=reference_type,
                                 further_serializable=further_serializable,
                                 drop_before_serialization=drop_before_serialization)
             if hasattr(item, 'dict_serialize') else item
             for item in list_of_objects]

        return serialized_list

    @staticmethod
    def serialize_2d_numpy_array(np_array_of_objects: np.ndarray,
                                 deactivate_id_filter=False,
                                 drop_before_serialization: dict[str, list[str]] = None,
                                 further_serializable: dict[str, list[str]] = None,
                                 reference_type: str = "identification"):
        """
        Serializes a list of objects.

        Parameters
        ----------
        np_array_of_objects: The list of objects to serialize.
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
                    tuple_item = tuple_item.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                                           reference_type=reference_type,
                                                           further_serializable=further_serializable,
                                                           drop_before_serialization=drop_before_serialization)
                new_tuple_list.append(tuple_item)
            np_array_as_list[index] = tuple(new_tuple_list)

        return np_array_as_list

    @staticmethod
    def serialize_dict(dictionary: dict[Any, Any],
                       deactivate_id_filter: bool = False,
                       use_reference: bool = False,
                       drop_before_serialization: dict[str, list[str]] = None,
                       further_serializable: dict[str, list[str]] = None,
                       reference_type: str = "identification") -> dict[Any, Any]:
        """
        Serializes a dictionary.

        Parameters
        ----------
        dictionary: The dictionary to serialize.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_reference: Whether to represent the objects in the dict by their static model id.
        drop_before_serialization: object parameter that should be dropped before serialization
        further_serializable: attributes that should be further serialized (based on their type ...)

        Returns
        -------
        new_attribute: A dictionary containing the serialized objects.
        """

        new_attribute = dict([Serializable.serialize_dict_item(key,
                                                               value,
                                                               deactivate_id_filter,
                                                               reference_type,
                                                               use_reference,
                                                               further_serializable,
                                                               drop_before_serialization)
                              for key, value in dictionary.items()])

        return new_attribute

    @staticmethod
    def serialize_dict_item(key, value, deactivate_id_filter, reference_type,
                            use_reference, further_serializable, drop_before_serialization):
        if hasattr(key, 'dict_serialize'):
            # Check if key is an instance or a class.
            # If it is a class return the name, otherwise serialize it.
            if not inspect.isclass(key):
                new_key = key.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                             reference_type=reference_type,
                                             use_reference=use_reference,
                                             further_serializable=further_serializable,
                                             drop_before_serialization=drop_before_serialization)
                if not isinstance(new_key, str):
                    if reference_type == "identification":
                        new_key = "id." + str(new_key['identification'])
                    elif reference_type == "label":
                        new_key = new_key['label']
            else:
                new_key = key.__name__
        else:
            new_key = key
        if isinstance(value, list):
            new_value = [obj.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                            reference_type=reference_type,
                                            use_reference=use_reference,
                                            further_serializable=further_serializable,
                                            drop_before_serialization=drop_before_serialization
                                            ) if hasattr(obj, "dict_serialize") else obj
                         for obj in value]
        elif hasattr(value, 'dict_serialize'):
            if not inspect.isclass(value):
                new_value = value.dict_serialize(deactivate_id_filter=deactivate_id_filter,
                                                 reference_type=reference_type,
                                                 use_reference=use_reference,
                                                 further_serializable=further_serializable,
                                                 drop_before_serialization=drop_before_serialization)
            else:
                new_value = value.__name__
        else:
            new_value = value

        return new_key, new_value

    @staticmethod
    def serialize_list_of_tuple(list_of_tuples: list[tuple],
                                deactivate_id_filter: bool = False,
                                use_reference: bool = False,
                                drop_before_serialization: dict[str, list[str]] = None,
                                further_serializable: dict[str, list[str]] = None,
                                reference_type: str = "identification") -> list[tuple]:
        """
        Serializes a list of tuples.

        Parameters
        ----------
        list_of_tuple: The list of tuples to serialize.
        deactivate_id_filter: Whether to deactivate the ID filter. Defaults to False.
        use_reference: Whether to represent the objects in the list by their static model id.

        Returns
        -------
        list_of_tuple: A list of tuples containing the serialized objects.
        """
        list_of_tuples_serialized = \
            [tuple(Serializable.serialize_list(list_of_objects=tuple_,
                                               use_reference=use_reference,
                                               deactivate_id_filter=deactivate_id_filter,
                                               further_serializable=further_serializable,
                                               drop_before_serialization=drop_before_serialization,
                                               reference_type=reference_type))
             for tuple_ in list_of_tuples]

        return list_of_tuples_serialized

    @staticmethod
    def __are_values_keys(values: Union[list[str] | set[str]], dictionary: dict[Any, Any]) -> set:
        """
        Checks whether all values in a list are keys in a dictionary.

        Parameters
        ----------
        values: The list of values to check.
        dictionary: The dictionary to check against.

        Returns
        -------
        True if all values in the list are keys in the dictionary, False otherwise.
        """

        difference = set(values) - set(dictionary.keys()) - {'object_type'} - {'version'}
        return difference

    def dict_serialize(self,
                       deactivate_id_filter: bool = False,
                       use_reference: bool = False,
                       drop_before_serialization: dict[str, list[str]] = None,
                       further_serializable: dict[str, list[str]] = None,
                       reference_type: str = "identification") -> Union[dict | str]:
        """
        Creates a dict representation of the object.
        Also serialize an object in attributes by calling the function on them.

        Parameters
        ----------
        deactivate_id_filter: Whether to check if an obj has already been serialized.
            Then the static model id is used. Defaults to False.
        use_reference: Whether to represent this object by its label.
            Defaults to False.
        drop_before_serialization: object parameter that should be dropped before serialization
        further_serializable: attributes that should be further serialized (based on their type ...)
        reference_type: specify if the static model id (label) should be used or the identification of the object

        Returns
        -------
        object_dict: The dict representation of the entity type or the static model id.
        """

        is_digital_twin_object = hasattr(self, "external_identifications")
        already_serialized = self in Serializable.already_serialized
        if (((not deactivate_id_filter and already_serialized) or use_reference) and
                is_digital_twin_object):
            return self.get_reference(reference_type)

        # Should we change this to be self.__class__.__name__ instead?
        object_dict = {'version': self.version,
                       'object_type': self.__class__.__name__}
        attributes_dict = copy(self.__dict__)
        object_dict.update(attributes_dict)

        Serializable.already_serialized.add(id(self))

        if hasattr(self, 'get_static_model_id'):
            # This is hideous but needed for import
            if reference_type == "label":
                object_dict["label"] = self.get_reference(reference_type)

        if further_serializable is not None:
            if self.__class__.__name__ in further_serializable:
                further_serializable_self = further_serializable[self.__class__.__name__]
                object_dict = self._serialize_attributes(object_dict, drop_before_serialization,
                                                         further_serializable_self, further_serializable,
                                                         reference_type)

        if self.__class__.__name__ in drop_before_serialization:
            drop_before_serialization_self = drop_before_serialization[self.__class__.__name__]
            object_dict = self._drop_attributes(object_dict, drop_before_serialization_self)

        # Check if attributes are missing
        if self.__class__.__name__ in drop_before_serialization:
            ignore = drop_before_serialization[self.__class__.__name__]
        else:
            ignore = []

        if isinstance(object_dict, str):
            return object_dict

        Serializable.warn_if_attributes_are_missing(list(self.__dict__.keys()),
                                                    ignore=ignore,
                                                    dictionary=object_dict)

        return object_dict

    def _drop_attributes(self, object_dict, drop_before_serialization_self):

        for delete in drop_before_serialization_self:
            if delete in object_dict:
                del object_dict[delete]

        return object_dict

    def _serialize_attributes(self, object_dict, drop_before_serialization,
                              further_serializable_self, further_serializable, reference_type):

        if "object" in further_serializable_self:
            further_serializable_object = further_serializable_self["object"]
            for further_serializable_attribute in further_serializable_object:
                if further_serializable_attribute not in object_dict:
                    continue

                if (isinstance(object_dict[further_serializable_attribute], str | dict) or
                        object_dict[further_serializable_attribute] is None):
                    continue
                if hasattr(object_dict[further_serializable_attribute], "external_identifications"):
                    if "static_model" not in object_dict[further_serializable_attribute].external_identifications \
                            and reference_type == "label":
                        object_dict[further_serializable_attribute] = (
                            object_dict[further_serializable_attribute].dict_serialize(
                                further_serializable=further_serializable,
                                drop_before_serialization=drop_before_serialization,
                                reference_type=reference_type))
                    else:
                        object_dict[further_serializable_attribute] = (
                            object_dict[further_serializable_attribute].get_reference(reference_type))

                else:
                    try:
                        object_dict[further_serializable_attribute] = (
                            object_dict[further_serializable_attribute].dict_serialize(
                                further_serializable=further_serializable,
                                drop_before_serialization=drop_before_serialization,
                                reference_type=reference_type))
                    except:
                        pass

        if "list" in further_serializable_self:
            further_serializable_list = further_serializable_self["list"]
            for further_serializable_attribute in further_serializable_list:
                if further_serializable_attribute not in object_dict:
                    continue

                if object_dict[further_serializable_attribute] is not None:
                    serialized_list = (
                        Serializable.serialize_list(object_dict[further_serializable_attribute],
                                                    use_reference=True,
                                                    further_serializable=further_serialization,
                                                    drop_before_serialization=drop_before_serialization,
                                                    reference_type=reference_type))
                    object_dict[further_serializable_attribute] = serialized_list

        if "storage_places" in further_serializable_self:
            further_serializable_storage_places = further_serializable_self["storage_places"]
            for further_serializable_attribute in further_serializable_storage_places:
                if further_serializable_attribute not in object_dict:
                    continue

                if object_dict[further_serializable_attribute] is not None:
                    object_dict[further_serializable_attribute[1:]] = \
                        ["id." + str(st.identification)  # use reference
                         for st_lst in list(object_dict[further_serializable_attribute]._storage_places.values())
                         for st in st_lst]
                    object_dict["capacity"] = object_dict[further_serializable_attribute].capacity

        if "dict" in further_serializable_self:
            further_serializable_dict = further_serializable_self["dict"]
            for further_serializable_attribute in further_serializable_dict:
                if further_serializable_attribute not in object_dict:
                    continue

                if object_dict[further_serializable_attribute] is not None:
                    serialized_dict = {}
                    for key, value in object_dict[further_serializable_attribute].items():

                        if hasattr(key, "external_identifications"):
                            serialized_key = (
                                Serializable.dict_serialize(key,
                                                            use_reference=True,
                                                            further_serializable=further_serialization,
                                                            drop_before_serialization=drop_before_serialization,
                                                            reference_type=reference_type))
                        else:
                            serialized_key = str(key)

                        if hasattr(value, "external_identifications"):
                            serialized_value = (
                                Serializable.dict_serialize(value,
                                                            use_reference=True,
                                                            further_serializable=further_serialization,
                                                            drop_before_serialization=drop_before_serialization,
                                                            reference_type=reference_type))
                        elif isinstance(value, list):
                            serialized_value = (
                                Serializable.serialize_list(value,
                                                            use_reference=True,
                                                            further_serializable=further_serialization,
                                                            drop_before_serialization=drop_before_serialization,
                                                            reference_type=reference_type))
                        else:
                            serialized_value = value

                        serialized_dict[serialized_key] = serialized_value

                    object_dict[further_serializable_attribute] = serialized_dict

        if "list_of_tuples" in further_serializable_self:
            further_serializable_list_of_tuples = further_serializable_self["list_of_tuples"]
            for further_serializable_attribute in further_serializable_list_of_tuples:
                if further_serializable_attribute not in object_dict:
                    continue

                if object_dict[further_serializable_attribute] is not None:
                    serialized_list_of_tuples = (
                        Serializable.serialize_list_of_tuple(object_dict[further_serializable_attribute],
                                                             use_reference=True,
                                                             further_serializable=further_serialization,
                                                             drop_before_serialization=drop_before_serialization,
                                                             reference_type=reference_type))
                    object_dict[further_serializable_attribute] = serialized_list_of_tuples

        if "datetime" in further_serializable_self:
            further_serializable_datetime = further_serializable_self["datetime"]
            for further_serializable_attribute in further_serializable_datetime:
                if further_serializable_attribute not in object_dict:
                    continue

                if isinstance(object_dict[further_serializable_attribute], datetime):
                    try:
                        timestamp_ = datetime_to_timestamp(object_dict[further_serializable_attribute])
                    except OSError:  # before 1970
                        timestamp_ = 0
                    if timestamp_ < 0:
                        timestamp_ = 0
                    object_dict[further_serializable_attribute] = timestamp_

        if "enum" in further_serializable_self:
            further_serializable_enum = further_serializable_self["enum"]
            for further_serializable_attribute in further_serializable_enum:
                if further_serializable_attribute not in object_dict:
                    continue

                if further_serializable_attribute == "transformation_type":
                    object_dict[further_serializable_attribute] = (
                            'EntityTransformationNode.TransformationTypes.' +
                            object_dict[further_serializable_attribute].name)
                elif further_serializable_attribute == "io_behaviour":
                    object_dict[further_serializable_attribute] = ('EntityTransformationNode.IoBehaviours.' +
                                                                   object_dict[further_serializable_attribute].name)
                elif further_serializable_attribute == "_event_type":
                    object_dict[further_serializable_attribute] = ('ProcessExecution.EventTypes.' +
                                                                   object_dict[further_serializable_attribute].name)
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

    def get_reference(self: DigitalTwinObject, reference_type):
        if reference_type == "identification":
            return "id." + str(self.identification)
        elif reference_type == "label":
            return self.get_static_model_id()[1:]
        else:
            raise ValueError(f"Unknown reference type: {reference_type}")

    def to_json(self,
                human_readable=False,
                file_path=None,
                drop_before_serialization: dict[str, list[str]] = None) -> str | None:
        """
        Converts the dict representation of a class instance to the json representation.

        Parameters
        ----------
        human_readable: If set to true, the json string is returned with indentation.
        file_path: If set to a path, the json string is written to the file.

        Returns
        -------
        json_string: The json representation as string.
        """
        raise NotImplementedError
        indent = 4 if human_readable else None
        print("Create serializable Representation")
        object_dict = self.dict_serialize(drop_before_serialization=drop_before_serialization)
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
