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

The file is used to instantiate the objects from an Excel file.

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
import ast
import inspect
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, List

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

from ofact.twin.state_model.entities import Entity
# Imports Part 3: Project Imports
from ofact.helpers import Singleton


class Mapping:
    """
    Used to map Excel element to python classes.
    object_columns
    - None for objects
    """
    mappings = {}
    object_columns = {}
    distributions = {}
    to_be_defined_object_columns = {}  # used to model things with python code


def check_list(func):
    """
    Wrapper to check if element is a string list. (Convert them to list)
    """

    def wrapper(read_element):
        # query if list element
        if isinstance(read_element, list):
            return read_element
        elif read_element.startswith('[') and read_element.endswith(']'):
            return func(read_element)
        else:
            if read_element == "":
                return []
            else:
                return read_element

    return wrapper


def convert_str_to_int(read_element):
    if read_element == "":
        return None
    else:
        return int(read_element)

@check_list
def convert_str_to_list(read_element):
    """
    Convert a string element with brackets at the ends to a list element

    Parameters
    ----------
    read_element: a string containing representing a list that looks like this --> "[some_text, some_more_text]"

    Returns
    -------
    a list that looks like this ["some_text", "some_more_text"]
    """
    if read_element == "" or read_element == [""] or read_element == '[]':
        read_element = []
    else:
        try:
            read_element = ast.literal_eval(read_element)
        except:
            Exception(read_element)

    return read_element


def convert_str_to_tuple(read_element):
    return tuple(ast.literal_eval(read_element))


def check_dict(func):
    """
    Wrapper to check if element is a string dict. (Convert them to dict)
    """

    def wrapper(read_element):
        # query if dict element
        if isinstance(read_element, dict):
            return read_element
        elif read_element.startswith('{') and read_element.endswith('}'):
            return func(read_element)
        else:
            if read_element == "":
                return {}
            else:
                return read_element

    return wrapper


@check_dict
def convert_str_to_dict(read_element):
    """
    Convert a string element with {} at the ends to a dict element

    Parameters
    ----------
    read_element: a string representing a dict that looks like this --> "{"some_text": some_more_text}"

    Returns
    -------
    a dict that looks like this {"some_text": "some_more_text"} # ToDo: What is with strings
    """
    if read_element == "" or read_element == {""} or read_element == '{}':
        read_element = {}
    else:
        try:
            read_element = ast.literal_eval(read_element)
        except:
            Exception(read_element)
    return read_element


def convert_str_to_datetime(read_element):
    if type(read_element) == list:
        return [datetime(*elem) for elem in read_element]  # is converted in a function before
    else:
        return datetime(*ast.literal_eval(read_element))


def convert_str_to_timedelta(read_element):
    """Assuming the read_element is a dict"""
    if not isinstance(read_element, dict):
        attributes_dict = convert_str_to_dict(read_element)
    else:
        attributes_dict = read_element
    if isinstance(attributes_dict, dict):
        return timedelta(**attributes_dict)
    else:
        return None


def convert_str_to_python_objects(element):
    """
    Convert string elements (read from the Excel file) that should be an object to real python object

    Parameters
    ----------
    element: string element that should be a python object

    Returns
    -------
    if possible a python object
    """
    # todo: this is ugly as hell, but for some reason i failed with all the proposed methods, which are supposed to be
    #  better --> test them again, I just had a bad day, probably did something very wrong
    if element is not None and element == element:
        try:
            element = ast.literal_eval(str(element))
        except:
            pass
    return element


def split_df(df, intersection_point=-2):
    """
    Split a df at the intersection_point

    Parameters
    ----------
    df: take a df and split them
    intersection_point: Specify the point at which the df is divided.

    Returns
    -------
    two part df's
    """
    return df.iloc[:, intersection_point:], df.iloc[:, :intersection_point]


def _check_independent_objects(params, class_):
    """
    Check df according independent objects (no dependencies to other python objects).

    Parameters
    ----------
    params: series of object attributes
    class_: mapping_object (like MappingFactory)

    Returns
    -------
    a boolean that specify if the dependency of an object of "class_" with the "params"
    """
    for key in class_["columns"]:
        if key in params:
            if not isinstance(params[key], list) and not isinstance(params[key], dict):
                if params[key] not in {np.nan, None} and pd.notna(params[key]):
                    return False
    return True


def _find_objects(find_in, fill_in, mapping_class):
    """
    Find objects and replace string elements with the real objects

    Parameters
    ----------
    find_in: dict - {(object type, object name): object}
    fill_in: dict of an object that should be filled in
    mapping_class: mapping class

    Returns
    -------
    a filled (with objects) dict
    """

    # mapping: {object name: object}
    if find_in:
        if type(list(find_in.keys())[0]) == tuple:
            find_in = dict(zip(list(map(lambda x: x[1], list(find_in.keys()))),
                               find_in.values()))

    for attribute_name, type_ in mapping_class["columns"].items():
        if attribute_name not in fill_in:
            continue

        value = fill_in[attribute_name]
        if value == "False":
            fill_in[attribute_name] = False
            continue
        elif value == "True":
            fill_in[attribute_name] = True
            continue

        if isinstance(value, str):
            # to_be_defined means that others methods are called,
            # that model things with code instead of explicit modelling
            to_be_defined = False
            if "to_be_defined" in value:
                value = value.split("(")[1][:-1]
                type_ = mapping_class.to_be_defined_object_columns[attribute_name]
                to_be_defined = True

            if value.startswith('[') and value.endswith(']'):
                value = convert_str_to_list(read_element=value)
            elif value.startswith('{') and value.endswith('}'):
                value = convert_str_to_dict(read_element=value)

            if to_be_defined:
                if isinstance(value, list):
                    new_value = []
                    for value_ in value:
                        if value_ in find_in:
                            value_ = find_in[value_]
                        new_value.append(value_)
                    value = new_value
                else:
                    if value in find_in:
                        value = find_in[value]

        if value == value:  # nan
            if value:
                # handle list (replacing of strings with objects)
                if isinstance(value, list):
                    if isinstance(value[0], tuple) and type_ is convert_str_to_list:
                        new_value = []
                        for k in value:
                            new_value.append(tuple([find_in[l] if l in find_in else l for l in k]))
                        fill_in[attribute_name] = new_value
                    elif isinstance(value[0], tuple) and type_ is not convert_str_to_list:
                        fill_in[attribute_name] = type_(value)
                    else:
                        # use find_in.get(k, k) instead
                        fill_in[attribute_name] = [find_in[k] if k in find_in else k for k in value]

                # handle dict (replacing of strings with objects)
                elif isinstance(value, dict):
                    key_change = []
                    for idx, (key, val) in enumerate(value.items()):
                        if isinstance(key, str):
                            if key in find_in:
                                new_key = find_in[key]
                                key_change.append((key, new_key))
                        elif isinstance(key, tuple):
                            new_key = (find_in[k] if k in find_in else k for k in val)
                            key_change.append((key, new_key))

                        if isinstance(val, str):
                            if val in find_in:
                                value[key] = find_in[val]
                        elif isinstance(val, list):
                            value[key] = [find_in[k] if k in find_in else k for k in val]
                    for (k, nk) in key_change:
                        value[nk] = value.pop(k)

                    fill_in[attribute_name] = value

            if callable(type_) and type_ is not convert_str_to_list and type_ is not convert_str_to_dict:
                fill_in[attribute_name] = type_(value)

            elif (type(fill_in[attribute_name]) != dict) and (type(fill_in[attribute_name]) != list):
                if fill_in[attribute_name] in find_in:
                    fill_in[attribute_name] = find_in[value]

            if value == [] or value == {}:  # ToDo: semester project - no other problems???
                fill_in[attribute_name] = value
        elif type_ is convert_str_to_list:
            fill_in[attribute_name] = []
        elif type_ is convert_str_to_dict:
            fill_in[attribute_name] = {}

    return fill_in


def _string_to_class_instance(class_parameters, mapping_class, class_name=None):
    """
    Correct the input parameters.
    mapping_class: can be MappingFactory or MappingProcess

    Parameters
    ----------
    class_parameters: dict
    mapping_class: can be MappingFactory or MappingProcess
    class_name: the name of a digital twin class which should be instantiated with the class_parameters

    Returns
    -------
    adjusted_input_parameters
    """
    str_to_class_instance = {}
    for attr_name, attr_value in class_parameters.items():
        if type(attr_value) == str and attr_value.split('(')[0] in mapping_class["distributions"]:
            input_param_s = eval(attr_value.split('(')[-1][:-1])
            if isinstance(input_param_s, tuple):
                attr_value = mapping_class["distributions"][attr_value.split('(')[0]](*input_param_s)
            else:
                attr_value = mapping_class["distributions"][attr_value.split('(')[0]](input_param_s)
        str_to_class_instance[attr_name] = attr_value

    # determine the required attributes
    attributes_of_class = list(inspect.signature(mapping_class["classes"][class_name]).parameters)

    # differentiate between attributes required anyway and attributes have default values but can be overwritten
    attributes_of_class_required = \
        [True if type(elem.default) == type else False
         for elem in list(inspect.signature(mapping_class["classes"][class_name]).parameters.values())]

    # adjust the parameters' nan to None and take only the attributes required from the class
    adjusted_input_parameters = {}
    for idx, key in enumerate(attributes_of_class):
        if attributes_of_class_required[idx] or key in str_to_class_instance:
            if str_to_class_instance[key] == str_to_class_instance[key]:
                adjusted_input_parameters[key] = str_to_class_instance[key]
            else:
                adjusted_input_parameters[key] = None  # check if nan

    return adjusted_input_parameters


class ObjectInstantiation(metaclass=Singleton):
    """
    All objects are instantiated and stored in the already_existing_objects dict. This avoids the creation
    of multiple instances of the same object.
    """

    __metaclass__ = Singleton

    def __init__(self):
        super().__init__()
        self.already_existing_objects = {}  # {(class_name, label): object}

    def add_existing_object(self, existing_object: dict):
        """
        Add an instantiated object to the

        Parameters
        ----------
        existing_object: an object that is instantiated but not yet in the already_existing_objects dict
        """
        self.already_existing_objects.update(existing_object)

    def get_object_already_exist(self, new_object: str):
        """
        Checks if a new_object already exist

        Parameters
        ----------
        new_object: (class_name, label) - reference to a possible new object.

        Returns
        -------
        True if the object already exist else false
        """
        if new_object in self.already_existing_objects:
            return True
        else:
            return False

    def get_existing_object(self, object_finder):
        """
        Returns
        -------
        a object if existing in the stored objects
        """
        return self.already_existing_objects[object_finder]

    def instantiate_independent_objects(self, object_df, mapping_class):
        """
        Instantiates objects without other objects in their attributes

        Parameters
        ----------
        object_df: Excel sheet in a df
        mapping_class: object of a mapping class

        Returns
        -------
        instantiated independent objects
        """

        independent_objects = {}
        for class_name, new_df in object_df.groupby(level=0):
            new_df = self._combine_multiple_rows(new_df, mapping_class)

            for label, params_to_class in new_df.loc[class_name].iterrows():

                if not _check_independent_objects(params_to_class, class_=mapping_class):
                    continue
                if self.get_object_already_exist((class_name, label)):
                    continue

                object_ = mapping_class["classes"][class_name](
                    **self.get_class_parameters(params_to_class=params_to_class,
                                                mapping_class=mapping_class,
                                                class_name=class_name))

                self.add_existing_object({(class_name, label): object_})

                independent_objects[(class_name, label)] = object_

        return independent_objects

    def instantiate_dependent_objects(self, independent_objects, objects_df, mapping_class, find_in,
                                      check_independent_objects=True):
        """
        Instantiates objects with other objects in their attributes

        Parameters
        ----------
        independent_objects: instantiated independent objects
        objects_df: Excel sheet in a df-format
        mapping_class: object of a mapping class
        find_in: archive of already instantiated objects that can be used to fill the attributes with the objects
        check_independent_objects:

        Returns
        -------
        instantiated dependent and independent objects
        """

        for class_name, new_df in objects_df.groupby(level=0):
            new_df = self._combine_multiple_rows(new_df, mapping_class)
            for label, params_to_class in new_df.loc[class_name].iterrows():
                if not ((class_name, label) not in independent_objects or not check_independent_objects):
                    continue

                # check if object already exists?
                class_parameter_dict = self.get_class_parameters(params_to_class=params_to_class,
                                                                 mapping_class=mapping_class,
                                                                 class_name=class_name,
                                                                 find_in=find_in)
                if not self.get_object_already_exist((class_name, label)):
                    try:
                        object_ = mapping_class["classes"][class_name](**class_parameter_dict)
                    except AttributeError:
                        raise AttributeError(class_name, class_parameter_dict)


                    self.add_existing_object({(class_name, label): object_})
                    find_in[(class_name, label)] = object_
                    continue

                # if the object already exist only the parameter are replaced (but not the identification)
                object_ = self.get_existing_object((class_name, label))
                if "identification" in class_parameter_dict:
                    del class_parameter_dict["identification"]  # the identification should remain the old one

                # maybe not the best style to update an object
                for attr_name, attr_value in class_parameter_dict.items():
                    try:
                        if getattr(object_, attr_name) != attr_value:
                            setattr(object_, attr_name, class_parameter_dict[attr_name])
                    except:
                        pass
                        # until now only used for work calendar end_time and input parameters that do not become
                        #   attributes of the instance (position from entity PhysicalBody)
                        # (should be the same (old and new_value))

        return find_in

    def _combine_multiple_rows(self, df, mapping_class):
        level_0_values = list(df.index.get_level_values(0).unique())
        level_1_counts = df.index.get_level_values(1).value_counts()
        non_unique_values_list = level_1_counts[level_1_counts > 1].index.tolist()

        if not non_unique_values_list:
            return df

        if len(level_0_values) != 1:
            raise NotImplementedError

        # Create a MultiIndex from unique values
        multi_index = pd.MultiIndex.from_product([level_0_values, non_unique_values_list])

        new_df = df.copy()
        columns_not_to_consider = ["identification", "external_identifications"]
        for index in multi_index:
            positions = df.index.get_indexer_for([index])

            columns = set(df.columns) - set(columns_not_to_consider)
            for column in columns:
                rows = df.loc[index, column].dropna()
                if len(rows) > 1:
                    df.loc[df.index[positions[0]], column] = "".join(rows)

            multi_row_df = pd.DataFrame(df.loc[df.index[positions[0]]].iloc[0]).T
            new_df = new_df.drop(df.index[positions[0]])
            if new_df.empty:
                new_df = multi_row_df
            else:
                new_df = pd.concat([new_df, multi_row_df], axis=0)

        return new_df

    def change_attributes(self, objects_to_change, correction_dict, attributes_to_change=None):
        """
        The idea is to extend attributes which has been changed because of the amount property
        (more than one of one object is created - especially for the Parts which have to be also added as stored_entity
        to the respective warehouse)
        Note: currently only for changes in lists implemented !!! - another use case available?
            single_values cannot be extended normally and dicts are not used for the init

        Returns
        -------
        objects_to_change: objects that are updated
        """

        if attributes_to_change:
            objects_to_change = {attribute_object_label[1][0]: objects_to_change[attribute_object_label[1][0]]
                                 for entity, attribute_object_label in attributes_to_change.items()
                                 if attribute_object_label[1]}

        objects_changed = list(correction_dict.keys())
        objects_need_change = [object_
                               for (class_name, name), object_ in objects_to_change.items()
                               if self._determine_extension_needed(object_, correction_dict, objects_changed)]

        _ = [self.change_attributes_object(object_, correction_dict, objects_changed)
             for object_ in objects_need_change]

        return objects_to_change

    def _determine_extension_needed(self, object_, correction_dict, objects_changed):
        if not isinstance(object_, Entity):  # use dfs
            return False

        extension_needed = bool([elem
                                 for attr_key, attr_value in object_.__dict__.items()
                                 if type(attr_value) is list
                                 for elem in attr_value
                                 if elem in objects_changed
                                 if correction_dict[elem] != attr_value])

        return extension_needed

    def change_attributes_object(self, object_, correction_dict, objects_changed):
        # ToDo: very slow

        for attr_key, attr_value in object_.__dict__.items():
            if type(attr_value) is not list:
                continue

            nested_extension = [correction_dict[elem] for elem in attr_value if elem in objects_changed]
            if nested_extension:
                attr_value = getattr(object_, attr_key)
                attr_value += [item for sublist in nested_extension for item in sublist]

        return object_

    def load_dict(self, object_df, mapping_class, input_objects=None, correction_needed=False, repetitions=2):
        """
        Load the objects ... ToDo

        Parameters
        ----------
        object_df:
        mapping_class:
        correction_needed:
        input_objects:
        repetitions:

        Returns
        -------
        the instantiated objects of the objects df
        """
        if input_objects is None:
            input_objects = []
        elif type(input_objects) == dict:
            input_objects = [input_objects]

        independent_objects = self.instantiate_independent_objects(object_df=object_df, mapping_class=mapping_class)

        check_independent_objects = True
        find_in = independent_objects
        for input_objects_dict in input_objects:
            find_in.update(input_objects_dict)

        for i in range(repetitions):
            # objects from the other sheets that can be used for the instantiation
            if i > 0:
                check_independent_objects = False
            # all objects, which do have dependencies to other objects already constructed in the previous dictionary
            find_in = self.instantiate_dependent_objects(independent_objects=independent_objects,
                                                         objects_df=object_df, mapping_class=mapping_class,
                                                         find_in=find_in,
                                                         check_independent_objects=check_independent_objects)

        independent_objects = find_in
        if correction_needed:
            creation_domain = mapping_class["source"]

            independent_objects, correction_dict = self.duplicate_objects(independent_objects, object_df,
                                                                          creation_domain)

            return independent_objects, correction_dict

        return independent_objects, None

    def duplicate_objects(self, independent_objects, objects_df, creation_domain):
        """
        Used to duplicate already existing objects

        Parameters
        ----------
        independent_objects: instantiated independent objects
        objects_df: Excel sheet in a df-format

        Returns
        -------
        independent_objects: updated by duplicates,
        correction_dict: shows which elements in other objects_dicts are concerned
        and must be corrected in further steps
        """
        correction_dict = {}

        for class_name, new_df in objects_df.groupby(level=0):
            try:
                concerned_elements = new_df.index[new_df["amount"].astype(int) > 1].to_list()
            except:
                concerned_elements = new_df.index[new_df["amount"] > 1].to_list()
            for concerned_element in concerned_elements:
                duplication_factor = objects_df.loc[concerned_element]["amount"]
                concerned_object = independent_objects[concerned_element]

                correction_dict[concerned_object] = []

                independent_objects, correction_dict = \
                    self.duplicate_object(independent_objects, correction_dict, duplication_factor, concerned_object,
                                          concerned_element, creation_domain)

        return independent_objects, correction_dict

    def duplicate_object(self, independent_objects, correction_dict, duplication_factor, concerned_object,
                         concerned_element, creation_domain):
        """Duplicate objects that have the same attributes except the identification"""

        for idx in range(int(duplication_factor) - 1):
            duplicated_object = concerned_object.duplicate(external_name=False)

            t_lst = list(concerned_element)
            t_lst[1] += str(idx)
            duplicate_element = tuple(t_lst)

            independent_objects[duplicate_element] = duplicated_object
            correction_dict[concerned_object].append(duplicated_object)
            self.add_existing_object({duplicate_element: duplicated_object})

        if creation_domain == "StateModel":
            if concerned_object.situated_in:
                concerned_object.situated_in.add_entities(correction_dict[concerned_object])

        elif creation_domain == "AgentsModel":
            for idx, duplicated_object in enumerate(correction_dict[concerned_object]):
                agent_number = idx + 1
                duplicated_object.adapt_duplicate(agent_number)

        return independent_objects, correction_dict

    def get_class_parameters(self, params_to_class, mapping_class, class_name, find_in=None):
        if find_in is not None:
            class_parameters = _find_objects(fill_in=params_to_class.to_dict(),
                                             find_in=find_in,
                                             mapping_class=mapping_class)
        else:
            class_parameters = params_to_class.to_dict()

        class_attributes = _string_to_class_instance(class_parameters=class_parameters,
                                                     mapping_class=mapping_class,
                                                     class_name=class_name)
        return class_attributes


def convert_tuple_keys_to_nested_dict(dict_, allowed_key_names: Optional[List[str]] = None):
    """Used for the mapping of objects to entity_type"""
    dict_keys = list(set(dict_.keys()))

    mapping_dict = defaultdict(list)
    if allowed_key_names is not None:
        for (key1, key2), val in dict_.items():
            if key1 in allowed_key_names:
                mapping_dict[key1].append((key2, val))
    else:
        for (key1, key2), val in dict_.items():
            mapping_dict[key1].append((key2, val))

    nested_dict = {key_class: dict(mapping_dict[key_class])
                   for key_class, _ in dict_keys
                   if key_class in mapping_dict}

    return nested_dict


def delete_appendages(elem) -> str:
    """
    delete appendages from the elem read from the excel file.

    Parameters
    ----------
    elem: elem which may have [" ", ")", "("]

    Returns
    -------
    a appendages free elem
    """
    for symbol in [" ", ")", "("]:
        elem = elem.replace(symbol, "")
    return elem


def execute_function_calls(func_calls_df, objects_):
    """
    execute functions and overwrite attributes from the objects_ (indexes of the df).

    Parameters
    ----------
    func_calls_df: df with functions to execute
    objects_: objects that can be matched with the indexes of the df - they are mapped with the method names
    """

    for index, row in func_calls_df.iterrows():
        if row["Function calls"] == row["Function calls"]:  # no nan/ function available
            # convert string to list
            string_list = row["Function calls"][1:-1].split("), (")

            for string_elem in string_list:
                function_str = delete_appendages(string_elem.split(", (")[0])

                parameter_str = string_elem.split(", (")[1]
                parameter_str_list = parameter_str.split(",")
                parameter_str_list = [delete_appendages(param) for param in parameter_str_list]

                object_ = [real_object for (object_type, object_name), real_object in objects_.items()
                           if object_name == index[1]][0]
                parameter_as_objects = [real_object for (object_type, object_name), real_object in objects_.items()
                                        for param in parameter_str_list
                                        if object_name == param]

                # call the method 'function_str' from the object 'object_' with the parameters 'parameters_as_objects'
                try:
                    getattr(object_, function_str)(*parameter_as_objects)
                except TypeError:
                    object_name = [object_name for (object_type, object_name), real_object in objects_.items()
                                   if object_name == index[1]][0]
                    raise TypeError(f"Object: {object_name} \n Function: {function_str}({parameter_as_objects})")

        elif row["Overwrite Attributes"] == row["Overwrite Attributes"]:
            # convert string_list to list
            string_list = row["Overwrite Attributes"][1:-1].split("), (")

            for string_elem in string_list:
                attribute_str = delete_appendages(string_elem.split(", ")[0])
                parameter_str = delete_appendages(string_elem.split(", ")[1])

                object_ = [real_object for (object_type, object_name), real_object in objects_.items()
                           if object_name == index[1]][0]
                parameter_as_objects = [real_object for (object_type, object_name), real_object in objects_.items()
                                        if object_name == parameter_str]

                setattr(object_, attribute_str, parameter_as_objects[0])


def combine_all_objects(d1, *dicts) -> dict[str: object]:
    """Combines dicts"""
    if not dicts:
        return d1
    else:
        for d_ in dicts:
            d1.update(d_)
        return d1
