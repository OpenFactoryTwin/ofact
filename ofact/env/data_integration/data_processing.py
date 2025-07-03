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

classes:
    DataProcessing
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import ast
from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union, Dict

# Imports Part 2: PIP Imports
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.helpers import convert_to_datetime
from ofact.twin.state_model.basic_elements import DomainSpecificAttributes, ProcessExecutionTypes
from ofact.twin.state_model.processes import ProcessExecution
from ofact.env.model_administration.helper import get_attr_value

if TYPE_CHECKING:
    from ofact.env.data_integration.data_transformation_management import ObjectCache
    from ofact.twin.state_model.entities import Entity
    from ofact.twin.state_model.basic_elements import DigitalTwinObject
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.change_handler.change_handler import ChangeHandlerPhysicalWorld


def _preprocess_raw_df(raw_data_df):
    # remove elements not matchable to the digital twin model
    columns_to_drop = [(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)]
    additional_columns_to_drop = \
        [(id_, reference_id, class_, attribute, handling, depend_on)
         for (id_, reference_id, class_, attribute, handling, depend_on) in raw_data_df.columns
         if attribute is None]

    columns_to_drop += additional_columns_to_drop

    raw_data_df.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)

    return raw_data_df


def _handle_domain_specific_attributes(raw_data_df):
    """
    Create dicts for the domain_specific attributes ....
    """
    domain_specific_attribute_columns = \
        [(id_, reference_id, class_, attribute, handling, depend_on)
         for (id_, reference_id, class_, attribute, handling, depend_on) in raw_data_df.columns
         if handling == "domain specific attribute"]

    domain_specific_attribute_names = \
        [attribute + '": "'
         for (id_, reference_id, class_, attribute, handling, depend_on) in domain_specific_attribute_columns]
    psa_length = raw_data_df.shape[0]
    attribute_names_np = np.tile(domain_specific_attribute_names, (psa_length, 1))
    attribute_names_df = pd.DataFrame(attribute_names_np)
    attribute_names_df.columns = domain_specific_attribute_columns

    raw_data_df[domain_specific_attribute_columns] = (
            attribute_names_df + raw_data_df[domain_specific_attribute_columns].astype(str))

    _create_domain_specific_attributes_df(raw_data_df, domain_specific_attribute_columns)

    return raw_data_df


def _create_domain_specific_attributes_df(raw_data_df, domain_specific_attribute_columns):
    domain_specific_attribute_groups = {}
    for (mapping_id, mapping_ref, class_, attribute, handling, depend_on) in domain_specific_attribute_columns:
        domain_specific_attribute_groups.setdefault(mapping_ref,
                                                    []).append((mapping_id, mapping_ref, class_, attribute,
                                                                handling, depend_on))

    for group, group_participant_columns in domain_specific_attribute_groups.items():
        domain_specific = raw_data_df[group_participant_columns].astype(str)
        domain_specific_attribute = (pd.Series(np.tile('{"', raw_data_df.shape[0])) +
                                     domain_specific.apply('", "'.join, axis=1) +
                                     pd.Series(np.tile('"}', raw_data_df.shape[0])))

        first_column = group_participant_columns[0]
        mapping_id = int(str(9) + str(int(first_column[0])))
        column_name = (mapping_id, group, np.nan, "domain_specific_attribute", "domain_specific_attribute", np.nan)

        raw_data_df.drop(group_participant_columns,
                         axis=1, errors='ignore', inplace=True)

        raw_data_df[column_name] = domain_specific_attribute.fillna("{}")

        # create a dict
        raw_data_df[column_name] = raw_data_df[column_name].apply(lambda row: ast.literal_eval(row))

    return raw_data_df


def _sort_column_names(column_names):
    """objects that are maybe later needed should be handled/instantiated first"""

    not_used_columns = list(column_names).copy()
    used_mapping_ids = []
    column_names_sorted = []

    while not_used_columns:
        not_used_mapping_new = []
        for (id_, reference_id, class_, attribute, handling, depend_on) in not_used_columns:
            if reference_id == reference_id:
                if reference_id not in used_mapping_ids:
                    not_used_mapping_new.append((id_, reference_id, class_, attribute, handling, depend_on))
                    continue

            if depend_on == depend_on:
                if depend_on not in used_mapping_ids:
                    not_used_mapping_new.append((id_, reference_id, class_, attribute, handling, depend_on))
                    continue

            used_mapping_ids.append(id_)
            column_names_sorted.append((id_, reference_id, class_, attribute, handling, depend_on))

        not_used_columns = not_used_mapping_new.copy()

    return column_names_sorted


def set_value(object_: dict | object, attr: str, value):
    """Set a value to the attr of the object_. The object_ can be a dict or an DT-object"""

    attr_value_object = get_attr_value(object_, attr)
    attr_value_object = copy(attr_value_object)
    value_type = type(value) if value == value else np.nan
    operation_ = operation_mapper(attr_value_object, value, value_type)

    attr_value_object = operation_(attr_value_object, value)

    if isinstance(object_, dict):
        object_[attr] = attr_value_object

    else:
        if isinstance(object_, ProcessExecution):
            completely_filled, _ = object_.completely_filled()
            if completely_filled:  # if already filled, maybe ToDo: an update needed
                return object_
            try:
                setattr(object_, attr, attr_value_object)
            except:
                pass
        else:
            setattr(object_, attr, attr_value_object)

    return object_


def append_object(objects, mapping_id, class_name, object_):
    if mapping_id not in objects:
        objects[mapping_id] = object_
        return objects

    return objects


memory = {}


def memoize_operation_mapper_string(operation_mapper_string):
    def inner(attr_value_object, value_type):
        if (attr_value_object, value_type) not in memory:
            memory[(attr_value_object, value_type)] = operation_mapper_string(attr_value_object, value_type)
        return memory[(attr_value_object, value_type)]

    return inner


def operation_mapper(attr_value_object, value, value_type):
    if isinstance(attr_value_object, str):
        operation_ = operation_mapper_string(attr_value_object, value_type)

    elif isinstance(attr_value_object, list):
        operation_ = operation_mapper_list(attr_value_object, value, value_type)

    elif isinstance(attr_value_object, dict):
        operation_ = operation_mapper_dict(attr_value_object, value_type)

    elif isinstance(attr_value_object, DomainSpecificAttributes):
        operation_ = do_nothing_value

    else:
        operation_ = do_nothing_value

    return operation_


@memoize_operation_mapper_string
def operation_mapper_string(attr_value_object, value_type):
    list_and_not_dict = [elem
                         for elem in attr_value_object.split(" | ")
                         if "list" in elem and not "dict" in elem]  # [0: (4 if len(elem) >= 4 else len(elem))]
    if list_and_not_dict:
        if value_type is None or value_type != value_type:
            operation_ = to_none  # would be overwritten at instantiation

        elif value_type == list:
            operation_ = do_nothing_value

        else:
            operation_ = to_list

    elif "dict" in attr_value_object:  # [0: (4 if len(attr_value_object) >= 4 else len(attr_value_object))]
        operation_ = do_nothing_value

    elif "datetime" in attr_value_object and value_type == str:
        operation_ = str_to_datetime

    else:
        operation_ = do_nothing_value

    return operation_


def operation_mapper_list(attr_value_object, value, value_type):
    if value and value_type == value_type:
        if len(attr_value_object) > 0:
            # check the list if the list contains tuples
            if isinstance(attr_value_object[0], tuple):
                operation_ = extend_tuple_by_list

            else:
                operation_ = append_to_list

        else:
            operation_ = append_to_list

    else:
        operation_ = do_nothing

    return operation_


def operation_mapper_dict(attr_value_object, value_type):
    operation_ = update_dict

    return operation_

def to_none(attr_value_object, value_):
    return None


def str_to_datetime(attr_value_object, value_):
    datetime_object = convert_to_datetime(value_)
    attr_value_object = datetime_object

    return attr_value_object


def to_list(attr_value_object, value_):
    if value_ is not None:
        return [value_]
    return []


def to_empty_list(attr_value_object, value_):
    return []


def append_to_list(attr_value_object, value_):
    if not isinstance(value_, list):
        if value_ is not None:
            attr_value_object.append(value_)
    else:
        attr_value_object.extend([elem for elem in value_ if elem is not None])
    return attr_value_object


def extend_tuple_by_list(attr_value_object, value_):
    value = get_tuple_list(value_)

    attr_value_object.extend(value)
    return attr_value_object


def update_dict(attr_value_object, value_):
    attr_value_object.update(value_)
    return attr_value_object


def do_nothing_value(attr_value_object, value_):
    return value_


def do_nothing(attr_value_object, value):
    return attr_value_object


def get_tuple_list(elem_):
    if not isinstance(elem_, list):
        tuple_list = [(elem_,)]
        return tuple_list

    if not isinstance(elem_[0], tuple):
        tuple_list = [(single_elem,) for single_elem in elem_]
    else:
        tuple_list = elem_

    return tuple_list


def memoize_get_column_name_row(get_column_name_row):
    def inner(row_index, mapping_ids):
        row_index_str = str(row_index)
        mapping_ids_str = str(mapping_ids)
        if (row_index_str, mapping_ids_str) not in memory:
            memory[(row_index_str, mapping_ids_str)] = get_column_name_row(row_index, mapping_ids)
        return memory[(row_index_str, mapping_ids_str)]

    return inner


@memoize_get_column_name_row
def get_column_name_row(row_index, mapping_ids):
    column_names = [(id_, reference_id, class_, attribute, handling, depend_on)
                    for (id_, reference_id, class_, attribute, handling, depend_on) in row_index
                    if id_ in mapping_ids]
    return column_names


class DataProcessing:

    def __init__(self, object_cache: ObjectCache, state_model: StateModel,
                 change_handler: ChangeHandlerPhysicalWorld, progress_tracker, cache, dtm):

        self.dtm = dtm

        self._object_cache = object_cache
        self.mapping_dict = {("", "None"): None,
                             ("ProcessExecution.EventTypes", "ACTUAL"): ProcessExecutionTypes.ACTUAL,
                             ("ProcessExecution.EventTypes", "PLAN"): ProcessExecutionTypes.PLAN}
        self._input_data = {}

        self.cache = cache

        self._objects_already_updated = []  # track the objects already existing in the digital twin and updated

        self._state_model = state_model
        self._change_handler = change_handler

        self.progress_tracker = progress_tracker

    def add_change_handler(self, change_handler):
        self._change_handler = change_handler

    def read_data(self, sources: list[dict], start_datetime: datetime, end_datetime: datetime):
        """Read the data from the sources and store them in the _input_data dict"""

        for idx, source in enumerate(sources):
            data_source_entry = source["data_source_entry"]
            mapping_name = data_source_entry["mapping"]
            name_space = data_source_entry["name space"]

            # async?
            data_batch_received = source["adapter"].get_data(self._input_data, start_datetime, end_datetime)
            self._input_data[mapping_name] = (data_batch_received, name_space)

        self._adapt_consideration_period(start_datetime, end_datetime)

    def create_state_model_objects(self, sources, data_batches_received, priorities):
        """
        Mapping the data to digital_twin state model objects
        prioritization of available data inputs (can be found in the data_source_model) determine the sequence
        """
        mapping_methods = \
            {True: self._create_state_model_objects_dicts,
             # the source is handled in a generalized way (maybe parts are domain specific)
             False: self._create_state_model_objects_dicts_domain_specific}  # the source is handled completely domain specific

        for idx, source in enumerate(sources):
            data_source_entry = source["data_source_entry"]
            mapping_name = data_source_entry["mapping"]
            domain_specific_refinements = source["static_refinements_batch_df"]

            data_batch_received, name_space = self._input_data[mapping_name]

            # Fill the available DT-objects or create new ones if not already available
            map_objects_method = mapping_methods[source["general"]]
            data_batch_refined = map_objects_method(name_space=name_space, raw_data_df=data_batch_received,
                                                    domain_specific_static_refinements=domain_specific_refinements)

            if self.progress_tracker is not None and idx == 2:
                self.progress_tracker.announce(40.0)

            data_batches_received[mapping_name] = (data_batch_refined, name_space)
            priorities[mapping_name] = data_source_entry["priority"]
            # alternative a wide-column solution

            print(f"[{datetime.now()}] Source '{mapping_name}' ({idx + 1}/{len(sources)}) finished")

        return data_batches_received, priorities

    def _adapt_consideration_period(self, start_datetime, end_datetime):
        pass

    def _create_state_model_objects_dicts(self, name_space, raw_data_df, domain_specific_static_refinements):
        """
        Data entry formats are adapted and mapped to digital twin objects.
        Objects that did not exist in the digital twin model are stored in the object cache for later instantiation.
        Objects already exist are adapted if needed?
        """

        if raw_data_df.empty:
            return raw_data_df

        raw_data_df: pd.DataFrame = _preprocess_raw_df(raw_data_df)
        raw_data_df: pd.DataFrame = self._adapt_values_domain_specific(raw_data_df, name_space)
        raw_data_df: pd.DataFrame = _handle_domain_specific_attributes(raw_data_df)

        column_names_sorted = _sort_column_names(raw_data_df.columns)
        raw_data_df = raw_data_df.reindex(column_names_sorted, axis=1)

        columns_only_considered_in_aggregation = \
            [(id_, reference_id, class_, attribute, handling, depend_on)
             for (id_, reference_id, class_, attribute, handling, depend_on) in raw_data_df.columns
             if handling != "aggregation"]

        raw_data_df[columns_only_considered_in_aggregation].apply(
            lambda source_entry: self._handle_source_entry(name_space, source_entry,
                                                           domain_specific_static_refinements),
            axis=1)

        return raw_data_df

    def _handle_source_entry(self, name_space, source_entry, domain_specific_static_refinements):

        # logger.debug(f"DT Object Mapping {idx}")
        currently_available_object_dicts = {}
        changed_objects = {}

        for (id_, reference_id, class_, attribute, handling, depend_on), value in source_entry.items():

            # not completed process_executions are not considered ...
            if value != value or (not value and value != 0):
                continue

            if handling == handling:
                value, situated_in, skip = (
                    self._handle_entries(handling, source_entry, currently_available_object_dicts, id_, reference_id,
                                         class_, attribute, value, name_space, depend_on))
                if skip:
                    continue
            else:
                situated_in = None

            if attribute == "identification":
                current_object, currently_available_object_dicts, changed_objects = (
                    self._get_object_from_source_entry(currently_available_object_dicts, id_, name_space, class_, value,
                                                       situated_in, domain_specific_static_refinements, changed_objects))

            else:
                reference_object = currently_available_object_dicts[int(reference_id)]

                # add an attribute
                if isinstance(reference_object, dict):
                    has_attr = attribute in reference_object
                else:
                    has_attr = hasattr(reference_object, attribute)

                if has_attr:
                    if class_ == class_:
                        value, currently_available_object_dicts, changed_objects = \
                            self._get_object_from_source_entry(currently_available_object_dicts, id_,
                                                               name_space, class_, value, situated_in,
                                                               domain_specific_static_refinements, changed_objects)

                    current_object = set_value(reference_object, attribute, value)
                    changed_objects.setdefault(class_,
                                               []).append(current_object)

        if "ProcessExecution" not in changed_objects:
            return

        self._update_process_executions(process_executions=changed_objects["ProcessExecution"],
                                        name_space=name_space)

    def _get_object_from_source_entry(self, currently_available_object_dicts, mapping_id,
                                      name_space, class_name, value, situated_in,
                                      domain_specific_static_refinements, changed_objects):
        value, object_changed = (
            self._get_state_model_object(currently_available_object_dicts=currently_available_object_dicts,
                                         mapping_id=mapping_id, name_space=name_space,
                                         class_name=class_name, value=value, situated_in=situated_in,
                                         domain_specific_static_refinements=domain_specific_static_refinements))
        currently_available_object_dicts = append_object(currently_available_object_dicts, mapping_id,
                                                         class_name, value)

        if object_changed:
            changed_objects.setdefault(class_name,
                                       []).append(value)

        return value, currently_available_object_dicts, changed_objects

    def _handle_entries(self, handling, row, currently_used_object_dicts, id_, reference_id, class_, attribute, value,
                        name_space, depend_on):
        """Currently the handling operations domain specific and domain_specific_attribute are supported"""

        situated_in = None
        skip = False
        if handling == "domain specific":
            value, situated_in, skip = (
                self._value_adaption_domain_specific(row, currently_used_object_dicts, id_,
                                                     class_, attribute, value, name_space,
                                                     currently_used_object_dicts, depend_on))

        elif "domain_specific_attribute" == attribute:
            # map the value the object associated with the class_name

            reference_object = currently_used_object_dicts[int(reference_id)]
            if isinstance(reference_object, dict):
                set_value(reference_object, "domain_specific_attributes", value)

            else:
                if reference_object.domain_specific_attributes is None:
                    reference_object.domain_specific_attributes = DomainSpecificAttributes(attributes=value)
                else:
                    reference_object.domain_specific_attributes.add_attributes(value, all_attributes_allowed=True)

            skip = True

        return value, situated_in, skip

    def _update_process_executions(self, process_executions, name_space):
        # ProcessExecutions should be stored a second time because the process_executions are connected to orders
        for process_execution in process_executions:
            self.store_batch(class_name="ProcessExecution",
                             object_dict=process_execution,
                             name_space=name_space)

    def store_batch(self, class_name, object_dict, name_space):
        # domain specific refinement - call a method

        external_identifications = get_attr_value(object_dict, "external_identifications")
        external_id = external_identifications[name_space]

        self._object_cache.store_object_dict(class_name=class_name, name_space=name_space,
                                             external_id=external_id, object_dict=object_dict)

    def _adapt_values_domain_specific(self, raw_data_df, name_space):
        """Should be overwritten by the domain-specific part"""
        return raw_data_df

    def _value_adaption_domain_specific(self, row, object_dicts, mapping_id, class_, attr, value, name_space,
                                        currently_used_object_dicts, depend_on):
        return None, None, False

    def _create_state_model_objects_dicts_domain_specific(self, name_space, raw_data_df,
                                                          domain_specific_static_refinements):
        pass

    def _update_objects_domain_specific(self, object_):
        """
        Used to update already existing objects
        Use Case: e.g. the abilities of the worker changes over time (this can be determined from the new data)
        """
        pass

    def _get_state_model_object(self, name_space, class_name, value, currently_available_object_dicts={},
                                mapping_id=None,
                                situated_in=None, new_possible=True, domain_specific_static_refinements=None) -> (
            [Optional[Union[Dict, DigitalTwinObject]], bool]):

        object_changed = False
        # 1. level: currently_available_object_dicts
        if mapping_id in currently_available_object_dicts:
            return currently_available_object_dicts[mapping_id], object_changed

        # 2. level: physical world cache
        current_object_from_cache = self._object_cache.get_object(name_space=name_space, external_id=value,
                                                                  class_name=class_name)

        if current_object_from_cache is not None:
            return current_object_from_cache, object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            objects_already_planned = self._object_cache.get_objects_already_planned(type_=class_name)

        else:
            objects_already_planned = []

        # 3. level: digital twin
        unique_value = value
        old_value = value

        current_object_from_dt = (
            self.get_object_by_external_identification_dt(name_space=name_space, external_id=value,
                                                          class_name=class_name, situated_in=situated_in,
                                                          objects_already_planned=objects_already_planned))

        self._update_objects_domain_specific(current_object_from_dt)

        if domain_specific_static_refinements is not None:
            current_object_from_dt, object_changed = (
                self.refine_with_static_attributes(current_object_from_dt, domain_specific_static_refinements,
                                                   class_name))
        else:
            object_changed = False

        if current_object_from_dt is not None:
            if class_name == "Part":  # ToDo: Why only the part
                current_object_from_dt.external_identifications = copy({name_space: [unique_value]})
                self._object_cache.store_object_already_planned(type_=class_name, object_=current_object_from_dt)

            # object available in the digital_twin - should be updated if new information available
            self.store_batch(class_name, current_object_from_dt, name_space)

            return current_object_from_dt, object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            properties = self._get_object_properties(current_object_from_dt, objects_already_planned, class_name, value,
                                                     name_space, situated_in, old_value)

            if properties:
                if "storage_places" in properties:
                    storages_to_store = properties["storage_places"]
                    # maybe integrate objects that are not a dict in the subsequent instantiation process
                    for _, storages in storages_to_store.items():
                        for storage in storages:
                            self._change_handler.add_object(storage)
            else:
                print("Class name:", class_name)

        else:
            properties = None

        # 4. level: digital twin class kwargs
        if new_possible is True:
            current_object_dict = self._get_digital_twin_class_dict(class_name, name_space, value)
            if domain_specific_static_refinements is not None:
                current_object_dict, object_changed = (
                    self.refine_with_static_attributes(current_object_dict, domain_specific_static_refinements,
                                                       class_name))
            else:
                object_changed = False
        else:
            return None, object_changed

        if properties is not None:
            for key, value in current_object_dict.items():
                if key in properties:
                    current_object_dict[key] = properties[key]

        if class_name == "PassiveMovingResource":
            del current_object_dict["physical_body"]

        if class_name == "Part" and get_attr_value(current_object_dict, "name") == "str":
            raise Exception("The object of type part is not completely filled:", current_object_dict, value)

        # logger.debug(f"Current object: {class_name}  {value}")
        self.store_batch(class_name, current_object_dict, name_space)
        return current_object_dict, object_changed

    def refine_with_static_attributes(self, current_object, domain_specific_static_refinements, class_name):

        object_changed = False
        # static attributes are only set to dict/ not to already existing objects
        if not isinstance(current_object, dict):
            return current_object, object_changed

        if domain_specific_static_refinements.loc[domain_specific_static_refinements["class"] == class_name].empty:
            return current_object, object_changed

        ps_attributes = domain_specific_static_refinements.loc[
            domain_specific_static_refinements["class"] == class_name]
        for ps_attributes_idx, ps_attributes_row in ps_attributes.iterrows():
            if ps_attributes_row["type"] == ps_attributes_row["type"]:
                if ps_attributes_row["type"] == ps_attributes_row["value"]:
                    continue

            mapping_key = (ps_attributes_row["type"], ps_attributes_row["value"])
            if mapping_key in self.mapping_dict:
                ps_attributes_row["value"] = self.mapping_dict[mapping_key]

            current_object = set_value(current_object,
                                       ps_attributes_row["attribute"],
                                       ps_attributes_row["value"])
            object_changed = True

        return current_object, object_changed

    def _get_digital_twin_class_dict(self, class_name, name_space, value):
        """Returns an unwritten class params dict from class 'class_name'"""
        current_object_dict = self._state_model.get_object_attributes(object_class_name=class_name)
        if not current_object_dict:
            raise NotImplementedError

        current_object_dict = set_value(object_=current_object_dict,
                                        attr="external_identifications",
                                        value={name_space: [value]})

        return current_object_dict

    def get_object_by_external_identification_dt(self, name_space, external_id, class_name,
                                                 situated_in: None | Entity = None,
                                                 objects_already_planned: None | list = None):
        """
        Get digital_twin object by external identification

        Parameters
        ----------
        situated_in: is used to determine only objects that are situated in the situated_in
        objects_already_planned: objects that are excluded in the determination
        because they are already selected in another context
        """

        digital_twin_objects = (
            self._state_model.get_object_by_external_identification(name_space=name_space, external_id=external_id,
                                                                     class_name=class_name,
                                                                     from_cache=True))

        if len(digital_twin_objects) == 0:
            return None

        if situated_in is not None:
            if hasattr(digital_twin_objects[0], "situated_in"):
                try:
                    digital_twin_objects = [digital_twin_object
                                            for digital_twin_object in digital_twin_objects
                                            if digital_twin_object.check_situated_in(entity=situated_in) is True]
                except:
                    raise Exception("Situated could not be checked: ..", situated_in)

        if objects_already_planned is not None:
            digital_twin_objects = list(set(digital_twin_objects).difference(set(objects_already_planned)))

        if len(digital_twin_objects) == 1:
            return digital_twin_objects[0]

        elif len(digital_twin_objects) > 1:
            return digital_twin_objects[0]

    def _get_object_properties(self, current_object_from_dt, objects_already_planned, class_name, value, name_space,
                               situated_in, old_value):

        properties = None
        if current_object_from_dt is None:
            objects_already_planned = [object_ for object_ in objects_already_planned
                                       if old_value in object_.external_identifications[name_space][0]]

        # maybe to special
        if objects_already_planned:
            if class_name == "Part":
                properties = self._get_part_properties(situated_in, objects_already_planned, name_space, old_value)

        else:
            if class_name == "PassiveMovingResource":
                properties = self._get_pmr_properties_without_reference(name_space=name_space,
                                                                        class_name=class_name,
                                                                        situated_in=situated_in,
                                                                        external_name=value)
            elif class_name == "ActiveMovingResource":
                properties = self._get_amr_properties_without_reference(name_space=name_space,
                                                                        class_name=class_name,
                                                                        situated_in=situated_in,
                                                                        external_name=value)

        return properties

    def _get_part_properties(self, situated_in, objects_already_planned, name_space, old_value):

        if situated_in is not None:
            objects_already_planned_situated_in = \
                [object_already_planned
                 for object_already_planned in objects_already_planned
                 if object_already_planned.check_situated_in(entity=situated_in) is True]

        else:
            objects_already_planned_situated_in = objects_already_planned

        part_properties = None
        if objects_already_planned_situated_in:
            for object_ in objects_already_planned_situated_in:
                if not [name for name in object_.external_identifications[name_space]
                        if old_value == " ".join([elem for elem in name.split(" ")[1:2]])]:
                    continue
                object_already_planned_situated_in = object_
                part_properties = \
                    {"entity_type": object_already_planned_situated_in.entity_type,
                     "name": object_already_planned_situated_in.name,
                     "situated_in": object_already_planned_situated_in.situated_in}
                break

        return part_properties

    def _get_pmr_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        reference_object, nsr_properties = (
            self._get_non_stationary_properties_without_reference(name_space, class_name, situated_in, external_name))

        # maybe differentiate the objects
        pmr_properties = (nsr_properties |
                          {"service_life": reference_object.service_life})

        return pmr_properties

    def _get_amr_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        reference_object, nsr_properties = (
            self._get_non_stationary_properties_without_reference(name_space, class_name, situated_in, external_name))

        # maybe differentiate the objects
        amr_properties = (nsr_properties |
                          {"energy_level": reference_object.speed,
                           "speed": reference_object.speed,
                           "energy_consumption": reference_object.speed,
                           "energy_capacity": reference_object.speed})

        return amr_properties

    def _get_non_stationary_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        available_objects = self._state_model.get_objects_by_class_name(class_name)

        reference_object = available_objects[0]

        # maybe differentiate the objects
        nsr_properties = {"entity_type": reference_object.entity_type,
                          "name": external_name,
                          "situated_in": reference_object.situated_in,
                          "process_execution_plan": reference_object.process_execution_plan.duplicate(),
                          "plant": reference_object.plant,
                          "costs_per_second": reference_object.costs_per_second,
                          "orientation": reference_object.orientation,
                          "storage_places":
                              reference_object._storage_places.duplicate_for_instantiation(without_situated_in=True),
                          "position": reference_object.get_position(),
                          "width": reference_object.get_width(),
                          "length": reference_object.get_length()}

        return reference_object, nsr_properties

    def aggregate_data(self, data_batches_received: dict[str, tuple[pd.DataFrame, str]],
                        aggregation_combinations, domain_specific_refinements):
        """Aggregate the data snippets from different sources."""

        all_dfs = {}
        # prioritize data_elements based on the excel file (data_sources) .get_data() - parallelization
        for (source_name, other_source_name), column_names_lst in aggregation_combinations.items():
            # create Orders, ProcessExecutions
            source_element, name_space_one = data_batches_received[source_name]
            all_dfs[source_name] = source_element
            column_names = \
                [(id_, reference_id, class_, attribute, handling, depend_on)
                 for (id_, reference_id, class_, attribute, handling, depend_on) in list(source_element.columns)
                 for column_name in column_names_lst
                 if class_ == column_name and attribute == "identification"]
            column_name = column_names[0]

            source_element_df = source_element.copy()

            other_df, name_space_two = data_batches_received[other_source_name]
            other_df = other_df.copy()
            all_dfs[other_source_name] = other_df

            intersection_domain_specific_refinements = \
                {source_name: refinements
                 for source_name, refinements in domain_specific_refinements.items()}

            self._aggregate_data_domain_specific((source_element_df, source_name),
                                                 (other_df, other_source_name),
                                                 column_name, name_space_one, name_space_two,
                                                 intersection_domain_specific_refinements)

        self._aggregate_data_domain_specific_all(all_dfs, domain_specific_refinements)

        # create process_execution_paths
        # basis: all process_execution from one AID
        # all objects with identification from the same AID

    def _get_intersections_between_sources(self, intersections_between_sources, source_name):
        """Combines the other than source_name source with the common feature_column_names"""
        current_intersections_between_sources = {}
        for (source_1, source_2), intersections_lst in intersections_between_sources.items():
            if source_name == source_1:
                current_intersections_between_sources[source_2] = intersections_lst
            elif source_name == source_2:
                current_intersections_between_sources[source_1] = intersections_lst

        return current_intersections_between_sources

    def _aggregate_data_domain_specific(self, one_df_with_name, other_df_with_name, column_name, name_space_one,
                                        name_space_two, domain_specific_static_refinements):
        pass

    def _aggregate_data_domain_specific_all(self, all_dfs, domain_specific_refinements):
        pass
