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

from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union, Dict

# Imports Part 2: PIP Imports
import pandas as pd

from ofact.env.model_administration.sm_object_handling import abbreviations
from ofact.env.model_administration.standardization.event_log_standard import EventLogStandardAttributes

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.env.model_administration.helper import get_attr_value

if TYPE_CHECKING:
    from ofact.env.model_administration.standardization.data_entry_mapping import DataEntryMapping
    from ofact.env.model_administration.cache import ObjectCacheDataIntegration
    from ofact.twin.state_model.entities import Entity
    from ofact.twin.state_model.basic_elements import DigitalTwinObject
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.change_handler.change_handler import ChangeHandlerPhysicalWorld


def _sort_column_names(data_entry_mappings):
    """objects that are maybe later needed should be handled/instantiated first"""

    not_used_data_entries = data_entry_mappings.copy()
    used_mapping_ids = []
    column_names_sorted = []

    while not_used_data_entries:
        not_used_mapping_new = []
        for data_entry_mapping in not_used_data_entries:
            if data_entry_mapping.reference_identification is not None:
                if data_entry_mapping.reference_identification not in used_mapping_ids:
                    not_used_mapping_new.append(data_entry_mapping)

                    continue

            # if depend_on == depend_on:
            #     if depend_on not in used_mapping_ids:
            #         not_used_mapping_new.append((data_entry_mapping)
            #         continue

            used_mapping_ids.append(data_entry_mapping.identification)
            column_names_sorted.append(data_entry_mapping.get_new_column_name())

        not_used_data_entries = not_used_mapping_new.copy()
        print(len(not_used_data_entries))

    return column_names_sorted


class StateModelObjectCreation:

    def __init__(self, object_cache: ObjectCacheDataIntegration, state_model: StateModel,
                 change_handler: ChangeHandlerPhysicalWorld, progress_tracker, dtm):

        self.dtm = dtm

        self._object_cache = object_cache
        self.mapping_dict = {("", "None"): None,
                             ("ProcessExecution.EventTypes", "ACTUAL"): ProcessExecutionTypes.ACTUAL,
                             ("ProcessExecution.EventTypes", "PLAN"): ProcessExecutionTypes.PLAN}
        self._input_data = {}

        self._objects_already_updated = []  # track the objects already existing in the digital twin and updated

        self._state_model = state_model
        self._change_handler = change_handler

        self.progress_tracker = progress_tracker

    def add_change_handler(self, change_handler):
        self._change_handler = change_handler

    def read_data(self, sources: list[dict], start_datetime: datetime, end_datetime: datetime):
        """Read the data from the sources and store them in the _input_data dict"""

        for idx, source in enumerate(sources):
            mapping_name = source["name"]
            name_space = source["name_space"]

            # async?
            standardization_class = source["standardization_class"]
            adapter_class = source["adapter"]

            standardized_event_log, static_refinements = (
                standardization_class.standardize(adapter_class, start_datetime, end_datetime))


            # self._input_data
            self._input_data[mapping_name] = (standardized_event_log, name_space)

        self._adapt_consideration_period(start_datetime, end_datetime)

    def create_state_model_objects(self, source_modules, data):
        """
        Mapping the data to digital_twin state model objects
        prioritization of available data inputs (can be found in the data_source_model) determine the sequence
        """
        mapping_methods = \
            {True: self._create_state_model_objects_dicts,
             # the source is handled in a generalized way (maybe parts are domain specific)
             False: self._create_state_model_objects_dicts_domain_specific}  # the source is handled completely domain specific

        for idx, (source_name, name_space, adapter, data_source_entry) in enumerate(source_modules):
            data_batch_received = data[source_name]

            # Fill the available DT-objects or create new ones if not already available
            map_objects_method = mapping_methods[True]
            data_batch_refined = map_objects_method(name_space=name_space, raw_data_df=data_batch_received,
                                                    data_entry_mappings=data_source_entry.data_entry_mapper)

            if self.progress_tracker is not None and idx == 2:
                self.progress_tracker.announce(40.0)

            data[source_name] = (data_batch_refined, name_space)
            # alternative a wide-column solution

            print(f"[{datetime.now()}] Source '{source_name}' ({idx + 1}/{len(source_modules)}) finished")

        return data

    def _adapt_consideration_period(self, start_datetime, end_datetime):
        pass

    def _create_state_model_objects_dicts(self, name_space, raw_data_df, data_entry_mappings):
        """
        Data entry formats are adapted and mapped to digital twin objects.
        Objects that did not exist in the digital twin model are stored in the object cache for later instantiation.
        Objects already exist are adapted if needed?
        """

        if raw_data_df.empty:
            return raw_data_df

        column_names_sorted = _sort_column_names(data_entry_mappings)
        raw_data_df = raw_data_df.reindex(column_names_sorted, axis=1)

        columns_data_entry_mappings = {data_entry_mapping.get_new_column_name(): data_entry_mapping
                                      for data_entry_mapping in data_entry_mappings}

        raw_data_df[column_names_sorted].apply(
            lambda source_entry: self._handle_source_entry(name_space, source_entry, columns_data_entry_mappings),
            axis=1)

        # todo handle time columns

        event_time_columns_available = [column_name
                                        for column_name in column_names_sorted
                                        if (EventLogStandardAttributes.EVENT_TIME_TRACE.string in column_name or
                                            EventLogStandardAttributes.EVENT_TIME_SINGLE.string in column_name)]

        if event_time_columns_available:
            print(event_time_columns_available)

            # handling is also important for this area

        return raw_data_df

    def _handle_source_entry(self, name_space, source_entry, columns_data_entry_mappings):

        # ToDo: class and attribute are different

        # logger.debug(f"DT Object Mapping {idx}")
        currently_available_object_dicts = {}
        currently_available_values = {}
        changed_objects = {}
        object_classes = {}
        for column_name, value in source_entry.items():
            data_entry_mapping: DataEntryMapping = columns_data_entry_mappings[column_name]
            object_classes[data_entry_mapping.identification] = data_entry_mapping.get_state_model_class()

            if (EventLogStandardAttributes.EVENT_TIME_TRACE.string in column_name or
                        EventLogStandardAttributes.EVENT_TIME_SINGLE.string in column_name):
                continue
            # not completed process_executions are not considered ...
            value_not_set = value != value or (not value and value != 0)
            if value_not_set:
                continue

            situated_in = None

            if data_entry_mapping.state_model_class:
                if not isinstance(value, str):
                    value = str(value)
                value = str(value)
                if value.endswith(abbreviations[data_entry_mapping.state_model_class]):
                    value += abbreviations[data_entry_mapping.state_model_class]

                if name_space == "static_model":
                    value = "_" + value

            if (data_entry_mapping.state_model_class is not None and
                    data_entry_mapping.state_model_attribute is None):
                current_object, currently_available_object_dicts, changed_objects = (
                    self._get_object_from_source_entry(value, data_entry_mapping, currently_available_object_dicts,
                                                       situated_in, changed_objects, name_space))

            elif data_entry_mapping.state_model_class is not None:
                value, currently_available_object_dicts, changed_objects = (
                        self._get_object_from_source_entry(value, data_entry_mapping,
                                                           currently_available_object_dicts,
                                                           situated_in, changed_objects, name_space))

            if data_entry_mapping.reference_identification in currently_available_object_dicts:
                if data_entry_mapping.handling is not None:
                    print("Handling not yet implemented")
                    continue

                reference_object = currently_available_object_dicts[data_entry_mapping.reference_identification]

                # add an attribute
                if isinstance(reference_object, dict):
                    has_attr = data_entry_mapping.state_model_attribute in reference_object
                else:
                    has_attr = hasattr(reference_object, data_entry_mapping.state_model_attribute)

                if has_attr:
                    reference_object = set_value(reference_object, data_entry_mapping.state_model_attribute, value)

                    changed_objects.setdefault(object_classes[data_entry_mapping.reference_identification],
                                               []).append(reference_object)
                    currently_available_values[data_entry_mapping.identification] = value

        if "ProcessExecution" not in changed_objects:
            return

        self._update_process_executions(process_executions=changed_objects["ProcessExecution"],
                                        name_space=name_space)

    def _get_object_from_source_entry(self, value, data_entry_mapping, currently_available_object_dicts, situated_in,
                                      changed_objects, name_space):

        value, object_changed = (
            self._get_state_model_object(currently_available_object_dicts=currently_available_object_dicts,
                                         value=value, situated_in=situated_in, data_entry_mapping=data_entry_mapping,
                                         name_space=name_space))

        currently_available_object_dicts = append_object(currently_available_object_dicts,
                                                         data_entry_mapping.identification,
                                                         data_entry_mapping.get_state_model_class(), value)

        if object_changed:
            changed_objects.setdefault(data_entry_mapping.get_state_model_class(),
                                       []).append(value)

        return value, currently_available_object_dicts, changed_objects

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

        self._object_cache.cache_object(class_name=class_name, name_space=name_space,
                                        external_id=external_id, object_dict=object_dict)


    def _create_state_model_objects_dicts_domain_specific(self, name_space, raw_data_df,
                                                          domain_specific_static_refinements):
        pass

    def _update_objects_domain_specific(self, object_):
        """
        Used to update already existing objects
        Use Case: e.g. the abilities of the worker changes over time (this can be determined from the new data)
        """
        pass

    def _get_state_model_object(self, value, data_entry_mapping: DataEntryMapping, currently_available_object_dicts={},
                                situated_in=None, name_space=None, new_possible=True) -> (
            [Optional[Union[Dict, DigitalTwinObject]], bool]):

        object_changed = False
        # 1. level: currently_available_object_dicts
        if data_entry_mapping.identification in currently_available_object_dicts:
            return currently_available_object_dicts[data_entry_mapping.identification], object_changed

        # 2. level: physical world cache
        class_name = data_entry_mapping.get_state_model_class()
        current_object_from_cache = self._object_cache.get_object(name_space=name_space, external_id=value,
                                                                  class_name=class_name)

        if current_object_from_cache is not None:
            return current_object_from_cache, object_changed

        if data_entry_mapping.state_model_class in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            objects_already_planned = (
                self._object_cache.get_objects_already_planned(type_=class_name))

        else:
            objects_already_planned = []

        # 3. level: digital twin
        unique_value = value
        old_value = value

        current_object_from_dt = (
            self.get_object_by_external_identification_dt(name_space=name_space, external_id=value,
                                                          class_name=class_name,
                                                          situated_in=situated_in,
                                                          objects_already_planned=objects_already_planned))

        self._update_objects_domain_specific(current_object_from_dt)

        # ToDo
        # if domain_specific_static_refinements is not None:
        #     current_object_from_dt, object_changed = (
        #         self.refine_with_static_attributes(current_object_from_dt, domain_specific_static_refinements,
        #                                            class_name))
        # else:
        #     object_changed = False

        if current_object_from_dt is not None:
            if class_name == "Part":  # ToDo: Why only the part
                current_object_from_dt.external_identifications = copy({name_space: [unique_value]})
                self._object_cache.cache_object_already_planned(type_=class_name,
                                                                object_=current_object_from_dt)

            # object available in the digital_twin - should be updated if new information available
            self.store_batch(class_name, current_object_from_dt, name_space)

            return current_object_from_dt, object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            properties = self._get_object_properties(current_object_from_dt, objects_already_planned,
                                                     class_name, value,
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

        # 4. level: state model class kwargs
        if new_possible is True:
            current_object_dict = self._get_state_model_class_dict(data_entry_mapping.state_model_class,
                                                                   name_space, value)

            # ToDo
            # if domain_specific_static_refinements is not None:
            #     current_object_dict, object_changed = (
            #         self.refine_with_static_attributes(current_object_dict, domain_specific_static_refinements,
            #                                            data_entry_mapping.state_model_class))
            # else:
            object_changed = False
        else:
            return None, object_changed

        if properties is not None:
            for key, value in current_object_dict.items():
                if key in properties:
                    current_object_dict[key] = properties[key]

        if data_entry_mapping.state_model_class == "PassiveMovingResource":
            del current_object_dict["physical_body"]

        if (data_entry_mapping.state_model_class == "Part" and
                get_attr_value(current_object_dict, "name") == "str"):
            raise Exception("The object of type part is not completely filled:", current_object_dict, value)

        # logger.debug(f"Current object: {class_name}  {value}")
        self.store_batch(data_entry_mapping.state_model_class, current_object_dict, name_space)
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

    def _get_state_model_class_dict(self, class_name, name_space, value):
        """Returns an unwritten class params dict from class 'class_name'"""
        current_object_dict = self._state_model.get_object_attributes(object_class_name=class_name)
        if not current_object_dict:
            raise NotImplementedError

        current_object_dict = set_value(object_=current_object_dict,
                                        attr="external_identifications",
                                        value={name_space: [value]})

        if "name" in current_object_dict:
            current_object_dict = set_value(object_=current_object_dict,
                                            attr="name",
                                            value=value)

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
            objects_already_planned = [object_
                                       for object_ in objects_already_planned
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
                if not [name
                        for name in object_.external_identifications[name_space]
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
