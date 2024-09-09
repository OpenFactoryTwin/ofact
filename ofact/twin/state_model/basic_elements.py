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

All the basic elements of the twin model that are needed across the whole model

Classes:
    DigitalTwinObject: Base class for all other classes administrating the ids
    ---
    DomainSpecificAttributes: Administrates domain specific attributes
    ---
    DynamicDigitalTwinObject: Inherit from the DigitalTwinObject and include the dynamic mapping of attribute changes
    ---
    DynamicAttributes: Describes the change of attributes over time
    ---
    DynamicAttributeChangeTracking: Describes the change of a specific attribute over time
    AttributeChangeTracker: Abstract class for tracking attribute changes
    SingleObjectAttributeChangeTracker: Describes the change of a specific single object attribute over time
    ListAttributeChangeTracker: Describes the change of a specific list attribute over time

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter
from copy import copy
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Union, Optional, Type, Any, get_type_hints

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.helpers.helpers import handle_bool, handle_str, handle_numerical_value, get_clean_attribute_name
from ofact.twin.state_model.serialization import InstantiationFromDict

if TYPE_CHECKING:
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.state_model.entities import EntityType

# constants
prints_visible = False

types = {bool: handle_bool,
         str: handle_str,
         float: handle_numerical_value,
         int: handle_numerical_value}


def _get_operator(type_):
    if type_ in types:
        operator = types[type_]
    else:
        raise NotImplementedError(f"Operator type '{type}' is not implemented")

    return operator


class DomainSpecificAttributes(InstantiationFromDict):
    """
    Parameters
    ----------
    attribute allowed_attributes: a list of allowed attribute names and their associated allowed types
    mapped to an object_type characterized through a dict of object class and entity_type
    """
    allowed_attributes: dict[tuple[DigitalTwinObject, EntityType], dict[str, Type]] = {}

    def __init__(self,
                 attributes: Optional[dict[str: object]] = None,
                 attributes_type: Optional[tuple[DigitalTwinObject, EntityType]] = None):
        """
        Used to administrate domain-specific attributes of a digital twin object.
        They are especially used for model learning tasks and further domain-specific analytics.

        Parameters
        ----------
        attributes: a dictionary that maps the attribute values to the attribute names
        attributes_type: describes the relation to object associated with the domain-specific attributes.
        """
        self.type_: Optional[tuple[DigitalTwinObject, EntityType]] = attributes_type

        self.cross_domain_attributes_definition = None
        if attributes_type is not None:
            parent_object = attributes_type[0]
            if parent_object is not None:
                if hasattr(parent_object, "cross_domain_attributes"):
                    self.cross_domain_attributes_definition = parent_object.cross_domain_attributes

        if attributes is None:
            attributes = {}
        self.attributes: dict[str: object] = attributes

    def copy(self):
        """Copy the object"""
        domain_specific_attributes_copy = copy(self)
        domain_specific_attributes_copy.allowed_attributes = \
            domain_specific_attributes_copy.allowed_attributes.copy()
        domain_specific_attributes_copy.attributes = domain_specific_attributes_copy.attributes.copy()

        return domain_specific_attributes_copy

    def get_allowed_attributes(self) -> dict[str, Type]:
        """Return the list of allowed attributes based on the current type."""
        if self.type_ in self.allowed_attributes:
            allowed_attributes = self.allowed_attributes[self.type_]
        else:
            allowed_attributes = {}

        return allowed_attributes

    def add_allowed_attributes(self) -> None:
        pass

    def add_attributes(self, new_attributes: dict, all_attributes_allowed: bool = False):
        """Add an attribute to the (domain-specific) attributes"""
        allowed_attributes = self.get_allowed_attributes()
        for attribute_name, attribute_value in new_attributes.items():
            if attribute_name in allowed_attributes:
                if not isinstance(attribute_value, allowed_attributes[attribute_name]):
                    attribute_value = _get_operator(allowed_attributes[attribute_name])(attribute_value)
                self.attributes[attribute_name] = attribute_value
            elif all_attributes_allowed:
                self.attributes[attribute_name] = attribute_value
            else:
                raise NotImplementedError("The attribute name is not in the allowed_attributes list")

    def remove_attributes(self, old_attributes: dict):
        """Remove an attribute from the (domain-specific) attributes"""
        for attribute_name, attribute_value in old_attributes.items():
            if attribute_name in self.attributes:
                if self.attributes[attribute_name] == attribute_value:
                    del self.attributes[attribute_name]

    def get_attributes(self):
        """Get all (domain-specific) attributes"""
        return self.attributes

    def get_attribute_value(self, attribute_name):
        """Get a specific attribute value for an attribute_name"""
        if attribute_name in self.attributes:
            return self.attributes[attribute_name]
        else:
            raise ValueError(f"No attribute found, matching to the attribute name: {attribute_name} \n"
                             f"The following attributes are available: {self.attributes}")

    def __str__(self):
        return (f"DomainSpecificAttributes of type: '{self.type_}' with attributes: '{self.attributes}'; "
                f"cross_domain_attributes_definition: '{self.cross_domain_attributes_definition}'")


def _create_new_dt_object_label(static_model_original_label):
    """
    Create a new label with a higher number for the object

    Parameters
    ----------
    static_model_original_label: original label that is adapted to be individual.
    Individuality is not ensured.

    Returns
    -------
    new_label: a new label with a higher number for the "new" object to generate a hopefully individual label.
    """

    label_as_list = static_model_original_label.split("_")
    element_with_number = label_as_list[-2]
    number_ = re.search(r'\d+', element_with_number)
    if number_:
        number_extracted = number_.group()
        element_without_number = element_with_number[:-(len(number_extracted))]
        number_extracted = int(number_extracted) + 1
        element_with_number = element_without_number + str(number_extracted)
    else:
        element_with_number = element_with_number + "1"

    label_as_list[-2] = element_with_number

    new_label = "_".join(label_as_list)

    return new_label


def _create_new_dt_object_name(name):
    """Create a new label with a higher number for the object"""

    number_ = re.search(r'\d+', name)
    if number_:
        number_extracted = number_.group()
        name_without_number = name[:-(len(number_extracted))]
        number_extracted = int(number_extracted) + 1
        name_with_number = name_without_number + str(number_extracted)
    else:
        name_with_number = name + " 1"

    return name_with_number


class DigitalTwinObject(InstantiationFromDict, metaclass=ABCMeta):
    next_id: int = 0

    # Note: maybe the id arises from different DT - models which should not the problem
    # But it could be the case that the next_id is set from the loading of a new model that is lower than the next_id of
    # another model - then, the ids are not unique anymore
    # ToDo: the next_id should be model dependent (relevant for more than one digital twin state model)

    @staticmethod
    def get_next_id() -> int:
        return DigitalTwinObject.next_id

    def __init__(self,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Base class for all classes of the digital_twin

        Parameters
        ----------
        identification: unique id, if None is passed identification is generated automatically
        external_identifications: a mapper that maps for external systems individually identifications/
        names/ designations
        domain_specific_attributes: a list of domain_specific_attributes
        """
        if identification is None:
            self.identification: int = self.get_next_id()
            DigitalTwinObject.next_id += 1
        else:
            self.identification: int = identification

        if external_identifications is None:
            external_identifications = {}
        self.external_identifications: dict[object, list[object]] = external_identifications

        if domain_specific_attributes:
            domain_specific_attributes = DomainSpecificAttributes(attributes=domain_specific_attributes)
        self._domain_specific_attributes: Optional[DomainSpecificAttributes] = domain_specific_attributes

    @property
    def domain_specific_attributes(self):
        return self._domain_specific_attributes

    @domain_specific_attributes.setter
    def domain_specific_attributes(self, domain_specific_attributes):
        self._domain_specific_attributes = domain_specific_attributes

    def get_static_model_id(self):
        """
        Used for the static model creation (gives a reference to the object in the "excel_sheets"),
        The sting returned begins with a "_" - character.
        Needed for the excel_importer to ensure that the string remains.
        Respectively, the original label is not replaced by the object referenced to the string
        """
        if "static_model" not in self.external_identifications:
            warn_message = "No label (external_identifications['static_model']) specified for object:" + str({self})
            warnings.warn(warn_message)
            static_model_identification = ""
        else:
            static_model_identification = self.external_identifications["static_model"][0]

        return static_model_identification

    def get_all_external_identifications(self):
        """Return all recorded external_identifications"""
        all_external_identifications = [external_id
                                        for external_id_list in list(self.external_identifications.values())
                                        for external_id in external_id_list]
        return all_external_identifications

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""

        digital_twin_object_copy = copy(self)
        if digital_twin_object_copy._domain_specific_attributes is not None:
            digital_twin_object_copy._domain_specific_attributes = \
                digital_twin_object_copy._domain_specific_attributes.copy()
        digital_twin_object_copy.external_identifications = \
            {external: external_identification_lst.copy()
             for external, external_identification_lst in digital_twin_object_copy.external_identifications.items()}

        return digital_twin_object_copy

    def duplicate(self, external_name: bool = False):
        """
        Duplicates the object with a new identification.
        Entities have an overwritten duplicate method because they really create new objects, also in their attributes.
        Example: An order has the same features if created, only the product could not be the same, but an entity
        (resource or part) could not have the same attributes because the changes of the attributes proceed
        in different directions

        Parameters
        ----------
        external_name: used to manage the static model identification that should be individual in the duplication

        Returns
        -------
        the duplicated object
        """
        digital_twin_object_copy = self.copy()
        digital_twin_object_copy.identification = DigitalTwinObject.next_id
        digital_twin_object_duplicate = digital_twin_object_copy
        if "static_model" in digital_twin_object_duplicate.external_identifications and external_name:
            static_model_original_label = digital_twin_object_duplicate.external_identifications["static_model"][0]
            new_label = _create_new_dt_object_label(static_model_original_label)
            if hasattr(digital_twin_object_duplicate, "name"):
                new_name = _create_new_dt_object_name(digital_twin_object_duplicate.name)
                digital_twin_object_duplicate.name = new_name
            digital_twin_object_duplicate.external_identifications["static_model"][0] = new_label

        DigitalTwinObject.next_id += 1

        return digital_twin_object_duplicate

    @classmethod
    def get_init_parameter_type_hints(cls):
        init_parameter_type_hints = get_type_hints(cls.__init__)
        return init_parameter_type_hints

    @classmethod
    def get_init_parameters_with_defaults(cls):
        pass

    def add_external_identifications(self, name_space, external_id):
        """Add external identification/ designation/ name"""
        self.external_identifications.setdefault(name_space,
                                                 []).append(external_id)

    def remove_external_identifications(self, name_space):
        """Add external identification/ designation/ name"""
        del self.external_identifications[name_space]

    def get_external_identifications_name_space(self, name_space) -> list:
        """Returns a list of external identifications associated with the name space"""

        if name_space in self.external_identifications:
            return self.external_identifications[name_space]
        else:
            return []

    def get_digital_twin_id_by_external_id(self, name_space, external_id) -> Optional[int]:
        """Returns the identification (internal) if external_id matches to the internal DTObject identification
        else None (no matching)"""
        if name_space in self.external_identifications:
            matching = [external_id_intern for external_id_intern in self.external_identifications[name_space]
                        if external_id == external_id_intern]
            if matching:
                return self.identification

        return None

    def get_self_by_external_id(self, name_space, external_id) -> Optional[DigitalTwinObject]:
        """Returns self if external_id matches to the internal DTObject identification
        else None (no matching)"""
        if name_space in self.external_identifications:
            matching = [external_id_intern
                        for external_id_intern in self.external_identifications[name_space]
                        if external_id == external_id_intern]
            if matching:
                return self

        return None

    def add_domain_specific_attributes(self, new_attributes):
        """Add domain-specific attributes to the object"""
        if self._domain_specific_attributes is not None:
            self._domain_specific_attributes.add_attributes(new_attributes)
        else:
            raise NotImplementedError("No domain-specific attributes are specified at the initialization")

    def remove_domain_specific_attributes(self, old_attributes):
        """Remove domain-specific attributes to the object"""
        if self._domain_specific_attributes is not None:
            self._domain_specific_attributes.remove_attributes(old_attributes)
        else:
            raise NotImplementedError("No domain-specific attributes are specified at the initialization")

    def get_domain_specific_attributes(self):
        """Get all (domain-specific) attributes"""
        if self._domain_specific_attributes is not None:
            return self._domain_specific_attributes.get_attributes()
        else:
            return {}

    def get_domain_specific_attribute_value(self, attribute_name):
        """Get a specific attribute value for a domain specific attribute_name"""
        if self._domain_specific_attributes is not None:
            return self._domain_specific_attributes.get_attribute_value(attribute_name)
        else:
            return None


def _get_current_time_change_tracker(current_time):
    if isinstance(current_time, datetime):
        current_time = np.datetime64(current_time, "ns")
    elif current_time != current_time or current_time is None:
        datetime_min = datetime.min
        current_time = np.datetime64(datetime_min, "ns")

    return current_time


def _transform_time_stamp(time_stamp: Optional[Union[datetime, np.datetime64]]) -> Optional[np.datetime64]:
    if time_stamp is not None:
        if isinstance(time_stamp, datetime):
            time_stamp = np.datetime64(time_stamp, "ns")

    return time_stamp


class DynamicAttributeChangeTracking:
    recent_changes_max_memory = 10000
    recent_changes_min_memory = 1000

    def __init__(self, attribute_change_tracker_class: Union[(SingleObjectAttributeChangeTracker,
                                                              ListAttributeChangeTracker)],
                 current_time, attribute_value: Any, process_execution: Optional[ProcessExecution]):
        """
        Used for the tracking of dynamic attributes. In general, the digital twin is created for operative tasks,
        where only a few attribute changes are expected. But since, e.g., for learning tasks or analytic tasks,
        also longer time periods can be required. The dynamic attribute history grows over longer observation periods,
        which means that the performance of the tracking would decrease significantly
        (e.g., significantly more than 10.000 entries). This can quickly lead to a decline in performance as in
        each process execution at least one change can be assumed (with a high probability more).
        To avoid a performance decline, recent and distant past changes are differentiated.
        If a number of "recent_changes_max_memory" entries are crossed, the "number_of_older_entries_to_transfer"
        (="recent_changes_max_memory" - "recent_changes_min_memory") is transferred to the distant past changes.
        In the requests, both recent and distant past changes are considered, but it can be assumed that
        for the standard use cases the recent changes are used solely.
        Nevertheless, depending on the project/ use case, the constant numbers "recent_changes_min_memory" and
        "recent_changes_max_memory" can be changed regarding the requirements.

        Parameters
        ----------
        attribute_change_tracker_class: the class is dependent on the attribute type
        attribute recent_changes: changes of the recent past that is usually used in operative tasks
        attribute distant_past_changes: changes of the distant past that is usually used to avoid performance issues
        attribute start_time_stamp_recent_changes: time stamp of the start of the recent changes
        (can be understood as the time stamp of the first entry)
        attribute distant_past_changes_filled: determines if the distant past is filled
        """

        self.attribute_change_tracker_class: Union[(Type[SingleObjectAttributeChangeTracker],
                                                    Type[ListAttributeChangeTracker])] = attribute_change_tracker_class

        current_time = _get_current_time_change_tracker(current_time)

        self.recent_changes: Union[SingleObjectAttributeChangeTracker, ListAttributeChangeTracker] = (
            self.attribute_change_tracker_class(current_time=current_time,
                                                attribute_value=attribute_value,
                                                process_execution=process_execution))
        self.distant_past_changes: Union[SingleObjectAttributeChangeTracker, ListAttributeChangeTracker] = (
            self.attribute_change_tracker_class())

        self.start_time_stamp_recent_changes: np.datetime64 = np.datetime64(datetime.min, "ns")
        self.distant_past_changes_filled: bool = False

    def copy(self):
        dynamic_attribute_change_tracking_copy = copy(self)
        dynamic_attribute_change_tracking_copy.recent_changes = self.recent_changes.copy()
        dynamic_attribute_change_tracking_copy.distant_past_changes = self.distant_past_changes.copy()
        dynamic_attribute_change_tracking_copy.considered_time_period_recent_changes = (
            copy(self.start_time_stamp_recent_changes))

        return dynamic_attribute_change_tracking_copy

    def duplicate(self):
        dynamic_attribute_change_tracking_duplicate = self.copy()

        return dynamic_attribute_change_tracking_duplicate

    def get_change_history_length(self):
        """Determine the length of both, the recent changes and the distant past changes"""

        recent_changes_length = self.recent_changes.get_change_history_length()
        distant_past_changes_length = self.distant_past_changes.get_change_history_length()
        change_history_length = recent_changes_length + distant_past_changes_length
        return change_history_length

    def get_process_execution_history(self) -> list[ProcessExecution]:
        """Return a list of process_executions that is mapped to attribute changes"""
        changes = self.get_change_array()
        process_executions: list[ProcessExecution] = list(set(changes["ProcessExecution"]))

        return process_executions

    def get_change_array(self, ):
        """Return the changes of the attribute as numpy-array"""

        if self.start_time_stamp_recent_changes is not None:
            recent_changes_array = self.recent_changes.get_change_array()
        else:
            recent_changes_array = np.array([],
                                            dtype=self.recent_changes.attribute_change_tracker_data_type)
        if self.distant_past_changes_filled:
            distant_past_changes_array = self.distant_past_changes.get_change_array()
            change_array = np.concatenate((recent_changes_array, distant_past_changes_array), axis=0)
        else:
            change_array = recent_changes_array

        return change_array

    def add_change(self, current_time, attribute_value, process_execution_plan, process_execution,
                   sequence_already_ensured: bool = False, **kwargs):
        """Add an attribute change to the change tracker"""

        if process_execution is None or current_time != current_time or current_time is None:
            raise Exception(process_execution, current_time)

        current_time = _transform_time_stamp(current_time)

        # determine the tracker, where the change is recorded
        if self.start_time_stamp_recent_changes is not None:
            if current_time >= self.start_time_stamp_recent_changes:
                change_tracker = self.recent_changes
                check_change_transfer = True
            else:
                change_tracker = self.distant_past_changes
                check_change_transfer = False
        else:
            change_tracker = self.recent_changes
            check_change_transfer = True

        change_tracker.add_change(current_time=current_time,
                                  attribute_value=attribute_value,
                                  process_execution_plan=process_execution_plan,
                                  process_execution=process_execution,
                                  sequence_already_ensured=sequence_already_ensured,
                                  **kwargs)

        # transfer betweén the change trackers
        if check_change_transfer:
            change_history_length = self.recent_changes.get_change_history_length()

            if change_history_length > type(self).recent_changes_max_memory:
                # transfer
                self._transfer_old_changes_to_distant_past_changes()

    def _transfer_old_changes_to_distant_past_changes(self):
        """Transfer the old entries from the recent changes to the distant past changes"""

        quantity_of_changes_to_pop = (type(self).recent_changes_max_memory -
                                      type(self).recent_changes_min_memory)
        old_changes_popped = (
            self.recent_changes.pop_old_changes_with_a_quantity_of(
                quantity_of_changes_to_pop=quantity_of_changes_to_pop))
        self.distant_past_changes.add_change_history(old_changes_popped)

        self.distant_past_changes_filled = True

    def get_changes(self, start_time_stamp: datetime, end_time_stamp: datetime) -> dict:

        start_time_stamp = _transform_time_stamp(start_time_stamp)
        end_time_stamp = _transform_time_stamp(end_time_stamp)

        change_histories_required_start = self._get_change_trackers_required(start_time_stamp)
        change_histories_required_end = self._get_change_trackers_required(end_time_stamp)
        change_histories_required = change_histories_required_start + change_histories_required_end

        if "RECENT" in change_histories_required:
            recent_changes = self.recent_changes.get_changes(start_time_stamp, end_time_stamp)
        else:
            recent_changes: dict = {}

        if "DISTANT_PAST" in change_histories_required:
            distant_past_changes: dict = self.distant_past_changes.get_changes(start_time_stamp, end_time_stamp)
            changes: dict = distant_past_changes | recent_changes
        else:
            changes: dict = recent_changes

        return changes

    def get_last_change_before(self, time_stamp: datetime) -> [Optional[datetime], object]:

        time_stamp = _transform_time_stamp(time_stamp)

        change_histories_required = self._get_change_trackers_required(time_stamp)

        if "RECENT" in change_histories_required:
            change_time_stamp, attribute_value = self.recent_changes.get_last_change_before(time_stamp)
        elif "DISTANT_PAST" in change_histories_required:
            change_time_stamp, attribute_value = self.distant_past_changes.get_last_change_before(time_stamp)
        else:
            raise Exception

        return change_time_stamp, attribute_value

    def _get_change_trackers_required(self, time_stamp) -> list[str]:
        change_histories_required = []

        if time_stamp is not None:
            if self.start_time_stamp_recent_changes <= time_stamp:
                change_histories_required.append("RECENT")
            else:
                change_histories_required.append("DISTANT_PAST")

        return change_histories_required

    def get_latest_version(self):
        return self.recent_changes.get_latest_version()

    def get_version(self, req_time_stamp: datetime = None):
        req_time_stamp = _transform_time_stamp(req_time_stamp)

        if self.start_time_stamp_recent_changes <= req_time_stamp:
            version_at_req_time_stamp = self.recent_changes.get_version(req_time_stamp)
        else:
            version_at_req_time_stamp = self.distant_past_changes.get_version(req_time_stamp)

        return version_at_req_time_stamp


class AttributeChangeTracker:

    def __init__(self, current_time, attribute_value, process_execution):
        """
        Used for attributes to track their changes over time.

        Parameters
        ----------
        attribute changes:
            - time_stamp: time_stamp of the change
            - attribute_value: new value of the attribute
            - process_execution: responsible for the change

        Note: A chronological order should be ensured from the beginning.
        A later sorting (e.g., in the requesting) cannot ensure that two process executions that are executed
        at the same time are in the right order.

        Note: first instantiation in the Excel import could contain not complete values
        """

        self.changes = np.array([[current_time, attribute_value, process_execution]],
                                dtype=[("Timestamp", "datetime64[ns]"),
                                       ("Value", object),
                                       ("ProcessExecution", object)])

    def copy(self):
        attribute_change_tracker_copy = copy(self)
        attribute_change_tracker_copy.changes = self.changes.copy()

        return attribute_change_tracker_copy

    def duplicate(self):
        attribute_change_tracker_duplicate = copy(self)

        return attribute_change_tracker_duplicate

    def get_change_history_length(self):
        """
        Determine the history length of the attribute change tracker
        Note: the "INITIAL" value is also counted
        """
        change_history_length = self.changes.shape[0]
        return change_history_length

    def pop_old_changes_with_a_quantity_of(self, quantity_of_changes_to_pop: int = 1):
        """Pop the old entries from the recent change tracker"""

        old_entries_popped = self.changes[:quantity_of_changes_to_pop]
        self.changes = self.changes[quantity_of_changes_to_pop:]

        return old_entries_popped

    def add_change_history(self, change_history):
        """Append the entries from the recent change tracker to the distant past change tracker"""

        self.changes = np.concatenate((self.changes, change_history),
                                      axis=0)

    def _get_last_entry_before_index(self, current_time: np.datetime64, sequence_already_ensured: bool):

        if not sequence_already_ensured:
            time_mask = self.changes["Timestamp"] <= current_time
            entries_before = self.changes[time_mask]
            last_entry_before_idx = entries_before.shape[0] - 1
        else:
            last_entry_before_idx = self.changes.shape[0] - 1

        return last_entry_before_idx

    def get_change_array(self, ):
        """Return the changes of the attribute as numpy-array"""
        return self.changes

    @abstractmethod
    def add_change(self, current_time, attribute_value, process_execution_plan, process_execution,
                   sequence_already_ensured: bool = False, **kwargs):
        """Add an attribute change to the change tracker"""
        pass

    @abstractmethod
    def get_changes(self, start_time_stamp: np.datetime64, end_time_stamp: np.datetime64) -> dict:
        changes: dict
        return changes

    @abstractmethod
    def get_last_change_before(self, time_stamp: np.datetime64) -> [Optional[datetime], object]:
        change_time_stamp: datetime
        attribute_value: object
        return change_time_stamp, attribute_value

    @abstractmethod
    def get_latest_version(self):
        return

    @abstractmethod
    def get_version(self, req_time_stamp: int = None):
        return


single_object_attribute_change_tracker_data_type = [("Timestamp", "datetime64[ns]"),
                                                    ("Value", object),
                                                    ("ProcessExecution", object)]


class SingleObjectAttributeChangeTracker(AttributeChangeTracker):
    attribute_change_tracker_data_type = single_object_attribute_change_tracker_data_type

    def __init__(self, current_time: Optional[datetime] = None, attribute_value: Optional[Any] = None,
                 process_execution: Optional[ProcessExecution] = None):
        """
        Used for single object attributes such as quality value (float) to track their changes over time.
        Parameters
        ----------
        attribute changes:
            - time_stamp: time_stamp of the change
            - attribute_value: new value of the attribute
            - process_execution: responsible for the change
        """

        current_time = _get_current_time_change_tracker(current_time)
        if process_execution is None:
            process_execution = "INITIAL"

        self.changes = np.array([(current_time, attribute_value, process_execution)],
                                dtype=type(self).attribute_change_tracker_data_type)

    def add_change(self, current_time, attribute_value, process_execution_plan, process_execution,
                   sequence_already_ensured: bool = False, **kwargs):

        # if an actual process_executions updates the attribute,
        # the connected process_execution plan is deleted replaced
        if process_execution_plan is not None:
            process_execution_mask = self.changes["ProcessExecution"] == process_execution_plan
            process_execution_plan_entry = self.changes[process_execution_mask]
            if process_execution_plan_entry.shape[0] == 1:
                new_change = np.array([(current_time, attribute_value, process_execution)],
                                      dtype=type(self).attribute_change_tracker_data_type)
                self.changes[process_execution_mask] = new_change
                return

        # some processes have a zero seconds process_time.
        # therefore, it is also possible that attribute changes can occur at the same time
        # caused by different process_executions

        last_entry_before_idx = self._get_last_entry_before_index(current_time, sequence_already_ensured)

        if len(self.changes) > 0:
            if self.changes[last_entry_before_idx][1] == attribute_value:
                return

        new_change = np.array([(current_time, attribute_value, process_execution)],
                              dtype=type(self).attribute_change_tracker_data_type)
        self.changes = np.insert(self.changes, last_entry_before_idx + 1, new_change,
                                 axis=0)

    def get_changes(self, start_time_stamp: np.datetime64, end_time_stamp: np.datetime64) -> dict:

        attribute_history = self.changes[["Timestamp", "Value"]]

        first_values = attribute_history[attribute_history["Timestamp"] <= start_time_stamp]
        if first_values.size == 0:
            idx = 0
        else:
            idx = len(first_values)

        if idx >= attribute_history.shape[0]:
            idx -= 1

        if attribute_history.size == 0:
            return {start_time_stamp: None}

        attribute_changes_first_time_stamp = {start_time_stamp: attribute_history["Value"][idx]}

        if start_time_stamp is not None and end_time_stamp is not None:
            entries = attribute_history[(start_time_stamp < attribute_history["Timestamp"]) &
                                        (attribute_history["Timestamp"] < end_time_stamp)]
            attribute_changes_in_time_period = dict(zip(entries["Timestamp"], entries["Value"]))
            attribute_changes_in_time_period = attribute_changes_first_time_stamp | attribute_changes_in_time_period

        elif start_time_stamp is not None:
            entries = attribute_history[start_time_stamp < attribute_history["Timestamp"]]
            attribute_changes_in_time_period = dict(zip(entries["Timestamp"], entries["Value"]))
            attribute_changes_in_time_period = attribute_changes_first_time_stamp | attribute_changes_in_time_period

        elif end_time_stamp is not None:
            entries = attribute_history[attribute_history["Timestamp"] < end_time_stamp]
            attribute_changes_in_time_period = dict(zip(entries["Timestamp"], entries["Value"]))
            attribute_changes_in_time_period = dict(zip(attribute_history[0])) | attribute_changes_in_time_period

        else:
            attribute_changes_in_time_period = attribute_history

        return attribute_changes_in_time_period

    def get_last_change_before(self, time_stamp: np.datetime64) -> [Optional[datetime], object]:

        attribute_history = self.changes[:2]

        # attribute_history[1:] the first time_stamp is always 0/ not a datetime object -
        # therefore, it cannot be compared with datetime objects

        first_values = attribute_history[1:][attribute_history["Timestamp"][1:] <= time_stamp]
        if first_values.size == 0:
            idx = 0
            change_time_stamp = None

        else:
            idx = len(first_values)
            change_time_stamp = attribute_history["Timestamp"][idx]

        attribute_value = attribute_history["Value"][idx]

        return change_time_stamp, attribute_value

    def get_latest_version(self):
        if len(self.changes[-1]) > 0:
            return self.changes[-1][1]
        else:
            return None

    def get_version(self, req_time_stamp: int = None):
        changes_before = self.changes[self.changes["Timestamp"] <= req_time_stamp]
        if changes_before.shape[0] == 0:
            return None

        version = changes_before[-1]
        return version


list_attribute_change_tracker_data_type = [("Timestamp", "datetime64[ns]"),
                                           ("Value", object),
                                           ("ChangeType", "float16"),
                                           ("ProcessExecution", object)]


class ListAttributeChangeTracker(AttributeChangeTracker):
    attribute_change_tracker_data_type = list_attribute_change_tracker_data_type

    def __init__(self, current_time: Optional[datetime] = None, attribute_value: Optional[Any] = None,
                 process_execution: Optional[ProcessExecution] = None):
        """
        Used for list attributes such as a list of stored entities (List[Entity]) to track their changes over time.
        It extends the single object change tracker with the additional column change_type
        Parameters
        ----------
        attribute changes:
            - time_stamp: time_stamp of the change
            - attribute_value: new value of the attribute
            - change_type: "ADD" (1) or "REMOVE" (-1)
            - process_execution: responsible for the change
        if None, the process_executions responsible for the list can be found in the history
        """
        change_type = 1
        if attribute_value is None:
            attribute_value = []
        if process_execution is None:
            process_execution = "INITIAL"

        change_entries = [(current_time, list_entry, change_type, process_execution)
                          for list_entry in attribute_value]

        self.changes = np.array(change_entries,
                                dtype=type(self).attribute_change_tracker_data_type)

    def pop_old_changes_with_a_quantity_of(self, quantity_of_changes_to_pop: int = 1):

        old_entries_popped = self.changes[:quantity_of_changes_to_pop]
        new_first_time_stamp = self.changes["Timestamp"][quantity_of_changes_to_pop - 1]
        initial_list = self._create_list(new_first_time_stamp)

        initial_current_time = None
        change_type = 1
        process_execution = "INITIAL"
        initial_list_change_entries = (
            np.array([(initial_current_time, list_entry, change_type, process_execution)
                      for list_entry in initial_list],
                     dtype=type(self).attribute_change_tracker_data_type))

        self.changes = np.concatenate((initial_list_change_entries, self.changes[quantity_of_changes_to_pop:]),
                                      axis=0)

        return old_entries_popped

    def add_change_history(self, change_history):
        """Append the entries from the recent change tracker to the distant past change tracker"""

        self.changes = np.concatenate((self.changes, change_history),
                                      axis=0)

    def add_change(self, current_time, attribute_value, process_execution_plan: Optional[ProcessExecution],
                   process_execution: ProcessExecution,
                   change_type="ADD", sequence_already_ensured: bool = False, **kwargs):

        change_type_mapper = {"ADD": 1, "REMOVE": -1}
        change_type = change_type_mapper[change_type]

        # if an actual process_executions updates the attribute,
        # the connected process_execution, plan is deleted replaced
        if process_execution_plan is not None:
            process_execution_mask = self.changes["ProcessExecution"] == process_execution_plan
            process_execution_plan_entry = self.changes[process_execution_mask]
            if process_execution_plan_entry.shape[0] == 1:
                self.changes[process_execution_mask] = (
                    np.array([(current_time, attribute_value, change_type, process_execution)],
                             dtype=type(self).attribute_change_tracker_data_type))
                return

        # some processes have a zero seconds process_time;
        # therefore, it is also possible that attribute changes can occur at the same time
        # caused by different process_executions
        last_entry_before_idx = self._get_last_entry_before_index(current_time, sequence_already_ensured)

        new_change = np.array([(current_time, attribute_value, change_type, process_execution)],
                              dtype=type(self).attribute_change_tracker_data_type)
        self.changes = np.insert(self.changes, last_entry_before_idx + 1, new_change,
                                 axis=0)

    def get_changes(self, start_time_stamp: np.datetime64, end_time_stamp: np.datetime64) -> dict:

        # list at the start_time_stamp
        first_time_stamp_list = self._create_list(start_time_stamp)
        attribute_changes_in_time_period = {start_time_stamp: first_time_stamp_list}

        # list over time
        attribute_history_after_start_time_complete = self.changes[(start_time_stamp < self.changes["Timestamp"]) &
                                                                   (self.changes["Timestamp"] < end_time_stamp)]
        attribute_history_after_start_time = (
            attribute_history_after_start_time_complete[['Timestamp', 'Value', 'ChangeType']])

        time_stamp_list = first_time_stamp_list.copy()
        if attribute_history_after_start_time.shape[0] > 0:

            for index in range(attribute_history_after_start_time.shape[0]):
                entry = attribute_history_after_start_time[index]

                if entry['ChangeType'] == 1:
                    time_stamp_list += [entry['Value']]
                else:
                    time_stamp_list.remove(entry['Value'])

                attribute_changes_in_time_period[entry['Timestamp']] = time_stamp_list.copy()

        # list at the end_time_stamp
        end_time_stamp_entries = self.changes[self.changes["Timestamp"] <= end_time_stamp]
        if end_time_stamp_entries.size == 0:
            return attribute_changes_in_time_period

        end_time_stamp_entry = end_time_stamp_entries[-1]
        attr_value, change_type = end_time_stamp_entry[['Value', 'ChangeType']]
        if change_type == 1:
            time_stamp_list += [attr_value]
        else:
            time_stamp_list.remove(attr_value)

        attribute_changes_in_time_period[end_time_stamp] = time_stamp_list

        return attribute_changes_in_time_period

    def get_last_change_before(self, time_stamp: datetime) -> [Optional[datetime], object]:
        raise NotImplementedError

    def get_latest_version(self):
        latest_version_list = self._create_list(self.changes[-1][0])
        return latest_version_list

    def get_version(self, req_time_stamp: int = None):
        list_at_req_time_stamp = self._create_list(req_time_stamp)
        return list_at_req_time_stamp

    def _create_list(self, req_time_stamp) -> list:
        if isinstance(req_time_stamp, datetime):
            req_time_stamp = np.datetime64(req_time_stamp, "ns")

        # self.changes = self.changes[self.changes["Timestamp"] == self.changes["Timestamp"]]  # exclude nan values
        change_before = self.changes[self.changes["Timestamp"] <= req_time_stamp]

        changes_added = Counter(change_before[change_before["ChangeType"] == 1]["Value"])
        changes_removed = Counter(change_before[change_before["ChangeType"] == -1]["Value"])

        # Note: e.g.: in the features_requested list, a feature can be occurred more than one time (set not usable)
        # This is also True for the stored_entities of a storage
        changes_remain_dict = changes_added - changes_removed
        changes_remain = list(changes_remain_dict.keys())

        return changes_remain


class DynamicAttributes(InstantiationFromDict):

    def __init__(self,
                 attributes: dict[str: object],
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime.min):
        """
        Used to work with attributes that change their values over time,

        Parameters
        ----------
        attributes: DynamicAttributes contain a dictionary.
        The key contains the dynamic attribute name
        and the value contains the change history identifiable by timestamp,
        value that changes over time and the process_execution responsible for the change.
        process_execution: ProcessExecution that has triggered the change/ None if at beginning
        current_time: time when the change happened/0 if at beginning
         ToDo: all changes should be mapped, the changes that are no changes (same value) too
          because in the interim time, further changes can occur -> if the writing sequence is not time chronological
          Real Use Cases?
        """
        if current_time is None or current_time == 0:
            current_time = datetime.min
        self.time_stamps = [current_time]

        self.attributes: dict[str, DynamicAttributeChangeTracking] = \
            {get_clean_attribute_name(attribute_name):
                 self.get_change_tracking(current_time, attribute_value, process_execution)
             for attribute_name, attribute_value in attributes.items()}

        self.latest_requested_version = current_time

    @staticmethod
    def get_change_tracking(current_time, attribute_value, process_execution) -> DynamicAttributeChangeTracking:
        """
        Depending on the attribute value type,
        a DynamicAttributeChangeTracking with a different AttributeChangeTracker is returned
        """

        if not isinstance(attribute_value, list):
            change_tracker = SingleObjectAttributeChangeTracker
        else:
            change_tracker = ListAttributeChangeTracker

        change_tracking = (
            DynamicAttributeChangeTracking(attribute_change_tracker_class=change_tracker, current_time=current_time,
                                           attribute_value=attribute_value, process_execution=process_execution))

        return change_tracking

    def copy(self):
        dynamic_attributes_copy = copy(self)
        dynamic_attributes_copy.time_stamps = dynamic_attributes_copy.time_stamps.copy()
        dynamic_attributes_copy.attributes = dynamic_attributes_copy.attributes.copy()

        return dynamic_attributes_copy

    def duplicate(self):

        dynamic_attributes_duplicate = self.copy()
        dynamic_attributes_duplicate.attributes: dict[str, DynamicAttributeChangeTracking] = \
            {attribute_name: attribute_change_tracking.duplicate()
             for attribute_name, attribute_change_tracking in dynamic_attributes_duplicate.attributes.items()}

        return dynamic_attributes_duplicate

    def add_change(self, attributes: dict[str: object], process_execution, current_time: datetime,
                   change_type: str, sequence_already_ensured: bool = False):
        """
        Append a new change_record (process_execution and timestamp) to the attributes.

        Parameters
        ----------
        attributes: dict with the attribute_names (key) and their respective (new_)values
        process_execution: process_execution which is responsible for the change
        current_time: timestamp on which the changes take effect
        change_type: "ADD" or "REMOVE" only used for list attributes
        sequence_already_ensured: used for performance assuming that the execution of the process is performed
        sequentially in a time chronological order.
        """
        self.time_stamps.append(current_time)

        process_execution_plan = None
        if process_execution.check_actual():
            process_execution_plan = process_execution.connected_process_execution

        for attribute_name, attribute_value in attributes.items():
            attribute_name_clean = get_clean_attribute_name(attribute_name)
            change_tracker = self.attributes[attribute_name_clean]
            # print("Attribute Name:", attribute_name, change_type)
            change_tracker.add_change(current_time=current_time, attribute_value=attribute_value,
                                      process_execution_plan=process_execution_plan,
                                      process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured, change_type=change_type)

        self.latest_requested_version = current_time

    def get_process_execution_history(self) -> list[ProcessExecution]:
        """ Return a list of process_executions that is mapped to attribute changes"""
        process_executions: list[ProcessExecution] = []

        dynamic_attribute_change_tracking_list: list[DynamicAttributeChangeTracking] = list(self.attributes.values())
        for dynamic_attribute_change_tracking in dynamic_attribute_change_tracking_list:
            process_executions_attribute = dynamic_attribute_change_tracking.get_process_execution_history()
            process_executions.extend(process_executions_attribute)

        process_executions = list(set(process_executions))

        return process_executions

    def get_changes_attribute(self, attribute: str, start_time_stamp: datetime, end_time_stamp: datetime):
        """Get changes for the attribute in the period between start_time_stamp and end_time_stamp"""

        if attribute not in self.attributes:
            raise Exception("Attribute not in attributes")

        attribute_change_tracker = self.attributes[attribute]
        attribute_changes_in_time_period = attribute_change_tracker.get_changes(start_time_stamp, end_time_stamp)

        return attribute_changes_in_time_period

    def get_last_change_of_attribute_before(self, attribute: str, time_stamp: datetime) -> [Optional[datetime], object]:
        """
        Get the last change before time_stamp with the change time stamp

        Returns
        -------
        change_time_stamp: time_stamp, the change is executed
        attribute_value: the new value of the attribute
        """

        if attribute not in self.attributes:
            raise Exception("Attribute not in attributes")

        attribute_change_tracker: DynamicAttributeChangeTracking = self.attributes[attribute]
        change_time_stamp, attribute_value = attribute_change_tracker.get_last_change_before(time_stamp)

        return change_time_stamp, attribute_value

    def get_latest_version(self):
        """
        Used to get the latest version of all dynamic attributes of the entity.

        Returns
        -------
        attributes_in_new_version: a dict with the attribute_names (keys) and their current values (value)
        """
        attributes_in_new_version = {attribute_name: attribute_change_tracker.get_latest_version()
                                     for attribute_name, attribute_change_tracker in self.attributes.items()}

        self.latest_requested_version = self.time_stamps[-1]
        return attributes_in_new_version

    def get_version(self, req_time_stamp: int = None):
        """
        Used to get the version at a specific timestamp.

        Parameters
        ----------
        req_time_stamp: a timestamp

        Returns
        -------
        attributes_in_new_version: a dict with attribute_names (keys) and their values (value) at the req_timestamp
        """
        time_stamp = 0
        attributes_in_new_version = {}
        for attribute_name, attribute_change_tracker in self.attributes.items():
            time_stamp, attributes_in_new_version[attribute_name] = attribute_change_tracker.get_version(req_time_stamp)

        self.latest_requested_version = time_stamp
        return attributes_in_new_version

    def get_attribute_at(self, req_time_stamp, attribute):
        attribute_change_tracker = self.attributes[attribute]
        attribute_state_at_req_time_stamp = attribute_change_tracker.get_version(req_time_stamp)

        return attribute_state_at_req_time_stamp


class DynamicDigitalTwinObject(DigitalTwinObject):

    def __init__(self,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime.min,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Base Class for a dynamic object in the digital twin, that change some of their values over time
        Classes derived from this class must firstly introduce the attributes,
        that can change over time in the init.
        After the super().init() call, the non-changing attributes are set.

        Example
        -------
        self.temperature = temperature
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name = name

        The "temperature" attribute changes over time and is therefore tracked in the Dynamic attributes.
        The name is assumed to be constant over time and not tracked.


        Parameters
        ----------
        process_execution: ProcessExecution that has triggered the change/ None if at beginning
        current_time: time when the change happened/0 if at beginning
        attribute dynamic_attributes: DynamicAttributes contain a dictionary of lists that includes for every dynamic
        attribute a list with a tuple of timestamps and corresponding values, the process_execution and the time_stamp
        """
        self.dynamic_attributes: DynamicAttributes = DynamicAttributes(self.__dict__, process_execution, current_time)
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        dynamic_digital_twin_object_copy: DynamicDigitalTwinObject = super(DynamicDigitalTwinObject, self).copy()
        dynamic_digital_twin_object_copy.dynamic_attributes = dynamic_digital_twin_object_copy.dynamic_attributes.copy()

        return dynamic_digital_twin_object_copy

    def duplicate(self, external_name=False):

        dynamic_digital_twin_object_duplicate = super(DynamicDigitalTwinObject, self).duplicate(external_name)
        dynamic_digital_twin_object_duplicate.dynamic_attributes = (
            dynamic_digital_twin_object_duplicate.dynamic_attributes.duplicate())

        return dynamic_digital_twin_object_duplicate

    def update_attributes(self, process_execution: ProcessExecution, current_time: datetime, change_type=None,
                          sequence_already_ensured: bool = False, **attributes):
        """
        Used to add a new version of the dynamic attributes

        Parameters
        ----------
        process_execution: process execution
        current_time: timestamp
        change_type: used if attribute is a list
        sequence_already_ensured: used for performance assuming that the execution of the process is performed
        sequentially in a time chronological order.
        attributes: attributes
        """
        attributes_dict = {key: value
                           for key, value in attributes.items()}

        if current_time is None:
            raise Exception(f"The current time must be specified (to use the time line over attributes)!")

        self.dynamic_attributes.add_change(attributes=attributes_dict, process_execution=process_execution,
                                           current_time=current_time, change_type=change_type,
                                           sequence_already_ensured=sequence_already_ensured)

    def set_different_version(self, time_stamp):
        """
        Used to change the version of the dynamic attributes

        Parameters
        ----------
        time_stamp: timestamp of the requested_version
        """
        attributes = self.dynamic_attributes.get_version(time_stamp)
        for key, value in attributes.items():
            self.__dict__[key] = value

    def get_active_time_stamp(self):
        """
        Get the last active timestamp (creation or change occurred)

        Returns
        -------
        the last_requested_version timestamp
        """
        return self.dynamic_attributes.latest_requested_version

    def get_attribute_at(self, req_time_stamp, attribute):
        """
        Parameters
        ----------
        req_time_stamp: Timestamp from which the attribute state is requested
        attribute: attribute name requested (if None, all attributes meant)
        """
        return self.dynamic_attributes.get_attribute_at(req_time_stamp, attribute)

    def get_changes_attribute(self, attribute, start_time_stamp, end_time_stamp):
        """Get changes in the period between start_time_stamp and end_time_stamp for the attribute"""
        return self.dynamic_attributes.get_changes_attribute(attribute, start_time_stamp, end_time_stamp)

    def get_process_execution_history(self):
        """Return a list of process_executions that is mapped to attribute changes"""
        return self.dynamic_attributes.get_process_execution_history()

    def get_last_change_of_attribute_before(self, attribute, time_stamp) -> [Optional[datetime], object]:
        """Get the last change before time_stamp"""
        return self.dynamic_attributes.get_last_change_of_attribute_before(attribute, time_stamp)


# to provide the opportunity to import the enum also in the 'entities' file
# (not possible to store it in processes because of circular calls)
ProcessExecutionTypes = Enum('ProcessExecutionEventTypes',
                             'PLAN ACTUAL',
                             module='DigitalTwin.model.processes',
                             qualname='ProcessExecution.EventTypes')

if __name__ == "__main__":
    pass
