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

All the process elements of the twin.
Processes are used within the process execution to change the state of the digital twin.
They show the opportunities as well as the restrictions the digital twin model can be emerged.
Classes:
    WorkOrder: Manage the process completion of the sales order
    ---
    ProcessExecution: Describe the dynamics in the system by assigning timestamps and used entities to processes
    ---
    Process: Transform parts using resources
    ValueAddedProcess: Assembly process that is executed by the assembly station
    ---
    ProcessTimeController: Describes the time a process need
    QualityController: Describes the change of the quality of the process
    ResourceController: All resources in a group are needed to execute a process
    TransitionController: Describes the change of position/ resource
    TransformationController: Describes the precedence (priority) graph/ transformation of a process

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Union, Optional

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import (DigitalTwinObject, DynamicDigitalTwinObject, ProcessExecutionTypes,
                                                   prints_visible)
from ofact.twin.state_model.entities import Part, Resource, NonStationaryResource, StationaryResource, Storage, \
    ActiveMovingResource
from ofact.twin.state_model.helpers.helpers import convert_to_datetime
from ofact.twin.state_model.process_models import (ProcessTimeModel, QualityModel, ResourceModel, ResourceGroup,
                                                   TransitionModel, EntityTransformationNode, TransformationModel,
                                                   EntityTransformationNodeIoBehaviours,
                                                   EntityTransformationNodeTransformationTypes)
from ofact.twin.state_model.time import ProcessExecutionPlan, WorkCalender

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.entities import EntityType, PartType, Entity
    from ofact.twin.state_model.sales import Feature, Order
    from ofact.twin.state_model.process_models import DTModel

logging.debug("DigitalTwin/processes")


class ProcessController(DigitalTwinObject, metaclass=ABCMeta):

    def __init__(self,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The process is used as a base class for all process controllers.
        A process controller is used to execute the effects of the process or
        checks if all required elements needed are available.
        The effects, respectively, the requirements are derived from the process model.
        Additionally, the model management is conducted by the process controller,
        which includes versioning and model update (retraining).
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

    @abstractmethod
    def get_model(self) -> DTModel:
        """Returns the process_model"""
        pass

    ####################################################################################################################
    # #### METHODS FOR MACHINE LEARNING MODELS
    ####################################################################################################################

    def set_digital_twin(self, digital_twin_model: StateModel):
        """Set the digital twin model after the instantiation of the digital twin model,
        because the process models are firstly instantiated"""
        process_model = self.get_model()
        process_model.set_digital_twin_model(digital_twin_model=digital_twin_model)

    def save_model(self, model_path=None, persistent_saving: bool = False):
        """Save process model (e.g., the weights of the neural network)"""
        process_model = self.get_model()
        if process_model.is_re_trainable():
            process_model.save_prediction_model(model_path, persistent_saving)

    def update_initial(self):
        if self.model_is_re_trainable():
            process_model = self.get_model()
            process_model.update_initial()

    def update_learning_parameters(self, digital_twin_model: StateModel):
        """
        Used to update learning parameters such as the workers available.
        For example, the workers can vary over time, because some workers enter the shop floor or others leave them.
        This is done to hold the models up to date and react to the changes on the shop floor
        """
        if self.model_is_re_trainable():
            model = self.get_model()
            model.update_learning_parameters(digital_twin_model)

    def delete_run_time_attributes(self):
        process_model = self.get_model()
        if process_model.is_re_trainable():
            process_model.delete_run_time_attributes()

    def model_is_re_trainable(self) -> bool:
        process_model = self.get_model()
        if not hasattr(process_model, "is_re_trainable"):
            print("Object has no is_re_trainable method:", self.external_identifications, type(self))
            raise Exception(self.external_identifications)
        return process_model.is_re_trainable()

    def model_retraining_needed(self):
        process_model = self.get_model()
        return process_model.retraining_needed()

    def retrain(self, test_batch=None, batch_size=None):
        """
        Retrain a duplicate of the process model and replace it if suits better.
        """
        process_model = self.get_model()
        process_model_duplicate: DTModel = process_model.duplicate()
        if not process_model_duplicate.is_re_trainable():
            raise Exception
        process_model_duplicate.retrain(test_batch, batch_size)

        print(f"[{self.__class__.__name__:20}] Retraining finished")


def _check_resource_group_applicable(available_resource_ets: list[EntityType], resource_group: ResourceGroup) -> bool:
    resource_entity_types = resource_group.resources

    matches = [False
               for resource_et in available_resource_ets
               if not _check_entity_type_in_entity_type_list(resource_et, resource_entity_types)]

    resource_group_applicable = bool(bool(matches)) - 1  # inversion needed

    return resource_group_applicable


def _check_entity_type_in_entity_type_list(entity_type: EntityType, entity_types: list[EntityType]) -> bool:
    if entity_type in entity_types:
        return True

    if entity_type.super_entity_type in entity_types:
        return True

    return False


class ResourceController(ProcessController):

    def __init__(self,
                 resource_model: ResourceModel,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._resource_model: ResourceModel = resource_model

    @property
    def resource_model(self):
        return self._resource_model

    @resource_model.setter
    def resource_model(self, resource_model):
        self._resource_model = resource_model

    def get_model(self) -> ResourceModel:
        return self._resource_model

    def copy(self):
        """Copy the object with the same identification."""
        resource_model_copy: ResourceController = super(ResourceController, self).copy()
        resource_model_copy._resource_model = resource_model_copy._resource_model.copy()

        return resource_model_copy

    def get_resource_groups(self, process_execution: Optional[ProcessExecution] = None) -> list[ResourceGroup]:
        """
        Resource groups are specified through a list of resources and a list of main resource needed
        to successfully execute a process.
        """

        resource_groups = self._resource_model.get_resource_groups(process_execution)

        return resource_groups

    def check_ability_to_perform_process_as_resource(self, resource, already_assigned_resources: list[Resource] = []) \
            -> bool:
        """
        Check the ability of the resource to perform the process

        Parameters
        resource: a resource
        already_assigned_resources: resources that are already used for the process (not needed anymore)

        Returns
        -------
        ability_to_perform_process_as_resource: True if the resource can perform the process, else False
        """
        resource_groups = self.get_resource_groups()
        ability_to_perform_process_as_resource_per_resource_group = \
            [resource_group.check_ability_to_perform_process_as_resource(
                resource=resource,
                already_assigned_resources=already_assigned_resources)
                for resource_group in resource_groups]
        ability_to_perform_process_as_resource = any(ability_to_perform_process_as_resource_per_resource_group)

        return ability_to_perform_process_as_resource

    def check_ability_to_perform_process_as_main_resource(self, resource) -> bool:
        """
        Check the ability of the resource to perform the process as main_resource

        Parameters
        ----------
        resource: a resource

        Returns
        -------
        ability_to_perform_process_as_main_resource:
        True if the resource can perform the process as main_resource, else False
        """
        resource_groups = self.get_resource_groups()
        ability_to_perform_process_as_main_resource_per_resource_group = \
            [resource_group.check_ability_to_perform_process_as_main_resource(resource=resource)
             for resource_group in resource_groups]
        ability_to_perform_process_as_main_resource = any(
            ability_to_perform_process_as_main_resource_per_resource_group)

        return ability_to_perform_process_as_main_resource

    def get_possible_resource_groups(self, resources, main_resource) -> list[ResourceGroup]:
        """Get possible resource models with the resources and main_resource given."""
        resource_groups = self.get_resource_groups()
        resource_groups = [resource_group
                           for resource_group in resource_groups
                           if resource_group.check_resource_group_applicability(resources, main_resource)]
        return resource_groups

    def get_resource_groups_for_main_resource(self, main_resource) -> list[ResourceGroup]:
        """Find all resource_models that can be processed by the main_resource (input_parameter) as main_resource"""
        resource_groups = self.get_resource_groups()
        resource_groups = [resource_group
                           for resource_group in resource_groups
                           if resource_group.check_ability_to_perform_process_as_main_resource(main_resource)]
        return resource_groups

    def get_usable_resources_for_process(self, available_resources) -> list[tuple[Resource, EntityTransformationNode]]:
        """Used to determine which (available) resources (for example, organized in processes before)
                can be used in this resource_models of the process"""
        resource_groups = self.get_resource_groups()
        usable_resources = [resource_group.get_usable_resources_for_process(available_resources)
                            for resource_group in resource_groups
                            if resource_group.get_usable_resources_for_process(available_resources)]
        if len(usable_resources) == 1:
            usable_resources = usable_resources[0]
        elif len(usable_resources) > 1:
            usable_resources = []
        else:
            usable_resources = []

        return usable_resources

    def check_resources_build_resource_group(self, available_resources: list[Resource],
                                             available_main_resource: Resource) -> bool:
        """Check if the input resources can complete at most one resource_model/-group"""
        resource_groups = self.get_resource_groups()
        usable_resource_models = \
            [resource_group
             for resource_group in resource_groups
             if resource_group.check_resources_build_resource_group(available_resources=available_resources,
                                                                    available_main_resource=available_main_resource)]
        if usable_resource_models:
            return True
        else:
            return False

    def get_possible_resource_entity_types(self, available_resources: list[Resource] = []):
        resource_groups = self.get_resource_groups()
        available_resource_ets = [available_resource.entity_type
                                  for available_resource in available_resources]

        possible_resource_entity_types = []
        for resource_group in resource_groups:
            applicable = True
            for resource_et in available_resource_ets:
                if resource_et in resource_group.resources:
                    continue

                if resource_et.super_entity_type in resource_group.resources:
                    continue

                applicable = False

            if applicable:
                possible_resource_entity_types += resource_group.resources

        # possible_resource_entity_types = [resource
        #                                   for resource_group in resource_groups
        #                                   if _check_resource_group_applicable(available_resource_ets, resource_group)
        #                                   for resource in resource_group.resources]  # ToDo: test alternative
        possible_resource_entity_types_set = list(set(possible_resource_entity_types))

        return possible_resource_entity_types_set

    def get_possible_main_resource_entity_types(self, available_resources=[], available_main_resource=None) \
            -> list[EntityType]:
        resource_groups = self.get_resource_groups()

        if available_main_resource is not None:
            return [available_main_resource.entity_type]

        available_resource_ets = [available_resource.entity_type
                                  for available_resource in available_resources]

        possible_main_resource_entity_types = []
        for resource_group in resource_groups:

            if available_resource_ets:
                applicable = False
                for resource_et in available_resource_ets:
                    if resource_et in resource_group.main_resources:
                        applicable = True
                        break

                    if resource_et.super_entity_type in resource_group.main_resources:
                        applicable = True
                        break

                if applicable:
                    possible_main_resource_entity_types += resource_group.main_resources

            else:
                possible_main_resource_entity_types += resource_group.main_resources

        possible_main_resource_entity_types = list(set(possible_main_resource_entity_types))

        return possible_main_resource_entity_types

    def __str__(self):
        return f"ResourceController with ID '{self.identification}'; {self._resource_model}'"


class ProcessTimeController(ProcessController):

    def __init__(self,
                 process_time_model: ProcessTimeModel,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Calculates the process time that the process needs, based on the information given by a process_execution.

        Parameters
        ----------
        process_time_model: Probability distribution of the process lead time
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._process_time_model: ProcessTimeModel = process_time_model

    @property
    def process_time_model(self):
        return self._process_time_model

    @process_time_model.setter
    def process_time_model(self, process_time_model):
        self._process_time_model = process_time_model

    def get_model(self) -> ProcessTimeModel:
        return self._process_time_model

    def copy(self):
        """Copy the object with the same identification."""
        process_time_model_copy = super(ProcessTimeController, self).copy()
        process_time_model_copy._process_time_model = process_time_model_copy._process_time_model.copy()

        return process_time_model_copy

    def get_estimated_process_lead_time(self, event_type: ProcessExecutionTypes = ProcessExecutionTypes.PLAN,
                                        process: Optional[Process] = None,
                                        parts_involved: Optional[list[tuple[Part, EntityTransformationNode]]] = None,
                                        resources_used: Optional[
                                            list[tuple[Resource, EntityTransformationNode]]] = None,
                                        resulting_quality: Optional[float] = None,
                                        main_resource: Optional[Resource] = None,
                                        origin: Optional[Resource] = None,
                                        destination: Optional[Resource] = None,
                                        order: Optional[Order] = None,
                                        executed_start_time: Optional[datetime] = None,
                                        executed_end_time: Optional[datetime] = None,
                                        source_application: Optional[str] = None,
                                        distance: Optional[float] = None) -> float:
        """
        The method is used to determine the estimated process_lead_time
        e.g. for the planned_process_execution not created at the state of calling the method.
        """

        estimated_process_lead_time = (
            self._process_time_model.get_estimated_process_lead_time(
                event_type=event_type, process=process, parts_involved=parts_involved, resources_used=resources_used,
                resulting_quality=resulting_quality, main_resource=main_resource, origin=origin,
                destination=destination,
                order=order, executed_start_time=executed_start_time, executed_end_time=executed_end_time,
                source_application=source_application,
                distance=distance))

        return estimated_process_lead_time

    def get_expected_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the expected_process_time e.g., for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time determination
        distance: used if the distance is not directly determinable because the current position of the resource
        can be different to the position in the execution

        Returns
        -------
        expected_process_time: the expected process time
        """
        expected_process_time = (
            self._process_time_model.get_expected_process_lead_time(process_execution=process_execution,
                                                                    distance=distance))

        return expected_process_time

    def get_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the process_time e.g., for the actual_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time determination
        distance: Used if the distance is not directly determinable because the current position of the resource
        can be different to the position in the execution

        Returns
        -------
        process_lead_time: the process time
        """
        process_lead_time = self._process_time_model.get_process_lead_time(process_execution=process_execution,
                                                                           distance=distance)
        return process_lead_time

    def __str__(self):
        return f"ProcessTimeController with ID '{self.identification}'; {self._process_time_model}'"


ProcessTransitionTypes = Enum('ProcessTransitionTypes',
                              'NO_TRANSITION TRANSFER TRANSPORT',
                              module='ofact.twin.state_model.processes',
                              qualname='TransitionController.Types')


def get_process_transition_type(origin: Resource, destination: Resource, class_name: str) -> tuple[str, bool]:
    """Determine the process transition type based on origin and destination"""

    if origin is not None and destination is not None:
        if origin.identification == destination.identification:
            # maybe intern
            process_transition_type = TransitionController.Types.NO_TRANSITION
            intersection = True
            return process_transition_type, intersection

        intersection = None
    else:
        intersection = True

    if intersection is None:
        intersection = _determine_intersection(origin, destination, class_name=class_name)

    if intersection:
        process_transition_type = TransitionController.Types.TRANSFER
    else:
        process_transition_type = TransitionController.Types.TRANSPORT

    return process_transition_type, intersection


def _determine_intersection(origin: Resource, destination: Resource, class_name: str):
    """Determine if the physical bodies of the origin and the destination intersect"""

    if not (origin is not None and destination is not None):
        intersection = True  # actually false
    elif origin.physical_body.position is not None and destination.physical_body.position is not None:
        intersection = origin.physical_body.check_intersection_base_areas(
            other_physical_body=destination.physical_body)
    elif origin.situated_in is None and origin.physical_body.position is None:
        # origin not determinable
        intersection = True  # assumption
    elif destination.situated_in is None and destination.physical_body.position is None:
        # origin not determinable
        intersection = True  # assumption
    elif origin.physical_body.position is None and destination.physical_body.position is None:
        intersection = origin.situated_in.physical_body.check_intersection_base_areas(
            other_physical_body=destination.situated_in.physical_body)
    elif origin.physical_body.position is None:
        intersection = origin.situated_in.physical_body.check_intersection_base_areas(
            other_physical_body=destination.physical_body)
    elif destination.physical_body.position is None:
        intersection = origin.physical_body.check_intersection_base_areas(
            other_physical_body=destination.situated_in.physical_body)
    else:
        debug_str = f"[{class_name}] Intersection: {origin} - {destination}"
        logging.debug(debug_str)
        raise Exception(debug_str)

    return intersection


def _get_entities_to_remove(entities, destination=None):
    """
    Get the entities that should be transferred and firstly removed
    That can be supports or main_entities
    """
    entities_to_transfer = []
    for entity_tuple in entities:
        if len(entity_tuple) == 2:
            if entity_tuple[1].transformation_type_main_entity():
                entities_to_transfer.append(entity_tuple[0])

    return entities_to_transfer


def get_transferred_entities(origin: Resource, destination: Resource,
                             entities: list[tuple[Entity, EntityTransformationNode]], half_transited_entities=[],
                             intersection=None, process_execution=None, transition_forced: bool = True,
                             sequence_already_ensured: bool = False) -> [list[Entity], list[Entity]]:
    """
    Transfer (resource change) entities from origin to destination

    Parameters
    ----------
    origin: the origin of the entity transfer
    destination: the destination for the entity transfer
    entities: entities that are transferred from the origin to the destination
    intersection: True if the origin and destination resources have a base area intersection
    process_execution: the process_execution responsible for the transfer
    transition_forced: means that the entities should be available in the origin.
    This is especially not the case for the data transformation. The initial origins of the entities are only assumed
    and are corrected with the first process_execution.

    Returns
    -------
    not_removed_entities: not transferred entities that are not removed
    not_added_entities: not transferred entities that are not added
    """

    # check if transfer possible - origin and destination must intersect
    if intersection is None:
        intersection = origin.physical_body.check_intersection_base_areas(
            other_physical_body=destination.physical_body)
    if not intersection:
        return entities, []

    # remove_entities from origin and add_entities to destination
    entities_to_remove = _get_entities_to_remove(entities, destination)
    if origin is not None:
        if half_transited_entities:
            entities_to_remove_half = [entity
                                       for entity in half_transited_entities
                                       if entity not in half_transited_entities]
            not_removed_entities = origin.remove_entities(entities_to_remove_half, process_execution,
                                                          sequence_already_ensured=sequence_already_ensured)
        else:
            try:
                not_removed_entities = origin.remove_entities(entities_to_remove, process_execution,
                                                              sequence_already_ensured=sequence_already_ensured)
            except:
                raise Exception(f"Transfer from origin '{origin.name}' "
                                f"in process execution '{process_execution.get_name()}' failed."
                                f"Entities to remove are {[(e.name, e.situated_in) for e in entities_to_remove]}")

    else:
        not_removed_entities = entities_to_remove

    if transition_forced:
        entities_to_add = [entity
                           for entity in entities_to_remove
                           if entity not in not_removed_entities]  # or
        # case: box content and part to stock (was part of box content)
        # (entity.situated_in is None and isinstance(entity, Part))]

    else:
        entities_to_add = entities_to_remove

    if destination is not None:
        not_added_entities = destination.add_entities(entities_to_add, process_execution=process_execution,
                                                      sequence_already_ensured=sequence_already_ensured)
    else:
        not_added_entities = []

    return not_removed_entities, not_added_entities


def _raise_transport_exception(origin, destination, transport_resource, process_execution, class_name):
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_rows', 2000)
    if origin is not None:
        print(f"[{class_name:20}] Origin: {origin.name, origin.situated_in}")
    if destination is not None:
        print(f"[{class_name:20}] Destination: {destination.name}")
        if destination.situated_in is not None:
            print(f"[{class_name:20}] Destination: {destination.situated_in.name}")
            if destination.situated_in.situated_in is not None:
                print(f"[{class_name:20}] Destination: {destination.situated_in.situated_in.name}")

    print(transport_resource.name)
    print(transport_resource.process_execution_plan._time_schedule)
    debug_str = f"[{class_name}] The transport is only possible between stationary resources \n" \
                f"PE connected ID: {process_execution.connected_process_execution.identification}"
    logging.debug(debug_str)
    raise Exception(debug_str)


def _raise_transport_exception2(transport_resource, process_execution, class_name):
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_rows', 2000)
    print(f"[{class_name:20}]", transport_resource.process_execution_plan._time_schedule)
    debug_str = f"[{class_name}] " \
                f"PE connected ID: {process_execution.connected_process_execution.identification}"
    logging.debug(debug_str)
    print(process_execution.process.external_identifications['static_model'][0])
    raise Exception(debug_str)


def transport_entities(origin: StationaryResource, destination: StationaryResource,
                       transport_resource: Resource, intersection=None, process_execution=None, transition_forced=True,
                       class_name=None, sequence_already_ensured: bool = False):
    """
    Transport (position change) entities from origin to destination

    Parameters
    ----------
    origin: the starting resource of the transport
    destination: the destination resource of the transport
    transport_resource: the resource that transports e.g. parts
    intersection: True if the origin and destination resources have a base area intersection
    process_execution: responsible for the transport
    transition_forced: means that the entities should be available in the origin
    class_name: name of the class (used for debugging)
    This is especially not the case for the data transformation. The initial origins of the entities are only assumed
    and are corrected with the first process_execution.
    """

    if not ((isinstance(origin, StationaryResource) or origin is None) and
            (isinstance(destination, StationaryResource) or destination is None)):
        if not transition_forced:
            return

        _raise_transport_exception(origin, destination, transport_resource, process_execution, class_name)

    # check if transport possible - origin and destination do not intersect
    if intersection is None:
        intersection = _determine_intersection(origin, destination, class_name)

    if not intersection:
        # transport - update position of the transport_resource if the resource has the type NonStationaryResource
        if isinstance(transport_resource, NonStationaryResource):
            if destination is not None:
                new_position = destination.get_approach_position()
                print(f"[{class_name:20}] Position from resource {transport_resource.name} in "
                      f"{process_execution.get_process_name()} changed "
                      f"from {origin.name, transport_resource.get_position()} "
                      f"to {destination.name, new_position}")
                if new_position == transport_resource.get_position() and \
                        origin.get_position() != destination.get_position():
                    _raise_transport_exception2(transport_resource, process_execution, class_name)

            else:
                new_position = (None, None)

            transport_resource.change_position(new_position=new_position, process_execution=process_execution,
                                               sequence_already_ensured=sequence_already_ensured)


class TransitionController(ProcessController):
    Types = ProcessTransitionTypes

    def __init__(self,
                 transition_model: TransitionModel,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Describes the possible Transitions from one (Stationary)Resource to another.

        Parameters
        ----------
        transition_model: Control the transition_model ...
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._transition_model: TransitionModel = transition_model

    @property
    def transition_model(self):
        return self._transition_model

    @transition_model.setter
    def transition_model(self, transition_model):
        self._transition_model = transition_model

    def get_model(self) -> TransitionModel:
        return self._transition_model

    def copy(self):
        """Copy the object with the same identification."""
        transition_model_copy = super(TransitionController, self).copy()
        transition_model_copy.transition_model = transition_model_copy.transition_model.copy()

        return transition_model_copy

    def check_resource_intern(self) -> bool:
        """
        Check if the process is performed process intern. The method checks the intersection of possible_origins and
        possible_destinations. (Assumption: if the intersection is equal to the set of possible_destinations
        (or possible_origins), the process is resource intern and therefore, e.g., a production process)

        Returns
        -------
        resource_intern: True if the process is resource intern else False
        """
        possible_origins = self._transition_model.get_possible_origins()
        possible_destinations = self._transition_model.get_possible_destinations()
        intersection_origin_destination = list(set(possible_origins) & set(possible_destinations))
        if len(intersection_origin_destination) == len(possible_origins):
            resource_intern = True
        else:
            resource_intern = False
        return resource_intern

    def check_resource_in_possible_origins(self, resource: Resource):
        """
        Check if the resource is in possible origins

        Returns
        -------
        resource_in_possible_origin: True if resource in possible_origins else False
        """
        possible_origins = self._transition_model.get_possible_origins()
        resource_in_possible_origin = any([True
                                           for possible_origin in possible_origins
                                           if possible_origin.identification == resource.identification])
        return resource_in_possible_origin

    def get_possible_origins(self) -> list[Resource]:
        """
        The method is used to determine the possible origins.

        Returns
        -------
        possible_origins: a list of possible origins (Resources)
        """
        return self._transition_model.get_possible_origins()

    def get_possible_destinations(self, origin: Optional[Resource] = None) -> list[Resource]:
        """
        The method is used to determine the possible destinations.

        Returns
        -------
        possible_destinations: a list of possible destinations (Resources)
        """
        return self._transition_model.get_possible_destinations(origin)

    def get_destination(self, process_execution: ProcessExecution):
        """Use Case not specified until now"""
        return self._transition_model.get_destination(process_execution=process_execution)

    def get_transition_type(self, origin: Optional[Resource], destination: Optional[Resource]):
        process_transition_type, intersection = get_process_transition_type(origin, destination,
                                                                            self.__class__.__name__)

        return process_transition_type

    def transit_entities(self, origin: StationaryResource, destination: StationaryResource,
                         transport_resource: Resource, parts_involved: list[tuple[Part, EntityTransformationNode]],
                         resources_used: list[tuple[Resource, EntityTransformationNode]], half_transited_entities,
                         process_execution, transition_forced=True, sequence_already_ensured: bool = False) -> \
            [list[Entity], list[Entity]]:
        """
        transit the entities from origin to destination (can be a logical transfer or spatial transport)
        the transport_resource is only used if it is of type NonStationaryResource
        """
        # print("Transition: ", transport_resource.name, process_execution.identification,
        # process_execution.process.name)
        process_transition_type, intersection = get_process_transition_type(origin, destination,
                                                                            self.__class__.__name__)
        # ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo
        if process_execution.process.external_identifications['static_model'][0][-19:] == 'loading_warehouse_p':
            process_transition_type = TransitionController.Types.TRANSFER


        if process_execution.process.external_identifications['static_model'][0] =='_main_part_transport_p':
            print('Debug _main_part_transport')
        # ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo: ToDo
        if process_transition_type == TransitionController.Types.NO_TRANSITION:
            return

        elif process_transition_type == TransitionController.Types.TRANSFER:
            entities = parts_involved + resources_used  # ToDo: neglect origin and destination
            get_transferred_entities(origin=origin, destination=destination, entities=entities,
                                     process_execution=process_execution, intersection=intersection,
                                     half_transited_entities=half_transited_entities,
                                     transition_forced=transition_forced)
        elif process_transition_type == TransitionController.Types.TRANSPORT:
            transport_entities(origin=origin, destination=destination, transport_resource=transport_resource,
                               process_execution=process_execution, intersection=intersection,
                               transition_forced=transition_forced, sequence_already_ensured=sequence_already_ensured,
                               class_name=self.__class__.__name__)
        else:
            raise Exception(f"Process transition type not supported {process_transition_type}")

    def __str__(self):
        return f"TransitionController with ID '{self.identification}'; {self._transition_model}'"


class QualityController(ProcessController):

    def __init__(self,
                 quality_model: QualityModel,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Describes how the quality values of the parts are calculated that are effected by the process. They may depend
        on the qualities of the ingoing parts and of the qualities of the used resources.

        Parameters
        ----------
        quality_model: a probability distribution that describes the quality behavior
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._quality_model: QualityModel = quality_model

    @property
    def quality_model(self):
        return self._quality_model

    @quality_model.setter
    def quality_model(self, quality_model):
        self._quality_model = quality_model

    def get_model(self) -> QualityModel:
        return self._quality_model

    def copy(self):
        """Copy the object with the same identification."""
        quality_model_copy: QualityController = super(QualityController, self).copy()
        quality_model_copy._quality_model = quality_model_copy._quality_model.copy()

        return quality_model_copy

    def get_estimated_quality(self, event_type: ProcessExecutionTypes = ProcessExecutionTypes.PLAN,
                              process: Optional[Process] = None,
                              parts_involved: Optional[list[tuple[Part, EntityTransformationNode]]] = None,
                              resources_used: Optional[list[tuple[Resource, EntityTransformationNode]]] = None,
                              resulting_quality: Optional[float] = None,
                              main_resource: Optional[Resource] = None,
                              origin: Optional[Resource] = None,
                              destination: Optional[Resource] = None,
                              order: Optional[Order] = None,
                              source_application: Optional[str] = None,
                              distance=None) -> float:
        """
        The method is used to determine the estimated quality,
        e.g., for the planned_process_execution not created at the state of calling the method.
        """

        estimated_quality = (
            self._quality_model.get_estimated_quality(event_type=event_type, process=process,
                                                      parts_involved=parts_involved,
                                                      resources_used=resources_used,
                                                      resulting_quality=resulting_quality,
                                                      main_resource=main_resource,
                                                      origin=origin, destination=destination,
                                                      order=order, source_application=source_application,
                                                      distance=distance))

        return estimated_quality

    def get_expected_quality(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the expected_process_time e.g., for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the quality determination
        distance: used if the distance is not directly determinable because the current position of the resource
        can be different to the position in the execution

        Returns
        -------
        expected_quality: the expected quality
        """
        expected_quality = self._quality_model.get_expected_quality(process_execution=process_execution,
                                                                    distance=distance)

        return expected_quality

    def get_quality(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the quality, e.g., for the actual_process_execution.

        Parameters
        ----------
        process_execution: provides the data needed for the quality determination
        distance: used if the distance is not directly determinable because the current position of the resource
        can be different to the position in the execution

        Returns
        -------
        quality: the process time
        """
        quality = self._quality_model.get_quality(process_execution=process_execution, distance=distance)
        return quality

    def __str__(self):
        return f"QualityController with ID '{self.identification}'; {self._quality_model}'"


def _get_nodes_entity_types(entity_transformation_nodes: list[EntityTransformationNode]):
    nodes_entity_types = [entity_transformation_node.entity_type
                          for entity_transformation_node in entity_transformation_nodes]
    return nodes_entity_types


def get_next_children_nodes(parent_nodes) -> list[EntityTransformationNode]:
    """
    The method is used to find the children nodes for the first transformation.
    Here it is necessary to have no main_entity in the predecessors to keep the right sequence.

    Returns
    -------
    children_nodes: children nodes without main_entity as a predecessor
    """
    possible_children_nodes = list(set([children_node
                                        for parent_node in parent_nodes
                                        for children_node in parent_node.children]))
    children_nodes = possible_children_nodes
    # ToDo: testing for all iobehaviours and transformation types required

    return children_nodes


def get_main_entity(main_entity: Part | Resource, nodes: list[EntityTransformationNode],
                    entities: list[Part | Resource]) -> [Part | Resource, EntityTransformationNode]:
    """Determine the main entity based on the entity_transformation_node and return them"""

    for node in nodes:
        if not node.transformation_type_main_entity():
            continue

        if main_entity:
            if main_entity.entity_type.check_entity_type_match(node.entity_type):
                return main_entity, node

        for entity in entities:
            if entity.entity_type.check_entity_type_match(node.entity_type):
                return entity, node

    return None, None


def _append_to_main_entity(node: EntityTransformationNode, main_entity: Part | Resource, entities: list[Part | Resource],
                           removable: bool, process_execution, sequence_already_ensured):
    """Used for sub entity and ingredient 'assembly'"""

    for entity in entities:
        if node.io_behaviour_exist():
            main_entity.add_entity(entity=entity,
                                       removable=removable,
                                       process_execution=process_execution)

            if not process_execution:
                print("Warning: No transition for sub entity or ingredient assembly")
                continue

            # change situated in attribute of the sub_entities
            entity.change_situated_in(situated_in=None,
                                      process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)

    return entities, main_entity


def get_necessary_entities_by_entity_type(entity_type: EntityType, amount: int, input_entities: list[Entity]):
    """
    Determine the entities needed based on the entity_types and the amount needed from that entity_type

    Parameters
    ----------
    entity_type:
    amount: an int that specify the number of needed parts
    input_entities: a list of entities

    Returns
    -------
    necessary_entities: a list with input parts that matches to the entity_type
    (but only as much as specified by the amount)
    """
    all_founded_entities = [entity
                            for entity in input_entities
                            if entity.entity_type.check_entity_type_match(entity_type)]
    necessary_entities = all_founded_entities[:int(amount)]
    return necessary_entities


def _create_and_destroy(parent_nodes: list[EntityTransformationNode],
                        input_parts_to_transform: list[Part],
                        transformed_parts: list[tuple[Part, EntityTransformationNode]],
                        destroyed_parts: list[tuple[Part, EntityTransformationNode]]):
    """Create and destroy entities (normally creation is in the root nodes)"""

    for parent_node in parent_nodes:
        if parent_node.io_behaviour_created():
            created_parts = _create_parts(parent_node)
            input_parts_to_transform += created_parts
            transformed_parts += [(created_part, parent_node)
                                  for created_part in created_parts]

        elif parent_node.io_behaviour_destroyed():
            parts_to_destroy = _destroy_parts(parent_node, input_parts_to_transform)
            input_parts_to_transform = [part
                                        for part in input_parts_to_transform
                                        if part not in parts_to_destroy]
            transformed_parts = [(part, node)
                                 for part, node in transformed_parts
                                 if part not in parts_to_destroy]
            destroyed_parts += [(part_to_destroy, parent_node)
                                for part_to_destroy in parts_to_destroy]

    return input_parts_to_transform, transformed_parts, destroyed_parts


def _create_parts(node: EntityTransformationNode):
    """Create parts based on the transformation node"""

    created_parts = []
    for i in range(int(node.amount)):
        created_part = Part(identification=None,
                            name=node.entity_type.name,
                            entity_type=node.entity_type,
                            situated_in=None,
                            quality=node.quality,
                            part_of=None,
                            parts=None,
                            part_removable=None)
        created_parts.append(created_part)

    return created_parts


def _destroy_parts(node, parts):
    """Get the parts to destroy based on the transformation node"""
    return get_necessary_entities_by_entity_type(node.entity_type, node.amount, parts)


def _execute_disassemble(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                         transformed_entities, class_name, sequence_already_ensured):
    sub_entities = transform_existing_disassemble(children_node, main_entity, process_execution,
                                                  sequence_already_ensured)
    input_entities_to_transform += sub_entities
    transformed_entities += [(sub_entity, children_node)
                             for sub_entity in sub_entities]
    transited_entity = None
    half_transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def _execute_unsupport(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                       transformed_entities, class_name, sequence_already_ensured):
    support, half_transited_entity = transform_existing_unsupport(children_node, main_entity, process_execution,
                                                                  class_name, sequence_already_ensured)
    transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def _inspect_quality(root_nodes, entities_to_inspect, process_execution):
    for root_node in root_nodes:
        if not root_node.transformation_type_quality_inspection():
            continue

        # does not consider the amount
        for entity in entities_to_inspect:
            if root_node.entity_type.check_entity_type_match(entity.entity_type):
                # ToDo: measurement tools should be also considered and have an impact
                entity.inspect_quality(process_execution)

    return entities_to_inspect


def _execute_main_entity(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                         transformed_entities, class_name, sequence_already_ensured):
    transform_existing_main_entity(main_entity, children_node)
    transited_entity = None
    half_transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def _execute_sub_entity(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                        transformed_entities, class_name, sequence_already_ensured):
    entities, main_entity = _transform_existing_sub_entity(main_entity, parent_node,
                                                           input_entities_to_transform,
                                                           process_execution, sequence_already_ensured)
    input_entities_to_transform = list(set(input_entities_to_transform) - set(entities))
    transited_entity = None
    half_transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def _execute_ingredient(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                        transformed_entities, class_name, sequence_already_ensured):
    entities, main_entity = _transform_existing_ingredients(main_entity, parent_node, input_entities_to_transform,
                                                       process_execution, sequence_already_ensured)
    input_entities_to_transform = list(set(input_entities_to_transform) - set(entities))
    transited_entity = None
    half_transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def _execute_support(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                     transformed_entities, class_name, sequence_already_ensured):
    support, main_entity, transited_entity = _transform_existing_support(main_entity, parent_node,
                                                                         input_entities_to_transform,
                                                                         process_execution,
                                                                         sequence_already_ensured)
    input_entities_to_transform.remove(support)
    half_transited_entity = None
    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


children_transformations = {EntityTransformationNodeTransformationTypes.DISASSEMBLE: _execute_disassemble,
                            EntityTransformationNodeTransformationTypes.UNSUPPORT: _execute_unsupport}
parent_transformations = {EntityTransformationNodeTransformationTypes.MAIN_ENTITY: _execute_main_entity,
                          EntityTransformationNodeTransformationTypes.SUB_ENTITY: _execute_sub_entity,
                          EntityTransformationNodeTransformationTypes.INGREDIENT: _execute_ingredient,
                          EntityTransformationNodeTransformationTypes.SUPPORT: _execute_support}


def _transform_with_main_entity(input_entities_to_transform, transformed_entities, parent_node, children_node,
                                main_entity, process_execution, class_name, sequence_already_ensured: bool = False):
    if children_node.compare_transformation_type_self(children_transformations):
        execution_func = children_transformations[children_node.transformation_type]
        input_entities_to_transform, main_entity, transited_entity, half_transited_entity = (
            execution_func(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                           transformed_entities, class_name, sequence_already_ensured))

    elif parent_node.compare_transformation_type_self(parent_transformations):
        execution_func = parent_transformations[parent_node.transformation_type]
        input_entities_to_transform, main_entity, transited_entity, half_transited_entity = (
            execution_func(parent_node, children_node, main_entity, process_execution, input_entities_to_transform,
                           transformed_entities, class_name, sequence_already_ensured))

    else:
        raise Exception(parent_node.transformation_type, children_node.transformation_type)

    return input_entities_to_transform, main_entity, transited_entity, half_transited_entity


def transform_existing_main_entity(main_entity, children_node):
    # no transformation needed until now - the sub_entities and the raw_material is appended in their transformation
    return main_entity


def transform_existing_disassemble(children_node: EntityTransformationNode, main_entity: Part | Resource,
                                   process_execution: ProcessExecution,
                                   sequence_already_ensured: bool = False) -> list[Part | Resource]:
    """
    Disassemble a number of entities (defined by the EntityTransformationNode) from the main_entity.

    Parameters
    ----------
    children_node: a children entity transformation_node
    main_entity: main_entity that is disassembled (described by the children nodes)
    process_execution: the process execution responsible for the disassembly
    sequence_already_ensured: says if the process execution sequence is time chronological or not

    Returns
    -------
    sub_entities: a list with main_entity and the disassembled entities
    """
    sub_entities = main_entity.get_disassembled_parts(part_entity_type=children_node.entity_type,
                                                      amount=children_node.amount,
                                                      process_execution=process_execution,
                                                      sequence_already_ensured=sequence_already_ensured)

    if not process_execution:
        print("Warning: No transition for disassembly")
        return sub_entities

    # change situated in attribute of the sub_entities
    for sub_entity in sub_entities:
        sub_entity.change_situated_in(situated_in=process_execution.destination,
                                    process_execution=process_execution,
                                    sequence_already_ensured=sequence_already_ensured)

    return sub_entities


def transform_existing_unsupport(children_node: EntityTransformationNode, main_entity, process_execution, class_name,
                                 sequence_already_ensured) -> Resource:
    support = main_entity.situated_in

    if support is not None:
        support.remove_entity(main_entity, process_execution, sequence_already_ensured)
        half_transited_entity = main_entity

    else:
        debug_str = f"[{class_name}] Main entity has no support {main_entity.external_identifications}"
        logging.debug(debug_str)
        raise Exception(debug_str)

    main_entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                   sequence_already_ensured=sequence_already_ensured)

    if not children_node.entity_type.check_entity_type_match(support.entity_type):
        if prints_visible:
            print(f"[{class_name:20}] Warning: Support {support.name} has the wrong entity_type")

    return support, half_transited_entity


def _transform_existing_blank(parent_node: EntityTransformationNode, children_node: EntityTransformationNode,
                              input_parts, process_execution, sequence_already_ensured: bool = False) -> list[Part]:
    """
    The method is used to transform a blank part. Therefore, a new part is created and
    the old one is attached as irremovable part.

    Returns
    -------
    processed_parts: the processed part
    """
    parts_to_process = get_necessary_entities_by_entity_type(parent_node.entity_type, parent_node.amount,
                                                             input_parts)
    processed_parts = []
    for part_to_process in parts_to_process:
        processed_part_type = parent_node.children[0].entity_type

        # execute the transition
        if not process_execution:
            print("Warning: No transition for blank part or ingredient assembly")

        part_to_process.change_situated_in(situated_in=None,
                                           process_execution=process_execution,
                                           sequence_already_ensured=sequence_already_ensured)

        processed_part_situated_in = part_to_process.situated_in
        if isinstance(processed_part_situated_in, Storage):
            if processed_part_situated_in.situated_in is not None:
                # remove the entity from the "old" storage to add them to the "new" with a different entity_type

                processed_part_situated_in = processed_part_situated_in.situated_in
                processed_part_situated_in.remove_entity(entity=part_to_process,
                                                         process_execution=process_execution,
                                                         sequence_already_ensured=sequence_already_ensured)
                processed_part_storages = processed_part_situated_in.get_storages(entity_type=processed_part_type)

                if processed_part_type not in processed_part_storages:
                    print("Warning: No storage available for blank transformation")
                elif not processed_part_storages[processed_part_type]:
                    print("Warning: No storage available for blank transformation")
                else:
                    # take the first storage as new storage for the processed part
                    processed_part_situated_in = processed_part_storages[processed_part_type][0]

        # create a new part
        processed_part = Part(identification=None,
                              name=processed_part_type.name,
                              entity_type=processed_part_type,
                              parts=[part_to_process],
                              situated_in=processed_part_situated_in,
                              parts_removable=[False])

        processed_parts.append(processed_part)

    return processed_parts


def _transform_existing_sub_entity(main_entity, node: EntityTransformationNode, input_parts,
                                   process_execution: ProcessExecution, sequence_already_ensured):
    """[REMOVABLE] Add one or more entities as sub-entities (equipment or sub entity) to another entity"""

    parts = get_necessary_entities_by_entity_type(node.entity_type, node.amount, input_parts)
    parts, main_entity = _append_to_main_entity(node, main_entity, parts, True, process_execution,
                                              sequence_already_ensured)

    return parts, main_entity


def _transform_existing_ingredients(main_entity, node: EntityTransformationNode, input_parts,
                                    process_execution: ProcessExecution, sequence_already_ensured):
    """[Not REMOVABLE] Add one or more entities as sub-entities (equipment or sub entity) to another entity"""

    parts = get_necessary_entities_by_entity_type(node.entity_type, node.amount, input_parts)
    parts, main_entity = _append_to_main_entity(node, main_entity, parts, False, process_execution,
                                              sequence_already_ensured)
    return parts, main_entity


def _transform_existing_support(main_entity, node: EntityTransformationNode, input_parts,
                                process_execution: ProcessExecution, sequence_already_ensured: bool = True) -> \
        [Resource, Entity, Entity]:
    """Transform an existing support means that the main_entity is added to the support if not already in the storages"""

    support = get_necessary_entities_by_entity_type(entity_type=node.entity_type, amount=1,
                                                    input_entities=input_parts)[0]

    main_entity_stored_in_support = support.check_entity_stored(main_entity)
    transited_entity = None
    if main_entity_stored_in_support:
        return support, main_entity, transited_entity

    if main_entity.situated_in:
        main_entity.situated_in.remove_entity(entity=main_entity, process_execution=process_execution,
                                            sequence_already_ensured=sequence_already_ensured)
    support.add_entity(entity=main_entity, process_execution=process_execution)
    transited_entity = main_entity

    return support, main_entity, transited_entity


def _raise_not_all_entities_available_exception(process_execution, input_parts, input_resources, class_name):
    process_name = None
    process_execution_identification = None
    if process_execution is not None:
        process_name = process_execution.get_process_name()
        process_execution_identification = process_execution.get_all_external_identifications()

    available_input_resources = [(input_resource.name, type(input_resource))
                                 for input_resource in input_resources]
    available_input_parts = [input_part.name
                             for input_part in input_parts]

    exception_str = (f"[{class_name}] Not all entities needed for the transformation of the process '{process_name}' - "
                     f"'{process_execution_identification}' are available. \n"
                     f"Available input resources are: '{available_input_resources}' \n"
                     f"Available input parts are: '{available_input_parts}'")
    logging.debug(exception_str)
    raise Exception(exception_str)


def _check_entity_type_node_match(entity_transformation_nodes: list[EntityTransformationNode],
                                  entity_type: EntityType):
    entity_match_with_entity_transformation_node = (
        bool([entity_type
              for entity_transformation_node in entity_transformation_nodes
              if entity_type.check_entity_type_match(entity_transformation_node.entity_type)]))

    return entity_match_with_entity_transformation_node


class TransformationController(ProcessController):
    io_behaviour_sequence = [EntityTransformationNodeIoBehaviours.CREATED,
                             EntityTransformationNodeIoBehaviours.DESTROYED,
                             EntityTransformationNodeIoBehaviours.EXIST]
    transformation_types_sequence = \
        [EntityTransformationNodeTransformationTypes.QUALITY_INSPECTION,
         EntityTransformationNodeTransformationTypes.MAIN_ENTITY,
         EntityTransformationNodeTransformationTypes.SUB_ENTITY,
         EntityTransformationNodeTransformationTypes.INGREDIENT,
         EntityTransformationNodeTransformationTypes.BLANK,
         EntityTransformationNodeTransformationTypes.DISASSEMBLE,
         EntityTransformationNodeTransformationTypes.SUPPORT,
         EntityTransformationNodeTransformationTypes.UNSUPPORT]

    def __init__(self,
                 transformation_model: TransformationModel,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Describes the physical transformation of parts by a process as a graph. Root nodes are needed inputs. Leaf nodes
        are created outputs.

        Parameters
        ----------
        transformation_model: describes the transformation behavior of a process executed
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._transformation_model: TransformationModel = transformation_model

    @property
    def transformation_model(self):
        return self._transformation_model

    @transformation_model.setter
    def transformation_model(self, transformation_model):
        self._transformation_model = transformation_model

    def get_model(self) -> TransformationModel:
        return self._transformation_model

    def copy(self):
        """Copy the object with the same identification."""
        transformation_model_copy = super(TransformationController, self).copy()
        transformation_model_copy.transformation_model = transformation_model_copy.transformation_model.copy()

        return transformation_model_copy

    def get_root_nodes(self, process_execution: Optional[ProcessExecution] = None) -> list[EntityTransformationNode]:
        return self._transformation_model.get_root_nodes(process_execution=process_execution)

    def get_input_entity_types_set(self) -> list[EntityType]:
        """
        Used to determine the necessary entity types. Can be used e.g., in planning of processes executions.

        Returns
        -------
        input_entity_types: a list of necessary entity types
        """
        root_nodes = self.get_root_nodes()
        root_node_entity_types = _get_nodes_entity_types(entity_transformation_nodes=root_nodes)
        input_entity_types = list(set(root_node_entity_types))
        return input_entity_types

    def get_necessary_input_entity_types(self) -> list[tuple[EntityType, int]]:
        """
        The method is used to determine the necessary entity types and their amounts.
        Can be used e.g., in planning of processes executions.

        Returns
        -------
        necessary_input_entity_types: a list of necessary entity types and their amounts.
        """
        root_nodes = self.get_root_nodes()
        necessary_input_entity_types = [(root_node.entity_type, root_node.amount)
                                        for root_node in root_nodes
                                        if not root_node.io_behaviour_created()]
        return necessary_input_entity_types

    def get_support_entity_type(self) -> Optional[EntityType]:
        """return the entity_types of transformation nodes with the transformation type SUPPORT or UNSUPPORT"""
        return self.get_entity_type_by_transformation_type_input([EntityTransformationNodeTransformationTypes.SUPPORT,
                                                                  EntityTransformationNodeTransformationTypes.UNSUPPORT])

    def get_entity_type_by_transformation_type_input(self, allowed_transformation_types: (
            list[EntityTransformationNodeTransformationTypes])) -> Optional[EntityType]:
        """return the entity_types of transformation nodes with the transformation type allowed_transformation_types"""

        root_nodes = self.get_root_nodes()
        entity_type = self._get_entity_type_of_nodes(nodes=root_nodes,
                                                     allowed_transformation_types=allowed_transformation_types)
        return entity_type

    def get_entity_type_by_transformation_type_output(self, allowed_transformation_types: (
            list[EntityTransformationNodeTransformationTypes])) -> Optional[EntityType]:
        """return the entity_types of transformation nodes with the transformation type allowed_transformation_types"""

        end_nodes = self._get_end_nodes()
        entity_type = self._get_entity_type_of_nodes(nodes=end_nodes,
                                                     allowed_transformation_types=allowed_transformation_types)
        return entity_type

    def _get_entity_type_of_nodes(self, nodes: list[EntityTransformationNode], allowed_transformation_types: (
            list[EntityTransformationNodeTransformationTypes])) -> Optional[EntityType]:

        entity_types = [node.entity_type
                        for node in nodes
                        if node.compare_transformation_type_self(allowed_transformation_types)]

        if len(entity_types) == 1:
            return entity_types[0]
        elif len(entity_types) > 1:
            raise NotImplementedError(f"[{self.__class__.__name__}]")
        else:
            return None

    def get_necessary_input_entity_types_without_support(self) -> list[tuple[EntityType, int]]:
        """
        The method is used to determine the necessary entity types and their amounts without support as
        transformation_type (for example, because the agv is not requested)

        Returns
        -------
        necessary_input_entity_types: a list of necessary entity types and their amounts.
        """
        root_nodes = self.get_root_nodes()
        necessary_input_entity_types_without_support = \
            [(root_node.entity_type, root_node.amount)
             for root_node in root_nodes
             if not (root_node.transformation_type_support() or root_node.transformation_type_un_support())]

        return necessary_input_entity_types_without_support

    def get_support_entity_types(self, support_entity_type):
        """Get support nodes that match to the support entity_type specified in the input parameters"""
        root_nodes = self.get_root_nodes()
        support_nodes = [root_node
                         for root_node in root_nodes
                         if root_node.entity_type.check_entity_type_match(support_entity_type)
                         if root_node.transformation_type_support() or root_node.transformation_type_un_support()]
        return support_nodes

    def get_possible_output_entity_types(self) -> list[tuple[EntityType, int]]:
        """
        The method is used to determine the output entity types and their amounts. Can be used e.g., in planning
        of processes executions.

        Returns
        -------
        necessary_input_entity_types: the list of possible output entities and their amounts.
        """

        parent_nodes = self.get_root_nodes()
        further_children_node_exist = True
        used_output_nodes = []
        possible_output_entity_types = []
        while further_children_node_exist:  # condition
            further_children_node_exist = False

            for parent_node in parent_nodes:
                if not parent_node.children and parent_node not in used_output_nodes:
                    possible_output_entity_types.append((parent_node.entity_type, parent_node.amount))
                    used_output_nodes.append(parent_node)

                else:
                    further_children_node_exist = True

            parent_nodes = list(set([parent_node.children[0]
                                     for parent_node in parent_nodes
                                     if parent_node.children]))

        return possible_output_entity_types

    def get_planned_transformations_with_entity_transformation_nodes(self, input_parts: list[Part],
                                                                     input_resources: list[Resource] = [],
                                                                     ignore_resources=False) \
            -> [list[tuple[Part, EntityTransformationNode]], list[tuple[Resource, EntityTransformationNode]]]:
        """
        Used to check if all parts needed for the transformation are in the input_parts and also combine them with
        TransformationNodes.

        Parameters
        ----------
        input_parts: a list of possible parts to use
        input_resources: a list of possible resources to use
        ignore_resources: if set - resource availabilities are not checked

        Returns
        -------
        available: True if all required resources and parts are available
        parts_with_entity_transformation_node: a list of (Part, TransformationNode) tuples,
        which are planned for the transformation.
        resources_with_entity_transformation_node: List of resources needed for the transformation,
        which are planned for the transformation.
        """

        # check if all needed input parts available
        input_parts_to_transform, input_resources_to_transform, available = \
            self.check_availability_of_needed_entities(input_parts=input_parts,
                                                       input_resources=input_resources,
                                                       ignore_resources=ignore_resources)

        input_entities = input_parts_to_transform + input_resources_to_transform

        # matching
        parts_with_entity_transformation_node = []
        resources_with_entity_transformation_node = []
        root_nodes = self.get_root_nodes()
        for root_node in root_nodes:
            for input_entity in input_entities:
                amount = 0
                # check part match
                if root_node.entity_usable(input_entity) and not root_node.amount_available(amount):
                    # check quality requirements
                    quality = root_node.quality_sufficient(input_entity)
                    if quality:
                        if isinstance(input_entity, Part):
                            parts_with_entity_transformation_node.append((input_entity, root_node))
                            input_parts_to_transform.remove(input_entity)
                        else:
                            resources_with_entity_transformation_node.append((input_entity, root_node))
                            input_resources_to_transform.remove(input_entity)
                        amount += 1
                # add created entities (parts)
                elif root_node.io_behaviour_created():
                    parts_with_entity_transformation_node.append((None, root_node))
                    break

        return available, parts_with_entity_transformation_node, resources_with_entity_transformation_node

    def get_entity_with_entity_transformation_node(self, entity: Entity) \
            -> list[tuple[Entity, EntityTransformationNode]]:
        """Match an entity_transformation_node to an entity"""

        entity_type = entity.entity_type
        root_nodes = self.get_root_nodes()
        possible_entity_with_entity_transformation_node = \
            [(entity, entity_transformation_node)
             for entity_transformation_node in root_nodes
             if entity_transformation_node.entity_type.check_entity_type_match(entity_type)]

        return possible_entity_with_entity_transformation_node

    def get_transformed_entities(self, process_execution: ProcessExecution, sequence_already_ensured: bool = False) -> \
            [list[tuple[Part, EntityTransformationNode]], list[tuple[Resource, EntityTransformationNode]], list, list]:
        """
        The method is used for the entity transformation.

        Transitions executed by the transformation controller based on the transformation type:
        No Action:
        - MAIN_ENTITY
        - SUPPORT
        - UNSUPPORT

        Change the Storage (managed by the main entity):
        - BLANK (change the storage in the transformation model)

        Remove situated in:
        - SUB_ENTITY
        - INGREDIENT

        Situated in destination:
        - DISASSEMBLE

        Set inspected quality:
        - QUALITY_INSPECTION

        Parameters
        ----------
        process_execution: process_execution that is responsible for the transformation
        sequence_already_ensured: used for dynamic attributes

        Returns
        -------
        necessary_input_entity_types: transformed_parts and destroyed_parts
        transited: relevant for the support and unsupport transformation type
        half_transited: relevant for the support and unsupport transformation type
        """
        class_name = self.__class__.__name__

        input_parts: list[Part] = process_execution.get_parts()
        input_resources: list[Resource] = process_execution.get_resources()

        # check if the input parts are available
        input_parts_to_transform, input_resources_to_transform, available = \
            self.check_availability_of_needed_entities(input_parts=input_parts,
                                                       input_resources=input_resources)

        if not available:
            _raise_not_all_entities_available_exception(process_execution, input_parts, input_resources,
                                                        class_name)

        # execute the transformation
        transited_entities, half_transited_entities = [], []
        transformed_entities = []  # [(Part, EntityTransformationNode)]
        destroyed_parts = []
        root_nodes = self.get_root_nodes()
        children_nodes = get_next_children_nodes(root_nodes)
        parent_nodes = self.get_sorted_nodes_io_behaviour(root_nodes)

        input_entities_to_transform = input_parts_to_transform + input_resources_to_transform
        # case: No children nodes available
        # Create and destroy parts before the assembly can start to ensure that the required parts are available
        input_entities_to_transform, transformed_entities, destroyed_parts = \
            _create_and_destroy(parent_nodes, input_entities_to_transform, transformed_entities, destroyed_parts)

        input_entities_to_transform = _inspect_quality(root_nodes, input_entities_to_transform, process_execution)

        # Parts are appended to the main_entity (SUB_ENTITY, INGREDIENT). Therefore, the main_entity is determined.
        main_entity, node = get_main_entity(None, parent_nodes, input_entities_to_transform)
        if main_entity in input_entities_to_transform:
            transformed_entities.append((main_entity, node))
            input_entities_to_transform.remove(main_entity)

        # case: Children nodes available
        iteration = 0 # reset the inspected quality based on the root nodes
        while children_nodes:
            # determine the main_entity
            main_entity, node = get_main_entity(main_entity, parent_nodes, input_entities_to_transform)
            if main_entity in input_entities_to_transform:
                input_entities_to_transform.remove(main_entity)
            main_entity_in_transformed_entities = False
            for idx, transformed_entity in enumerate(transformed_entities):
                if main_entity.identification == transformed_entity[0].identification:
                    # update the node
                    transformed_entities_lst = list(transformed_entities[idx])
                    transformed_entities_lst[1] = node
                    transformed_entities[idx] = tuple(transformed_entities_lst)

                    main_entity_in_transformed_entities = True
                    break

            if main_entity and not main_entity_in_transformed_entities:
                transformed_entities.append((main_entity, node))

            # process the existing entities
            for children_node in children_nodes:
                if not children_node.io_behaviour_exist():
                    continue

                parent_nodes = self.get_sorted_nodes_transformation_type(children_node.parents)
                for parent_node in parent_nodes:
                    if main_entity:
                        input_entities_to_transform, main_entity, transited_entity, half_transited_entity = \
                            _transform_with_main_entity(input_entities_to_transform, transformed_entities,
                                                        parent_node, children_node, main_entity,
                                                        process_execution, class_name,
                                                        sequence_already_ensured=sequence_already_ensured)
                        if transited_entity:
                            transited_entities.append(transited_entity)
                        if half_transited_entity:
                            half_transited_entities.append(half_transited_entity)

                        if iteration == 0:
                            if not parent_node.reset_inspected_quality:
                                continue

                            main_entity.reset_inspected_quality(process_execution)

                    elif parent_node.transformation_type_blank():
                        processed_parts = _transform_existing_blank(parent_node, children_node,
                                                                    input_entities_to_transform,
                                                                    process_execution,
                                                                    sequence_already_ensured=sequence_already_ensured)
                        transformed_entities += [(processed_part, children_node)
                                                 for processed_part in processed_parts]

                        if iteration == 0:
                            if not parent_node.reset_inspected_quality:
                                continue

                            for process_part in processed_parts:
                                process_part.reset_inspected_quality(process_execution)

            parent_nodes = self.get_sorted_nodes_io_behaviour(children_nodes)
            children_nodes = get_next_children_nodes(parent_nodes)

        transformed_parts = []
        transformed_resources = []
        for entity, entity_transformation_node in transformed_entities:
            if isinstance(entity, Part):
                transformed_parts.append((entity, entity_transformation_node))
            elif isinstance(entity, Resource):
                transformed_resources.append((entity, entity_transformation_node))

        return transformed_parts, transformed_resources, destroyed_parts, transited_entities, half_transited_entities

    def check_availability_of_needed_entities(self, input_parts: list[Part], input_resources: list[Resource] = [],
                                              ignore_resources: bool = False, event_type="PLAN") -> \
            [list[Part], list[Resource], bool]:
        """
        Check if all necessary input_parts are available and their quality is "okay".

        Parameters
        ----------
        input_parts: parts that are possible to use
        input_resources: resources that are possible to use
        ignore_resources: resources are faded out in the availability checking
        event_type: specify if the root_nodes are the consideration_objects or the end/ frontier nodes
        of the transformation graph

        Returns
        -------
        input_parts_to_transform: the necessary parts
        input_resources_to_transform: the necessary resources
        availability: True if possible to perform the transformation based on input_parts and resources
        """
        input_parts_to_transform = []
        input_resources_to_transform = []
        input_parts_and_resources = input_parts + input_resources
        availability = True

        if not isinstance(event_type, str):
            event_type = event_type.name

        if event_type == "PLAN":
            nodes = self.get_root_nodes()
        else:
            nodes = self._get_end_nodes()

        for node in nodes:
            available = False
            amount_input_parts = 0
            amount_input_resources = 0
            for input_part_or_resource in input_parts_and_resources:
                if input_part_or_resource in input_parts_to_transform or \
                        input_part_or_resource in input_resources_to_transform:
                    continue

                # check if the right part is available
                if not input_part_or_resource.entity_type.check_entity_type_match(node.entity_type):
                    continue

                # check if the right quality is available
                quality = node.quality_sufficient(input_part_or_resource)
                if quality:
                    if input_part_or_resource in input_parts:
                        amount_input_parts += 1
                        input_parts_to_transform.append(input_part_or_resource)
                    elif input_part_or_resource in input_resources:
                        amount_input_resources += 1
                        input_resources_to_transform.append(input_part_or_resource)

                # check if the right amount is available
                if amount_input_parts == node.amount:
                    available = True
                    break
                elif amount_input_resources == node.amount:
                    available = True
                    break

            # for the transformation "creation" is no input_part necessary
            if node.io_behaviour_created():
                pass

            # the transformation is not "creation" and no input_part available
            elif not available:
                availability = False
                if not ignore_resources:
                    continue

                # assume that resources are used as support
                if node.transformation_type_support() or node.transformation_type_un_support():
                    availability = True

        return input_parts_to_transform, input_resources_to_transform, availability

    def _get_end_nodes(self) -> list[EntityTransformationNode]:
        """Get the end nodes of the transformation graph"""

        frontier = copy(self.get_root_nodes())
        visited_states = []
        end_nodes = []
        while len(frontier) > 0:
            current_node: EntityTransformationNode = frontier.pop(0)
            if current_node not in visited_states:
                visited_states.append(current_node)
                if current_node.children:
                    frontier += current_node.children
                else:
                    end_nodes.append(current_node)

        return end_nodes

    def check_possible_input_part(self, possible_input_part) -> bool:
        """Check if the possible_input_part is usable in the transformation"""
        root_nodes = self.get_root_nodes()
        return _check_entity_type_node_match(entity_transformation_nodes=root_nodes,
                                             entity_type=possible_input_part.entity_type)

    def check_possible_output_part(self, possible_output_part) -> bool:
        """Check if the possible_output_part is usable in the transformation"""
        end_nodes = self._get_end_nodes()
        return _check_entity_type_node_match(entity_transformation_nodes=end_nodes,
                                             entity_type=possible_output_part.entity_type)

    def get_sorted_nodes_io_behaviour(self, nodes: list[EntityTransformationNode]) -> list[EntityTransformationNode]:
        """
        Used to sort the io_behaviour. First creation destroying is performed, and later other executions

        Returns
        -------
        nodes_sorted_according_sequence: a sorted list
        """
        nodes_sorted_according_sequence = [node
                                           for io_behaviour in type(self).io_behaviour_sequence
                                           for node in nodes
                                           if node.io_behaviour == io_behaviour]
        return nodes_sorted_according_sequence

    def get_sorted_nodes_transformation_type(self, nodes: list[EntityTransformationNode]) \
            -> list[EntityTransformationNode]:
        """To execute the transformation, the nodes must be sorted, e.g., the main_entity should be the first node"""

        transformation_types_sorted_according_sequence = \
            [node
             for transformation_type in type(self).transformation_types_sequence
             for node in nodes
             if node.transformation_type == transformation_type]
        return transformation_types_sorted_according_sequence

    def __str__(self):
        return f"TransformationController with ID '{self.identification}'; {self._transformation_model}'"


class Process(DigitalTwinObject):

    def __init__(self,
                 name: str,
                 lead_time_controller: ProcessTimeController,
                 transition_controller: TransitionController,
                 quality_controller: QualityController,
                 transformation_controller: TransformationController,
                 resource_controller: ResourceController,
                 group: Optional[EntityType] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        A process transforms entities (mainly parts, but also resource when they have to by maintained or converted).
        Transformations always take time, can affect space (transportation) or the physical constitution of the entity
        (e.g., assembly, packing, unpacking, transformation of physical form).

        Parameters
        ----------
        name: Name of process
        lead_time_controller: Controls the probability distribution for the time that the processes needs,
        might depend on the efficiency of the resources (lead_time, also called throughput_time or cycle_time)
        transition_controller: Controls the transition model that describes a possible spacial transition
        from one stationary resource to another
        quality_controller: Controls the probability distribution that describes the resulting quality
        of the entities
        transformation_controller: Controls the model that describes how the parts being involved in the process
        are transformed.
        The parts in the root nodes have to be available before execution, which means they have to be on/in
        the main_resource (or a resource in/on the main resource e.g. a buffer)
        resource_controller: Controls the list of resource groups (one of them be used to execute the process).
        Each group can be used alternatively, but all resources of the chosen groups have to be used imperative
        group: describes the type of the processes
        (used e.g. in the kpi calculation to differentiate the calculation method)
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name

        self._lead_time_controller: ProcessTimeController = lead_time_controller
        self._quality_controller: QualityController = quality_controller
        self._resource_controller: ResourceController = resource_controller
        self._transition_controller: TransitionController = transition_controller
        self._transformation_controller: TransformationController = transformation_controller

        # self.order_controller: = order_controller

        self.group: Optional[EntityType] = group  # ToDo: decide if the usage is necessary ...

    # getter and setter needed for the Excel instantiation if the attributes are private ...

    @property
    def quality_controller(self):
        return self._quality_controller

    @quality_controller.setter
    def quality_controller(self, quality_controller):
        self._quality_controller = quality_controller

    @property
    def lead_time_controller(self):
        return self._lead_time_controller

    @lead_time_controller.setter
    def lead_time_controller(self, lead_time_controller):
        self._lead_time_controller = lead_time_controller

    @property
    def resource_controller(self):
        return self._resource_controller

    @resource_controller.setter
    def resource_controller(self, resource_controller):
        self._resource_controller = resource_controller

    @property
    def transition_controller(self):
        return self._transition_controller

    @transition_controller.setter
    def transition_controller(self, transition_controller):
        self._transition_controller = transition_controller

    @property
    def transformation_controller(self):
        return self._transformation_controller

    @transformation_controller.setter
    def transformation_controller(self, transformation_controller):
        self._transformation_controller = transformation_controller

    def copy(self):
        process_copy = super(Process, self).copy()
        process_copy._lead_time_controller = self._lead_time_controller.copy()
        process_copy._quality_controller = self._quality_controller.copy()
        process_copy._resource_controller = self._resource_controller.copy()
        process_copy._transformation_controller = self._transformation_controller.copy()
        process_copy._transition_controller = self._transition_controller.copy()

        return process_copy

    def get_all_controllers(self) -> (
            list[(ProcessTimeController, QualityController, ResourceController, TransitionController,
                  TransitionController)]):
        models = [self._lead_time_controller,
                  self._quality_controller,
                  self._resource_controller,
                  self._transition_controller,
                  self._transformation_controller]
        return models

    def get_process_time_model(self) -> ProcessTimeModel:
        return self._lead_time_controller.process_time_model

    def get_quality_model(self) -> QualityModel:
        return self._quality_controller.quality_model

    def get_resource_model(self) -> ResourceModel:
        return self._resource_controller.resource_model

    def get_transition_model(self) -> TransitionModel:
        return self._transition_controller.transition_model

    def get_transformation_model(self) -> TransformationModel:
        return self._transformation_controller.transformation_model

    # # # # LEAD TIME controller based methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_estimated_process_lead_time(self, event_type: ProcessExecutionTypes = ProcessExecutionTypes.PLAN,
                                        process: Optional[Process] = None,
                                        parts_involved: Optional[list[tuple[Part, EntityTransformationNode]]] = None,
                                        resources_used: Optional[
                                            list[tuple[Resource, EntityTransformationNode]]] = None,
                                        resulting_quality: Optional[float] = None,
                                        main_resource: Optional[Resource] = None,
                                        origin: Optional[Resource] = None, destination: Optional[Resource] = None,
                                        order: Optional[Order] = None, source_application: Optional[str] = None,
                                        executed_start_time: Optional[datetime] = None,
                                        executed_end_time: Optional[datetime] = None,
                                        distance=None) -> float:
        """see description the lead_time_controller"""
        if process is None:
            process = self
        elif process != self:
            raise Exception(f"The process differs from the process called {self.name} - {process.name}.")
        if (isinstance(origin, NonStationaryResource) or
                isinstance(destination, NonStationaryResource)):
            distance = 0
        estimated_process_lead_time = (
            self._lead_time_controller.get_estimated_process_lead_time(
                event_type=event_type, process=process, parts_involved=parts_involved, resources_used=resources_used,
                resulting_quality=resulting_quality, main_resource=main_resource, origin=origin,
                destination=destination, order=order, source_application=source_application,
                executed_start_time=executed_start_time, executed_end_time=executed_end_time, distance=distance))

        return estimated_process_lead_time

    def get_expected_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """see description the lead_time_controller"""

        expected_process_lead_time = (
            self._lead_time_controller.get_expected_process_lead_time(process_execution=process_execution,
                                                                      distance=distance))

        return expected_process_lead_time

    def get_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """see description the lead_time_controller"""

        process_lead_time = self._lead_time_controller.get_process_lead_time(process_execution=process_execution,
                                                                             distance=distance)

        return process_lead_time

    # # # # RESOURCE controller based methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def check_ability_to_perform_process_as_resource(self, resource, already_assigned_resources: list[Resource] = []) \
            -> bool:
        """
        Check the ability of the resource to perform the process

        Parameters
        ----------
        resource: a resource
        already_assigned_resources: resources that are already used for the process (not needed anymore)

        Returns
        -------
        ability_to_perform_process_as_resource: True if the resource can perform the process, else False
        """
        return self._resource_controller.check_ability_to_perform_process_as_resource(
            resource=resource,
            already_assigned_resources=already_assigned_resources)

    def check_ability_to_perform_process_as_main_resource(self, resource) -> bool:
        """
        Check the ability of the resource to perform the process as main_resource

        Parameters
        ----------
        resource: a resource

        Returns
        -------
        ability_to_perform_process_as_main_resource: True if the resource can perform the process as main_resource,
        else False.
        """
        return self._resource_controller.check_ability_to_perform_process_as_main_resource(resource=resource)

    def get_resource_groups(self, process_execution: ProcessExecution = None):
        """Return all resource groups associated with the process_execution"""
        return self._resource_controller.get_resource_groups(process_execution)

    def get_possible_resource_groups(self, resources, main_resource):
        """Get possible resource models with the resources and main_resource given."""
        return self._resource_controller.get_possible_resource_groups(resources=resources, main_resource=main_resource)

    def get_resource_groups_for_main_resource(self, main_resource) -> list[ResourceGroup]:
        """Find all resource_models that can be processed by the main_resource (input_parameter) as main_resource"""
        return self._resource_controller.get_resource_groups_for_main_resource(main_resource=main_resource)

    def get_usable_resources_for_process(self, available_resources) -> list[tuple[Resource, EntityTransformationNode]]:
        """Used to determine which (available) resources (for example, organized in processes before)
                can be used in this resource_models of the process"""
        return self._resource_controller.get_usable_resources_for_process(available_resources=available_resources)

    def check_resources_build_resource_group(self, available_resources: list[Resource],
                                             available_main_resource: Resource):
        """Check if the input resources can complete at most one resource_model/-group"""
        return self._resource_controller.check_resources_build_resource_group(
            available_resources=available_resources, available_main_resource=available_main_resource)

    def get_all_entity_types_required(self, available_resources=[]):
        """Return possible entity types required for the process"""
        resource_entity_types = \
            self._resource_controller.get_possible_resource_entity_types(available_resources=available_resources)
        transformation_entity_types = \
            self._transformation_controller.get_input_entity_types_set()

        return list(set(resource_entity_types + transformation_entity_types))

    def get_possible_resource_entity_types(self, available_resources=[]):
        """Return possible entity types for resources needed for the process"""
        return self._resource_controller.get_possible_resource_entity_types(available_resources=available_resources)

    def get_possible_main_resource_entity_types(self, available_resources=[], available_main_resource=None) \
            -> list[EntityType]:
        """Return the possible main resources entity types based on the given resources"""
        return self._resource_controller.get_possible_main_resource_entity_types(
            available_resources=available_resources,
            available_main_resource=available_main_resource)

    # # # # TRANSITION controller based methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_possible_origins(self):
        return self._transition_controller.get_possible_origins()

    def get_possible_destinations(self):
        return self._transition_controller.get_possible_destinations()

    def get_transition_type(self, origin: Optional[Resource], destination: Optional[Resource]):
        return self._transition_controller.get_transition_type(origin, destination)

    def check_resource_intern(self) -> bool:
        """Requesting if the process is performed resource intern. Further information in the transition_model method"""
        return self._transition_controller.check_resource_intern()

    def check_resource_in_possible_origins(self, resource):
        """Requesting if the resource is in possible origins. Further information in the transition_model method"""
        return self._transition_controller.check_resource_in_possible_origins(resource)

    # # # # TRANSFORMATION controller based methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_necessary_input_part_entity_types(self) -> list[(EntityType, int)]:
        """Determine the required part input entity_types to execute the process"""
        all_necessary_entity_types: list[(EntityType, int)] = self.get_necessary_input_entity_types()
        resources_entity_types = self.get_possible_resource_entity_types()
        resources_entity_types += [entity_type.super_entity_type
                                   for entity_type in resources_entity_types
                                   if entity_type.super_entity_type is not None]
        part_entity_types = [(necessary_entity_type, amount)
                             for necessary_entity_type, amount in all_necessary_entity_types
                             if necessary_entity_type not in resources_entity_types]
        return part_entity_types

    def get_possible_output_entity_types(self) -> list[tuple[EntityType, int]]:
        """Determine the possible output entity_type when performing the process"""
        return self._transformation_controller.get_possible_output_entity_types()

    def get_necessary_input_entity_types(self) -> list[tuple[EntityType, int]]:
        """Determine the required input entity_types to execute the process"""
        return self._transformation_controller.get_necessary_input_entity_types()

    def get_necessary_input_entity_types_without_support(self) -> list[tuple[EntityType, int]]:
        """Determine the required input entity_types to execute the process without the consideration of
        the input resources - SUPPORT"""
        return self._transformation_controller.get_necessary_input_entity_types_without_support()

    def get_input_entity_types_set(self) -> list[EntityType]:
        """Determine the needed entity_types but not their amounts"""
        return self._transformation_controller.get_input_entity_types_set()

    def get_necessary_input_amount_of_parts(self) -> int:
        """Determine the required input number of parts to execute the process"""
        amount_per_input_part = \
            [amount for entity_type, amount in self._transformation_controller.get_necessary_input_entity_types()]
        necessary_input_amount_of_parts = sum(amount_per_input_part)
        return necessary_input_amount_of_parts

    def get_support_entity_type(self) -> Optional[EntityType]:
        return self._transformation_controller.get_support_entity_type()

    def get_main_entity_entity_type(self) -> Optional[EntityType]:
        return self.get_entity_type_by_transformation_type_input(
            [EntityTransformationNodeTransformationTypes.MAIN_ENTITY])

    def get_entity_type_by_transformation_type_input(self, allowed_transformation_types: (
            list[EntityTransformationNodeTransformationTypes])) -> Optional[EntityType]:
        return self._transformation_controller.get_entity_type_by_transformation_type_input(
            allowed_transformation_types)

    def get_entity_type_by_transformation_type_output(self, allowed_transformation_types: (
            list[EntityTransformationNodeTransformationTypes])) -> Optional[EntityType]:
        return self._transformation_controller.get_entity_type_by_transformation_type_output(
            allowed_transformation_types)

    def get_support_entity_types(self, support_entity_type):
        """Get support nodes from the transformation model"""
        support_nodes = self._transformation_controller.get_support_entity_types(support_entity_type)
        return support_nodes

    def get_planned_transformations_with_entity_transformation_nodes(self, input_parts: list[Part],
                                                                     input_resources: list[Resource] = [],
                                                                     ignore_resources=False) \
            -> [bool, list[tuple[Part, EntityTransformationNode]], list[Resource]]:
        """Determine the part transformation nodes for the input_parts, needed for the process_execution"""
        planned_transformations_with_entity_transformation_nodes = (
            self._transformation_controller.get_planned_transformations_with_entity_transformation_nodes(
                input_parts, input_resources, ignore_resources))

        return planned_transformations_with_entity_transformation_nodes

    def get_entity_with_entity_transformation_node(self, entity: Union[Resource, Part]) \
            -> list[Union[tuple[Union[Resource, Part], EntityTransformationNode], tuple[Entity,]]]:
        """Specify the entity_transformation_nodes for a resource or a part
        Note: Can return more than one entity_transformation node for the same resource/ part"""

        entity_with_transformation_type = (
            self._transformation_controller.get_entity_with_entity_transformation_node(entity=entity))
        if not entity_with_transformation_type:
            entity_with_transformation_type = [(entity,)]

        return entity_with_transformation_type

    def choose_needed_parts_involved(self, available_parts: list[Part]):
        """Choose from a list of parts only the needed parts for the process"""
        needed_parts_involved = [(part,) for part in available_parts
                                 if self.check_possible_input_part(part)]
        return needed_parts_involved

    def check_possible_input_part(self, possible_input_part) -> bool:
        """Check if the possible_input_part is usable in the transformation"""
        return self._transformation_controller.check_possible_input_part(possible_input_part)

    def check_possible_output_part(self, possible_output_part) -> bool:
        """Check if the possible_output_part is usable in the transformation"""
        return self._transformation_controller.check_possible_output_part(possible_output_part)

    def get_quality_transformed_entities(self, process_execution: ProcessExecution, distance=None,
                                         sequence_already_ensured: bool = False):
        """Transform the entities based on the quality of the quality_model and the transformed_parts and
        resources from the transformation_model"""
        quality = self._quality_controller.get_quality(process_execution=process_execution, distance=distance)

        transformed_parts, transformed_resources, destroyed_parts, transited_entities, half_transited_entities = (
            self.get_transformed_entities(process_execution=process_execution,
                                          sequence_already_ensured=sequence_already_ensured))

        # quality_transformation
        for idx, part_with_entity_trans_node in enumerate(transformed_parts):
            transformed_parts[idx][0].change_quality(quality, process_execution,
                                                     sequence_already_ensured=sequence_already_ensured)
        for idx, resource_with_entity_trans_node in enumerate(transformed_resources):
            transformed_resources[idx][0].change_quality(quality, process_execution,
                                                         sequence_already_ensured=sequence_already_ensured)

        return (transformed_parts, transformed_resources, destroyed_parts, transited_entities, half_transited_entities,
                quality)

    def get_transformed_entities(self, process_execution: ProcessExecution, sequence_already_ensured: bool = False):
        return self._transformation_controller.get_transformed_entities(
            process_execution=process_execution, sequence_already_ensured=sequence_already_ensured)

    def get_entities_needed(self) -> list[EntityType | PartType]:
        input_part_entity_types: list[EntityType | PartType] = \
            [entity_type
             for entity_type, amount in self.get_necessary_input_part_entity_types()
             for i in range(int(amount))]
        resource_groups: list[ResourceGroup] = self.get_resource_groups()
        input_entity_types: list[EntityType | PartType] = \
            [resource_group.resources + input_part_entity_types
             for resource_group in resource_groups]

        return input_entity_types

    def completely_filled(self):
        not_completely_filled_attributes = []
        if not isinstance(self.name, str):
            not_completely_filled_attributes.append("name")
        if not isinstance(self._lead_time_controller, ProcessTimeController):
            not_completely_filled_attributes.append("lead_time_controller")
        if not isinstance(self._quality_controller, QualityController):
            not_completely_filled_attributes.append("quality_controller")
        if not isinstance(self._transition_controller, TransitionController):
            not_completely_filled_attributes.append("transition_controller")
        if not isinstance(self._resource_controller, ResourceController):
            not_completely_filled_attributes.append("resource_controller")
        if not isinstance(self.transformation_controller, TransformationController):
            not_completely_filled_attributes.append("transformation_controller")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True

        return completely_filled, not_completely_filled_attributes

    def __str__(self):
        return (f"Process with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.lead_time_controller}', '{self.transition_controller}', "
                f"'{self.quality_controller}', '{self.transformation_controller}', '{self.resource_controller}', "
                f"'{self.group}'")


class ValueAddedProcess(Process):

    def __init__(self,
                 name: str,
                 lead_time_controller: ProcessTimeController,
                 transition_controller: TransitionController,
                 quality_controller: QualityController,
                 transformation_controller: TransformationController,
                 resource_controller: ResourceController,
                 feature: Feature,
                 predecessors: list[tuple[ValueAddedProcess]],
                 successors: list[ValueAddedProcess],
                 group: Optional[EntityType] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        A value-added process completes a feature of a product according to the order. They can only be executed
        in a certain order given by the successor relationship (the sequence is arbitrary unless a successor is defined,
        the successor can only be executed after the predecessor)

        Parameters
        ----------
        feature: The feature that is completed by this process
        predecessors: list of value-added processes that must be executed (at most one of each tuple)
        before the current process can start.
        E.g. [(ProductionProcessI, ProductionProcessII), (ProductionProcessII, ProductionProcessIII)] and for example,
        ProductionProcessI and ProductionProcessIII can meet the "process"-preconditions for the ValueAddedProcess
        successors: list of value-added processes that only can be executed after the execution of the current process.
        """
        super().__init__(identification=identification, name=name, lead_time_controller=lead_time_controller,
                         transition_controller=transition_controller, quality_controller=quality_controller,
                         transformation_controller=transformation_controller, resource_controller=resource_controller,
                         group=group, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.feature: Feature = feature
        self.predecessors: list[tuple[ValueAddedProcess]] = predecessors
        self.successors: list[ValueAddedProcess] = successors

    def copy(self):
        value_added_process_copy = super(Process, self).copy()
        value_added_process_copy.predecessors = value_added_process_copy.predecessors.copy()
        value_added_process_copy.successors = value_added_process_copy.successors.copy()

        return value_added_process_copy

    def add_successor(self, successor):
        """Append successor value_added_process"""
        self.successors.append(successor)

    def possible_to_execute(self, processes_executed: list[Union[Process, ValueAddedProcess]],
                            processes_requested: list[Union[Process, ValueAddedProcess]]) -> bool:
        """
        States if the process (self) could be executed based on the priority chart given by the predecessors and
        successors.

        Parameters
        ----------
        processes_executed: a list of processes that are already executed for the order
        processes_requested: a list of processes not executed until now

        Returns
        -------
        possibility_to_execute: a bool that states if the predecessors of the process defined as input are already met.
        """
        processes_executed_set = set(processes_executed)
        for process_tuple in self.predecessors:
            if not process_tuple:  # no process to choose
                continue

            predecessor_process_fulfilled = set(process_tuple).intersection(processes_executed_set)

            if not predecessor_process_fulfilled:
                predecessor_process_required = set(process_tuple).intersection(set(processes_requested))
                if predecessor_process_required:
                    return False

        return True

    def get_bill_of_material(self) -> dict[EntityType:float]:
        """
        Determines the bill of material on the process level

        Returns
        -------
        a dict with needed entity_types as key and the needed amount of that respective entity_type as value
        """
        needed_entity_types = self._transformation_controller.get_necessary_input_entity_types_without_support()
        bill_of_material = {}
        for needed_entity_type, needed_amount in needed_entity_types:
            if needed_entity_type in bill_of_material:
                bill_of_material[needed_entity_type] += 1
            else:
                bill_of_material[needed_entity_type] = needed_amount

        return bill_of_material

    def completely_filled(self):
        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if self.feature is None:
            not_completely_filled_attributes.append("feature")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True

        return completely_filled, not_completely_filled_attributes

    def __str__(self):
        return (f"ValueAddedProcess with ID '{self.identification}' and name '{self.name}';  "
                f"'{self.lead_time_controller}', '{self.transition_controller}', "
                f"'{self.quality_controller}', '{self.transformation_controller}', '{self.resource_controller}', "
                f"'{self.group}', '{self.feature.name}', '{self.predecessors}', '{self.successors}'")


def _delete_doubles(lst_of_tuples):
    lst_with_entities = [tuple_[0] for tuple_ in lst_of_tuples]

    if not len(list(set(lst_with_entities))) < len(lst_with_entities):
        return lst_of_tuples

    lst_without_doubles = []
    for entity in list(set(lst_with_entities)):
        for tuple_ in lst_of_tuples:
            if entity.identification == tuple_[0].identification:
                lst_without_doubles.append(tuple_)
                break

    return lst_without_doubles


def _get_resource_or_part_tuple(resource_or_part_tuple, process_execution, class_name):
    """Check the parts_involved or resources_used list according the right format"""

    if not isinstance(resource_or_part_tuple, tuple):
        if not isinstance(resource_or_part_tuple, list):  # create a tuple with the entity as first element
            resource_or_part_tuple = (resource_or_part_tuple,)

        else:
            debug_str = f"[{class_name}] " \
                        f"The object '{resource_or_part_tuple}' is not of the type 'Entity' or 'tuple[Entity]'. \n" \
                        f"Information according the process_execution, the entity is used in: " \
                        f"'{process_execution.process}', '{process_execution.get_all_external_identifications()}' " \
                        f"(Process Name, External identifications)"
            logging.debug(debug_str)
            raise Exception(debug_str)

    return resource_or_part_tuple


class ProcessExecution(DigitalTwinObject):
    # Todo: if time, parts or resources that were planned have to be used in Execution or can be changed has to be
    #  explicitly said here. an attribute will be set by execution that re-planning is necessary @Christian

    EventTypes = ProcessExecutionTypes

    def __init__(self,
                 event_type: EventTypes,
                 process: Process,
                 executed_start_time: Optional[Union[int, datetime]] = None,
                 executed_end_time: Optional[Union[int, datetime]] = None,
                 parts_involved: list[Union[tuple[Part, EntityTransformationNode], tuple[Part,]]] = None,
                 resources_used: list[Union[tuple[Resource, EntityTransformationNode], tuple[Resource,]]] = None,
                 main_resource: Optional[Resource] = None,
                 origin: Optional[Resource] = None,
                 destination: Optional[Resource] = None,
                 resulting_quality: Optional[float] = None,
                 order: Optional[Order] = None,
                 source_application: Optional[str] = None,
                 connected_process_execution: Optional[ProcessExecution] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None,
                 etn_specification: bool = True):
        """
        Process executions describe the dynamics of the value creation. They can be used for planning the future or
        describing the past. It can be created by a Simulation run or the physcial world.

        Parameters
        ----------
        event_type: ACTUAL if it has already occurred (also in Simulation) or PLAN if it is juts a plan
        - PLAN: specify that the process_execution is just a plan
        - ACTUAL: specify that the process_execution is already executed or at least started
        process: the process that is started or ended
        executed_start_time: int in seconds that describe the time of the actual start
        executed_end_time: int in seconds that describe the time of the actual end
        parts_involved: list of tuples of concrete parts (can also be resources, like boxes or
        transport vehicles that transport parts) involved and the corresponding EntityTransformation that
        describes how they were involved (must not be set if in planning)
        resources_used: list of resources used to execute the process
        main_resource: the main resource where the process takes place
        origin: resource (stationary if transport_process) where the process started
        destination: resource (stationary if transport_process) where the process ended
        resulting_quality: Quality value that resulted for the main entity
        order: the order for which the process_execution is performed for
        source_application: Where the ProcessExecution comes from (e.g. "Simulation scenario 123", "Shop floor")
        connected_process_execution: ProcessExecution that is either the PLAN or ACTUAL occurrence of this execution
        etn_specification: If true, the entity_transformation_nodes associated with the entities_used
        (part/ resource) are checked and specified within the method (if not available)
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._event_type: ProcessExecutionTypes = event_type

        self._executed_start_time: datetime = executed_start_time
        self._executed_end_time: datetime = executed_end_time

        self.process: Process = process

        if etn_specification:
            parts_involved = self._specify_entity_transformation_nodes(parts_involved)
        self._parts_involved: list[Union[tuple[Part, EntityTransformationNode], tuple[Part,]]] = parts_involved

        # self.resource_model: Optional[ResourceGroup] = resource_group ToDo: maybe add the actual chosen resource model
        if etn_specification:
            resources_used = self._specify_entity_transformation_nodes(resources_used)
        self._resources_used: list[Union[tuple[Resource, EntityTransformationNode], tuple[Resource,]]] = resources_used
        # self.main_entity: Entity = main_entity
        self._main_resource: Resource = main_resource

        self._resulting_quality: float = resulting_quality

        self._origin: Optional[Resource] = origin
        self._destination: Optional[Resource] = destination

        self._order: Optional[Order] = order

        self.source_application: Optional[str] = source_application
        self._connected_process_execution: Optional[ProcessExecution] = connected_process_execution

    def __str__(self):
        process_name = self.get_process_name()
        resource_names = self.get_resource_names()
        part_names = self.get_part_names()
        main_resource_name = self.get_main_resource_name()
        origin_name = self.get_origin_name()
        destination_name = self.get_destination_name()
        connected_process_execution_identification = self.get_connected_process_execution_identification()
        return (f"ProcessExecution with ID '{self.identification}' executed the process '{process_name}' "
                f"of the type '{self._event_type}' from '{self._executed_start_time}' to '{self._executed_end_time}'; "
                f"'{part_names}', '{resource_names}', '{main_resource_name}', "
                f"'{self._resulting_quality}', '{origin_name}', '{destination_name}', '{self._order}', "
                f"'{self.source_application}', '{connected_process_execution_identification}'")

    def copy(self):
        process_execution_copy = super(ProcessExecution, self).copy()
        process_execution_copy.parts_involved = process_execution_copy.parts_involved.copy()
        process_execution_copy.resources_used = process_execution_copy.resources_used.copy()

        return process_execution_copy

    @property
    def event_type(self):
        return self._event_type

    @property
    def executed_start_time(self):
        return self._executed_start_time

    @property
    def executed_end_time(self):
        return self._executed_end_time

    @property
    def parts_involved(self) -> list[tuple[Part, EntityTransformationNode]]:
        return self._parts_involved

    @property
    def resulting_quality(self):
        return self._resulting_quality

    @property
    def resources_used(self):
        return self._resources_used

    @property
    def main_resource(self):
        return self._main_resource

    @property
    def origin(self):
        return self._origin

    @property
    def destination(self):
        return self._destination

    @property
    def order(self):
        return self._order

    @property
    def connected_process_execution(self):
        return self._connected_process_execution

    @event_type.setter
    def event_type(self, event_type):
        self._event_type = event_type

    @executed_start_time.setter
    def executed_start_time(self, executed_start_time):
        if executed_start_time is not None:
            self._executed_start_time = convert_to_datetime(executed_start_time)
        else:
            self._executed_start_time = None

    @executed_end_time.setter
    def executed_end_time(self, executed_end_time):
        if executed_end_time is not None:
            self._executed_end_time = convert_to_datetime(executed_end_time)
        else:
            self._executed_end_time = None

    @parts_involved.setter
    def parts_involved(self, parts_involved):
        # format checking
        if not isinstance(parts_involved, list):
            raise ValueError(f"[{self.__class__.__name__}] The parts_involved value should have the type list, "
                             f"but has the type '{type(parts_involved)}'")

        # ToDo: avoid that the same object is stored twice

        if parts_involved:
            # check only the first element of the tuple
            if isinstance(parts_involved[0], tuple) and 1 <= len(parts_involved[0]) <= 2:
                parts_involved = parts_involved
            elif isinstance(parts_involved, list) and isinstance(parts_involved[0][0], Part):
                parts_involved = [(part_involved,) for part_involved in parts_involved]
            else:
                raise ValueError(f"[{self.__class__.__name__}] "
                                 f"Given parts_involved '{parts_involved}' are available in the false format")

            parts_used_clean = _delete_doubles(lst_of_tuples=parts_involved)

            if self.event_type == ProcessExecutionTypes.PLAN:
                all_parts_usable = all([self.process.check_possible_input_part(part_tuple[0])
                                        for part_tuple in parts_used_clean])
            elif self.event_type == ProcessExecutionTypes.ACTUAL:
                all_parts_usable = all([self.process.check_possible_output_part(part_tuple[0])
                                        for part_tuple in parts_used_clean
                                        if part_tuple[1].io_behaviour.name != "CREATED"])
                if all_parts_usable:
                    all_parts_usable = all([self.process.check_possible_input_part(part_tuple[0])
                                            for part_tuple in parts_used_clean
                                            if part_tuple[1].io_behaviour.name == "CREATED"])

            else:
                debug_str = f"[{self.__class__.__name__}] The updating of the parts_involved attribute failed" \
                            f"for the process: '{self.process.name}' because the event_type is not usable \n" \
                            f"'{[part_tuple[0].name for part_tuple in parts_involved]}'"
                logging.debug(debug_str)
                raise Exception(debug_str)

            if not all_parts_usable:
                parts_not_usable = [(part_tuple[0].name,
                                     self.process.check_possible_input_part(part_tuple[0]))
                                    for part_tuple in parts_used_clean
                                    if part_tuple[1].io_behaviour.name == "CREATED"]

                debug_str = f"[{self.__class__.__name__}] " \
                            f"The updating of the parts_involved attribute failed because not all parts usable " \
                            f"for the process: '{self.process.name}' \n " \
                            f"'{parts_not_usable}' (Part Name, Usability)"
                logging.debug(debug_str)
                raise Exception(debug_str)

            # update the entity_transformation_nodes for unfilled elements  # ToDo: Maybe not relevant here?
            parts_used_clean = self._specify_entity_transformation_nodes(parts_used_clean)

            self._parts_involved = parts_used_clean

        elif isinstance(parts_involved, list):
            self._parts_involved = []

    @resulting_quality.setter
    def resulting_quality(self, resulting_quality):
        if isinstance(resulting_quality, int):
            pass
        elif isinstance(resulting_quality, float):
            resulting_quality = float(resulting_quality)
        else:
            raise ValueError(f"[{self.__class__.__name__}] "
                             f"The resulting quality of a process_execution should have the type int or float, "
                             f"but has the type '{type(resulting_quality)}'.")

        if 0 <= resulting_quality <= 1:
            self._resulting_quality = resulting_quality
        else:
            raise ValueError(f"The resulting quality is out of the range 0 and 1 with the value '{resulting_quality}'.")

    @resources_used.setter
    def resources_used(self, resources_used):
        # format checking
        if not isinstance(resources_used, list):
            raise ValueError(f"[{self.__class__.__name__}] The resources_used value should have the type list, "
                             f"but has the type '{type(resources_used)}'")

        # ToDo: avoid that the same object is stored twice

        if resources_used:
            # check only the first element of the tuple
            if isinstance(resources_used[0], tuple) and 1 <= len(resources_used[0]) <= 2:
                resources_used = resources_used
            elif isinstance(resources_used, list) and isinstance(resources_used[0][0], Resource):
                resources_used = [(resource_used,) for resource_used in resources_used]
            else:
                raise ValueError(f"[{self.__class__.__name__}] "
                                 f"Given resources_used '{resources_used}' are available in the false format")

            resources_used_clean = _delete_doubles(lst_of_tuples=resources_used)

            all_resources_usable = all([self.process.check_ability_to_perform_process_as_resource(resource_tuple[0])
                                        for resource_tuple in resources_used_clean])

            if not all_resources_usable:
                resources_not_usable = [(resource_tuple[0].name,
                                         type(resource_tuple[0]),
                                         self.process.check_ability_to_perform_process_as_resource(resource_tuple[0]))
                                        for resource_tuple in resources_used_clean]

                debug_str = f"[{self.__class__.__name__}] " \
                            f"The updating of the resources_used attribute failed because not all resources usable " \
                            f"for the process: '{self.process.name}' \n " \
                            f"'{resources_not_usable}' (Resource Name, Resource Type, Usability)"
                logging.debug(debug_str)
                raise Exception(debug_str)

            # update the entity_transformation_nodes for unfilled elements
            resources_used_with_etn = self._specify_entity_transformation_nodes(resources_used_clean)

            self._resources_used = resources_used_with_etn

        elif isinstance(resources_used, list):
            self._resources_used = []

    @main_resource.setter
    def main_resource(self, main_resource):
        if not isinstance(main_resource, Resource):
            raise ValueError(f"[{self.__class__.__name__}] The main_resource value should have the type resource, "
                             f"but has the type '{type(main_resource)}'")

        # should also be a possible candidate to be the main_resource
        if self.process.check_ability_to_perform_process_as_main_resource(resource=main_resource):
            self._main_resource = main_resource
        else:
            raise ValueError(f"[{self.__class__.__name__}] The chosen main_resource is not usable for the process")

    @origin.setter
    def origin(self, origin):
        possible_origins = self.process.get_possible_origins()
        if not isinstance(origin, Resource):
            if possible_origins:
                raise ValueError(f"[{self.__class__.__name__}] The origin value should have the type origin, "
                                 f"but has the type '{type(origin)}'")

        if origin in possible_origins or (not possible_origins and origin is None):
            self._origin = origin

        else:
            # ToDo: Short cut because the destination is set in the data integration process and is unknown before
            if possible_origins:
                if possible_origins[0].entity_type == origin.entity_type:
                    self._origin = origin
                    return

            possible_origins_str = [(possible_origin.name, type(possible_origin))
                                    for possible_origin in possible_origins]
            debug_str = f"[{self.__class__.__name__}] " \
                        f"The origin '{origin.name}' is not within the possible origins " \
                        f"of the process '{self.process.name}', '{self.external_identifications}' \n" \
                        f"Possible origins are: {possible_origins_str}"
            logging.debug(debug_str)
            raise Exception(debug_str)

    @destination.setter
    def destination(self, destination):
        possible_destinations = self.process.get_possible_destinations()
        if not isinstance(destination, Resource):
            if possible_destinations:
                process_name = None
                if self.process:
                    process_name = self.process.name

                raise ValueError(f"[{self.__class__.__name__}] The destination value should have the type origin, "
                                 f"but has the type '{type(destination)}' for process '{process_name}'")

        if destination in possible_destinations or (not possible_destinations and destination is None):
            self._destination = destination

        else:
            # ToDo: Short cut ...
            if possible_destinations:
                if possible_destinations[0].entity_type == destination.entity_type:
                    self._destination = destination
                    return

            possible_destinations_str = [(possible_destination.name, type(possible_destination))
                                         for possible_destination in possible_destinations]

            debug_str = f"[{self.__class__.__name__}] " \
                        f"The destination '{destination.name}' is not within the possible destinations " \
                        f"of the process '{self.process.name}', '{self.external_identifications}' \n" \
                        f"Possible destinations are: {possible_destinations_str}"
            logging.debug(debug_str)
            raise Exception(debug_str)

    @order.setter
    def order(self, order):
        self._order = order

    @connected_process_execution.setter
    def connected_process_execution(self, connected_process_execution):
        if isinstance(connected_process_execution, ProcessExecution):
            if connected_process_execution.event_type != self.event_type:
                self._connected_process_execution = connected_process_execution
            else:
                raise ValueError(f"[{self.__class__.__name__}] No difference between process_execution event_types")
        else:
            raise ValueError(f"[{self.__class__.__name__}] "
                             f"The process_execution to connect is not an instance of ProcessExecution")

    def get_plan_process_execution(self) -> Optional[ProcessExecution]:
        """Returns the PLAN ProcessExecution if available or None"""

        if self.check_plan():
            return self
        else:
            return self.connected_process_execution

    def get_actual_process_execution(self) -> Optional[ProcessExecution]:
        """Returns the ACTUAL ProcessExecution if available or None"""

        if self.check_actual():
            return self
        else:
            return self.connected_process_execution

    def check_plan(self):
        if self.event_type == ProcessExecutionTypes.PLAN:
            return True
        else:
            return False

    def check_actual(self):
        if self.event_type == ProcessExecutionTypes.ACTUAL:
            return True
        else:
            return False

    def get_end_time_deviation_from_plan(self) -> Optional[timedelta]:
        """Determine the end time deviation from the PLAN ProcessExecution"""

        process_execution_actual = self.get_actual_process_execution()
        if process_execution_actual is None:
            return None

        process_execution_plan = process_execution_actual.connected_process_execution

        if process_execution_plan is None:
            return None

        if process_execution_actual.executed_end_time is None:
            return None

        end_time_deviation_from_plan = (process_execution_actual.executed_end_time -
                                        process_execution_plan.executed_end_time)

        return end_time_deviation_from_plan

    def get_main_resource_name(self):
        if self._main_resource is not None:
            return self._main_resource.name
        else:
            return None

    def get_resource_names(self):
        resource_names = [resource.name for resource in self.get_resources()]
        return resource_names

    def get_part_names(self):
        part_names = [part.name for part in self.get_parts()]
        return part_names

    def get_origin_name(self):
        if self._origin is not None:
            return self._origin.name
        else:
            return None

    def get_destination_name(self):
        if self._destination is not None:
            return self._destination.name
        else:
            return None

    def get_connected_process_execution_identification(self):
        if self._connected_process_execution is not None:
            return self._connected_process_execution.identification
        else:
            return None

    def get_possible_resource_entity_types(self, available_resources=None):
        if available_resources is None:
            available_resources = self.get_resources()
            if self.main_resource is not None:
                available_resources.append(self.main_resource)
        return self.process.get_possible_resource_entity_types(available_resources=available_resources)

    def get_all_entity_types_required(self, available_resources=[]):
        """Return possible entity types required for the process"""
        resource_entity_types = \
            self.get_possible_resource_entity_types(available_resources=available_resources)
        transformation_entity_types = \
            self.process.get_input_entity_types_set()

        return list(set(resource_entity_types + transformation_entity_types))

    def get_possible_main_resource_entity_types(self, available_resources=None, available_main_resource=None):
        if available_resources is None:
            available_resources = self.get_resources()
        if available_main_resource is None:
            available_main_resource = self.main_resource

        possible_main_resource_entity_types = \
            self.process.get_possible_main_resource_entity_types(available_resources=available_resources,
                                                                 available_main_resource=available_main_resource)
        return possible_main_resource_entity_types

    def get_main_resource_from_resources(self):
        available_resources = self.get_resources()
        possible_main_resource_entity_types = self.get_possible_main_resource_entity_types()
        main_resource = None
        for resource in available_resources:
            if resource.entity_type in possible_main_resource_entity_types:
                main_resource = resource
                break
            elif resource.entity_type.super_entity_type in possible_main_resource_entity_types:
                main_resource = resource
                break

        return main_resource

    def check_resources_build_resource_group(self) -> bool:
        return self.process.check_resources_build_resource_group(available_resources=self.get_resources(),
                                                                 available_main_resource=self.main_resource)

    def check_availability_of_needed_entities(self, event_type):
        return self.process.transformation_controller.check_availability_of_needed_entities(
            input_parts=self.get_parts(),
            input_resources=self.get_resources(),
            event_type=event_type)

    def _specify_entity_transformation_nodes(self, resources_used_or_parts_involved) \
            -> list[Union[tuple[Resource, EntityTransformationNode], tuple[Resource,]]]:
        """Specify the entity_transformation_nodes of the resources used"""
        if resources_used_or_parts_involved is None:
            return []

        new_resources_used_or_parts_involved = []
        entity_transformation_nodes_used = []
        for resource_or_part_tuple in resources_used_or_parts_involved:
            resource_or_part_tuple = _get_resource_or_part_tuple(resource_or_part_tuple=resource_or_part_tuple,
                                                                 process_execution=self,
                                                                 class_name=self.__class__.__name__)

            new_resources_parts_tuples = (
                self.process.get_entity_with_entity_transformation_node(entity=resource_or_part_tuple[0]))

            if resource_or_part_tuple in new_resources_parts_tuples:
                new_resources_or_parts_tuple = resource_or_part_tuple
            else:
                new_resources_or_parts_tuple = new_resources_parts_tuples[0]

            if len(new_resources_or_parts_tuple) == 2:
                if new_resources_or_parts_tuple[1] in entity_transformation_nodes_used and \
                        new_resources_or_parts_tuple[1].amount == 1:
                    try:
                        entity_description = [(entity_tuple[0].identification, entity_tuple[0].name,
                                               entity_tuple[1].entity_type.name if len(entity_tuple) > 1 else None
                                               , entity_tuple[1].amount if len(entity_tuple) > 1 else None)
                                              for entity_tuple in resources_used_or_parts_involved]
                    except:
                        print
                    raise NotImplementedError(f"[{self.__class__.__name__}] "
                                              f"Entity Transformation can only be used one time for process "
                                              f"'{self.process.name}' with entities \n '{entity_description}' ... \n "
                                              f"{self._executed_start_time} {self._executed_end_time}")

                entity_transformation_nodes_used.append(new_resources_or_parts_tuple[1])

            new_resources_used_or_parts_involved.append(new_resources_or_parts_tuple)

        return new_resources_used_or_parts_involved

    def connect_process_executions(self, another_process_execution):
        """Connect two process_executions"""
        self.connected_process_execution = another_process_execution
        another_process_execution.connected_process_execution = self

    def get_parts(self) -> list[Part]:
        """Return parts involved"""

        if self._parts_involved is not None:
            parts_involved = self._parts_involved
        else:
            parts_involved = []
        parts = [parts_involved_tuple[0] for parts_involved_tuple in parts_involved]
        return parts

    def get_main_entity(self) -> Optional[Part, Resource]:
        """Return the entity with the transformation_type MAIN_ENTITY in the entity_transformation_nodes
        Assumption: only one main_entity in the root_nodes"""

        main_parts = [part_tuple[0]
                      for part_tuple in self.parts_involved
                      if len(part_tuple) == 2
                      if part_tuple[1].transformation_type_main_entity()
                      and not part_tuple[1].io_behaviour_created()]
        main_resources = \
            [resource_tuple[0]
             for resource_tuple in self.resources_used
             if len(resource_tuple) == 2
             if resource_tuple[1].transformation_type_main_entity()]

        main_entities = main_parts + main_resources

        if len(main_entities) == 1:
            main_entity = main_entities[0]

        elif not main_entities:
            main_entity = None

        else:
            # ToDo: for different products
            debug_str = f"[{self.__class__.__name__}] To much entities with the transformation_type 'MAIN_ENTITY'."
            logging.debug(debug_str)
            raise Exception(debug_str)

        return main_entity

    def get_part_entity_types_needed(self) -> list[tuple[EntityType, int]]:
        """Determine the part entity types needed for the process_execution/ not already organized"""
        if self.event_type != ProcessExecutionTypes.PLAN:
            debug_str = f"[{self.__class__.__name__}] " \
                        f"Only usable for process_execution object with the event_type PLAN not {self.event_type}."
            logging.debug(debug_str)
            raise Exception(debug_str)

        needed_part_entity_types: list[tuple[EntityType, int]] = self.process.get_necessary_input_part_entity_types()
        available_entity_types = [part.entity_type for part in self.get_parts()]

        # convert the list of tuples to dict for easier handling
        needed_part_entity_types_d = dict(needed_part_entity_types)

        for available_entity_type in available_entity_types:
            needed_entity_types_available = \
                [needed_entity_type for needed_entity_type in list(needed_part_entity_types_d.keys())
                 if available_entity_type.check_entity_type_match(needed_entity_type)]
            if len(needed_entity_types_available) != 1:
                continue

            if needed_entity_types_available[0]:
                needed_part_entity_types_d[needed_entity_types_available[0]] -= 1

                if needed_part_entity_types_d[needed_entity_types_available[0]] <= 0:
                    del needed_part_entity_types_d[needed_entity_types_available[0]]

        needed_part_entity_types = list(needed_part_entity_types_d.items())

        return needed_part_entity_types

    def choose_needed_parts_involved(self, available_parts: list[Part]):
        """Choose from a list of parts only the needed parts for the process_execution"""
        return self.process.choose_needed_parts_involved(available_parts)

    def get_parts_involved_without_etn(self) -> list[tuple[Part,]]:
        """Return parts involved with empty entity_transformation_node slots"""
        parts_involved = [(parts_involved_tuple[0],) for parts_involved_tuple in self.parts_involved]
        return parts_involved

    def get_necessary_input_amount_of_parts(self) -> int:
        return self.process.get_necessary_input_amount_of_parts()

    def get_support_resource(self, support_entity_type=None):
        if support_entity_type is None:
            support_entity_type = self.get_support_entity_type()

        support_resources = [resource
                             for resource in self.get_resources()
                             if support_entity_type.check_entity_type_match_lower(resource.entity_type)]

        if len(support_resources) == 1:
            return support_resources[0]
        elif len(support_resources) > 1:
            debug_str = f"[{self.__class__.__name__}] Is this case wanted?"
            logging.debug(debug_str)
            raise Exception(debug_str)

        return None

    def get_support_entity_type(self) -> Optional[EntityType]:
        return self.process.get_support_entity_type()

    def get_main_entity_entity_type(self) -> Optional[EntityType]:
        return self.process.get_main_entity_entity_type()

    def get_support(self):
        """Searching for resources that are used as support in the process_execution and return them"""
        # Note: on transport is a support a MAIN_RESOURCE
        return [(resource_tuple[0],)
                for resource_tuple in self.resources_used
                if len(resource_tuple) == 2
                if resource_tuple[1].transformation_type_support() or
                resource_tuple[1].transformation_type_main_entity()]

    def get_resources(self) -> list[Resource]:
        """Return resources used"""
        if self._resources_used is not None:
            resources_used = self._resources_used
        elif self._resources_used is None:
            resources_used = []

        resources = [resource_tuple[0] for resource_tuple in resources_used]
        return resources

    def get_possible_resource_groups(self, resources=None, main_resource=None):
        """Return possible resource models based on given resources and main_resource as well as the process given"""
        if main_resource is None:
            main_resource = self.main_resource

        if resources is None:
            resources = self.get_resources()
            # assumption: the origin and destination can decide on which resource model is chosen
            # if the resources are irrelevant they are not considered for the resource model selection
            if self.origin:
                resources_with_origin = resources + [self.origin]
                if self.process.get_possible_resource_groups(resources_with_origin, main_resource):
                    resources.append(self.origin)
            if self.destination:
                resources_with_destination = resources + [self.destination]
                if self.process.get_possible_resource_groups(resources_with_destination, main_resource):
                    resources.append(self.destination)

        possible_resource_models = self.process.get_possible_resource_groups(resources, main_resource)
        return possible_resource_models

    def get_possible_origins(self):
        """
        Get the possible origins of the process. If more than one origin is possible
        the available resources are considered.
         ToDo: maybe the resource consideration should be done by the agents
        """
        possible_origin_resources = self.process.get_possible_origins()
        if len(possible_origin_resources) == 1:
            return possible_origin_resources
        elif not possible_origin_resources:
            return []

        available_resources = self.get_resources()

        assigned_possible_origin_resources = [resource
                                              for resource in available_resources
                                              if resource in possible_origin_resources]

        return assigned_possible_origin_resources

    def get_possible_destinations(self):
        """
        Get the possible destinations of the process. If more than one destination is possible
        the available resources are considered.
        ToDo: maybe the resource consideration should be done by the agents
        """

        possible_destination_resources = self.process.get_possible_destinations()
        if len(possible_destination_resources) == 1:
            return possible_destination_resources
        elif not possible_destination_resources:
            return []

        available_resources = self.get_resources()

        assigned_possible_destination_resources = [resource
                                                   for resource in available_resources
                                                   if resource in possible_destination_resources]

        return assigned_possible_destination_resources

    def get_main_entity_types(self):
        """Get entity_types with the transformation_type MAIN_ENTITY in the EntityTransformationNode"""
        return [main_entity.entity_type
                for main_entity in self.get_main_entities()]

    def get_main_entities(self):
        """Get entities with the transformation_type MAIN_ENTITY in the EntityTransformationNode"""
        entities_participating = self.get_entities_participating()
        main_entities = [entity_tuple[0]
                         for entity_tuple in entities_participating
                         if len(entity_tuple) == 2
                         if entity_tuple[1].transformation_type_main_entity()]
        return main_entities

    def get_entities_participating(self):
        """Get the combined lists of parts_involved and resources_used"""
        return self.parts_involved + self.resources_used

    def get_entities(self):
        """Get the combined lists of parts and resources"""
        parts: list[Part] = self.get_parts()
        resources: list[Resource] = self.get_resources()
        entities = parts + resources

        return entities

    def get_max_process_time(self, distance=None, main_resource=None):
        """Calculate the maximum expected execution time for the process_execution
        based on the current information"""
        if self.origin:
            possible_origins = [self.origin]
        else:
            possible_origins = self.process.get_possible_origins()

        if self.destination:
            possible_destinations = [self.destination]
        else:
            possible_destinations = self.process.get_possible_destinations()

        if not possible_destinations and possible_origins:
            possible_destinations = [None]
        elif not possible_origins and possible_destinations:
            possible_origins = [None]

        if main_resource is None:
            main_resource = self.main_resource

        all_possible_process_times = [self.process.get_estimated_process_lead_time(origin=possible_origin,
                                                          destination=possible_destination,
                                                          main_resource=main_resource,
                                                          distance=distance)
                                      for possible_origin in possible_origins
                                      for possible_destination in possible_destinations]

        if all_possible_process_times:
            max_execution_time = max(all_possible_process_times)
        else:
            max_execution_time = self.process.get_estimated_process_lead_time(distance=distance,
                                                                              main_resource=main_resource)
        return max_execution_time

    def get_expected_process_lead_time(self, distance=None):
        """Return the expected_process_time based on the information given by the ProcessExecution"""

        if self.origin is None:
            if self.process:
                if self.process.get_possible_origins():
                    raise NotImplementedError(f"[{self.__class__.__name__}] Origin not available - "
                                              f"get_max_process_time could be chosen instead")

        elif self.destination is None:
            if self.process:
                if self.process.get_possible_destinations():
                    raise NotImplementedError(f"[{self.__class__.__name__}] Destination not available - "
                                              f"get_max_process_time could be chosen instead")

        expected_process_time = self.process.get_expected_process_lead_time(process_execution=self, distance=distance)

        return expected_process_time

    def get_distance(self):
        """Determine the distance transited in the process_execution"""

        possible_origin, possible_destination = self.get_origin_destination()

        if (isinstance(possible_origin, NonStationaryResource) and
            isinstance(possible_destination, StationaryResource)) or \
                (isinstance(possible_destination, NonStationaryResource) and
                 isinstance(possible_origin, StationaryResource)):
            distance = 0

        else:
            distance = None

        return distance

    def get_origin_destination(self):
        """Get possible origins and destinations of the resource with use of already specified resources"""

        if self.origin is not None:
            possible_origin = self.origin
        else:
            possible_origins = self.get_possible_origins()
            if not possible_origins:
                possible_origins = self.process.get_possible_origins()
            if possible_origins:
                possible_origin = possible_origins[0]
            else:
                possible_origin = None

        if self.destination is not None:
            possible_destination = self.destination
        else:
            possible_destinations = self.get_possible_destinations()
            if not possible_destinations:
                possible_destinations = self.process.get_possible_destinations()
            if possible_destinations:
                possible_destination = possible_destinations[0]
            else:
                possible_destination = None

        return possible_origin, possible_destination

    def get_process_lead_time(self, distance: Optional[float] = None, ignore_specified_times: bool = False):
        """Sampling from process_time distribution"""
        if self._executed_start_time and self._executed_end_time and not ignore_specified_times:
            process_time = (self._executed_end_time - self._executed_start_time).seconds

            return process_time

        if self.main_resource is None:
            debug_str = f"[{self.__class__.__name__}] " \
                        f"Main resource is not recorded in the attributes {self.process.name}"
            logging.debug(debug_str)
            raise Exception(debug_str)
        if self.origin is None:
            origin_needed = self._check_origin_needed()
            if origin_needed:
                debug_str = f"[{self.__class__.__name__}] " \
                            f"Origin is not recorded in the attributes {self.process.name}"
                logging.debug(debug_str)
                raise Exception(debug_str)

        if self.destination is None:
            destination_needed = self._check_destination_needed()
            if destination_needed:
                debug_str = f"[{self.__class__.__name__}] " \
                            f"Destination is not recorded in the attributes {self.process.name}"
                logging.debug(debug_str)
                raise Exception(debug_str)

        process_time = self.process.get_process_lead_time(process_execution=self,
                                                          distance=distance)

        return process_time

    def _check_origin_needed(self) -> Optional[bool]:
        """Check if the origin can be specified"""
        if not self.process:
            return None

        if self.process.get_possible_origins():
            origin_needed = True
            return origin_needed
        else:
            origin_needed = False
            return origin_needed

    def _check_destination_needed(self) -> Optional[bool]:
        """Check if the destination can be specified"""
        if not self.process:
            return

        if self.process.get_possible_destinations():
            destination_needed = True
            return destination_needed
        else:
            destination_needed = False
            return destination_needed

    def get_latest_available_time_stamp(self):
        """Return the latest available time stamp,
        respectively first the executed end time and then the executed start time"""
        if self._executed_end_time is not None:
            return self._executed_end_time

        return self.executed_start_time

    def create_actual(self, source_application, time_specification=False, enforce_time_specification=False,
                      executed_start_time=None, process_lead_time: Optional[timedelta] = None,
                      end_time: bool = True, from_plan: bool = False) -> Optional[ProcessExecution]:
        """
        Create an actual ProcessExecution object based on the planned one (self)

        Parameters
        ----------
        source_application: determines the source_application of the actual process_execution
        time_specification: a bool that specifies if the time_impact should be executed in the method
        enforce_time_specification: a bool that specifies if the time_impact is necessary,
        also if not possible to determine (imputation not possible)
        executed_start_time: used for the time_impact
        process_lead_time: the process_lead_time set from outside (used if given and not from_plan)
        end_time: used for the time_impact
        from_plan: determines if the time entries are taken from plan process_execution
        """
        if not self.event_type == ProcessExecutionTypes.PLAN:
            return None

        self._create_process_execution_match_identification()

        actual_process_execution = ProcessExecution(event_type=ProcessExecutionTypes.ACTUAL,
                                                    process=self.process,
                                                    source_application=source_application)
        actual_process_execution.event_type = ProcessExecutionTypes.ACTUAL

        actual_process_execution.connect_process_executions(self)
        actual_process_execution.order = self.order

        if time_specification:
            actual_process_execution._specify_time_impact(executed_start_time=executed_start_time,
                                                          process_lead_time=process_lead_time,
                                                          end_time=end_time,
                                                          from_plan=from_plan,
                                                          enforce_time_specification=enforce_time_specification)

        return actual_process_execution

    def _create_process_execution_match_identification(self):
        """
        The process_execution_match_identification is used to match process_executions (exclusively PLAN)
        from different digital_twin models and consists from order_match_identification and the
        process_match_id_identification
        process_execution_match_identification = '{order_match_identification}: {process_match_id_identification}'
        """

        if self.process is None:
            return
        if "match" not in self.process.external_identifications:
            return

        if not self.order:
            return

        process_match_identification = copy(self.process.external_identifications["match"][0])
        external_identifications = list(self.order.external_identifications.values())
        if external_identifications:
            order_match_identification = external_identifications[0][0]
        else:
            order_match_identification = "Order ID not provided"
        process_execution_match_identification = f"{order_match_identification}: {process_match_identification}"
        self.external_identifications["match"] = [process_execution_match_identification]

    def _specify_time_impact(self, executed_start_time: Optional[datetime],
                             process_lead_time: Optional[timedelta] = None, end_time: bool = True,
                             from_plan: bool = False, enforce_time_specification: bool = False):
        """
        Specify the start_time and if end_time is True also the end_time based on the process_time_model

        Parameters
        ----------
        executed_start_time: the start_time can be deviated from the planned executed_start_time
        process_lead_time: the process_lead_time set from outside (used if given and not from_plan)
        end_time: a bool that specifies if the end_time should be set
        from_plan: determines if the time entries are taken from plan process_execution
        enforce_time_specification: a bool that specifies if the time_impact should be executed in the method,
        if the time model is not able to deliver the time and the time is not specified in plan
        """

        if self.event_type != ProcessExecutionTypes.ACTUAL:
            debug_str = (f"[{self.__class__.__name__}] "
                         f"Impacts can only become effective from a ProcessExecution with event_type ACTUAL, "
                         f"but the event_type is '{self.event_type}'!")
            logging.debug(debug_str)
            raise Exception(debug_str)

        # handle executed_start_time
        if executed_start_time is not None:
            # adapt the start_time
            self.executed_start_time = executed_start_time
        elif from_plan and self.connected_process_execution.executed_start_time:
            self.executed_start_time = self.connected_process_execution.executed_start_time

        # handle executed_end_time
        if not end_time:
            self._executed_end_time = None
            return

        if from_plan and self.connected_process_execution.executed_end_time:
            self._executed_end_time = self.connected_process_execution.executed_end_time

        else:
            if process_lead_time is None:
                process_execution_plan = self.connected_process_execution
                process_lead_time = process_execution_plan.get_process_lead_time(ignore_specified_times=True)

            # try:
            #     process_lead_time = process_execution_plan.get_process_lead_time(ignore_specified_times=True)
            # except:
            #     if enforce_time_specification:
            #         debug_str = f"[{self.__class__.__name__}] " \
            #                     f"The time cannot be specified through the process time model given."
            #         logging.debug(debug_str)
            #         raise Exception(debug_str)
            #     else:
            #         return

            int_delta = round(process_lead_time, 0)  # ToDo: accuracy problem
            if int_delta > 500:
                print
            timedelta_process_execution = timedelta(seconds=int_delta)

            self._executed_end_time = executed_start_time + timedelta_process_execution

    def execute(self, transition_forced: bool = True, sequence_already_ensured: bool = False):
        """
        Execute the impacts of the process_execution
        The impacts differ between actual and plan process_execution
        While the plan process_execution only change dynamic attributes the actual also
        """

        if self.event_type == ProcessExecutionTypes.ACTUAL:
            self._execute_actual(transition_forced=transition_forced, sequence_already_ensured=sequence_already_ensured)
        elif self.event_type == ProcessExecutionTypes.PLAN:
            self._execute_plan(sequence_already_ensured=sequence_already_ensured)

    def _execute_actual(self, transition_forced=True, sequence_already_ensured: bool = False):
        """
        Simulate the actual impacts based on the actual process_execution, that is created based on the planned one
        """
        transited_entities, half_transited_entities = (
            self._execute_transformation_impact(sequence_already_ensured=sequence_already_ensured))
        self._execute_transition_impact(transited_entities=transited_entities,
                                        half_transited_entities=half_transited_entities,
                                        transition_forced=transition_forced,
                                        sequence_already_ensured=sequence_already_ensured)

    def _execute_plan(self, sequence_already_ensured):
        """
        Execute the impacts of the process_execution for the dynamic attributes
         ToDo: implement a general solution for all dynamic attributes
        """

        # print("PLAN Execution: ", self.main_resource.name, self.origin, self.destination, self.process.name)
        # main resource can change the position? - non_stationary
        if not isinstance(self.main_resource, NonStationaryResource):
            return None

        # Transition type: Transport
        if not (isinstance(self.origin, StationaryResource) and isinstance(self.destination, StationaryResource)):
            return None

        # Position change of the main_resource and their parts
        # print(f"{self.executed_start_time} PLAN "
        #       f"Transition of {self.main_resource.name} to {self.destination.name}")
        new_position = self.destination.get_position()
        self.main_resource: NonStationaryResource
        self.main_resource.change_position(new_position=new_position,
                                           process_execution=self,
                                           sequence_already_ensured=sequence_already_ensured)

    def _execute_transformation_impact(self, quality=True, from_plan=True, distance=None,
                                       sequence_already_ensured: bool = False):
        """
        Execute the transformation of the process

        Parameters
        ----------
        quality: True if the quality should be determined through the quality else False
        from_plan: True if the execution is based on the PLAN ProcessExecution
        """

        if from_plan:
            if self.connected_process_execution is not None:
                process_execution = self.connected_process_execution
            else:
                raise Exception(f"Connected process_execution not determined - execution cannot be performed ...")
        else:
            process_execution = self

        if not self.process:
            debug_str = f"[{self.__class__.__name__}] " \
                        f"Input parts and resources as well as the process should be specified " \
                        f"before the transformation results can be identified/ specified!" \
                        f"Input parts: {process_execution.get_parts()}" \
                        f"Input resources: {process_execution.get_resources()}" \
                        f"Process Name: {self.process}"
            logging.debug(debug_str)
            raise Exception(debug_str)

        input_resources = process_execution.get_resources().copy()

        # execute quality + transformation
        if quality:
            (parts_involved, resources_used, destroyed_parts, transited_entities, half_transited_entities,
             resulting_quality) = (
                process_execution.process.get_quality_transformed_entities(
                    process_execution=process_execution, distance=distance,
                    sequence_already_ensured=sequence_already_ensured))

            self.resulting_quality = resulting_quality

        else:
            parts_involved, resources_used, destroyed_parts, transited_entities, half_transited_entities = \
                self.process.get_transformed_entities(process_execution=process_execution,
                                                      sequence_already_ensured=sequence_already_ensured)
        self.parts_involved = parts_involved + destroyed_parts

        if from_plan:
            not_transformed_resources = list(set(input_resources) -
                                             set([resource_tuple[0] for resource_tuple in resources_used
                                                  if resource_tuple[0] in input_resources]))
            not_transformed_resources_used = [(resource,) for resource in not_transformed_resources]
            self.resources_used = resources_used + not_transformed_resources_used

        else:
            self.resources_used += resources_used

        return transited_entities, half_transited_entities

    def _execute_transition_impact(self, origin=None, destination=None, main_resource=None, from_plan=True,
                                   transited_entities=[], half_transited_entities=[], transition_forced=True,
                                   sequence_already_ensured: bool = False):
        """
        Execute the transition of the process

        Parameters
        ----------
        origin: transition origin
        destination: transition destination
        main_resource: the main_resource is also used as transport resource (only used in the transition
        if process_type is transportation and the resource type is NonStationaryResource)
        from_plan: True if the execution is based on the PLAN ProcessExecution
        """

        if from_plan and not (origin and destination and main_resource):
            if self._connected_process_execution:
                origin = self._connected_process_execution.origin
                destination = self._connected_process_execution.destination
                main_resource = self._connected_process_execution.main_resource

        if not (self._parts_involved is not None and self._resources_used is not None and main_resource):
            debug_str = f"[{self.__class__.__name__}] Parts and/ or resources and/  or main_resource not specified"
            logging.debug(debug_str)
            raise Exception(debug_str)
        if not (origin and destination):
            if not origin:
                possible_origins = self.get_possible_origins()
                if possible_origins:
                    debug_str = f"[{self.__class__.__name__}] Origin not determinable" \
                                f"Possible origins are {possible_origins}"
                    logging.debug(debug_str)
                    raise Exception(debug_str)
            if not destination:
                possible_destinations = self.get_possible_destinations()
                if possible_destinations:
                    debug_str = f"[{self.__class__.__name__}] Destination not determinable" \
                                f"Possible destinations are {possible_destinations}"
                    logging.debug(debug_str)
                    raise Exception(debug_str)

        if from_plan:
            if not self._origin:
                self.origin = origin
            if not self._destination:
                self.destination = destination
            if not self._main_resource:
                self.main_resource = main_resource

        # execute transition
        if transited_entities:
            parts_involved = [part_tuple for part_tuple in self.connected_process_execution.parts_involved
                              if part_tuple[0] not in transited_entities]
            resources_used = [resource_tuple for resource_tuple in self.connected_process_execution.resources_used
                              if resource_tuple[0] not in transited_entities]
        else:
            disassembled_parts_involved = \
                [part_tuple
                 for part_tuple in self.parts_involved
                 if len(part_tuple) == 2
                 if part_tuple[1].transformation_type_disassemble()]
            parts_involved = self.connected_process_execution.parts_involved + disassembled_parts_involved
            resources_used = self.connected_process_execution.resources_used

        self.process.transition_controller.transit_entities(
            origin=origin, destination=destination, transport_resource=self._main_resource,
            parts_involved=parts_involved, resources_used=resources_used, process_execution=self,
            half_transited_entities=half_transited_entities, transition_forced=transition_forced,
            sequence_already_ensured=sequence_already_ensured)

    def completely_filled(self) -> [bool, list[str]]:
        """Check if the process_execution is already filled completely"""
        not_completely_filled_attributes = []
        if self.identification is None:
            not_completely_filled_attributes.append("identification")
        if not (self.event_type == ProcessExecutionTypes.PLAN or self.event_type == ProcessExecutionTypes.ACTUAL):
            not_completely_filled_attributes.append("event_type")
        if not (isinstance(self._executed_start_time, int) or isinstance(self._executed_start_time, datetime)):
            not_completely_filled_attributes.append("executed_start_time")
        if not (isinstance(self._executed_end_time, int) or isinstance(self._executed_end_time, datetime)):
            not_completely_filled_attributes.append("executed_end_time")
        if not isinstance(self.process, Process):
            not_completely_filled_attributes.append("process")

        if not self.check_resources_build_resource_group():
            not_completely_filled_attributes.append("resources_used")

        parts_with_etn, resources_with_etn, all_entities_available = \
            self.check_availability_of_needed_entities(event_type=self.event_type)
        if not all_entities_available:
            parts_with_etn, resources_with_etn, all_entities_available = \
                self.check_availability_of_needed_entities(event_type=self.event_type)
            not_completely_filled_attributes.append("parts_involved")

        if isinstance(self.resulting_quality, int) or isinstance(self.resulting_quality, float):
            if not (0 <= self.resulting_quality <= 1):
                not_completely_filled_attributes.append("resulting_quality")
        else:
            not_completely_filled_attributes.append("resulting_quality")
        if not isinstance(self.main_resource, Resource):
            not_completely_filled_attributes.append("main_resource")
        if not isinstance(self.origin, Resource):
            origin_needed = self._check_origin_needed()
            if origin_needed is True or origin_needed is None:
                not_completely_filled_attributes.append("origin")
        if not isinstance(self.destination, Resource):
            # it is possible that the destination is None but also wanted (for example customer delivery)
            destination_needed = self._check_destination_needed()
            if destination_needed is True or destination_needed is None:
                not_completely_filled_attributes.append("destination")
        if not isinstance(self.source_application, str):
            not_completely_filled_attributes.append("source_application")
        if self.event_type == ProcessExecutionTypes.ACTUAL:
            if not isinstance(self.connected_process_execution, ProcessExecution):
                not_completely_filled_attributes.append("connected_process_execution")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes

    def get_process_name(self):
        process_name = None
        if self.process:
            process_name = self.process.name
        elif self._connected_process_execution:
            if self._connected_process_execution.process:
                process_name = self._connected_process_execution.process.name
        return process_name

    def get_name(self):
        """Returns a name for the process execution for debugging purposes"""
        process_name = self.get_process_name()
        process_execution_name = f"{process_name} - {self.identification}"

        return process_execution_name


def _unroll_value_added_process_list(nested_valued_added_processes) -> list[ValueAddedProcess]:
    value_added_process_list = [value_added_process
                                for nested_dict in list(nested_valued_added_processes.values())
                                for value_added_process_list in list(nested_dict.values())
                                for value_added_process in value_added_process_list]

    return value_added_process_list


class WorkOrder(DynamicDigitalTwinObject):

    @classmethod
    def convert_features_to_value_added_processes_requested(cls, features_with_value_added_processes: list[Feature],
                                                            feature_value_added_process_mapper: (
                                                                    dict[Feature, list[ValueAddedProcess]])) -> (
            dict[Feature, dict[int, list[ValueAddedProcess]]]):
        """
        Convert the features given by the Sales order to value added processes in the format required
        from the work order

        Parameters
        ----------
        features_with_value_added_processes:
        A list of the features requested coming e.g. from the (sales) order.
        feature_value_added_process_mapper:
        A mapper that maps the value added processes required to finish a feature to the features in a dict.
        The dict can be requested from the digital twin model.

        Returns
        -------
        value_added_processes_requested: value added processes mapped to the features requested
        """
        # translate the features from the order (sales area) to value_added_processes (production_logistics area)
        features_requested = {}
        for feature_requested in features_with_value_added_processes:
            if feature_requested in features_requested:
                features_requested[feature_requested] += 1
            else:
                features_requested[feature_requested] = 1

        value_added_processes_requested = {}
        for feature, amount_needed in features_requested.items():
            value_added_processes_needed: list[ValueAddedProcess] = feature_value_added_process_mapper[feature]
            value_added_processes_requested[feature] = {i: value_added_processes_needed.copy()
                                                        for i in range(int(amount_needed))}

        return value_added_processes_requested

    def __init__(self,
                 value_added_processes_completed: dict[Feature: dict[int: list[ValueAddedProcess]]],
                 value_added_processes_requested: dict[Feature: dict[int: list[ValueAddedProcess]]],
                 order: Order,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 work_calendar: Optional[WorkCalender] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The WorkOrder is created based on the sales order and represent the order in contrast to the sales order
        on the process level.

        Parameters
        ----------
        value_added_processes_completed: a dict that maps the completed value_added_process_executions to
        features of the sales order
        value_added_processes_requested: a dict that maps the requested value_added_process_executions to
        features of the sales order
        order: a sales order that determines the production order
        process_execution_plan: a work schedule that is used to plan the production order
        work_calendar: used if the process execution plan is not given as input parameter
        """
        self._value_added_processes_completed: dict[Feature: dict[int: list[ValueAddedProcess]]] = \
            value_added_processes_completed
        super().__init__(identification=identification, process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._value_added_processes_requested: dict[Feature: dict[int: list[ValueAddedProcess]]] = \
            value_added_processes_requested

        # Note: if the sales_order is created in the product_configurator mode, possible_value_added_processes should be
        # determinable without predecessor (they must have en entry point)
        # if that is not the case, it is also possible, that the entry points are not chosen/ no entry_point is given,
        # and the processes are chosen that have no selected process in the predecessors
        self._possible_value_added_processes: list[ValueAddedProcess] = \
            self._get_possible_value_added_processes_without_predecessor()
        if not self._possible_value_added_processes:
            self._possible_value_added_processes = self.get_possible_value_added_processes()
        self.order = order

        if process_execution_plan is None:
            if order.customer is not None:
                process_execution_plan_name = order.get_customer_name()
            else:
                process_execution_plan_name = None

            process_execution_plan = ProcessExecutionPlan(name=process_execution_plan_name,
                                                          work_calendar=work_calendar)
        self._process_execution_plan: ProcessExecutionPlan = process_execution_plan

        self.value_added_process_process_execution_match: dict[Feature: list[ProcessExecution]] = {}

        self.all_predecessors = True

        # self._bill_of_materials = self._create_bill_of_materials()  # time intensive, but it works

        # further possible attributes: the earliest production start and latest production start, ...

    def copy(self):
        work_order_copy: WorkOrder = super(WorkOrder, self).copy()
        work_order_copy._value_added_processes_completed = work_order_copy._value_added_processes_completed.copy()
        work_order_copy._value_added_processes_requested = work_order_copy._value_added_processes_requested.copy()
        work_order_copy.possible_value_added_processes = work_order_copy.possible_value_added_processes
        work_order_copy.process_execution_plan = work_order_copy.process_execution_plan
        work_order_copy.value_added_process_process_execution_match = \
            work_order_copy.value_added_process_process_execution_match.copy()

        return work_order_copy

    @property
    def possible_value_added_processes(self) -> list[ValueAddedProcess]:
        """Returns a list of possible value_added_processes based on the order progress"""
        return self._possible_value_added_processes

    # @property
    # def bill_of_materials(self) -> dict[EntityType: float]:
    #     """Returns the bill of material that is associated with the product of the (Sales-)Order"""
    #     return self._bill_of_materials

    @property
    def value_added_processes_requested(self):
        """Returns value_added_processes which are until now requested"""
        return self._value_added_processes_requested

    @property
    def value_added_processes_completed(self):
        """Returns value_added_processes which are until now completed"""
        return self._value_added_processes_completed

    @possible_value_added_processes.setter
    def possible_value_added_processes(self, possible_value_added_processes):
        self._possible_value_added_processes = possible_value_added_processes

    # @bill_of_materials.setter
    # def bill_of_materials(self, bill_of_materials):
    #     self._bill_of_materials = bill_of_materials

    @value_added_processes_requested.setter
    def value_added_processes_requested(self, value_added_processes_requested):
        self._value_added_processes_requested = value_added_processes_requested

    @value_added_processes_completed.setter
    def value_added_processes_completed(self, value_added_processes_completed):
        self._value_added_processes_completed = value_added_processes_completed

    ####################################################################################################################
    # #### Process Execution Plan ####
    ####################################################################################################################

    def block_period(self, start_time, end_time, blocker_name, process_execution_id: int, work_order_id: int,
                     issue_id: int = None, block_before: bool = False):
        """Block a period in the _process_execution_plan"""
        return self._process_execution_plan.block_period(start_time=start_time, end_time=end_time,
                                                         issue_id=issue_id, blocker_name=blocker_name,
                                                         process_execution_id=process_execution_id,
                                                         work_order_id=work_order_id, block_before=block_before)

    def unblock_period(self, unblocker_name, process_execution_id):
        """Unlock a period in the _process_execution_plan"""
        return self._process_execution_plan.unblock_period(unblocker_name=unblocker_name,
                                                           process_execution_id=process_execution_id)

    def update_period(self, start_time, end_time, process_execution_id: int):
        """Update a period, respectively, their start and end time"""
        return self._process_execution_plan.update_period(start_time=start_time, end_time=end_time,
                                                          process_execution_id=process_execution_id)

    def update_period_by_actual(self, start_time, end_time, process_execution_id: int, plan_process_execution_id: int):
        self._process_execution_plan.update_period_by_actual(start_time=start_time, end_time=end_time,
                                                             process_execution_id=process_execution_id,
                                                             plan_process_execution_id=plan_process_execution_id)

    def get_next_possible_period(self, period_length: timedelta, start_time=None, issue_id=None,
                                 last_element: bool = False):
        """Get the next possible period from the _process_execution_plan"""
        return self._process_execution_plan.get_next_possible_period(period_length=period_length,
                                                                     start_time=start_time,
                                                                     issue_id=issue_id,
                                                                     last_element=last_element)

    def get_free_periods_calendar_extract(self, start_time=None, end_time=None, issue_id=None,
                                          time_slot_duration: Optional[pd.Timedelta] = None,
                                          long_time_reservation_duration=None):
        """Get the free periods calendar extract from the _process_execution_plan"""
        return self._process_execution_plan.get_free_periods_calendar_extract(
            start_time=start_time, end_time=end_time, issue_id=issue_id,
            time_slot_duration=time_slot_duration, long_time_reservation_duration=long_time_reservation_duration)

    ####################################################################################################################
    # #### Order Progress ####
    ####################################################################################################################

    def _get_possible_value_added_processes_without_predecessor(self, value_added_processes_requested=None):
        """
        The method is used to find all ValueAddedProcesses of the ProductionOrder without a predecessor.

        Returns
        -------
        possible_value_added_processes: possible value_added_processes that did not have any predecessors
        """
        value_added_processes_lst = self.get_value_added_processes_requested_lst(value_added_processes_requested)
        possible_value_added_processes = \
            [value_added_process
             for value_added_process in value_added_processes_lst
             if not value_added_process.predecessors or value_added_process.predecessors == [()]]
        return possible_value_added_processes

    def get_possible_value_added_processes(self, value_added_process_completed: Optional[ValueAddedProcess] = None,
                                           possible_value_added_processes: Optional[list[ValueAddedProcess]] = None,
                                           value_added_processes_requested=None,
                                           value_added_processes_completed=None):
        """
        The method is used to find all ValueAddedProcesses that are recorded with predecessors.
        """
        if value_added_processes_requested is None:
            value_added_processes_requested = self.value_added_processes_requested

        if possible_value_added_processes is None:
            possible_value_added_processes = self.possible_value_added_processes

        if value_added_process_completed:
            possible_value_added_processes = \
                [possible_value_added_process
                 for possible_value_added_process in possible_value_added_processes
                 if possible_value_added_process.identification != value_added_process_completed.identification]

        if value_added_processes_completed is None:
            value_added_processes_completed = self.value_added_processes_completed

        value_added_processes_requested = \
            self.get_value_added_processes_requested_lst(value_added_processes_requested)
        value_added_processes_requested_ids = \
            [value_added_process_requested.identification
             for value_added_process_requested in value_added_processes_requested]

        if value_added_process_completed:
            # avoid a comparison of different objects that have the same id (deepcopy)
            value_added_processes_completed_ids = \
                [value_added_processes_completed.identification
                 for value_added_processes_completed_d in list(value_added_processes_completed.values())
                 for value_added_processes_completed in list(value_added_processes_completed_d.values())[0]]

            # value_added_processes that occur more often than once are not seen as completed
            # logic: all of them should be completed before the
            # ToDo: maybe not applicable for all use cases
            value_added_processes_completed_ids = \
                [value_added_processes_completed_id
                 for value_added_processes_completed_id in value_added_processes_completed_ids if
                 value_added_processes_completed_id not in value_added_processes_requested_ids]
        else:
            value_added_processes_completed_ids = []

        all_value_added_processes_ids = value_added_processes_requested_ids + value_added_processes_completed_ids

        # check all requested_value_added_processes if they can be performed next
        for possible_next_value_added_process in value_added_processes_requested:
            # check if no requested process is in the predecessor
            all_predecessor_finished = True
            # at least one of each predecessor_process_lst must be chosen
            for predecessor_process_lst in possible_next_value_added_process.predecessors:
                predecessor_finished = False
                all_predecessors_not_chosen = True
                for possible_predecessor in predecessor_process_lst:
                    if possible_predecessor.identification in value_added_processes_completed_ids:
                        if possible_predecessor.identification in value_added_processes_requested_ids and \
                                self.all_predecessors:
                            continue
                        # at least one is completed
                        predecessor_finished = True
                        break

                    elif possible_predecessor.identification in all_value_added_processes_ids:
                        all_predecessors_not_chosen = False
                # ToDo: one while loop should be possible
                # case: all the predecessors are not mandatory/ not chosen/ not needed because of the feature choice
                if not all_predecessors_not_chosen or predecessor_finished:
                    # ToDo: speak with Christian
                    if not predecessor_finished:
                        # at least one of the lists has no completed process
                        all_predecessor_finished = False
                        break
                    continue

                # iterate through the predecessors of the predecessors
                predecessor_found = False
                while not predecessor_found:
                    new_predecessor_process_lst = []
                    predecessors_lst = [predecessor_process_lst1
                                        for possible_predecessors in predecessor_process_lst
                                        for predecessor_process_lst1 in possible_predecessors.predecessors]

                    if not predecessors_lst or predecessors_lst == [()]:
                        # case: no predecessor available
                        predecessor_finished = True
                        break

                    for predecessor_process_lst1 in predecessors_lst:
                        for pre_predecessor in predecessor_process_lst1:
                            new_predecessor_process_lst.append(pre_predecessor)
                            if pre_predecessor.identification in value_added_processes_completed_ids:
                                # at least one is completed
                                predecessor_finished = True
                                break
                            elif pre_predecessor.identification in all_value_added_processes_ids:
                                # further search needed
                                predecessor_found = True
                        # if not predecessor_finished:
                        #     # at least one of the lists has no completed process
                        #     all_predecessor_finished = False
                        #     break

                    predecessor_process_lst = new_predecessor_process_lst

            if all_predecessor_finished:
                possible_value_added_processes.append(possible_next_value_added_process)

        possible_value_added_processes = [value_added_process_requested
                                          for value_added_process_requested in value_added_processes_requested
                                          if value_added_process_requested in list(set(possible_value_added_processes))]

        return possible_value_added_processes

    def get_value_added_processes_requested_lst(self, value_added_processes_requested=None):
        """Returns all value_added_processes_requested as list"""
        if value_added_processes_requested is None:
            value_added_processes_requested = self.value_added_processes_requested

        value_added_processes_requested_lst = _unroll_value_added_process_list(value_added_processes_requested)

        return value_added_processes_requested_lst

    def complete_value_added_process(self, value_added_process_completed: ValueAddedProcess,
                                     process_executions: list[ProcessExecution] = [],
                                     sequence_already_ensured: bool = False):
        """
        Complete a value_added_process

        Parameters
        ----------
        value_added_process_completed: a value_added_process that is completed
        process_executions: process_executions that are responsible for the change
        sequence_already_ensured: bool value
        """
        # find the value_added_process with the feature
        feature_value_added_process = {}
        for feature, value_added_processes_d in self.value_added_processes_requested.items():
            value_added_process_d = \
                {i: value_added_process
                 for i, value_added_processes in value_added_processes_d.items()
                 for value_added_process in value_added_processes
                 if value_added_process.identification == value_added_process_completed.identification}

            if value_added_process_d:
                feature_value_added_process[feature] = value_added_process_d
                break

        feasible = self.check_priority_chart_consistency(value_added_process_completed)
        if not feasible:
            print(f"Warning: The process '{value_added_process_completed.name}' is not possible to execute "
                  f"at this point of the priority chart")

        # exceptions
        if len(feature_value_added_process) == 0:
            pass
            # value_added_process_already_completed = \
            #     [value_added_process
            #      for feature, value_added_process_d in self.value_added_processes_completed.items()
            #      for value_added_process_list in list(value_added_process_d.values())
            #      for value_added_process in value_added_process_list
            #      if value_added_process.identification == value_added_process_completed.identification]
            # if len(value_added_process_already_completed) > 0:
            #     raise ValueError(f"[{self.__class__.__name__}] complete_value_added_process cannot be conducted "
            #                      f"because the value_added_process is already completed")
            # else:
            #     raise ValueError(f"[{self.__class__.__name__}] complete_value_added_process cannot be conducted "
            #                      f"because the value_added_process is not requested")

        elif len(feature_value_added_process) > 1:
            raise ValueError(f"[{self.__class__.__name__}] complete_value_added_process cannot be conducted "
                             f"because more than one value_added_process is matched")

        # remove the value_added_process_completed from requested and add them to completed
        features = list(feature_value_added_process.keys())
        if not features:
            debug_str = (f"[{self.__class__.__name__}] No feature matchable to the process. "
                         f"Process Name: {value_added_process_completed.name} \n"
                         f"Identifier: {self.order.identifier} \n"
                         f"Order internal id: {self.order.identification} \n"
                         f"Features: {features}")
            logging.debug(debug_str)
            raise Exception(debug_str)

        feature = features[0]
        value_added_processes_ = feature_value_added_process[feature]
        value_added_process_completed_key = list(value_added_processes_.keys())[0]
        value_added_process_completed = value_added_processes_.pop(value_added_process_completed_key)

        updated_features_requested = []
        removed = False
        for value_added_process in self.value_added_processes_requested[feature][value_added_process_completed_key]:
            if value_added_process.identification == value_added_process_completed.identification and not removed:
                removed = True
            else:
                updated_features_requested.append(value_added_process)
        self.value_added_processes_requested[feature][value_added_process_completed_key] = updated_features_requested

        self.value_added_processes_completed.setdefault(feature,
                                                        {}).setdefault(value_added_process_completed_key,
                                                                       []).append(value_added_process_completed)

        # update possible_value_added_processes - Push ToDo: maybe a pull approach would be the better alternative
        self.update_possible_value_added_processes(value_added_process_completed=value_added_process_completed)

        # complete feature if no more value_added_process is requested for them
        feature_completed = self.value_added_processes_requested[feature][value_added_process_completed_key]
        if not feature_completed:
            self.order.complete_feature(feature_completed=feature,
                                        process_executions=process_executions,
                                        sequence_already_ensured=sequence_already_ensured)
        else:
            self.order.match_feature_process_executions(feature=feature,
                                                        process_executions=process_executions)

        # update the dynamic attribute
        # find the last process_execution
        if process_executions:
            last_process_execution_idx = np.argmax([process_execution.executed_end_time
                                                    for process_execution in process_executions])
            current_time = process_executions[last_process_execution_idx].get_latest_available_time_stamp()
            self.update_attributes(process_execution=process_executions[last_process_execution_idx],
                                   current_time=current_time,
                                   value_added_processes_completed=value_added_process_completed,
                                   change_type="ADD", sequence_already_ensured=sequence_already_ensured)

    def check_priority_chart_consistency(self, value_added_process: ValueAddedProcess) -> bool:
        """
        Check if the value_added_process is possible to execute based on the priority chart defined in the
        value_added_processes needed to be executed for the order completion to detect not possible processing sequences

        Parameters
        ----------
        value_added_process: the value_added_process that is checked according to predecessors fulfilled.

        Returns
        -------
        consistent: a bool value that states if the execution of the value_added_process is consistent
        with the priority chart
        """
        if not isinstance(value_added_process, ValueAddedProcess):
            return False

        processes_executed: list[Union[Process, ValueAddedProcess]] = (
            [process
             for process_dict in list(self._value_added_processes_completed.values())
             for lst in list(process_dict.values())
             for process in lst])
        processes_requested: list[Union[Process, ValueAddedProcess]] = (
            [process
             for process_dict in list(self.value_added_processes_requested.values())
             for lst in list(process_dict.values())
             for process in lst])
        if value_added_process in processes_requested:
            # remove the value_added_process from the list of requested processes
            processes_requested.remove(value_added_process)

        consistent = value_added_process.possible_to_execute(processes_executed, processes_requested)

        return consistent

    def update_possible_value_added_processes(self, value_added_process_completed: ValueAddedProcess,
                                              possible_value_added_processes=None,
                                              value_added_processes_requested=None,
                                              value_added_processes_completed=None):
        """Update the possible_value_added_processes"""
        new_possible_value_added_processes = (
            self.get_possible_value_added_processes(value_added_process_completed, possible_value_added_processes,
                                                    value_added_processes_requested, value_added_processes_completed))
        self.possible_value_added_processes = new_possible_value_added_processes

    def match_value_added_process_process_execution(self, value_added_process: ValueAddedProcess,
                                                    process_execution: ProcessExecution):
        """
        Map a new process_execution to the value_added_process (needed for the kpi calculation)

        Parameters
        ----------
        value_added_process: needed to fulfill the order
        process_execution: matched to the process
        """
        self.value_added_process_process_execution_match.setdefault(value_added_process,
                                                                    []).append(process_execution)

    ####################################################################################################################
    # #### Finished part type ####
    ####################################################################################################################

    def get_finished_part_entity_type(self) -> EntityType:
        """
        Note: The method is implemented based on the assumption that the last value added process has no successor.

        Returns
        -------
        main_entity_entity_type: the entity_type of the finished main_entity
        """

        all_value_added_processes = self._get_all_value_added_processes()
        last_value_added_process: ValueAddedProcess = self._get_possible_last_value_added_process()

        transformation_types_main_entity = EntityTransformationNodeTransformationTypes.MAIN_ENTITY

        root_nodes = last_value_added_process.transformation_controller.get_root_nodes()
        main_entity_entity_types = \
            [root_node.entity_type
             for root_node in root_nodes
             if root_node.compare_transformation_type_self([transformation_types_main_entity])]
        main_entity_entity_type = main_entity_entity_types[0]

        # check if super_entity_type exist assuming that the last process has a specific entity_type
        # - no super_entity_type
        predecessor_process = [vap
                               for vap_lst in last_value_added_process.predecessors
                               for vap in vap_lst
                               if vap in all_value_added_processes][0]

        root_nodes_predecessor = predecessor_process.transformation_controller.get_root_nodes()
        entity_types = \
            [root_node.entity_type
             for root_node in root_nodes_predecessor
             if root_node.compare_transformation_type_self([transformation_types_main_entity])]
        last_entity_type = entity_types[0]
        if last_entity_type.super_entity_type:
            main_entity_entity_type = last_entity_type

        return main_entity_entity_type

    def _get_possible_last_value_added_process(self) -> ValueAddedProcess:

        all_value_added_processes = self._get_all_value_added_processes()
        possible_last_value_added_process = [value_added_process
                                             for value_added_process in all_value_added_processes
                                             if not value_added_process.successors][0]

        return possible_last_value_added_process

    def _get_all_value_added_processes(self) -> list[ValueAddedProcess]:
        """Combine the requested and completed value added process into one list"""

        all_value_added_processes = (_unroll_value_added_process_list(self.value_added_processes_requested) +
                                     _unroll_value_added_process_list(self.value_added_processes_completed))

        return all_value_added_processes

    ####################################################################################################################
    # #### BOM ####
    ####################################################################################################################

    def create_bill_of_materials(self) -> dict[EntityType: float]:
        """
        Perform a forward simulation through the transformation graph to get all materials needed/ the BOM.

        Returns
        -------
        needed_entity_types_bom: a list with all materials needed to execute the production_order
        """
        # ToDo: frame problem(super_entity_types   - done?

        needed_entity_types_bom = {}  # ToDo: link to the value_added_processes
        available_entity_types = {}
        # go through all value_added_processes

        forward_simulation = self._forward_simulation()  # create a generator
        for next_value_added_process in forward_simulation:

            needed_entity_types_dict = next_value_added_process.get_bill_of_material()

            for needed_entity_type, needed_amount in needed_entity_types_dict.items():
                needed_amount_abs = needed_amount

                if needed_entity_type in available_entity_types:
                    available_entity_types[needed_entity_type] -= needed_amount
                    if available_entity_types[needed_entity_type] < 0:
                        needed_amount_abs = abs(available_entity_types[needed_entity_type])
                        available_entity_types[needed_entity_type] = 0
                    else:
                        needed_amount_abs = 0
                # super entity_types can be replaced by normal entity_types
                elif needed_entity_type in [et.super_entity_type for et in available_entity_types]:
                    needed_entity_type = [et for et in available_entity_types
                                          if et.super_entity_type == needed_entity_type][0]
                    available_entity_types[needed_entity_type] -= needed_amount
                    if available_entity_types[needed_entity_type] < 0:
                        needed_amount_abs = abs(available_entity_types[needed_entity_type])
                        available_entity_types[needed_entity_type] = 0
                    else:
                        needed_amount_abs = 0

                if needed_entity_type in needed_entity_types_bom:
                    needed_entity_types_bom[needed_entity_type] += needed_amount_abs
                else:
                    needed_entity_types_bom[needed_entity_type] = needed_amount_abs

            output_entity_types = next_value_added_process.get_possible_output_entity_types()
            for entity_type, amount in output_entity_types:
                if entity_type in available_entity_types:
                    if entity_type in [et.super_entity_type for et in list(available_entity_types.keys())]:
                        # assumptions - already chosen parts are used in further processes
                        entity_type = [et for et in available_entity_types if et.super_entity_type == entity_type][0]
                    available_entity_types[entity_type] += amount
                else:
                    if entity_type in [et.super_entity_type for et in list(available_entity_types.keys())]:
                        # assumptions - already chosen parts are used in further processes
                        entity_type = [et for et in available_entity_types if et.super_entity_type == entity_type][0]
                    available_entity_types[entity_type] = amount

        return needed_entity_types_bom

    def get_possible_process_sequence(self):
        """
        Used for the odoo import to roll out the processes/ perform a forward simulation on the priority chart

        Returns
        ------
        possible_process_sequence: A possible process execution sequence for the work order, based on the priority chart
        """

        forward_simulation = self._forward_simulation()  # create a generator
        possible_process_sequence = [next_value_added_process
                                     for next_value_added_process in forward_simulation]

        return possible_process_sequence

    def _forward_simulation(self):
        """
        Simulate through the transformation graph (take always the first possible process)

        Yield
        -----
        next_value_added_process: the first found possible next value_added_process
        """

        value_added_processes_completed: dict[Feature: list[ValueAddedProcess]] = {}
        value_added_processes_requested = deepcopy(self.value_added_processes_requested)

        possible_value_added_processes = \
            self._get_possible_value_added_processes_without_predecessor(value_added_processes_requested)
        if not possible_value_added_processes:
            possible_value_added_processes = \
                self.get_possible_value_added_processes(value_added_processes_requested=value_added_processes_requested)

        # iterate through the value added processes
        while value_added_processes_requested:
            next_value_added_process = possible_value_added_processes[0]
            yield next_value_added_process

            if next_value_added_process:
                value_added_processes_requested, value_added_processes_completed = (
                    self._simulate_next_value_added_process(value_added_processes_requested,
                                                            value_added_processes_completed, next_value_added_process))

            possible_value_added_processes = \
                self.get_possible_value_added_processes(next_value_added_process, possible_value_added_processes,
                                                        value_added_processes_requested,
                                                        value_added_processes_completed)

    def _simulate_next_value_added_process(self, value_added_processes_requested, value_added_processes_completed,
                                           next_value_added_process):

        feature_with_completed_value_added_process = None
        for feature, value_added_processes_dict in value_added_processes_requested.items():
            for i, value_added_processes in value_added_processes_dict.items():
                for value_added_process in value_added_processes:
                    if next_value_added_process.identification == value_added_process.identification:
                        feature_with_completed_value_added_process = feature
                        break

        vap_batches_length_index = len(value_added_processes_requested[feature_with_completed_value_added_process]) - 1
        value_added_processes_requested[feature_with_completed_value_added_process][vap_batches_length_index].remove(
            next_value_added_process)
        if not value_added_processes_requested[feature_with_completed_value_added_process][vap_batches_length_index]:
            del value_added_processes_requested[feature_with_completed_value_added_process][vap_batches_length_index]

        vap_completed_dict = value_added_processes_completed.setdefault(feature_with_completed_value_added_process,
                                                                        {}).setdefault(vap_batches_length_index,
                                                                                       {})
        vap_completed_dict.setdefault(feature_with_completed_value_added_process,
                                      []).append(next_value_added_process)

        if not value_added_processes_requested[feature_with_completed_value_added_process]:
            del value_added_processes_requested[feature_with_completed_value_added_process]

        return value_added_processes_requested, value_added_processes_completed

