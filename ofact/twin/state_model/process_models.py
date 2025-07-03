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

Contains the models of the process

Classes:
    DTModel: The base class for all models (process_models)
    ---
    ProcessTimeModel: Describes the time a process need
    SimpleProcessTimeModel:
    SimpleNormalDistributedProcessTimeModel:
    SimpleSingleValueDistributedProcessTimeModel:
    ---
    QualityModel: Describes the change of the quality of the process
    SimpleQualityModel:
    SimpleBernoulliDistributedQualityModel:
    ---
    ResourceModel: Controls the resource groups possible for a process
    ResourceGroup: All resources in a group are needed to execute a process
    ---
    TransitionModel: Describes the change of position/ resource
    ---
    TransformationModel: Describes the precedence (priority) graph/ transformation of a process
    EntityTransformationNode: Describes a transformation and is part of a precedence graph

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from copy import copy
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Union, Optional

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes, DigitalTwinObject
from ofact.twin.state_model.entities import ConveyorBelt
from ofact.twin.state_model.probabilities import (ProbabilityDistribution, NormalDistribution, SingleValueDistribution,
                                                  BernoulliDistribution)

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import (EntityType, Entity, Resource, StationaryResource,
                                                 NonStationaryResource,
                                                 Part)
    from ofact.twin.state_model.processes import ProcessExecution, Process
    from ofact.twin.state_model.sales import Order
    from ofact.twin.state_model.model import StateModel

logging.debug("DigitalTwin/process_models")


class DTModel(DigitalTwinObject, metaclass=ABCMeta):

    def __init__(self,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        General dt model class currently used only for the process models.

        Parameters
        ----------
        is_re_trainable: defines if the model can be re-trained (for example the neural network)
        """

        DigitalTwinObject.__init__(self, identification=identification,
                                   external_identifications=external_identifications,
                                   domain_specific_attributes=domain_specific_attributes)
        self._is_re_trainable: bool = is_re_trainable

    def set_digital_twin_model(self, digital_twin_model: StateModel):
        """
        The method should be overwritten if used (intelligent prediction model given)

        Parameters
        ----------
        digital_twin_model: used to get information needed for the feature generation for the lead time prediction
        """
        pass

    def is_re_trainable(self) -> bool:
        if not hasattr(self, "_is_re_trainable"):
            self._is_re_trainable = False  # currently not in the static twin model
        return self._is_re_trainable


class ProcessTimeModel(DTModel, ProbabilityDistribution, metaclass=ABCMeta):

    def __init__(self,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        DTModel.__init__(self, is_re_trainable=is_re_trainable,
                         identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

    @abstractmethod
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
                                        distance=None) -> float:
        """
        The method is used to calculate the estimated_process_time e.g. for a process_execution not created until now.
        Assuming mean/ median for unknown values.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        event_type: PLAN or ACTUAL
        process: a process
        parts_involved: parts with their entity_transformation_nodes
        resources_used: resources with their entity_transformation_nodes
        resulting_quality: quality after process execution
        main_resource: main resource
        origin: origin resource
        destination: destination resource
        order: order that is the customer
        executed_start_time: start time of the process
        executed_end_time: end time of the process
        source_application: source application e.g., "Scenario 1"
        distance: used for transport processes if the process time is dependent on the transport length
        (between origin and destination)

        Returns
        -------
        the expected process time
        """
        pass

    @abstractmethod
    def get_expected_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the expected_process_time e.g. for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: A process_execution
        distance: used for transport processes if the process time is dependent on the transport length
        (between origin and destination)

        Returns
        -------
        the expected process time
        """
        pass

    @abstractmethod
    def get_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the process_time e.g. for the actual_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: A process_execution
        distance: used for transport processes if the process time is transport length-dependent

        Returns
        -------
        the process time
        """
        pass

    @classmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters = {}
        return parameters

    def get_parameters_with_values(self) -> dict[str, object]:
        parameters = {}
        return parameters

    def get_expected_value(self) -> float:
        pass

    def get_random_number(self,
                          weight=None,
                          negative_values_allowed: bool = False) -> float:
        pass

    def copy(self):
        return copy(self)


def _get_lead_time_factor(main_resource: Optional[StationaryResource, NonStationaryResource] = None,
                          origin: Optional[Resource] = None, destination: Optional[Resource] = None,
                          distance: Optional[int, float] = None, expectation: bool = False) -> float:
    """
    The method is used to calculate the factor, which is determined by the involved entities and their attributes
    (like speed). Also, the distance plays a role.

    Parameters
    ----------
    distance: the parameter distance can be alternatively used for 'distance' between origin and destination
    It is needed because the position of a resource is changing over time - therefore, the distance is also changing
    and makes planning into the future difficult (alternative - derivation from dynamic attributes)
    expectation: if the expected value of the performance probability distribution should be used

    Returns
    -------
    the calculated factor
    """
    factor = 1
    if main_resource is not None:
        if not expectation:
            performance = main_resource.get_performance()
        else:
            performance = main_resource.get_expected_performance()

        try:
            factor /= performance
        except:
            print(f"Problem with the performance value of the main resource: '{main_resource.name}' '{performance}'")

    if distance is None:
        distance = _get_distance(origin, destination, main_resource)

    if distance > 0:
        factor *= distance

    return factor


def _get_distance(origin: Optional[Resource] = None, destination: Optional[Resource] = None,
                  main_resource: Optional[Resource] = None) -> float:
    """
    Calculate the distance between the origin and destination resource

    Parameters
    ----------
    main_resource: if the main_resource is a conveyor belt, the distance is determined via the length of
    the conveyor belt and not the distance between origin and destination

    Returns
    -------
    distance: length between origin and destination
    """

    # case: conveyor belt - the conveyor belt should be the main_resource
    if main_resource is not None:
        if isinstance(main_resource, ConveyorBelt):
            if origin is None and destination is None:
                distance = main_resource.conveyor_length
                return distance
            else:
                if origin == main_resource.origin and destination == main_resource.destination:
                    distance = main_resource.conveyor_length
                    return distance

    # case: free connection
    distance = 0
    if origin and destination:
        if isinstance(origin, ConveyorBelt) or isinstance(destination, ConveyorBelt):  # must be a transfer
            return distance

        if origin.physical_body.position != destination.physical_body.position:
            vector_ = np.array(origin.physical_body.position) - np.array(destination.physical_body.position)
            distance = np.linalg.norm(vector_)

    return distance


class SimpleProcessTimeModel(ProcessTimeModel):

    def __init__(self,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Calculates the process time that the process needs. Depends on a probability distribution and other factors
        (like distance, resource efficiency, etc.)
        """
        ProcessTimeModel.__init__(self, is_re_trainable=is_re_trainable,
                                  identification=identification, external_identifications=external_identifications,
                                  domain_specific_attributes=domain_specific_attributes)

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
                                        distance=None) -> float:
        # and also the other process_execution attributes

        factor = _get_lead_time_factor(main_resource, origin, destination, distance, expectation=True)
        expected_process_time: float = self.get_expected_value()
        expected_process_time *= factor

        return expected_process_time

    def get_expected_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the expected_process_time e.g. for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        expected_process_time: the expected process time
        """
        main_resource = process_execution.main_resource
        origin = process_execution.origin
        destination = process_execution.destination
        expected_process_time = self.get_estimated_process_lead_time(main_resource=main_resource,
                                                                     origin=origin,
                                                                     destination=destination)

        return expected_process_time

    def get_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the process_time e.g. for the actual_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        process_time: the process time
        """
        main_resource = process_execution.main_resource
        origin = process_execution.origin
        destination = process_execution.destination

        factor = _get_lead_time_factor(main_resource, origin, destination, distance)
        process_time = self.get_random_number()
        process_time *= factor

        return process_time


class SimpleNormalDistributedProcessTimeModel(SimpleProcessTimeModel, NormalDistribution):

    def __init__(self,
                 mue: float,
                 sigma: float = 1,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        SimpleProcessTimeModel.__init__(self, is_re_trainable=is_re_trainable,
                                        identification=identification,
                                        external_identifications=external_identifications,
                                        domain_specific_attributes=domain_specific_attributes)
        NormalDistribution.__init__(self, mue=mue, sigma=sigma)

    def get_expected_value(self):
        expected_value = NormalDistribution.get_expected_value(self)
        return expected_value

    def get_random_number(self, weight=None, negative_values_allowed: bool = False):
        get_random_value = NormalDistribution.get_random_number(self, weight=weight,
                                                                negative_values_allowed=negative_values_allowed)
        return get_random_value


class SimpleSingleValueDistributedProcessTimeModel(SimpleProcessTimeModel, SingleValueDistribution):

    def __init__(self,
                 value,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        SimpleProcessTimeModel.__init__(self, is_re_trainable=is_re_trainable,
                                        identification=identification,
                                        external_identifications=external_identifications,
                                        domain_specific_attributes=domain_specific_attributes)
        SingleValueDistribution.__init__(self, value=value)

    def get_expected_value(self):
        expected_value = SingleValueDistribution.get_expected_value(self)
        return expected_value

    def get_random_number(self, weight=None,
                          negative_values_allowed: bool = False):
        get_random_value = SingleValueDistribution.get_random_number(self, negative_values_allowed)
        return get_random_value


class QualityModel(DTModel, ProbabilityDistribution, metaclass=ABCMeta):

    def __init__(self,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Determine the quality that the process creates. Depends on a probability distribution and other factors
        (like distance, resource efficiency, etc.)
        """
        DTModel.__init__(self, is_re_trainable=is_re_trainable,
                         identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

    def copy(self):
        """Copy the object with the same identification."""
        quality_model_copy = super(QualityModel, self).copy()
        return quality_model_copy

    @abstractmethod
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
        pass

    @abstractmethod
    def get_expected_quality(self, process_execution: ProcessExecution, distance: Optional[float] = None) -> float:
        """
        The method is used to calculate the expected_quality e.g., for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        the quality value
        """
        return self.get_expected_value()

    @abstractmethod
    def get_quality(self, process_execution: ProcessExecution, distance: Optional[float] = None) -> float:
        """
        The method is used to calculate the quality, e.g., for the actual_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        the quality value
        """
        return self.get_random_number()

    @abstractmethod
    def get_expected_value(self):
        pass

    @abstractmethod
    def get_random_number(self,
                          weight=None,
                          negative_values_allowed: bool = False):
        pass

    @classmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters = {}
        return parameters

    def get_parameters_with_values(self) -> dict[str, object]:
        parameters = {}
        return parameters


class SimpleQualityModel(QualityModel):

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
        raise NotImplementedError

    def get_expected_quality(self, process_execution: ProcessExecution, distance: Optional[float] = None) -> float:
        """
        The method is used to calculate the expected_quality e.g. for the planned_process_execution.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        the quality value
        """
        raise NotImplementedError

    def get_quality(self, process_execution: ProcessExecution, distance: Optional[float] = None) -> float:
        """
        The method is used to determine the quality based on the single qualities of the parts and resources.
        The calculation is based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.

        Parameters
        ----------
        process_execution: provides the data needed for the lead_time sample
        distance: explained in method get_lead_time_factor

        Returns
        -------
        the quality (a float that can be 0 or 1)
        """

        resulting_quality: float = 1.0
        resulting_quality *= np.prod([part_tuple[0].quality
                                      for part_tuple in process_execution.parts_involved])
        resulting_quality *= np.prod([resource_tuple[0].quality
                                      for resource_tuple in process_execution.resources_used])
        quality = self.get_random_number(weight=resulting_quality)
        return quality


class SimpleBernoulliDistributedQualityModel(SimpleQualityModel, BernoulliDistribution):

    def __init__(self,
                 probability: float = 1.0,
                 not_successful_value: float = 0.0,
                 successful_value: float = 1.0,
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        BernoulliDistribution.__init__(self, probability=probability, not_successful_value=not_successful_value,
                                       successful_value=successful_value)
        SimpleQualityModel.__init__(self, is_re_trainable=is_re_trainable,
                                    identification=identification,
                                    external_identifications=external_identifications,
                                    domain_specific_attributes=domain_specific_attributes)

    def get_random_number(self, weight=None, negative_values_allowed: bool = False):
        get_random_value = BernoulliDistribution.get_random_number(self, weight=weight,
                                                                   negative_values_allowed=negative_values_allowed)
        return get_random_value

    def get_expected_value(self):
        pass


def create_count_dict(lst):
    count_dict = {}
    for elem in lst:
        if elem not in count_dict:
            count_dict[elem] = 1
        else:
            count_dict[elem] += 1

    return count_dict


class ResourceGroup(DigitalTwinObject):

    def __init__(self,
                 resources: list[EntityType],
                 main_resources: list[EntityType],
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The ResourceGroup list all the resource types a process needs to be executed

        Parameters
        ----------
        resources: list of resource types that need to be used, the first resource in the list is
        the main resource that defines the place where the process takes place (workstation, forklift),
        all other resources and parts are situated in this resource when the process is executed
        main_resources: Resources that actively execute the corresponding process
        """
        super().__init__(identification=identification,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        self.resources: list[EntityType] = resources
        self.main_resources: list[EntityType] = main_resources  # TODO Check if list is necessary

    def __str__(self):
        resource_names = self.get_resource_names()
        main_resource_names = self.get_main_resource_names()
        return (f"ResourceGroup with ID '{self.identification}' and resources '{resource_names}' and "
                f"main_resources {main_resource_names}")

    def copy(self):
        """Copy the object with the same identification."""
        resource_group_copy = super(ResourceGroup, self).copy()
        resource_group_copy.resources = resource_group_copy.resources.copy()
        resource_group_copy.main_resources = resource_group_copy.main_resources.copy()

        return resource_group_copy

    def get_resource_names(self):
        return [resource.name for resource in self.resources]

    def get_main_resource_names(self):
        return [resource.name for resource in self.main_resources]

    def check_ability_to_perform_process_as_resource(self, resource, already_assigned_resources):
        """
        Check the ability of the resource to be part of the resource model to execute a process that has
        the resource model as a possibility.

        Parameters
        ----------
        resource: a resource
        already_assigned_resources: resources that are already used for the process (not needed anymore)

        Returns
        -------
        occurrence: True if the resource can be part of the resource model, else False
        """

        # consider the already assigned resources
        already_assigned_entity_types = [resource.entity_type
                                         for resource in already_assigned_resources]
        open_entity_types = []
        for resource_entity_type in self.resources:
            if resource_entity_type not in already_assigned_entity_types:
                open_entity_types.append(resource_entity_type)
            else:
                already_assigned_entity_types.remove(resource_entity_type)

        occurrence_list = [resource
                           for resource_entity_type in open_entity_types
                           if resource_entity_type.check_entity_type_match_lower(resource.entity_type)]
        occurrence = bool(occurrence_list)

        return occurrence

    def check_ability_to_perform_process_as_main_resource(self, resource: Optional[Resource]):
        """
        Check the ability of the resource to be the main_resource of the resource model to execute a process that has
        the resource model as a possibility.

        Parameters
        ----------
        resource: a resource

        Returns
        -------
        occurrence: True if the resource can be the main_resource of the resource model, else False
        """
        if resource is None:
            return False

        occurrence_list = [resource
                           for resource_entity_type in self.main_resources
                           if resource_entity_type.check_entity_type_match(resource.entity_type)]
        occurrence = bool(occurrence_list)

        return occurrence

    def get_resources_without_main_resource(self) -> list[EntityType]:
        """Get resource entity_types without main_resource entity_types"""
        main_resource_idx = self.resources.index(self.main_resources[0])
        resources_without_main_resource = [resource_entity_type
                                           for idx, resource_entity_type in enumerate(self.resources)
                                           if idx != main_resource_idx]

        return resources_without_main_resource

    def check_resource_group_applicability(self, resources: list[Resource], main_resource: Optional[Resource]):
        """
        Resources and main_resource do not have to be complete

        Parameters
        ----------
        resources: a list of resources
        main_resource: a resource if given

        Returns
        -------
        resource_group_applicable: if the length of given and required resources matches
        """
        resource_entity_types = [resource.entity_type
                                 for resource in resources]

        if main_resource is not None:
            main_resource_entity_type = main_resource.entity_type
            if main_resource_entity_type.identification != self.main_resources[0].identification:
                return False

        # check if the length of given and required resources matches
        # ToDo: consider amount (more than one resource with the same entity_type)
        # resources = self.resources.copy()
        # for resource_entity_type in resource_entity_types:
        #     if resource_entity_type in self.resources:
        #         resources.remove(resource_entity_type)
        #     elif resource_entity_type.super_entity_type in self.resources:
        #         resources.remove(resource_entity_type.super_entity_type)

        matching_resources = [resource_entity_type
                              for resource_entity_type in resource_entity_types
                              if resource_entity_type in self.resources or
                              resource_entity_type.super_entity_type in self.resources]
        resource_group_applicable = len(matching_resources) == len(resource_entity_types)

        return resource_group_applicable

    def get_needed_resources(self, already_assigned_resources: list[Resource]):
        """
        Determine needed resources from a resource list

        Parameters
        ----------
        already_assigned_resources: A list of resources that are already assigned for a process_execution
        if possible to use

        Returns
        -------
        needed_resource_entity_types: A list of resources that are able to use within the resource group
        """

        # consider the already assigned resources
        already_assigned_entity_types = [resource.entity_type
                                         for resource in already_assigned_resources]
        already_assigned_entity_types_with_amount = create_count_dict(already_assigned_entity_types)

        needed_resource_entity_types = []
        for resource_entity_type in self.resources:
            if resource_entity_type not in already_assigned_entity_types_with_amount:
                needed_resource_entity_types.append(resource_entity_type)
            else:
                already_assigned_entity_types_with_amount[resource_entity_type] -= 1
                if already_assigned_entity_types_with_amount[resource_entity_type] == 0:
                    del already_assigned_entity_types_with_amount[resource_entity_type]

        return needed_resource_entity_types

    def get_usable_resources_for_process(self, available_resources: list[tuple[Resource, EntityTransformationNode]]) \
            -> list[tuple[Resource,]]:
        """
        Used to determine which (available) resources (e.g., organized in processes before)
        can be used in this resource_group

        Parameters
        ----------
        available_resources: A list of resources as tuples that are available
        if possible to use

        Returns
        -------
        usable_resources: A list of resources that are able to use within the resource group as tuples
        """
        usable_resources = [(resource_tuple[0],)
                            for resource_tuple in available_resources
                            for entity_type in self.resources
                            if resource_tuple[0].entity_type.check_entity_type_match(entity_type)]
        if len(usable_resources) > len(self.resources):
            raise ValueError("Further restrictions for the resource selection needed")

        return usable_resources

    def check_resources_build_resource_group(self, available_resources: list[Resource],
                                             available_main_resource: Resource) -> bool:
        """Check if the input resources can complete at most the resource_group"""
        already_assigned_resources = []
        for available_resource in available_resources:
            ability_to_perform_process_as_resource = \
                self.check_ability_to_perform_process_as_resource(resource=available_resource,
                                                                  already_assigned_resources=already_assigned_resources)
            if ability_to_perform_process_as_resource:
                already_assigned_resources.append(available_resource)
            else:
                return False

        ability_to_perform_process_as_main_resource = \
            self.check_ability_to_perform_process_as_main_resource(resource=available_main_resource)

        if ability_to_perform_process_as_main_resource:
            return True
        else:
            return False

    def completely_filled(self):

        not_completely_filled_attributes = []
        if not isinstance(self.resources, list):
            not_completely_filled_attributes.append("resources")
            not_correctly_instantiated_attributes = [resource
                                                     for resource in self.resources
                                                     if not isinstance(resource, DigitalTwinObject)]
            if not_correctly_instantiated_attributes:
                not_completely_filled_attributes.append("resources")
        if not isinstance(self.main_resources, list):
            not_completely_filled_attributes.append("main_resources")
            not_correctly_instantiated_attributes = [resource
                                                     for resource in self.main_resources
                                                     if not isinstance(resource, DigitalTwinObject)]
            if not_correctly_instantiated_attributes:
                not_completely_filled_attributes.append("main_resources")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class ResourceModel(DTModel):

    def __init__(self,
                 resource_groups: list[ResourceGroup],
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        DTModel.__init__(self, is_re_trainable=is_re_trainable,
                         identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        self._resource_groups: list[ResourceGroup] = resource_groups

    @property
    def resource_groups(self):
        return self._resource_groups

    @resource_groups.setter
    def resource_groups(self, resource_groups):
        self._resource_groups = resource_groups

    def copy(self):
        """Copy the object with the same identification."""
        resource_model_copy: ResourceModel = super(ResourceModel, self).copy()
        resource_model_copy._resource_groups = resource_model_copy._resource_groups.copy()

    def get_resource_groups(self, process_execution: ProcessExecution):
        return self._resource_groups


class TransitionModel(DTModel):

    def __init__(self,
                 possible_origins: list[Resource],
                 possible_destinations: list[Resource],
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Describes the possible Transitions from one (Stationary)Resource to another.

        Parameters
        ----------
        possible_origins: Possible starting point for the (transport) process are these (stationary) resources
        possible_destinations: Possible ending point for the (transport) process are these (stationary) resources
        """
        DTModel.__init__(self, is_re_trainable=is_re_trainable,
                         identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._possible_origins: list[Resource] = possible_origins
        self._possible_destinations: list[Resource] = possible_destinations

    @property
    def possible_origins(self):
        return self._possible_origins

    @possible_origins.setter
    def possible_origins(self, possible_origins):
        self._possible_origins = possible_origins

    @property
    def possible_destinations(self):
        return self._possible_destinations

    @possible_destinations.setter
    def possible_destinations(self, possible_destinations):
        self._possible_destinations = possible_destinations

    def copy(self):
        """Copy the object with the same identification."""
        transition_model_copy = super(TransitionModel, self).copy()
        transition_model_copy.possible_origins = transition_model_copy.possible_origins.copy()
        transition_model_copy.possible_destinations = transition_model_copy.possible_destinations.copy()

        return transition_model_copy

    def get_possible_origins(self) -> list[Resource]:
        """
        The method is used to determine the possible origins.

        Returns
        -------
        possible_origins: A list of possible origins (Resources)
        """
        return self._possible_origins

    def get_possible_destinations(self, origin: Optional[Resource] = None) -> list[Resource]:
        """
        The method is used to determine the possible destinations.

        Returns
        -------
        possible_origins: A list of possible destinations (Resources)
        """
        if origin is None:
            return self._possible_destinations
        else:
            if origin in self._possible_origins:
                return self._possible_destinations
            else:
                return []

    def get_destination(self, process_execution: ProcessExecution) -> Optional[Resource]:
        """

        Parameters
        ----------
        process_execution: a process_execution

        Returns
        -------
        destination: not implemented until now
        """
        raise NotImplementedError("Currently no use case for the method 'get_destination()'")
        destination: Resource
        return destination


# # # # TRANSFORMATION MODEL # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


EntityTransformationNodeTransformationTypes = Enum('EntityTransformationNodeTransformationTypes',
                                                   'MAIN_ENTITY '
                                                   'BLANK '
                                                   'INGREDIENT '
                                                   'SUB_ENTITY '
                                                   'DISASSEMBLE '
                                                   'SUPPORT '
                                                   'UNSUPPORT '
                                                   'QUALITY_INSPECTION',
                                                   qualname='EntityTransformationNode.TransformationTypes')

EntityTransformationNodeIoBehaviours = Enum('EntityTransformationNodeIoBehaviours',
                                            'EXIST CREATED DESTROYED',
                                            qualname='EntityTransformationNode.IoBehaviours')


class EntityTransformationNode(DigitalTwinObject):
    TransformationTypes = EntityTransformationNodeTransformationTypes
    IoBehaviours = EntityTransformationNodeIoBehaviours

    @classmethod
    def get_transformation_types(cls):
        return cls.TransformationTypes

    @classmethod
    def get_io_behaviours(cls):
        return cls.IoBehaviours

    def __init__(self,
                 entity_type: EntityType,  # | PartType
                 amount: int,
                 transformation_type: TransformationTypes,
                 io_behaviour: IoBehaviours,
                 quality: Optional[float] = None,
                 reset_inspected_quality: Optional[bool] = False,
                 parents: Optional[list[EntityTransformationNode]] = None,
                 children: Optional[list[EntityTransformationNode]] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The EntityTransformationNode (ETN) describes how an entity is transformed in a process.
        The transformation is described through a transformation graph.
        The root node(s) (ETNs) of the graph build the required input entities and
        the leaf/ end node(s) (ETNs) the output entities.
        The transformation is done by transforming the entities from the parent nodes
        (at the beginning of the transformation the parent nodes are the root nodes)
        considering the transformation_types and io_behaviours.
        Io_behaviour defines if entities have to be destroyed or created in the current node.

        Parameters
        ----------
        entity_type: EntityType that is transformed
        amount: amount of part
        quality: if set, the minimum quality (of the entity matched with the entity_type) needed
        to start the transformation
        else, the transformation can start at any quality
        reset_inspected_quality: if set, the inspected_quality is reset after the transformation
        - only relevant for root nodes of the transformation type exists, others are ignored ..
        (meaning that the quality should be inspected again if required - a kind of quality life span)
        children: list of EntityTransformationNode. Successor nodes of the current one.
        Transformation_type is used in these nodes
        parents: list of EntityTransformationNode. Possible predecessors of the current node

        ###
        transformation_type: determine the transformation (see below)

        Possible variables:
        - MAIN_ENTITY: Necessary to start the process if not created.
        Entity leaves the process unchanged or extended

        - BLANK: Necessary to start the process if not created. Part is transformed/ processed.
        The entity type is changed, but the attributes remain untouched (no further parts attached) (e.g., bending)

        - INGREDIENT: Necessary to start the process if not created. Part ist transformed into or
        combined with the main entity. Cannot be removed later (e.g., surface coating)

        - SUB_ENTITY: Necessary to start the process if not created. Entity is built into the main_entity and
        can be taken apart later (e.g., assembly, packing)
        - DISASSEMBLE: SubParts can be disassembled in the children nodes.

        - SUPPORT: Necessary to marry (NonStationaryResource and Parts) or
        (NonStationaryResource and NonStationaryResource).
        The marriage is needed to create a (longer) connection, for example, for transport.
        E.g.: AGV and main_product (can be identified if the SUPPORT is also found in the successor processes) or
        Bin and screws.
        - UNSUPPORT: cancel/ undo the SUPPORT transformation.

        - QUALITY_INSPECTION: The quality of the entity (matching with the entity_type) is inspected.

        ###
        io_behaviour: describe if a part is created, exist or destroyed (see below)

        Possible variables:
        - EXIST: Part is neither created nor destroyed by the process it has to exist before and still
        exists at the end
        - CREATED: Part is created in the process
        - DESTROYED: Part is destroyed at the end of the process
        (e.g. scrap bad quality, parts with no further tracking)
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.entity_type: EntityType = entity_type
        self.amount: int = amount
        self.quality: Optional[float] = quality
        self.reset_inspected_quality: bool = reset_inspected_quality

        if isinstance(transformation_type, EntityTransformationNode.TransformationTypes):
            self.transformation_type = transformation_type
        else:
            raise TypeError(f"[{self.__class__.__name__}] Transformation {transformation_type} type not valid")
        if isinstance(io_behaviour, EntityTransformationNode.IoBehaviours):
            self.io_behaviour = io_behaviour
        else:
            raise TypeError(f"[{self.__class__.__name__}] IO Behaviour {io_behaviour} type not valid")

        if parents is None:
            parents = []
        self.parents: list[EntityTransformationNode] = parents
        if children is None:
            children = []
        self.children: list[EntityTransformationNode] = children

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        return (f"EntityTransformationNode with ID '{self.identification}' and entity_type_name '{entity_type_name}'; "
                f"amount: '{self.amount}', quality: '{self.quality}', "
                f"reset_inspected_quality: '{self.reset_inspected_quality}'"
                f"transformation_type: '{self.transformation_type.name}', io_behaviour: '{self.io_behaviour.name}'")

    def copy(self):
        """Copy the object with the same identification."""
        entity_transformation_node_copy = super(EntityTransformationNode, self).copy()
        entity_transformation_node_copy.children = entity_transformation_node_copy.children.copy()
        entity_transformation_node_copy.parents = entity_transformation_node_copy.parents.copy()

        return entity_transformation_node_copy

    def get_entity_type_name(self):
        if self.entity_type is not None:
            return self.entity_type.name
        else:
            return ""

    def add_child(self, child_node, *further_children_nodes):
        """Add one or more child's to the children"""
        self.children.append(child_node)
        for further_children_node in further_children_nodes:
            self.children.append(further_children_node)

    def add_parent(self, parent_node, *further_parent_nodes):
        """Add one or more parent's to the parents"""
        self.parents.append(parent_node)
        for further_parent_node in further_parent_nodes:
            self.parents.append(further_parent_node)

    def entity_usable(self, entity: Entity) -> bool:
        """Check if the input is available"""
        return entity.entity_type.check_entity_type_match(self.entity_type)

    def amount_available(self, amount: int) -> bool:
        """Check if the amount is available"""
        return amount >= self.amount

    def quality_sufficient(self, entity: Entity) -> bool:
        """Check if the quality is sufficient to execute the transformation"""
        if self.quality is None:
            return True  # quality check isn't required
        elif entity.inspected_quality is None:
            return False # quality isn't inspected before
        else:
            return entity.inspected_quality >= self.quality # quality isn't enough

    def transformation_type_main_entity(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.MAIN_ENTITY

    def transformation_type_blank(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.BLANK

    def transformation_type_ingredient(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.INGREDIENT

    def transformation_type_sub_part(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.SUB_ENTITY

    def transformation_type_disassemble(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.DISASSEMBLE

    def transformation_type_support(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.SUPPORT

    def transformation_type_un_support(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.UNSUPPORT

    def transformation_type_quality_inspection(self) -> bool:
        return self.transformation_type == EntityTransformationNodeTransformationTypes.QUALITY_INSPECTION

    def compare_transformation_type_self(self, possible_transformation_types: list[TransformationTypes]) -> bool:
        """Check if the own transformation_type match to other transformation_types"""
        return self.transformation_type in possible_transformation_types

    def io_behaviour_exist(self) -> bool:
        return self.io_behaviour == EntityTransformationNodeIoBehaviours.EXIST

    def io_behaviour_created(self) -> bool:
        return self.io_behaviour == EntityTransformationNodeIoBehaviours.CREATED

    def io_behaviour_destroyed(self) -> bool:
        return self.io_behaviour == EntityTransformationNodeIoBehaviours.DESTROYED

    def compare_io_behaviour_self(self, io_behaviors: list[IoBehaviours]) -> bool:
        """Check if the own io_behaviour match to other io_behaviours"""
        return self.io_behaviour in io_behaviors

    def completely_filled(self):

        not_completely_filled_attributes = []
        if self.entity_type.__class__.__name__ not in {"EntityType", "PartType"}:
            not_completely_filled_attributes.append("entity_type")
        if not (isinstance(self.amount, int) or isinstance(self.amount, float)):
            not_completely_filled_attributes.append("amount")
        if not (isinstance(self.quality, float) or isinstance(self.quality, int) or self.quality is None):
            not_completely_filled_attributes.append("quality")
        if not isinstance(self.transformation_type, EntityTransformationNodeTransformationTypes):
            not_completely_filled_attributes.append("transformation_type")
        if not isinstance(self.io_behaviour, EntityTransformationNodeIoBehaviours):
            not_completely_filled_attributes.append("io_behaviour")
        if not isinstance(self.parents, list):
            not_completely_filled_attributes.append("parents")
            not_correctly_instantiated_attributes = [parent
                                                     for parent in self.parents
                                                     if not isinstance(parent, EntityTransformationNode)]
            if not_correctly_instantiated_attributes:
                not_completely_filled_attributes.append("parents")
        if not isinstance(self.children, list):
            not_completely_filled_attributes.append("children")
            not_correctly_instantiated_attributes = [child
                                                     for child in self.children
                                                     if not isinstance(child, EntityTransformationNode)]
            if not_correctly_instantiated_attributes:
                not_completely_filled_attributes.append("children")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class TransformationModel(DTModel):

    def __init__(self,
                 root_nodes: list[EntityTransformationNode | str],
                 is_re_trainable: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Describes the physical transformation of parts by a process as a graph. Root nodes are needed inputs. Leaf nodes
        are created outputs.

        Parameters
        ----------
        root_nodes: A list of entity_transformation_nodes that builds the first level of the transformation_model
        A list of strings is also recognized so that the root nodes can be loaded by the database. [id.1234']
        """
        DTModel.__init__(self, is_re_trainable=is_re_trainable,
                         identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        root_nodes_with_wrong_type = [root_node
                                      for root_node in root_nodes
                                      if not isinstance(root_node, EntityTransformationNode | str)]
        if root_nodes_with_wrong_type:
            error_description = [(root_node, type(root_node)) for root_node in root_nodes_with_wrong_type]
            raise TypeError(f"[{self.__class__.__name__}] The root_nodes have a wrong type: {error_description}. \n "
                            f"They should be EntityTransformationNode's")
        self._root_nodes: list[EntityTransformationNode] = root_nodes

    @property
    def root_nodes(self):
        return self._root_nodes

    @root_nodes.setter
    def root_nodes(self, root_nodes):
        self._root_nodes = root_nodes

    def copy(self):
        """Copy the object with the same identification."""
        transformation_model_copy: TransformationModel = super(TransformationModel, self).copy()
        transformation_model_copy.root_nodes = transformation_model_copy.root_nodes.copy()

        return transformation_model_copy

    def get_root_nodes(self, process_execution: ProcessExecution = None) -> list[EntityTransformationNode]:
        """Returns the root nodes of the transformation model."""
        return self._root_nodes
