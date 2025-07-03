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

This file contains the entrance point/ interface to all state model objects of a scenario
(e.g., virtual representation of a shop floor, simulation, etc.).

Classes:
    StateModel: The state model is a representation of the digital twin and serves as access point.
    The model can be emerged by processes executed (process execution) on the model.

@contact persons: Christian Schwede & Adrian Freiter
@last update: 16.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import inspect
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from functools import wraps, reduce
from operator import concat
from typing import TYPE_CHECKING, Union, Optional, Type
from types import NoneType
# Imports Part 2: PIP Imports
import dill as pickle
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import (DigitalTwinObject, DynamicAttributes, DynamicDigitalTwinObject,
                                                   ProcessExecutionTypes, prints_visible)
from ofact.twin.state_model.entities import (Plant, PhysicalBody, EntityType, PartType, Entity, Part, Resource,
                                             StationaryResource, Storage, WorkStation, Warehouse, ConveyorBelt,
                                             NonStationaryResource, ActiveMovingResource, PassiveMovingResource)
from ofact.twin.state_model.helpers.helpers import convert_lst_of_lst_to_lst, load_from_pickle, convert_to_datetime
from ofact.twin.state_model.probabilities import (ProbabilityDistribution, SingleValueDistribution,
                                                  BernoulliDistribution, NormalDistribution)
from ofact.twin.state_model.process_models import (
    _get_distance, ResourceGroup, EntityTransformationNode,
    ResourceModel, ProcessTimeModel, TransitionModel, QualityModel, TransformationModel,
    SimpleSingleValueDistributedProcessTimeModel, SimpleNormalDistributedProcessTimeModel,
    SimpleBernoulliDistributedQualityModel,
    EntityTransformationNodeIoBehaviours, EntityTransformationNodeTransformationTypes)
from ofact.twin.state_model.processes import (ResourceController, ProcessTimeController, TransitionController,
                                              QualityController, TransformationController, Process, ValueAddedProcess,
                                              ProcessExecution, WorkOrder, ProcessController)
from ofact.twin.state_model.sales import Customer, FeatureCluster, Feature, Order
from ofact.twin.state_model.time import WorkCalender, ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt

if TYPE_CHECKING:
    from pathlib import Path

    process_execution_types = Union[ProcessExecutionTypes.PLAN, ProcessExecutionTypes.ACTUAL]
    stationary_resources_types = Union[StationaryResource, Warehouse, Storage, WorkStation]
    non_stationary_resources_types = Union[NonStationaryResource | ActiveMovingResource | PassiveMovingResource]
    all_resources_types = stationary_resources_types | non_stationary_resources_types

EventTypes = ProcessExecutionTypes

enums_used = {"Types": ProcessExecutionTypes,
              "TransformationTypes": EntityTransformationNodeTransformationTypes,
              "IoBehaviours": EntityTransformationNodeIoBehaviours}

# constants
MEMOIZATION_MAX = 100


def memoize_get_stationary_resource_at_position(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_stationary_resources_by_entity_types(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_distance_matrix(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


process_execution_data_type = [("Executed Start Time", "datetime64[ns]"),
                               ("Executed End Time", "datetime64[ns]"),
                               ("Event Type", object),
                               ("Process Execution", object)]


def _transform_process_executions_list_to_np_array(process_executions: list[ProcessExecution]) -> \
        np.array([np.datetime64, np.datetime64, np.string_, ProcessExecution]):
    """
    Transform the process_executions from type list to numpy-array for better access ...

    Parameters
    ----------
    process_executions: a list of process_executions

    Returns
    -------
    process_executions_a: a numpy array with process_executions as fourth element
    (mapped to start_time, end_time, event_type)
    """
    process_executions_by_timestamps = [_get_process_execution_entry(process_execution)
                                        for process_execution in process_executions]
    process_executions_a = np.array(process_executions_by_timestamps,
                                    dtype=process_execution_data_type)

    return process_executions_a


def _get_process_execution_entry(process_execution: ProcessExecution) -> (np.datetime64, np.datetime64, str,
                                                                          ProcessExecution):
    """Set up a process_execution numpy entry for faster access"""
    process_execution_entry = (process_execution.executed_start_time,
                               process_execution.executed_end_time,
                               process_execution.event_type.name,
                               process_execution)
    return process_execution_entry


order_data_type = [("Order Date", "datetime64[ns]"),
                   ("Delivery Date Planned", "datetime64[ns]"),
                   ("Delivery Date Actual", "datetime64[ns]"),
                   ("Order", object)]


def _transform_order_pool_list_to_np_array(order_pool: list[Order]) -> (
        np.array([np.datetime64, np.datetime64, np.datetime64, Order])):
    """
    Transform the order_pool from type list to numpy array for better access ...

    Parameters
    ----------
    order_pool: a list of orders

    Returns
    -------
    order_pool_a: a numpy array with orders as the fourth element (mapped to order_date, delivery_date_planned,
    delivery_date_actual)
    """
    orders_by_timestamps = [_get_order_entry(order)
                            for order in order_pool]
    order_pool_a = np.array(orders_by_timestamps, dtype=order_data_type)

    return order_pool_a


def _get_order_entry(order: Order) -> (np.datetime64, np.datetime64, np.datetime64, Order):
    """Set up an order numpy array entry for faster access"""
    order_entry = (order.order_date if order.order_date else np.datetime64('NaT'),
                   order.delivery_date_planned if order.delivery_date_planned else np.datetime64('NaT'),
                   order.delivery_date_actual if order.delivery_date_actual else np.datetime64('NaT'),
                   order)
    return order_entry


def _set_up_objects_by_external_identification_cache(name_space: str, dt_objects: (
        Union[dict[object, DigitalTwinObject], list[DigitalTwinObject], np.array])):
    """Set up a cache for digital twin objects for faster access in the case of need"""
    if not isinstance(dt_objects, list):
        if isinstance(dt_objects, dict):
            dt_objects = convert_lst_of_lst_to_lst(list(dt_objects.values()))

        elif type(dt_objects).__module__ == np.__name__:  # np.array
            last_column_index: int = list(dt_objects.dtype.fields)[-1]
            # Note: only valid for the 'ProcessExecution' attribute
            dt_objects: list = dt_objects[last_column_index].tolist()

        elif isinstance(dt_objects, DigitalTwinObject):
            dt_objects = [dt_objects]

        else:
            raise NotImplementedError(dt_objects)

    objects_by_external_identification_cache = {}
    for dt_object in dt_objects:
        external_identifications_dt_object = dt_object.get_external_identifications_name_space(name_space=name_space)
        for external_id in external_identifications_dt_object:
            objects_by_external_identification_cache.setdefault(external_id,
                                                                []).append(dt_object)

    return objects_by_external_identification_cache


def _get_objects_by_external_identification(name_space, external_id, dt_objects: list[DigitalTwinObject]):
    """
    Iterate through the objects existing in the digital twin and check the match to the external_identification
    (related to the name_space)
     ToDo: it is not the best way to iterate for each object through the list
      Maybe for the data integration a alternative way should be evaluated
    """

    state_model_objects = \
        list({dt_object.get_self_by_external_id(name_space=name_space,
                                                external_id=external_id)
              for dt_object in dt_objects
              if dt_object.get_self_by_external_id(name_space=name_space,
                                                   external_id=external_id) is not None})

    return state_model_objects


def get_orders_in_progress(orders: list[Order], at: Optional[datetime] = None):
    """get the orders in progress (begun but not finished)"""
    orders_in_progress = [order
                          for order in orders
                          if _check_order_in_progress(order, at)]

    return orders_in_progress


def _check_order_in_progress(order: Order, at: Optional[datetime] = None):
    """
    Check if order is in progress (begun but not finished)
    Note: the function would be very simple, if the delivery_date_actual and the release_date are always set
    Because this is not always the case, some other conditions are introduced that support the determination if
    the order is in progress

    Parameters
    ----------

    order: sales order object
    at: for query times that are different from the current time of the digital twin

    Note: currently it is assumed that the time_stamp at is in the past
    """

    release_date = get_order_release_date(order)
    if release_date is None:
        return False
    elif at is not None:
        if at < release_date:
            return False

    delivery_date_actual = get_order_delivery_date_actual(order)
    if delivery_date_actual is None:
        return True
    elif at is not None:
        if at < delivery_date_actual:
            return True

    return False


def get_order_delivery_date_actual(order) -> Optional[datetime]:
    """
    Determine the delivery date actual
    Note: Since the date is not always set,
    it can also be derived from the last process execution if no features are requested anymore.
    """

    if order.delivery_date_actual is not None:
        return order.delivery_date_actual

    delivery_date_actual_from_process_executions = order.get_delivery_date_actual_from_process_executions()

    return delivery_date_actual_from_process_executions


def get_order_release_date(order) -> Optional[datetime]:
    if order.release_date_actual is not None:
        return order.release_date_actual

    release_date_from_process_executions = order.get_release_date_from_process_executions()

    return release_date_from_process_executions


state_model_classes: list[Type] = \
    [DigitalTwinObject, DynamicAttributes, DynamicDigitalTwinObject,
     Plant, PhysicalBody, EntityType, PartType, Entity, Part, Resource,
     StationaryResource, Storage, WorkStation, Warehouse, ConveyorBelt,
     NonStationaryResource, ActiveMovingResource, PassiveMovingResource,
     ResourceGroup, EntityTransformationNode,
     ResourceModel, ProcessTimeModel, TransitionModel, QualityModel, TransformationModel,
     ResourceController, ProcessTimeController, TransitionController, QualityController,
     TransformationController,
     Process, ValueAddedProcess, ProcessExecution, WorkOrder,
     WorkCalender, ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt,
     Customer, FeatureCluster, Feature, Order,
     ProbabilityDistribution, SingleValueDistribution, BernoulliDistribution, NormalDistribution]


def _filter_order_pool(start_date, end_date, order_pool_filtered, ignore_na_values, index):
    """Filter the order pool based on the time window restricted with start and end date"""

    order_dates_masked = np.ma.array(order_pool_filtered[index], mask=np.isnan(order_pool_filtered[index]))
    if start_date is not None and ignore_na_values:
        order_date_start_mask = ((order_dates_masked >= start_date) or
                                 (order_pool_filtered[index] != order_pool_filtered[index]))
    elif start_date is not None and not ignore_na_values:
        order_date_start_mask = order_dates_masked >= start_date
    else:
        order_date_start_mask = True

    if end_date is not None and ignore_na_values:
        order_date_end_mask = ((order_dates_masked <= end_date) or
                               (order_pool_filtered[index] != order_pool_filtered[index]))
    elif end_date is not None and not ignore_na_values:
        order_date_end_mask = order_dates_masked <= end_date
    else:
        order_date_end_mask = True

    if not isinstance(order_date_start_mask, bool) and not isinstance(order_date_end_mask, bool):
        order_mask = order_date_start_mask & order_date_end_mask
    elif isinstance(order_date_start_mask, bool) and not isinstance(order_date_end_mask, bool):
        order_mask = order_date_start_mask
    elif not isinstance(order_date_start_mask, bool) and isinstance(order_date_end_mask, bool):
        order_mask = order_date_end_mask
    else:
        return order_pool_filtered

    order_pool_filtered = order_pool_filtered[order_mask]

    return order_pool_filtered


class StateModel:
    verison = '1.0.0'
    # could be class attributes (but that could cause problems with dill/ pickle persistence ...)
    state_model_classes = state_model_classes

    @classmethod
    def _get_digital_twin_class(cls, digital_twin_class):
        if isinstance(digital_twin_class, str):
            try:
                digital_twin_class = eval(digital_twin_class)
            except NameError:
                digital_twin_class = None

        return digital_twin_class

    @classmethod
    def get_init_parameter_type_hints(cls, digital_twin_class: Union[str, Type]) -> Optional[dict[str, object]]:
        """
        Used to determine and return the parameter values of a digital twin class.

        Parameters
        ----------
        digital_twin_class: one class of the DigitalTwin.state_model_classes

        Returns
        -------
        new_dict: the init parameters type hints of the state twin
        """
        digital_twin_class = cls._get_digital_twin_class(digital_twin_class)

        if digital_twin_class == WorkCalender or digital_twin_class is None:
            return None

        new_dict = {}
        init_method_input_parameters = inspect.signature(digital_twin_class.__init__).parameters
        for param, hint in init_method_input_parameters.items():

            if param == 'self':
                continue

            if isinstance(hint, inspect.Parameter):
                if hint.annotation not in enums_used:
                    if isinstance(hint.annotation, str):
                        hint_annotation = eval(str(hint.annotation))
                    else:
                        hint_annotation = hint.annotation
                else:
                    hint_annotation = enums_used[hint.annotation]
            elif isinstance(hint, str):
                hint_annotation = eval(hint)
            else:
                hint_annotation = hint.annotation

            new_dict[param] = hint_annotation

        return new_dict

    @classmethod
    def get_init_parameter_default_values(cls, digital_twin_class: Union[str, Type]) -> Optional[dict[str, object]]:
        """
        Used to determine and return the parameter values of a digital twin class with default values.

        Parameters
        ----------
        digital_twin_class: one class of the DigitalTwin.state_model_classes

        Returns
        -------
        the default values of the init parameters of the digital twin
        """
        digital_twin_class = cls._get_digital_twin_class(digital_twin_class)

        if digital_twin_class == WorkCalender or digital_twin_class is None:
            return None

        new_dict = {}
        init_method_input_parameters = inspect.signature(digital_twin_class.__init__).parameters
        for param, hint in init_method_input_parameters.items():

            if param == 'self':
                continue

            if not isinstance(hint, inspect.Parameter):
                continue

            if isinstance(hint.default, (dict, list)):
                default_value = eval(str(hint.default)) if isinstance(hint.default, str) else hint.default
            elif hint.default not in enums_used:
                default_value = eval(str(hint.default)) if isinstance(hint.default, str) else hint.default
            else:
                # enum
                default_value = enums_used[hint.default]

            if default_value != inspect.Parameter.empty:
                new_dict[param] = default_value

        return new_dict

    @staticmethod
    def prepare_processes(process_list) -> dict[type[Union[Process], ValueAddedProcess], list[Process]]:
        """
        Prepare the processes for the state model
        """
        processes = {Process: [],
                     ValueAddedProcess: []}
        for process in process_list:
            processes[process.__class__].append(process)

        return processes

    # used for the frontend to know which probability distributions available in the digital twin
    process_time_model_classes: list = [SimpleSingleValueDistributedProcessTimeModel,
                                        SimpleNormalDistributedProcessTimeModel]
    quality_model_classes: list = [SimpleBernoulliDistributedQualityModel]

    drop_before_serialization = ['_objects_by_external_identification',
                                 'digital_twin_class_mapper',
                                 'dt_objects_directory']

    def __init__(self,
                 entity_types: list[Union[EntityType, PartType]],
                 plant: Optional[Plant],
                 parts: dict[EntityType: list[Part]],
                 obstacles: list[Resource],
                 stationary_resources: dict[EntityType: list[StationaryResource]],
                 passive_moving_resources: dict[EntityType: list[PassiveMovingResource]],
                 active_moving_resources: dict[EntityType: list[ActiveMovingResource]],
                 entity_transformation_nodes: list[EntityTransformationNode],
                 processes: dict[Union[Process, ValueAddedProcess]: list[Process]],
                 process_executions: list[ProcessExecution],
                 order_pool: list[Order],
                 customer_base: list[Customer],
                 features: list[Feature],
                 feature_clusters: dict[EntityType: list[FeatureCluster]],
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        """
        The state model holds all objects of one scenario. A scenario could be the representation of the physical world
        via data integration or virtual simulation.
        Hybrid scenarios that contain objects from the physical world as well as simulated objects are also possible.
        They are especially relevant if data gaps need to be fixed.
        In general, the state model serves as standard interface to all the state model objects.

        Parameters
        ----------
        name: scenario name / use case name / source application
        description: describes for example parameters chosen
        entity_types: Types that specifies parts and resources (entities)
        plant: the plant contains resources
        parts: entities that are processed
        obstacles: resources, that take up space
        stationary_resources: resources that are used to process the parts
        active_moving_resources: resources that can transport the parts
        entity_transformation_nodes: transformation_nodes that describes the transformation executed by a process
        processes: processes that can be executed within the state model
        process_executions: PLAN and ACTUAL executions of processes
        order_pool: orders that should be achieved
        customer_base: a list of all customers
        features: a list of possible features
        feature_clusters: a mapping of feature_clusters to product_classes (entity_types)
        attribute state_model_classes: A list of all state model classes that built the digital twin model
        if instantiated
        """

        # attributes for the digital twin identification
        self.name: Optional[str] = name
        self.description: Optional[str] = description

        self.plant: Optional[Plant] = plant

        # entities
        self.entity_types: list[Union[EntityType, PartType]] = entity_types
        self.parts: dict[EntityType: list[Part]] = parts
        self.obstacles: list[Resource] = obstacles
        self.stationary_resources: dict[EntityType: list[StationaryResource]] = stationary_resources
        self.passive_moving_resources: dict[EntityType: list[PassiveMovingResource]] = passive_moving_resources
        self.active_moving_resources: dict[EntityType: list[ActiveMovingResource]] = active_moving_resources

        # processes
        self.processes: dict[Union[Process, ValueAddedProcess]: list[Process]] = processes

        process_executions_a = _transform_process_executions_list_to_np_array(process_executions)
        self.process_executions: np.array([np.datetime64, np.datetime64, np.string_, ProcessExecution]) = (
            process_executions_a)

        # sales
        order_pool_a = _transform_order_pool_list_to_np_array(order_pool)
        self.order_pool: np.array([np.datetime64, np.datetime64, np.datetime64, Order]) = order_pool_a
        self.customer_base: list[Customer] = customer_base
        self.features: list[Feature] = features
        self.feature_clusters: dict[Union[EntityType, PartType]: list[FeatureCluster]] = feature_clusters

        self.state_model_class_mapper: dict[str, object] = {class_.__name__: class_
                                                            for class_ in type(self).state_model_classes}

        # caching for performance issues
        self._objects_by_external_identification: dict = {}

        # derivable attributes
        self.processes_by_main_parts: dict[EntityType: list[Process]] = {}
        self.processes_by_main_resource: dict[EntityType: list[Process]] = {}

        self.physical_bodies: list[PhysicalBody] = []
        self.process_executions_plans: list[Union[ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt]] = []
        self.resource_groups: list[ResourceGroup] = []
        # ToDo: Always Required ??
        self.entity_transformation_nodes: list[EntityTransformationNode] = entity_transformation_nodes
        self.process_controllers: list[ProcessController] = []
        self.process_models: dict[object, list] = {}

        # objects by class name string
        resources = self.stationary_resources | self.passive_moving_resources | self.active_moving_resources
        # note: assuming that the different resource types did not have the same entity types
        self.dt_objects_directory: dict[str: object] = \
            {"EntityType": self.entity_types,
             "PartType": self.entity_types,
             "Customer": self.customer_base,
             "Order": self.order_pool,
             "Feature": self.features,
             "FeatureCluster": self.feature_clusters,
             "Resource": resources,
             "ProcessExecutionPlan": self.process_executions_plans,
             "Plant": self.plant,
             "NonStationaryResource": self.active_moving_resources | self.passive_moving_resources,
             "PassiveMovingResource": self.passive_moving_resources,
             "ActiveMovingResource": self.active_moving_resources,
             "StationaryResource": self.stationary_resources,
             "ConveyorBelt": self.stationary_resources,
             "WorkStation": self.stationary_resources,
             "Warehouse": self.stationary_resources,
             "Storage": self.stationary_resources,
             "Part": self.parts,
             "ProcessExecution": self.process_executions,
             "Process": self.processes,
             "ValueAddedProcess": self.processes}

    def duplicate(self):
        """
        Create a duplicate of the state model with the same objects and content/
        identifications of objects but diverging python objects
        """
        if prints_visible:
            print(f"[{self.__class__.__name__:20}] Testing needed")

        state_model = self
        new_state_model = {"entity_types": state_model.entity_types,
                           "plant": state_model.plant,
                           "parts": state_model.parts,
                           "obstacles": state_model.obstacles,
                           "stationary_resources": state_model.stationary_resources,
                           "passive_moving_resources": state_model.passive_moving_resources,
                           "active_moving_resources": state_model.active_moving_resources,
                           "entity_transformation_nodes": state_model.entity_transformation_nodes,
                           "processes_by_main_parts": state_model.processes_by_main_parts,
                           "processes_by_main_resource": state_model.processes_by_main_resource,
                           "processes": state_model.processes,
                           "process_executions": state_model.get_process_executions_list(),
                           "order_pool": state_model.get_orders(),
                           "customer_base": state_model.customer_base,
                           "features": state_model.features,
                           "feature_clusters": state_model.feature_clusters,
                           "name": state_model.name,
                           "description": state_model.description}
        state_model_duplicated_dict = deepcopy(new_state_model)

        state_model_duplicated = StateModel(**state_model_duplicated_dict)
        return state_model_duplicated

    def update_initial(self):
        process_controllers = self.get_all_process_controllers()
        for process_controller in process_controllers:
            process_controller.update_initial()

    def get_consideration_period(self) -> [datetime, datetime]:
        """Return the consideration period of the ..."""
        first_execution_np = self.process_executions["Executed Start Time"].min()  # ToDo: datetime?
        last_execution_np = self.process_executions["Executed End Time"].max()

        first_execution = convert_to_datetime(first_execution_np)
        last_execution = convert_to_datetime(last_execution_np)

        return first_execution, last_execution

    # #### GET DIGITAL TWIN OBJECTS ####################################################################################

    def get_all_resources_by_entity_types(self, entity_types: list[EntityType]) -> (
            dict[EntityType: list[all_resources_types]]):
        """Combine all resources to a dict"""
        stationary_resources = self.get_stationary_resources_by_entity_types(entity_types)
        passive_moving_resources = self.get_passive_moving_resources_by_entity_types(entity_types)
        active_moving_resources = self.get_active_moving_resources_by_entity_types(entity_types)

        entities = {entity_type: (stationary_resources[entity_type] + passive_moving_resources[entity_type] +
                                  active_moving_resources[entity_type])
                    for entity_type in entity_types}

        return entities

    def get_entities_by_entity_types(self, entity_types: list[EntityType]) -> (
            dict[EntityType: list[all_resources_types | Part]]):

        all_resources = self.get_all_resources_by_entity_types(entity_types)
        parts = self.get_parts_by_entity_types(entity_types)

        entities = {entity_type: all_resources[entity_type] + parts[entity_type]
                    for entity_type in entity_types}

        return entities

    def get_plant(self) -> Plant:
        """Return the plant object"""
        return self.plant

    def get_process_time_model_classes(self) -> list:
        process_time_model_classes = type(self).process_time_model_classes
        return process_time_model_classes

    def get_quality_model_classes(self) -> list:
        quality_model_classes = type(self).quality_model_classes
        return quality_model_classes

    # entity_types

    def get_entity_types(self) -> list[Union[EntityType, PartType]]:
        entity_types = self.entity_types

        return entity_types

    def get_part_types(self) -> list[PartType]:
        part_types = [entity_type
                      for entity_type in self.entity_types
                      if isinstance(entity_type, PartType)]

        return part_types

    # resources
    def get_all_resources(self) -> list[all_resources_types]:
        """Extract/ return all resources from the state model"""
        stationary_resources = self.get_stationary_resources()
        non_stationary_resources = self.get_non_stationary_resources()
        all_resources = stationary_resources + non_stationary_resources

        return all_resources

    def get_stationary_resources(self) -> list[StationaryResource]:
        """Extract/ return the stationary_resources from the state model"""
        obstacle_resources = self.obstacles  # ToDo

        stationary_resources_nested = self.stationary_resources.values()
        if stationary_resources_nested:
            stationary_resources = reduce(concat, stationary_resources_nested)
        else:
            stationary_resources = []

        stationary_resources.extend(obstacle_resources)

        return stationary_resources

    @memoize_get_stationary_resources_by_entity_types
    def get_stationary_resources_by_entity_types(self, entity_types):
        """Return stationary resources that have the entity_type"""
        return self.get_entities_of_type_by_entity_types("StationaryResource", entity_types)

    def get_passive_moving_resources_by_entity_types(self, entity_types):
        """Return passive_moving_resources that have the entity_type"""
        return self.get_entities_of_type_by_entity_types("PassiveMovingResource", entity_types)

    def get_active_moving_resources_by_entity_types(self, entity_types):
        """Return active_moving_resources that have the entity_type"""
        return self.get_entities_of_type_by_entity_types("ActiveMovingResource", entity_types)

    def get_parts_by_entity_types(self, entity_types):
        """Return parts that have the entity_type"""
        return self.get_entities_of_type_by_entity_types("Part", entity_types)

    def get_entities_of_type_by_entity_types(self, type_, entity_types):
        """Return entities that have the entity_type"""

        entities_dict = self.dt_objects_directory[type_]
        if type_ == "StationaryResource":
            entities = self.get_stationary_resources()
        elif type_ == "PassiveMovingResource":
            entities = self.get_passive_moving_resources()
        elif type_ == "ActiveMovingResource":
            entities = self.get_active_moving_resources()
        elif type_ == "Part":
            entities = self.get_parts()
        else:
            raise NotImplementedError

        entities_by_entity_types = {}
        for entity_type in entity_types:
            entities_by_entity_type = []
            if entity_type in entities_dict:
                entities_by_entity_type += entities_dict[entity_type]

            entities_by_entity_type += [entity for entity in entities
                                        if entity.entity_type.super_entity_type == entity_type]
            entities_by_entity_types[entity_type] = entities_by_entity_type

        return entities_by_entity_types

    def get_work_stations(self) -> list[WorkStation]:
        """Return all work_stations of the state model"""
        stationary_resources = self.get_stationary_resources()
        work_stations = list(set([stationary_resource
                                  for stationary_resource in stationary_resources
                                  if isinstance(stationary_resource, WorkStation)]))

        return work_stations

    def get_conveyor_belts(self) -> list[ConveyorBelt]:
        """Return all conveyor_belts of the state model"""
        stationary_resources = self.get_stationary_resources()
        conveyor_belts = list(set([stationary_resource
                                   for stationary_resource in stationary_resources
                                   if isinstance(stationary_resource, ConveyorBelt)]))

        return conveyor_belts

    def get_warehouses(self) -> list[Warehouse]:
        """Return all warehouses of the state model"""
        stationary_resources = self.get_stationary_resources()
        warehouses = list(set([stationary_resource
                               for stationary_resource in stationary_resources
                               if isinstance(stationary_resource, Warehouse)]))

        return warehouses

    def get_storages(self) -> list[Storage]:
        """Return all storages of the state model"""
        stationary_resources = self.get_stationary_resources()
        storages = list(set([stationary_resource
                             for stationary_resource in stationary_resources
                             if isinstance(stationary_resource, Storage)]))

        return storages

    def get_non_stationary_resources(self) -> list[non_stationary_resources_types]:
        """Extract/ return the non_stationary_resources from the state model"""
        passive_moving_resources = self.get_passive_moving_resources()
        active_moving_resources = self.get_active_moving_resources()
        non_stationary_resources = active_moving_resources + passive_moving_resources

        return non_stationary_resources

    def get_passive_moving_resources(self) -> list[PassiveMovingResource]:
        """Return all passive_moving_resources of the state model"""
        passive_moving_resources_nested = self.passive_moving_resources.values()
        if passive_moving_resources_nested:
            passive_moving_resources = reduce(concat, passive_moving_resources_nested)
        else:
            passive_moving_resources = []

        return passive_moving_resources

    def get_active_moving_resources(self) -> list[ActiveMovingResource]:
        """Return all active_moving_resources of the state model"""
        active_moving_resources_nested = self.active_moving_resources.values()
        if active_moving_resources_nested:
            active_moving_resources = reduce(concat, active_moving_resources_nested)
        else:
            active_moving_resources = []

        return active_moving_resources

    # parts

    def get_parts(self) -> list[Part]:
        parts_nested = self.parts.values()
        if parts_nested:
            parts = list(set(reduce(concat, parts_nested)))
        else:
            parts = []

        return parts

    def get_entities(self):

        parts = self.get_parts()
        resources = self.get_all_resources()

        entities = parts + resources
        return entities

    def get_physical_bodies(self):
        all_resources = self.get_all_resources()

        physical_bodies = [resource.physical_body
                           for resource in all_resources
                           if hasattr(resource, "_physical_body")]
        return physical_bodies

    # time

    def get_process_executions_plans(self, cache: bool = True) -> (
            list[Union[ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt]]):
        """
        Return all process_executions_plans of the state model

        Parameters
        ----------
        cache: if True, the calculations are cached in the attribute "process_executions_plans"
        """

        if self.process_executions_plans:
            # cached - ensure that changes are considered in the meantime ...
            return self.process_executions_plans

        all_resources = self.get_all_resources()
        process_executions_plans = list(set([resource.process_execution_plan
                                             for resource in all_resources
                                             if resource.process_execution_plan is not None]))
        if cache:
            self.process_executions_plans = process_executions_plans

        return process_executions_plans

    # processes

    def get_all_processes(self) -> list[Union[Process, ValueAddedProcess]]:
        """Return all processes stored in the state model - Including processes and value added processes"""
        processes_nested = self.processes.values()
        if processes_nested:
            all_processes = reduce(concat, processes_nested)
        else:
            all_processes = []

        return all_processes

    def get_process_by_main_resources(self):
        # processes by main resources
        processes_by_main_resources = {}
        processes = self.get_all_processes()
        for process in processes:
            single_resource_group = process.get_possible_resource_groups([], None)[0]
            main_resource = single_resource_group.main_resources[0]

            processes_by_main_resources.setdefault(main_resource,
                                                   []).append(process)
        return processes_by_main_resources

    def get_processes_by_main_parts(self):
        part_ets = [etn.entity_type
                    for etn in self.entity_transformation_nodes
                    if etn.compare_transformation_type_self([EntityTransformationNode.TransformationTypes.MAIN_ENTITY])]
        processes = self.get_all_processes()

        processes_by_main_parts = {part_et: []
                                   for part_et in part_ets}

        for process in processes:
            for etn in process.transformation_controller.get_root_nodes():
                if etn.entity_type in processes_by_main_parts:
                    processes_by_main_parts[etn.entity_type].append(process)
                    break
                elif etn.entity_type is not None:
                    if etn.entity_type.super_entity_type in processes_by_main_parts:
                        processes_by_main_parts[etn.entity_type.super_entity_type].append(process)
                        break

        return processes_by_main_parts

    def get_value_added_processes(self) -> list[ValueAddedProcess]:
        if ValueAddedProcess in self.processes:
            value_added_processes = self.processes[ValueAddedProcess]
        else:
            value_added_processes = []
        return value_added_processes

    def get_processes(self) -> list[Process]:
        if Process in self.processes:
            processes = self.processes[Process]
        else:
            processes = []
        return processes

    def get_all_process_controllers(self) -> list:

        all_processes = self.get_all_processes()

        all_process_controllers = list((set(process_controller
                                            for process in all_processes
                                            for process_controller in process.get_all_controllers())))
        return all_process_controllers

    def get_process_executions_list(self, event_type: Optional[process_execution_types] = None,
                                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> (
            list[ProcessExecution]):
        """
        Determine and return the process_executions according the chosen filter options

        Parameters
        ----------
        event_type: the event_type of a process_execution can be 'ACTUAL' or 'PLAN'.
        If set (unequal None), only the process_executions of the respective event_type are used.
        start_time: accept only process_executions that are not finished before the start_time
        end_time: accept only process_executions that are begun before the end_time

        Returns
        -------
        process_executions: a list of all process_executions existing in the digital twin
        """

        # filter according event_type
        if event_type is not None:
            process_executions: np.array = (
                self.process_executions[self.process_executions["Event Type"] == event_type.name])
        else:
            process_executions: np.array = self.process_executions

        # filter according start and end time
        if start_time is not None:
            frame_mask = (process_executions["Executed End Time"] >= np.datetime64(start_time, "ns"))
        else:
            frame_mask = None

        if end_time is not None:
            end_mask = (process_executions["Executed Start Time"] <= np.datetime64(end_time, "ns"))
            if frame_mask is not None:
                frame_mask = frame_mask & end_mask
        else:
            frame_mask = None

        if frame_mask is not None:
            process_executions = process_executions[frame_mask]
        if len(process_executions.shape) == 2:
            print("Squeezing needed ...")
            process_executions = process_executions.squeeze()
        process_executions_list = process_executions["Process Execution"].tolist()

        return process_executions_list

    def get_process_executions_list_for_main_resource(self, event_type, resource_as_main_resource,
                                                      start_time: Optional[datetime] = None,
                                                      end_time: Optional[datetime] = None) -> list[ProcessExecution]:
        """
        Get a list of process executions for which have the resource_as_main_resource as main_resource.

        Parameters
        ----------
        event_type: The type of event associated with the process executions.
        resource_as_main_resource: The main resource to filter process executions by.
        start_time: The start time to filter process executions by. Defaults to None.
        end_time: The end time to filter process executions by. Defaults to None.

        Returns
        -------
        A list of `ProcessExecution` objects with the main resource matching the `resource_as_main_resource` parameter.
        """
        process_execution_list = self.get_process_executions_list(event_type=event_type, start_time=start_time,
                                                                  end_time=end_time)

        process_executions_with_resource_as_main_resource = \
            [process_execution
             for process_execution in process_execution_list
             if process_execution.main_resource == resource_as_main_resource]

        return process_executions_with_resource_as_main_resource

    def get_process_executions_list_for_resource(self, event_type, resource, start_time: Optional[datetime] = None,
                                                 end_time: Optional[datetime] = None) -> list[ProcessExecution]:
        """
        Get a list of process executions for which have the resource in resources_used.

        Parameters
        ----------
        event_type: The type of event associated with the process executions.
        resource: The resource to filter process executions by.
        start_time: The start time to filter process executions by. Defaults to None.
        end_time: The end time to filter process executions by. Defaults to None.

        Returns
        -------
        process_executions_resource_participate_in: A list of `ProcessExecution` objects with the main resource matching
        the `resource_as_main_resource` parameter.
        """
        process_execution_list = self.get_process_executions_list(event_type=event_type, start_time=start_time,
                                                                  end_time=end_time)

        process_executions_resource_participate_in = [process_execution
                                                      for process_execution in process_execution_list
                                                      if resource in process_execution.get_resources()]

        return process_executions_resource_participate_in

    def get_resource_models(self) -> list[ResourceModel]:
        all_processes = self.get_all_processes()
        resource_models = [process.resource_controller.resource_model
                           for process in all_processes]

        return resource_models

    def get_time_models(self, cache: bool = False) -> list[ProcessTimeModel]:
        """
        Return all process time models of the state model

        Parameters
        ----------
        cache: if True, the derivations are cached in the attribute "self.process_models[ProcessTimeModel]"
        """

        if ProcessTimeModel in self.process_models:
            # cached - ensure that changes are considered in the meantime ...
            return self.process_models[ProcessTimeModel]

        all_processes = self.get_all_processes()
        time_models = list(set([process.lead_time_controller.process_time_model
                                for process in all_processes]))

        if cache:
            self.process_models[ProcessTimeModel] = time_models

        return time_models

    def get_transition_models(self, cache: bool = False) -> list[TransitionModel]:
        """
        Return all transition models of the state model

        Parameters
        ----------
        cache: if True, the derivations are cached in the attribute "self.process_models[TransitionModel]"
        """

        if TransitionModel in self.process_models:
            # cached - ensure that changes are considered in the meantime ...
            return self.process_models[TransitionModel]

        all_processes = self.get_all_processes()
        transition_models = list(set([process.transition_controller.transition_model
                                      for process in all_processes]))

        if cache:
            self.process_models[TransitionModel] = transition_models

        return transition_models

    def get_quality_models(self, cache: bool = False) -> list[QualityModel]:
        """
        Return all quality models of the state model

        Parameters
        ----------
        cache: if True, the derivations are cached in the attribute "self.process_models[QualityModel]"
        """

        if QualityModel in self.process_models:
            # cached - ensure that changes are considered in the meantime ...
            return self.process_models[QualityModel]

        all_processes = self.get_all_processes()
        quality_models: list[QualityModel] = list(set([process.quality_controller.quality_model
                                                       for process in all_processes]))

        if cache:
            self.process_models[QualityModel] = quality_models

        return quality_models

    def get_transformation_models(self, cache: bool = False) -> list[TransformationModel]:
        """
        Return all transformation models of the state model

        Parameters
        ----------
        cache: if True, the derivations are cached in the attribute "self.process_models[TransformationModel]"
        """

        if TransformationModel in self.process_models:
            # cached - ensure that changes are considered in the meantime ...
            return self.process_models[TransformationModel]

        all_processes = self.get_all_processes()
        transformation_models = list(set([process.transformation_controller.transformation_model
                                          for process in all_processes]))

        if cache:
            self.process_models[TransformationModel] = transformation_models

        return transformation_models

    def get_transformation_types(self) -> list:
        return EntityTransformationNode.get_transformation_types()

    def get_io_behaviours(self) -> list:
        return EntityTransformationNode.get_io_behaviours()

    def get_entity_transformation_nodes(self) -> list[EntityTransformationNode]:
        entity_transformation_nodes = self.entity_transformation_nodes
        return entity_transformation_nodes

    def get_orders(self, start_date=None, end_date=None, ignore_na_values=True,
                   consider_order_date=False, consider_release_date=False, consider_delivery_date_requested=False,
                   consider_delivery_date_planned=False, consider_delivery_date_actual=False) -> (
            list[Order]):
        """
        Return a list of orders.
        The orders can be filtered regarding their attributes.
        Note: All of the parameters given should be inside the start_date and end_date frame
        """

        if start_date is None and end_date is None:
            order_a = self.order_pool["Order"]
            orders = order_a.tolist()
            return orders

        order_pool_filtered = self.order_pool
        if consider_order_date:
            order_pool_filtered = _filter_order_pool(start_date, end_date, order_pool_filtered, ignore_na_values,
                                                     index="Order Date")

        if consider_delivery_date_planned:
            order_pool_filtered = _filter_order_pool(start_date, end_date, order_pool_filtered, ignore_na_values,
                                                     index="Delivery Date Planned")

        if consider_delivery_date_actual:
            order_pool_filtered = _filter_order_pool(start_date, end_date, order_pool_filtered, ignore_na_values,
                                                     index="Delivery Date Actual")

        orders_filtered = order_pool_filtered["Order"].tolist()

        if consider_delivery_date_requested:
            if start_date is not None and end_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.delivery_date_requested is not None
                                   if (order.delivery_date_requested >= start_date) and
                                   (order.delivery_date_requested <= end_date)]
            elif start_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.delivery_date_requested is not None
                                   if order.delivery_date_requested >= start_date]
            elif end_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.delivery_date_requested is not None
                                   if order.delivery_date_requested <= end_date]

        if consider_release_date:
            if start_date is not None and end_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.release_date_actual is not None
                                   if (order.release_date_actual >= start_date) and
                                   (order.release_date_actual <= end_date)]
            elif start_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.release_date_actual is not None
                                   if order.release_date_actual >= start_date]
            elif end_date is not None:
                orders_filtered = [order
                                   for order in orders_filtered
                                   if order.release_date_actual is not None
                                   if order.release_date_actual <= end_date]

        return orders_filtered

    def get_features(self) -> list[Feature]:
        features = self.features
        return features

    def get_feature_clusters(self) -> list[FeatureCluster]:
        features_clusters_nested = list(self.feature_clusters.values())
        if features_clusters_nested:
            features_clusters = reduce(concat, features_clusters_nested)
        else:
            features_clusters = []
        return features_clusters

    def get_customers(self):
        customers = self.customer_base
        return customers

    # #### GET DIGITAL TWIN OBJECTS BY ID/ ... #########################################################################

    def get_object_by_id(self, id_: Optional[int] = None) -> Optional[int]:
        """
        The method gets an id_ and searches in the state model for the object (counterpart)
        with the same identifier.

        Parameters
        ----------
        id_: DigitalTwinObject identification (every object of the digital twin has an id)

        Returns
        -------
        object_: the requested object
        """
        if type(id_) != int:
            try:
                id_ = int(id_)
            except ValueError:
                if prints_visible:
                    print(f"[{self.__class__.__name__:20}] Could not convert id to an integer.")
                return None

        all_objects = []
        # iterate through all attributes of the instance and create a single list of lists
        for attribute in list(self.__dict__.values()):
            if type(attribute) == dict:
                if list(attribute.values()) == list:
                    for lst in list(attribute.values()):
                        all_objects += list(lst.values())
                else:
                    all_objects += list(attribute.values())
            elif type(attribute) == list:
                all_objects.append(list(attribute))
            elif type(attribute) is None:
                pass
            else:
                all_objects.append([attribute])

        # flatten the list of lists
        all_objects_list = [object_ for lst in all_objects for object_ in lst if object_]

        # find the object with the appropriate object id
        for object_ in all_objects_list:
            if not hasattr(object_, "identification"):
                pass
            elif object_.identification == id_:
                if prints_visible:
                    print(f"[{self.__class__.__name__:20}] "
                          f"The object with the Object-ID '{object_.identification}' has been found!\n")
                return object_

        if prints_visible:
            print(f"[{self.__class__.__name__:20}] There is no object with the id '{id_}' stored in the state model.")
        return None

    def get_object_by_external_identification(self, name_space: str, external_id: str,
                                              class_name: Optional[str] = None,
                                              from_cache: bool = False) -> list[DigitalTwinObject]:
        """
        Get the state model object by external identification

        Parameters
        ----------
        name_space: namespace of the external id
        external_id: identification from the environment associated with the object
        class_name: type of the object as string
        from_cache: in some cases, e.g. in the data_integration, lots of objects are requested,
        and it is assumed that the objects lists do not change at this time, respectively, the changes can be neglected.
        In this case, the data is cached with the purpose of faster access ...
        """
        if class_name is None:
            return []

        if not hasattr(self, "_objects_by_external_identification"):
            self._objects_by_external_identification = {}

        if from_cache:
            if name_space not in self._objects_by_external_identification:
                # set up the dict for the name_space
                self._objects_by_external_identification[name_space] = {}

            if class_name not in self._objects_by_external_identification[name_space]:
                # set up the dict for the object_type
                if class_name in self.dt_objects_directory:
                    if class_name == "ProcessExecutionPlan":
                        self.get_process_executions_plans()  # set them up initially
                        self.dt_objects_directory[class_name] = self.process_executions_plans

                    self._objects_by_external_identification[name_space][class_name] = (
                        _set_up_objects_by_external_identification_cache(
                            name_space=name_space, dt_objects=self.dt_objects_directory[class_name]))
                else:
                    raise NotImplementedError(f"The class_name: {class_name} with external_id: {external_id}")

            if class_name == "Part":  # ToDo: should be replaced!
                external_id = "_".join(external_id.split("_")[:-2]) + " " + external_id.split("_")[-2]

            if external_id in self._objects_by_external_identification[name_space][class_name]:

                return self._objects_by_external_identification[name_space][class_name][external_id]
            else:
                return []

        available_objects = self.get_objects_by_class_name(class_name)
        state_model_objects = \
            _get_objects_by_external_identification(name_space=name_space,
                                                    external_id=external_id,
                                                    dt_objects=available_objects)

        return state_model_objects

    def get_objects_by_class_name(self, class_name: str) -> list:
        """Get digital twin objects associated with the class_name as string"""

        if class_name in self.dt_objects_directory:
            available_objects = self.dt_objects_directory[class_name]

            if not isinstance(available_objects, list):
                if isinstance(available_objects, dict):
                    available_objects = convert_lst_of_lst_to_lst(list(available_objects.values()))
                else:
                    raise NotImplementedError

            available_objects = [available_object
                                 for available_object in available_objects
                                 if available_object.__class__.__name__ == class_name]

        else:
            raise NotImplementedError(class_name)

        return available_objects

    def get_object_attributes(self, object_class_name) -> Optional[dict]:
        """Get the attributes that are needed for the state model object instantiation"""
        class_ = self.state_model_class_mapper[object_class_name]

        return deepcopy(class_.__init__.__annotations__)

    def get_class_by_class_name(self, class_name: str):
        """Get the class (for example, to instantiate the class/ create an object) by class_name (str)"""

        class_ = self.state_model_class_mapper[class_name]
        return class_

    # #### HANDLE (NESTED-) OBJECTS DERIVABLE FROM OTHER OBJECTS #######################################################

    def set_implicit_objects_explicit(self):
        self.physical_bodies = self.get_physical_bodies()

        process_executions_plans = self.get_process_executions_plans()
        process_executions_plans_cb = [pep
                                       for pep in process_executions_plans
                                       if isinstance(pep, ProcessExecutionPlanConveyorBelt)]
        process_executions_plans = list(set(process_executions_plans).difference(set(process_executions_plans_cb)))
        self.process_executions_plans = process_executions_plans + process_executions_plans_cb  # sorted

        process_controllers = self.get_all_process_controllers()

        process_time_controllers = [process_controller
                                    for process_controller in process_controllers
                                    if isinstance(process_controller.get_model(), ProcessTimeModel)]
        quality_controllers = [process_controller
                               for process_controller in process_controllers
                               if isinstance(process_controller.get_model(), QualityModel)]
        transition_controllers = [process_controller
                                  for process_controller in process_controllers
                                  if isinstance(process_controller.get_model(), TransitionModel)]
        transformation_controllers = [process_controller
                                      for process_controller in process_controllers
                                      if isinstance(process_controller.get_model(), TransformationModel)]
        resource_controllers = [process_controller
                                for process_controller in process_controllers
                                if isinstance(process_controller.get_model(), ResourceModel)]

        resource_models = []
        self.resource_groups = []
        for process_controller in resource_controllers:
            process_model = process_controller.get_model()
            if not isinstance(process_model, ResourceModel):
                continue

            resource_models.append(process_model)
            self.resource_groups.extend(process_model.resource_groups)

        self.process_controllers = (process_time_controllers + quality_controllers + transition_controllers +
                                    transformation_controllers + resource_controllers)
        self.process_models = {ProcessTimeModel: [ptc.get_model() for ptc in process_time_controllers],
                               QualityModel: [qc.get_model() for qc in quality_controllers],
                               TransitionModel: [tsc.get_model() for tsc in transition_controllers],
                               TransformationModel: [tfc.get_model() for tfc in transformation_controllers],
                               ResourceModel: resource_models}

    def delete_explicit_objects(self):
        self.physical_bodies = []
        self.process_executions_plans = []
        self.resource_groups = []
        self.process_controllers = []
        self.process_models = {}

    # #### SET DIGITAL TWIN OBJECTS ####################################################################################

    def set_plant_layout(self, name: str, corners: list):
        self.plant.name = name
        self.plant.corners = corners

    # #### ADD DIGITAL TWIN OBJECTS ####################################################################################

    def add_plant(self, plant):
        if isinstance(plant, Plant) or plant is None:
            self.plant = plant

    def add_entity_type(self, entity_type: EntityType):
        """
        Adds an entity type to the list of entity types.

        Parameters
        ----------
        entity_type: The entity type to be added.
        """
        if not isinstance(entity_type, EntityType):
            raise Exception("False format!")
        if entity_type not in self.entity_types:
            self.entity_types.append(entity_type)

    def add_process(self, process):
        if not isinstance(process, Process) and not isinstance(process, ValueAddedProcess):
            raise Exception("False format!")

        process_type = Process if isinstance(process, Process) else ValueAddedProcess
        if process not in self.processes[process_type]:
            self.processes[process_type].append(process)

    def add_process_executions(self, process_executions: list[ProcessExecution]):
        """
        Add process_executions to the state model
        Faster than adding a single process_execution to the state model

        Parameters
        ----------
        process_executions: The process_executions to be added.
        """

        new_process_execution_entries = np.array([_get_process_execution_entry(process_execution)
                                                  for process_execution in process_executions],
                                                 dtype=process_execution_data_type)

        new_process_execution_entries = new_process_execution_entries.sort_values("Executed Start Time")

        executed_start_times = self.process_executions["Executed Start Time"].dropna()
        if executed_start_times.any():
            last_executed_start_time = executed_start_times[-1]
            process_executions_in_between_between_mask = (
                    new_process_execution_entries["Executed Start Time"] < last_executed_start_time)
            process_executions_in_between = new_process_execution_entries[process_executions_in_between_between_mask]
            for process_execution in process_executions_in_between["Process Execution"]:
                self.add_process_execution(process_execution)  # separate handling

            new_process_execution_entries_afterwards = (
                new_process_execution_entries[~process_executions_in_between_between_mask])
        else:
            new_process_execution_entries_afterwards = new_process_execution_entries

        self.process_executions = np.concatenate([self.process_executions, new_process_execution_entries_afterwards],
                                                 axis=0)

    def add_process_execution(self, process_execution: ProcessExecution,
                              time_chronological_order_ensured: bool = False):
        """
        Adds a process_execution to the state model

        Parameters
        ----------
        process_execution: The process_execution to be added.
        time_chronological_order_ensured: if true, the process_execution is added to the end of the list because
        it can be assumed that the added element is in the time_line the last element
        """

        process_execution_entry = np.array([_get_process_execution_entry(process_execution)],
                                           dtype=process_execution_data_type)

        resources = process_execution.get_resources()
        for resource in resources:  # ToDo: faster alternatives ... (known if creation?)
            resource_dict = self.dt_objects_directory[resource.__class__.__name__]  # ToDo: not suitable for obstacles

            resource_list = resource_dict[resource.entity_type]
            if resource not in resource_list:
                self.add_resource(resource)

        parts = process_execution.get_parts()
        for part in parts:
            try:
                parts_list = self.parts[part.entity_type]
            except:
                print(part.entity_type.name,
                      part.entity_type.identification)
                raise Exception
            if part not in parts_list:
                self.add_part(part)

        if time_chronological_order_ensured:
            self.process_executions = np.concatenate([self.process_executions, process_execution_entry], axis=0)
            return

        if process_execution.executed_start_time is not None:
            process_executions_after = np.where(self.process_executions["Executed Start Time"] >
                                                np.datetime64(process_execution.executed_start_time, "ns"))[0]
            next_process_executions_index = self.process_executions.size - process_executions_after.size
            self.process_executions = (
                np.insert(self.process_executions, next_process_executions_index, process_execution_entry,
                          axis=0))
        else:
            self.process_executions = np.concatenate([self.process_executions, process_execution_entry], axis=0)

        # ToDo: check if parts created are in the state model

    def add_resource(self, resource: all_resources_types):
        """Add a resource to the digital twin model"""

        resource_dict = self.dt_objects_directory[resource.__class__.__name__]
        resource_entity_type = resource.entity_type
        resource_super_entity_type = resource_entity_type.super_entity_type

        # add the resource
        print(f"Add resource '{resource.name}' to digital twin")
        if resource not in resource_dict[resource_entity_type]:
            resource_dict[resource_entity_type].append(resource)

        if resource_super_entity_type is None:
            return

        if resource not in resource_dict[resource_super_entity_type]:
            resource_dict[resource_super_entity_type].append(resource)

    def add_non_stationary_resource(self, non_stationary_resource: non_stationary_resources_types):
        if isinstance(non_stationary_resource, ActiveMovingResource):
            self.add_active_moving_resource(active_moving_resource=non_stationary_resource)
        elif isinstance(non_stationary_resource, PassiveMovingResource):
            self.add_passive_moving_resource(passive_moving_resource=non_stationary_resource)
        else:
            print(f"The non_stationary_resource couldn't be identified as "
                  f"passive_moving_resource or active_moving_resource")

    def add_active_moving_resources(self, active_moving_resources: list[ActiveMovingResource]):
        """Add active_moving_resources to the digital twin model"""
        for active_moving_resource in active_moving_resources:
            self.add_active_moving_resource(active_moving_resource)

    def add_active_moving_resource(self, active_moving_resource: ActiveMovingResource):
        """Add an active moving resource to the digital twin model.

        Parameters
        ----------
        active_moving_resource: The active moving resource object to add.

        Raises
        ------
        Exception: If the type of active_moving_resource is not ActiveMovingResource.
        Exception: If the entity type of active_moving_resource is not specified.
        """
        if not isinstance(active_moving_resource, ActiveMovingResource):
            raise Exception("The type should be ActiveMovingResource!")
        if active_moving_resource.entity_type is None:
            raise Exception("Entity Type should be specified")
        sub_state_model_objects = active_moving_resource.get_sub_instances_to_add()
        for dt_object_class, dt_objects in sub_state_model_objects.items():
            if dt_object_class == EntityType:
                for dt_object in dt_objects:
                    self.add_entity_type(dt_object)
            elif dt_object_class == Storage:
                for dt_object in dt_objects:
                    self.add_stationary_resource(dt_object)

        if active_moving_resource.entity_type in self.passive_moving_resources:
            if active_moving_resource in self.passive_moving_resources[active_moving_resource.entity_type]:
                return  # resource already available

        self.active_moving_resources[active_moving_resource.entity_type].append(active_moving_resource)

    def add_passive_moving_resources(self, passive_moving_resources: list[PassiveMovingResource]):
        """Add passive_moving_resources to the state model"""
        for passive_moving_resource in passive_moving_resources:
            self.add_passive_moving_resource(passive_moving_resource)

    def add_passive_moving_resource(self, passive_moving_resource: PassiveMovingResource):
        """Add active_moving_resource to the state model"""
        if not isinstance(passive_moving_resource, PassiveMovingResource):
            raise Exception("The type should be PassiveMovingResource!")
        if passive_moving_resource.entity_type is None:
            raise Exception("Entity Type should be specified")
        if passive_moving_resource.entity_type in self.passive_moving_resources:
            if passive_moving_resource in self.passive_moving_resources[passive_moving_resource.entity_type]:
                return  # resource already available

        self.passive_moving_resources.setdefault(passive_moving_resource.entity_type,
                                                 []).append(passive_moving_resource)

    def add_stationary_resources(self, stationary_resources: list[StationaryResource]):
        """Add stationary_resources to the state model"""
        for stationary_resource in stationary_resources:
            self.add_stationary_resource(stationary_resource)

    def add_stationary_resource(self, stationary_resource: stationary_resources_types):
        """Add stationary_resource to the state model"""
        if not isinstance(stationary_resource, StationaryResource):
            raise Exception("The type should be StationaryResource!")
        if stationary_resource.entity_type is None:
            raise Exception("Entity Type should be specified")
        if stationary_resource.entity_type in self.stationary_resources:
            if stationary_resource in self.stationary_resources[stationary_resource.entity_type]:
                return  # resource already available

        self.stationary_resources.setdefault(stationary_resource.entity_type,
                                             []).append(stationary_resource)

    def add_customers(self, customers: list[Customer]):
        """Add customers to the state model"""
        self.customer_base.extend(customers)

    def add_customer(self, customer: Customer):
        """Add customer to the state model customer_base"""
        self.customer_base.append(customer)

    def add_orders(self, orders):
        orders_entries = np.array([_get_order_entry(order)
                                   for order in orders],
                                  dtype=order_data_type)
        self.order_pool = np.concatenate([self.order_pool, orders_entries], axis=0)

    def add_order(self, order: Order):
        """Add order to the state model order_pool"""
        order_entry = np.array([_get_order_entry(order)],
                               dtype=order_data_type)
        self.order_pool = np.concatenate([self.order_pool, order_entry], axis=0)

    def update_orders_access_variables(self):
        """ensure that the access data is always up-to-date"""

        self.order_pool = _transform_order_pool_list_to_np_array(self.order_pool["Order"].tolist())

    def add_feature_cluster(self, feature_cluster: FeatureCluster):

        if not isinstance(feature_cluster, FeatureCluster):
            print(f"The feature cluster {feature_cluster} is not a feature cluster.")
            return

        product_class = feature_cluster.product_class
        self.feature_clusters.setdefault(product_class,
                                         []).append(feature_cluster)

    def add_feature(self, feature: Feature):
        if not isinstance(feature, Feature):
            print(f"The feature {feature} is not a feature.")
            return

        self.features.append(feature)

    def delete_order(self, order: Order):
        """Delete order from the state model order_pool"""
        self.order_pool = np.delete(self.order_pool, np.where(self.order_pool["Order"] == order), axis=0)

    def add_parts(self, parts: list[Part]):
        """Add parts to the state model"""
        for part in parts:
            self.add_part(part)

    def add_part(self, part: Part):
        """Add part to the state model"""
        if part.entity_type not in self.parts:
            self.parts[part.entity_type] = []

        parts_list = self.parts[part.entity_type]
        if part not in parts_list:
            self.parts[part.entity_type].append(part)
            if part.entity_type.super_entity_type is not None:
                if part.entity_type.super_entity_type not in self.parts:
                    self.parts[part.entity_type.super_entity_type] = []

                parts_list = self.parts[part.entity_type.super_entity_type]
                if part not in parts_list:
                    parts_list.append(part)

    # interim elements

    def add_process_executions_plan(self, process_executions_plan: (
            Union[ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt])):
        self.process_executions_plans.append(process_executions_plan)

    def add_process_time_model(self, process_time_model: ProcessTimeModel):
        if process_time_model not in self.process_models[ProcessTimeModel]:
            self.process_models[ProcessTimeModel].append(process_time_model)

    def add_quality_model(self, quality_model: QualityModel):
        if quality_model not in self.process_models[QualityModel]:
            self.process_models[QualityModel].append(quality_model)

    def add_resource_model(self, resource_model: ResourceModel):
        if resource_model not in self.process_models[ResourceModel]:
            self.process_models[ResourceModel].append(resource_model)

    def add_resource_group(self, resource_group: ResourceGroup):
        if resource_group not in self.resource_groups:
            self.resource_groups.append(resource_group)

    def add_transition_model(self, transition_model: TransitionModel):
        if transition_model not in self.process_models[TransitionModel]:
            self.process_models[TransitionModel].append(transition_model)

    def add_transformation_model(self, transformation_model: TransformationModel):
        if transformation_model not in self.process_models[TransformationModel]:
            self.process_models[TransformationModel].append(transformation_model)

    def add_entity_transformation_node(self, entity_transformation_node: EntityTransformationNode):
        if entity_transformation_node not in self.entity_transformation_nodes:
            self.entity_transformation_nodes.append(entity_transformation_node)

    # #### DELETE DIGITAL TWIN OBJECTS #################################################################################

    def delete_entity_type(self, entity_type: EntityType):
        if entity_type in self.entity_types:
            self.entity_types.remove(entity_type)
        else:
            print(f"Entity Type {entity_type} not found")

    def delete_plant(self, plant: Plant):
        if self.plant == plant:
            self.plant = None
        else:
            print(f"Plant {plant} not found")

    def delete_stationary_resource(self, stationary_resource):
        deleted = False
        if stationary_resource.entity_type in self.stationary_resources:
            if stationary_resource in self.stationary_resources[stationary_resource.entity_type]:
                self.stationary_resources[stationary_resource.entity_type].remove(stationary_resource)
                deleted = True

        if stationary_resource.entity_type.super_entity_type in self.stationary_resources:
            if stationary_resource in self.stationary_resources[stationary_resource.entity_type.super_entity_type]:
                self.stationary_resources[stationary_resource.entity_type.super_entity_type].remove(stationary_resource)
                deleted = True

        if stationary_resource.entity_type in self.obstacles:
            self.obstacles.remove(stationary_resource)
            deleted = True

        if not deleted:
            print(f"Stationary Resource {stationary_resource} not found")

    def delete_non_stationary_resource(self, non_stationary_resource):
        deleted = False
        if non_stationary_resource.entity_type in self.passive_moving_resources:
            if non_stationary_resource in self.passive_moving_resources[non_stationary_resource.entity_type]:
                self.passive_moving_resources[non_stationary_resource.entity_type].remove(non_stationary_resource)
                deleted = True

        if non_stationary_resource.entity_type.super_entity_type in self.passive_moving_resources:
            pmr_list = self.passive_moving_resources[non_stationary_resource.entity_type.super_entity_type]
            if non_stationary_resource in pmr_list:
                pmr_list.remove(non_stationary_resource)
                deleted = True

        if non_stationary_resource.entity_type in self.active_moving_resources:
            if non_stationary_resource in self.active_moving_resources[non_stationary_resource.entity_type]:
                self.active_moving_resources[non_stationary_resource.entity_type].remove(non_stationary_resource)
                deleted = True

        if non_stationary_resource.entity_type.super_entity_type in self.active_moving_resources:
            amr_list = self.active_moving_resources[non_stationary_resource.entity_type.super_entity_type]
            if non_stationary_resource in amr_list:
                amr_list.remove(non_stationary_resource)
                deleted = True

        if not deleted:
            print(f"Non Stationary Resource {non_stationary_resource} not found")

    def delete_process_execution_plan(self, process_execution_plan: ProcessExecutionPlan):
        if process_execution_plan in self.process_executions_plans:
            self.process_executions_plans.remove(process_execution_plan)
        else:
            print(f"Process Execution Plan {process_execution_plan} not found")

    def delete_customer(self, customer: Customer):
        if customer in self.customer_base:
            self.customer_base.remove(customer)
        else:
            print(f"Customer {customer} not found")

    def delete_feature_cluster(self, feature_cluster: FeatureCluster):
        product_class = feature_cluster.product_class
        if product_class in self.feature_clusters:
            if feature_cluster in self.feature_clusters[product_class]:
                self.feature_clusters[product_class].remove(feature_cluster)
                return

        print(f"Feature Cluster {feature_cluster} not found")

    def delete_feature(self, feature: Feature):
        if feature in self.features:
            self.features.remove(feature)
        else:
            print(f"Feature {feature} not found")

    def delete_part(self, part: Part):
        """Delete part from the state model"""
        print("Delete Part")
        if part.entity_type in self.parts:
            try:
                self.parts[part.entity_type].remove(part)
            except ValueError:
                if prints_visible:
                    print(f"[{self.__class__.__name__:20}]"
                          f" The part '{part.name}' with the ID '{part.identification}' cannot be removed "
                          f"from the state model because the part is not in list")

            if part.entity_type.super_entity_type is not None:
                try:
                    self.parts[part.entity_type.super_entity_type].remove(part)
                except ValueError:
                    if prints_visible:
                        print(f"[{self.__class__.__name__:20}]"
                              f" The part '{part.name}' with the ID '{part.identification}' cannot be removed "
                              f"from the state model because the part is not in list")
        else:
            raise ValueError(f"[{self.__class__.__name__:20}] "
                             f"The part '{part.name}' with the ID '{part.identification}' cannot be removed "
                             f"from the state model because the part entity type does not exist")

    def delete_time_model(self, process_time_model: ProcessTimeModel):
        all_processes = self.get_all_processes()
        for process in all_processes:
            process_time_model_from_process = process.lead_time_controller.get_model()
            if process_time_model_from_process == process_time_model:
                process.lead_time_controller.delete_dt_object()

    def delete_quality_model(self, quality_model: QualityModel):
        all_processes = self.get_all_processes()
        for process in all_processes:
            quality_model_from_process = process.quality_controller.get_model()
            if quality_model_from_process == quality_model:
                process.lead_time_controller.delete_dt_object()

    def delete_resource_model(self, resource_model: ResourceModel):
        all_processes = self.get_all_processes()
        for process in all_processes:
            resource_model_from_process = process.resource_controller.get_model()
            if resource_model_from_process == resource_model:
                process.lead_time_controller.delete_dt_object()

    def delete_transition_model(self, transition_model: TransitionModel):
        all_processes = self.get_all_processes()
        for process in all_processes:
            transition_model_from_process = process.transition_controller.get_model()
            if transition_model_from_process == transition_model:
                process.lead_time_controller.delete_dt_object()

    def delete_transformation_model(self, transformation_model: TransformationModel):
        all_processes = self.get_all_processes()
        for process in all_processes:
            transformation_model_from_process = process.lead_time_controller.get_model()
            if transformation_model_from_process == transformation_model:
                process.lead_time_controller.delete_dt_object()

    # #### TRANSFORM DIGITAL TWIN OBJECTS ##############################################################################

    def transform_part_entity_type(self, part):
        """Map the part to another entity_type if needed"""
        # Currently, not needed, because the part transformations do not have the use case
        old_entity_type = [entity_type
                           for entity_type, part_list in self.parts.items()
                           for part_ in part_list
                           if part.identification == part_.identification]
        if old_entity_type != part.entity_type:
            self.parts.remove(part)
            self.add_part(part)

    # #### OTHER METHODS ###############################################################################################

    def get_first_unassigned_identification(self) -> Optional[int]:
        """
        Returns
        -------
        first_unassigned_identification: the first identification that is unassigned/ not related to an object
        determined through an entity_type object assuming that an entity_type is available in model
        with the highest probability
        """

        if not self.entity_types:
            return None

        first_entity_type: EntityType = self.entity_types[0]
        first_unassigned_identification = first_entity_type.next_id
        return first_unassigned_identification

    @memoize_distance_matrix
    def get_distance_matrix(self) -> dict[tuple[StationaryResource, StationaryResource], float]:
        """Calculate the distances between all stationary_resources available in the state model"""

        stationary_resources = [stationary_resource
                                for stationary_resources_lst in list(self.stationary_resources.values())
                                for stationary_resource in stationary_resources_lst]

        distance_matrix = \
            {(stationary_resource1, stationary_resource2):
                 _get_distance(origin=stationary_resource1, destination=stationary_resource2)
             for idx, stationary_resource1 in enumerate(stationary_resources)
             for stationary_resource2 in stationary_resources[idx:]}

        return distance_matrix

    @memoize_get_stationary_resource_at_position
    def get_stationary_resource_at_position(self, pos_tuple):

        stationary_resources = \
            [stationary_resource
             for s_resource_lst in list(self.stationary_resources.values())
             for stationary_resource in s_resource_lst
             if stationary_resource.get_position() == pos_tuple and stationary_resource.situated_in is None]

        if not stationary_resources:
            print(pos_tuple)

        return stationary_resources

    def get_feature_process_mapper(self):
        """Maps all value_added_processes to the features"""

        feature_process_mapping = defaultdict(list)
        value_added_processes = self.processes[ValueAddedProcess]
        for process in value_added_processes:
            feature_process_mapping[process.feature].append(process)

        return feature_process_mapping

    def get_orders_in_progress(self, at: Optional[datetime] = None) -> list[Order]:
        """
        Parameters
        ----------
        at: for request times that are unequal to the current time of the state model

        Returns
        ----------
        orders_in_progress: a list of orders that are started but not finished
        """

        orders_without_delivery_date_actual = self.get_orders_not_finished(at=at)

        orders_in_progress = get_orders_in_progress(orders=orders_without_delivery_date_actual, at=at)

        return orders_in_progress

    def get_orders_not_finished(self, at: Optional[datetime] = None) -> list[Order]:
        """Return a list of orders that have no delivery_date_actual"""

        delivery_date_actual_mask = (
                self.order_pool["Delivery Date Actual"] != self.order_pool["Delivery Date Actual"])
        orders_without_delivery_date_actual = self.order_pool[delivery_date_actual_mask]

        if at is not None:
            # considering also orders that have already an actual delivery date
            delivery_date_actual_available_mask = (
                    self.order_pool["Delivery Date Actual"] == self.order_pool["Delivery Date Actual"])
            orders_with_delivery_date_actual = self.order_pool[delivery_date_actual_available_mask]

            # delivery dates after 'at'
            delivery_date_actual_mask = (
                (orders_with_delivery_date_actual["Delivery Date Actual"] > np.datetime64(at)))
            # with delivery date actual after 'at'
            orders_with_delivery_date_actual = orders_with_delivery_date_actual[delivery_date_actual_mask]
            orders_without_delivery_date_actual = np.concatenate([orders_without_delivery_date_actual,
                                                                  orders_with_delivery_date_actual])

        orders_list = orders_without_delivery_date_actual["Order"]
        orders_without_delivery_date_actual=[]
        for order in orders_list:
            if type(order.delivery_date_actual) == NoneType:
                orders_without_delivery_date_actual.append(order)

        return orders_without_delivery_date_actual

    def get_number_of_orders_in_progress(self, at: Optional[datetime] = None):
        """Return the number of orders that are in the system and currently processed"""

        orders_in_progress = self.get_orders_in_progress(at)
        number_orders_in_progress = len(orders_in_progress)

        return number_orders_in_progress

    def get_number_of_orders_not_finished(self):
        """Return the number of orders that are not finished"""

        orders_not_finished = self.get_orders_not_finished()
        number_orders_not_finished = len(orders_not_finished)

        return number_orders_not_finished

    # #### PERSISTENCE METHOD ##########################################################################################

    def store_as_pickle(self, pickle_path):
        """Store the state model as pickle file"""
        with open(pickle_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def to_pickle(self, digital_twin_file_path: Union[Path, str]):
        """
        Store the state model as pickle file to the path location.

        Parameters
        ----------
        digital_twin_file_path: the path where the pickle file should be stored
        """

        # ensure that the next id is stored (the pickle could not store the class attribute)
        self.next_id = DigitalTwinObject.next_id

        with open(digital_twin_file_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, digital_twin_file_path: Union[Path, str]) -> StateModel:
        """
        Read the state model from the pickle file.

        Parameters
        ----------
        digital_twin_file_path: file path where the digital_twin is stored

        Returns
        -------
        the read in state model
        """

        state_model = load_from_pickle(digital_twin_file_path)

        # needed because the class attributes are not stored by the pickle object
        DigitalTwinObject.next_id = state_model.next_id
        del state_model.next_id

        return state_model

    def get_resource_capacity_utilization(self, resource_names: list[str]):
        all_resources = self.get_all_resources()
        relevant_resources = [resource
                              for resource in all_resources
                              if resource.name in resource_names]

        start_time = self.process_executions[0]["Executed Start Time"]
        end_time = self.process_executions[-1]["Executed End Time"]

        utilization_lst = [round(float(resource.get_utilization(start_time, end_time)), 2) * 100
                           for resource in relevant_resources]
        return utilization_lst

    def get_order_lead_time_mean(self):
        lead_times = self.get_order_lead_times()

        order_lead_time_mean = np.mean(lead_times)
        print("order_lead_time_mean", order_lead_time_mean)
        return order_lead_time_mean

    def get_order_lead_times(self):
        orders = self.get_orders()

        lead_times = [order.get_lead_time()
                      for order in orders
                      if order.get_lead_time() is not None]
        return lead_times

    def get_estimated_order_lead_time_mean(self, orders: Optional[list[Order]] = None):
        lead_times = self.get_estimated_order_lead_times(orders)

        estimated_order_lead_time_mean = np.mean(lead_times)
        print("estimated_order_lead_time_mean", estimated_order_lead_time_mean)
        return estimated_order_lead_time_mean

    def get_estimated_order_lead_times(self, orders: Optional[list[Order]] = None):
        if orders is None:
            orders = self.get_orders()
        feature_value_added_processes = self.get_feature_process_mapper()
        lead_times = [self.get_estimated_order_lead_time(order, feature_value_added_processes)
                      for order in orders]

        return lead_times

    def get_estimated_order_lead_time(self, order, feature_value_added_processes: dict = None):
        # feature value_added_process required mapping
        if feature_value_added_processes is None:
            feature_value_added_processes = self.get_feature_process_mapper()

        value_added_processes_required = \
            [value_added_processes
             for feature in order.features_requested
             for value_added_processes in feature_value_added_processes[feature]]
        if not value_added_processes_required:
            return timedelta(seconds=0)

        # value added process order lead_time mapping
        estimated_order_lead_time = sum([value_added_process.get_estimated_process_lead_time()
                                         for value_added_process in value_added_processes_required])

        return timedelta(seconds=estimated_order_lead_time)



    def get_number_of_orders_finished(self):
        orders = self.get_orders()
        finished_orders = [order.is_finished()
                           for order in orders]
        number_of_orders_finished = sum(finished_orders)
        return number_of_orders_finished

    def get_delivery_reliability(self, end_of_consideration_period: Optional[datetime] = None):
        """Return the delivery reliability of the orders."""
        if end_of_consideration_period is None:
            _, end_of_consideration_period = self.get_consideration_period()

        orders = self.get_orders()
        reliability_status = [order.get_reliability_status(current_time=end_of_consideration_period)
                              for order in orders]
        reliable_count = reliability_status.count(True)
        delivery_reliability = (reliable_count / len(orders)) * 100

        return delivery_reliability
