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

The file is used to instantiate the state model objects from persistence.

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import json
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.helpers import Singleton, timestamp_to_datetime
from ofact.twin.state_model.basic_elements import (DigitalTwinObject, DynamicAttributes,
                                                   SingleObjectAttributeChangeTracker, ListAttributeChangeTracker,
                                                   DynamicAttributeChangeTracking, DomainSpecificAttributes,
                                                   list_attribute_change_tracker_data_type,
                                                   single_object_attribute_change_tracker_data_type)
from ofact.twin.state_model.entities import (Plant, EntityType, PartType, Part, Resource,
                                             StationaryResource, Storage, WorkStation, Warehouse,
                                             ActiveMovingResource, PassiveMovingResource, ConveyorBelt,
                                             PhysicalBody, StoragePlaces)
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.probabilities import (NormalDistribution, SingleValueDistribution,
                                                  BernoulliDistribution)
from ofact.twin.state_model.process_models import (EntityTransformationNode, TransformationModel, ResourceGroup,
                                                   ResourceModel, ProcessTimeModel, QualityModel, TransitionModel,
                                                   SimpleSingleValueDistributedProcessTimeModel,
                                                   SimpleNormalDistributedProcessTimeModel,
                                                   SimpleBernoulliDistributedQualityModel)
from ofact.twin.state_model.processes import (Process, ValueAddedProcess, ProcessExecution,
                                              ProcessTimeController, TransitionController, QualityController,
                                              ResourceController, TransformationController)
from ofact.twin.state_model.sales import Customer, FeatureCluster, Feature, Order
from ofact.twin.state_model.seralized_model import SerializedStateModel
from ofact.twin.state_model.time import ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt

if TYPE_CHECKING:
    pass

# must be imported
change_trackers = [SingleObjectAttributeChangeTracker, ListAttributeChangeTracker]
controller = [ProcessTimeController, TransitionController, QualityController, ResourceController,
              TransformationController]
storage_places_ = StoragePlaces
models = [TransformationModel, ResourceGroup, ResourceModel, ProcessTimeModel, QualityModel, TransitionModel,
          SimpleSingleValueDistributedProcessTimeModel, SimpleNormalDistributedProcessTimeModel,
          SimpleBernoulliDistributedQualityModel]


def _get_object_instantiated(object_dict):
    if object_dict is None:
        object_ = None
        return object_
    elif not isinstance(object_dict, dict):
        object_ = object_dict
        return object_

    object_type = eval(object_dict.pop("object_type"))
    instantiated_object = object_type.from_dict(object_dict)

    if hasattr(instantiated_object, "dynamic_attributes"):
        if isinstance(instantiated_object.dynamic_attributes, dict):
            dynamic_attributes: DynamicAttributes = (
                DynamicAttributes.from_dict(instantiated_object.dynamic_attributes))
            instantiated_object.dynamic_attributes = dynamic_attributes

    if hasattr(instantiated_object, "_domain_specific_attributes"):
        if (isinstance(instantiated_object._domain_specific_attributes, dict) and
                instantiated_object._domain_specific_attributes):
            domain_specific_attributes: dict = instantiated_object._domain_specific_attributes
            del domain_specific_attributes["object_type"]
            if "version" in domain_specific_attributes:
                del domain_specific_attributes["version"]

            domain_specific_attributes: DomainSpecificAttributes = (
                DomainSpecificAttributes.from_dict(domain_specific_attributes))
            instantiated_object._domain_specific_attributes = domain_specific_attributes

    return instantiated_object


def _get_feature_instantiated(feature_light):
    object_type = eval(feature_light["selection_probability_distribution"].pop("object_type"))
    feature_light["selection_probability_distribution"] = (
        object_type.from_dict(feature_light["selection_probability_distribution"]))
    feature_object = _get_object_instantiated(feature_light)
    return feature_object


def _get_process_execution_plan_instantiated(process_execution_plan_dict):
    process_execution_plan_dict["_time_schedule"]["Start"] = \
        [np.datetime64(int(t), "ns")
         for t in process_execution_plan_dict["_time_schedule"]["Start"]]
    process_execution_plan_dict["_time_schedule"]["End"] = \
        [np.datetime64(int(t), "ns")
         for t in process_execution_plan_dict["_time_schedule"]["End"]]

    time_schedule_entries = list(process_execution_plan_dict["_time_schedule"].values())
    process_execution_plan_dict["_time_schedule"] = (
        np.array([tuple(e) for e in np.array(time_schedule_entries).T],
                 dtype=ProcessExecutionPlan.time_schedule_data_type))
    process_execution_plan = _get_object_instantiated(process_execution_plan_dict)
    return process_execution_plan


def _get_process_execution_instantiated(process_execution_dict, processes, resources, parts,
                                        entity_transformation_nodes):
    number = eval(process_execution_dict["process"].split('.')[1])
    process_execution_dict["process"] = processes[number]

    entities = parts | resources
    process_execution_dict["_parts_involved"] = \
        [_get_entities_used_tuple(part_tuple, entities, entity_transformation_nodes)
         for part_tuple in process_execution_dict["_parts_involved"]]
    process_execution_dict["_resources_used"] = \
        [_get_entities_used_tuple(resource_used_tuple, entities, entity_transformation_nodes)
         for resource_used_tuple in process_execution_dict["_resources_used"]]
    process_execution_dict |= {"etn_specification": False}
    light_process_execution = _get_object_instantiated(process_execution_dict)

    return light_process_execution


def _get_entities_used_tuple(entities_used_tuple_light, available_entities, entity_transformation_nodes):
    entity_used = available_entities[eval(entities_used_tuple_light[0].split('.')[1])]

    if len(entities_used_tuple_light) > 1:
        entity_transformation_node = entity_transformation_nodes[eval(entities_used_tuple_light[1].split('.')[1])]
        entity_used_tuple = (entity_used, entity_transformation_node)

    else:
        entity_used_tuple = (entity_used,)

    return entity_used_tuple


def _get_process_controller_instantiated(process_controllers_list, process_models):
    process_controllers = {}
    for controller in process_controllers_list:
        if '_process_time_model' in controller.keys():
            process_controllers[controller['identification']] = eval(controller['object_type'])(
                identification=controller['identification'],
                external_identifications=controller['external_identifications'],
                domain_specific_attributes=controller['_domain_specific_attributes'],
                process_time_model=process_models[eval(controller['_process_time_model'].split('.')[1])]
            )
        if '_quality_model' in controller.keys():
            process_controllers[controller['identification']] = eval(controller['object_type'])(
                identification=controller['identification'],
                external_identifications=controller['external_identifications'],
                domain_specific_attributes=controller['_domain_specific_attributes'],
                quality_model=process_models[eval(controller['_quality_model'].split('.')[1])]
            )
        if '_transition_model' in controller.keys():
            process_controllers[controller['identification']] = eval(controller['object_type'])(
                identification=controller['identification'],
                external_identifications=controller['external_identifications'],
                domain_specific_attributes=controller['_domain_specific_attributes'],
                transition_model=process_models[eval(controller['_transition_model'].split('.')[1])]
            )
        if '_transformation_model' in controller.keys():
            process_controllers[controller['identification']] = eval(controller['object_type'])(
                identification=controller['identification'],
                external_identifications=controller['external_identifications'],
                domain_specific_attributes=controller['_domain_specific_attributes'],
                transformation_model=process_models[eval(controller['_transformation_model'].split('.')[1])]
            )
        if '_resource_model' in controller.keys():
            process_controllers[controller['identification']] = eval(controller['object_type'])(
                identification=controller['identification'],
                external_identifications=controller['external_identifications'],
                domain_specific_attributes=controller['_domain_specific_attributes'],
                resource_model=process_models[eval(controller['_resource_model'].split('.')[1])]
            )
    return process_controllers


def _get_times(time_stamp):
    return timestamp_to_datetime(time_stamp)

def _get_timestamps(time_stamp):
    if time_stamp == -1:
        return np.nan

    return np.datetime64(int(time_stamp * 1e9), "ns")


v_get_times = np.vectorize(_get_times, otypes=[np.datetime64])
v_get_timestamps = np.vectorize(_get_timestamps, otypes=[np.datetime64])


class DynamicStateModelDeserialization(metaclass=Singleton):
    __metaclass__ = Singleton

    def __init__(self, db=None):
        """
        Deserializes the dynamic state model

        Parameters
        ----------
        db: StateModelDB
            StateModelDB object
        """

        self.db = db

        self.state_model = None
        self.all_objects = {}

        self.v_get_object_by_id = np.vectorize(self.get_object_by_id, otypes=[object])

    def get_state_model(self, state_model_dict: dict) -> StateModel:
        """
        Main method of the class to deserialize the state model

        Parameters
        ----------
        state_model_dict: dict
            state model dictionary of the serialized state model

        Returns
        -------
        state_model: StateModel
            state model
        """

        next_id = int(eval(state_model_dict['next_id']))
        DigitalTwinObject.next_id = next_id

        (all_objects, entity_types, plant, parts, obstacles, storages, other_stationary_resources,
         passive_moving_resources, active_moving_resources, entity_transformation_nodes, processes, process_executions,
         order_pool, customer_base, features, feature_clusters, resource_group) = (
            self._instantiate_objects_from_key_value(state_model_dict))

        (entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
         active_moving_resources, entity_transformation_nodes, processes, process_executions,
         order_pool, customer_base, features, feature_clusters) = (
            self._fill_objects_from_key_value(all_objects, entity_types, plant, parts, obstacles, storages,
                                              other_stationary_resources, passive_moving_resources,
                                              active_moving_resources, entity_transformation_nodes, processes,
                                              process_executions, order_pool, customer_base, features, feature_clusters,
                                              resource_group))

        self.state_model = (
            self.create_state_model(entity_types, plant, parts, obstacles, stationary_resources,
                                    passive_moving_resources, active_moving_resources, entity_transformation_nodes,
                                    processes, process_executions, order_pool, customer_base, features,
                                    feature_clusters))

        return self.state_model

    def _instantiate_objects_from_key_value(self, sorted_elements):

        if "entity_types" in sorted_elements:

            list_entity_types = eval(eval(sorted_elements['entity_types']))
            entity_types = {entity_type_light["identification"]: _get_object_instantiated(entity_type_light)
                            for entity_type_light in list_entity_types}
            entity_types: dict[str, EntityType | PartType]
        else:
            entity_types = {}

        if "plant" in sorted_elements:
            list_plant = eval(eval(sorted_elements['plant']))
            plant = {list_plant["identification"]:
                         _get_object_instantiated(list_plant)}
            plant: dict[str, Plant]
        else:
            plant = {}

        if "parts" in sorted_elements:
            list_part = eval(eval(sorted_elements['parts']))
            parts = {part_light["identification"]: _get_object_instantiated(part_light)
                     for part_light in list_part}
        else:
            parts = {}

        if 'process_executions_plans' in sorted_elements:
            process_executions_plans_list = eval(eval(sorted_elements["process_executions_plans"]))
            process_executions_plans = {plan['identification']: _get_process_execution_plan_instantiated(plan)
                                        for plan in process_executions_plans_list}
        else:
            process_executions_plans = {}

        # ToDo: how to handle
        if 'obstacles' in sorted_elements:
            list_obstacles = eval(eval(sorted_elements['obstacles']))
            obstacles = {obstacle["identification"]: _get_object_instantiated(obstacle)
                         for obstacle in list_obstacles}
        else:
            obstacles = {}

        storages = {}
        if "stationary_resources" in sorted_elements:
            stationary_resources_list = eval(eval(sorted_elements["stationary_resources"]))
            # firstly all storages
            storages = {sr_light["identification"]:
                            self._get_resource_instantiated(sr_light, process_executions_plans)
                        for sr_light in stationary_resources_list
                        if sr_light['object_type'] == 'Storage'}
            other_stationary_resources = \
                {sr_light["identification"]:
                     self._get_resource_with_storage_instantiated(sr_light, process_executions_plans, storages)
                                    for sr_light in stationary_resources_list
                                    if "object_type" in sr_light}
            stationary_resources = other_stationary_resources | storages
            stationary_resources: dict[str, ConveyorBelt | Storage | WorkStation | Warehouse]
        else:
            other_stationary_resources = {}
            stationary_resources = {}

        if "passive_moving_resources" in sorted_elements:
            passive_moving_resources_list = eval(eval(sorted_elements["passive_moving_resources"]))
            passive_moving_resources = \
                {pmr_light["identification"]:
                     self._get_resource_with_storage_instantiated(pmr_light, process_executions_plans, storages)
                 for pmr_light in passive_moving_resources_list}

        else:
            passive_moving_resources = {}

        if "active_moving_resources" in sorted_elements:
            active_moving_resource_list = eval(eval(sorted_elements['active_moving_resources']))
            active_moving_resources = \
                {amr_light["identification"]:
                     self._get_resource_with_storage_instantiated(amr_light, process_executions_plans, storages)
                 for amr_light in active_moving_resource_list}
        else:
            active_moving_resources = {}

        resources = stationary_resources | passive_moving_resources | active_moving_resources

        if "entity_transformation_nodes" in sorted_elements:
            entity_transformation_nodes_list = eval(eval(sorted_elements["entity_transformation_nodes"]))
            entity_transformation_nodes = {etn_light["identification"]:
                                               self._get_entity_transformation_node_object_instantiated(etn_light)
                                           for etn_light in entity_transformation_nodes_list}
        else:
            entity_transformation_nodes = {}

        if 'process_models' in sorted_elements:
            process_models_list = eval(eval(sorted_elements['process_models']))
            process_models = self._get_process_model_instantiated(process_models_list)
        else:
            process_models = {}

        if 'process_controllers' in sorted_elements:
            process_controllers_list = eval(eval(sorted_elements['process_controllers']))
            process_controllers = _get_process_controller_instantiated(process_controllers_list, process_models)
        else:
            process_controllers = {}

        if "processes" in sorted_elements:
            process_list = eval(eval(sorted_elements["processes"]))

            processes = self._create_processes(process_list, process_controllers)
        else:
            processes = {}

        if "process_executions" in sorted_elements:
            process_executions_list = eval(eval(sorted_elements["process_executions"]))
            process_executions = {pe["identification"]:
                                      _get_process_execution_instantiated(
                                          pe, processes=processes, parts=parts, resources=resources,
                                          entity_transformation_nodes=entity_transformation_nodes)
                                  for pe in process_executions_list}
        else:
            process_executions = {}

        if "order_pool" in sorted_elements:
            order_pool_list = eval(eval(sorted_elements["order_pool"]))
            order_pool = {order["identification"]: _get_object_instantiated(order)
                          for order in order_pool_list}
        else:
            order_pool = {}

        if "customer_base" in sorted_elements:
            customer_base_list = eval(eval(sorted_elements["customer_base"]))
            customer_base = {customer_light["identification"]: _get_object_instantiated(customer_light)
                             for customer_light in customer_base_list}
        else:
            customer_base = {}

        if "features" in sorted_elements:
            features_list = eval(eval(sorted_elements["features"]))
            features = {feature["identification"]: _get_feature_instantiated(feature)
                        for feature in features_list}
        else:
            features = {}

        if "feature_clusters" in sorted_elements:
            feature_clusters_list = eval(eval(sorted_elements["feature_clusters"]))
            feature_clusters: dict[EntityType: list[FeatureCluster]] = \
                {feature_cluster_light["identification"]: _get_object_instantiated(feature_cluster_light)
                 for feature_cluster_light in feature_clusters_list}
        else:
            feature_clusters = {}

        if 'resource_groups' in sorted_elements:
            resource_groups_list = eval(eval(sorted_elements["resource_groups"]))
            resource_groups = {resource['identification']: _get_object_instantiated(resource)
                               for resource in resource_groups_list}
        else:
            resource_groups = {}

        if 'physical_bodies' in sorted_elements:
            physical_bodies_list = eval(eval(sorted_elements["physical_bodies"]))
            physical_bodies = {bodie['identification']: _get_object_instantiated(bodie)
                               for bodie in physical_bodies_list}
        else:
            physical_bodies = {}

        all_objects = (entity_types | plant | parts | obstacles | stationary_resources | passive_moving_resources |
                       active_moving_resources | entity_transformation_nodes | processes | process_executions |
                       order_pool | customer_base | features | feature_clusters | resource_groups |
                       process_executions_plans | physical_bodies)
        self.all_objects |= all_objects

        return (all_objects, entity_types, plant, parts, obstacles, storages, other_stationary_resources,
                passive_moving_resources, active_moving_resources, entity_transformation_nodes, processes,
                process_executions, order_pool, customer_base, features, feature_clusters, resource_groups)

    def _get_process_model_instantiated(self, process_models_list):
        process_models = {}
        for model_list in process_models_list:
            for model in process_models_list[model_list]:
                model_object_type = eval(model.pop("object_type"))
                process_models[model['identification']] = model_object_type.from_dict(model)
        return process_models

    def _get_resource_with_storage_instantiated(self, light_object, process_executions_plans, storages):

        object_ = self._get_resource_instantiated(light_object, process_executions_plans)

        # only for a short time - set after all storages are available
        if "buffer_stations" in light_object:
            buffer_stations = {None: [storages[int(bs.split("id.")[1])]
                                      for bs in set(light_object["buffer_stations"])]}
            object_._buffer_stations = StoragePlaces(buffer_stations, name=object_.name,
                                                     capacity=light_object["capacity"], situated_in=object_)
        elif "storage_places" in light_object:
            storage_places = {None: [storages[int(bs.split("id.")[1])]
                                     for bs in set(light_object["storage_places"])]}  # Should be adapted later
            object_._storage_places = StoragePlaces(storage_places, name=object_.name,
                                                    capacity=light_object["capacity"], situated_in=object_)

        return object_

    def _get_resource_instantiated(self, light_object, process_executions_plans):
        object_ = _get_object_instantiated(light_object)
        if object_ is None:
            return object_

        if hasattr(object_, "physical_body"):
            if not type(object_.physical_body) == str:
                object_.physical_body: PhysicalBody = _get_object_instantiated(object_.physical_body)

        if hasattr(object_, "_process_execution_plan"):
            if not object_._process_execution_plan:
                return object_

            process_execution_plan = process_executions_plans[int(object_.process_execution_plan.split("id.")[1])]
            object_._process_execution_plan: ProcessExecutionPlan | ProcessExecutionPlanConveyorBelt = (
                process_execution_plan)

        return object_

    def _get_entity_transformation_node_object_instantiated(self, light_object):
        if light_object is not None:
            transformation_type = (
                vars(EntityTransformationNode.TransformationTypes)[light_object["transformation_type"].split(".")[2]])
            io_behaviour = vars(EntityTransformationNode.IoBehaviours)[light_object["io_behaviour"].split(".")[2]]
            light_object["transformation_type"] = transformation_type
            light_object["io_behaviour"] = io_behaviour
            light_object = _get_object_instantiated(light_object)
        else:
            light_object = None

        return light_object

    def _create_processes(self, processes, controller):

        processes_dict = {}
        for pro in processes:
            process = self._get_process_instantiated(pro, controller)
            processes_dict[process.identification] = process

        return processes_dict

    def _get_process_instantiated(self, pro, controller):
        process = _get_object_instantiated(pro)
        process.lead_time_controller = controller[eval(process.lead_time_controller.split('.')[1])]
        process.quality_controller = controller[eval(process.quality_controller.split('.')[1])]
        process.transition_controller = controller[eval(process.transition_controller.split('.')[1])]
        process.transformation_controller = controller[eval(process.transformation_controller.split('.')[1])]
        process.resource_controller = controller[eval(process.resource_controller.split('.')[1])]

        return process

    def _fill_objects_from_key_value(self, all_objects, entity_types, plants, parts, obstacles, storages,
                                     other_stationary_resources,
                                     passive_moving_resources, active_moving_resources, entity_transformation_nodes,
                                     processes, process_executions, order_pool, customer_base, features,
                                     feature_clusters, resource_groups):
        entity_types = {entity_type_id: self._update_entity_type(entity_type)
                        for entity_type_id, entity_type in entity_types.items()}

        if plants:
            for id_, plant in plants.items():
                if not plant.work_calendar == None:
                    plant = self._update_plant(list(plant)[0])
        else:
            plant = None

        entity_type_parts: dict[EntityType: list[Part]] = {}
        for part in list(parts.values()):
            part_updated = self._update_part(part)
            entity_type_parts.setdefault(part_updated.entity_type,
                                         []).append(part_updated)
        parts = entity_type_parts

        obstacles: list[Resource] = [self._update_resource(obstacle)
                                     for obstacle in list(obstacles.values())]

        entity_type_stationary_resources: dict[EntityType:list[StationaryResource]] = {}
        for stationary_resource in list(storages.values()):
            updated_stationary_resource = self._update_stationary_resource(stationary_resource)
            entity_type_stationary_resources.setdefault(updated_stationary_resource.entity_type,
                                                        []).append(updated_stationary_resource)
        for stationary_resource in list(other_stationary_resources.values()):
            updated_stationary_resource = self._update_stationary_resource(stationary_resource)
            entity_type_stationary_resources.setdefault(updated_stationary_resource.entity_type,
                                                        []).append(updated_stationary_resource)
        stationary_resources = entity_type_stationary_resources

        entity_type_passive_moving_resources: dict[EntityType:list[PassiveMovingResource]] = {}
        for pmr in list(passive_moving_resources.values()):
            pmr_updated = self._update_non_stationary_resource(pmr)
            entity_type_passive_moving_resources.setdefault(pmr_updated.entity_type,
                                                            []).append(pmr_updated)
        passive_moving_resources = entity_type_passive_moving_resources

        entity_type_active_moving_resources: dict[EntityType:list[ActiveMovingResource]] = {}
        for amr in list(active_moving_resources.values()):
            amr_updated = self._update_non_stationary_resource(amr)
            entity_type_active_moving_resources.setdefault(amr_updated.entity_type,
                                                           []).append(amr_updated)
        active_moving_resources = entity_type_active_moving_resources

        entity_transformation_nodes_dict = {etn.identification: self._update_entity_transformation_node(etn)
                                            for etn in list(entity_transformation_nodes.values())}
        resource_groups = {rg.identification: self.update_resource_group(rg)
                           for rg in list(resource_groups.values())}
        processes_lst = list(processes.values())
        self._update_process_models(processes_lst, resource_groups, entity_types, all_objects,
                                    entity_transformation_nodes_dict)
        entity_transformation_nodes = list(entity_transformation_nodes_dict.values())
        processes_dict: dict[Process | ValueAddedProcess: list[Process]] = {}
        for process in processes_lst:
            processes_dict.setdefault(process.__class__,
                                      []).append(self._update_process(process))

        process_executions: list[ProcessExecution] = [self._update_process_execution(pe)
                                                      for pe in list(process_executions.values())]
        order_pool: list[Order] = [self._update_order(order)
                                   for order in list(order_pool.values())]
        customer_base: list[Customer] = [self._update_customer(customer)
                                         for customer in list(customer_base.values())]
        features: list[Feature] = [self._update_feature(feature)
                                   for feature in list(features.values())]
        feature_clusters_dict: dict[EntityType | PartType, list[FeatureCluster]] = {}
        for feature_cluster in list(feature_clusters.values()):
            feature_clusters_dict.setdefault(feature_cluster.product_class,
                                             []).append(feature_cluster)

        return (list(entity_types.values()), plant, parts, obstacles, stationary_resources, passive_moving_resources,
                active_moving_resources, entity_transformation_nodes, processes_dict, process_executions,
                order_pool, customer_base, features, feature_clusters_dict)

    def _update_entity_type(self, entity_type):

        if entity_type.super_entity_type is not None:
            entity_type.super_entity_type = self.all_objects[eval(entity_type.super_entity_type.split('.')[1])]

        return entity_type

    def _update_plant(self, plant):

        if plant.work_calendar is not None:
            plant.work_calendar = self.all_objects[eval(plant.work_calendar.split('.')[1])]

        return plant

    def _update_part(self, part):

        part = self._update_entity(part)

        if part.part_of is not None:
            part.part_of = self.all_objects[eval(part.part_of.split('.')[1])]
        if len(part.parts) > 0:
            part.parts = [self.all_objects[eval(part_.split('.')[1])] for part_ in part.parts]
        return part

    def _update_entity(self, entity):

        if entity.entity_type is not None:
            entity.entity_type = self.all_objects[eval(entity.entity_type.split('.')[1])]
        if entity.situated_in is not None:
            try:
                entity.situated_in = self.all_objects[eval(entity.situated_in.split('.')[1])]
            except:
                print("Entity Dict:", entity)

        entity.dynamic_attributes = self._update_dynamic_attributes(entity.dynamic_attributes)

        return entity

    def _update_dynamic_attributes(self, object_):
        # where are the dynamic attributes needed ...?
        if object_.attributes is not None:
            if not isinstance(list(object_.attributes.values())[0], DynamicAttributeChangeTracking):
                object_.attributes = \
                    {attr_name: self._get_dynamic_attribute_value(dynamic_array)
                     for attr_name, dynamic_array in object_.attributes.items()}

                object_.time_stamps = v_get_times(object_.time_stamps).tolist()
                object_.latest_requested_version = timestamp_to_datetime(object_.latest_requested_version)

        return object_

    def _get_dynamic_attribute_value(self, dynamic_attribute_value):

        dynamic_attribute_value['recent_changes'] = (
            self._get_changes(dynamic_attribute_value['recent_changes']))
        dynamic_attribute_value['distant_past_changes'] = (
            self._get_changes(dynamic_attribute_value['distant_past_changes']))

        dynamic_change_tracking = DynamicAttributeChangeTracking.from_serialization(attribute_dict=dynamic_attribute_value)
        return dynamic_change_tracking

    def _get_changes(self, dynamic_attribute_change_tracker):
        if len(dynamic_attribute_change_tracker['Value']) != 0:
            if "id." in dynamic_attribute_change_tracker['Value'][0]:  # ToDo: it is not always the right solution
                dynamic_attribute_change_tracker['Value'] = (
                    self.v_get_object_by_id(dynamic_attribute_change_tracker['Value']))

            dynamic_attribute_change_tracker['ProcessExecution'] = (
                self.v_get_object_by_id(dynamic_attribute_change_tracker['ProcessExecution']))

            dynamic_attribute_change_tracker['Timestamp'] = (
                v_get_timestamps(dynamic_attribute_change_tracker['Timestamp']))

            dynamic_attribute_change_tracker[
                dynamic_attribute_change_tracker != dynamic_attribute_change_tracker] = None

        if "ChangeType" in dynamic_attribute_change_tracker:
            dtype_ = list_attribute_change_tracker_data_type
        else:
            dtype_ = single_object_attribute_change_tracker_data_type

        if False in dynamic_attribute_change_tracker:
            del dynamic_attribute_change_tracker[False]  # ToDo: Why

        changes = np.array(list(zip(*list(dynamic_attribute_change_tracker.values()))),
                                   dtype=dtype_)

        return changes

    def _update_resource(self, resource):

        resource = self._update_entity(resource)

        resource.physical_body = self._update_physical_body(resource.physical_body)
        # resource.process_execution_plan = self._update_process_execution_plan(resource.process_execution_plan)

        if resource.plant is not None:
            resource.plant = self.all_objects[eval(resource.plant.split('.')[1])]
        return resource

    def _update_physical_body(self, physical_body):

        physical_body = self.all_objects[eval(physical_body.split('.')[1])]
        physical_body.dynamic_attributes = self._update_dynamic_attributes(physical_body.dynamic_attributes)

        return physical_body

    def _update_stationary_resource(self, stationary_resource):

        stationary_resource = self._update_resource(stationary_resource)
        model_object_type = eval(stationary_resource.efficiency.pop("object_type"))
        stationary_resource.efficiency = model_object_type.from_dict(stationary_resource.efficiency)
        stationary_resource.efficiency: SingleValueDistribution | NormalDistribution | BernoulliDistribution

        # react on differences
        if isinstance(stationary_resource, Storage):
            stationary_resource.stored_entities = [self.all_objects[eval(stored_entity.split('.')[1])]
                                                   for stored_entity in stationary_resource.stored_entities]
            if stationary_resource.allowed_entity_type is not None:
                stationary_resource.allowed_entity_type = self.all_objects[
                    eval(stationary_resource.allowed_entity_type.split('.')[1])]

        elif isinstance(stationary_resource, WorkStation):
            stationary_resource._buffer_stations = self._update_storage_places(stationary_resource._buffer_stations)
            stationary_resource.__dict__["buffer_stations"] = stationary_resource._buffer_stations._storage_places  # ToDo: ..

        elif isinstance(stationary_resource, Warehouse):
            stationary_resource._storage_places = self._update_storage_places(stationary_resource._storage_places)
            stationary_resource.__dict__["storage_places"] = stationary_resource._storage_places._storage_places  # ToDo: ..

        elif isinstance(stationary_resource, ConveyorBelt):
            stationary_resource.entities_on_transport = [self.all_objects[eval(stored_entity.split('.')[1])]
                                                         for stored_entity in stationary_resource.entities_on_transport]
            stationary_resource.allowed_entity_types = [self.all_objects[eval(stored_entity.split('.')[1])]
                                                        for stored_entity in stationary_resource.allowed_entity_types]

        return stationary_resource

    def _update_process_execution_plan(self, process_execution_plan):
        process_execution_plan = self.all_objects[eval(process_execution_plan.split('.')[1])]
        return process_execution_plan

    def _update_non_stationary_resource(self, non_stationary_resource):

        non_stationary_resource = self._update_resource(non_stationary_resource)
        non_stationary_resource._storage_places = self._update_storage_places(non_stationary_resource._storage_places)
        non_stationary_resource.__dict__["storage_places"] = non_stationary_resource._storage_places._storage_places
        # ToDo: ..

        return non_stationary_resource

    def _update_storage_places(self, storage_places):
        storage_places_dict = {}
        for storage in storage_places._storage_places[None]:
            storage_places_dict.setdefault(storage.allowed_entity_type,
                                      []).append(storage)
            if storage.allowed_entity_type.super_entity_type is not None:
                storage_places_dict.setdefault(storage.allowed_entity_type.super_entity_type,
                                          []).append(storage)
        storage_places._storage_places = storage_places_dict
        return storage_places

    def _update_entity_transformation_node(self, entity_transformation_node):

        if entity_transformation_node.entity_type is not None:
            entity_transformation_node.entity_type = (
                self.all_objects)[eval(entity_transformation_node.entity_type.split('.')[1])]

        if len(entity_transformation_node.parents) > 0:
            entity_transformation_node.parents = [self.all_objects[eval(parent.split('.')[1])]
                                                  for parent in entity_transformation_node.parents]
        if entity_transformation_node.children is not None:
            entity_transformation_node.children = [self.all_objects[eval(child.split('.')[1])]
                                                   for child in entity_transformation_node.children]

        return entity_transformation_node

    def _update_process_models(self, processes, resource_groups_obj, entity_types, all_objects,
                               entity_transformation_node):

        for process in processes:
            if process.lead_time_controller.process_time_model.is_re_trainable():
                process.lead_time_controller.process_time_model.set_prediction_model()
            group_list = []
            for group in process.resource_controller.resource_model.resource_groups:
                if type(group) == str:
                    group_list.append(resource_groups_obj[eval(group.split('.')[1])])
            if len(process.resource_controller.resource_model.resource_groups) == len(group_list):
                process.resource_controller.resource_model.resource_groups = group_list
            possible_destination_list = []
            possible_origins_list = []
            for destination in process.transition_controller.transition_model.possible_destinations:
                if type(destination) == str:
                    possible_destination_list.append(all_objects[eval(destination.split('.')[1])])
            for origin in process.transition_controller.transition_model.possible_origins:
                if type(origin) == str:
                    possible_origins_list.append(all_objects[eval(origin.split('.')[1])])
            if len(process.transition_controller.transition_model.possible_destinations) == len(
                    possible_destination_list):
                process.transition_controller.transition_model.possible_destinations = possible_destination_list
            if len(process.transition_controller.transition_model.possible_origins) == len(possible_origins_list):
                process.transition_controller.transition_model.possible_origins = possible_origins_list
            nodes_list = []
            for nodes in process.transformation_controller.transformation_model.root_nodes:
                if type(nodes) == str:
                    nodes_list.append(entity_transformation_node[eval(nodes.split('.')[1])])
            if len(process.transformation_controller.transformation_model.root_nodes) == len(nodes_list):
                process.transformation_controller.transformation_model.root_nodes = nodes_list
            if type(process.group) == str:
                if "." in process.group:
                    process.group = entity_types[eval(process.group.split('.')[1])]

    def update_resource_group(self, resource_group):
        resource_list = []
        main_list = []
        for resource in resource_group.resources:
            resource_list.append(self.all_objects[eval(resource.split('.')[1])])
        for main in resource_group.main_resources:
            main_list.append(self.all_objects[eval(main.split('.')[1])])
        resource_group.resources = resource_list
        resource_group.main_resources = main_list
        return resource_group

    def _update_process(self, process):

        # ToDo: light_object = _update_list_attr_light_object(light_object, "group")

        if isinstance(process, ValueAddedProcess):
            process.feature = self.all_objects[eval(process.feature.split('.')[1])]
            process.predecessors = [tuple([self.all_objects[eval(predecessor_id.split('.')[1])]
                                           for predecessor_id in predecessor_lst])
                                    for predecessor_lst in process.predecessors]
            process.successors = [self.all_objects[eval(successor_id.split('.')[1])]
                                  for successor_id in process.successors]

        return process

    def _update_process_execution(self, process_execution):

        # process_execution['_event_type'] = vars(ProcessExecution.EventTypes)[process_execution['_event_type']]
        process_execution.event_type = eval(process_execution.event_type)
        if process_execution.main_resource is not None:
            process_execution.main_resource = self.all_objects[eval(process_execution.main_resource.split('.')[1])]
        if process_execution.origin is not None:
            process_execution.origin = self.all_objects[eval(process_execution.origin.split('.')[1])]
        if process_execution.destination is not None:
            process_execution.destination = self.all_objects[eval(process_execution.destination.split('.')[1])]
        if process_execution.order is not None:
            process_execution.order = self.all_objects[eval(process_execution.order.split('.')[1])]
        if (process_execution.connected_process_execution is not None and
                process_execution.connected_process_execution == process_execution.connected_process_execution):
            try:
                process_execution.connected_process_execution = (
                    self.all_objects[eval(process_execution.connected_process_execution.split('.')[1])])
            except:
                print("Problem:", process_execution._connected_process_execution)
        # else:
        #     process_execution.connected_process_execution = None

        if type(process_execution.executed_start_time) == int:
            process_execution.executed_start_time = timestamp_to_datetime(process_execution.executed_start_time)
        if  type(process_execution.executed_end_time) == int:
            process_execution.executed_end_time = timestamp_to_datetime(process_execution.executed_end_time)

        return process_execution

    def _update_order(self, order: Order):
        """Update the order object"""

        order.dynamic_attributes = self._update_dynamic_attributes(order.dynamic_attributes)
        order.features_completed = [self.all_objects[eval(feature_completed.split('.')[1])]
                                    for feature_completed in order.features_completed]
        order.features_requested = [self.all_objects[eval(feature_requested.split('.')[1])]
                                    for feature_requested in order.features_requested]
        if order.product_classes:
            order.product_classes = [self.all_objects[eval(product_class.split('.')[1])]
                                     for product_class in order.product_classes]
        if order.customer:
            try:
                customer = self.all_objects[eval(order.customer.split('.')[1])]
                if isinstance(customer, Customer):
                    order.customer = customer
            except:
                if not isinstance(order.customer, Customer):
                    order.customer = None
                print("Warning - Customer initialization fails:", order.customer)
        if order.products:
            order.products = [self.all_objects[eval(product.split('.')[1])]
                              for product in order.products]
        order.feature_process_execution_match = \
            {(self.all_objects[eval(dict_id.split('.')[1])] if dict_id != "None" else None):
                 [self.all_objects[eval(process_execution_id.split('.')[1])]
                  for process_execution_id in process_execution_ids]
             for dict_id, process_execution_ids in order.feature_process_execution_match.items()}

        if type(order.order_date) == int:
            order.order_date = timestamp_to_datetime(order.order_date)
        if type(order.release_date_planned) == int:
            order.release_date_planned = timestamp_to_datetime(order.release_date_planned)
        if type(order.release_date_actual) == int:
            order.release_date_actual = timestamp_to_datetime(order.release_date_actual)
        if type(order.start_time_planned) == int:
            order.start_time_planned = timestamp_to_datetime(order.start_time_planned)
        if type(order.start_time_actual) == int:
            order.start_time_actual = timestamp_to_datetime(order.start_time_actual)
        if type(order.end_time_planned) == int:
            order.end_time_planned = timestamp_to_datetime(order.end_time_planned)
        if type(order.end_time_actual) == int:
            order.end_time_actual = timestamp_to_datetime(order.end_time_actual)
        if type(order.delivery_date_requested) == int:
            order.delivery_date_requested = timestamp_to_datetime(order.delivery_date_requested)
        if type(order.delivery_date_actual) == int:
            order.delivery_date_actual = timestamp_to_datetime(order.delivery_date_actual)
        if type(order.delivery_date_planned) == int:
            order.delivery_date_planned = timestamp_to_datetime(order.delivery_date_planned)

        return order

    def _update_customer(self, customer):

        return customer

    def _update_feature(self, feature):
        if feature.feature_cluster is not None:
            feature.feature_cluster = self.all_objects[eval(feature.feature_cluster.split('.')[1])]
            if type(feature.feature_cluster.product_class) == str:
                feature.feature_cluster.product_class = self.all_objects[
                    eval(feature.feature_cluster.product_class.split('.')[1])]
        return feature

    def get_object_by_id(self, object_id):

        if "id." not in object_id:
            return None

        object_id = int(object_id.split("id.")[1])
        if object_id in self.all_objects:
            return self.all_objects[object_id]

        else:
            return None

    def create_state_model(self, entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                           active_moving_resources, entity_transformation_nodes, processes, process_executions,
                           order_pool, customer_base, features, feature_clusters):

        state_model = StateModel(entity_types=entity_types, plant=plant, parts=parts,
                                 obstacles=obstacles, stationary_resources=stationary_resources,
                                 passive_moving_resources=passive_moving_resources,
                                 active_moving_resources=active_moving_resources,
                                 entity_transformation_nodes=entity_transformation_nodes,
                                 processes=processes, process_executions=process_executions,
                                 order_pool=order_pool, customer_base=customer_base,
                                 features=features, feature_clusters=feature_clusters)

        all_process_controllers = state_model.get_all_process_controllers()

        for process_controller in all_process_controllers:
            process_controller.set_digital_twin(digital_twin_model=state_model)

        return state_model

    @staticmethod
    def from_json(state_model_json):
        state_model_str = {}
        for key, value in state_model_json.items():
            state_model_str[key] = eval(json.loads(value))

        print(state_model_str)

    def from_db(self):
        file_id = ObjectId('66cd877a2db633db7efb6158')
        fs = gridfs.GridFS(self.db.db)
        json_data = fs.get(file_id).read().decode('UTF-8')
        digital_twin_state_model_dict = json.loads(json_data)
        return digital_twin_state_model_dict


if __name__ == "__main__":
    from pathlib import Path
    from ofact.settings import ROOT_PATH

    try:
        import gridfs
        from bson import ObjectId
    except:
        pass

    # state_model_dict = SerializedStateModel.load_from_json(Path(ROOT_PATH.split("ofact")[0],
    #                                                     "ofact/twin/repository_services/serialization/test_result.json"))
    state_model_dict = SerializedStateModel.load_from_pickle(Path(ROOT_PATH.split("ofact")[0],
                                                                  "ofact/twin/repository_services/serialization/test_result.pkl"))
    # state_model_dict = SerializedStateModel.load_from_parquet_folder(Path(ROOT_PATH.split("ofact")[0],
    #                                                                  "ofact/twin/repository_services/serialization/test_result"))
    # db=StateModelMongoDB()
    importer = DynamicStateModelDeserialization()  # StateModelMongoDB())
    # state_model_dict=deserialization.from_db()
    state_model = importer.get_state_model(state_model_dict)

    print(state_model)
    # ToDo: speed
    # state_model_dict = serialization.from_db()
