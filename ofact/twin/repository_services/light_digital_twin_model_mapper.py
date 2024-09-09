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

This file is used for the persistence of the digital twin model.
Therefore, the digital twin model elements are transformed. In the transformation, the python object references
are replaced by digital twin object identifications. These objects are called in the following as light objects.
The identifications avoid the recursion problem in the subsequent persistence.
On the way back from the light objects to complete digital twin objects,
replacing the identifications by digital twin objects again.

"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from collections import defaultdict
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING, Dict, List, Union

# Imports Part 2: PIP Imports
import dill as pickle
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import (DigitalTwinObject, DynamicAttributes,
                                                   SingleObjectAttributeChangeTracker, ListAttributeChangeTracker)
from ofact.twin.state_model.entities import (Plant, EntityType, PartType, Part, Resource,
                                             StationaryResource, Storage, WorkStation, Warehouse,
                                             ActiveMovingResource, PassiveMovingResource, ConveyorBelt,
                                             PhysicalBody)
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.process_models import (EntityTransformationNode, TransformationModel, ResourceGroup,
                                                   ResourceModel, ProcessTimeModel, QualityModel, TransitionModel)
from ofact.twin.state_model.processes import (Process, ValueAddedProcess, ProcessExecution,
                                              ProcessTimeController, TransitionController, QualityController,
                                              ResourceController, TransformationController)
from ofact.twin.state_model.sales import Customer, FeatureCluster, Feature, Order
from ofact.twin.state_model.time import ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt
from ofact.twin.state_model.helpers.helpers import load_from_pickle

if TYPE_CHECKING:
    pass


def _get_light_object(object_):
    if object_ is None:
        return None

    object_class = object_.__class__.__name__
    try:
        object_attributes = vars(object_).copy()
    except TypeError:
        raise TypeError(object_)

    light_object = {"cls": object_class,
                    "attr": object_attributes}

    return light_object


def _update_attr_light_object(light_object, attr):
    # capture none and np.nan
    if light_object["attr"][attr] is not None and light_object["attr"][attr] == light_object["attr"][attr]:
        if not isinstance(light_object["attr"][attr], int):
            try:
                light_object["attr"][attr] = light_object["attr"][attr].identification
            except AttributeError:
                raise AttributeError(attr, light_object["attr"])

    return light_object


def _update_list_attr_light_object(light_object, attr):
    if light_object["attr"][attr] is not None:
        try:
            light_object["attr"][attr] = [elem.identification for elem in light_object["attr"][attr]]
        except:
            print(light_object, attr)
        # light_object["attr"][attr] = [elem.identification
        #                               for elem in light_object["attr"][attr]]

    return light_object


def _update_tuple_list_attr_light_object(light_object, attr):
    if light_object["attr"][attr] is not None:
        try:
            light_object["attr"][attr] = [[elem.identification for elem in list(tuple_elem)]
                                          for tuple_elem in light_object["attr"][attr]]
        except TypeError:
            raise TypeError(attr, light_object["attr"])

    return light_object


def _update_enum_attr_light_object(light_object, attr):
    if light_object["attr"][attr] is not None:
        light_object["attr"][attr] = light_object["attr"][attr].name

    return light_object


def _update_entities_used_attr_light_object(light_object, attr):
    if light_object["attr"][attr] is not None:
        light_object["attr"][attr] = [(tuple_elem[0].identification,
                                       tuple_elem[1].identification if len(tuple_elem) == 2 else None)
                                      for tuple_elem in light_object["attr"][attr]]

    return light_object


def _get_entity_type_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_attr_light_object(light_object, "super_entity_type")

    return light_object


def _update_dynamic_light_object(light_object):
    # ToDo: also used for work order

    dynamic_attributes = light_object["attr"]["dynamic_attributes"]
    if dynamic_attributes is None:
        return light_object

    object_class = dynamic_attributes.__class__.__name__
    object_attributes = vars(dynamic_attributes).copy()

    dynamic_attributes_light_object = {"cls": object_class,
                                       "attr": object_attributes}

    dynamic_attributes_light_object["attr"]["attributes"] = \
        {attr_name: _transform_dynamic_attribute_value(dynamic_array)
         for attr_name, dynamic_array in dynamic_attributes_light_object["attr"]["attributes"].items()}
    light_object["attr"]["dynamic_attributes"] = dynamic_attributes_light_object

    return light_object


def _get_id(object_):
    if object_ is not None and object_ == object_:
        if isinstance(object_, DigitalTwinObject):
            return np.float32(object_.identification)
        else:
            return object_

    else:
        return np.nan


v_get_id = np.vectorize(_get_id)


def _transform_dynamic_attribute_value(dynamic_attribute_value):
    actual_value_type = "not_changed"
    recent_tracker, value_type_recent = _transform_dynamic_attribute_tracker(dynamic_attribute_value.recent_changes)

    distant_past_tracker, value_type_distant_past = (
        _transform_dynamic_attribute_tracker(dynamic_attribute_value.distant_past_changes))
    if value_type_recent == "DigitalTwinObject" or value_type_distant_past == "DigitalTwinObject":
        actual_value_type = "DigitalTwinObject"

    return dynamic_attribute_value, actual_value_type


def _transform_dynamic_attribute_tracker(dynamic_attribute_tracker):

    value_type = "not_changed"
    if len(dynamic_attribute_tracker.changes["Value"]) == 0:
        return dynamic_attribute_tracker, value_type

    if (isinstance(dynamic_attribute_tracker, SingleObjectAttributeChangeTracker) or
            isinstance(dynamic_attribute_tracker, ListAttributeChangeTracker)):
        value = dynamic_attribute_tracker.changes["Value"]
        if not isinstance(value[0], tuple):
            if not (isinstance(value[0], float) or isinstance(value[0], int)):
                value_type = "DigitalTwinObject"  # ToDo: fond more general ways ...

            try:
                dynamic_attribute_tracker.changes["Value"] = v_get_id(dynamic_attribute_tracker.changes["Value"])
            except:
                print(dynamic_attribute_tracker)
                print(dynamic_attribute_tracker.changes)
                raise Exception

    dynamic_attribute_tracker.changes["ProcessExecution"] = (
        v_get_id(dynamic_attribute_tracker.changes["ProcessExecution"]))

    return dynamic_attribute_tracker, value_type


def _update_entity_light_object(light_object):
    light_object = _update_attr_light_object(light_object, "_entity_type")
    light_object = _update_attr_light_object(light_object, "_situated_in")
    light_object = _update_dynamic_light_object(light_object)

    return light_object


def _get_part_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_entity_light_object(light_object)

    light_object = _update_attr_light_object(light_object, "part_of")
    light_object = _update_list_attr_light_object(light_object, "parts")

    return light_object


def _update_physical_body_attr_light_object(light_object):
    if light_object["attr"]["_physical_body"] is not None:
        physical_body_light_object = _get_light_object(light_object["attr"]["_physical_body"])
        try:
            physical_body_light_object = _update_dynamic_light_object(physical_body_light_object)
        except:
            raise Exception(light_object)
        light_object["attr"]["_physical_body"] = physical_body_light_object

    return light_object


def _update_process_executions_plan_light_object(light_object):
    if light_object["attr"]["_process_execution_plan"] is not None:
        process_executions_plan_light_object = _get_light_object(light_object["attr"]["_process_execution_plan"])
        # ToDo: work calendar
        light_object["attr"]["_process_execution_plan"] = process_executions_plan_light_object

    return light_object


def _update_resource_light_object(light_object):
    light_object = _update_entity_light_object(light_object)

    light_object = _update_attr_light_object(light_object, "plant")
    light_object = _update_physical_body_attr_light_object(light_object)
    light_object = _update_process_executions_plan_light_object(light_object)

    return light_object


def _get_resource_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)
    light_object = _update_resource_light_object(light_object)

    return light_object


def _update_storage_places_attr_light_object(light_object, attr):
    if light_object["attr"][attr] is not None:
        storage_places = light_object["attr"][attr]
        storage_places_dict = vars(storage_places).copy()
        storage_places_dict["_storage_places"] = \
            {entity_type.identification: [entity.identification for entity in entities]
             for entity_type, entities in storage_places_dict["_storage_places"].items()}
        light_object["attr"][attr] = storage_places_dict

    return light_object


def _get_stationary_resource_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_resource_light_object(light_object)

    # react on differences
    if isinstance(object_, Storage):
        light_object = _update_list_attr_light_object(light_object, "stored_entities")
        light_object = _update_attr_light_object(light_object, "allowed_entity_type")

    elif isinstance(object_, WorkStation):
        light_object = _update_storage_places_attr_light_object(light_object, "_buffer_stations")

    elif isinstance(object_, Warehouse):
        light_object = _update_storage_places_attr_light_object(light_object, "_storage_places")

    return light_object


def _get_non_stationary_resource_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_resource_light_object(light_object)

    light_object = _update_storage_places_attr_light_object(light_object, "_storage_places")

    return light_object


def _get_entity_transformation_node_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)
    light_object = _update_attr_light_object(light_object, "entity_type")
    light_object = _update_list_attr_light_object(light_object, "parents")
    light_object = _update_list_attr_light_object(light_object, "children")
    light_object = _update_enum_attr_light_object(light_object, "io_behaviour")
    light_object = _update_enum_attr_light_object(light_object, "transformation_type")

    return light_object


def _update_process_model(object_):
    if isinstance(object_, dict):
        return object_

    object_.save_model(persistent_saving=True)
    if object_.model_is_re_trainable():
        object_.delete_run_time_attributes()
    return object_


def _update_process_time_model_attribute(object_):
    object_ = _update_process_model(object_)
    light_object = _get_light_object(object_)

    return light_object


def _update_transition_model_attribute(object_):
    if isinstance(object_, dict):
        return object_

    object_ = _update_process_model(object_)

    light_object = _get_light_object(object_)
    if not isinstance(light_object["attr"]["_transition_model"], dict):
        light_object_transition_model = _get_light_object(light_object["attr"]["_transition_model"])
        light_object_transition_model = (
            _update_list_attr_light_object(light_object_transition_model, "_possible_origins"))
        light_object_transition_model = (
            _update_list_attr_light_object(light_object_transition_model, "_possible_destinations"))
        light_object["attr"]["_transition_model"] = light_object_transition_model

    return light_object


def _update_quality_model_attribute(object_):
    object_ = _update_process_model(object_)
    light_object = _get_light_object(object_)

    return light_object


def _update_transformation_model_attribute(object_):
    if isinstance(object_, dict):
        return object_

    object_ = _update_process_model(object_)

    light_object = _get_light_object(object_)
    if not isinstance(light_object["attr"]["_transformation_model"], dict):
        light_object_transformation_model = _get_light_object(light_object["attr"]["_transformation_model"])
        light_object_transformation_model = (
            _update_list_attr_light_object(light_object_transformation_model, "_root_nodes"))
        light_object["attr"]["_transformation_model"] = light_object_transformation_model

    return light_object


def _update_resource_model_attribute(object_):
    object_ = _update_process_model(object_)

    light_object = _get_light_object(object_)
    light_object["attr"]["_resource_model"] = _get_light_object(light_object["attr"]["_resource_model"])
    light_object["attr"]["_resource_model"]["attr"]["_resource_groups"] = \
        [_update_resource_group_attribute(elem)
         for elem in light_object["attr"]["_resource_model"]["attr"]["_resource_groups"]]

    return light_object


def _update_resource_group_attribute(object_):
    if isinstance(object_, dict):
        return object_

    light_object = _get_light_object(object_)
    light_object = _update_list_attr_light_object(light_object, "resources")
    light_object = _update_list_attr_light_object(light_object, "main_resources")

    return light_object


def _get_process_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    # ToDo: sample_extractor with digital twin

    light_object["attr"]["_lead_time_controller"] = (
        _update_process_time_model_attribute(light_object["attr"]["_lead_time_controller"]))
    light_object["attr"]["_quality_controller"] = (
        _update_quality_model_attribute(light_object["attr"]["_quality_controller"]))
    light_object["attr"]["_resource_controller"] = (
        _update_resource_model_attribute(light_object["attr"]["_resource_controller"]))
    light_object["attr"]["_transition_controller"] = (
        _update_transition_model_attribute(light_object["attr"]["_transition_controller"]))
    light_object["attr"]["_transformation_controller"] = (
        _update_transformation_model_attribute(light_object["attr"]["_transformation_controller"]))

    # light_object = _update_list_attr_light_object(light_object, "group")

    if light_object["cls"] == "ValueAddedProcess":
        light_object = _update_attr_light_object(light_object, "feature")

        light_object = _update_tuple_list_attr_light_object(light_object, "predecessors")
        light_object = _update_list_attr_light_object(light_object, "successors")

    return light_object


def _get_process_execution_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    # _executed_start_time - datetime
    # _executed_end_time - datetime
    light_object = _update_enum_attr_light_object(light_object, "_event_type")
    light_object = _update_attr_light_object(light_object, "process")
    light_object = _update_entities_used_attr_light_object(light_object, "_parts_involved")
    light_object = _update_entities_used_attr_light_object(light_object, "_resources_used")
    light_object = _update_attr_light_object(light_object, "_main_resource")
    light_object = _update_attr_light_object(light_object, "_origin")
    light_object = _update_attr_light_object(light_object, "_destination")
    light_object = _update_attr_light_object(light_object, "_order")
    light_object = _update_attr_light_object(light_object, "_connected_process_execution")

    return light_object


def _get_order_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)
    light_object = _update_dynamic_light_object(light_object)
    light_object = _update_list_attr_light_object(light_object, "features_completed")
    light_object = _update_list_attr_light_object(light_object, "features_requested")
    light_object = _update_attr_light_object(light_object, "product_class")
    light_object = _update_attr_light_object(light_object, "customer")
    light_object = _update_attr_light_object(light_object, "product")

    if light_object["attr"]["feature_process_execution_match"] is not None:
        light_object["attr"]["feature_process_execution_match"] = \
            {(feature.identification if feature else None):
                 [process_execution.identification
                  for process_execution in process_executions]
             for feature, process_executions in light_object["attr"]["feature_process_execution_match"].items()}

    return light_object


def _get_customer_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    return light_object


def _get_feature_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_attr_light_object(light_object, "feature_cluster")

    return light_object


def _get_feature_cluster_light_object(object_):
    if object_ is None:
        return None

    light_object = _get_light_object(object_)

    light_object = _update_attr_light_object(light_object, "product_class")

    return light_object


class DigitalTwinModelMapper:  # (metaclass=Singleton):
    # __metaclass__ = Singleton

    def __init__(self, digital_twin_model: StateModel = None, digital_twin_model_light: StateModel = None,
                 digital_twin_model_light_dict=None, pickle_path=None):
        self.digital_twin_model = digital_twin_model
        self.digital_twin_model_light = digital_twin_model_light
        self.digital_twin_model_light_dict = digital_twin_model_light_dict
        self.pickle_path = pickle_path

        # ToDo: changes should be tracked in the change_handler

    def get_digital_twin_objects_as_key_value(self):
        digital_twin_objects_as_key_value = self.create_digital_twin_objects_as_key_value()
        return digital_twin_objects_as_key_value

    def create_digital_twin_objects_as_key_value(self):
        if self.digital_twin_model is None:
            raise Exception("No digital twin available ...")

        entity_types_light = self._get_entity_types_light(self.digital_twin_model.entity_types)

        plant_light = self._get_plant_light(self.digital_twin_model.plant)

        if self.digital_twin_model.parts:
            parts_list = list(set(reduce(concat, list(self.digital_twin_model.parts.values()))))
            parts_light = self._get_parts_light(parts_list)
        else:
            parts_light = {}

        obstacles_light = self._get_resources_light(self.digital_twin_model.obstacles)

        if self.digital_twin_model.stationary_resources:
            stationary_resources_list = (
                list(set(reduce(concat, list(self.digital_twin_model.stationary_resources.values())))))
            stationary_resources_light = self._get_stationary_resources_light(stationary_resources_list)
        else:
            stationary_resources_light = {}

        if self.digital_twin_model.passive_moving_resources:
            passive_moving_resources_list = (
                list(set(reduce(concat, list(self.digital_twin_model.passive_moving_resources.values())))))
            passive_moving_resources_light = self._get_non_stationary_resources_light(passive_moving_resources_list)
        else:
            passive_moving_resources_light = {}

        if self.digital_twin_model.active_moving_resources:
            active_moving_resources_list = (
                list(set(reduce(concat, list(self.digital_twin_model.active_moving_resources.values())))))
            active_moving_resources_light = self._get_non_stationary_resources_light(active_moving_resources_list)
        else:
            active_moving_resources_light = {}

        entity_transformation_nodes_light = (
            self._get_entity_transformation_nodes_light(self.digital_twin_model.entity_transformation_nodes))

        if self.digital_twin_model.processes:
            processes_list = reduce(concat, self.digital_twin_model.processes.values())
            processes_light = self._get_processes_light(processes_list)
        else:
            processes_light = {}

        process_executions_light = (
            self._get_process_executions_light(self.digital_twin_model.get_process_executions_list()))

        order_pool_light = self._get_orders_light(self.digital_twin_model.get_orders())

        customer_base_light = self._get_customers_light(self.digital_twin_model.customer_base)

        features_light = self._get_features_light(self.digital_twin_model.features)

        if self.digital_twin_model.feature_clusters:
            features_clusters_list = reduce(concat, self.digital_twin_model.feature_clusters.values())
            feature_clusters_light = self._get_feature_clusters(features_clusters_list)
        else:
            feature_clusters_light = {}

        dt_objects = (entity_types_light | plant_light | parts_light | obstacles_light | stationary_resources_light |
                      passive_moving_resources_light | active_moving_resources_light |
                      entity_transformation_nodes_light | processes_light | process_executions_light |
                      order_pool_light | customer_base_light | features_light | feature_clusters_light)

        self.digital_twin_model_light_dict = {"next_id": DigitalTwinObject.next_id} | dt_objects

        return self.digital_twin_model_light_dict

    def _get_entity_types_light(self, entity_types):
        entity_types_light = {entity_type.identification: _get_entity_type_light_object(entity_type)
                              for entity_type in entity_types}
        return entity_types_light

    def _get_plant_light(self, plant):
        if plant:
            plant_light = {plant.identification: _get_light_object(plant)}  # in the future -> work calendar
        else:
            plant_light = {}
        return plant_light

    def _get_parts_light(self, parts):
        parts_light = {part.identification: _get_part_light_object(part)
                       for part in parts}
        return parts_light

    def _get_resources_light(self, resources):
        resources_light = {resource.identification: _get_resource_light_object(resource)
                           for resource in resources}
        return resources_light

    def _get_stationary_resources_light(self, stationary_resources):
        resources_light = {stationary_resource.identification:
                               _get_stationary_resource_light_object(stationary_resource)
                           for stationary_resource in stationary_resources}
        return resources_light

    def _get_non_stationary_resources_light(self, non_stationary_resources):
        resources_light = {non_stationary_resource.identification:
                               _get_non_stationary_resource_light_object(non_stationary_resource)
                           for non_stationary_resource in non_stationary_resources}
        return resources_light

    def _get_entity_transformation_nodes_light(self, entity_transformation_nodes):
        entity_transformation_nodes_light = {etn.identification: _get_entity_transformation_node_light_object(etn)
                                             for etn in entity_transformation_nodes}
        return entity_transformation_nodes_light

    def _get_processes_light(self, processes):
        processes_light = {pro.identification: _get_process_light_object(pro)
                           for pro in processes}
        return processes_light

    def _get_process_executions_light(self, process_executions):
        process_executions_light = {pe.identification: _get_process_execution_light_object(pe)
                                    for pe in process_executions}
        return process_executions_light

    def _get_orders_light(self, orders):
        orders_light = {order.identification: _get_order_light_object(order)
                        for order in orders}
        return orders_light

    def _get_customers_light(self, customers):
        customer_base_light = {customer.identification: _get_customer_light_object(customer)
                               for customer in customers}
        return customer_base_light

    def _get_features_light(self, features):
        features_light = {feature.identification: _get_feature_light_object(feature)
                          for feature in features}
        return features_light

    def _get_feature_clusters(self, feature_clusters):
        feature_clusters_light = {feature_cluster.identification: _get_feature_cluster_light_object(feature_cluster)
                                  for feature_cluster in feature_clusters}
        return feature_clusters_light

    def to_pickle(self, pickle_path=None):
        """Store the digital twin into a pickle file"""

        if pickle_path is None:
            pickle_path = self.pickle_path

        with open(pickle_path, 'wb') as outp:
            pickle.dump(self.digital_twin_model_light_dict, outp, pickle.HIGHEST_PROTOCOL)


def _get_object_instantiated(light_object):
    if light_object is None:
        object_ = None
        return object_

    object_ = eval(light_object["cls"]).from_dict(light_object["attr"])

    if hasattr(object_, "dynamic_attributes"):
        object_.dynamic_attributes: DynamicAttributes = (
            eval(object_.dynamic_attributes["cls"]).from_dict(object_.dynamic_attributes["attr"]))

    return object_


def _get_entity_transformation_node_object_instantiated(light_object):
    if light_object is not None:
        transformation_type = (
            vars(EntityTransformationNode.TransformationTypes))[light_object["attr"]["transformation_type"]]
        io_behaviour = vars(EntityTransformationNode.IoBehaviours)[light_object["attr"]["io_behaviour"]]
        light_object["attr"]["transformation_type"] = transformation_type
        light_object["attr"]["io_behaviour"] = io_behaviour

        object_ = eval(light_object["cls"]).from_dict(light_object["attr"])

    else:
        object_ = None

    return object_


def _get_process_execution_instantiated(light_process_execution, processes, resources, parts,
                                        entity_transformation_nodes):
    light_process_execution["attr"]["process"] = processes[light_process_execution["attr"]["process"]]

    entities = parts | resources
    light_process_execution["attr"]["_parts_involved"] = \
        [_get_entities_used_tuple(part_tuple, entities, entity_transformation_nodes)
         for part_tuple in light_process_execution["attr"]["_parts_involved"]]
    light_process_execution["attr"]["_resources_used"] = \
        [_get_entities_used_tuple(resource_used_tuple, entities, entity_transformation_nodes)
         for resource_used_tuple in light_process_execution["attr"]["_resources_used"]]

    object_ = eval(light_process_execution["cls"]).from_dict(light_process_execution["attr"] |
                                                             {"etn_specification": False})

    return object_


def _get_entities_used_tuple(entities_used_tuple_light, available_entities, entity_transformation_nodes):

    entity_used = available_entities[entities_used_tuple_light[0]]

    if entities_used_tuple_light[1] is not None:
        entity_transformation_node = entity_transformation_nodes[entities_used_tuple_light[1]]
        entity_used_tuple = (entity_used, entity_transformation_node)

    else:
        entity_used_tuple = (entity_used,)

    return entity_used_tuple


def _get_process_instantiated(pro, process_models, entity_transformation_nodes):
    process = _get_object_instantiated(pro)

    if process.lead_time_controller["attr"]["identification"] not in process_models:
        process.lead_time_controller = _get_object_instantiated(process.lead_time_controller)
        process.lead_time_controller: ProcessTimeController
        process_models[process.lead_time_controller.identification] = process.lead_time_controller
        process.lead_time_controller.process_time_model: ProcessTimeModel
    else:
        process.lead_time_controller = process_models[process.lead_time_controller["attr"]["identification"]]

    if process.quality_controller["attr"]["identification"] not in process_models:
        process.quality_controller = _get_object_instantiated(process.quality_controller)
        process.quality_controller: QualityController
        process_models[process.quality_controller.identification] = process.quality_controller
        process.quality_controller.quality_model: QualityModel
    else:
        process.quality_controller = process_models[process.quality_controller["attr"]["identification"]]

    if process.transition_controller["attr"]["identification"] not in process_models:
        process.transition_controller = _get_object_instantiated(process.transition_controller)
        process.transition_controller: TransitionController
        process_models[process.transition_controller.identification] = process.transition_controller
        process.transition_controller.transition_model: TransitionModel = (
            _get_object_instantiated(process.transition_controller.transition_model))
    else:
        process.transition_controller = process_models[process.transition_controller["attr"]["identification"]]

    if process.transformation_controller["attr"]["identification"] not in process_models:
        process.transformation_controller: TransformationController = (
            _get_object_instantiated(process.transformation_controller))

        process.transformation_controller.transformation_model["attr"]["_root_nodes"] = \
            [entity_transformation_nodes[root_node_id]
             if root_node_id in entity_transformation_nodes
             else root_node_id
             for root_node_id in process.transformation_controller.transformation_model["attr"]["_root_nodes"]]
        process.transformation_controller.transformation_model: TransformationModel = (
            _get_object_instantiated(process.transformation_controller.transformation_model))

        process_models[process.transformation_controller.identification] = process.transformation_controller

    else:
        process.transformation_controller = process_models[process.transformation_controller["attr"]["identification"]]

    if process.resource_controller["attr"]["identification"] not in process_models:
        process.resource_controller: ResourceController = _get_object_instantiated(process.resource_controller)
        process.resource_controller.resource_model: ResourceModel = (
            _get_object_instantiated(process.resource_controller.resource_model))
        resource_model = process.resource_controller.resource_model

        resource_groups = []
        for single_resource_group in resource_model.resource_groups:
            if single_resource_group["attr"]["identification"] not in process_models:
                single_resource_group_updated: ResourceGroup = _get_object_instantiated(single_resource_group)
                process_models[single_resource_group_updated.identification] = single_resource_group_updated

            else:
                single_resource_group_updated = process_models[single_resource_group["attr"]["identification"]]

            resource_groups.append(single_resource_group_updated)

        resource_model.resource_groups = resource_groups
        process_models[process.resource_controller.identification] = process.resource_controller
    else:
        process.resource_controller = process_models[process.resource_controller["attr"]["identification"]]

    return process, process_models


class LightDigitalTwinModelMapper:  # (metaclass=Singleton):
    # __metaclass__ = Singleton

    def __init__(self, digital_twin_model: StateModel = None, digital_twin_model_light: StateModel = None,
                 digital_twin_model_light_dict: dict = None, pickle_path=None):
        self.digital_twin_model = digital_twin_model
        self.digital_twin_model_light = digital_twin_model_light
        self.digital_twin_model_light_dict = digital_twin_model_light_dict
        self.all_objects = {}

        self.v_get_object_by_id = np.vectorize(self.get_object_by_id, otypes=[object])

        self.pickle_path = pickle_path

        if self.pickle_path is not None and self.digital_twin_model_light_dict is None:
            self.digital_twin_model_light_dict = self.from_pickle()

    def get_digital_twin_model_by_key_value(self):

        next_id = 0
        sorted_elements = defaultdict(list)

        for id_, dict_object_ in self.digital_twin_model_light_dict.items():
            if id_ == "next_id":
                next_id = dict_object_
                continue

            sorted_elements[dict_object_["cls"]].append(dict_object_)

        (all_objects, entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
         active_moving_resources, entity_transformation_nodes, processes, process_executions, order_pool,
         customer_base, features, feature_clusters) = self._instantiate_objects_from_key_value(sorted_elements)

        (entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
         active_moving_resources, entity_transformation_nodes, processes, process_executions,
         order_pool, customer_base, features, feature_clusters) = (
            self._fill_objects_from_key_value(all_objects, entity_types, plant, parts, obstacles, stationary_resources,
                                              passive_moving_resources, active_moving_resources,
                                              entity_transformation_nodes, processes, process_executions, order_pool,
                                              customer_base, features, feature_clusters))

        self.digital_twin_model = (
            self.create_digital_twin(entity_types, plant, parts, obstacles, stationary_resources,
                                     passive_moving_resources, active_moving_resources, entity_transformation_nodes,
                                     processes, process_executions, order_pool, customer_base, features,
                                     feature_clusters, next_id))

        return self.digital_twin_model

    def _instantiate_objects_from_key_value(self, sorted_elements):
        if "EntityType" in sorted_elements:
            entity_types = {entity_type_light["attr"]["identification"]: _get_object_instantiated(entity_type_light)
                            for entity_type_light in sorted_elements["EntityType"]}
        else:
            entity_types = {}

        if "PartType" in sorted_elements:
            entity_types: Dict[str, PartType]
            entity_types |= {entity_type_light["attr"]["identification"]: _get_object_instantiated(entity_type_light)
                             for entity_type_light in sorted_elements["PartType"]}

        if "Plant" in sorted_elements:
            plant = {sorted_elements["Plant"][0]["attr"]["identification"]:
                         _get_object_instantiated(sorted_elements["Plant"][0])}
        else:
            plant = {}

        if "Part" in sorted_elements:
            parts = {part_light["attr"]["identification"]: _get_object_instantiated(part_light)
                     for part_light in sorted_elements["Part"]}
        else:
            parts = {}

        # ToDo: how to handle
        obstacles = {obstacle["attr"]["identification"]: _get_object_instantiated(obstacle)
                     for obstacle in []}

        stationary_resources_list = []
        if "Storage" in sorted_elements:
            stationary_resources_list.extend(sorted_elements["Storage"])
        if "WorkStation" in sorted_elements:
            stationary_resources_list.extend(sorted_elements["WorkStation"])
        if "Warehouse" in sorted_elements:
            stationary_resources_list.extend(sorted_elements["Warehouse"])
        if "ConveyorBelt" in sorted_elements:
            stationary_resources_list: List[Union[ConveyorBelt, ...]]
            stationary_resources_list.extend(sorted_elements["ConveyorBelt"])

        stationary_resources = {sr_light["attr"]["identification"]: self._get_resource_instantiated(sr_light)
                                for sr_light in stationary_resources_list}

        if "PassiveMovingResource" in sorted_elements:
            passive_moving_resources = {pmr_light["attr"]["identification"]:
                                            self._get_resource_instantiated(pmr_light)
                                        for pmr_light in sorted_elements["PassiveMovingResource"]}

        else:
            passive_moving_resources = {}

        if "ActiveMovingResource" in sorted_elements:
            active_moving_resources = {amr_light["attr"]["identification"]: self._get_resource_instantiated(amr_light)
                                       for amr_light in sorted_elements["ActiveMovingResource"]}
        else:
            active_moving_resources = {}

        resources = stationary_resources | passive_moving_resources | active_moving_resources

        if "EntityTransformationNode" in sorted_elements:
            entity_transformation_nodes = {etn_light["attr"]["identification"]:
                                               _get_entity_transformation_node_object_instantiated(etn_light)
                                           for etn_light in sorted_elements["EntityTransformationNode"]}
        else:
            entity_transformation_nodes = {}

        all_processes = []
        if "Process" in sorted_elements:
            all_processes.extend(sorted_elements["Process"])
        if "ValueAddedProcess" in sorted_elements:
            all_processes.extend(sorted_elements["ValueAddedProcess"])
        processes = self._create_processes(all_processes, entity_transformation_nodes)

        if "ProcessExecution" in sorted_elements:
            process_executions = {pe["attr"]["identification"]:
                _get_process_execution_instantiated(
                    pe, processes=processes, parts=parts, resources=resources,
                    entity_transformation_nodes=entity_transformation_nodes)
                for pe in sorted_elements["ProcessExecution"]}
        else:
            process_executions = {}

        if "Order" in sorted_elements:
            order_pool = {order["attr"]["identification"]: _get_object_instantiated(order)
                          for order in sorted_elements["Order"]}
        else:
            order_pool = {}

        if "Customer" in sorted_elements:
            customer_base = {customer_light["attr"]["identification"]: _get_object_instantiated(customer_light)
                             for customer_light in sorted_elements["Customer"]}
        else:
            customer_base = {}

        if "Feature" in sorted_elements:
            features = {feature["attr"]["identification"]: _get_object_instantiated(feature)
                        for feature in sorted_elements["Feature"]}
        else:
            features = {}

        if "FeatureCluster" in sorted_elements:
            feature_clusters: dict[EntityType: list[FeatureCluster]] = \
                {feature_cluster_light["attr"]["identification"]: _get_object_instantiated(feature_cluster_light)
                 for feature_cluster_light in sorted_elements["FeatureCluster"]}
        else:
            feature_clusters = {}

        all_objects = (entity_types | plant | parts | obstacles | stationary_resources | passive_moving_resources |
                       active_moving_resources | entity_transformation_nodes | processes | process_executions |
                       order_pool | customer_base | features | feature_clusters)
        self.all_objects |= all_objects

        return (all_objects, entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                active_moving_resources, entity_transformation_nodes, processes, process_executions, order_pool,
                customer_base, features, feature_clusters)

    def _get_resource_instantiated(self, light_object):
        if "_storage_places" in light_object["attr"]:
            storage_places = light_object["attr"]["_storage_places"]
            light_object["attr"]["capacity"] = storage_places["capacity"]
            light_object["attr"]["storage_places"] = storage_places["_storage_places"]

            del light_object["attr"]["_storage_places"]

        elif "_buffer_stations" in light_object["attr"]:
            buffer_stations = light_object["attr"]["_buffer_stations"]
            light_object["attr"]["capacity"] = buffer_stations["capacity"]
            light_object["attr"]["buffer_stations"] = buffer_stations["_storage_places"]

            del light_object["attr"]["_buffer_stations"]

        object_ = _get_object_instantiated(light_object)
        if object_ is None:
            return object_

        if hasattr(object_, "physical_body"):
            object_.physical_body: PhysicalBody = _get_object_instantiated(object_.physical_body)

        if hasattr(object_, "_process_execution_plan"):
            if object_._process_execution_plan["attr"]["identification"] not in self.all_objects:
                process_execution_plan = _get_object_instantiated(object_._process_execution_plan)
                self.all_objects[object_._process_execution_plan["attr"]["identification"]] = process_execution_plan

            object_._process_execution_plan: ProcessExecutionPlan | ProcessExecutionPlanConveyorBelt = (
                self.all_objects[object_._process_execution_plan["attr"]["identification"]])

        return object_

    def _fill_objects_from_key_value(self, all_objects, entity_types, plant, parts, obstacles, stationary_resources,
                                     passive_moving_resources, active_moving_resources, entity_transformation_nodes,
                                     processes, process_executions, order_pool, customer_base, features,
                                     feature_clusters):
        entity_types = [self._update_entity_type(entity_type)
                        for entity_type_id, entity_type in entity_types.items()]
        if plant:
            plant: Plant = self._update_plant(list(plant.values())[0])
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
        for stationary_resource in list(stationary_resources.values()):
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

        entity_type_active_moving_resources: dict[EntityType:list[PassiveMovingResource]] = {}
        for amr in list(active_moving_resources.values()):
            amr_updated = self._update_non_stationary_resource(amr)
            entity_type_active_moving_resources.setdefault(amr_updated.entity_type,
                                                           []).append(amr_updated)
        active_moving_resources = entity_type_active_moving_resources

        entity_transformation_nodes = [self._update_entity_transformation_node(etn)
                                       for etn in list(entity_transformation_nodes.values())]
        processes_lst = list(processes.values())
        self._update_process_models(processes_lst)
        processes_dict: dict[Process | ValueAddedProcess:list[Process]] = {}
        for process in processes_lst:
            processes_dict.setdefault(process.__class__,
                                      []).append(process)
        processes = {process_class: [self._update_process(pro)
                                     for pro in pro_s]
                     for process_class, pro_s in processes_dict.items()}

        process_executions: list[ProcessExecution] = [self._update_process_execution(pe)
                                                      for pe in list(process_executions.values())]
        order_pool: list[Order] = [self._update_order(order)
                                   for order in list(order_pool.values())]
        customer_base: list[Customer] = [self._update_customer(customer)
                                         for customer in list(customer_base.values())]
        features: list[Feature] = [self._update_feature(feature)
                                   for feature in list(features.values())]

        entity_types_feature_clusters: dict[EntityType: list[FeatureCluster]] = {}
        for feature_cluster in list(feature_clusters.values()):
            feature_cluster_updated = self._update_feature_cluster(feature_cluster)
            entity_types_feature_clusters.setdefault(feature_cluster_updated.product_class,
                                                     []).append(feature_cluster_updated)
        feature_clusters = entity_types_feature_clusters

        return (entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                active_moving_resources, entity_transformation_nodes, processes, process_executions,
                order_pool, customer_base, features, feature_clusters)

    def instantiate_objects(self):
        entity_types = [_get_object_instantiated(entity_type_light)
                        for entity_type_light in self.digital_twin_model_light.entity_types]
        plant: Plant = _get_object_instantiated(self.digital_twin_model_light.plant)
        parts: dict[EntityType: list[Part]] = \
            {entity_type_id:
                 [_get_object_instantiated(part_light) for part_light in parts_light]
             for entity_type_id, parts_light in self.digital_twin_model_light.parts.items()}
        obstacles: list[Resource] = [_get_object_instantiated(obstacle)
                                     for obstacle in self.digital_twin_model_light.obstacles]
        stationary_resources: dict[EntityType:list[StationaryResource]] = \
            {entity_type_id:
                 [self._get_resource_instantiated(sr_light) for sr_light in sr_s_light]
             for entity_type_id, sr_s_light in self.digital_twin_model_light.stationary_resources.items()}
        passive_moving_resources: dict[EntityType:list[PassiveMovingResource]] = \
            {entity_type_id:
                 [self._get_resource_instantiated(pmr_light) for pmr_light in pmr_s_light]
             for entity_type_id, pmr_s_light in self.digital_twin_model_light.passive_moving_resources.items()}
        active_moving_resources: dict[EntityType:list[ActiveMovingResource]] = \
            {entity_type_id:
                 [self._get_resource_instantiated(amr_light) for amr_light in amr_s_light]
             for entity_type_id, amr_s_light in self.digital_twin_model_light.active_moving_resources.items()}
        entity_transformation_nodes = [_get_entity_transformation_node_object_instantiated(etn_light)
                                       for etn_light in self.digital_twin_model_light.entity_transformation_nodes]
        processes_ = reduce(concat, self.digital_twin_model_light.processes.values())
        process_dict = self._create_processes(processes_, entity_transformation_nodes)
        processes: dict[Process | ValueAddedProcess:list[Process]] = \
            {str(process_class):
                 [process_dict[pro["attr"]["identification"]] for pro in pro_s]
             for process_class, pro_s in self.digital_twin_model_light.processes.items()}
        process_executions: list[ProcessExecution] = \
            [_get_object_instantiated(pe)
             for pe in self.digital_twin_model_light.get_process_executions_list()]
        order_pool: list[Order] = [_get_object_instantiated(order)
                                   for order in self.digital_twin_model_light.order_pool]
        customer_base: list[Customer] = [_get_object_instantiated(customer_light)
                                         for customer_light in self.digital_twin_model_light.customer_base]
        features: list[Feature] = [_get_object_instantiated(feature)
                                   for feature in self.digital_twin_model_light.features]
        feature_clusters: dict[EntityType: list[FeatureCluster]] = \
            {entity_type_id:
                 [_get_object_instantiated(feature_cluster_light) for feature_cluster_light in feature_clusters_light]
             for entity_type_id, feature_clusters_light in self.digital_twin_model_light.feature_clusters.items()}

        return (entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                active_moving_resources, entity_transformation_nodes, processes, process_executions,
                order_pool, customer_base, features, feature_clusters)

    def create_all_objects_dict(self, entity_types, plant, parts, obstacles, stationary_resources,
                                passive_moving_resources, active_moving_resources, entity_transformation_nodes,
                                processes, process_executions, order_pool, customer_base, features, feature_clusters):
        all_objects = {}

        entity_types_dict = {entity_type.identification: entity_type for entity_type in entity_types}
        all_objects |= entity_types_dict

        if plant is not None:
            plant_dict = {plant.identification: plant}
            all_objects |= plant_dict

        part_dict = {part.identification: part for part in reduce(concat, list(parts.values()))}
        all_objects |= part_dict

        obstacle_dict = {obstacle.identification: obstacle for obstacle in obstacles}
        all_objects |= obstacle_dict

        if stationary_resources:
            stationary_resources_dict = {stationary_resource.identification: stationary_resource
                                         for stationary_resource in reduce(concat, list(stationary_resources.values()))}
            all_objects |= stationary_resources_dict

        if passive_moving_resources:
            passive_moving_resources_dict = \
                {passive_moving_resource.identification: passive_moving_resource
                 for passive_moving_resource in reduce(concat, list(passive_moving_resources.values()))}
            all_objects |= passive_moving_resources_dict

        if active_moving_resources:
            active_moving_resources_dict = \
                {active_moving_resource.identification: active_moving_resource
                 for active_moving_resource in reduce(concat, list(active_moving_resources.values()))}
            all_objects |= active_moving_resources_dict

        entity_transformation_nodes_dict = {entity_transformation_node.identification: entity_transformation_node
                                            for entity_transformation_node in entity_transformation_nodes}
        all_objects |= entity_transformation_nodes_dict

        if processes:
            processes_dict = {process.identification: process for process in reduce(concat, list(processes.values()))}
            all_objects |= processes_dict

        process_executions_dict = {process_execution.identification: process_execution
                                   for process_execution in process_executions}
        all_objects |= process_executions_dict

        order_dict = {order.identification: order for order in order_pool}
        all_objects |= order_dict

        customers_dict = {customer.identification: customer for customer in customer_base}
        all_objects |= customers_dict

        features_dict = {feature.identification: feature for feature in features}
        all_objects |= features_dict

        if feature_clusters:
            feature_clusters_dict = {feature_cluster.identification: feature_cluster
                                     for feature_cluster in reduce(concat, list(feature_clusters.values()))}
            all_objects |= feature_clusters_dict

        return all_objects

    def _create_processes(self, processes, entity_transformation_nodes):

        processes_dict = {}
        process_models = {}
        for pro in processes:
            process, process_models = _get_process_instantiated(pro, process_models, entity_transformation_nodes)
            processes_dict[process.identification] = process

        return processes_dict

    def _update_dynamic_attributes(self, object_):
        # where are the dynamic attributes needed ...?

        object_.attributes = {attr_name: self._get_dynamic_attribute_value(dynamic_array, value_type)
                              for attr_name, (dynamic_array, value_type) in object_.attributes.items()}

        return object_

    def _get_dynamic_attribute_value(self, dynamic_attribute_value, value_type):

        dynamic_attribute_value.recent_changes = (
            self._get_changes(dynamic_attribute_value.recent_changes, value_type))
        dynamic_attribute_value.distant_past_changes = (
            self._get_changes(dynamic_attribute_value.distant_past_changes, value_type))

        return dynamic_attribute_value

    def _get_changes(self, dynamic_attribute_change_tracker, value_type):
        if len(dynamic_attribute_change_tracker.changes["Value"]) == 0:
            return dynamic_attribute_change_tracker

        if value_type == "DigitalTwinObject":
            dynamic_attribute_change_tracker.changes["Value"] = (
                self.v_get_object_by_id(dynamic_attribute_change_tracker.changes["Value"]))

        dynamic_attribute_change_tracker.changes["ProcessExecution"] = (
            v_get_id(dynamic_attribute_change_tracker.changes["ProcessExecution"]))

        dynamic_attribute_change_tracker.changes[
            dynamic_attribute_change_tracker.changes != dynamic_attribute_change_tracker.changes] = None

        return dynamic_attribute_change_tracker

    def get_object_by_id(self, object_id):

        if object_id in self.all_objects:
            return self.all_objects[object_id]

        else:
            return object_id

    def _update_entity_type(self, entity_type):

        if entity_type.super_entity_type is not None:
            entity_type.super_entity_type = self.all_objects[entity_type.super_entity_type]

        return entity_type

    def _update_plant(self, plant):

        if plant.work_calendar is not None:
            plant.work_calendar = self.all_objects[plant.work_calendar]

        return plant

    def _update_entity(self, entity):

        if entity._entity_type is not None:
            entity._entity_type = self.all_objects[entity._entity_type]
        if entity._situated_in is not None:
            try:
                entity._situated_in = self.all_objects[entity._situated_in]
            except:
                print("Entity Dict:", entity.__dict__)

        entity.dynamic_attributes = self._update_dynamic_attributes(entity.dynamic_attributes)

        return entity

    def _update_part(self, part):

        part = self._update_entity(part)

        if part.part_of is not None:
            part.part_of = self.all_objects[part.part_of]
        part.parts = [self.all_objects[part_] for part_ in part.parts]

        return part

    def _update_physical_body(self, physical_body):
        physical_body.dynamic_attributes = self._update_dynamic_attributes(physical_body.dynamic_attributes)

        return physical_body

    def _update_process_execution_plan(self, process_execution_plan):

        # ToDo: work calendar

        return process_execution_plan

    def _update_resource(self, resource):

        resource = self._update_entity(resource)

        resource.physical_body = self._update_physical_body(resource.physical_body)
        resource._process_execution_plan = self._update_process_execution_plan(resource._process_execution_plan)

        if resource.plant is not None:
            resource.plant = self.all_objects[resource.plant]

        return resource

    def _update_storage_places(self, storage_places):

        storage_places._storage_places = {self.all_objects[entity_type_id]:
                                              [self.all_objects[storage_id] for storage_id in storage_places_]
                                          for entity_type_id, storage_places_ in storage_places._storage_places.items()}

        return storage_places

    def _update_stationary_resource(self, stationary_resource):

        stationary_resource = self._update_resource(stationary_resource)

        # react on differences
        if isinstance(stationary_resource, Storage):
            stationary_resource.stored_entities = [self.all_objects[stored_entity]
                                                   for stored_entity in stationary_resource.stored_entities]
            if stationary_resource.allowed_entity_type is not None:
                stationary_resource.allowed_entity_type = self.all_objects[stationary_resource.allowed_entity_type]

        elif isinstance(stationary_resource, WorkStation):
            stationary_resource._buffer_stations = self._update_storage_places(stationary_resource._buffer_stations)

        elif isinstance(stationary_resource, Warehouse):
            stationary_resource._storage_places = self._update_storage_places(stationary_resource._storage_places)

        return stationary_resource

    def _update_non_stationary_resource(self, non_stationary_resource):

        non_stationary_resource = self._update_resource(non_stationary_resource)

        non_stationary_resource._storage_places = self._update_storage_places(non_stationary_resource._storage_places)

        return non_stationary_resource

    def _update_entity_transformation_node(self, entity_transformation_node):

        if entity_transformation_node.entity_type is not None:
            entity_transformation_node.entity_type = self.all_objects[entity_transformation_node.entity_type]

        if entity_transformation_node.parents is not None:
            entity_transformation_node.parents = [self.all_objects[parent]
                                                  for parent in entity_transformation_node.parents]
        if entity_transformation_node.entity_type is not None:
            entity_transformation_node.children = [self.all_objects[child]
                                                   for child in entity_transformation_node.children]

        return entity_transformation_node

    def _update_process_models(self, processes):

        process_time_models = [process.lead_time_controller.process_time_model
                               for process in processes
                               if process.lead_time_controller.process_time_model.is_re_trainable()]

        process_time_models = list(set(process_time_models))
        for process_time_model in process_time_models:
            process_time_model.set_prediction_model()

        resource_groups = list(set(resource_group
                                   for process in processes
                                   for resource_group in process.resource_controller.resource_model.resource_groups))

        transition_models = list(set(process.transition_controller.transition_model
                                     for process in processes))

        for resource_group in resource_groups:
            resource_group.resources = [self.all_objects[resource]
                                        for resource in resource_group.resources]
            resource_group.main_resources = [self.all_objects[resource]
                                             for resource in resource_group.main_resources]

        for transition_model in transition_models:
            transition_model.possible_origins = \
                [self.all_objects[possible_origin]
                 for possible_origin in transition_model.possible_origins]
            transition_model.possible_destinations = \
                [self.all_objects[possible_destination]
                 for possible_destination in transition_model.possible_destinations]

    def _update_process(self, process):

        # ToDo: light_object = _update_list_attr_light_object(light_object, "group")

        if isinstance(process, ValueAddedProcess):
            process.feature = self.all_objects[process.feature]
            process.predecessors = [tuple([self.all_objects[predecessor_id]
                                           for predecessor_id in predecessor_lst])
                                    for predecessor_lst in process.predecessors]
            process.successors = [self.all_objects[successor_id]
                                  for successor_id in process.successors]

        return process

    def _update_process_execution(self, process_execution):

        process_execution._event_type = vars(ProcessExecution.EventTypes)[process_execution._event_type]
        if process_execution._main_resource is not None:
            process_execution._main_resource = self.all_objects[process_execution._main_resource]
        if process_execution._origin is not None:
            process_execution._origin = self.all_objects[process_execution._origin]
        if process_execution._destination is not None:
            process_execution._destination = self.all_objects[process_execution._destination]
        if process_execution._order is not None:
            process_execution._order = self.all_objects[process_execution._order]
        if (process_execution._connected_process_execution is not None and
                process_execution._connected_process_execution == process_execution._connected_process_execution):
            try:
                process_execution._connected_process_execution = (
                    self.all_objects[process_execution._connected_process_execution])
            except:
                print("Problem:", process_execution._connected_process_execution)
        else:
            process_execution._connected_process_execution = None

        return process_execution

    def _update_order(self, order):

        order.dynamic_attributes = self._update_dynamic_attributes(order.dynamic_attributes)
        order.features_completed = [self.all_objects[feature_completed]
                                    for feature_completed in order.features_completed]
        order.features_requested = [self.all_objects[feature_requested]
                                    for feature_requested in order.features_requested]
        if order.product_class:
            order.product_class = self.all_objects[order.product_class]
        if order.customer:
            try:
                customer = self.all_objects[order.customer]
                if isinstance(customer, Customer):
                    order.customer = customer
            except:
                if not isinstance(order.customer, Customer):
                    order.customer = None
                print("Warning - Customer initialization fails:", order.customer)
        if order.product is not None:
            order.product = self.all_objects[order.product]

        order.feature_process_execution_match = \
            {(self.all_objects[feature_id] if feature_id is not None else None):
                 [self.all_objects[process_execution_id]
                  for process_execution_id in process_executions]
             for feature_id, process_executions in order.feature_process_execution_match.items()}

        return order

    def _update_customer(self, customer):

        return customer

    def _update_feature(self, feature):

        if feature.feature_cluster is not None:
            feature.feature_cluster = self.all_objects[feature.feature_cluster]

        return feature

    def _update_feature_cluster(self, feature_cluster):

        feature_cluster.product_class = self.all_objects[feature_cluster.product_class]

        return feature_cluster

    def create_digital_twin(self, entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                            active_moving_resources, entity_transformation_nodes, processes, process_executions,
                            order_pool, customer_base, features, feature_clusters, next_id):

        digital_twin_model = StateModel(entity_types=entity_types, plant=plant, parts=parts,
                                        obstacles=obstacles, stationary_resources=stationary_resources,
                                        passive_moving_resources=passive_moving_resources,
                                        active_moving_resources=active_moving_resources,
                                        entity_transformation_nodes=entity_transformation_nodes,
                                        processes=processes, process_executions=process_executions,
                                        order_pool=order_pool, customer_base=customer_base,
                                        features=features, feature_clusters=feature_clusters)

        all_process_controllers = digital_twin_model.get_all_process_controllers()

        for process_controller in all_process_controllers:
            process_controller.set_digital_twin(digital_twin_model=digital_twin_model)

        DigitalTwinObject.next_id = next_id

        return digital_twin_model

    def from_pickle(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = self.pickle_path

        digital_twin_objects = load_from_pickle(pickle_path)

        return digital_twin_objects


if "__main__" == __name__:
    import os

    from ofact.planning_services.model_generation.twin_generator import StaticModelGenerator
    from ofact.settings import ROOT_PATH

    twin_scenario_name = ""

    digital_twin_file_name = f"base_general{twin_scenario_name}"
    digital_twin_pickle_path = \
        os.path.join(ROOT_PATH,
                     fr'DigitalTwin\projects\bicycle_world\scenarios\base\models\twin\{digital_twin_file_name}.pkl')
    digital_twin_objects = StaticModelGenerator.from_pickle(digital_twin_pickle_path=digital_twin_pickle_path)

    all_objects_digital_twin = digital_twin_objects.get_all_objects_digital_twin()
    digital_twin_model = digital_twin_objects.get_digital_twin()

    digital_twin_model_mapper = DigitalTwinModelMapper(digital_twin_model=digital_twin_model)
    digital_twin_objects_key_value = digital_twin_model_mapper.get_digital_twin_objects_as_key_value()
    pickle_path = os.path.join(ROOT_PATH, r"DigitalTwin/apps_tech/repository_services/digital_twin_model_test.pkl")
    digital_twin_model_mapper.to_pickle(pickle_path)

    light_digital_twin_model_mapper = LightDigitalTwinModelMapper(pickle_path=pickle_path)
    digital_twin_model = light_digital_twin_model_mapper.get_digital_twin_model_by_key_value()
