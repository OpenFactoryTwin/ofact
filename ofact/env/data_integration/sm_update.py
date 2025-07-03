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
    StateModelUpdating
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import operator
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

from ofact.env.model_administration.helper import get_attr_value

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.twin.change_handler.change_handler import ChangeHandlerPhysicalWorld
from ofact.twin.state_model.basic_elements import get_clean_attribute_name, ProcessExecutionTypes
from ofact.twin.state_model.entities import Resource
from ofact.twin.state_model.processes import ProcessExecution, ValueAddedProcess, WorkOrder
from ofact.twin.state_model.sales import Order

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class StateModelUpdate:

    def __init__(self, object_cache, change_handler: ChangeHandlerPhysicalWorld, state_model: StateModel,
                 artificial_simulation_need: bool, progress_tracker, cache, dtm):

        self.dtm = dtm

        self._object_cache = object_cache

        self._change_handler = change_handler
        self._state_model = state_model

        self.cache = cache

        self.order_work_order_mapping = {}

        # the work orders are needed to ensure that the features are completed based on the value added processes given
        # this is needed because more than one process can be needed for a feature
        self.feature_process_mapping = self._state_model.get_feature_process_mapper()

        self.resources_added = []  # checking needed if the resource should be added within process models  # ToDo

        self.artificial_simulation_need = artificial_simulation_need

        self.counter_success = 0
        self.counter_fails = 0

        self.progress_tracker = progress_tracker

    def add_change_handler(self, change_handler):
        self._change_handler = change_handler

    def get_process_model_updates(self) -> dict:
        return {}

    def update_process_models(self, process_models_updates, instantiated_resources):
        """Update the process_models with new objects available"""

        if "TransitionModel" not in process_models_updates:
            return

        add_as_origin_transition_controllers, add_as_destination_transition_controllers = (
            process_models_updates["TransitionModel"])

        resources_name_resource_mapping = {resource.name: resource
                                           for resource in instantiated_resources}

        transition_controller_additional_origins = {}
        for resource_name, transition_controllers in add_as_origin_transition_controllers.items():
            resource = resources_name_resource_mapping[resource_name]

            for transition_controller in transition_controllers:
                transition_controller_additional_origins.setdefault(transition_controller,
                                                                    []).append(resource)

        transition_controller_additional_destinations = {}
        for resource_name, transition_controllers in add_as_destination_transition_controllers.items():
            resource = resources_name_resource_mapping[resource_name]

            for transition_controller in transition_controllers:
                transition_controller_additional_destinations.setdefault(transition_controller,
                                                                         []).append(resource)

        for transition_controller, additional_resources in transition_controller_additional_origins.items():
            possible_origins = transition_controller.get_possible_origins()
            possible_origins += additional_resources

        for transition_controller, additional_resources in transition_controller_additional_destinations.items():
            possible_destinations = transition_controller.get_possible_destinations()
            possible_destinations += additional_resources

    def update_state_model(self):
        """Instantiate the objects from the _objects_cache, sort them in the right order to store them
        through the change_handler into the digital_twin"""

        self._instantiate_objects()
        dt_objects_cache = self._sort_cache()
        instantiated_resources = self._store_objects(dt_objects_cache)
        self.cache = {}

        # resulting evaluation of the procedure measured by the successful execution of process_executions
        failing_rate = round((self.counter_fails + 1e-12) / (self.counter_fails + self.counter_success + 1e-12), 3)
        print(f"[{datetime.now()}] Data Transformation Process ended with the instantiation and execution of objects.\n"
              f"    ProcessExecutions successful: {self.counter_success} \n"
              f"    ProcessExecutions failed: {self.counter_fails} \n"
              f"The failing rate of the executed data integration is therefore: {failing_rate}")

        return instantiated_resources

    def _instantiate_objects(self):
        """Instantiate the objects if they have all attributes needed (for example the order)"""

        v_instantiate_object_func = np.vectorize(self._instantiate_object)

        class_name, object_batch = self._object_cache.pop()
        while not self._object_cache.empty() or object_batch is not None:
            print(f"[{datetime.now()}] Create {class_name} instantiation dict ...")

            try:
                object_batch_length = len(object_batch)
            except:
                print(f"[{datetime.now()}] Create {class_name} instantiation dict ...")
                print(object_batch)
                raise Exception
            object_dict_to_creates = self._create_instantiation_dict_batch(class_name, object_batch_length,
                                                                           object_batch)

            print(f"[{datetime.now()}] Instantiate {class_name} objects ...")
            v_instantiate_object_func(np.repeat(class_name, object_batch_length),
                                      object_batch,
                                      object_dict_to_creates)

            class_name, object_batch = self._object_cache.pop()

    def _create_instantiation_dict_batch(self, class_name, object_batch_length, object_batch):
        v_create_instantiation_dict_func = np.vectorize(self._create_instantiation_dict)

        object_dict_to_creates = v_create_instantiation_dict_func(np.repeat(class_name, object_batch_length),
                                                                  object_batch)

        object_dict_to_creates = object_dict_to_creates[object_dict_to_creates != np.array(None)]

        return object_dict_to_creates

    def _create_instantiation_dict(self, type_, object_):
        """
        Create an instantiation dict which means the including of other objects already instantiated and referenced.
        """

        if type(object_) != dict:
            # the object is already stored in the digital twin
            # maybe the current data_transformation_batch could update the object
            self._change_handler.update_object(object_=object_)
            return

        object_attr_types = self._state_model.get_object_attributes(object_class_name=type_)

        object_dict_to_create = {key: self._get_key_value_pair(object_attr_types, key, value)[1]
                                 for key, value in object_.items()
                                 if self._get_key_value_pair(object_attr_types, key, value)[0] is not None}

        return object_dict_to_create

    def _get_key_value_pair(self, object_attr_types, key, value):
        """Create a new value for the key"""
        # determine if the attribute is filled with a value
        filled = True
        if type(value) == type:
            filled = False
        elif type(value) == str:
            if object_attr_types[key] == value:
                filled = False

        if not filled:
            return None, None

        if isinstance(value, dict) and "dict" not in object_attr_types[key]:
            try:
                cache_key = str(value["external_identifications"])
            except:
                raise Exception("External ID from the domain does not exist:", value, key, object_attr_types)
            if cache_key in self.cache:
                value = self.cache[cache_key]

            else:
                # create the object because the object is not created - should be instantiated first
                if "Order" in object_attr_types[key]:
                    object_dict_to_create_attr = \
                        self._create_instantiation_dict(type_="Order", object_=value)

                    self._instantiate_object(type_="Order", object_dict=value,
                                             object_para_dict=object_dict_to_create_attr)

                elif "Part" in object_attr_types[key] and not "PartType" in object_attr_types[key]:
                    object_dict_to_create_attr = \
                        self._create_instantiation_dict(type_="Part", object_=value)

                    self._instantiate_object(type_="Part", object_dict=value,
                                             object_para_dict=object_dict_to_create_attr)

                elif "EntityType" in object_attr_types[key]:
                    object_dict_to_create_attr = \
                        self._create_instantiation_dict(type_="EntityType", object_=value)

                    self._instantiate_object(type_="EntityType", object_dict=value,
                                             object_para_dict=object_dict_to_create_attr)

                else:
                    raise NotImplementedError(f'{object_attr_types} \n {key} \n {value}')

                if cache_key not in self.cache:
                    raise NotImplementedError

                value = self.cache[cache_key]

        elif isinstance(value, list):
            dicts_to_objects_required = [elem
                                         for elem in value
                                         if isinstance(elem, tuple)
                                         if isinstance(elem[0], dict)]

            if not value:
                return None, None
            # case parts_involved
            elif dicts_to_objects_required and "dict" not in object_attr_types[key]:

                if "parts_involved" in key:
                    new_values = list(set((self._create_new_values(dict_[0], "Part"),)
                                          if isinstance(dict_[0], dict) else dict_
                                          for dict_ in value
                                          if dict_ == dict_))

                    return key, new_values

                elif "resources_used" in key:
                    new_values = list(set((self._create_new_values(dict_[0], "StationaryResource"),)  # ToDo
                                          if isinstance(dict_[0], dict) else dict_
                                          for dict_ in value
                                          if dict_ == dict_))

                    return key, new_values

                raise NotImplementedError(key, value)

            if "parts_involved" in key or "resources_used" in key:
                try:
                    value = list(set(value))  # feature for example can be more than one of the same object
                except TypeError:
                    pass  # unique check not executed

        return key, value

    def _create_new_values(self, dict_, type_):
        """Instantiate the parts as attribute for the process_execution"""
        object_dict_to_create_attr = \
            self._create_instantiation_dict(type_=type_, object_=dict_)
        if object_dict_to_create_attr is None:
            return
        self._instantiate_object(type_=type_, object_dict=dict_,
                                 object_para_dict=object_dict_to_create_attr)

        cache_key = str(dict_["external_identifications"])
        new_values = self.cache[cache_key]

        return new_values

    def _instantiate_object(self, type_, object_dict, object_para_dict):
        """Instantiate the objects"""

        if object_para_dict is None:
            return

        if type_ == "Order":
            if 'features_requested' not in object_para_dict:
                object_para_dict['features_requested'] = []  # not available until now

        cache_key = str(object_dict["external_identifications"])
        if cache_key in self.cache:
            # the object already exist
            return

        dt_class = self._state_model.get_class_by_class_name(class_name=type_)
        if type_ == "ProcessExecution":
            object_para_dict["event_type"] = ProcessExecutionTypes.PLAN

        try:
            dt_object = dt_class(**object_para_dict)
        except TypeError as t:
            try:
                print(object_para_dict["external_identifications"])
            except KeyError:
                pass
            raise TypeError(f"The object of type '{type_}' cannot be instantiated. "
                            f"The object identifier should be found a row above. \n"
                            f"Parameter dict: '{object_para_dict}' \n"
                            f"Error: {t}")

        except AttributeError as a:
            try:
                print(object_para_dict["external_identifications"])
            except KeyError:
                pass
            raise AttributeError(f"The object of type '{type_}' cannot be instantiated. "
                                 f"The object identifier should be found a row above. \n"
                                 f"Parameter dict: '{object_para_dict}' \n"
                                 f"Error: {a}")

        except:
            print(f"Warning: The object of type {type_} is not instantiated. "
                  f"The object is therefore ignored in the follow up processes. \n"
                  f"The following parameters are in the parameter dict: {object_para_dict}.")
            return

        if cache_key in self.cache:
            if self.cache[cache_key] != dt_object:
                raise Exception
        self.cache[cache_key] = dt_object

    def _sort_cache(self):
        """map each object to his class"""
        dt_objects_cache = {}
        for dt_dict, dt_object in self.cache.items():
            dt_objects_cache.setdefault(dt_object.__class__,
                                        []).append(dt_object)

        return dt_objects_cache

    def _store_objects(self, dt_objects_cache):
        """Store the objects through the change_handler into the digital_twin"""
        instantiated_resources = []
        for dt_object_type, dt_objects_lst in dt_objects_cache.items():
            if dt_object_type == ProcessExecution:
                self.store_dynamic_objects(dt_objects_lst)

            else:
                instantiated_resources = self.store_static_objects(dt_objects_lst, instantiated_resources)

        return instantiated_resources

    def store_static_objects(self, dt_objects_lst, instantiated_resources):
        instantiated_resources += [self._instantiate_static_object(dt_object)
                                   for dt_object in dt_objects_lst
                                   if self._instantiate_static_object(dt_object) is not None]

        return instantiated_resources

    def _instantiate_static_object(self, dt_object):  # ToDo: refactor - _refine_static_object

        # ToDo: check the completeness is the task of the change_handler
        # completely_filled, not_completely_filled_attributes = dt_object_.completely_filled()
        #
        # if not completely_filled:
        #    print(f"[{datetime.now()}] {not_completely_filled_attributes}")

        self._change_handler.add_object(dt_object)

        if isinstance(dt_object, Resource):  # ensure that both sides mark the situated in...
            if dt_object.situated_in is not None:
                if not dt_object.situated_in.check_entity_stored(dt_object):
                    dt_object.situated_in.add_entity(dt_object)

        if isinstance(dt_object, Order):
            self._create_work_order(dt_object)
            print("Order:", dt_object.features_requested[0].name if dt_object.features_requested else None, dt_object.delivery_date_actual)

        elif isinstance(dt_object, Resource):
            return dt_object

    def store_dynamic_objects(self, dt_objects_lst: list[ProcessExecution]):
        """
        Store dynamic objects - because only process_executions are dynamic objects in the digital twin,
        they are named as process_executions
        """

        self._enforce_consistency(dt_objects_lst)

        # sort the process_executions_components according their executed_start_time
        # to ensure the execution in the right order
        process_executions_finished = \
            [process_execution
             for process_execution in dt_objects_lst
             if get_attr_value(process_execution, "executed_start_time") is not None]
        dt_objects_lst = sorted(process_executions_finished,
                                key=operator.attrgetter("executed_start_time"))

        self._change_handler.validate_process_chain_consistency(dt_objects_lst)

        # self.store_dynamic_objects_with_cut(dt_objects_lst)
        # return

        elements_created = len(dt_objects_lst)

        if tqdm is not None:
            process_executions_for_loop = tqdm(dt_objects_lst[:elements_created],
                                               colour="green", mininterval=30, maxinterval=300)
        else:
            process_executions_for_loop = dt_objects_lst[:elements_created]
        for idx, process_execution in enumerate(process_executions_for_loop):  # sequence important

            # Check process_execution is completed
            # completely_filled, not_completely_filled_attributes = process_execution.completely_filled()
            # if not completely_filled:
            #     print(f"[{datetime.now()}] Not completed attributes PE {not_completely_filled_attributes}")

            planned_process_execution, actual_process_execution, successful = (
                self._handle_process_execution_impacts(process_execution))

            # ToDo: find a better concept ...

            if successful:
                self.counter_success += 1
            else:
                # print(f"The execution of the process '{actual_process_execution.get_name()}' fails. \n"
                #       f"The process_execution has the following parameters: '{planned_process_execution.__dict__}'")
                self.counter_fails += 1

            planned_completely_filled = True
            if planned_process_execution is not None:
                try:
                    self._change_handler.add_planned_process_execution(planned_process_execution,
                                                                      completely_filled_enforced=False)
                except:
                    planned_completely_filled = False

            actual_completely_filled = True
            if actual_process_execution is not None:
                try:
                    self._change_handler.add_actual_process_execution(actual_process_execution,
                                                                     completely_filled_enforced=False)
                except:
                    actual_completely_filled = False

            if not (planned_completely_filled and actual_completely_filled) and successful:
                # correction
                self.counter_success -= 1
                self.counter_fails += 1

                # print(f"The execution of the process '{actual_process_execution.get_name()}' has not all attributes. \n"
                #       f"The process_execution has the following parameters: '{planned_process_execution.__dict__}'")
            if actual_process_execution:
                self._complete_features(planned_process_execution, actual_process_execution)

        # ToDo: used for testing the simulation by leave process_executions places empty ...

        process_executions_not_added = dt_objects_lst[elements_created:]

        orders_not_need_transport_anymore = []
        for process_execution in process_executions_not_added:  # sequence important
            order = process_execution.order

            if "_vap" in process_execution.process.name:
                orders_not_need_transport_anymore.append(order)

            if order in orders_not_need_transport_anymore:
                continue

            planned_process_execution, actual_process_execution, successful = \
                self._handle_process_execution_impacts(process_execution)

            # ToDo: use add_process_executions (batch processing would be faster ...)

            if planned_process_execution is not None:
                self._change_handler.add_planned_process_execution(planned_process_execution)
            if actual_process_execution is not None:
                self._change_handler.add_actual_process_execution(actual_process_execution)

            self._complete_features(planned_process_execution, actual_process_execution)

        self._change_handler.ensure_open_features_consistency(dt_objects_lst[:elements_created])

    def store_dynamic_objects_with_cut(self, dt_objects_lst: list[ProcessExecution]):
        """used for testing the simulation by leave process_executions places empty ..."""

        if self.artificial_simulation_need:
            elements_created = -1000
        else:
            elements_created = len(dt_objects_lst)
        last_update_idx = None
        consideration_end_time = datetime(2023, 7, 24)
        if tqdm is not None:
            process_executions_for_loop = tqdm(dt_objects_lst[:elements_created],
                                               colour="green", mininterval=30, maxinterval=300)
        else:
            process_executions_for_loop = dt_objects_lst[:elements_created]
        for idx, process_execution in enumerate(process_executions_for_loop):  # sequence important
            if process_execution.executed_start_time > consideration_end_time:  # for experiments
                continue
            else:
                last_update_idx = idx

            # Check process_execution is completed
            # completely_filled, not_completely_filled_attributes = process_execution.completely_filled()
            # if not completely_filled:
            #     print(f"[{datetime.now()}] Not completed attributes PE {not_completely_filled_attributes}")

            planned_process_execution, actual_process_execution, successful = (
                self._handle_process_execution_impacts(process_execution))

            # ToDo: find a better concept ...

            if successful:
                self.counter_success += 1
            else:
                # print(f"The execution of the process '{actual_process_execution.get_name()}' fails. \n"
                #       f"The process_execution has the following parameters: '{planned_process_execution.__dict__}'")
                self.counter_fails += 1

            planned_completely_filled = True
            if planned_process_execution is not None:
                try:
                    self._change_handler.add_planned_process_execution(planned_process_execution,
                                                                       completely_filled_enforced=False)
                except:
                    planned_completely_filled = False

            actual_completely_filled = True
            if actual_process_execution is not None:
                try:
                    self._change_handler.add_actual_process_execution(actual_process_execution,
                                                                      completely_filled_enforced=False)
                except:
                    actual_completely_filled = False

            if not (planned_completely_filled and actual_completely_filled) and successful:
                # correction
                self.counter_success -= 1
                self.counter_fails += 1

                # print(f"The execution of the process '{actual_process_execution.get_name()}' has not all attributes. \n"
                #       f"The process_execution has the following parameters: '{planned_process_execution.__dict__}'")
            if actual_process_execution:
                self._complete_features(planned_process_execution, actual_process_execution)

        if last_update_idx is not None:
            process_executions_not_added = dt_objects_lst[last_update_idx:]
            orders_not_need_transport_anymore = [process_execution.order
                                                 for process_execution in process_executions_not_added]
            print(len(orders_not_need_transport_anymore))
        else:
            process_executions_not_added = dt_objects_lst[elements_created:]

            orders_not_need_transport_anymore = []
            for process_execution in process_executions_not_added:  # sequence important
                order = process_execution.order

                if "_vap" in process_execution.process.name:
                    orders_not_need_transport_anymore.append(order)

                if order in orders_not_need_transport_anymore:
                    continue

                planned_process_execution, actual_process_execution, successful = \
                    self._handle_process_execution_impacts(process_execution)

                # ToDo: use add_process_executions (batch processing would be faster ...)

                if planned_process_execution is not None:
                    self._change_handler.add_planned_process_execution(planned_process_execution)
                if actual_process_execution is not None:
                    self._change_handler.add_actual_process_execution(actual_process_execution)

                self._complete_features(planned_process_execution, actual_process_execution)

        self._change_handler.ensure_open_features_consistency(dt_objects_lst[:elements_created])

        if orders_not_need_transport_anymore:
            for order in orders_not_need_transport_anymore:
                order.delivery_date_actual = None


                if None not in order.feature_process_execution_match:
                    if (order.release_date_actual is not None and
                            isinstance(consideration_end_time, datetime)):
                        if order.release_date_actual > consideration_end_time:
                            order.release_date_actual = None

            orders = self._state_model.get_orders()
            for order in orders:
                if not order.get_process_executions() and order.features_requested:
                    print("Order delivery date actual reset:", order.identifier)
                    order.delivery_date_actual = None
                if order.release_date_actual:
                    if order.release_date_actual > consideration_end_time:
                        order.release_date_actual = None

            self._state_model.update_orders_access_variables()

    def _enforce_consistency(self, process_executions):
        return process_executions

    def _create_work_order(self, sales_order):
        """Create a work order based on the sales order to mark features as done"""

        features_with_value_added_processes = sales_order.get_features_with_value_added_processes()

        value_added_processes_requested = \
            WorkOrder.convert_features_to_value_added_processes_requested(features_with_value_added_processes,
                                                                          self.feature_process_mapping)

        # create the production order
        work_order = WorkOrder(value_added_processes_completed={},
                               value_added_processes_requested=value_added_processes_requested,
                               order=sales_order)

        self.order_work_order_mapping[sales_order] = work_order

    def _handle_process_execution_impacts(self, planned_process_execution):
        """Crate an actual process_execution based on the planned one and execute the actual process_execution"""

        # Option 1: a planned process_execution is available
        # planned one should be already specified in the attribute connected_process_execution
        # Option 2: no planned process_execution is available
        # create a plan process_execution

        if planned_process_execution.executed_start_time is None:  # ToDo: how to handle them??? - ignore???
            actual_process_execution = None
            return planned_process_execution, actual_process_execution

        elif planned_process_execution.executed_end_time is None:
            executed_end_time = self._get_estimated_executed_end_time(planned_process_execution)
            planned_process_execution.executed_end_time = executed_end_time

        # How, from an actual process_execution the impacts can be derived
        actual_process_execution = \
            planned_process_execution.create_actual(source_application=planned_process_execution.source_application,
                                                    time_specification=True, enforce_time_specification=False,
                                                    end_time=True, from_plan=True)

        # assuming that the process_executions are ordered before execution
        # (currently used in the insertion new attribute values in the dynamic attributes ...)
        sequence_already_ensured = True
        # if True it is checked if the transitions are executed correctly
        # for the case that a resource is not at the place, modeled in the static model it would raise an exception
        # if transition_forced is True
        transition_forced = False
        try:
            actual_process_execution.execute(transition_forced=transition_forced,
                                             sequence_already_ensured=sequence_already_ensured)

            # transition_forced is exposed because for the first transition the origins can differ (because unknown)
            successful = True

        except:
            print(f"The process '{planned_process_execution.get_name()}' failed to execute")
            successful = False

        return planned_process_execution, actual_process_execution, successful

    def _get_estimated_executed_end_time(self, planned_process_execution):
        known_attributes = planned_process_execution.__dict__.copy()
        not_necessary_attributes = ["identification", "external_identifications", "_domain_specific_attributes",
                                    "_executed_start_time", "_executed_end_time", "_connected_process_execution"]
        known_attributes = {get_clean_attribute_name(known_attribute_name): value
                            for known_attribute_name, value in known_attributes.items()
                            if known_attribute_name not in not_necessary_attributes}
        try:
            estimated_process_time = planned_process_execution.process.get_estimated_process_lead_time(
                distance=0, **known_attributes)
            process_time = timedelta(seconds=estimated_process_time)
            executed_end_time = planned_process_execution.executed_start_time + process_time

        except:
            executed_end_time = self._get_estimated_executed_end_time_without_estimation(planned_process_execution)

        return executed_end_time

    def _get_estimated_executed_end_time_without_estimation(self, planned_process_execution):
        executed_end_time = None
        return executed_end_time

    def _complete_features(self, actual_process_execution, planned_process_execution):
        """
        Complete the feature of the order associated with the process executed.
        Only value_added_processes can complete orders.
        """

        if not actual_process_execution.order:
            return planned_process_execution, actual_process_execution

        sales_order: Order = actual_process_execution.order

        if isinstance(actual_process_execution.process, ValueAddedProcess):
            if sales_order not in self.order_work_order_mapping:
                self._create_work_order(sales_order)

            self.order_work_order_mapping[sales_order].complete_value_added_process(
                value_added_process_completed=actual_process_execution.process,
                process_executions=[actual_process_execution],
                sequence_already_ensured=True)  # assumption for better performance

        else:
            sales_order.add_process_execution(actual_process_execution)
