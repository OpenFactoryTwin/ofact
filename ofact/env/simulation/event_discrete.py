"""
This file encapsulates the event discrete simulation, triggered from the change handler, that passes process_executions
and later maybe orders and other objects to the simulation.
On the way back to the agents, the actual process_executions and orders are passed to the agents
(also through the change handler)
Note: for the shop_floor (if only sensor data available, the agent calls a service)
@last update: 1.8.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Union, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.env.environment import Environment
from ofact.env.simulation.entity_queue import EntitiesEventsQueue, ConveyorBeltsQueue
from ofact.env.helper import np_datetime64_to_datetime
from ofact.twin.state_model.entities import (
    Resource, StationaryResource, WorkStation, Warehouse, Storage, ConveyorBelt,
    NonStationaryResource, ActiveMovingResource, PassiveMovingResource)
from ofact.twin.state_model.processes import (ProcessExecution, ValueAddedProcess, Process, TransitionController,
                                              get_process_transition_type)

if TYPE_CHECKING:
    from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation
    from ofact.twin.state_model.time import ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt

    resources_type_hinting = (
        Union[Resource, StationaryResource, WorkStation, Warehouse, Storage, ConveyorBelt,
        NonStationaryResource, ActiveMovingResource, PassiveMovingResource])

logger = logging.getLogger("event_discrete_simulation")


def _update_possible_start_time_stamp(possible_start_time_stamp, possible_start_time_stamp_entity):
    if possible_start_time_stamp_entity is None:
        return possible_start_time_stamp

    if possible_start_time_stamp is None:
        possible_start_time_stamp = possible_start_time_stamp_entity
    elif possible_start_time_stamp_entity > possible_start_time_stamp:
        possible_start_time_stamp = possible_start_time_stamp_entity

    return possible_start_time_stamp


class EventDiscreteSimulation(Environment):
    """
    The event discrete simulation goes from event to event and effect the environment/ digital_twin.
    After each event batch, the simulation stops.
    Event batch: Events that are handled at the same time (same second are handled with one stop after execution)

    ToDo: include stochastic behaviour ...

    ToDo: in which cases the agents should be informed
    """

    def __init__(self, change_handler: ChangeHandlerSimulation, start_time, processes_to_stop: set[Process]):
        super(EventDiscreteSimulation, self).__init__(change_handler=change_handler, start_time=start_time)
        self.change_handler: ChangeHandlerSimulation

        # all processes where the simulation stops
        # normally not set in the init (therefore the set is empty),
        # but directly after it, when the agents are initialized
        self.processes_to_stop = processes_to_stop

        self.from_plan = False  # sample or get expected lead time

        self.simulation_queue_empty_counter = 0

        self.waiting_times_resources = {}

        # for differences between PLAN and ACTUAL process executions
        self._entity_event_queue_mapper: (
            Dict[resources_type_hinting, Union[EntitiesEventsQueue, ConveyorBeltsQueue]]) = \
            {Resource: EntitiesEventsQueue,
             StationaryResource: EntitiesEventsQueue,
             Storage: EntitiesEventsQueue,
             WorkStation: EntitiesEventsQueue,
             Warehouse: EntitiesEventsQueue,
             ConveyorBelt: ConveyorBeltsQueue,
             NonStationaryResource: EntitiesEventsQueue,
             ActiveMovingResource: EntitiesEventsQueue,
             PassiveMovingResource: EntitiesEventsQueue}

        # using the process_executions_plan as reference
        self._entity_events_queues: (
            Dict[Union[ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt],
            Union[EntitiesEventsQueue, ConveyorBeltsQueue]]) = {}

        self.process_executions_in_entity_event_queues = []

    def get_simulation_output(self) -> Dict:
        """return information about the simulation run"""
        simulation_output = {}

        return simulation_output

    def check_simulation_queue_empty(self):
        return self._planned_events_queue.empty()

    def execute_process_execution(self, process_execution: ProcessExecution,
                                  deviation_tolerance_time_delta: Optional[timedelta] = None,
                                  notification_time_delta: Optional[timedelta] = None):
        # print("Execute process",
        #     process_execution.get_process_name(),
        #       process_execution.executed_start_time,
        #       process_execution.executed_end_time)
        self._planned_events_queue.store_process_execution(
            process_execution=process_execution, type_="PLAN",
            deviation_tolerance_time_delta=deviation_tolerance_time_delta,
            notification_time_delta=notification_time_delta)

        # print(f"{datetime.now()} [{'Env':35} | {self.__class__.__name__:35}] Process Executions to simulate",
        #       len(self._planned_events_queue.process_executions_to_simulate))

        # update the dynamic attributes with planned process_execution effects
        process_execution.execute()

    async def simulate(self):
        # print(f"{datetime.now()} [{'Env':35} | {self.__class__.__name__:35}] simulate")

        if self._simulation_end_arrived():
            await self._end_simulation()
            return None

        queue_empty_stop = await self._queue_empty_stop()
        if queue_empty_stop:
            self.frequency_without_execution += 1
            return None

        self.frequency_without_execution = 0

        process_executions_actual = []
        while True:
            if self._planned_events_queue.empty():
                if process_executions_actual:
                    break
                else:
                    return None

            # iterating through the simulation queue and return if a change occurred
            (event_object, time_stamp, deviation_tolerance_time_delta, notification_time_delta,
             objects_for_notification) = (
                self._planned_events_queue.pop_next_event())

            if time_stamp is not None:
                self._current_time = time_stamp
            # print("simulate ...")
            process_execution_actual, time_stamp, objects_for_notification = (
                self.simulate_until_next_stop(event_object, time_stamp, objects_for_notification,
                                              deviation_tolerance_time_delta, notification_time_delta))

            if process_execution_actual is not None:
                process_executions_actual.append(process_execution_actual)

            # check termination
            if self._simulation_end_arrived():
                await self._end_simulation()
                return None

            if not process_executions_actual:
                continue

            # Assumption: only the value_added_process_execution are relevant for the order agents to
            process_execution_to_stop = self._process_execution_to_stop(process_executions_actual[-1])
            if process_execution_to_stop:
                break

        if objects_for_notification:
            self.change_handler.set_notification_for_executions_ending_soon(objects_for_notification)
        self.change_handler.add_actual_process_executions(actual_process_executions=process_executions_actual)

        print(f"{datetime.now().time()} [{'Env':35} | {self.__class__.__name__:35}] Time: {self._current_time} -"
              f" {self._planned_events_queue.get_queue_length()} (Queue Level)")

        return process_executions_actual

    async def _queue_empty_stop(self):
        if not self._planned_events_queue.empty():
            self.simulation_queue_empty_counter = 0

            return False

        self.simulation_queue_empty_counter += 1
        print(f"{datetime.now()} [{self.__class__.__name__:35}] Simulation waiting for new process_executions")
        if self.simulation_queue_empty_counter >= 50:
            print("End the simulation because no new events are added to the event_queue ...")
            await self.change_handler.end_simulation(end_simulation_agents=True)

        return True

    def _process_execution_to_stop(self, actual_process_execution):

        if (isinstance(actual_process_execution.process, ValueAddedProcess) or
                actual_process_execution.process in self.processes_to_stop):
            process_execution_to_stop = True
        else:
            process_execution_to_stop = False

        return process_execution_to_stop

    def simulate_until_next_stop(self, event_object, time_stamp, objects_for_notification,
                                 deviation_tolerance_time_delta: Optional[timedelta],
                                 notification_time_delta: Optional[timedelta]):
        """Simulate until actual process_execution is executed"""

        process_execution_actual = None
        while event_object is not None:

            if isinstance(event_object, ProcessExecution):
                process_execution_actual = (
                    self._handle_process_execution(process_execution=event_object, start_time=time_stamp,
                                                   deviation_tolerance_time_delta=deviation_tolerance_time_delta,
                                                   notification_time_delta=notification_time_delta))

                if process_execution_actual is not None:
                    break

                # take the next event until an actual process execution is available
                (event_object, time_stamp, deviation_tolerance_time_delta, notification_time_delta,
                 objects_for_notification) = (
                    self._planned_events_queue.pop_next_event(time_stamp))

        return process_execution_actual, time_stamp, objects_for_notification

    def _handle_process_execution(self, process_execution: ProcessExecution, start_time,
                                  deviation_tolerance_time_delta: Optional[timedelta],
                                  notification_time_delta: Optional[timedelta]) -> Optional[ProcessExecution]:
        """Introduce a different handling of actual and plan process_executions"""

        if process_execution.event_type == ProcessExecution.EventTypes.ACTUAL:
            process_execution_actual = (
                self._handle_actual_process_execution(process_execution_actual=process_execution,
                                                      deviation_tolerance_time_delta=deviation_tolerance_time_delta))

        else:
            self._handle_plan_process_execution(process_execution_plan=process_execution,
                                                start_time=start_time,
                                                deviation_tolerance_time_delta=deviation_tolerance_time_delta,
                                                notification_time_delta=notification_time_delta)
            process_execution_actual = None

        return process_execution_actual

    def _handle_plan_process_execution(self, process_execution_plan: ProcessExecution, start_time,
                                       deviation_tolerance_time_delta: Optional[timedelta],
                                       notification_time_delta: Optional[timedelta],
                                       next_event: bool = True):
        """
        Set the process_execution in the entity events queues and execute the process execution plan.
        After that the process_execution is stored in the planned_events_queue for actual execution.
        :param process_execution_plan: a process execution with event type PLAN
        :param start_time: start_time of the process execution
        :param deviation_tolerance_time_delta: ToDo
        :param next_event: mean the process_execution would be the next event in the planned events queue,
        which is not the case if a process execution is tried to execute before the planned start time
        """
        # print("PLAN PE:",  # process_execution_plan.executed_start_time, process_execution_plan.executed_end_time,
        #       process_execution_plan.get_name(), process_execution_plan.order.external_identifications)

        # could be done also after checking if shifting needed
        already_in_queue = True
        if process_execution_plan not in self.process_executions_in_entity_event_queues:
            self._set_process_execution_in_entity_events_queues(process_execution=process_execution_plan,
                                                                start_time=start_time,
                                                                event_type="PLAN")
            already_in_queue = False

        all_required_entities_available = self.check_entities_availability(process_execution_plan, start_time,
                                                                           deviation_tolerance_time_delta,
                                                                           notification_time_delta, next_event)
        if not all_required_entities_available:
            if not next_event and not already_in_queue:
                # remove entries from entity event queues
                print("Remove entries from entity event queues")
                self._remove_process_execution_from_entity_event_queue(process_execution_plan)
            return

        actual_process_execution = \
            self._create_actual_process_execution(plan_process_execution=process_execution_plan,
                                                  source_application="Event discrete simulation",
                                                  executed_start_time=start_time,
                                                  time_execution=True, from_plan=self.from_plan, end_time=True)

        if not next_event:
            self._planned_events_queue.remove_event(event=process_execution_plan)

        self._planned_events_queue.store_process_execution(
            process_execution=actual_process_execution, type_="ACTUAL",
            deviation_tolerance_time_delta=deviation_tolerance_time_delta,
            notification_time_delta=notification_time_delta)

        self._remove_process_execution_from_entity_event_queue(process_execution_plan)
        self._set_process_execution_in_entity_events_queues(process_execution=actual_process_execution,
                                                            start_time=start_time, event_type="ACTUAL",
                                                            resources_from_plan=True)

    def _set_process_execution_in_entity_events_queues(self, process_execution, start_time, event_type,
                                                       resources_from_plan=False):
        """
        Set the process_execution in the entity events queues to handle disturbances or
        different lead times in PLAN and ACTUAL process execution.
        :param process_execution: a process_execution with event type PLAN
        :param resources_from_plan: resources from process execution not available until now
        """
        process_execution_plan = process_execution.get_plan_process_execution()
        if not resources_from_plan:
            resources = process_execution.get_resources()
        else:

            resources = process_execution_plan.get_resources()

        for resource in resources:
            if resource.process_execution_plan not in self._entity_events_queues:
                entity_events_queue: EntitiesEventsQueue = self._entity_event_queue_mapper[resource.__class__]
                self._entity_events_queues[resource.process_execution_plan] = entity_events_queue(entity=resource)

            resource_event_queue = self._entity_events_queues[resource.process_execution_plan]
            resource_event_queue.set_entity(entity=resource)  # ensure that the resource set in event queue
            resource_event_queue.store_process_execution(process_execution=process_execution,
                                                         time_stamp=process_execution_plan.executed_start_time,
                                                         type_=event_type)
            resource_event_queue.update_event(event=process_execution,
                                              time_stamp=start_time)

        self.process_executions_in_entity_event_queues.append(process_execution)  # needed for the other PE's

    def check_entities_availability(self, process_execution: ProcessExecution, start_time,
                                    deviation_tolerance_time_delta: Optional[timedelta],
                                    notification_time_delta: Optional[timedelta],
                                    next_event: bool = True) -> bool:
        """
        Check if all required entities are available at the start_time.
        If not the process execution is shifted into the future
        :param next_event: False if a later process_execution is tried to give a priority because
        e.g. the process_execution is ended before the planned end time
        """

        available = True

        # check resources availability
        possible_start_time_stamp_resource = (
            self._get_possible_start_time_stamp_resources(process_execution=process_execution,
                                                          time_now=start_time))
        possible_start_time_stamp_part = (
            self._get_possible_start_time_stamp_parts(process_execution=process_execution,
                                                      time_now=start_time))

        possible_start_time_stamp64 = deepcopy(possible_start_time_stamp_resource)
        if possible_start_time_stamp_part is not None:
            if possible_start_time_stamp64 is None:
                possible_start_time_stamp64 = possible_start_time_stamp_part
            elif possible_start_time_stamp64 < possible_start_time_stamp_part:
                possible_start_time_stamp64 = possible_start_time_stamp_part

        if possible_start_time_stamp64 is None:
            return available

        # Note: loose the nanosecond entries
        possible_start_time_stamp = np_datetime64_to_datetime(possible_start_time_stamp64)
        # equal means that a process need to be executed before that "does not consume time"
        if start_time <= possible_start_time_stamp and next_event:
            if deviation_tolerance_time_delta is not None:
                print(f'deviation_tolerance_time_delta:{deviation_tolerance_time_delta}')
                possible_start_time_stamp.replace(microsecond=possible_start_time_stamp.microsecond + 1)
                #deviation_tolerance_time_delta=deviation_tolerance_time_delta+pd.Timedelta(minutes=1)
                return available
                #pass  # ToDo: comparison with ... not implemented until now
            # right order in the queue
            # possible_start_time_stamp = (
            #     possible_start_time_stamp.replace(microsecond=possible_start_time_stamp.microsecond + 1))

            print(f"Shifting '{process_execution.get_process_name()}' from {start_time} to {possible_start_time_stamp}")
            resources = process_execution.get_resources()
            possible_start_time_stamp = None

            for resource in resources:
                queue=self._entity_events_queues[resource.process_execution_plan]
                print(queue)

            self._planned_events_queue.store_process_execution(
                process_execution=process_execution, time_stamp=possible_start_time_stamp, type_="PLAN",
                deviation_tolerance_time_delta=deviation_tolerance_time_delta,
                notification_time_delta=notification_time_delta)

            resources = process_execution.get_resources()
            for resource in resources:
                resource_event_queue = self._entity_events_queues[resource.process_execution_plan]
                resource_event_queue.update_event(process_execution, possible_start_time_stamp)

            # print([(event.executed_start_time, event.identification, event.get_process_name(),
            #         self._planned_events_queue._event_time_stamps[index])
            #        for index, event in enumerate(self._planned_events_queue._events_to_simulate)])

            available = False

        return available

    def _get_possible_start_time_stamp_resources(self, process_execution: ProcessExecution, time_now) -> (
            Optional[np.datetime64]):
        """
        If no process_execution before, the return value must be None to avoid a false oder
        :return: if None no restriction found
        """
        resources = process_execution.get_resources()
        possible_start_time_stamp = None

        for resource in resources:
            entity_events_queue: EntitiesEventsQueue = self._entity_events_queues[resource.process_execution_plan]

            possible_start_time_stamp_resource = (
                entity_events_queue.get_possible_start_time_stamp(process_execution, time_now))

            possible_start_time_stamp = _update_possible_start_time_stamp(possible_start_time_stamp,
                                                                          possible_start_time_stamp_resource)

        return possible_start_time_stamp

    def _get_possible_start_time_stamp_parts(self, process_execution: ProcessExecution, time_now) -> (
            Optional[np.datetime64]):
        """
        :return: if None no restriction found
        """

        parts = process_execution.get_parts()
        main_resource = process_execution.main_resource
        origin = process_execution.origin
        destination = process_execution.destination
        process_transition_type, intersection = get_process_transition_type(origin=origin, destination=destination,
                                                                            class_name=self.__class__.__name__)
        possible_start_time_stamp = None
        for part in parts:
            part_found = False
            if process_transition_type == TransitionController.Types.TRANSFER and origin is None:
                # Case entities are fed in the system
                continue

            possible_start_time_stamp = None
            if (process_transition_type == TransitionController.Types.TRANSFER or
                    process_transition_type == TransitionController.Types.NO_TRANSITION):
                # Case entities are fed in the system
                part_found, possible_start_time_stamp_part = (
                    self.get_part_availability_time(origin, part, process_execution, time_now))

                if part_found:
                    possible_start_time_stamp = (
                        _update_possible_start_time_stamp(possible_start_time_stamp, possible_start_time_stamp_part))
                    continue

            if process_transition_type == TransitionController.Types.TRANSPORT:
                # Case entities are fed in the system
                part_found, possible_start_time_stamp_part = (
                    self.get_part_availability_time(main_resource, part, process_execution, time_now))

                if part_found:
                    possible_start_time_stamp = _update_possible_start_time_stamp(possible_start_time_stamp,
                                                                                  possible_start_time_stamp_part)
                    continue

            resources = process_execution.get_resources()  # Evaluation needed ...
            resources = list(set(resources) - {main_resource})

            for resource in resources:
                part_found, possible_start_time_stamp_part = (
                    self.get_part_availability_time(resource, part, process_execution, time_now))
                if part_found:
                    break

            if not part_found:
                # return None
                raise Exception(f"[{process_execution.identification} | "
                                f"{process_execution.order.external_identifications}] \n"
                                f"part not found {part} - {part.identification} in {process_execution.process} \n"
                                f"of transition type {process_transition_type} from {origin} to {destination} \n"
                                f"Order {process_execution.order.identification}")

        return possible_start_time_stamp

    def get_part_availability_time(self, resource, part, process_execution, time_now):
        possible_start_time_stamp_part = None
        part_found = False
        if resource is None:
            return part_found, possible_start_time_stamp_part
        # if resource.process_execution_plan not in self._entity_events_queues:
        #     resource = part.situated_in

        resource_events_queue = self._entity_events_queues[resource.process_execution_plan]
        part_found, possible_start_time_stamp_part = (
            resource_events_queue.get_parts_availability_time(resource, part, process_execution, time_now))

        return part_found, possible_start_time_stamp_part

    def _handle_actual_process_execution(self, process_execution_actual: ProcessExecution,
                                         deviation_tolerance_time_delta: Optional[timedelta]):
        """Handle the actual process execution"""

        # print("Actual PE:", process_execution_actual.identification,
        #       process_execution_actual.executed_start_time, process_execution_actual.executed_end_time,
        #       process_execution_actual.get_name(), process_execution_actual.order.external_identifications)

        process_execution_actual.execute()
        end_time_deviation_from_plan = process_execution_actual.get_end_time_deviation_from_plan()
        negative_deviation_to_handle_within_simulation = False
        if abs(end_time_deviation_from_plan) > timedelta(0):
            if abs(end_time_deviation_from_plan) < deviation_tolerance_time_delta:
                if end_time_deviation_from_plan < timedelta(0):
                    negative_deviation_to_handle_within_simulation = True
            else:
                pass  # ToDo: inform the agent who will maybe take another option
        # print("Process Name Execution:", process_execution_actual.process.name,
        #       process_execution_actual.executed_start_time, process_execution_actual.executed_end_time)
        self._remove_process_execution_from_entity_event_queue(process_execution_actual, resources_from_plan=False)
        if not negative_deviation_to_handle_within_simulation:
            return process_execution_actual

        process_executions_to_recheck = []
        resources = process_execution_actual.get_resources()
        for resource in resources:
            entity_events_queue: EntitiesEventsQueue = self._entity_events_queues[resource.process_execution_plan]

            event_object, time_stamp64 = entity_events_queue.get_next_event()
            if event_object:
                time_stamp = np_datetime64_to_datetime(time_stamp64)
                if event_object.check_actual():
                    continue
                process_executions_to_recheck.append((event_object, time_stamp))

        if process_executions_to_recheck:
            print("PE to recheck")
            self._try_to_bring_process_executions_forward(process_executions_to_recheck)

        return process_execution_actual

    def _remove_process_execution_from_entity_event_queue(self, process_execution, resources_from_plan=False):
        """
        Remove the event from the entities event queues
        :param resources_from_plan: if True, the resources are taken from the plan
        """

        if not resources_from_plan:
            resources = process_execution.get_resources()
        else:
            # resources from process execution not available until now
            process_execution_plan = process_execution.get_plan_process_execution()
            resources = process_execution_plan.get_resources()

        for resource in resources:
            entity_events_queue: EntitiesEventsQueue = self._entity_events_queues[resource.process_execution_plan]
            entity_events_queue.remove_event(process_execution)

        self.process_executions_in_entity_event_queues.remove(process_execution)

    def _try_to_bring_process_executions_forward(self, process_executions_to_recheck: (
            List[Tuple[ProcessExecution, datetime]])):
        """
        Iterate through the process executions and try to prioritize them because
        e.g. the process execution ended before the planned end time,
        respectively the start time of the follow-up process execution can be earlier, if possible.
        :param process_executions_to_recheck: a list with the
        :return:
        """

        for process_execution, possibly_new_start_time in process_executions_to_recheck:
            deviation_tolerance_time_delta = (
                self._planned_events_queue.get_deviation_tolerance_time_delta(process_execution))
            notification_time_delta = self._planned_events_queue.get_notification_time_delta(process_execution)


            self._handle_plan_process_execution(process_execution_plan=process_execution,
                                                start_time=possibly_new_start_time,
                                                deviation_tolerance_time_delta=deviation_tolerance_time_delta,
                                                notification_time_delta=notification_time_delta,
                                                next_event=False)

    # #### SIMULATION END ##############################################################################################

    def _simulation_end_arrived(self):
        simulation_end_arrived = False
        if self._time_limit is not None:
            if self._current_time > self._time_limit:
                simulation_end_arrived = True

        return simulation_end_arrived

    async def _end_simulation(self):
        print(f"End the simulation because time limit '{self._time_limit}' reached "
              f"with current time '{self._current_time}'...")
        await self.change_handler.end_simulation(end_simulation_agents=True)

    # not used

    def _set_waiting_times(self, process_execution):
        waiting_time_delta = process_execution.executed_start_time - \
                             process_execution.connected_process_execution.executed_start_time
        waiting_time_delta_seconds = waiting_time_delta.seconds
        participating_resources = process_execution.get_resources()
        for participating_resource in participating_resources:
            self.waiting_times_resources.setdefault(participating_resource,
                                                    []).append(waiting_time_delta_seconds)

