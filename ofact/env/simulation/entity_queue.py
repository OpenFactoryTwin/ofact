"""
Used to handle differences between PLAN and ACTUAL process executions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Set, Dict

import numpy as np

from ofact.env.helper import np_datetime64_to_datetime
from ofact.twin.state_model.entities import ConveyorBelt
from ofact.env.planned_event_queue import EventsQueue

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Entity
    from ofact.twin.state_model.processes import ProcessExecution


class EntitiesEventsQueue(EventsQueue):
    """
    An EntitiesEventsQueue is used for one process execution plan.
    This means that more than one entity can share their event queue with each other.
    Use case:
    A work station has its own event queue and all its storage places a second one.
    This ensures that not more than one material supply can enter the storage places
    but the work is separated in a parallel event queue.

    The events_to_simulate contain only start time stamp to ensure the planned sequence.
    This means, that it is assumed that the order is always the planned one ...
    """

    def __init__(self, entity):
        super().__init__()
        self._entities: Set[Entity] = {entity}

        # idea: the planned event time stamp is stored in the event_time_stamps array
        # to leave the order as planned, the updated time stamp is stored in the event_time_stamps_updated array
        self._event_time_stamps_updated: Dict[ProcessExecution, datetime] = {}
        self.event_expected_process_times = {}

    def set_entity(self, entity):
        self._entities.update({entity})

    def store_process_execution(self, process_execution: ProcessExecution, time_stamp: Optional[datetime] = None,
                                type_: str = "PLAN", deviation_tolerance_time_delta: Optional[timedelta] = None):
        super().store_process_execution(process_execution=process_execution, time_stamp=time_stamp, type_=type_,
                                        deviation_tolerance_time_delta=deviation_tolerance_time_delta)
        self._event_time_stamps_updated[process_execution] = time_stamp

        process_time_seconds = process_execution.get_process_lead_time()
        self.event_expected_process_times[process_execution] = (
            timedelta(seconds=process_time_seconds))

    def update_event(self, event, time_stamp):
        self._event_time_stamps_updated[event] = time_stamp

    def remove_event(self, event):
        super().remove_event(event=event)
        del self.event_expected_process_times[event]
        del self._event_time_stamps_updated[event]

    def get_next_event(self) -> [Optional[ProcessExecution], Optional[datetime]]:
        if not self._events_to_simulate:
            return None, None

        event_object = self._events_to_simulate[0]
        time_stamp = self._event_time_stamps[0]
        return event_object, time_stamp

    def get_possible_start_time_stamp(self, process_execution, time_now) -> Optional[np.datetime64]:
        """return the executed end time from the process_execution before ..."""

        if not self._events_to_simulate:
            raise Exception

        if self._events_to_simulate[0] == process_execution:
            return None

        # print("Length Event Queue:", len(self._events_to_simulate))
        start_time_stamp = time_now
        if isinstance(time_now, np.datetime64):
            start_time_stamp = np_datetime64_to_datetime(time_now)

        for index, event in enumerate(self._events_to_simulate):
            if event == process_execution:
                break
            event_end_time = self._event_time_stamps_updated[event] + self.event_expected_process_times[event]
            if event_end_time < start_time_stamp and event.check_plan():
                start_time_stamp += self.event_expected_process_times[event]
            else:
                start_time_stamp = event_end_time

        start_time_stamp = np.datetime64(start_time_stamp, "ns")
        return start_time_stamp

    def get_parts_availability_time(self, resource, part, process_execution, time_now) -> \
            [bool, Optional[np.datetime64]]:
        possible_start_time_stamp_part = None

        part_found = resource.check_entity_stored(part)

        if part_found:
            return part_found, possible_start_time_stamp_part

        entity_type_storable = resource.check_entity_type_storable(part.entity_type)
        if not entity_type_storable:
            return part_found, possible_start_time_stamp_part

        # consider the given process_executions for the part
        possible_start_time_stamp_part = self._get_part_delivery_start_time_stamp(resource, part, process_execution,
                                                                                  time_now)
        if possible_start_time_stamp_part is not None:
            part_found = True

        return part_found, possible_start_time_stamp_part

    def _get_part_delivery_start_time_stamp(self, resource, part, process_execution, time_now) -> (
            Optional[np.datetime64]):
        """
        Determine the process_execution before that stores the part in the resource.
        :param resource:
        :param part:
        :param process_execution:
        :return:
        """
        if self._events_to_simulate[0] == process_execution:
            return None

        possible_resources = [resource] + resource.get_storages_without_entity_types()
        start_time_stamp = time_now
        for index, event in enumerate(self._events_to_simulate):
            if event == process_execution:
                start_time_stamp = np.datetime64(start_time_stamp, "ns")
                return start_time_stamp

            event_end_time = self._event_time_stamps_updated[event] + self.event_expected_process_times[event]
            if event_end_time < start_time_stamp and event.check_plan():
                start_time_stamp += self.event_expected_process_times[event]
            else:
                start_time_stamp = event_end_time

            if part not in process_execution.get_parts():
                continue

            if process_execution.destination in possible_resources:
                start_time_stamp = np.datetime64(start_time_stamp, "ns")
                return start_time_stamp

        return None

    def _get_end_time_stamp_by_index(self, index) -> np.datetime64:
        """Return an end_time_stamp of the process_execution"""

        process_execution: ProcessExecution = self._events_to_simulate[index]
        end_time_stamp: np.datetime64 = np.datetime64(self._event_time_stamps_updated[process_execution], "ns")
        end_time_stamp += np.timedelta64(process_execution.get_process_lead_time())

        return end_time_stamp


class ConveyorBeltsQueue(EntitiesEventsQueue):

    def __init__(self, entity: ConveyorBelt):
        super().__init__(entity=entity)

        self.time_interval = entity.process_execution_plan.time_interval

        # check how many entities are on the conveyor belt and when the next entity leave the belt
        # that should be checked in the planning before!

    def get_possible_start_time_stamp(self, process_execution, time_now) -> Optional[np.datetime64]:
        """
        return the executed end time from the process_execution before ...
        Conveyor Belts have a little different logic because the executed_start is mapped to the point
        where the entity to transport start driving and the end, where the entity to transport end driving.
        This means that the next entity to transport can already start
        while another entity drives over the conveyor belt.
        To manage the different behaviour, from the transport processes, the start time end is taken.
        This time is combined with the time interval that is required between two entities on the conveyor belt.
        """
        # ToDo: lead time with variance not considered

        if not self._events_to_simulate:
            raise Exception

        if self._events_to_simulate[0] == process_execution:
            return None

        for index, event in enumerate(self._events_to_simulate):
            if event == process_execution:
                break

        process_execution_before = self._events_to_simulate[index - 1]

        if (isinstance(process_execution_before.origin, ConveyorBelt) or
                isinstance(process_execution_before.destination, ConveyorBelt)):
            # transfer from/ to conveyor_belt
            possible_start_time_stamp = self._get_end_time_stamp_by_index(index - 1)

        else:
            last_start = self._get_start_time_stamp_by_index(index - 1)
            last_start = last_start + np.timedelta64(int(self.time_interval * 1e9), "ns")
            possible_start_time_stamp = last_start

        if possible_start_time_stamp <= np.datetime64(time_now, "ns"):
            return None

        return possible_start_time_stamp

    def _get_start_time_stamp_by_index(self, index) -> np.datetime64:
        """Return the start_time_stamp of the process_execution"""
        process_execution: ProcessExecution = self._events_to_simulate[index]
        start_time_stamp = np.datetime64(self._event_time_stamps_updated[process_execution], "ns")

        return start_time_stamp
