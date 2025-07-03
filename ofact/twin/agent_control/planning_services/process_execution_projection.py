"""
# TODO Add Module Description
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import (Warehouse, Storage, NonStationaryResource,
                                             WorkStation, ConveyorBelt, StationaryResource)
from ofact.twin.state_model.processes import ProcessExecution

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import EntityType


class ProcessExecutionsProjection:

    def __init__(self, resource):
        """
        The projection into the future is based on planned process_executions_components which determine the future behaviour
        :param resource: The projection object which is focussed
        """
        if not (isinstance(resource, Warehouse) or isinstance(resource, Storage) or
                isinstance(resource, NonStationaryResource) or isinstance(resource, WorkStation) or
                isinstance(resource, ConveyorBelt)):
            raise NotImplementedError

        self.resource = resource
        self.process_executions = np.array([], dtype=np.dtype([('timestamps', 'datetime64[s]'),
                                                               ('duration', 'timedelta64[s]'),
                                                               ('process_executions', 'object')]))

    def add_planned_process_execution(self, planned_process_execution: ProcessExecution):
        """Add planned process_executions_component to project their future impact for example for
        the available capacity of an entity"""

        # Maybe adding only the entries and exits would be the best option

        # ToDo: batch-wise update??? - impossible if more than one object negotiated in a proposal round - issues?
        #  (should be handled if necessary)

        # process_executions_components without a planned executed_start_time are ignored
        if not planned_process_execution.executed_start_time:
            return

        if self.resource not in planned_process_execution.get_resources():
            return

        timestamp_ = planned_process_execution.executed_start_time
        time_duration = planned_process_execution.get_process_lead_time()
        self.process_executions = \
            np.insert(self.process_executions, [0],
                      [(timestamp_, time_duration, planned_process_execution)], axis=0)

        # to ensure that the processes that have time are noted first, the duration is also considered
        self.process_executions = self.process_executions[self.process_executions[["timestamps", "duration"]].argsort()]

    def remove_planned_process_executions(self, process_executions_to_remove: list[ProcessExecution]):
        """Remove a list of process_executions_components because they lost their relevance for projections.
        Reasons can be rejections of proposals or actual_process_executions take place"""
        removed = False
        for process_execution_to_remove in process_executions_to_remove:
            idx = np.where(process_execution_to_remove ==
                           self.process_executions["process_executions"])

            if idx[0].size:
                self.process_executions = np.delete(self.process_executions, obj=idx, axis=0)
                removed = True
                # print(self.resource, "ProcessExecution Removed")

        # if not removed:
        #     pass

    def get_free_storage_capacities_at(self, entity_types: list[EntityType], at: datetime,
                                       duration: timedelta | None = None):
        """Get the capacity for a specific 'entity_type' 'at' a specific datetime"""
        if at is None:
            raise ValueError("Projection not possible")

        # Warehouse/ NonStationaryResource/ WorkStation
        # loading/ unloading/ transport (referred to the transport resource
        # no process_execution with origin/ dest. == resource)/ assembly
        already_executed_process_executions = []
        entry_exit_abs = {entity_type: 0 for entity_type in entity_types}
        for process_execution in self.process_executions["process_executions"]:
            # detect process_executions_components that are already finished and do not need to project in the future
            if process_execution.connected_process_execution:
                already_executed_process_executions.append(process_execution)
                continue

            if process_execution.executed_start_time > at:
                break  # ToDo: consider duration

            # query if resource is the origin or the destination
            if process_execution.origin.identification != process_execution.destination.identification:
                # did not remain in the same resource
                for entity_type_process_execution in process_execution.get_main_entity_types():
                    for entity_type in entity_types:
                        if entity_type.check_entity_type_match(entity_type_process_execution):
                            if self.resource.identification == process_execution.origin.identification:
                                entry_exit_abs[entity_type] -= 1
                            elif self.resource.identification == process_execution.destination.identification:
                                entry_exit_abs[entity_type] += 1

        if already_executed_process_executions:
            self.remove_planned_process_executions(already_executed_process_executions)

        available_capacities_now = {entity_type: self.resource.get_available_capacity_entity_type(entity_type)
                                    for entity_type in entity_types}

        capacities_at = {entity_type: available_capacities_now[entity_type] + single_entry_exit_abs
                         for entity_type, single_entry_exit_abs in entry_exit_abs.items()}

        return capacities_at

    def get_position_at(self, at: datetime):
        """Find the last process_execution before the datetime and
        return the destination of the determined process execution"""
        filter_ = np.where(at > self.process_executions["timestamps"])
        if filter_[0].shape[0]:
            index = filter_[0][-1]

            if isinstance(self.process_executions["process_executions"][index].destination, StationaryResource):
                resource = self.process_executions["process_executions"][index].destination

            else:
                resource = self.process_executions["process_executions"][index].origin
        else:
            resource = self.resource

        return resource.get_position()

    def get_positions_time_periods(self, start_time: datetime | None, end_time: datetime | None,
                                   time_slot_duration_restriction=None):
        """
        Determine the positions of the resource (self) in a time_period (additional mapping positions and
        free time periods/ unplanned time)
        :param start_time: the start_time of the time_period
        :param end_time: the end_time of the time_period
        :param time_slot_duration_restriction: a timeslot duration after a process_execution
        Used to filter the positions to possible start positions for a process_execution afterwards
        """
        if start_time is None:
            pass  # ToDo: get the current time

        index_start = self._get_index(timestamp=start_time)
        index_end = self._get_index(timestamp=end_time)

        if index_start is None and any(self.process_executions["timestamps"]):
            index_start = 0
        if index_end is None and any(self.process_executions["timestamps"]):
            index_end = len(self.process_executions["timestamps"]) - 1

        if index_start is not None and index_end is not None:
            resource_positions = []
            consider_first_origin = False
            process_execution = self.process_executions["process_executions"][index_start]
            if start_time is None or end_time is None:
                consider_first_origin = True
            elif process_execution.executed_time <= end_time:
                consider_first_origin = True
            if consider_first_origin:
                resource_positions.append(((None, process_execution.executed_start_time),
                                           process_execution.origin.get_position()))
                # ToDo: time_slot_duration_restriction should be used if #current_time available

            first_process_execution = self.process_executions["process_executions"][index_start]
            while index_start <= index_end:
                index_start += 1

                if index_start > index_end or index_end > len(self.process_executions["process_executions"]):
                    break

                second_process_execution = self.process_executions["process_executions"][index_start]

                slot_relevant = True
                if time_slot_duration_restriction:
                    real_time_slot = \
                        second_process_execution.executed_start_time - first_process_execution.executed_end_time

                    if not (real_time_slot >= time_slot_duration_restriction):
                        # non-stationary resources change their positions -
                        # therefore, the current positions are not usable
                        slot_relevant = False
                    if isinstance(first_process_execution.destination, NonStationaryResource):
                        slot_relevant = False

                if slot_relevant:
                    resource_positions.append(
                        ((first_process_execution.executed_end_time,
                          second_process_execution.executed_start_time),
                         first_process_execution.destination.get_position()))

                first_process_execution = second_process_execution

        else:
            resource_positions = [((start_time, end_time), self.resource.get_position())]

        resource_positions_unique = list(set(resource_positions))

        return resource_positions_unique

    def get_positions_within_time_period(self, start_time: datetime | None, end_time: datetime | None,
                                         time_slot_duration_restriction=None):
        """
        Determine the positions of the resource (self) in a time_period
        :param start_time: the start_time of the time_period
        :param end_time: the end_time of the time_period
        :param time_slot_duration_restriction: a timeslot duration after a process_execution
        Used to filter the positions to possible start positions for a process_execution afterwards
        """

        positions_time_periods = self.get_positions_time_periods(start_time, end_time, time_slot_duration_restriction)
        # take always the second element of each tuple in the list
        return list(list(zip(*positions_time_periods))[1])

    def _get_index(self, timestamp):
        """Determine the index from a numpy array"""
        if timestamp is not None:
            filter_ = np.where(timestamp >= self.process_executions["timestamps"])
            if filter_[0].shape[0]:
                return filter_[0][-1]
            else:
                return None
        else:
            return None

    def get_resource_for_part(self):
        """Used for long time reservation"""
        pass
