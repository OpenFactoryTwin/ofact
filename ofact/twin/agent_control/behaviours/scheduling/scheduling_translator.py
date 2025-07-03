"""
Translates the central scheduling tasks to allow schedulers schedule them ...
Within this part, python objects are broken down from python references to ids ...
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import dill as pickle
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import Resource, Part, NonStationaryResource
from ofact.twin.state_model.processes import ProcessExecution

if TYPE_CHECKING:
    pass


def convert_distance_matrix(distance_matrix):
    distance_matrix = {(origin._process_execution_plan.identification,
                        destination._process_execution_plan.identification):
                           distance
                       for (origin, destination), distance in distance_matrix.items()}

    return distance_matrix


class HeuristicSchedulingInterface:
    """
    Change the input format given by the agents to an output format required by a scheduler.
    If the scheduler end the scheduling procedure, the solution is translated back again.
    """

    def __init__(self, distance_matrix):
        self.distance_matrix = convert_distance_matrix(distance_matrix)
        self.batch_number = 0

    def get_scheduling_input(self, resources_process_executions_components, resources_preferences, routing_service,
                             digital_twin, start_time_stamp):

        if not resources_process_executions_components:
            return {}

        # ToDo: What is about connectors
        # ToDo: What is about resources sharing a

        scheduling_interval_start_time_stamp: datetime = start_time_stamp

        distance_matrix: dict[tuple[int, int]: float] = self.distance_matrix  # ToDo: reachable through processes

        process_execution_paths = \
            [process_execution_path
             for process_execution_path_lst in list(resources_process_executions_components.values())
             for process_execution_path in process_execution_path_lst]

        process_execution_available_paths = {}
        process_execution_variant = {}
        for process_execution_path in process_execution_paths:
            if process_execution_path.type.value == "CONNECTOR":
                continue
            variant = process_execution_path.process_executions_components_parents[0]
            process_execution = variant.goal

            process_execution_available_paths.setdefault(process_execution,
                                                         []).append(process_execution_path)
            process_execution_variant.setdefault(process_execution,
                                                 []).append(variant)

        process_executions: dict[int, dict] = {}
        resources_: list[Resource] = []
        parts_: list[Part] = []
        predecessors = {}
        for process_execution, paths in process_execution_available_paths.items():
            process_execution_id, process_name, process_time = \
                self._get_process_specific_attributes(process_execution)

            resources_, parts_, linked_resources, linked_parts, \
                earliest_possible_start_time_stamp, latest_possible_end_time_stamp, \
                required_resource_entity_types, number_of_required_resources_per_entity_type, \
                required_part_entity_types, number_of_required_parts_per_entity_type = \
                self._get_process_execution_specific_attributes(resources_, parts_, paths, process_execution)

            variants = list(set(process_execution_variant[process_execution]))

            predecessors = self._determine_neighbour_relationships(predecessors, variants)

            process_executions[process_execution_id] = \
                {"process_name": process_name,
                 "process_time": process_time,
                 "linked_resources": linked_resources,
                 "linked_parts": linked_parts,
                 "earliest_possible_start_time_stamp": earliest_possible_start_time_stamp,
                 "latest_possible_end_time_stamp": latest_possible_end_time_stamp,
                 "required_resource_entity_types": required_resource_entity_types,
                 "number_of_required_resources_per_entity_type": number_of_required_resources_per_entity_type,
                 "required_part_entity_types": required_part_entity_types,
                 "number_of_required_parts_per_entity_type": number_of_required_parts_per_entity_type,
                 "predecessor_process_executions": None,
                 "successor_process_executions": None}

        latest_possible_end_time_stamps = [pe_dict["latest_possible_end_time_stamp"]
                                           for pe_dict in list(process_executions.values())]
        if not latest_possible_end_time_stamps:
            raise NotImplementedError
        scheduling_interval_end_time_stamp: datetime = min(latest_possible_end_time_stamps).item()

        predecessors = {pe1.identification: [pe.identification
                                             for pe in list(set(pe_s))]
                        for pe1, pe_s in predecessors.items()}
        successors = {}
        for process_execution_id, pe_dict in process_executions.items():
            predecessors_pe = predecessors[process_execution_id]
            pe_dict["predecessor_process_executions"] = predecessors_pe
            for predecessor_id in predecessors_pe:
                successors.setdefault(predecessor_id, []).append(process_execution_id)

        for process_execution_id, pe_dict in process_executions.items():
            if process_execution_id in successors:
                successors_pe = successors[process_execution_id]
            else:
                successors_pe = []
            pe_dict["successor_process_executions"] = successors_pe

        # resources
        resources_ = list(set(resources_))
        resources: list[int] = [resource._process_execution_plan.identification for resource in resources_]
        resource_types: list[tuple[int, int | None]] = \
            [(resource.entity_type.identification,
              resource.entity_type.super_entity_type.identification if resource.entity_type.super_entity_type else None)
             for resource in resources_]
        initial_positions_resources: list[int] = []  # ToDo: depend on time_stamps?  included in schedule
        resources_speeds: list[float] = [resource.get_expected_performance() for resource in resources_]

        resources_schedules = \
            self._get_resource_schedules(digital_twin, start_time_stamp, resources_,
                                         scheduling_interval_start_time_stamp, scheduling_interval_end_time_stamp)

        resources_dict: dict = {"resources": resources,
                                "resource_types": resource_types,
                                "initial_positions_resources": initial_positions_resources,
                                "resources_speeds": resources_speeds,
                                "resources_schedules": resources_schedules}

        # parts
        parts: list[int] = [part.identification for part in parts_]
        part_types: list[tuple[int, int | None]] = \
            [(part.entity_type.identification,
              part.entity_type.super_entity_type.identification if part.entity_type.super_entity_type else None)
             for part in parts_]
        initial_positions_parts: list[int] = [self._get_part_initial_position(part)
                                              for part in parts_]  # ToDo: depend on time_stamps?

        parts_dict: dict = {"parts": parts,
                            "part_types": part_types,
                            "initial_positions_parts": initial_positions_parts}

        # ToDo: preferences

        scheduling_input = {"scheduling_interval_start_time_stamp": scheduling_interval_start_time_stamp,
                            "scheduling_interval_end_time_stamp": scheduling_interval_end_time_stamp,
                            "distance_matrix": distance_matrix,
                            "process_executions": process_executions,
                            "resources": resources_dict,
                            "parts": parts_dict}

        scheduling_input_path = f"scheduling_input{self.batch_number}"
        with open(scheduling_input_path, 'wb') as outp:
            pickle.dump(scheduling_input, outp, pickle.HIGHEST_PROTOCOL)

        self.batch_number += 1

        return scheduling_input

    def _get_process_specific_attributes(self, process_execution):
        process_execution_id: int = process_execution.identification
        process_name: str = process_execution.get_name()
        process_time: int = process_execution.process.get_estimated_process_lead_time()

        return process_execution_id, process_name, process_time

    def _get_process_execution_specific_attributes(self, resources_, parts_, paths,
                                                   process_execution: ProcessExecution):
        cfp_goal_item_match = {}
        for path in paths:
            cfp_goal_item_match.setdefault(path.cfp_path[-1], []).append(path.goal_item)

        entities_used_example_batch = [entity_list[0] for entity_list in list(cfp_goal_item_match.values())]
        example_resources_used = [entity for entity in entities_used_example_batch if isinstance(entity, Resource)]
        example_parts_involved = [entity for entity in entities_used_example_batch if isinstance(entity, Part)]
        resources_.extend(example_resources_used)
        parts_.extend(example_parts_involved)

        possible_resource_models = process_execution.get_possible_resource_groups(resources=example_resources_used)
        if len(possible_resource_models) != 1:
            raise NotImplementedError

        possible_resource_model = possible_resource_models[0]
        resource_entity_types = possible_resource_model.resources

        # ToDo: super entity types
        number_of_required_resources_per_entity_type: dict[int, int] = \
            {entity_type.identification: resource_entity_types.count(entity_type)
             for entity_type in list(set(resource_entity_types))}
        required_resource_entity_types: list[int] = list(number_of_required_resources_per_entity_type.keys())

        part_entity_types = [part.entity_type for part in example_parts_involved]
        number_of_required_parts_per_entity_type: dict[int, int] = \
            {entity_type.identification: part_entity_types.count(entity_type)
             for entity_type in list(set(part_entity_types))}
        required_part_entity_types: list[int] = list(number_of_required_parts_per_entity_type.keys())

        linked_resources: list[int] = []
        linked_parts: list[int] = []

        earliest_possible_start_time_stamp: datetime | None = None  # ToDo:  ensure that it is always specified
        latest_possible_end_time_stamp: datetime | None = None
        for path in paths:

            preference = path.reference_preference

            # earliest_possible_start_time_stamp and latest_possible_start_time_stamp
            accepted_time_periods = preference.accepted_time_periods
            if accepted_time_periods.any():
                if earliest_possible_start_time_stamp is None:
                    earliest_possible_start_time_stamp = accepted_time_periods[0][0]
                elif earliest_possible_start_time_stamp > accepted_time_periods[0][0]:
                    earliest_possible_start_time_stamp = accepted_time_periods[0][0]

                if latest_possible_end_time_stamp is None:
                    latest_possible_end_time_stamp = accepted_time_periods[-1][1]
                elif latest_possible_end_time_stamp < accepted_time_periods[-1][1]:
                    latest_possible_end_time_stamp = accepted_time_periods[-1][1]

            # entities
            entity = path.goal_item
            if preference.long_time_reservation_duration:
                if isinstance(entity, Resource):
                    linked_resources.append(entity._process_execution_plan.identification)
                elif isinstance(entity, Part):
                    linked_parts.append(entity.identification)

        return resources_, parts_, linked_resources, linked_parts, \
            earliest_possible_start_time_stamp, latest_possible_end_time_stamp, \
            required_resource_entity_types, number_of_required_resources_per_entity_type, \
            required_part_entity_types, number_of_required_parts_per_entity_type

    def _determine_neighbour_relationships(self, predecessors, variants):
        for variant in variants:
            predecessors_ = list(set(predecessor.goal for predecessor in variant.predecessors if predecessor.type.value != "CONNECTOR"))
            predecessors.setdefault(variant.goal, []).extend(predecessors_)

        return predecessors

    def _get_resource_schedules(self, digital_twin, start_time_stamp, resources, scheduling_interval_start_time_stamp,
                                scheduling_interval_end_time_stamp):
        resource_positions = \
            {resource: (self._get_resource_position(digital_twin=digital_twin, resource=resource,
                                                    start_time_stamp=start_time_stamp)
                        if isinstance(resource, NonStationaryResource) else None)
             for resource in resources}

        scheduling_interval_start_time_stamp64 = np.datetime64(scheduling_interval_start_time_stamp)
        scheduling_interval_end_time_stamp64 = np.datetime64(scheduling_interval_end_time_stamp)

        resources_schedules = \
            {resource: resource.process_executions_plan._get_blocked_periods_calendar_extract(
                start_time=scheduling_interval_start_time_stamp64,
                end_time=scheduling_interval_end_time_stamp64).to_numpy()
             for resource in resources}

        resources_schedules_with_positions: dict[int: list[[datetime, datetime, int, int]]] = {}  # ToDo: schedules
        for resource, resource_position_d in resource_positions.items():
            schedule = resources_schedules[resource]
            resources_schedules_with_positions[resource._process_execution_plan.identification] = \
                self._create_resource_schedule_with_positions(resource, schedule, resource_position_d,
                                                              scheduling_interval_start_time_stamp64,
                                                              scheduling_interval_end_time_stamp64)

        return resources_schedules_with_positions

    def _create_resource_schedule_with_positions(self, resource, schedule, positions,
                                                 scheduling_interval_start_time_stamp64,
                                                 scheduling_interval_end_time_stamp64):

        resource_id = resource._process_execution_plan.identification

        if not schedule.size and positions is None:
            schedule = np.array([[scheduling_interval_start_time_stamp64, scheduling_interval_start_time_stamp64,
                                  resource_id, resource_id],
                                 [scheduling_interval_end_time_stamp64, scheduling_interval_end_time_stamp64,
                                  resource_id, resource_id]])
            return schedule

        elif not schedule.size:
            datetimes_ = list(positions.keys())
            if len(datetimes_) != 1:
                resource.process_executions_plan._get_blocked_periods_calendar_extract(
                    start_time=scheduling_interval_start_time_stamp64, end_time=scheduling_interval_end_time_stamp64)
                raise NotImplementedError
            first_time_stamp = datetimes_[0]
            if first_time_stamp != scheduling_interval_start_time_stamp64:
                raise NotImplementedError
            position_resource_id = positions[first_time_stamp]._process_execution_plan.identification
            schedule = np.array([[scheduling_interval_start_time_stamp64, scheduling_interval_start_time_stamp64,
                                  position_resource_id, position_resource_id],
                                 [scheduling_interval_end_time_stamp64, scheduling_interval_end_time_stamp64,
                                  position_resource_id, position_resource_id]])

            return schedule

        schedule = schedule.astype("object")
        positions_array = schedule.copy()
        schedule = np.hstack((schedule, positions_array))

        if positions is None:  # stationary resource

            schedule[:, 2:] = resource_id

        else:
            datetime_positions = list(positions.keys())
            dict_len = len(datetime_positions)
            for datetime_position_idx in range(dict_len):
                datetime_ = copy(datetime_positions[datetime_position_idx].replace(microsecond=000000000))
                datetime_ = datetime.timestamp(datetime_) * 1e9

                if not (datetime_position_idx + 1 >= dict_len):
                    datetime_1 = copy(datetime_positions[datetime_position_idx + 1].replace(microsecond=000000000))
                    datetime_1 = datetime.timestamp(datetime_1) * 1e9
                    if datetime_position_idx == 0:

                        mask1 = schedule[:, 0] < datetime_1
                        mask2 = schedule[:, 1] < datetime_1
                    else:
                        mask1 = (schedule[:, 0] >= datetime_) & (schedule[:, 0] < datetime_1)
                        mask2 = (schedule[:, 1] >= datetime_) & (schedule[:, 1] < datetime_1)

                else:
                    # end slice
                    mask1 = schedule[:, 0] >= datetime_
                    mask2 = schedule[:, 1] >= datetime_
                schedule[mask1, 2] = positions[datetime_positions[datetime_position_idx]].identification
                schedule[mask2, 3] = positions[datetime_positions[datetime_position_idx]].identification

        return schedule

    def _get_resource_position(self, digital_twin, resource, start_time_stamp):
        """Get the position changes mapped to time_stamps the position is changed for a resource"""

        position_changes = resource.get_positions(start_time_stamp)
        resource_position_changes = \
            {time_stamp if time_stamp != 0 else datetime(1970, 1, 1):
                 digital_twin.get_stationary_resource_at_position(pos_tuple=position)[0]
             for time_stamp, position in position_changes.items() if position is not None}

        return resource_position_changes

    def _get_part_initial_position(self, part: Part):
        origin_resource = None
        if part.situated_in:
            origin_resource = part.situated_in

        elif part.part_of:
            if part.part_of.situated_in:
                origin_resource = part.part_of.situated_in

        else:
            raise Exception

        if origin_resource is None:
            raise Exception

        return origin_resource._process_execution_plan.identification
