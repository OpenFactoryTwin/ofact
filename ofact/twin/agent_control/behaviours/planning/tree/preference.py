"""
handle preference
preferences
- time
- resources (intern) - currently evaluation of the proposals
- parts
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.planning.tree.helpers import get_overlaps_periods
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import \
    ProcessExecutionsPath
from ofact.twin.state_model.entities import Entity, ConveyorBelt, EntityType
from ofact.twin.state_model.processes import ProcessExecution
from ofact.twin.state_model.time import ProcessExecutionPlanConveyorBelt
from ofact.helpers import convert_to_datetime

if TYPE_CHECKING:
    from ofact.twin.state_model.processes import WorkOrder  # , ProcessExecution
    from ofact.twin.state_model.entities import Resource
    from ofact.twin.state_model.time import ProcessExecutionPlan


# ToDo: switch to numpy which seems to be faster

one_second64 = np.timedelta64(1, "s")


def arr_sequence_detection(arr):
    """Detect sequences in arrays by  determination of start and end index"""

    if arr.size == 0:
        return arr

    second_arr = arr + 1
    second_arr = np.insert(second_arr, 0, arr[0] - 1)[:-1]
    sequence_breaks = np.where(arr != second_arr)[0]

    second_sequence_breaks = (sequence_breaks - 1)[1:]
    second_sequence_breaks = np.insert(second_sequence_breaks, second_sequence_breaks.shape[0], second_arr.shape[0] - 1)

    slices = np.zeros((sequence_breaks.shape[0], 2), dtype=np.int64)
    slices[:, 0], slices[:, 1] = sequence_breaks, second_sequence_breaks

    return slices


def get_overlaps(arr_a, arr_b, data_type="float64"):
    """Determine the overlaps of periods (determined by start, end - arrays)"""
    if not arr_a.any() or not arr_b.any():
        return np.array([[]])

    i, j = 0, 0
    overlaps = np.array([], dtype=data_type)
    while i < len(arr_a) and j < len(arr_b):
        if arr_a[i][1] < arr_b[j][0]:
            # a[i] is too small to overlap
            i += 1
        elif arr_b[j][1] < arr_a[i][0]:
            # b[j] is too small to overlap
            j += 1
        else:
            start = max(arr_a[i][0], arr_b[j][0])
            end = min(arr_a[i][1], arr_b[j][1])
            overlaps = np.insert(overlaps, overlaps.shape[0], np.array([start, end]))
            # whichever ends first will be removed from consideration as
            # the next interval in the series might still overlap
            if arr_a[i][1] <= arr_b[j][1]:
                i += 1
            else:
                j += 1

    overlaps_formatted = overlaps.reshape((int(len(overlaps) / 2), 2))

    return overlaps_formatted


# ToDo: ... write test_case

def determine_lead_time_entity_entity_types(entity_entity_types, lead_times):
    if entity_entity_types is None or not lead_times:
        lead_time = 0
        return lead_time

    if entity_entity_types in lead_times:
        lead_time = lead_times[entity_entity_types]
    elif isinstance(entity_entity_types, EntityType):
        if entity_entity_types.super_entity_type in lead_times:
            lead_time = lead_times[entity_entity_types.super_entity_type]
        else:
            lead_time = 0
    elif isinstance(entity_entity_types, Entity):
        if entity_entity_types.entity_type in lead_times:
            lead_time = lead_times[entity_entity_types.entity_type]
        elif entity_entity_types.entity_type is not None:
            if entity_entity_types.entity_type.super_entity_type in lead_times:
                lead_time = lead_times[entity_entity_types.entity_type.super_entity_type]
            else:
                lead_time = 0
        else:
            lead_time = 0
    else:
        lead_time = 0

    return lead_time


def determine_follow_up_time_entity_entity_types(entity_entity_types, follow_up_times):
    if entity_entity_types is None or not follow_up_times:
        follow_up_time = 0
        return follow_up_time

    if entity_entity_types in follow_up_times:
        follow_up_time = follow_up_times[entity_entity_types]
    elif isinstance(entity_entity_types, EntityType):
        if entity_entity_types.super_entity_type in follow_up_times:
            follow_up_time = follow_up_times[entity_entity_types.super_entity_type]
        else:
            follow_up_time = 0
    elif isinstance(entity_entity_types, Entity):
        if entity_entity_types.entity_type in follow_up_times:
            follow_up_time = follow_up_times[entity_entity_types.entity_type]
        elif entity_entity_types.entity_type is not None:
            if entity_entity_types.entity_type.super_entity_type in follow_up_times:
                follow_up_time = follow_up_times[entity_entity_types.entity_type.super_entity_type]
            else:
                follow_up_time = 0
        else:
            follow_up_time = 0
    else:
        follow_up_time = 0

    return follow_up_time


class Preference(metaclass=ABCMeta):
    preference_values_data_type = np.dtype([("Timestamp", "datetime64[s]"),
                                            ("Value", "int8")])

    def __init__(self, reference_objects: None | list[Resource] | list[WorkOrder] | list[ProcessExecution]):
        """
        The Preference determines the 'utility' of time slots and is used to find 'best' matching periods between
        different parties.
        :param reference_objects: an object that has to schedule process_executions_components (Resource | WorkOrder) or
        no specified object or a process_execution that is in planning
        """
        if not isinstance(reference_objects, list):
            reference_objects = [reference_objects]
        self.reference_objects: list[Resource] | list[WorkOrder] = reference_objects

    def get_preference_values(self, accepted_time_periods, time_slot_duration,
                              destination, process, preference_function, lead_times=None, follow_up_times=None,
                              start_time=None, end_time=None, after_issue=None) -> np.array:
        # ToDo: destination
        #  duration = lead_time + time_slot_duration + follow_up_time

        # old_accepted_time_periods = accepted_time_periods.copy()
        accepted_time_periods = accepted_time_periods.copy()
        if isinstance(self, EntityPreference):  # relation to a process_executions_plan
            lead_time = 0
            follow_up_time = 0
            if lead_times or follow_up_times:
                reference_object = self.reference_objects[0]
                reference_entity_type = reference_object.entity_type

                lead_time = determine_lead_time_entity_entity_types(reference_entity_type, lead_times)
                follow_up_time = determine_follow_up_time_entity_entity_types(reference_entity_type, follow_up_times)

            accepted_time_periods = accepted_time_periods.astype(dtype="datetime64[s]")
            if start_time is None:
                start_time = accepted_time_periods[0][0] - np.timedelta64(int(np.ceil(round(lead_time, 1))), "s")
            if end_time is None:
                end_time = accepted_time_periods[-1][1] + np.timedelta64((int(np.ceil(round(follow_up_time, 1)))), "s")
            time_slot_duration_free_periods = time_slot_duration + lead_time + follow_up_time
            free_time_periods = \
                self.get_free_time_periods(time_slot_duration=time_slot_duration_free_periods, start_time=start_time,
                                           end_time=end_time, after_issue=after_issue)

            accepted_free_time_periods = get_overlaps(free_time_periods, accepted_time_periods)
            accepted_time_periods = accepted_free_time_periods.astype(dtype="datetime64[s]")

        preference_values = np.array([], dtype=type(self).preference_values_data_type)
        if not accepted_time_periods.any():
            return preference_values

        if not isinstance(accepted_time_periods[0][0], np.datetime64):
            raise Exception

        v_get_preference_value_timestamp = np.vectorize(preference_function,  # signature='(h),(h),(n)->(n)',
                                                        otypes=["int16"])

        input_params_lst = []
        for idx, accepted_time_period in enumerate(accepted_time_periods):
            time_stamps = np.arange(accepted_time_period[0], accepted_time_period[1],
                                    one_second64)
            len_time_stamps = time_stamps.size
            start_period = accepted_time_period[0].item()
            len_period = len_time_stamps - 1
            if len_period == 0:
                len_period = 0.5  # to avoid a zero value

            if isinstance(self, EntityPreference):
                weight = 1 / (1 + idx)
            else:
                weight = 1

            input_param = np.repeat(np.array([start_period, len_period, weight]),
                                    len_time_stamps).reshape(3, len_time_stamps)
            input_param = np.concatenate([time_stamps[None, :], input_param], axis=0)
            input_params_lst.append(input_param)

        input_params = np.concatenate(input_params_lst, axis=1)
        if not input_params.size:
            return preference_values

        preference_values_array = np.zeros(input_params.shape[1], dtype="int8")
        preference_values_array[:] = \
            v_get_preference_value_timestamp(input_params[0], input_params[1], input_params[2])

        preference_values_array *= input_params[3].astype("int8")  # int object would be better
        time_stamps_array = input_params[0].astype("datetime64[s]")

        preference_values = np.empty(preference_values_array.size, dtype=type(self).preference_values_data_type)
        preference_values['Timestamp'] = time_stamps_array
        preference_values['Value'] = preference_values_array

        if np.unique(preference_values["Timestamp"]).size < preference_values.shape[0]:
            raise Exception("Get the use case where timestamp are not unique")

        return preference_values

    @abstractmethod
    def get_free_time_periods(self, time_slot_duration, start_time=None, end_time=None):
        pass

    def _get_preference_value_timestamp(self, time_stamp, start_period, len_period):
        # it should be checked before if the timestamp is a free one

        time_dist_to_start = time_stamp - start_period
        time_dist_to_start_seconds = time_dist_to_start.seconds

        preference_value_unrounded = (1 - (time_dist_to_start_seconds + 1e-12) / (2 * len_period + 1e-12)) * 100
        preference_value = int(preference_value_unrounded)

        if preference_value > 100:
            raise Exception("Not normalized")

        return preference_value

    @abstractmethod
    def get_copy(self):
        # copy the self/ object
        new_process_execution_preference = copy(self)
        new_process_execution_preference.reference_objects = copy(self.reference_objects)

        return new_process_execution_preference


class EntityPreference(Preference):

    def __init__(self, reference_objects: None | list[Resource] | list[WorkOrder], accepted_time_horizont):
        """
        The Preference determines the 'utility' of time slots and is used to find 'best' matching periods between
        different parties.
        :param accepted_time_horizont: determines the period the reference_object can be planned in the future
        :param reference_objects: an object that has to schedule process_executions_components
        """
        super(EntityPreference, self).__init__(reference_objects=reference_objects)
        self.accepted_time_horizont64 = np.timedelta64(int(accepted_time_horizont), "s")
        self.accepted_time_horizont64_adapted = None
        self._process_execution_plan_copy: Optional[ProcessExecutionPlan] = None

    def get_copy(self):
        """Copy the EntityType preference object (self) and overwrite the attributes of them to avoid some things"""

        entity_preference_copy = super(EntityPreference, self).get_copy()
        entity_preference_copy.accepted_time_horizont = copy(self.accepted_time_horizont64)

        # the process_executions_plan is normally not stored in the entity_preference because it is reachable via
        # the reference_object but in case of copy, there is the original process_executions_plan found
        # Assumption: all reference_objects have the same process_executions_plan
        if len(entity_preference_copy.reference_objects) < 1:
            raise NotImplementedError
        reference_object = entity_preference_copy.reference_objects[0]
        entity_preference_copy._process_execution_plan_copy = reference_object.get_process_execution_plan_copy()

        return entity_preference_copy

    def get_process_execution_preference(self, reference_objects, accepted_time_periods, expected_time_slot_duration,
                                         lead_times={}, follow_up_times={}, origin=None, destination=None,
                                         process=None, weight=1, extension=None):
        """:param weight: describes the relevance of the next value_added_process"""

        start_time = None
        end_time = None
        if extension is None:
            self.accepted_time_horizont64_adapted = None

        if accepted_time_periods.any():
            if accepted_time_periods[0][0] == accepted_time_periods[0][0]:
                start_time = accepted_time_periods[0][0]
            if accepted_time_periods[-1][1] == accepted_time_periods[-1][1]:
                end_time = accepted_time_periods[-1][1]

        # determine start_time and end_time if not given
        if start_time is None:
            raise Exception

        if end_time is not None:
            accepted_time_periods_end = accepted_time_periods.copy()
            accepted_time_periods_end[np.where(accepted_time_periods[:, 1] != accepted_time_periods[:, 1]), 1] = \
                end_time

        else:
            if extension:
                self.accepted_time_horizont64_adapted = np.timedelta64(int(extension), "s")
                time_horizont = self.accepted_time_horizont64_adapted
            else:
                time_horizont = self.accepted_time_horizont64

            accepted_time_periods[-1][1] = start_time + time_horizont

        preference_values = \
            self.get_preference_values(accepted_time_periods, expected_time_slot_duration,
                                       process, destination, lead_times=lead_times, follow_up_times=follow_up_times,
                                       preference_function=self._get_preference_value_timestamp, start_time=start_time,
                                       end_time=end_time)  # reinlegen und Aufruf noch nicht tätigen ...

        preference_values["Value"] *= weight

        if not preference_values.size:
            preference_values = \
                self.get_preference_values(accepted_time_periods, expected_time_slot_duration,
                                           process, destination, lead_times=lead_times, follow_up_times=follow_up_times,
                                           preference_function=self._get_preference_value_timestamp,
                                           start_time=start_time,
                                           end_time=end_time)  # reinlegen und Aufruf noch nicht tätigen ...

        accepted_time_periods = accepted_time_periods.copy()
        process_execution_preference = \
            ProcessExecutionPreference(reference_objects=reference_objects, accepted_time_periods=accepted_time_periods,
                                       preference_values=preference_values, lead_time=lead_times,
                                       expected_process_execution_time=expected_time_slot_duration,
                                       follow_up_time=follow_up_times, origin=origin, destination=destination)

        return process_execution_preference

    def get_accepted_time_horizont64(self):
        if self.accepted_time_horizont64_adapted:
            return self.accepted_time_horizont64_adapted
        else:
            return self.accepted_time_horizont64

    def get_utilization(self, start_time, end_time):
        """Return the utilization in the period between start and end time"""
        if self._process_execution_plan_copy is None:
            raise Exception("Currently only usable for the process_executions_plan copy")

        utilization = self._process_execution_plan_copy.get_utilization(start_time, end_time)
        return utilization

    def get_last_time_stamp(self) -> pd.Timestamp | None:
        """used by process_execution_queue"""
        if self._process_execution_plan_copy is None:
            raise Exception("Currently only usable for the process_executions_plan copy")

        last_time_stamp = self._process_execution_plan_copy.get_last_time_stamp()
        return last_time_stamp

    def block_time_slot(self, start_time, end_time, blocker_name, process_execution_id, work_order_id, issue_id,
                        block_before=False):
        """
        Block a time slot in the scheduling process (therefore it is scheduled on a copy of the process_executions_plan)
        """

        if not self._process_execution_plan_copy:
            raise Exception

        successful, clashing_blocker_names, clashing_process_execution_ids = \
            self._process_execution_plan_copy.block_period(start_time=start_time,
                                                           end_time=end_time,
                                                           blocker_name=blocker_name,
                                                           process_execution_id=process_execution_id,
                                                           work_order_id=work_order_id,
                                                           issue_id=issue_id,
                                                           block_before=block_before)

        if not successful:
            i = 100
            while i > 0:
                print("Block not successful ...")
                i -= 1

            # successful, clashing_blocker_names, clashing_process_execution_ids = \
            #     self._process_execution_plan_copy.block_period(start_time=start_time,
            #                                                    end_time=end_time,
            #                                                    blocker_name=blocker_name,
            #                                                    process_execution_id=process_execution_id,
            #                                                    work_order_id=work_order_id,
            #                                                    issue_id=issue_id,
            #                                                    block_before=block_before)
            # process_execution was scheduled before

            formatted_data = np.array2string(
                self._process_execution_plan_copy._time_schedule,
                separator=', ',
                formatter={'float_kind': lambda x: f"{x:.2f}"}
            )
            operator = self.reference_objects[0].name
            exception_msg = (
                f"dtype:\n{str(self._process_execution_plan_copy._time_schedule.dtype)}\n"
                f"Array:\n{formatted_data}\n\n"
                f"Timeslot: {start_time}, {end_time}\n\n"
                f"Operator: {operator}\n"
                f"Issue ID: {issue_id}\n"
                f"Process Execution ID: {process_execution_id}\n"
                f"Blocker Name: {blocker_name}\n"
            )
            raise Exception(exception_msg)

        return successful

    def unblock_time_slot(self, unblocker_name, process_execution_id):
        """Unblock a time_slot because the time_slot is in combination not possible"""
        if not self._process_execution_plan_copy:
            raise Exception

        successful = self._process_execution_plan_copy.unblock_period(unblocker_name, process_execution_id)
        return successful

    def get_time_slot(self, blocker_name: str, process_execution_id: int, issue_id: int):
        """Get the time_slot suitable to the blocker_name and the process_execution_id"""
        if not self._process_execution_plan_copy:
            raise Exception

        successful, time_slot = self._process_execution_plan_copy.get_time_slot(blocker_name, process_execution_id,
                                                                                issue_id)
        return successful, time_slot

    def get_time_periods_matching(self, free_time_periods_self=None, free_time_periods_other=None,
                                  time_slot_duration: Optional[float, int] = None, start_time=None, end_time=None,
                                  issue_id=None, long_time_reservation_duration=None) \
            -> Optional[np.array([[np.datetime64, np.datetime64], ])]:
        """Get free_time_slots that match with the free_time_periods_self and the free_time_periods_other"""

        if free_time_periods_self is None:
            free_time_periods_self = \
                self.get_free_time_periods(time_slot_duration=time_slot_duration, start_time=start_time,
                                           end_time=end_time, issue_id=issue_id,
                                           long_time_reservation_duration=long_time_reservation_duration)
        if free_time_periods_other is None:
            raise ValueError("Free Time Period of the other entity is needed")

        matching_time_periods = None
        if not free_time_periods_self.size or not free_time_periods_other.size:
            return matching_time_periods

        matching_time_periods = get_overlaps(free_time_periods_self, free_time_periods_other)
        matching_time_periods = matching_time_periods.astype("datetime64[s]")

        if matching_time_periods is not None and not isinstance(self.reference_objects[0], ConveyorBelt):
            time_slot_duration = np.timedelta64(int(time_slot_duration), "s")
            # consider only the periods that match with the time_slot_duration restriction
            matching_time_periods = matching_time_periods[
                (matching_time_periods[:, 1] - matching_time_periods[:, 0]) >= time_slot_duration]

            if not matching_time_periods.shape[0]:
                matching_time_periods = None

        return matching_time_periods

    def get_free_time_periods(self, time_slot_duration: Union[float, int], start_time: Optional[np.datetime64] = None,
                              end_time: Optional[np.datetime64] = None, issue_id: int = None,
                              long_time_reservation_duration=None, after_issue=None) -> \
            np.array([[np.datetime64, np.datetime64], ]):
        """Get free time_slots based on the own _process_execution_plan"""

        time_slot_duration = np.timedelta64(int(time_slot_duration), "s")

        if (end_time - start_time) < time_slot_duration and not isinstance(self.reference_objects[0], ConveyorBelt):
            return np.array([[]])

        if self._process_execution_plan_copy is not None:
            reference_object = self._process_execution_plan_copy
        else:
            reference_object = self.reference_objects[0]

        if after_issue is not None:
            issue_df = reference_object.get_time_slots_from_issue(issue_id=after_issue)
            if issue_df.shape[0] > 0:
                last_time_stamp = issue_df["End"][-1]
                if isinstance(reference_object, ProcessExecutionPlanConveyorBelt):
                    pass  # the time_stamp should be the same like the timeslot requested before (with the same issue)
                    # last_time_stamp = reference_object.get_next_possible_start_time(last_time_stamp)

                    time_slot_duration = np.timedelta64(0, "s")
                free_periods_calendar_extract = reference_object.get_free_time_from(last_time_stamp,
                                                                                    issue_id=after_issue)
                if free_periods_calendar_extract.shape[0] > 0:
                    mask = (free_periods_calendar_extract[1] - free_periods_calendar_extract[0] >= time_slot_duration)
                    free_periods_calendar_extract = free_periods_calendar_extract[mask]
                    if end_time and free_periods_calendar_extract.shape[0] > 0:
                        if free_periods_calendar_extract[-1, 1] > end_time:
                            free_periods_calendar_extract[-1, 1] = end_time

                if free_periods_calendar_extract.dtype != "datetime64[s]":
                    free_periods_calendar_extract = free_periods_calendar_extract.astype("datetime64[s]")

                return free_periods_calendar_extract

        free_periods_calendar_extract = \
            reference_object.get_free_periods_calendar_extract(
                start_time=start_time, end_time=end_time, issue_id=issue_id, time_slot_duration=time_slot_duration,
                long_time_reservation_duration=long_time_reservation_duration)

        if free_periods_calendar_extract.dtype != "datetime64[s]":
            free_periods_calendar_extract = free_periods_calendar_extract.astype("datetime64[s]")

        return free_periods_calendar_extract

    def get_free_time_from(self, start_time_stamp, end_time_stamp=None) -> np.array:
        """Get free time period from the start_time_stamp until the next planned slot
        Used for the process_executions_plan_copy"""

        free_time_from_timestamp = \
            self._process_execution_plan_copy.get_free_time_from(start_time=start_time_stamp).copy()

        if end_time_stamp is not None and free_time_from_timestamp.size != 0:
            end_time_stamp = convert_to_datetime(end_time_stamp)
            if free_time_from_timestamp[-1] > end_time_stamp:
                free_time_from_timestamp[-1] = end_time_stamp
        return free_time_from_timestamp

    def check_clashing(self, start_time, end_time, issue_id, time_slot_duration):
        clashing_blocker_names, clashing_process_execution_ids = \
            self._process_execution_plan_copy._check_clash(start_time=start_time, end_time=end_time, issue_id=issue_id,
                                                           time_slot_duration=time_slot_duration)

        if clashing_blocker_names[0] is None:
            clash = False
        else:
            clash = True
        return clash


def _get_merged_time_periods_vertical(possible_time_periods, data_type="datetime64[ns]"):
    """Merge the possible time periods vertical"""

    merged_time_periods = possible_time_periods
    while len(merged_time_periods) > 1:
        new_merged_time_period = []
        for possible_time_period_idx in range(0, len(merged_time_periods), 2):
            if possible_time_period_idx + 1 < len(merged_time_periods):
                overlaps = get_overlaps(merged_time_periods[possible_time_period_idx],
                                        merged_time_periods[possible_time_period_idx + 1], data_type)

                overlaps = overlaps.astype(data_type)
                # overlaps.astype("datetime64[s]")
                new_merged_time_period.append(overlaps)
            else:
                new_merged_time_period.append(merged_time_periods[possible_time_period_idx])

        merged_time_periods = new_merged_time_period

    merged_time_periods_complete = merged_time_periods[0].astype(data_type)

    return merged_time_periods_complete


class ProcessExecutionPreference(Preference):

    def __init__(self, reference_objects, accepted_time_periods: np.array([]), preference_values,
                 expected_process_execution_time, origin, destination, lead_time=None, follow_up_time=None,
                 long_time_reservation_duration: None | float = None):
        """
        :param accepted_time_periods: determined via start and end timestamps - needed to ensure preconditions
        (other processes) and accelerate the negotiation process through restrictions

        :param long_time_reservation_duration: the long_time_reservation_duration is related to the goal_item
        :param origin: position where the component starts
        :param destination: position where the component ends
        """
        super(ProcessExecutionPreference, self).__init__(reference_objects=reference_objects)

        if accepted_time_periods.dtype != "datetime64[s]":
            accepted_time_periods = accepted_time_periods.astype("datetime64[s]")

        self._accepted_time_periods: np.array([[np.datetime64, np.datetime64], ]) = accepted_time_periods
        self.last_accepted_time_periods: None | list[np.array([[np.datetime64, np.datetime64], ], str)] = None
        self.accepted_time_periods_changed = False  # is used to determine if the price should be updated

        if preference_values.shape[0] == 0:
            print("Preference values df empty", expected_process_execution_time)
        self.preference_values: np.array = preference_values

        # the preference values related to goals are preference_values in vertical direction
        if expected_process_execution_time is None:
            expected_process_execution_time = 0
        expected_process_execution_time = int(np.ceil(round(expected_process_execution_time, 1)))
        self.expected_process_execution_time = expected_process_execution_time

        # used from the planning for adaptions
        self.accepted_time_periods_adapted = None
        self.required_process_execution_time = None

        # related to entity_type
        if lead_time is None:
            lead_time = {}
        self.lead_time: dict[Entity | EntityType, int] = lead_time
        if follow_up_time is None:
            follow_up_time = {}
        self.follow_up_time: dict[Entity | EntityType, int] = follow_up_time

        self.long_time_reservation_duration: None | float = long_time_reservation_duration

        self.origin = origin
        self.destination = destination

    @property
    def preference_values(self):
        return self._preference_values

    @preference_values.setter
    def preference_values(self, preference_values):
        self._preference_values = preference_values

    @property
    def accepted_time_periods(self):
        return self._accepted_time_periods

    @accepted_time_periods.setter
    def accepted_time_periods(self, accepted_time_periods):

        if np.array_equiv(accepted_time_periods, self._accepted_time_periods):
            return

        if accepted_time_periods.dtype != "datetime64[s]":
            accepted_time_periods = accepted_time_periods.astype("datetime64[s]")

        self.accepted_time_periods_changed = True
        # if self._accepted_time_periods is not None:
        #     if self.last_accepted_time_periods is None:
        #         self.last_accepted_time_periods = []
        #
        #     curframe = inspect.currentframe()
        #     calframe = inspect.getouterframes(curframe, 2)
        #     self.last_accepted_time_periods.append((self._accepted_time_periods.copy(), calframe[1][3]))
        #     for debugging use cases
        #
        #     print("Accepted time periods: ", accepted_time_periods)

        if accepted_time_periods.size == 0:
            self._accepted_time_periods = np.array([[]])
            return

        accepted_time_periods = \
            accepted_time_periods[accepted_time_periods[:, 1] - accepted_time_periods[:, 0] >=
                                  np.timedelta64(self.expected_process_execution_time, "s")]

        self._accepted_time_periods = accepted_time_periods.copy()

    def get_accepted_time_periods(self):
        if self.accepted_time_periods_adapted is not None:
            return self.accepted_time_periods_adapted
        else:
            return self._accepted_time_periods

    def get_shift_time(self):
        if self.accepted_time_periods_adapted is not None:
            return self.accepted_time_periods_adapted[0][0] - self.accepted_time_periods[-1][0]

        if self.last_accepted_time_periods is not None:
            return self.accepted_time_periods[0][0] - self.last_accepted_time_periods[-1][0][0][0]

    def get_process_execution_time(self):
        """If required process_execution_time not specified, take the expected process_execution time"""
        if self.required_process_execution_time is not None:
            return self.required_process_execution_time

        else:
            return self.expected_process_execution_time

    def expand_by(self, expansion_duration):
        if not (expansion_duration < 0):
            return

        expansion_duration64 = np.timedelta64(expansion_duration, "s")
        first_time_period = \
            np.array([[self.accepted_time_periods[0][0] + expansion_duration64,
                       self.accepted_time_periods[0][1]]])  # ensure to visit the property
        self.accepted_time_periods = np.concatenate((first_time_period, self.accepted_time_periods[1:]), axis=0)

        # ToDo

        date_range = \
            pd.date_range(self.preference_values.head(1).index[0] + expansion_duration64,
                          self._preference_values.head(1).index[0] - one_second64, freq="s")

        expansion_df = \
            pd.DataFrame(np.repeat(self.preference_values.head(1).values, abs(expansion_duration)),
                         index=date_range, columns=self.preference_values.columns)

        self.preference_values = pd.concat([expansion_df, self.preference_values], axis=0)

    def get_preferences_with_durations(self, entity_type=None) -> (pd.DataFrame, int, float):
        """
        :return preferences: preferences over period
        :return Duration: Duration needed to execute the process_executions_component
        :return Support times: are the combination of lead_times and follow_up_times
        """

        preferences = self.get_preferences()

        lead_time, follow_up_time = self._determine_support_times(entity_type)
        support_time = lead_time + follow_up_time
        duration = self.get_process_execution_time()

        accepted_time_periods = self.get_accepted_time_periods()
        if not accepted_time_periods.size:
            raise Exception

        return preferences, duration, support_time

    def get_preferences_with_durations_planning(self, start_time=None, end_time=None) -> (pd.DataFrame, int, {}, {}):
        """Get the preference_values for a process_execution to schedule"""

        preferences = self.get_preferences()

        if start_time is not None:
            preferences = preferences[preferences["Timestamp"] >= start_time]
        if end_time is not None:
            preferences = preferences[preferences["Timestamp"] <= end_time]

        accepted_time_periods = self.get_accepted_time_periods()

        if preferences.size == 0:
            pass

        elif preferences["Timestamp"][0] > accepted_time_periods[0][0]:
            date_range_lst = [np.arange(accepted_time_period[0], accepted_time_period[1],
                                        one_second64)
                              for accepted_time_period in accepted_time_periods]
            time_stamps = np.concatenate(date_range_lst)
            time_stamps = time_stamps[time_stamps < preferences["Timestamp"][0]]
            values = np.repeat(preferences["Value"][0], time_stamps.size)
            preference_values_data_type = np.dtype([("Timestamp", "datetime64[s]"),
                                                    ("Value", "int8")])
            pre_preference_values = np.empty(values.size, dtype=preference_values_data_type)
            pre_preference_values['Timestamp'] = time_stamps
            pre_preference_values['Value'] = values

            self.preference_values = np.concatenate([pre_preference_values, preferences], axis=0)

        elif preferences["Timestamp"][0] < accepted_time_periods[0][0]:
            raise Exception

        duration = self.get_process_execution_time()

        return preferences, duration, self.lead_time, self.follow_up_time

    def _determine_support_times(self, entity_type=None):
        """Support times are the combination of lead_times and follow_up_times"""
        if entity_type is not None:

            if entity_type in self.lead_time:
                lead_time = self.lead_time[entity_type]
            else:
                lead_time = 0
            if entity_type in self.follow_up_time:
                follow_up_time = self.follow_up_time[entity_type]
            else:
                follow_up_time = 0

        else:
            lead_time_lst = list(self.lead_time.values())
            if lead_time_lst:
                lead_time = lead_time_lst[0]
            else:
                lead_time = 0
            follow_up_time_lst = list(self.follow_up_time.values())
            if follow_up_time_lst:
                follow_up_time = follow_up_time_lst[0]
            else:
                follow_up_time = 0

        return int(np.ceil(round(lead_time, 1))), int(np.ceil(round(follow_up_time, 1)))

    def determine_lead_time_entity_entity_types(self, entity_entity_types):
        return determine_lead_time_entity_entity_types(entity_entity_types, self.lead_time)

    def determine_follow_up_time_entity_entity_types(self, entity_entity_types):
        return determine_follow_up_time_entity_entity_types(entity_entity_types, self.follow_up_time)

    def update_lead_time_time(self, entity_entity_types, lead_time):
        if entity_entity_types is None or not self.lead_time:
            return

        if entity_entity_types in self.lead_time:
            self.lead_time[entity_entity_types] = lead_time
        elif isinstance(entity_entity_types, EntityType):
            if entity_entity_types.super_entity_type in self.lead_time:
                self.lead_time[entity_entity_types.super_entity_type] = lead_time
        elif isinstance(entity_entity_types, Entity):
            if entity_entity_types.entity_type in self.lead_time:
                self.lead_time[entity_entity_types.entity_type] = lead_time
            elif entity_entity_types.entity_type is not None:
                if entity_entity_types.entity_type.super_entity_type in self.lead_time:
                    self.lead_time[entity_entity_types.entity_type.super_entity_type] = lead_time

    def update_follow_up_time(self, entity_entity_types, follow_up_time):
        if entity_entity_types is None or not self.follow_up_time:
            return

        if entity_entity_types in self.follow_up_time:
            self.follow_up_time[entity_entity_types] = follow_up_time
        elif isinstance(entity_entity_types, EntityType):
            if entity_entity_types.super_entity_type in self.follow_up_time:
                self.follow_up_time[entity_entity_types.super_entity_type] = follow_up_time
        elif isinstance(entity_entity_types, Entity):
            if entity_entity_types.entity_type in self.follow_up_time:
                self.follow_up_time[entity_entity_types.entity_type] = follow_up_time
            elif entity_entity_types.entity_type is not None:
                if entity_entity_types.entity_type.super_entity_type in self.follow_up_time:
                    self.follow_up_time[entity_entity_types.entity_type.super_entity_type] = follow_up_time

    def get_preferences(self):
        """Return the mean over the preference values in the accepted time period"""

        accepted_preference_values = self.get_accepted_preference_values()
        preference_values = self._get_preference_mean_df(accepted_preference_values)

        return preference_values

    def _get_preference_mean_df(self, preference_array):
        """return the mean df over the columns in preference_df"""

        # ToDo: more than one value array stacked
        # if preference_df.shape[1] > 1:
        #     preference_df_partial = preference_df.iloc[:, 1:]
        #     preference_series_mean = preference_df_partial.mean(axis=1, numeric_only=True)
        #     preference_df_mean = pd.DataFrame(preference_series_mean, index=preference_df.index)
        # else:
        #     preference_df_mean = preference_df
        #
        # return preference_df_mean

        return preference_array

    def get_accepted_preference_values(self) -> np.array:
        """consider the accepted_time_periods on the preference_values"""
        accepted_time_periods = self.get_accepted_time_periods()
        if accepted_time_periods.size == 0:
            return np.array([])

        # find the time stamps that are in the accepted_time_periods
        start_times = accepted_time_periods[:, 0][:, None]

        end_times = accepted_time_periods[:, 1][:, None]
        time_stamps_preferences = self.preference_values["Timestamp"]
        time_stamps_preferences_tile = np.tile(time_stamps_preferences, (start_times.size, 1))
        mask = ((start_times <= time_stamps_preferences_tile) & (time_stamps_preferences_tile <= end_times))

        accepted_preference_values = self.preference_values[mask.max(axis=0)]

        return accepted_preference_values

    def get_process_execution_preference_before(self, expected_process_execution_time, process_execution,
                                                lead_time, follow_up_time, min_time_restriction64, take_min=False):
        # if not self.accepted_time_periods.size:
        #     return None

        preference_values = self.preference_values[["Timestamp", "Value"]].copy()
        if take_min:
            new_accepted_time_periods = np.array([[min_time_restriction64, self.accepted_time_periods[-1][1]]])

        else:
            new_accepted_time_periods = np.array([[self.accepted_time_periods[0][0],
                                                   self.accepted_time_periods[-1][1]]])

            if self.accepted_time_periods[0][0] == self.preference_values["Timestamp"][0]:
                # the first start_time_stamp is in the accepted_time_periods -
                # therefore the time_stamps before are also possible to be available
                possible_duration = (min_time_restriction64 - self.accepted_time_periods[0][0]).item().seconds
                pre_time_delta = int(np.ceil(max(np.round(np.array(list(follow_up_time.values())), 1).tolist() + [0])))
                pre_time_delta = min(possible_duration, pre_time_delta)
                if pre_time_delta:
                    preference_values["Timestamp"] = (preference_values["Timestamp"] -
                                                      np.timedelta64(pre_time_delta, "s"))
                    new_accepted_time_periods -= np.timedelta64(pre_time_delta, "s")

        origin = process_execution.origin
        destination = process_execution.destination
        process_execution_preference = \
            ProcessExecutionPreference(reference_objects=process_execution,
                                       accepted_time_periods=new_accepted_time_periods,
                                       preference_values=preference_values, lead_time=lead_time,
                                       expected_process_execution_time=expected_process_execution_time,
                                       follow_up_time=follow_up_time, origin=origin, destination=destination,
                                       long_time_reservation_duration=self.long_time_reservation_duration)

        return process_execution_preference

    def get_copy(self):
        # copy the self/ object
        new_process_execution_preference = copy(self)
        # overwrite attributes that are StaticModelGenerator
        new_process_execution_preference.reference_objects = self.reference_objects.copy()
        new_process_execution_preference._accepted_time_periods = self.accepted_time_periods.copy()
        if self.last_accepted_time_periods is not None:
            new_process_execution_preference.last_accepted_time_periods = copy(self.last_accepted_time_periods)
        new_process_execution_preference.accepted_time_periods_changed = False
        new_process_execution_preference.preference_values = self.preference_values.copy()
        new_process_execution_preference.expected_process_execution_time = copy(self.expected_process_execution_time)
        new_process_execution_preference.required_process_execution_time = copy(self.required_process_execution_time)
        new_process_execution_preference.lead_time = self.lead_time.copy()
        new_process_execution_preference.follow_up_time = self.follow_up_time.copy()
        new_process_execution_preference.origin = self.origin
        new_process_execution_preference.destination = self.destination

        return new_process_execution_preference

    def get_process_execution_preference_after(self, expected_process_execution_time, process_execution,
                                               lead_time, follow_up_time):
        pass

    def merge_resource(self, resource_preference: EntityPreference):
        """Merge the preference values from the resource into the ProcessExecutionsPreference
        # ToDo: maybe not the right place (alternatively only merge the free time periods)
        """

        if not self.accepted_time_periods.any():
            return

        lead_time = 0
        follow_up_time = 0
        if self.lead_time or self.follow_up_time:
            reference_object = self.reference_objects[0]
            reference_entity_type = reference_object.entity_type

            if reference_entity_type in self.lead_time:
                lead_time = self.lead_time[reference_entity_type]

            if reference_entity_type in self.follow_up_time:
                follow_up_time = self.follow_up_time[reference_entity_type]

        accepted_time_periods = self.accepted_time_periods.copy()
        accepted_time_periods = accepted_time_periods.astype(dtype="datetime64[s]")
        start_time = accepted_time_periods[0][0] - np.timedelta64(int(np.ceil(round(lead_time, 1))), "s")
        end_time = accepted_time_periods[-1][1] + np.timedelta64(int(np.ceil(round(follow_up_time, 1))), "s")
        time_slot_duration_free_periods = self.get_process_execution_time() + lead_time + follow_up_time
        free_time_periods = \
            resource_preference.get_free_time_periods(time_slot_duration=time_slot_duration_free_periods,
                                                      start_time=start_time, end_time=end_time)
        self.accepted_time_periods = get_overlaps(self.accepted_time_periods, free_time_periods)

    def merge_vertical(self, process_executions_preferences):
        # merge anything over preference_values{x} (preference_values is always from the requester)

        possible_time_periods_components = self._get_possible_time_periods_vertical(process_executions_preferences)

        merged_time_periods = _get_merged_time_periods_vertical(possible_time_periods_components,
                                                                data_type="datetime64[s]")
        if not merged_time_periods.any():
            self.accepted_time_periods = np.array([[]])
            return

        possible_merged_time_periods = self._get_possible_merged_time_periods_vertical(merged_time_periods)
        # if not possible_time_periods.any():  # debug
        #     print()

        # update the accepted_time_period
        self.accepted_time_periods = possible_merged_time_periods
        for process_executions_preference in process_executions_preferences:
            process_executions_preference.accepted_time_periods = possible_merged_time_periods

            # if not possible_merged_time_periods.size:
            #     raise Exception

    def _get_possible_time_periods_vertical(self, process_executions_preferences):
        # goal can be more frequent
        accepted_time_periods = [process_executions_preference.get_accepted_time_periods()
                                 for process_executions_preference in process_executions_preferences]
        accepted_time_periods_self = self.get_accepted_time_periods()
        accepted_time_periods.append(accepted_time_periods_self)

        for process_executions_preference in process_executions_preferences:
            self.lead_time |= process_executions_preference.lead_time
            self.follow_up_time |= process_executions_preference.follow_up_time

        return accepted_time_periods

    def _get_possible_merged_time_periods_vertical(self, merged_time_periods_complete):
        """Consider the expected_process_execution_time"""
        mask = np.where(merged_time_periods_complete[:, 1] - merged_time_periods_complete[:, 0] >=
                        np.timedelta64(self.get_process_execution_time(), "s"))

        possible_merged_time_periods = merged_time_periods_complete[mask]

        return possible_merged_time_periods

    def feasible(self):
        if self.accepted_time_periods.any():
            feasible = True
        else:
            feasible = False

        return feasible

    def _get_possible_time_periods_raw(self, preference_df):
        """ToDo: description"""
        # if (preference_df_mean.index == np.float(0)).any() | (preference_df_mean.index == np.array(np.nan)).any():
        #     print
        # filter nan values and 0 values
        preference_times = preference_df.index
        possible_time_periods = np.where(~(preference_times == np.float64(0)) |
                                         (preference_times == np.array(np.nan)))[0]

        if possible_time_periods.size:
            slices = arr_sequence_detection(possible_time_periods)
            possible_time_periods_raw_flatten = preference_df.iloc[slices.flatten()].index.to_numpy()
            possible_time_periods_raw_a = possible_time_periods_raw_flatten.reshape(slices.shape)

            if self.get_accepted_time_periods().any():
                possible_time_periods_raw_a = \
                    get_overlaps_periods(arr_a=possible_time_periods_raw_a, arr_b=self.get_accepted_time_periods())

        else:
            possible_time_periods_raw_a = np.array(())

        return possible_time_periods_raw_a

    def merge_horizontal(self, preferences_before, connector_object_entity_type=None, path_link_type=None,
                         shift_time=None):
        """Merge horizontal means that a process chain is combined if possible"""

        if shift_time is not None:
            new_accepted_time_periods = self.accepted_time_periods.copy()
            if new_accepted_time_periods.size:
                new_accepted_time_periods[0][0] += shift_time
                self.accepted_time_periods = new_accepted_time_periods

        if self.get_accepted_time_periods().size == 0:
            self.accepted_time_periods = np.array([[]])
            return

        if not preferences_before:
            return

        accepted_time_periods_components_merged = \
            self._get_accepted_time_periods_components(preferences_before, connector_object_entity_type)

        if path_link_type == ProcessExecutionsPath.LINK_TYPES.LOOSE:
            self.merge_loose(accepted_time_periods_components_merged)

        else:
            if connector_object_entity_type is None:
                raise Exception

            self.merge_fixed(accepted_time_periods_components_merged)

    def merge_loose(self, accepted_time_periods_components_merged):
        """Merge loose means that the preferences before should be handled before, but when is not relevant
        for the process itself. This is used for the material supply that has the requirement to be finished before
        the processing process at the work station begins"""

        if accepted_time_periods_components_merged.size == 0:
            self.accepted_time_periods = np.array([[]])
            return

        accepted_time_periods = self.get_accepted_time_periods().copy()
        earliest_provision_time = accepted_time_periods_components_merged[0][0]
        accepted_time_periods = accepted_time_periods[accepted_time_periods[:, 1] > earliest_provision_time]
        if accepted_time_periods.size == 0:
            self.accepted_time_periods = np.array([[]])
            return

        # if accepted_time_periods[0][0] > earliest_provision_time:
        #     accepted_time_periods[0][0] = earliest_provision_time
        expected_process_execution_time = self.get_process_execution_time()
        expected_process_execution_time64 = np.timedelta64(expected_process_execution_time, "s")
        possible_time_periods = \
            accepted_time_periods[np.where(accepted_time_periods[:, 1] - accepted_time_periods[:, 0] >=
                                           expected_process_execution_time64)]
        self.accepted_time_periods = possible_time_periods

        # case: the material delivery comes earlier than the earliest execution_time
        # take the highest value of the times before as the first value
        # else:
        #     pass
        # possible_times = pd.date_range(merged_dfs.index[0],
        #                                self.preference_values.index[0] - one_second64, freq="s")
        # max_values = np.repeat(merged_dfs.loc[possible_times].max().iloc[0], len(possible_times))
        # im_possible_df = pd.DataFrame({"Value": max_values}, index=possible_times)

        # to influence the process_execution_time
        # expected_process_mean_time = int(self.get_process_execution_time() / 2) ToDo
        # interim_df = merged_dfs[: min(merged_dfs.index[-1] - one_second64, self.preference_values.index[-1])]
        # interim_df.columns = ["Value"]
        # # interim_df.index += np.timedelta64(expected_process_mean_time, "s")
        # end_df = pd.DataFrame()
        # if self.preference_values.index[0] < merged_dfs.index[-1] < self.preference_values.index[-1]:
        #     time_stamps_end = pd.date_range(merged_dfs.index[-1],
        #                                     self.preference_values.index[-1], freq="s")
        #     max_time = np.repeat(max(merged_dfs), len(time_stamps_end))
        #     end_df = pd.DataFrame(max_time, columns=["Value"], index=time_stamps_end)
        #
        # # combine the preferences before with the preferences now to restrict the accepted_time_period
        # combi_preference_values = self.preference_values.copy()
        # combi_preference_values["preference_values_before"] = pd.concat([interim_df, end_df], axis=0)
        #
        # slices_time_stamps = self._get_possible_time_periods_raw(combi_preference_values)
        #
        # if not slices_time_stamps.size:
        #     self.accepted_time_periods = np.array([[]])
        #     return
        #
        # expected_process_execution_time = np.timedelta64(self.get_process_execution_time(), "s")
        # possible_time_periods = \
        #     slices_time_stamps[np.where(slices_time_stamps[:, 1] - slices_time_stamps[:, 0] >=
        #                                 expected_process_execution_time)]

        # self.accepted_time_periods = possible_time_periods

    def merge_fixed(self, accepted_time_periods_horizontal_before):
        """Merge fixed is the case when for example transport is performed (all elements of the process chain
        are executed directly after each other if possible)"""

        accepted_time_periods = self.get_accepted_time_periods().copy()

        # assumption: the accepted_time_periods of the predecessor and the successor must intersect each other
        accepted_time_periods_combined = \
            get_overlaps(accepted_time_periods, accepted_time_periods_horizontal_before).astype("datetime64[s]")

        combi_accepted_time_periods = accepted_time_periods_combined
        if combi_accepted_time_periods.shape[0] == 0:
            self.accepted_time_periods = np.array([[]])
            return

        # lead_time = self.determine_lead_time_entity_entity_types(connector_object)
        # follow_up_time = self.determine_follow_up_time_entity_entity_types(connector_object)
        expected_process_time = self.get_process_execution_time()

        path_time = expected_process_time  # + lead_time + follow_up_time

        slices_time_stamps = combi_accepted_time_periods
        if slices_time_stamps.any():
            possible_time_periods = \
                slices_time_stamps[np.where(slices_time_stamps[:, 1] - slices_time_stamps[:, 0] >=
                                            np.timedelta64(int(path_time), "s"))]
        else:
            possible_time_periods = np.array([[]])

        self.accepted_time_periods = possible_time_periods

    def _get_adapted_preference_values(self, process_executions_preferences, connector_object):
        process_executions_preferences_adapted = []

        follow_up_time = []
        for process_executions_preference in reversed(process_executions_preferences):
            preference = process_executions_preference.get_accepted_preference_values()

            columns_idx = list(range(1, preference.shape[1]))
            if columns_idx:
                preference_adapted = preference.iloc[:, columns_idx].copy()
            else:
                preference_adapted = preference.copy()

            shift_time = int(process_executions_preference.expected_process_execution_time + sum(follow_up_time))
            # shift only with expected process_execution_time because the preference_values are built on it

            preference_adapted.index += np.timedelta64(shift_time, "s")
            if process_executions_preference.required_process_execution_time:
                process_executions_preference.update_follow_up_time(connector_object, sum(follow_up_time))

            follow_up_time.append(process_executions_preference.get_process_execution_time())

            process_executions_preferences_adapted.append(preference_adapted)

        for idx, process_executions_preference in enumerate(process_executions_preferences[1:]):
            if self.required_process_execution_time:
                process_executions_preference.update_lead_time(connector_object, sum(follow_up_time[idx + 1:]))

        return process_executions_preferences_adapted

    def _get_accepted_time_periods_components(self, process_executions_preferences, connector_object_entity_type):
        """Returns merged accepted time periods with consideration of expected time periods"""

        accepted_time_periods_components = []
        follow_up_time = []
        for process_executions_preference in reversed(process_executions_preferences):
            accepted_time_periods = process_executions_preference.get_accepted_time_periods().copy()

            expected_process_execution_time = process_executions_preference.expected_process_execution_time
            shift_time = int(expected_process_execution_time + sum(follow_up_time))
            try:
                accepted_time_periods += np.timedelta64(shift_time, "s")
                # shift only with expected process_execution_time because the preference_values are built on it
            except:
                print(accepted_time_periods, np.timedelta64(shift_time, "s"))
                raise Exception

            if process_executions_preference.required_process_execution_time:
                process_executions_preference.update_follow_up_time(connector_object_entity_type, sum(follow_up_time))

            follow_up_time.append(process_executions_preference.get_process_execution_time())

            accepted_time_periods_components.append(accepted_time_periods)

        for idx, process_executions_preference in enumerate(process_executions_preferences[1:]):
            if self.required_process_execution_time:
                process_executions_preference.update_lead_time(connector_object_entity_type,
                                                               sum(follow_up_time[idx + 1:]))

        accepted_time_periods_components_merged = \
            _get_merged_time_periods_vertical(accepted_time_periods_components, data_type="datetime64[s]")

        return accepted_time_periods_components_merged

    def get_accepted_start_period(self):
        start_ = np.min(self.accepted_time_periods)
        end_ = np.max(self.accepted_time_periods) - np.timedelta64(self.get_process_execution_time(), "s")
        return [start_, end_]

    def get_predecessor_preference(self, process_execution, origin, destination, period_of_demand):
        """Used for the material_supply that means that the material_supply should end (as a predecessor)
        before the demand occurs"""
        new_accepted_time_periods = period_of_demand.copy()
        # ToDo: maybe the time_period can be a little bit longer

        preference_values = \
            self.get_preference_values(new_accepted_time_periods, time_slot_duration=None, lead_times=None,
                                       follow_up_times=None, process=None, destination=None,
                                       preference_function=self._get_preference_value_predecessor_timestamp)
        expected_process_execution_time = None

        if preference_values.shape[0] == 0:
            successful = False
            return None, successful

        process_execution_preference = \
            ProcessExecutionPreference(reference_objects=process_execution,
                                       accepted_time_periods=new_accepted_time_periods,
                                       preference_values=preference_values,
                                       expected_process_execution_time=expected_process_execution_time,
                                       origin=origin, destination=destination)
        successful = True
        return process_execution_preference, successful

    def _get_preference_value_predecessor_timestamp(self, time_stamp, start_period, len_period):
        # it should be checked before if the timestamp is a free one

        if type(time_stamp) == datetime:
            preference_value = 1
        else:
            preference_value = 1

        if preference_value > 1:
            raise Exception("Not normalized")

        return preference_value

    def get_free_time_periods(self, time_slot_duration=None, start_time=None, end_time=None):
        """Get free time_slots based on the own _process_execution_plan"""

        accepted_time_periods = self.accepted_time_periods.copy()
        if start_time is not None:
            accepted_time_periods = accepted_time_periods[accepted_time_periods[:, 1] >= start_time]
            if accepted_time_periods.size == 0:
                return accepted_time_periods
            if accepted_time_periods[0][0] <= start_time:
                accepted_time_periods[0][0] = start_time

        if end_time is not None:
            accepted_time_periods = accepted_time_periods[accepted_time_periods[:, 0] <= end_time]
            if accepted_time_periods.size == 0:
                return accepted_time_periods
            if accepted_time_periods[-1][1] >= end_time:
                accepted_time_periods[-1][1] = end_time

        return accepted_time_periods

    def get_accepted_start_time_successor(self):
        """Get the accepted start time for a successor in the process chain, the preference participate"""

        accepted_time_periods = self.get_free_time_periods()
        if accepted_time_periods.size == 0:
            return None
        start_time = accepted_time_periods[0, 0]
        start_time += np.timedelta64(self.get_process_execution_time(), "s")

        return start_time

    def get_accepted_end_time_predecessor(self):
        """Get the accepted end time for a predecessor in the process chain, the preference participate"""

        accepted_time_periods = self.get_free_time_periods()
        if accepted_time_periods.size == 0:
            return None
        end_time = accepted_time_periods[-1, 1]
        end_time -= np.timedelta64(self.get_process_execution_time(), "s")

        return end_time

    def merge_free_time_period_into_accepted(self, free_time_periods):
        """Merge the free time period into the accepted_time_periods/ not free time is not considered as accepted"""

        accepted_free_time_periods = get_overlaps(self.accepted_time_periods, free_time_periods)

        self.accepted_time_periods = accepted_free_time_periods

    def update_accepted_time_periods_with_predecessor(self, start_time):
        """Update the accepted time period based on the start_time determined by the predecessor"""

        if self.accepted_time_periods.shape[0] == 0:
            return
        if self.accepted_time_periods[0, 0] < start_time:
            first_time_period = \
                np.array([[start_time, self.accepted_time_periods[0][1]]])  # ensure to visit the property
            self.accepted_time_periods = np.concatenate((first_time_period, self.accepted_time_periods[1:]), axis=0)

        self.accepted_time_periods[self.accepted_time_periods[:, 0] < start_time, 0] = start_time

    def update_accepted_time_periods_with_successor(self, end_time):
        """Update the accepted time period based on the end_time determined by the successor"""
        if self.accepted_time_periods.shape[0] == 0:
            return

        self.accepted_time_periods[self.accepted_time_periods[:, 1] > end_time, 1] = end_time

    def get_first_accepted_time_stamp(self):
        return self.accepted_time_periods[0][0]

    def get_last_accepted_time_stamp(self):
        return self.accepted_time_periods[-1][1]

    # methods used for scheduling from in the coordination behaviour
    def schedule_resources(self, resources_preferences_batch, predecessor_process_planned, successor_process_planned,
                           process_execution_id, work_order_id, blocker_name, issue_id, block_before):
        """
        Schedule a time slot for the process_execution (self: preference for the process_execution)
        on the resources got in the input parameters
        :param resources_preferences_batch: resources and preferences according the resources
        :param predecessor_process_planned: determines if the predecessor_process is planned
        :param successor_process_planned: determines if the successor_process is planned # ToDo: identifier needed
        :param process_execution_id: the process_execution_id is needed for the blocking as reference
        :param issue_id: the id of the last issue
        :param work_order_id: the work_order_id is needed for the blocking as reference
        :param blocker_name: the blocker_name is needed for the blocking as reference to know the right contact
        if change planning occur
        :param block_before: the block_before is needed for the blocking if interim times in a process_chain
        should not be blocked by other resources
        """

        # influences the lead and follow-up time, as well as the accepted time period
        last_issue_id = None
        self.accepted_time_periods_adapted = None
        # if planned before and the plan was not possible -> reset the planning
        if predecessor_process_planned:
            if isinstance(predecessor_process_planned, np.datetime64):
                start_time = pd.Timestamp(predecessor_process_planned)
            elif isinstance(predecessor_process_planned, datetime):
                start_time = pd.Timestamp(predecessor_process_planned)
            else:
                start_time = predecessor_process_planned.get_latest_end_time()[1]
                last_issue_id = predecessor_process_planned.issue_id
        else:
            start_time = None

        if successor_process_planned:
            end_time = successor_process_planned.get_earliest_start_time()[1]
            last_issue_id = successor_process_planned.issue_id
        else:
            end_time = None

        accepted_time_periods = self.get_free_time_periods(start_time=start_time, end_time=end_time)
        if last_issue_id == issue_id and accepted_time_periods.size != 0 and start_time:
            if accepted_time_periods[0, 0] > start_time:
                accepted_time_periods = self._expand_accepted_time_periods_planning(accepted_time_periods, start_time)

        process_execution_preference_values, time_slot_duration, lead_times, follow_up_times = \
            self.get_preferences_with_durations_planning(start_time, end_time)
        if process_execution_preference_values.size == 0:
            successful = False
            best_matching_time_slot = (None, None)
            return successful, best_matching_time_slot

        # reset lead_time or follow-up time if the time before/ after already scheduled
        if predecessor_process_planned:
            lead_times = {}
        if successor_process_planned:
            follow_up_times = {}

        successful, best_matching_time_slot, preference_df = \
            self._get_best_matching_time_slot(process_execution_preference=process_execution_preference_values,
                                              resources_preferences_batch=resources_preferences_batch,
                                              time_slot_duration=time_slot_duration, lead_times=lead_times,
                                              follow_up_times=follow_up_times,
                                              accepted_time_periods=accepted_time_periods,
                                              after_issue=issue_id)

        conveyor_belt_involved = [resource_preference
                                  for resource_preference in resources_preferences_batch
                                  if isinstance(resource_preference.reference_objects[0], ConveyorBelt)]
        if not successful and not conveyor_belt_involved:
            return successful, best_matching_time_slot

        if time_slot_duration <= 1 and successful:
            successful, best_matching_time_slot = \
                self._check_clash(resources_preferences_batch, best_matching_time_slot, time_slot_duration,
                                  preference_df, issue_id)

        if not successful:
            if conveyor_belt_involved:
                successful, best_matching_time_slot, preference_df = \
                    self._get_best_matching_time_slot(process_execution_preference=process_execution_preference_values,
                                                      resources_preferences_batch=resources_preferences_batch,
                                                      time_slot_duration=time_slot_duration, lead_times=lead_times,
                                                      follow_up_times=follow_up_times,
                                                      accepted_time_periods=accepted_time_periods,
                                                      after_issue=None)
                if successful:
                    successful, best_matching_time_slot = \
                        self._check_clash(resources_preferences_batch, best_matching_time_slot, time_slot_duration,
                                          preference_df, issue_id)

        if not successful:
            # print("Not successful")
            # for resource_preference in resources_preferences_batch:
            #     pref = resource_preference.get_preference_values(accepted_time_periods, time_slot_duration, lead_times=lead_times,
            #                                           follow_up_times=follow_up_times, destination=self.destination,
            #                                           process=None,
            #                                           preference_function=self._get_preference_value_timestamp,
            #                                           after_issue=issue_id)
            #     if pref.empty:
            #         print(resource_preference.reference_objects[0].name)

            return successful, best_matching_time_slot

        self.accepted_time_periods_adapted = np.array([[best_matching_time_slot[0], best_matching_time_slot[1]]],
                                                      dtype="datetime64[s]")
        for resource_preference in resources_preferences_batch:
            successful = resource_preference.block_time_slot(best_matching_time_slot[0], best_matching_time_slot[1],
                                                             blocker_name, process_execution_id, work_order_id,
                                                             issue_id, block_before)

            if not successful:
                return successful, best_matching_time_slot

        # 'update' (set the required) the expected process_execution time
        if predecessor_process_planned:
            self.required_process_execution_time = (best_matching_time_slot[1] - start_time).seconds
        else:
            self.required_process_execution_time = (best_matching_time_slot[1] - best_matching_time_slot[0]).seconds

        return successful, best_matching_time_slot

    def _expand_accepted_time_periods_planning(self, accepted_time_periods, start_time):
        """Expand accepted_time_periods before because the process before ends earlier as expected"""

        accepted_time_periods[0, 0] = start_time
        self.accepted_time_periods_adapted = accepted_time_periods
        end_time = self.preference_values["Timestamp"][0] - one_second64
        times_before = np.arange(start_time, end_time, one_second64)
        if times_before.size > 0:  # ToDo
            rows_before = pd.DataFrame(np.repeat(self.preference_values.values[0], times_before.shape[0], axis=0),
                                       index=times_before)
            rows_before.columns = self.preference_values.columns

            self.preference_values = pd.concat([rows_before, self.preference_values], axis=0)

        return accepted_time_periods

    def schedule_parts(self, predecessor_process_planned, successor_process_planned):
        """
        Schedule a time slot for the process_execution (self: preference for the process_execution)
        on the resources got in the input parameters
        :param predecessor_process_planned: determines if the predecessor_process is planned
        :param successor_process_planned: determines if the successor_process is planned # ToDo: identifier needed
        """

        # influences the lead and follow-up time, as well as the accepted time period
        if predecessor_process_planned:
            start_time = predecessor_process_planned.get_latest_end_time()[1]
        else:
            start_time = None
        if successor_process_planned:
            end_time = successor_process_planned.get_earliest_start_time()[1]
        else:
            end_time = None

        process_execution_preference_values, duration, lead_times, follow_up_times = \
            self.get_preferences_with_durations_planning(start_time, end_time)

        successful, best_matching_time_slot, preference_df = \
            self._determine_best_matching_time_slot(process_execution_preference_values, duration)

        return successful, best_matching_time_slot

    def _get_best_matching_time_slot(self, process_execution_preference,
                                     resources_preferences_batch: list[EntityPreference],
                                     time_slot_duration, lead_times, follow_up_times, accepted_time_periods,
                                     after_issue):
        """Returns the best matching time slot
        in relation to the process_execution preference and the resource preferences
        Return the start and the end time of the best matching time slot
        """
        # ToDo: lead_times and follow_up_times should be considered here

        resources_preference_values = \
            [pd.DataFrame(resource_preference.get_preference_values(
                accepted_time_periods, time_slot_duration, lead_times=lead_times, follow_up_times=follow_up_times,
                destination=self.destination, process=None, preference_function=self._get_preference_value_timestamp,
                after_issue=after_issue)).set_index('Timestamp')
             for resource_preference in resources_preferences_batch]

        process_resources_df = pd.concat([pd.DataFrame(process_execution_preference).set_index('Timestamp')] +
                                         resources_preference_values, axis=1)

        successful, best_matching_time_slot, preference_df = \
            self._determine_best_matching_time_slot(process_resources_df, time_slot_duration)

        return successful, best_matching_time_slot, preference_df

    def _determine_best_matching_time_slot(self, process_resources_df, time_slot_duration):
        """Determine the best matching time slot based on the given preference values (df) and the time needed"""

        process_resources_df = process_resources_df.dropna()
        process_resources_combined_df = process_resources_df.mean(axis=1)

        sequence_indexes = \
            arr_sequence_detection((process_resources_combined_df.index.to_numpy(dtype="datetime64") -
                                    np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))

        # consider the durations (build a combination of preference values)
        time_slot_duration = int(time_slot_duration)
        if time_slot_duration == 0:
            time_slot_duration = 1

        rolled_dfs = \
            [self.get_preferences_over_time(process_resources_combined_df, time_slot_duration, start_period, end_period)
             for start_period, end_period in sequence_indexes]

        if not rolled_dfs:
            successful = False
            return successful, (None, None), pd.DataFrame()

        preference_df = pd.concat(rolled_dfs, axis=0)
        try:
            end_time = preference_df.idxmax(axis=0) + one_second64
        except:
            successful = False
            return successful, (None, None), pd.DataFrame()

        start_time = end_time - np.timedelta64(time_slot_duration, 's')
        successful = True
        return successful, (start_time, end_time), preference_df

    def get_preferences_over_time(self, process_resources_df, time_slot_duration, start_period, end_period):
        """Determine the preferences over time through consideration of the duration"""

        preferences_over_time = \
            process_resources_df.iloc[int(start_period): int(end_period + 1)].rolling(
                window=time_slot_duration, min_periods=time_slot_duration).sum()[time_slot_duration - 1:]
        return preferences_over_time

    def _check_clash(self, resources_preferences_batch, best_matching_time_slot, time_slot_duration, preference_df,
                     issue_id):
        """needed because for one second or smaller time_slots
        (the preference value df has a sampling rate of 1 second)"""

        successful = True
        clash = True
        if time_slot_duration == 0:
            best_matching_time_slot = (best_matching_time_slot[0], best_matching_time_slot[0])
        time_slot_duration64 = np.timedelta64(time_slot_duration, 's')
        end_time = best_matching_time_slot[1]
        start_time = best_matching_time_slot[0]
        while clash:
            for resource_preference in resources_preferences_batch:
                clash = resource_preference.check_clashing(start_time=start_time, end_time=end_time, issue_id=issue_id,
                                                           time_slot_duration=time_slot_duration64)
                if clash:
                    break

            if clash:
                try:
                    preference_df.drop(end_time - one_second64, inplace=True)
                except:
                    preference_df.drop(end_time, inplace=True)
                if preference_df.empty:
                    successful = False
                    break

                end_time = preference_df.idxmax(axis=0) + one_second64
                start_time = end_time - time_slot_duration64
                best_matching_time_slot = (start_time, end_time)

        return successful, best_matching_time_slot
