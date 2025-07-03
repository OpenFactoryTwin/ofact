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

Provide a WorkCalender and a TimePlanner.
Classes
    WorkCalendar: the class is based on the timeboard (pip install timeboard) and provides a shift system.
    It has leap years and the possibility to set amendments (changes).
    ProcessExecutionPlan: is responsible for the time planning of resources and orders. Therefore, it is able
    to block, unblock time_slots and detect clashing time_slots.
    ProcessExecutionPlanConveyorBelt: because the ConveyorBelt has a different scheduling behavior
    it overwrites the ProcessExecutionPlan for ConveyorBelts

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from copy import copy, deepcopy
from datetime import datetime, time, timedelta
from typing import Optional, Union

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import DigitalTwinObject, prints_visible
from ofact.twin.state_model.helpers.helpers import convert_to_datetime, convert_to_np_timedelta, convert_to_np_datetime


# import timeboard as tb
# from timeboard.calendars.calendarbase import (nth_weekday_of_month, extend_weekends, from_easter)


class WorkCalender(DigitalTwinObject):

    def copy(self):
        work_calender_copy = super(WorkCalender, self).copy()
        return work_calender_copy


# ToDo: What is a one second entry?
# 12.00.00 - 12.00.00 or 12.00.00 - 12.00.01
one_second = np.timedelta64(1, "s")

import os
import pandas as pd
from datetime import datetime

EXCEL_FILE = 'log.xlsx'
SHEET_NAME = 'Logs'

def append_to_excel_manual(df_new, filename, sheet_name):
   if os.path.exists(filename):
       try:
           df_existing = pd.read_excel(filename, sheet_name=sheet_name)
           df_combined = pd.concat([df_existing, df_new], ignore_index=True)
       except Exception:
           # Falls das Sheet nicht existiert oder ein Fehler auftritt
           df_combined = df_new
   else:
       df_combined = df_new
   # Schreibe den kombinierten DataFrame in eine neue Datei bzw. ersetze das existierende Sheet
   with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
       df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

# Beispiel-Decorator
import functools

def log_args_to_excel(func):
   @functools.wraps(func)
   def wrapper(self, *args, **kwargs):
       log_entry = {
           'timestamp': datetime.now(),
           'self': str(self.identification) + str(self.name),
           'args': str(args),
           **kwargs
       }
       df_new = pd.DataFrame([log_entry])
       append_to_excel_manual(df_new, EXCEL_FILE, SHEET_NAME)
       return func(self, *args, **kwargs)
   return wrapper


class ProcessExecutionPlan(DigitalTwinObject):
    time_schedule_data_type = [("Start", "datetime64[ns]"),
                               ("End", "datetime64[ns]"),
                               ("Duration", "float32"),
                               ("Blocker Name", "U32"),
                               ("Process Execution ID", "int32"),
                               ("Event Type", "U32"),
                               ("Issue ID", "int32"),
                               ("Work Order ID", "int32")]

    def __init__(self,
                 name: str,
                 work_calendar: Optional[WorkCalender] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Used to plan events/ process_executions

        Parameters
        ----------
        name: name for the planner
        work_calendar: provide further time restrictions

        Attributes
        ----------
        time_now: time_stamp that says nothing before available

        time_schedule: contains the events of the resource in a structured numpy array
        - Start: start time stamp of the process_execution
        - End: end time stamp of the process_execution
        - Duration: the time between start and end time stamp
        - Blocker Name: name of the agent, who blocks the time slot
        - Process Execution ID: identification of the process_execution executed in the time_slot
        - Event Type: event type of the process_execution
        - Issue ID: issues consists of one or more than one time_slots for the same reference_process_execution_id
        (e.g. assembly of rims needs 4 rims - the part supply can be performed through 4 process_executions
        (for each part one) - therefore all of them have the same issue_id
        Use Case/ Value: If two work_stations can perform the process - both can request the parts
        with different process_execution_ids but the same issue_id (advantage: both can reserve the same parts
        (because of a XOR-relation) and the whole issue can be cancelled with little effort))
        -> issue id and process_execution_id can be the same
        - Work Order ID: identification of the production order for which the time_slot is used
        """
        super(ProcessExecutionPlan, self).__init__(identification=identification,
                                                   external_identifications=external_identifications,
                                                   domain_specific_attributes=domain_specific_attributes)
        self.name = name

        self.work_calendar = work_calendar
        self.time_now = None
        # ToDo: time_now should be the timestamp that says that previous time_slots are no longer available
        #  how to update? - simulation
        self._time_schedule: np.array = np.empty(0,
                                                 dtype=type(self).time_schedule_data_type)

    # @property
    # def time_schedule(self):
    #     return self._time_schedule
    #
    # @time_schedule.setter
    # def time_schedule(self, time_schedule):
    #     print("Time schedule setter:", self.name, time_schedule)
    #     self._time_schedule = time_schedule

    def copy(self):
        process_execution_plan_copy = super(ProcessExecutionPlan, self).copy()
        process_execution_plan_copy._time_schedule = process_execution_plan_copy._time_schedule

        return process_execution_plan_copy

    def get_scheduled_time_slots(self):
        return self._time_schedule

    def set_available_times(self, available_times: list[tuple[time, time]], start_time_stamp, horizont=5):
        """
        Set available times over the process_execution_plan
        ToDo: should be transferred to the WorkCalendar

        Parameters
        ----------
        available_times: a list of available times (start and end time)
        start_time_stamp: determine the time where the first blocks are set
        horizont: determines the amount of days the blocks are set
        """

        available_times = sorted(available_times, key=lambda x: x[0])

        if available_times:
            blocked_periods = []
        else:
            blocked_periods = [(time(0, 0), time(0, 0))]

        available_end_time = None
        start_time_time = deepcopy(start_time_stamp.time())

        while available_times:
            available_start_time, available_end_time = available_times.pop(0)
            if available_start_time > start_time_time:
                blocked_period = (start_time_time, available_start_time)
                start_time_time = available_end_time
                blocked_periods.append(blocked_period)

        if available_end_time is not None:
            if available_end_time != time(0, 0):
                end_block = (available_end_time, start_time_stamp.time())
                blocked_periods.append(end_block)

        date_ = start_time_stamp.date()
        blocked_times = []
        while horizont >= 0:
            for idx, (blocked_start_time, blocked_end_time) in enumerate(blocked_periods):
                if idx + 1 < len(blocked_periods):
                    blocked_period = (datetime.combine(date_, blocked_start_time),
                                      datetime.combine(date_, blocked_end_time))
                else:
                    blocked_period = (datetime.combine(date_, blocked_start_time),
                                      datetime.combine(date_ + timedelta(days=1), blocked_end_time))

                blocked_times.append(blocked_period)

            date_ += timedelta(days=1)
            horizont -= 1

        blocked_rows = [(np.datetime64(blocked_time[0]),  # "Start"
                         np.datetime64(blocked_time[1]),  # "End"
                         (blocked_time[1] - blocked_time[0]).seconds,  # "Duration"
                         "Planner",  # "Blocker Name"
                         0,  # "Process Execution ID"
                         "ACTUAL",  # "Event Type"
                         0,  # "Issue ID"
                         0)  # "Work Order ID"
                        for blocked_time in blocked_times]

        new_rows = np.array(blocked_rows, dtype=type(self).time_schedule_data_type)

        self._time_schedule = np.append(self._time_schedule, new_rows)

    def get_schedule(self, start_time):
        """Return the resource schedule ()"""

        schedule_with_blocks = self._time_schedule[self._time_schedule["Blocker Name"] == "Planner"][["Start", "End"]]

        if schedule_with_blocks.size == 0:
            end_time = start_time + timedelta(days=5)
            schedule_df = pd.DataFrame({"Start": [start_time],
                                        "End": [end_time]})
            return schedule_df

        # Compute unblocked times using broadcasting
        unblocked_start = schedule_with_blocks['End'][:-1]
        unblocked_end = schedule_with_blocks['Start'][1:]

        # Create mask where there is a gap
        mask = unblocked_start < unblocked_end

        # Extract unblocked intervals
        schedule = np.column_stack((unblocked_start[mask], unblocked_end[mask]))

        min_day = np.datetime64(start_time, "D")
        max_day = schedule_with_blocks['End'].astype('datetime64[D]')[-1]

        if min_day != schedule_with_blocks["Start"][0]:
            start_period = np.array([[min_day, schedule_with_blocks["Start"][0]]])
            schedule = np.row_stack((start_period, schedule))

        if max_day != schedule_with_blocks["End"][-1]:
            end_period = np.array([[schedule_with_blocks["End"][-1], max_day]])
            schedule = np.row_stack((schedule, end_period))

        schedule_df = pd.DataFrame(schedule,
                                   columns=["Start", "End"])
        return schedule_df

    def get_next_possible_period(self, period_length: timedelta, start_time=None, issue_id=None,
                                 last_element: bool = False, type_="ALL") -> (datetime, datetime):
        """
        Get a time_slot that begins at the next possible start_time with the period_length

        Parameters
        ----------
        period_length: a timedelta that describes the period between start_time and end_time
        start_time: option to restrict the search
        last_element: get the last element from the list/ no appointment in between possible

        Returns
        -------
        start_time: a start_time of the next possible time slot.
        end_time: an end_time of the next possible time slot.
        """
        raise NotImplementedError("Clashig Problems")
        if not isinstance(start_time, datetime):
            start_time = convert_to_datetime(start_time)

        if last_element and self._time_schedule.size == 0:
            start_time = self.get_last_time_stamp()

        end_time = start_time + period_length
        clashing_process_executions = self._get_clashing_process_executions(start_time=start_time,
                                                                            end_time=end_time,
                                                                            issue_id=issue_id,
                                                                            type_=type_)

        while not clashing_process_executions.empty:
            start_time = clashing_process_executions[-1]["End"]
            end_time = start_time + period_length
            clashing_process_executions = self._get_clashing_process_executions(start_time=start_time,
                                                                                end_time=end_time,
                                                                                issue_id=issue_id,
                                                                                type_=type_)
        # ToDo (later): blocking?
        return start_time, end_time

    def _get_blocked_periods_calendar_extract(self, start_time, end_time, time_slot_duration,
                                              issue_id: Optional[int] = None) -> (
            np.array):
        """
        Get a numpy array of blocked time periods

        Parameters
        ----------
        start_time: datetime, start of a time period
        end_time: datetime, end of a time period
        issue_id: identification of the issue

        Returns
        -------
        blocked_periods_calendar_extract: a numpy array with start and end timestamps of a blocked periods
        as array entries
        """
        time_slot_duration_s = time_slot_duration / one_second

        calendar_extract_df = self._get_clashing_process_executions(start_time=start_time,
                                                                    end_time=end_time,
                                                                    issue_id=issue_id,
                                                                    time_slot_duration=time_slot_duration_s)

        if time_slot_duration_s == 0 and calendar_extract_df.size != 0:  # needed?
            # all entries with other issue should be extended - avoiding blocks at the same time
            calendar_extract_df["Start"][calendar_extract_df["Issue ID"] != issue_id] -= one_second
            calendar_extract_df["End"][calendar_extract_df["Issue ID"] != issue_id] += one_second

        blocked_periods_calendar_extract = calendar_extract_df[["Start", "End"]]
        return blocked_periods_calendar_extract

    def get_free_periods_calendar_extract(self, start_time: Optional[np.datetime64] = None,
                                          end_time: Optional[np.datetime64] = None,
                                          issue_id: int = None,
                                          time_slot_duration: Optional[np.timedelta64] = None,
                                          long_time_reservation_duration=None) -> np.array([[]]):
        """
        Get a numpy array of free time periods that match the time_slot_duration restriction

        Parameters
        ----------
        start_time: datetime, start of a time period
        end_time: datetime, end of a time period
        issue_id: allows double blocking of more than one process_execution for one issue (XOR)
        time_slot_duration: if not None, it forms the lower boundary/ time limit for periods
        long_time_reservation_duration: the long_time_reservation_duration is used to ensure that the resource
        is also available later

        Returns
        -------
        free_periods_calendar_extract: a numpy array with start and end timestamps of a free periods as array entries
        """

        if time_slot_duration and not isinstance(self, ProcessExecutionPlanConveyorBelt):
            if (end_time - start_time) < time_slot_duration:
                # early stopping time slot duration is longer than the period given
                free_periods_calendar_extract = np.array([[]])
                return free_periods_calendar_extract

        blocked_periods_calendar_extract = \
            self._get_blocked_periods_calendar_extract(start_time=start_time, end_time=end_time,
                                                       issue_id=issue_id, time_slot_duration=time_slot_duration)

        free_periods_calendar_extract = None
        if not blocked_periods_calendar_extract.size:
            # no blocked periods
            free_periods_calendar_extract = np.array([[start_time, end_time]])
            # normal datetime would be better
            return free_periods_calendar_extract.astype("datetime64[s]").astype(datetime)

        elif blocked_periods_calendar_extract.shape[0] == 1:
            if blocked_periods_calendar_extract["Start"][0] >= start_time:
                free_periods_calendar_extract_start = \
                    np.array([[start_time, blocked_periods_calendar_extract["Start"][0]]])
                free_periods_calendar_extract = free_periods_calendar_extract_start

            blocked_end_time = blocked_periods_calendar_extract["End"][-1]
            if blocked_end_time <= end_time:

                if blocked_end_time != end_time:
                    free_periods_calendar_extract_end = np.array([[blocked_end_time, end_time]])
                    if free_periods_calendar_extract is not None:
                        free_periods_calendar_extract = np.row_stack((free_periods_calendar_extract,
                                                                      free_periods_calendar_extract_end))
                    else:
                        free_periods_calendar_extract = free_periods_calendar_extract_end

        else:
            start_period_times = blocked_periods_calendar_extract["End"][:-1]
            end_period_times = blocked_periods_calendar_extract["Start"][1:]
            free_periods_calendar_extract = np.column_stack((start_period_times, end_period_times))

            # delete periods with length 0
            # free_periods_calendar_extract[:, 1] - free_periods_calendar_extract[:, 0] > np.timedelta64(0, "s")

            if blocked_periods_calendar_extract["Start"][0] > start_time:
                start_period = np.array([[start_time,
                                          convert_to_np_datetime(blocked_periods_calendar_extract["Start"][0])]])
                free_periods_calendar_extract = np.row_stack((start_period, free_periods_calendar_extract))
            if blocked_periods_calendar_extract["End"][-1] < end_time:
                end_period = np.array([[convert_to_np_datetime(blocked_periods_calendar_extract["End"][-1]),
                                        end_time]])
                free_periods_calendar_extract = np.row_stack((free_periods_calendar_extract, end_period))

        if free_periods_calendar_extract is None:
            # early stopping - no free time
            free_periods_calendar_extract = np.array([[]])
            return free_periods_calendar_extract

        # consider only the periods that match with the time_slot_duration restriction
        periods = (free_periods_calendar_extract[:, 1] - free_periods_calendar_extract[:, 0]).astype("timedelta64[ns]")
        mask = periods >= convert_to_np_timedelta(time_slot_duration)
        free_periods_calendar_extract = free_periods_calendar_extract[mask]

        # if long_time_reservation_duration:
        #     warnings.warn("Not implemented . It should be ensured that the period of "
        #                   "the 'long_time_reservation_duration' is free of blocking")  # ToDo
        if free_periods_calendar_extract.size == 0:
            print
        return free_periods_calendar_extract.astype("datetime64[s]").astype(datetime)

    def get_free_time_from(self, start_time, issue_id=None) -> np.array:
        """Get free time period from the time_stamp until the next planned slot"""

        if start_time is None:
            start_time = np.datetime64(datetime.now(), "ns")

        mask = (self._time_schedule["Start"] > start_time) | (self._time_schedule["End"] > start_time)
        time_slots_after_start_time = self._time_schedule[mask]

        if time_slots_after_start_time.size != 0:
            next_time_slot = time_slots_after_start_time[0]
            if next_time_slot["Start"] == start_time:
                return np.array([])  # empty array, because the start time is blocked
            elif next_time_slot["Start"] < start_time:  # blocked case
                return np.array([])
            else:
                end_time = next_time_slot["Start"]
        else:
            end_time = np.datetime64("2100-01-01 00:00:00", "ns")

        free_time = np.array([start_time, end_time])

        return free_time

    def _get_clashing_process_executions(self, start_time, end_time, time_slot_duration: Union[int, float],
                                         issue_id=None, type_="ALL") -> np.array:
        """
        Get clashing_process_execution_df based on the time between start_time and end_time.

        Parameters
        ----------
        start_time: datetime, start of a time period
        end_time: datetime, end of a time period
        issue_id: time_slot blocking with the same issue_id are not considered as clash

        Returns
        -------
        clashing_process_executions_df: a dataframe with clashing process_executions
        """

        # ToDo: test max(start1, start2) < min(end1, end2)
        if self._time_schedule.size == 0:
            return self._time_schedule
        # it can be the case that two logical transfers of the same issue are at the same time

        overlap_mask = ((self._time_schedule["Start"] > start_time) & (self._time_schedule["Start"] < end_time) |
                        ((self._time_schedule["Start"] <= start_time) & (self._time_schedule["End"] > start_time)))
        # '23:26:59', '23:27:53'
        # '23:27:00', '23:27:53', 'frame and handlebar')

        # & (self._time_schedule["Start"] != self._time_schedule["End"]))
        if time_slot_duration == 0:
            # ToDo: comment
            non_zero_duration_mask = ((self._time_schedule["Start"] == self._time_schedule["End"]) &
                                      (self._time_schedule["Start"] >= start_time) &
                                      (self._time_schedule["End"] <= end_time) &
                                      (self._time_schedule["Issue ID"] != issue_id))
            mask = overlap_mask & non_zero_duration_mask
        else:
            mask = overlap_mask
        #
        # else:
        #     zero_duration_mask = ((self._time_schedule["Start"] == self._time_schedule["End"]) &
        #                           (self._time_schedule["Start"] > start_time) &
        #                           (self._time_schedule["End"] < end_time))
        #     mask = overlap_mask | zero_duration_mask

        clashing_process_executions_df = self._time_schedule[mask]

        # process_executions with the same issue_id cannot be blockers of the process_execution
        if issue_id is not None:
            clashing_process_executions_df = \
                clashing_process_executions_df[clashing_process_executions_df["Issue ID"] != issue_id]

        if type_ == "ACTUAL" or type_ == "PLAN":
            clashing_process_executions_df = \
                clashing_process_executions_df[clashing_process_executions_df["Event Type"] == type_]

        return clashing_process_executions_df

    #@log_args_to_excel
    def block_period(self, start_time, end_time, blocker_name, process_execution_id: int, work_order_id: int,
                     event_type: str = "PLAN", issue_id: Optional[int] = None, block_before=False) -> (
            [bool, list[Optional[str]], list[Optional[int]]]):
        """
        Blocks the period between start_time and end_time if possible for the process_execution_id
        (can only be unblocked by the blocker)

        Parameters
        ----------
        event_type: the event type is defined by the process_execution itself (PLAN or ACTUAL)
        start_time: start time of the period to block_period
        end_time: end time of the period to block_period
        blocker_name: (agent) name who blocks the period
        process_execution_id: the id of the process_execution that should be executed in that time
        work_order_id: the id of the production_order that is responsible for the process_execution
        issue_id: the id of an issue (for better understanding see the doc-string of the init)
        block_before: in some cases the time blocking before a process_execution is needed
        Example case: An AGV accesses the warehouse (1), load the material (2), transport them
        to the assembly station (3) and finally unload the material (4) for further processing
        Between these process_executions waiting times may occur between (1), (2), (3) and/ or (4)
        To avoid further blocking in between, it is possible to block the time inbetween by the process_execution
        after the waiting

        Returns
        -------
        successful: True if the blocking was successful
        clashing_blocker_names: the names of the agents who blocks the time needed for the new blocking
        clashing_process_execution_ids: the process_execution ids of the clashing appointments
        """
        # import inspect
        # print(f"Block: ({self.name}) {process_execution_id}", inspect.stack()[2])
        time_slot_duration = end_time - start_time

        successful = False
        # check clashing
        clashing_blocker_names, clashing_process_execution_ids = \
            self._check_clash(start_time=start_time, end_time=end_time, issue_id=issue_id,
                              time_slot_duration=time_slot_duration)

        if clashing_blocker_names[0] is not None:  # None
            return successful, clashing_blocker_names, clashing_process_execution_ids

        # check if the process_execution_id blocks already an event slot
        right_process_execution_ids = (self._time_schedule["Process Execution ID"] == process_execution_id)
        if right_process_execution_ids.any() and right_process_execution_ids.size > 0:
            clashing_blocker_names, clashing_process_execution_ids = \
                self._time_schedule[right_process_execution_ids][["Blocker Name",
                                                                  "Process Execution ID"]].tolist()[0]
            return successful, clashing_blocker_names, clashing_process_execution_ids

        # set the event / split the work schedule and set a blocked_period in the middle
        row_index = self._get_row_index(start_time, end_time)

        self._block_by_row(row_index, start_time, end_time, blocker_name, process_execution_id, event_type,
                           issue_id, work_order_id, time_slot_duration=time_slot_duration)

        if block_before is True and row_index > 1:
            if "Fließband" not in self.name:  # ToDo: name should be not the criteria
                try:
                    row_before = self._time_schedule[row_index - 1]
                except:
                    if prints_visible:
                        print(f"[{self.__class__.__name__:20}] ISSUE: BLOCKING")
                    row_before = self._time_schedule[
                        row_index - 2]  # ToDo: Where can I find the missing row - unblocking

                if row_before["Issue ID"] == issue_id:
                    start_time_waiting = row_before["End"]
                    end_time_waiting = start_time
                    if start_time_waiting != end_time_waiting and start_time_waiting < end_time_waiting:
                        event_type_before = "INTERIM"
                        self._block_by_row(row_index, start_time_waiting, end_time_waiting, blocker_name,
                                           process_execution_id, event_type_before, issue_id, work_order_id)
            else:
                pass
                # print("Warning: Fließband workaround")
                # ToDo: used to avoid the shifting of conveyor belt plannings because a slot inbetween is needed

        successful = True
        clashing_blocker_names = [None]
        clashing_process_execution_ids = [None]
        return successful, clashing_blocker_names, clashing_process_execution_ids

    def _get_row_index(self, start_time, end_time):
        row_index = 0
        if self._time_schedule.size != 0:
            index_ = np.where(self._time_schedule["End"] <= end_time)[0]
            if index_.size:
                row_index = index_[-1]
                if start_time >= self._time_schedule["Start"][row_index]:
                    row_index += 1

        # row_index -= 1
        row_index = max(0, row_index)
        return row_index

    def _block_by_row(self, row_index, start_time, end_time, blocker_name, process_execution_id, event_type,
                      issue_id, work_order_id, time_slot_duration=None):
        """Blck time slot (insert row by row_index)"""
        if end_time < start_time:
            raise Exception("Start time should be before end time")

        if time_slot_duration is None:
            time_slot_duration = end_time - start_time

        time_slot_duration_seconds = time_slot_duration / one_second
        new_entry = np.array([(start_time,  # "Start"
                               end_time,  # "End"
                               time_slot_duration_seconds,  # "Duration"
                               blocker_name,  # "Blocker Name"
                               process_execution_id,  # "Process Execution ID"
                               event_type,  # "Event Type"
                               issue_id,  # "Issue ID"
                               work_order_id)],  # "Work Order ID"
                             dtype=type(self).time_schedule_data_type)

        self._time_schedule = np.insert(self._time_schedule, row_index, new_entry)

    def update_period(self, start_time, end_time, process_execution_id: int):
        if process_execution_id not in self._time_schedule["Process Execution ID"]:
            return  # Case: cannot be updated

        process_execution_mask = self._time_schedule["Process Execution ID"] == process_execution_id
        series_to_adapt_ = self._time_schedule[process_execution_mask]
        if series_to_adapt_.size > 0:
            series_to_adapt = series_to_adapt_[0]
        else:
            print("Exception:", self.name, process_execution_id)
            raise Exception(process_execution_id)

        time_slot_duration = (end_time - start_time).seconds
        clashing_blocker_names, clashing_process_execution_ids = (
            self._check_clash(start_time=np.datetime64(start_time, "ns"),
                              end_time=np.datetime64(end_time, "ns"),
                              issue_id=series_to_adapt_["Issue ID"][0],
                              time_slot_duration=np.timedelta64(time_slot_duration, "ns")))
        if clashing_process_execution_ids[0] is not None:
            successful = False
            print("Not successful")
            return successful

        series_to_adapt["Start"] = start_time
        series_to_adapt["End"] = end_time
        series_to_adapt["Process Execution ID"] = process_execution_id
        series_to_adapt["Event Type"] = "PLAN"  # Can an actual process_execution be updated?

        # check if the sequence should be adjusted
        changed_entry_index = np.where(self._time_schedule["Process Execution ID"] == process_execution_id)[0][0]
        entries_afterwards_planned_before = (self._time_schedule["Start"][changed_entry_index] >
                                             self._time_schedule["Start"][changed_entry_index + 1:])
        if entries_afterwards_planned_before.any():  # ToDo: maybe logical not suitable
            process_execution_id_planned_before = entries_afterwards_planned_before["Process Execution ID"][-1]
            process_execution_planned_before_index = (
                np.where(self._time_schedule["Process Execution ID"] == process_execution_id_planned_before))[0][0]

            # delete from old position
            delete_block_before_mask = self._time_schedule["Process Execution ID"] != process_execution_id
            self._time_schedule = self._time_schedule[delete_block_before_mask]

            # set to new position
            self._time_schedule = np.insert(self._time_schedule, process_execution_planned_before_index - 1,
                                            series_to_adapt)

        successful = True
        return successful

    def update_period_by_actual(self, start_time, end_time, process_execution_id: int, plan_process_execution_id: int):
        """Update the time_slot scheduled with an actual time slot"""
        if process_execution_id in self._time_schedule["Process Execution ID"]:
            return  # Case: ACTUAL already stored
        elif plan_process_execution_id not in self._time_schedule["Process Execution ID"]:
            # Can be a transfer process (timedelta rounded up)
            return

        plan_process_execution_mask = self._time_schedule["Process Execution ID"] == plan_process_execution_id
        series_to_adapt_ = self._time_schedule[plan_process_execution_mask]
        if series_to_adapt_.size > 0:
            series_to_adapt = series_to_adapt_[0]
        else:
            print("Exception:", self.name, plan_process_execution_id)
            raise Exception(plan_process_execution_id)
        series_to_adapt["Start"] = start_time
        series_to_adapt["End"] = end_time
        series_to_adapt["Process Execution ID"] = process_execution_id
        series_to_adapt["Event Type"] = "ACTUAL"

        self._time_schedule[self._time_schedule["Process Execution ID"] == plan_process_execution_id] = series_to_adapt

        # ToDo: sort again ...

        # delete block_before
        delete_block_before_mask = self._time_schedule["Process Execution ID"] != plan_process_execution_id
        self._time_schedule = self._time_schedule[delete_block_before_mask]

    def get_time_slots_from_issue(self, issue_id: int) -> np.array:
        issue_time_rows = self._time_schedule[self._time_schedule["Issue ID"] == issue_id]
        return issue_time_rows

    def unblock_period(self, unblocker_name, process_execution_id) -> bool:
        """
        Blocks the period between start_time and end_time if possible for the process_execution_id
        (can only be unblocked by the blocker)

        Parameters
        ----------
        unblocker_name: (agent) name who unblocks the period
        process_execution_id: the id of the process_execution that should be executed in that time

        Returns
        -------
        successful: if the unblocking was successful True else False
        """

        # assumption: one process_execution can only block one time_slot in a calendar
        # (self._time_schedule["Blocker Name"] == unblocker_name) &
        unblock_mask = self._time_schedule["Process Execution ID"] == process_execution_id
        # (self._time_schedule["Blocker Name"] == unblocker_name) &
        if unblock_mask.any():
            self._time_schedule = self._time_schedule[~unblock_mask]

            successful = True
        else:

            print("Unblock not successful", self._time_schedule, process_execution_id, self.name)
            successful = False

        return successful

    def get_time_slot(self, blocker_name, process_execution_id, issue_id):
        """Get a specific time_slot matching to the blocker_name and the process_execution_id"""
        time_slot_mask = ((self._time_schedule["Blocker Name"] == blocker_name) &
                          (self._time_schedule["Process Execution ID"] == process_execution_id))

        if time_slot_mask.any():
            time_slot_extract = self._time_schedule[time_slot_mask]

            time_slot = time_slot_extract[["Start", "End"]]
            successful = True
            time_slot = np.array([[time_slot[0][0], time_slot[-1][1]]])

        else:
            successful = False
            time_slot = None

        return successful, time_slot

    def _check_clash(self, start_time, end_time, issue_id, time_slot_duration) -> [list[str], list[int]]:
        """
        Checks if the period between start_time and end_time is already blocked for another process_execution.

        Parameters
        ----------
        start_time: start time of the period to block_period
        end_time: end time of the period to block_period
        issue_id: time_slot blocking with the same issue_id are not considered as clash

        Returns
        -------
        clashing_blocker_names: list of names, which are responsible for the blocking between the start_time
        and end_time
        clashing_process_execution_ids: list of process_execution_ids, which are responsible for
        the blocking between the start_time and end_time
        """
        time_slot_duration_seconds = time_slot_duration / one_second

        clashing_process_executions = \
            self._get_clashing_process_executions(start_time=start_time, end_time=end_time, issue_id=issue_id,
                                                  time_slot_duration=time_slot_duration_seconds)

        # checking interference with work_calendar
        if self.work_calendar:
            # self.work_calendar.check_clash(start_time=start_time,
            #                                end_time=end_time)  # ToDo include the work_calendar
            pass

        if clashing_process_executions.size != 0:
            clashing_blocker_names = clashing_process_executions["Blocker Name"].tolist()
            clashing_process_execution_ids = clashing_process_executions["Process Execution ID"].tolist()

        else:
            clashing_blocker_names = [None]
            clashing_process_execution_ids = [None]

        return clashing_blocker_names, clashing_process_execution_ids

    def get_copy(self):
        """Crate a copy of the object self (used for planning objectives)"""
        process_executions_plan_copy = copy(self)
        # in use therefore it should have a different object id to avoid overwriting the original object
        process_executions_plan_copy._time_schedule = self._time_schedule

        return process_executions_plan_copy

    def get_utilization(self, start_time: np.datetime64, end_time: np.datetime64):
        """Calculate the utilization in the time_period between start and end time"""

        whole_time_delta = end_time - start_time
        blocked_time_periods = \
            self._get_blocked_periods_calendar_extract(start_time, end_time,
                                                       time_slot_duration=np.timedelta64(1, "s"))  # not relevant
        blocked_time_periods[blocked_time_periods["Start"] < start_time] = start_time
        blocked_time_periods[blocked_time_periods["End"] > end_time] = end_time

        blocked_time_periods = \
            blocked_time_periods[blocked_time_periods["End"] - blocked_time_periods["Start"] > np.timedelta64(0)]
        blocked_time_delta = (blocked_time_periods["End"] - blocked_time_periods["Start"]).sum(axis=0)

        blocked_seconds = (blocked_time_delta / one_second) + 1e-9
        whole_seconds = (whole_time_delta / one_second) + 1e-9

        utilization = blocked_seconds / whole_seconds

        return utilization

    def get_clashing_process_executions(self, start_time: np.datetime64, end_time: np.datetime64):
        clashing_mask = ((start_time < self._time_schedule["End"]) &
                         (self._time_schedule["Start"] < end_time))
        time_schedule_filtered = self._time_schedule[clashing_mask]
        process_execution_ids = time_schedule_filtered["Process Execution ID"].tolist()
        return process_execution_ids

    def get_events_of_type(self, event_type: str = "PLAN"):

        event_type_mask = (event_type == self._time_schedule["Event Type"])
        time_schedule_filtered = self._time_schedule[event_type_mask]
        process_execution_ids = time_schedule_filtered["Process Execution ID"].tolist()
        return process_execution_ids

    def get_last_time_stamp(self) -> Optional[np.datetime64]:
        """How to handle the Planner which consider a wider time range"""

        time_schedule = self._time_schedule[self._time_schedule["Blocker Name"] != "Planner"]

        if time_schedule.size != 0:
            last_time_stamp = time_schedule["End"][-1]
            return last_time_stamp
        else:
            return None

    def __str__(self):
        return (f"ProcessExecutionsPlan with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.work_calendar}', '{self.time_now}', '{self._time_schedule}'")

    def dict_serialize(self,
                       deactivate_id_filter: bool = False,
                       use_reference: bool = False,
                       drop_before_serialization: dict[str, list[str]] = None,
                       further_serializable: dict[str, list[str]] = None,
                       reference_type: str = "identification") -> Union[dict | str]:
        time_schedule = self._time_schedule.copy()

        not_to_process_columns = ["Duration", "Blocker Name", "Process Execution ID", "Event Type", "Issue ID",
                                  "Work Order ID"]
        time_columns = ["Start", "End"]
        if time_schedule.size:
            time_schedule_serialized = \
                {name: time_schedule[name].tolist() if name in not_to_process_columns else []
                 for name in time_schedule.dtype.names}

            for name in time_columns:
                time_stamp_array = time_schedule[name].astype('int64')
                time_stamp_array[time_stamp_array != time_stamp_array] = -1
                time_stamp_array = time_stamp_array
                time_schedule_serialized[name] = time_stamp_array.tolist()

        else:
            time_schedule_serialized = {name: []
                                        for name in time_schedule.dtype.names}

        self._time_schedule = time_schedule_serialized
        process_execution_plan_serialized = super(ProcessExecutionPlan, self).dict_serialize(deactivate_id_filter,
                                                                                             use_reference,
                                                                                             drop_before_serialization,
                                                                                             further_serializable,
                                                                                             reference_type)
        self._time_schedule = time_schedule
        return process_execution_plan_serialized


class ProcessExecutionPlanConveyorBelt(ProcessExecutionPlan):
    """
    The process_execution_plan for the conveyor belt has a slight different scheduling behavior,
    which means that only the transitions are scheduled, etc.
    """

    def __init__(self, name: str,
                 work_calendar: Optional[WorkCalender] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        super(ProcessExecutionPlanConveyorBelt, self).__init__(name=name,
                                                               work_calendar=work_calendar,
                                                               identification=identification,
                                                               external_identifications=external_identifications,
                                                               domain_specific_attributes=domain_specific_attributes)

        self._transfer_process_execution_mapper = {}
        self._transport_process_execution_mapper = {}
        self.time_interval = None

    def set_time_interval(self, time_interval):
        """called from the ConveyorBelt itself"""

        self.time_interval: int = time_interval

    def copy(self):
        return super(ProcessExecutionPlanConveyorBelt, self).copy()

    def block_period(self, start_time: np.datetime64, end_time: np.datetime64, blocker_name: str,
                     process_execution_id: int, work_order_id: int, event_type: str = "PLAN",
                     issue_id: Optional[int] = None, block_before: bool = False) -> (
            [bool, list[Optional[str]], list[Optional[int]]]):
        """Block a period in the _process_execution_plan
        Different from the normal timeslot blocking only the start/ entering time is considered
        Therefore a pitch between the entering of the conveyor_belt is ensured
        """

        time_slot_duration = end_time - start_time

        if not isinstance(time_slot_duration, pd.Timedelta):
            time_slot_duration = time_slot_duration.astype('timedelta64[s]').item()
        if time_slot_duration.total_seconds() == 0:  # Transfer process
            self._transfer_process_execution_mapper[process_execution_id] = start_time

            return True, [None], [None]

        self._transport_process_execution_mapper[issue_id] = (end_time, start_time)
        end_time = start_time
        block_before = False

        return super().block_period(start_time=start_time, end_time=end_time, issue_id=issue_id,
                                    blocker_name=blocker_name,
                                    process_execution_id=process_execution_id, work_order_id=work_order_id,
                                    block_before=block_before)

    def _get_clashing_process_executions(self, start_time, end_time, time_slot_duration=None, issue_id=None,
                                         type_="ALL") -> np.array:
        """
        Get clashing_process_execution_df based on the time between start_time and end_time.

        Parameters
        ----------
        start_time: datetime, start of a time period
        end_time: datetime, end of a time period
        issue_id: time_slot blocking with the same issue_id are not considered as clash

        Returns
        -------
        clashing_process_executions_df: a dataframe with clashing process_executions
        """

        if self._time_schedule.size == 0:
            return self._time_schedule
        timedelta_ = np.timedelta64(int(round(self.time_interval, 0) * 1e9), "ns")

        mask = ((self._time_schedule["Start"] > start_time - timedelta_) &
                (self._time_schedule["Start"] < end_time + timedelta_))  # a different mask
        clashing_process_executions_df = self._time_schedule[mask]

        # assuming 0 == "Start" and 1 == "End"
        clashing_process_executions_df["Start"] = clashing_process_executions_df["Start"] - timedelta_
        clashing_process_executions_df["End"] = clashing_process_executions_df["End"] + timedelta_

        if type_ == "ACTUAL" or type_ == "PLAN":
            clashing_process_executions_df = \
                clashing_process_executions_df[clashing_process_executions_df["Event Type"] == type_]

        return clashing_process_executions_df

    def get_free_time_from(self, start_time, issue_id=None) -> np.array:
        """Get free time period from the time_stamp until the next planned slot"""

        if start_time is None:
            start_time = np.datetime64(datetime.now(), "ns")

        timedelta_ = np.timedelta64(int(round(self.time_interval, 0) * 1e9), "ns")
        mask = (self._time_schedule["Start"] > start_time - timedelta_) | \
               (self._time_schedule["End"] > start_time - timedelta_)  # ToDo: > or >=
        time_slots_after_start_time = self._time_schedule[mask]

        if time_slots_after_start_time.shape[0]:
            next_time_slot = time_slots_after_start_time[0]
            if next_time_slot["Start"] == start_time:  # ToDo: <= or ==
                if next_time_slot["Issue ID"] != issue_id:  # same issue means that th
                    return np.array([])  # empty array, because the start time is blocked
                end_time = np.datetime64("2100-01-01 00:00:00", "ns")

            else:
                end_time = time_slots_after_start_time[0]["Start"]
        else:
            end_time = np.datetime64("2100-01-01 00:00:00", "ns")

        free_time_from = np.array([start_time, end_time])
        return free_time_from

    def update_period_by_actual(self, start_time, end_time, process_execution_id: int, plan_process_execution_id: int):
        """Update the time_slot scheduled with an actual time slot"""

        if end_time == start_time:  # Transfer process
            return

        end_time = start_time
        super().update_period_by_actual(start_time=start_time, end_time=end_time,
                                        process_execution_id=process_execution_id,
                                        plan_process_execution_id=plan_process_execution_id)

    def get_last_time_stamp(self) -> Optional[np.datetime64]:

        time_schedule = self._time_schedule[self._time_schedule["Blocker Name"] != "Planner"]
        if time_schedule.shape[0]:
            timedelta_ = np.timedelta64(int(round(self.time_interval, 0) * 1e9), "ns")
            last_time_stamp = time_schedule["End"][-1] + timedelta_
            return last_time_stamp
        else:
            return None

    def get_time_slot(self, blocker_name, process_execution_id, issue_id):
        """Get a specific time_slot matching to the blocker_name and the process_execution_id"""

        mask = ((self._time_schedule["Blocker Name"] == blocker_name) &
                (self._time_schedule["Process Execution ID"] == process_execution_id))
        if mask.any():
            time_slot_extract = self._time_schedule[mask]

            time_slot = time_slot_extract[["Start", "End"]]
            successful = True
            time_slot = np.array([[time_slot[0][0], time_slot[-1][1]]])
            return successful, time_slot
        try:
            time_stamp = np.datetime64(self._transfer_process_execution_mapper[process_execution_id])
        except KeyError:
            raise Exception(self._time_schedule)

        if time_stamp:
            time_slot = np.array([[time_stamp, time_stamp]])
            successful = True

        else:
            time_slot = None
            successful = False

        return successful, time_slot

    def get_next_possible_start_time(self, last_time_stamp):

        possibly_blocking_time_slots = \
            self._time_schedule[self._time_schedule["Start"] > last_time_stamp -
                                np.timedelta64(int(round(self.time_interval, 0) * 1e9), "ns")].copy()
        if possibly_blocking_time_slots.empty:
            return last_time_stamp

        differences = possibly_blocking_time_slots["Start"][1:] - possibly_blocking_time_slots["Start"][:-2]
        interim_time_slots = differences > np.timedelta64((2 * int(round(self.time_interval, 0)) * 1e9), "ns")
        if not interim_time_slots.any():
            return self.get_last_time_stamp()

        raise NotImplementedError

    def __str__(self):
        return (f"ProcessExecutionsPlanConveyorBelt with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.work_calendar}', '{self.time_now}', '{self._time_schedule}'")

    def dict_serialize(self,
                       deactivate_id_filter: bool = False,
                       use_reference: bool = False,
                       drop_before_serialization: dict[str, list[str]] = None,
                       further_serializable: dict[str, list[str]] = None,
                       reference_type: str = "identification") -> Union[dict | str]:
        self._transfer_process_execution_mapper = {}  # value is a timestamp
        self._transport_process_execution_mapper = {}  # value is a tuple of timestamps

        return super().dict_serialize(deactivate_id_filter,
                                      use_reference,
                                      drop_before_serialization,
                                      further_serializable,
                                      reference_type)
