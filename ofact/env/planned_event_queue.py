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
    EventsQueue: base class for event queues
    PlannedEventsQueue: Manages the planned events in a queue

Contains the planned event queue, with events to be executed in the simulation/ environment ...
@last update: 31.10.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, List

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.env.helper import np_datetime64_to_datetime
from ofact.twin.state_model.processes import ProcessExecution

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Order


def _get_time_stamp_process_execution(process_execution: ProcessExecution):
    if process_execution.check_actual():
        time_stamp_process_execution = process_execution.executed_end_time
    else:
        time_stamp_process_execution = process_execution.executed_start_time

    time_stamp_process_execution64 = np.datetime64(time_stamp_process_execution, "ns")

    return time_stamp_process_execution64


class EventsQueue:

    def __init__(self):
        self._event_time_stamps = np.array([],
                                           dtype="datetime64[ns]")  # datetime
        self._events_to_simulate = []
        self._events_deviation_tolerance_time_delta = {}

    def store_order(self, order: Order):
        time_stamp_o = np.datetime64(order.release_date_planned, "ns")

        idx = self._get_index(time_stamp_o, order, "ORDER")

        self._event_time_stamps = np.insert(self._event_time_stamps, idx, time_stamp_o)
        self._events_to_simulate.insert(idx, order)

        if order in self._events_deviation_tolerance_time_delta:
            raise Exception(order.identification, None)

        self._events_deviation_tolerance_time_delta[order] = None

    def store_process_execution(self, process_execution: ProcessExecution, time_stamp: Optional[datetime] = None,
                                type_: str = "PLAN", deviation_tolerance_time_delta: Optional[timedelta] = None):
        """
        Store the process_execution into the queue.
        Therefore, two queues are used, one for the event time stamps and another for events itself.
        """
        # print("ProcessExecution: ", process_execution.executed_start_time,
        #       process_execution.process.name, process_execution.order.identification, type_)

        if time_stamp is not None:
            time_stamp_pe = np.datetime64(time_stamp, "ns")
        else:
            time_stamp_pe = _get_time_stamp_process_execution(process_execution)

        idx = self._get_index(time_stamp_pe, process_execution, type_)

        self._event_time_stamps = np.insert(self._event_time_stamps, idx, time_stamp_pe)
        self._events_to_simulate.insert(idx, process_execution)

        if process_execution in self._events_deviation_tolerance_time_delta:
            raise Exception(process_execution.identification, deviation_tolerance_time_delta)

        self._events_deviation_tolerance_time_delta[process_execution] = deviation_tolerance_time_delta

    def _get_index(self, time_stamp, process_execution, type_):
        """Get the position index where the process_execution is stored in the time line"""
        if not self._event_time_stamps.size:
            idx = 0
            return idx

        if time_stamp < self._event_time_stamps[0]:
            idx = 0

        elif time_stamp > self._event_time_stamps[-1]:
            idx = self._event_time_stamps.shape[0]

        else:
            if type_ == "ACTUAL":
                indexes = np.where(self._event_time_stamps >= time_stamp)
                indexes_a = indexes[0]
                if indexes_a.any():
                    idx = int(indexes_a[0])
                else:
                    idx = 0

                indexes_same_time_stamp = np.where(self._event_time_stamps == time_stamp)

                process_execution_plan = process_execution.get_plan_process_execution()
                for index_same_timestamp in indexes_same_time_stamp[0]:
                    pe_plan = self._events_to_simulate[index_same_timestamp].get_plan_process_execution()

                    if (pe_plan.get_process_lead_time() == 0 and
                            pe_plan.order == process_execution_plan.order):
                        if pe_plan.destination == process_execution_plan.origin:  # in other cases a bad approach?
                            idx = int(index_same_timestamp) + 1
                            break

            elif type_ == "PLAN":
                indexes = np.where(self._event_time_stamps <= time_stamp)
                idx = int(indexes[0][-1]) + 1

            elif type_ == "ORDER":

                indexes = np.where(self._event_time_stamps >= time_stamp)
                indexes_a = indexes[0]
                if indexes_a.any():
                    idx = int(indexes_a[0])
                else:
                    idx = 0

            else:
                raise Exception

        return idx

    def pop_next_event(self, time_stamp=None) -> [Optional[ProcessExecution], Optional[datetime], Optional[timedelta]]:
        """
        Returns
        -------
        the next event in the queue
        """

        if not self._events_to_simulate:
            return None, None, None

        if time_stamp is None:
            time_stamp64 = self._event_time_stamps[0]
        else:
            time_stamp64 = np.datetime64(time_stamp, "ns")

        time_stamps64 = self._event_time_stamps[self._event_time_stamps == time_stamp64]
        if time_stamps64.size == 0:
            # print(f"Time stamp {time_stamp} not found \n "
            #       f"{self._event_time_stamps}")
            return None, time_stamp64, None

        self._event_time_stamps = self._event_time_stamps[1:]

        time_stamp = np_datetime64_to_datetime(time_stamp64)  # datetime ...

        event_object = self._events_to_simulate.pop(0)

        deviation_tolerance_time_delta = self._events_deviation_tolerance_time_delta.pop(event_object)

        return event_object, time_stamp, deviation_tolerance_time_delta

    def remove_event(self, event):
        """Remove the event from the queue(s)"""

        if not self._events_to_simulate:
            raise Exception(event)

        for index, stored_event in enumerate(self._events_to_simulate):
            if stored_event == event:
                break

        if self._events_to_simulate[index] != event:
            raise Exception

        del self._events_to_simulate[index]
        self._event_time_stamps = np.delete(self._event_time_stamps, index)

    def get_queue_length(self):
        return len(self._events_to_simulate)

    def empty(self):
        if self._events_to_simulate:
            return False
        else:
            return True


notification_queue_data_type = [("Notification Timestamp", "datetime64[ns]"),
                                ("ProcessExecution", object)]


class PlannedEventsQueue(EventsQueue):

    def __init__(self):
        super().__init__()

        self._notification_queue = np.array([],
                                            dtype=notification_queue_data_type)
        self._notification_queue.shape = (0, 2)

    def store_event(self, event):
        pass

    def store_process_execution(self, process_execution: ProcessExecution, time_stamp: Optional[datetime] = None,
                                type_: str = "PLAN", deviation_tolerance_time_delta: Optional[timedelta] = None,
                                notification_time_delta: Optional[timedelta] = None):
        """Store the process_execution into the queue"""

        # print("ProcessExecution: ", process_execution.executed_start_time,
        #       process_execution.process.name, process_execution.order.identification, type_)
        super().store_process_execution(process_execution=process_execution, time_stamp=time_stamp, type_=type_,
                                        deviation_tolerance_time_delta=deviation_tolerance_time_delta)

        if notification_time_delta:
            self._store_notification_request(process_execution, time_stamp, type_,
                                             notification_time_delta)

    def pop_next_event(self, time_stamp=None) -> \
            [Optional[ProcessExecution], Optional[datetime], Optional[timedelta], Optional[timedelta], List[object]]:
        """:return the next event in the queue"""
        event_object, time_stamp, deviation_tolerance_time_delta = super().pop_next_event(time_stamp)

        objects_for_notification = self._get_objects_for_notification(time_stamp)
        notification_time_delta = None  # ToDo
        return (event_object, time_stamp, deviation_tolerance_time_delta, notification_time_delta,
                objects_for_notification)

    def _store_notification_request(self, process_execution: ProcessExecution, time_stamp: Optional[datetime],
                                    type_: str = "PLAN",
                                    notification_duration_before_completion: Optional[timedelta] = None):
        """Store a notification request for the process_execution"""
        return None  # Complete implementation needed

        notification_time_stamp = process_execution.executed_start_time - notification_duration_before_completion
        new_notification_request = np.array([(np.datetime64(notification_time_stamp, "ns"), process_execution)],
                                            dtype=notification_queue_data_type)

        self._notification_queue = np.concatenate([self._notification_queue, new_notification_request], axis=0)

    def get_deviation_tolerance_time_delta(self, process_execution):
        deviation_tolerance_time_delta = self._events_deviation_tolerance_time_delta[process_execution]
        return deviation_tolerance_time_delta

    def get_notification_time_delta(self, process_execution):
        return None  # ToDo

    def _get_objects_for_notification(self, time_stamp) -> List[object]:
        notification_queue = self._notification_queue[self._notification_queue["Notification Timestamp"] <= time_stamp]
        return []
