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

The environment class can be inherited by a simulation or a physical world.
While the simulation is currently the event discrete simulation,
the physical world is depicted by the data transformation management (data integration).
The environment returns actions to the change handler
and can also receive actions from the agents also through the change handler.

Classes:
    Environment: the environment class

@last update: 26.05.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Union

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.env.planned_event_queue import PlannedEventsQueue

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Order
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation, ChangeHandlerPhysicalWorld


class Environment:
    """Simulation and data transformation/ integration inherit from the class"""

    def __init__(self, change_handler: Union[ChangeHandlerSimulation, ChangeHandlerPhysicalWorld], start_time):
        """
        The environment contains describes the behavior of the shop floor.

        Parameters
        ----------
        change_handler: the interface to the digital twin and the agent control
        start_time: the time, the environment is set up
        """

        self.change_handler: ChangeHandlerSimulation | ChangeHandlerPhysicalWorld = change_handler
        self._planned_events_queue = PlannedEventsQueue()
        self._current_time: datetime = start_time
        self.frequency_without_execution = 0
        self._time_limit = None

    def get_current_time(self) -> datetime:
        """The environment updates the current_time"""
        return self._current_time

    def get_current_time64(self):
        """The environment updates the current_time"""
        return np.datetime64(self.get_current_time().replace(microsecond=0), "s")

    def set_current_time(self, new_current_time: datetime):
        self._current_time = new_current_time

    def get_frequency_without_execution(self):
        return self.frequency_without_execution

    def add_change_handler(self, change_handler):
        self.change_handler = change_handler

    def planned_events_queue_empty(self):
        return self._planned_events_queue.empty()

    def set_time_limit(self, time_limit):
        self._time_limit = time_limit

    def plan_order_release(self, order: Order):
        """
        Store a planned order in the planned events queue to release them later

        Parameters
        ----------
        order: the planned order
        """
        self._planned_events_queue.store_event(order)

    def execute_process_execution(self, process_execution: ProcessExecution,
                                  deviation_tolerance_time_delta: Optional[timedelta] = None,
                                  notification_time_delta: Optional[timedelta] = None):
        """
        Store a planned process_execution in the planned events queue to execute the process later

        Parameters
        ----------
        process_execution: the planned process_execution that should be executed
        deviation_tolerance_time_delta: if set, the environment is permitted to handle deviations of
        process_execution from the planned executed_start_time within the time_delta
        notification_time_delta: if set, a notification is sent,
        "notification_duration_before_completion" (or less) before the process_execution will be completed.
        """
        self._planned_events_queue.store_event(process_execution)

    def _create_actual_process_execution(self, plan_process_execution: ProcessExecution, source_application,
                                         executed_start_time=None, time_execution=False, from_plan=False,
                                         end_time=False, enforce_time_specification=True):
        """Create an actual process_execution based on the planned one"""
        actual_process_execution = \
            plan_process_execution.create_actual(source_application=source_application,
                                                 executed_start_time=executed_start_time,
                                                 time_specification=time_execution,
                                                 from_plan=from_plan, end_time=end_time,
                                                 enforce_time_specification=enforce_time_specification)

        return actual_process_execution
