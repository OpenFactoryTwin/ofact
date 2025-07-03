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

An output of the simulation is the process executions list. They can be rolled out as a schedule.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import ActiveMovingResource

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Resource, WorkStation
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.state_model.model import StateModel


def get_order_name(order):
    if order.customer:
        return order.customer.name + str(order.identification)
    else:
        return "Unknown" + str(order.identification)


def get_resource_name(resource):
    return resource.name


def _create_schedule(process_executions_entity_mapping, entity_name_func) -> pd.DataFrame:
    schedule_input_data = \
        [(entity_name_func(real_world_entity), process_execution.process.name,
          process_execution.executed_start_time, process_execution.executed_end_time)
         for process_execution, real_world_entity in process_executions_entity_mapping]

    schedule_df = pd.DataFrame(schedule_input_data, columns=['reference', 'process name', 'start', 'end'])

    schedule_df.sort_values(['start', 'end'], ascending=[True, False], inplace=True)

    return schedule_df


def export_schedule_results(digital_twin: StateModel,
                            allowed_resource_types: Optional[
                                List[type[Resource, ActiveMovingResource, WorkStation]]] = None):
    """create schedules for order and worker"""
    actual_process_executions: list[ProcessExecution] = \
        [process_execution
         for process_execution in digital_twin.get_process_executions_list()
         if process_execution.check_actual()]

    process_execution_order_mapping = \
        [(actual_process_execution, actual_process_execution.order)
         for actual_process_execution in actual_process_executions
         if actual_process_execution.order]

    if allowed_resource_types is None:
        allowed_resource_types = [ActiveMovingResource]

    process_execution_worker_mapping = \
        [(actual_process_execution, resource)
         for actual_process_execution in actual_process_executions
         for resource in actual_process_execution.get_resources()
         if _check_resource_of_allowed_type(resource, allowed_resource_types)]

    worker_schedules = _create_schedule(process_execution_worker_mapping, get_resource_name)
    order_schedules = _create_schedule(process_execution_order_mapping, get_order_name)

    return worker_schedules, order_schedules


def _check_resource_of_allowed_type(resource,
                                    allowed_resource_types: List[Resource, ActiveMovingResource, WorkStation]):
    return any(isinstance(resource, resource_class)
               for resource_class in allowed_resource_types)


def store_schedules(worker_schedules_df, order_schedules_df, path_to=""):
    worker_schedules_df.to_excel(path_to + "worker_results.xlsx", index=False, sheet_name='Worker Schedule')
    order_schedules_df.to_excel(path_to + "order_results.xlsx", index=False, sheet_name='Order Schedule')
