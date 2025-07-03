"""
A transport scheduler that is used next to the central scheduler to arise the simulation performance.
It is a central unit too and managed through an inherited participant behaviour.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import random
from copy import copy
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.planning.process import \
    _determine_resource_demand_process_execution
from ofact.twin.agent_control.behaviours.planning.tree.preference import _get_merged_time_periods_vertical
from ofact.twin.agent_control.helpers.communication_objects import ListCO
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.env.helper import np_datetime64_to_datetime
from ofact.twin.state_model.entities import NonStationaryResource
from ofact.twin.state_model.processes import ProcessExecution

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Resource
    from ofact.twin.state_model.time import ProcessExecutionPlan


class TransportScheduler:

    def __init__(self):
        """
        process_executions_to_forward: a list of process_executions_components that should be forwarded to another agent
        """
        super(TransportScheduler, self).__init__()

        self.agent = None
        self.behaviour = None

        self.resources = {}
        self.process_executions = []
        self.transport_requests = {}
        self.issue_id = 0

    def initial_setting(self, agent, behaviour):
        self.agent = agent
        self.behaviour = behaviour

        resources = [resource
                     for resource_entity_type, resources in self.agent.resources.items()
                     for resource in resources]

        self.resources = {}
        for resource in resources:
            self.resources.setdefault(resource.entity_type,
                                      []).append(resource)
            if resource.entity_type.super_entity_type:
                self.resources.setdefault(resource.entity_type.super_entity_type,
                                          []).append(resource)

    def add_transport_request_to_schedule(self, msg_content, msg_sender):
        """Add a transport request to be scheduled"""

        call_for_proposal_id, transport_process_executions, support_resource, start_time_begin_64 = msg_content
        # print("ADDED", len(transport_process_executions))

        self.transport_requests.setdefault(start_time_begin_64,
                                           []).append((call_for_proposal_id, transport_process_executions,
                                                       support_resource, msg_sender))

    async def schedule(self):  # triggered by the agent
        # schedule when all requests available
        if not self.transport_requests:
            return

        print(get_debug_str(self.agent.name, "TransportScheduler") + " Start transports scheduling")
        process_executions_plan_copies: dict[Resource, ProcessExecutionPlan] = {}

        sorted_start_times = self.get_sorted_times()
        for start_time_stamp in sorted_start_times:
            for call_for_proposal_id, transport_process_executions, support_resource, requester in \
                    self.transport_requests[start_time_stamp]:

                process_executions_plan_copies, transport_process_executions = \
                    self._schedule_process_executions_path(start_time_stamp, transport_process_executions,
                                                           support_resource, process_executions_plan_copies)

                scheduled_process_executions = transport_process_executions

                await self.send_response(call_for_proposal_id, scheduled_process_executions, requester)
                self.process_executions.extend(scheduled_process_executions)

        self.transport_requests = {}
        # print("End transports scheduling")

    def get_sorted_times(self):
        sorted_start_times = sorted(self.transport_requests.keys())
        return sorted_start_times

    def _schedule_process_executions_path(self, start_time_stamp, transport_process_executions, support_resource,
                                          process_executions_plan_copies):
        # print("Scheduling started")
        # find the right resource_model
        self.issue_id -= 1
        start_time_stamp_pe = copy(start_time_stamp)
        for process_execution in transport_process_executions:
            process_execution: ProcessExecution
            resources_available = self._get_resources_to_schedule(process_execution, support_resource)

            process_executions_plans_batch = []
            for resource in resources_available:
                if resource not in process_executions_plan_copies:
                    process_executions_plan_copies[resource] = resource.get_process_execution_plan_copy()

                if process_executions_plan_copies[resource] not in process_executions_plans_batch:
                    process_executions_plans_batch.append(process_executions_plan_copies[resource])

            process_execution.resources_used = [(resource,) for resource in resources_available]

            process_execution.main_resource = process_execution.get_main_resource_from_resources()

            expected_process_execution_time = process_execution.get_expected_process_lead_time()
            expected_process_execution_time64 = np.timedelta64(int(expected_process_execution_time), "s")

            free_time_periods = \
                [process_executions_plan.get_free_periods_calendar_extract(
                    start_time=start_time_stamp_pe, end_time=start_time_stamp_pe + np.timedelta64(1, "D"),
                    issue_id=self.issue_id,
                    time_slot_duration=expected_process_execution_time64).astype("datetime64[ns]")
                    for process_executions_plan in process_executions_plans_batch]

            possible_time_periods = _get_merged_time_periods_vertical(free_time_periods,
                                                                      data_type="datetime64[ns]")
            time_slot_start_time_stamp = possible_time_periods[0][0]
            time_slot_end_time_stamp = time_slot_start_time_stamp + expected_process_execution_time64

            order_id = process_execution.order.identification
            schedules_blocked = []
            for resource in resources_available:
                if isinstance(resource, NonStationaryResource):
                    block_before = True
                else:
                    block_before = False

                if process_executions_plan_copies[resource] in schedules_blocked:
                    continue

                # issue_ids_before = \
                #     process_executions_plan_copies[resource]._time_schedule.loc[
                #         process_executions_plan_copies[resource]._time_schedule["Work Order ID"] == order_id,
                #         "Issue ID"]
                # if issue_ids_before.empty:
                issue_id = self.issue_id
                # else:
                #     issue_id = issue_ids_before.iloc[-1]

                successful, clashing_blocker_names, clashing_process_execution_ids = \
                    process_executions_plan_copies[resource].block_period(
                        start_time=time_slot_start_time_stamp,
                        end_time=time_slot_end_time_stamp,
                        blocker_name="TransportScheduler",
                        process_execution_id=process_execution.identification,
                        work_order_id=order_id,
                        issue_id=issue_id,
                        block_before=block_before)

                if not successful:
                    print(process_executions_plan_copies[resource]._time_schedule, time_slot_start_time_stamp,
                          time_slot_end_time_stamp, process_execution.identification, order_id, issue_id, block_before,
                          type(process_executions_plan_copies[resource]), resource.name)
                    raise Exception

                if successful:
                    schedules_blocked.append(process_executions_plan_copies[resource])

            process_execution.executed_start_time = np_datetime64_to_datetime(time_slot_start_time_stamp)
            process_execution.executed_end_time = np_datetime64_to_datetime(time_slot_end_time_stamp)
            start_time_stamp_pe = time_slot_end_time_stamp
        # print("Scheduling finished")
        return process_executions_plan_copies, transport_process_executions

    def _get_resources_to_schedule(self, process_execution, support_resource):
        resources_participated = process_execution.get_resources()
        resource_model_complete, resource_model_demands = \
            _determine_resource_demand_process_execution(process_execution)

        resource_model, resource_entity_types_needed = resource_model_demands.popitem()

        if support_resource.entity_type in resource_model.resources:
            resources_participated.append(support_resource)
            resources_participated = list(set(resources_participated))
        elif support_resource.entity_type.super_entity_type in resource_entity_types_needed:
            resources_participated.append(support_resource)
            resources_participated = list(set(resources_participated))

        for resource in resources_participated:
            if resource.entity_type in resource_entity_types_needed:
                resource_entity_types_needed.remove(resource.entity_type)
            elif resource.entity_type.super_entity_type in resource_entity_types_needed:
                resource_entity_types_needed.remove(resource.entity_type)

        resources_available = self._get_resources_for_entity_types(resource_entity_types_needed)

        resources_available += resources_participated

        return resources_available

    def _get_resources_for_entity_types(self, resource_entity_types_needed):
        """Get a resource for each required entity type"""
        resources_available = [random.choice(self.resources[resource_entity_type])
                               for resource_entity_type in resource_entity_types_needed]

        return resources_available

    # async def send_process_executions_to_resource_agents(self, process_executions_of_other_agents):
    #     for agent_name, process_executions in process_executions_of_other_agents.items():
    #         await self.agent.send_msg(behaviour=self, receiver_list=[agent_name], msg_body=process_executions,
    #                                   message_metadata={"performative": "inform",
    #                                                     "ontology": "PROCESS_EXECUTIONS",
    #                                                     "language": "OWL-S"})

    async def send_response(self, call_for_proposal_id, transport_process_executions, requester):

        process_executions_co = ListCO(transport_process_executions)
        msg_content = process_executions_co
        self.agent.NegotiationBehaviour._set_call_for_proposal_called(call_for_proposal_id, self.agent.name)
        await self.agent.send_msg(behaviour=self.behaviour, receiver_list=[requester], msg_body=msg_content,
                                  message_metadata={"performative": "inform",
                                                    "ontology": "PROPOSAL",
                                                    "language": "OWL-S"})

    def get_results(self):
        if self.process_executions:
            process_executions, self.process_executions = self.process_executions.copy(), []
            accepted_proposals = {self.agent.name: process_executions}
        else:
            accepted_proposals = {}
        # print("ACCEPTED: ",  accepted_proposals)
        return accepted_proposals
