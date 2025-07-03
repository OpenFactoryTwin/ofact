"""
# TODO Make Module Description
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
import random
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING

import numpy as np

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.env_release.process_executions_queue import (
    ProcessExecutionsOnHoldHandling)
from ofact.twin.agent_control.helpers.communication_objects import ListCO
from ofact.twin.state_model.entities import NonStationaryResource
from ofact.twin.state_model.processes import EntityTransformationNode, ProcessExecution

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ProcessExecutionsQueue")

# ToDo: Probably the behaviours must be updated to work again ...


class WorkStationProcessExecutionsOnHoldHandling(ProcessExecutionsOnHoldHandling):
    """
    Not in use anymore. Here, the decoupled material supply planning (paper) was executed.
    """

    def __init__(self):
        super(WorkStationProcessExecutionsOnHoldHandling, self).__init__()
        self.origin_resource = None
        self._material_supply_active_moving_resources = None

    def initial_setting(self):
        all_resources = [reduce(concat, list(agent.resources.values()))
                         for agent_type, agents in self.agent.agents.agents.items()
                         if agent_type.__name__ == "TransportAgent"
                         for agent in agents
                         if "individual_part" in agent.name][0]
        self._material_supply_active_moving_resources = [resource
                                                         for resource in all_resources
                                                         if isinstance(resource, NonStationaryResource)]
        self.origin_resource = set(reduce(concat, list(self.agent.entity_provider.values()))).pop()

    async def run(self):
        # get the new process_executions_components from the negotiation
        if await self.agent.NegotiationBehaviour.negotiation_round_finished():
            accepted_proposals, rejected_proposals = self.agent.NegotiationBehaviour.get_results()

            await self._save_agreed_process_executions(accepted_proposals)
            self._handle_rejected_process_executions(rejected_proposals)

            # print(get_debug_str(self.agent.name, "Queue") + f"{self.agent.process_executions_on_hold}")

            if not ((rejected_proposals or accepted_proposals) and self.agent.new_process_executions_available):
                return
            if self.agent.new_process_executions_available.is_set():
                return
            # print("new_process_executions_available", self.agent.new_process_executions_available)
            self.agent.new_process_executions_available.set()
            return

        if self.agent.reaction_expected and self.agent.NegotiationBehaviour.negotiation_round_finished_info():
            # if str(datetime.now().microsecond)[0:3] == "100":
            #     print(get_debug_str(self.agent.name, self.__class__.__name__) + " Reaction expected")

            if not self.agent.process_executions_on_hold:
                self.agent.reaction_expected = False
                await self.agent.change_handler.go_on(agent_name=self.agent.name)

    async def _save_agreed_process_executions(self, accepted_proposals):
        """save the agreed_process_executions in the process_executions_queue from the agent"""
        all_process_executions = await self._set_process_executions_on_hold(accepted_proposals)
        self._update_projection(all_process_executions)

    async def _set_process_executions_on_hold(self, accepted_proposals):

        all_process_executions = []

        process_executions_requester = []
        parts_to_reserve = []
        for requester_agent_name, process_executions in accepted_proposals.items():

            completely_filled_process_executions = [process_execution for process_execution in process_executions
                                                    if process_execution.completely_filled()[0]]
            all_process_executions += completely_filled_process_executions
            for process_execution in process_executions:
                completely_filled, _ = process_execution.completely_filled()

                if not completely_filled:
                    raise Exception("ProcessExecution", _)
                    continue  # ToDo: maybe further handling needed

                if process_execution.main_resource in self.agent._resources:
                    # update the order_queue
                    # : dict[Resource: list[tuple[str, list[ProcessExecution]]]] not up to date
                    # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" PE on Hold "
                    #       f"{process_execution.identification}")
                    parts = self.get_parts_to_transport(process_execution)
                    if parts:
                        parts_to_reserve.extend(parts)
                        transport_process_executions = self.get_transport_process_executions(parts, process_execution)

                        self.agent.process_executions_on_hold_sub.setdefault(process_execution,
                                                                             []).extend(transport_process_executions)
                    process_executions_requester.append((process_execution, requester_agent_name))

        await self.reserve_parts(parts_to_reserve)
        for process_execution, requester_agent_name in process_executions_requester:
            self.agent.process_executions_on_hold[process_execution] = requester_agent_name

        # if self.agent.new_process_executions_available is not None:
        #     if not self.agent.new_process_executions_available.done():
        #         self.agent.new_process_executions_available.set_result(True)
        return all_process_executions

    def get_parts_to_transport(self, process_execution):
        parts = []
        for part_tuple in process_execution.parts_involved:
            if len(part_tuple) == 2:
                if part_tuple[1] != EntityTransformationNode.TransformationTypes.SUB_ENTITY:
                    continue
            parts.append(part_tuple[0])

        return parts

    def get_transport_process_executions(self, parts, process_execution):
        location_of_demand = process_execution.origin
        process_executions_nested = [self.get_transport_process_executions_transport(part, location_of_demand,
                                                                                     process_execution.order)
                                     for part in parts]
        process_executions = reduce(concat, process_executions_nested)

        return process_executions

    def get_transport_process_executions_transport(self, part, location_of_demand, order):

        # available resources for this process
        current_time = self.agent.change_handler.get_current_time()

        resources_first_time_stamps = {}
        for resource in self._material_supply_active_moving_resources:
            first_time_stamp = resource.process_execution_plan.get_last_time_stamp()
            if first_time_stamp is None or first_time_stamp < current_time:
                first_time_stamp = current_time
            else:
                first_time_stamp = first_time_stamp.to_pydatetime()
            resources_first_time_stamps.setdefault(first_time_stamp, []).append(resource)
        general_first_time_stamp = min(list(resources_first_time_stamps.keys()))

        best_resources = resources_first_time_stamps[general_first_time_stamp]

        best_resource = random.choice(best_resources)
        if best_resource in self.agent.agents.resources_in_scheduling:
            process_executions_material_supply = \
                self.get_transport_process_executions_transport(part, location_of_demand, order)
            return process_executions_material_supply
        else:
            self.agent.agents.resources_in_scheduling.append(best_resource)

        process_executions_material_supply = \
            self.get_process_executions_material_supply_path(best_resource, general_first_time_stamp,
                                                             location_of_demand, part, order)

        # scheduling
        process_executions_material_supply = self.schedule_process_executions_path(process_executions_material_supply,
                                                                                   general_first_time_stamp)

        self.agent.agents.resources_in_scheduling.remove(best_resource)

        return process_executions_material_supply

    def get_process_executions_material_supply_path(self, best_resource, general_first_time_stamp, location_of_demand,
                                                    part, order):
        last_time_stamp, resource_position = best_resource.get_positions(general_first_time_stamp).popitem()

        stationary_resource = self.get_stationary_resource(best_resource, resource_position)
        transport_access_processes = \
            self.agent.routing_service.get_transit_processes(stationary_resource, self.origin_resource,
                                                             support_entity_type=best_resource.entity_type,
                                                             transport=False)
        transport_processes_d_lst: list[dict] = self.agent.routing_service.get_transit_processes(
            origin=self.origin_resource, destination=location_of_demand, entity_entity_type=part.entity_type)

        # from origin to support and from support to destination
        loading_process_d: dict = self.agent.routing_service.get_transfer_process(
            origin=self.origin_resource, entity_entity_type=part.entity_type,
            support_entity_type=transport_processes_d_lst[0]["process"].get_support_entity_type(),
            level_differences_allowed=True)
        unloading_process_d: dict = self.agent.routing_service.get_transfer_process(
            destination=location_of_demand, entity_entity_type=part.entity_type,
            support_entity_type=transport_processes_d_lst[-1]["process"].get_support_entity_type(),
            level_differences_allowed=True)

        processes_material_supply_d_lst = \
            transport_access_processes + [loading_process_d] + transport_processes_d_lst + [unloading_process_d]

        process_executions_material_supply = \
            [ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN,
                              process=supply_process_d["process"],
                              executed_start_time=None, executed_end_time=None,
                              parts_involved=None, resources_used=None,
                              main_resource=best_resource,
                              origin=supply_process_d["origin"] if supply_process_d["origin"] is not None
                              else best_resource,
                              destination=supply_process_d["destination"] if supply_process_d["destination"] is not None
                              else best_resource,
                              resulting_quality=1, order=order,
                              source_application=self.agent.source_application)
             for supply_process_d in processes_material_supply_d_lst]

        available_resources = [self.origin_resource, best_resource]

        for process_execution in process_executions_material_supply:
            resource_entity_types_needed = process_execution.get_possible_resource_entity_types()
            resources_usable = []
            individual_available_resources = available_resources.copy() + \
                                             [process_execution.origin, process_execution.destination]
            for available_resource in individual_available_resources:
                available_entity_type = available_resource.entity_type
                if available_entity_type in resource_entity_types_needed:
                    resources_usable.append((available_resource,))

            process_execution.resources_used = resources_usable

            parts_entity_types_needed = process_execution.get_part_entity_types_needed()
            if not parts_entity_types_needed:
                continue

            parts_entity_types_needed = [part_tuple[0] for part_tuple in parts_entity_types_needed]
            if part.entity_type in parts_entity_types_needed or part.entity_type.super_entity_type:
                process_execution.parts_involved = [(part,)]

        return process_executions_material_supply

    def get_stationary_resource(self, available_resource, resource_position):
        stationary_resources = \
            self.agent.digital_twin.get_stationary_resource_at_position(pos_tuple=resource_position)

        if len(stationary_resources) == 1:
            stationary_resource = stationary_resources[0]
        if len(stationary_resources) == 0:
            raise NotImplementedError("The position cannot be specified - no reference resource available")
        elif len(stationary_resources) > 1:
            stationary_resources_possible = []
            for stationary_resource in stationary_resources:
                possible_entity_types_to_store = stationary_resource.get_possible_entity_types_to_store()
                if available_resource.entity_type in possible_entity_types_to_store:
                    stationary_resources_possible.append(stationary_resource)
            if len(stationary_resources_possible) == 1:
                stationary_resource = stationary_resources_possible[0]
            else:
                raise NotImplementedError("The position cannot be specified")

        return stationary_resource

    def schedule_process_executions_path(self, process_executions_material_supply, first_time_stamp):

        first_time_stamp = np.datetime64(first_time_stamp)

        for idx, process_execution in enumerate(process_executions_material_supply):
            distance = None
            if isinstance(process_execution.origin, NonStationaryResource) or \
                    isinstance(process_execution.destination, NonStationaryResource):
                distance = 0

            expected_process_execution_time = \
                int(np.ceil(round(process_execution.get_expected_process_lead_time(distance=distance), 1)))

            resources = process_execution.get_resources()
            end_time = first_time_stamp + np.timedelta64(expected_process_execution_time, "s")

            while True:
                time_slot_failed = False
                for resource in resources:
                    free_periods_calendar_extract = \
                        resource.get_free_periods_calendar_extract(start_time=first_time_stamp, end_time=end_time,
                                                                   time_slot_duration=expected_process_execution_time)

                    if not free_periods_calendar_extract.size:
                        time_slot_failed = True
                        break

                if not time_slot_failed:
                    break

                first_time_stamp += np.timedelta64(1, "s")
                end_time += np.timedelta64(1, "s")

            process_execution.executed_start_time, process_execution.executed_end_time = \
                free_periods_calendar_extract[0]

            first_time_stamp = end_time

            for resource in process_execution.get_resources():
                if idx > 0 and isinstance(resource, NonStationaryResource):
                    block_before = True
                else:
                    block_before = False

                successful, clashing_blocker_names, clashing_process_execution_ids = \
                    resource.block_period(start_time=process_execution.executed_start_time,
                                          end_time=process_execution.executed_end_time, blocker_name=self.agent.name,
                                          process_execution_id=process_execution.identification,
                                          work_order_id=process_execution.order.identification,
                                          issue_id=None, block_before=block_before)

                # ToDo: additionally unreserve
                # ToDo: handling in the simulation

                if not successful:
                    raise Exception

        return process_executions_material_supply

    async def reserve_parts(self, parts):

        # store the negotiation_object in the agent_model
        list_communication_object = ListCO(parts)
        msg_content = list_communication_object

        await self.agent.send_msg(behaviour=self, receiver_list=[self.agent.address_book[self.origin_resource]],
                                  msg_body=msg_content, message_metadata={"performative": "reserve",
                                                                          "ontology": "PartReservation",
                                                                          "language": "OWL-S"})
