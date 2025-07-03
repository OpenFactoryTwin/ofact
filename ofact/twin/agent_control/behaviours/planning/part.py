"""
# ToDo: include block_before process_execution_plan
This file is used to encapsulate the plan process_execution_request behaviours
@last update: 28.11.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.negotiation.objects import ProcessCallForProposal
from ofact.twin.agent_control.behaviours.negotiation.objects import \
    ProcessExecutionsPathProposal
from ofact.twin.agent_control.behaviours.planning.planning_request import PlanningRequest
from ofact.twin.agent_control.behaviours.planning.process_group import get_process_group_information, \
    process_process_group
from ofact.twin.state_model.entities import Resource, Storage
from ofact.twin.state_model.processes import ProcessExecution
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Entity, EntityType, StationaryResource
    from ofact.twin.agent_control.behaviours.negotiation.objects import PartCallForProposal
    from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference

logger = logging.getLogger("PartRequest")


def _check_loading_process_suitability(loading_process, part_entity_type, possible_origin, origin):
    if loading_process.get_necessary_input_entity_types()[0][0].check_entity_type_match(part_entity_type) \
            and possible_origin.identification == origin.identification:
        return True
    else:
        return False


class PartRequest(PlanningRequest):

    def __init__(self):
        super(PartRequest, self).__init__()
        self.ability_to_transport = True
    async def process_request(self, negotiation_object_request: PartCallForProposal):
        """
        The request for a part is handled in several steps: (in some aspects it is similar to the resource request)
        1. determine the available parts that can match to the entity_type needed if no entity itself requested
        (long time reservation)
        2. according to the available parts, for each part a process_executions_path is organized
        - the path contains the process requested to be participated in and
          can be extended by further transport processes that bring the part to the location of demand
        :param negotiation_object_request: the negotiation_object_request
        :return: process_executions_components (/ input for the proposals)
        """

        (reference_cfp, client_object, order, preference_requester, request_type, issue_id,
         requested_entity_types, locations_of_demand, node_path) = (
            negotiation_object_request.unpack())  # transfer the object through the method
        request_id = negotiation_object_request.identification

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Client {client_object.identification} start PartRequest with ID {request_id}")

        if len(requested_entity_types) > 1:
            raise NotImplementedError
        elif requested_entity_types[0][1] != 1:
            raise Exception

        all_entities_available, process_executions_with_information, resources_available_entities = \
            self.satisfy_part_demand(requested_entity_types, locations_of_demand, order, preference_requester)

        # print("Part availability:", not all_entities_available, self.agent.name, requested_entity_types[0][0].name)

        if not all_entities_available or self.check_break_needed():  # ToDo: next  or "as_agentbase" in self.agent.name
            # early termination
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" End PartRequest not successfully with ID: {negotiation_object_request.identification}")

            return False, [], []

        # Process requirements
        process_executions_paths, proposals_to_reject = \
            await self._organize_process_requirements(
                process_executions_with_information=process_executions_with_information,
                resources_available_entities=resources_available_entities,
                preference_requester=preference_requester, cfp=negotiation_object_request, order=order,
                node_path=node_path, request_id=request_id, process_execution=client_object)

        if process_executions_paths:
            successful = True
        else:
            successful = False

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" End PartRequest with ID {request_id} {len(process_executions_paths)}, {len(proposals_to_reject)}")

        return successful, process_executions_paths, proposals_to_reject

    def check_break_needed(self):
        return False
        if "WorkStationAgent" in self.agent.__class__.__name__:
            return True
        else:
            return False

    def satisfy_part_demand(self, requested_entity_types, locations_of_demand, order, preference_requester):
        """Request/ Determine the parts from the buffers/ storage places"""

        if not self.ability_to_transport:

            if isinstance(locations_of_demand, list):
                if len(locations_of_demand) == 1:
                    location_of_demand = locations_of_demand[0]
                else:
                    raise NotImplementedError
            else:
                location_of_demand = locations_of_demand

            resource_without_transport_found = False
            for resource in self.agent._resources_without_storages:
                if resource.identification == location_of_demand.identification:
                    resource_without_transport_found = True
                    continue

            if not resource_without_transport_found:
                return False, [], []

        process_executions_with_information, lead_times = \
            self._get_process_requirements_rough(requested_entity_types=requested_entity_types,
                                                 locations_of_demand=locations_of_demand, order=order)

        all_entities_available, resources_available_entities = \
            self._get_entities_procurement_intern(requested_entity_types=requested_entity_types,
                                                  lead_times=lead_times,
                                                  preference=preference_requester, order=order)

        return all_entities_available, process_executions_with_information, resources_available_entities

    def _get_process_requirements_rough(self, requested_entity_types, locations_of_demand, order):

        process_executions_material_provision, lead_times = \
            self._derive_process_requirements(requested_entity_types=requested_entity_types,
                                              locations_of_demand=locations_of_demand, order=order)

        # ToDo: (later) more than one entity_type requested

        return process_executions_material_provision, lead_times

    def _derive_process_requirements(self, requested_entity_types, locations_of_demand, order):
        """Derive transport of the parts needed"""

        # Loading - needed for all entities if not in the same resource
        # Transport only for different position of storage and location of demand

        if locations_of_demand is None:
            loading_needed = True
        else:
            loading_needed = False

        if isinstance(locations_of_demand, list):
            if len(locations_of_demand) == 1:
                location_of_demand = locations_of_demand[0]
            else:
                raise NotImplementedError
        else:
            location_of_demand = locations_of_demand

        lead_times = {}

        loading_demands: dict[StationaryResource, list[dict]] = {}
        transport_demands: dict[StationaryResource, list[dict]] = {}
        for resource in self.agent._resources_without_storages:  # ToDo: should be preselected as dict
            if resource.identification == location_of_demand.identification:
                continue

            if not resource.check_entity_type_storable(requested_entity_types[0][0]):
                continue

            for part_entity_type, amount in requested_entity_types:

                transport_needed = False
                if loading_needed:
                    raise NotImplementedError

                else:
                    if location_of_demand.get_position() != resource.get_position():
                        transport_needed = True
                    elif (location_of_demand.situated_in != resource and
                          resource.situated_in != location_of_demand):
                        transport_needed = True

                if transport_needed:
                    # in transport the loading_process is included
                    processes_material_supply_d_lst = (
                        self._get_transport_processes(resource, location_of_demand, part_entity_type))

                    if not processes_material_supply_d_lst:
                        continue

                    process_executions_material_supply = \
                        [ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN,
                                          process=supply_process_d["process"],
                                          executed_start_time=None, executed_end_time=None,
                                          parts_involved=None, resources_used=None,
                                          main_resource=None,
                                          origin=supply_process_d["origin"],
                                          destination=supply_process_d["destination"],
                                          resulting_quality=1, order=order,
                                          source_application=self.agent.source_application)
                         for supply_process_d in processes_material_supply_d_lst]

                    transport_demand_d = get_process_group_information(process_executions_material_supply,
                                                                       part_entity_type, amount)
                    lead_time_part = sum(transport_demand_d["lead_times"])

                    lead_times[(resource, part_entity_type)] = lead_time_part
                    transport_demands.setdefault(resource,
                                                 []).append(transport_demand_d)

                # calculate the time needed between the unloading to the locations of demand and the loading process to
                # a transport resource (here the part should be available)
                # dependent on resource and entity_type
                # ToDo: calculate the lead_time (use heuristics) - max/ mean
                if loading_demands:
                    raise NotImplementedError
                if not transport_demands:
                    continue

        process_executions_material_provision = loading_demands | transport_demands

        return process_executions_material_provision, lead_times

    def _get_transport_processes(self, resource, location_of_demand, part_entity_type):
        transport_processes_d_lst: list[dict] = (
            self.agent.routing_service.get_transit_processes(
                origin=resource, destination=location_of_demand, entity_entity_type=part_entity_type))

        if not transport_processes_d_lst:
            return []

        # from origin to support and from support to destination
        support_entity_type_beginning = transport_processes_d_lst[0]["process"].get_support_entity_type()
        support_entity_type_ending = transport_processes_d_lst[-1]["process"].get_support_entity_type()
        loading_process_d: dict = self.agent.routing_service.get_transfer_process(
            origin=resource, entity_entity_type=part_entity_type,
            support_entity_type=support_entity_type_beginning,
            level_differences_allowed=True)
        unloading_process_d: dict = self.agent.routing_service.get_transfer_process(
            destination=location_of_demand, entity_entity_type=part_entity_type,
            support_entity_type=support_entity_type_ending,
            level_differences_allowed=True)

        processes_material_supply_d_lst = transport_processes_d_lst.copy()
        if transport_processes_d_lst:
            if loading_process_d["process"] != transport_processes_d_lst[0]["process"]:
                processes_material_supply_d_lst = [loading_process_d] + processes_material_supply_d_lst
            if unloading_process_d["process"] != transport_processes_d_lst[-1]["process"]:
                processes_material_supply_d_lst = processes_material_supply_d_lst + [unloading_process_d]

        return processes_material_supply_d_lst

    def _get_entities_procurement_intern(self, requested_entity_types: list[tuple[EntityType, int]], lead_times,
                                         preference, order) -> (bool, dict[Resource: list[Entity]]):
        """
        Method is used to search in the stocks of the resources to find the needed parts.
        :param requested_entity_types: a list of requested entity_types
        :return all_entities_available: True if all needed parts organized else False
        :return resources_available_entities: parts mapped to the resources
        """
        all_entities_available = True

        suitable_resources = self.agent._resources_without_storages.copy()  # already no storage?
        # Find Parts that can be used for the EntityTypes
        resources_available_entities: dict[Resource: dict[EntityType, list[Entity]]] = \
            {resource: {}
             for resource in suitable_resources}

        # it is assumed that a part that is available earlier is also available later but not vice versa
        first_time_stamp = preference.get_first_accepted_time_stamp()  # ToDo: (later) maybe time conversion needed

        for part_entity_type, demand_number in requested_entity_types:
            available_entities_entity_type = []
            for resource in suitable_resources:

                # ToDo: (later) maybe also poorer time_slots should be checked
                if (resource, part_entity_type) in lead_times:
                    first_time_stamp -= np.timedelta64(int(lead_times[(resource, part_entity_type)]), "s")

                available_entities = \
                    self.agent.storage_reservations[resource].get_unreserved_entities_at(
                        entity_type=part_entity_type, demand_number=demand_number, time_stamp=first_time_stamp,
                        round_=self.agent.agents.current_round_id)

                # if "as_agentbase" in self.agent.name:
                #     print("Part Behaviour AS:", requested_entity_types[0][0].name, [e[0].identification for e in available_entities])
                # print("Part Behaviour:", requested_entity_types[0][0].name,
                #       [e[0].identification for e in available_entities])

                # the time_stamp is currently not used
                if available_entities:
                    resources_available_entities[resource][part_entity_type] = available_entities
                    available_entities_entity_type.extend(available_entities)
                else:
                    del resources_available_entities[resource]

            if len(available_entities_entity_type) < demand_number:
                all_entities_available = False
                if "WarehouseAgent" not in self.agent.__class__.__name__:
                    continue

                current_demand = demand_number - len(available_entities_entity_type)
                # self._trigger_good_receipt(part_entity_type, first_time_stamp, current_demand)

            # print("Entity demand: ", part_entity_type.name, all_entities_available)

        return all_entities_available, resources_available_entities

    def _trigger_good_receipt(self, part_entity_type, first_time_stamp, current_demand):
        # reordering
        good_receipt_processes = \
            [process for process in self.agent.possible_processes
             if len(process.transformation_controller.get_root_nodes()) == 1  # ensure: good receipt and no loading process
             if part_entity_type in list(zip(*process.get_possible_output_entity_types()))[0]]

        if good_receipt_processes:
            good_receipt_process = good_receipt_processes[0]
            main_resource = self.agent._resources_without_storages[0]
            executed_start_time = self.agent.change_handler.get_current_time()
            executed_end_time = \
                executed_start_time + timedelta(seconds=good_receipt_process.get_estimated_process_lead_time())
            process_execution = \
                ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, process=good_receipt_process,
                                 executed_start_time=executed_start_time, executed_end_time=executed_end_time,
                                 resources_used=[(main_resource,)],
                                 main_resource=main_resource, origin=None, destination=main_resource,
                                 resulting_quality=1, order=None, source_application="MaterialSupply")

            good_receipt_process_executions = [process_execution]

            if current_demand > 1:
                further_process_executions = [process_execution.duplicate() for i in range(1, current_demand)]
                good_receipt_process_executions.extend(further_process_executions)

            self.agent.trigger_good_receipt(good_receipt_process_executions)

    async def _organize_process_requirements(self, process_executions_with_information, resources_available_entities,
                                             preference_requester, cfp, order, node_path, request_id,
                                             process_execution):
        """Handle the process requirements (means organize the material supply by specify process_executions_paths)"""
        process_execution_id = process_execution.identification

        if not process_executions_with_information:
            process_executions_paths = self._determine_process_executions_paths(
                preference_requester=preference_requester, resources_available_entities=resources_available_entities,
                cfp=cfp, node_path=node_path, process_execution_id=process_execution_id)
            proposals_to_reject = []
            return process_executions_paths, proposals_to_reject

        # ToDo: derive resource_demand
        # based on the transport demands, a preselection/ prioritization can be take place
        process_executions_with_information = \
            self._determine_possible_material_supply(process_executions_with_information,
                                                     resources_available_entities)

        current_time = np.datetime64(self.agent.change_handler.get_current_time().replace(microsecond=0))

        process_executions_paths, proposals_to_reject = \
            await process_process_group(self, process_executions_with_information, preference_requester, cfp,
                                        node_path, process_execution_id, current_time, order, request_id)

        # print("Part paths", len(process_executions_paths), node_path)
        # for path in process_executions_paths:
        #     for process_executions_component in path.get_process_executions_components_lst():
        #         print(process_executions_component.node_identification)
        # print("Part paths end. .. .")
        return process_executions_paths, proposals_to_reject

    def _determine_process_executions_paths(self, resources_available_entities,
                                            preference_requester: ProcessExecutionPreference, cfp, node_path,
                                            process_execution_id):
        """Create process_executions_paths for every part requested without transport"""

        process_executions_paths = []
        for origin_resource, part_dict in resources_available_entities.items():
            for part_entity_type, parts in part_dict.items():
                for part, time_stamp in parts:
                    preference = preference_requester.get_copy()
                    # adapt the preference based on the availability of the part
                    preference.accepted_time_periods = \
                        preference.accepted_time_periods[preference.accepted_time_periods[:, 1] > time_stamp]
                    if preference.accepted_time_periods.size:
                        preference.accepted_time_periods[0][0] = time_stamp

                    process_executions_path = \
                        ProcessExecutionsPathProposal(call_for_proposal=cfp, provider=self.agent.name,
                                                      issue_id=cfp.identification,
                                                      goal_item=part, goal=part.entity_type,
                                                      process_executions_components={}, reference_preference=preference,
                                                      cfp_path=node_path, process_execution_id=process_execution_id)

                    process_executions_paths.append(process_executions_path)

        return process_executions_paths

    def _determine_possible_material_supply(self, process_executions_material_provision,
                                            resources_available_entities):
        """Determine material supplies """
        process_executions_material_provision_updated = {}
        for resource_, parts_d in resources_available_entities.items():
            for part_entity_type, parts_with_time_stamp in parts_d.items():
                provision_lst = [provision_d for provision_d in process_executions_material_provision[resource_]
                                 if provision_d["entity_type"].check_entity_type_match(part_entity_type)]
                if len(provision_lst) > 1:
                    raise NotImplementedError
                parts, time_stamps = list(zip(*parts_with_time_stamp))
                provision_lst[0]["entities"] = parts
                for process_execution in provision_lst[0]["process_executions"]:
                    process_execution.parts_involved = list(zip(parts))

                process_executions_material_provision_updated[resource_] = provision_lst[0]
                process_executions_material_provision_updated[resource_]["pick_up_times"] = time_stamps

        return process_executions_material_provision_updated

    async def request_process_executions_paths_organization(self, process_executions_paths, reference_cfp, order,
                                                            node_path, request_id, long_time_reservation):
        """Request the material supplies"""
        cfp_paths_requested = []
        negotiation_object_identifications = []
        provider_negotiation_objects = {}
        for process_execution_path in process_executions_paths:

            last_negotiation_object = None
            for process_execution_component in process_execution_path.get_process_executions_components_lst():

                preference = \
                    process_execution_path.get_preference_process_executions_component(process_execution_component)
                process_execution = process_execution_component.goal
                if preference is None:
                    preference = process_execution_path.reference_preference
                negotiation_object = \
                    ProcessCallForProposal(reference_cfp=reference_cfp, predecessor_cfp=last_negotiation_object,
                                           sender_name=self.agent.name,
                                           client_object=process_execution_path.goal_item,
                                           order=order, issue_id=process_execution_path.issue_id,
                                           process_execution=process_execution, preference=preference,
                                           node_path=node_path, long_time_reservation={})

                # determine the providers
                if process_execution.main_resource is not None:
                    providers = [process_execution.main_resource]  # support entity_type
                else:
                    providers = self.agent.NegotiationBehaviour.convert_providers(
                        process_execution.get_possible_main_resource_entity_types())

                last_negotiation_object = negotiation_object
                for provider in self.agent.NegotiationBehaviour.convert_providers(providers):
                    provider_negotiation_objects.setdefault(provider, []).append(negotiation_object)

                negotiation_object_identifications.append(negotiation_object.identification)
                cfp_paths_requested.append(node_path + [negotiation_object.identification])

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" process Requested ID: {request_id}")
        for provider, negotiation_objects in provider_negotiation_objects.items():
            await self.agent.NegotiationBehaviour.call_for_proposal(negotiation_objects, [provider])

        if negotiation_object_identifications:
            proposals = await self.agent.NegotiationBehaviour.await_callback(negotiation_object_identifications,
                                                                             "part")
        else:
            proposals = []

        return proposals, cfp_paths_requested


class PartRequestIntern(PartRequest):

    def __init__(self):
        super(PartRequestIntern, self).__init__()
        self.ability_to_transport = False
