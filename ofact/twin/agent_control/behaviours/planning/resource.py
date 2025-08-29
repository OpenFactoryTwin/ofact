"""Used to handle resource requests ..."""

# Imports Part 1: Standard imports
from __future__ import annotations

from copy import copy
from functools import reduce, wraps
from operator import concat
from typing import TYPE_CHECKING, List

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.negotiation.objects import (
    ProcessCallForProposal, ProcessExecutionsPathProposal, ProcessExecutionsVariantProposal)
from ofact.twin.agent_control.behaviours.planning.planning_request import PlanningRequest
from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import (
    ProcessExecutionsComponent, ProcessExecutionsPath)
from ofact.twin.state_model.entities import (Resource, StationaryResource, NonStationaryResource,
                                             PassiveMovingResource, ActiveMovingResource)
from ofact.twin.state_model.processes import ProcessExecution
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.negotiation.objects import ResourceCallForProposal

# ToDo: check the length of the acc preference  - valid?


def memoize_transport_processes(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > 200:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def _coordinate_process_executions_variants(possible_path_combination):
    """Coordination means the matching of the different accepted_time_periods in the process_chain"""

    process_executions_preferences_before = \
        [process_execution_variant.reference_preference
         for process_execution_variant in possible_path_combination]

    for idx in range(1, len(possible_path_combination)):
        possible_path_combination[idx].predecessors.append(possible_path_combination[idx - 1])

    connector_preference = None
    connector_preference1 = None
    if possible_path_combination[0].process_executions_components:  # connector
        connector_components = \
            [component
             for component in possible_path_combination[0].get_process_executions_components_lst()
             if isinstance(component.goal_item, NonStationaryResource)]

        if connector_components:
            if connector_components[0].process_executions_components:
                connector_component = connector_components[0].get_process_executions_components_lst()[0]
                connector_preference = connector_component.reference_preference
                connector_preference1 = connector_component.reference_preference

    accepted_start_time = None
    for preference_before in process_executions_preferences_before:

        if accepted_start_time is not None:
            preference_before.update_accepted_time_periods_with_predecessor(accepted_start_time)

            if preference_before.accepted_time_periods.size == 0:
                return None
        accepted_start_time = preference_before.get_accepted_start_time_successor()
        if accepted_start_time is None:
            return None

    accepted_end_time = None
    for preference_before in reversed(process_executions_preferences_before):

        if accepted_end_time is not None:
            preference_before.update_accepted_time_periods_with_successor(accepted_end_time)
            if preference_before.accepted_time_periods.size == 0:
                return None

        accepted_end_time = preference_before.get_accepted_end_time_predecessor()
        if accepted_end_time is None:
            return None

    if connector_preference is not None and process_executions_preferences_before:
        accepted_end_time -= connector_preference.expected_process_execution_time
        connector_preference.update_accepted_time_periods_with_successor(accepted_end_time)

        connector_preference1.update_accepted_time_periods_with_successor(accepted_end_time)

    return process_executions_preferences_before


def _get_resource_position_through_projection(process_executions_projection, digital_twin, available_resource,
                                              matching_time_stamp, locations_of_demand):
    # good idea but not completely used - only the current position is used
    stationary_resource = None
    resource_position = \
        process_executions_projection[available_resource].get_position_at(matching_time_stamp)

    if resource_position is None:
        situated_in_resource = available_resource.situated_in
        if situated_in_resource is not None:
            if isinstance(situated_in_resource, StationaryResource):
                stationary_resource = situated_in_resource
            elif situated_in_resource in process_executions_projection:
                resource_position = \
                    process_executions_projection[situated_in_resource].get_position_at(matching_time_stamp)
            else:
                # no responsibility for the resource
                raise NotImplementedError
        else:
            situated_in_changes = (
                available_resource.dynamic_attributes.attributes["situated_in"].changes)["ProcessExecution"]
            for process_execution in situated_in_changes:
                print(process_execution)
            raise Exception(f"Resource {available_resource.external_identifications} "
                            f"gives no indication to find the position {matching_time_stamp}, {available_resource}")

    if stationary_resource is None:
        stationary_resources = digital_twin.get_stationary_resource_at_position(pos_tuple=resource_position)

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
                elif available_resource.entity_type.super_entity_type in possible_entity_types_to_store:
                    stationary_resources_possible.append(stationary_resource)
            if len(stationary_resources_possible) == 1:
                stationary_resource = stationary_resources_possible[0]
            elif locations_of_demand:
                if locations_of_demand[0] in stationary_resources_possible:
                    stationary_resource = locations_of_demand[0]
            else:
                raise NotImplementedError("The position cannot be specified", available_resource.name)

    return stationary_resource


def _determine_transport_needed(entity_to_transport, origin_resource, destination_resource, preference_requester):
    """Determine if transport between origin_resource and the destination resource is necessary"""

    if entity_to_transport.entity_type in preference_requester.lead_time:
        return False

    if entity_to_transport == origin_resource:
        return False

    if origin_resource.identification == destination_resource.identification:
        return False
    if origin_resource.situated_in:
        if origin_resource.situated_in.identification == destination_resource.identification:
            return False
    if destination_resource.situated_in:
        if origin_resource.identification == destination_resource.situated_in.identification:
            return False
    if origin_resource.check_intersection_base_areas(destination_resource):
        return False

    return True


def check_organization_needed(process_execution):
    """check if the organization of the process_execution is needed (resource demands not completely satisfied)"""

    process_organization_needed = False
    # identify resource demand
    resource_model_complete = process_execution.check_resources_build_resource_group()
    if not resource_model_complete:
        print("Error - HERE: resource model not complete")
        for resource in process_execution.get_resources():
            if isinstance(resource, NonStationaryResource):
                process_executions = resource.get_process_execution_history()
                orders = []
                for process_execution in process_executions:
                    if process_execution is None or isinstance(process_execution, str):
                        continue
                    print(process_execution.executed_start_time, process_execution.executed_end_time,
                          process_execution.get_process_name())
                    if process_execution.order:
                        print(process_execution.order.identifier, process_execution.order.external_identifications)

                        orders.append(process_execution.order)
                print("\n Order view")
                orders = list(set(orders))
                for order in orders:
                    print(order.identifier, order.external_identifications)
                    for process_execution in order.get_process_executions():
                        if process_execution is None or isinstance(process_execution, str):
                            continue

                        print(process_execution.executed_start_time, process_execution.executed_end_time,
                              process_execution.get_process_name())

                    print("Delivery Date actual", order.delivery_date_actual)

        print(process_execution.get_name())
        print(process_execution.order.external_identifications)
        print([resource.name for resource in process_execution.get_resources()])
        print(process_execution.main_resource)
        print([[resource.name for resource in rm.resources]
               for rm in process_execution.process.get_resource_groups()])


    # identity parts demand
    input_parts_to_transform, input_resources_to_transform, parts_complete = \
        process_execution.check_availability_of_needed_entities(event_type=ProcessExecution.EventTypes.PLAN)
    if not (resource_model_complete and parts_complete):
        print(process_execution.get_name(), resource_model_complete, process_execution.get_resources())
        process_organization_needed = True

    return process_organization_needed


def _get_process_executions_path_variant_updated(process_executions_path_variant):
    """Get the process_executions_path_variant through updating the process_executions_path preference"""

    process_executions_preferences_before = \
        process_executions_path_variant.get_process_executions_components_preferences()

    connector = process_executions_path_variant.get_entity_used()
    process_executions_path_variant.merge_horizontal(process_executions_preferences_before,
                                                     connector_object_entity_type=connector.entity_type)

    return process_executions_path_variant


def _get_entity_type_usage(resource_entity_types_before):
    entity_type_usage = {}
    for idx, resource_entity_types_before_single in enumerate(resource_entity_types_before):
        for resource_entity_type in resource_entity_types_before_single:
            entity_type_usage.setdefault(resource_entity_type, []).append(idx)

    return entity_type_usage


def _get_neighbour_times(current_entity_types, entity_type_usage, expected_process_times_before, entity_to_transport,
                         reference_preference, len_idx_process_executions_before, idx):
    """Determine the lead_time and the follow_up_time results from the integration of the transport processes"""

    lead_time = {}
    follow_up_time = {}
    for current_entity_type in current_entity_types:

        min_idx = min(entity_type_usage[current_entity_type])
        if (len(entity_type_usage[current_entity_type]) > 1) and (min_idx < idx):
            lead_time[current_entity_type] = sum(expected_process_times_before[min_idx:idx])

        max_idx = max(entity_type_usage[current_entity_type])
        if (len(entity_type_usage[current_entity_type]) > 1) and (max_idx > idx):
            follow_up_time[current_entity_type] = sum(expected_process_times_before[idx:max_idx])

    if entity_to_transport.entity_type in follow_up_time:
        follow_up_time[entity_to_transport.entity_type] += reference_preference.expected_process_execution_time
    elif entity_to_transport.entity_type.super_entity_type in follow_up_time:
        follow_up_time[entity_to_transport.entity_type.super_entity_type] += \
            reference_preference.expected_process_execution_time
    elif entity_to_transport in follow_up_time:
        if reference_preference.expected_process_execution_time > 0:
            follow_up_time[entity_to_transport] = reference_preference.expected_process_execution_time
    elif idx == len_idx_process_executions_before:
        # for the last transport process
        follow_up_time[entity_to_transport] = reference_preference.expected_process_execution_time
    # ToDo: Schmaus adaption follow up time from the element above

    return lead_time, follow_up_time


def _update_lead_time_reference(reference_preference, entity_to_transport, entity_type_usage,
                                expected_process_times_before):
    """Update the lead_time of the reference (for the process requested originally)"""

    lead_time = {}
    if entity_to_transport.entity_type in entity_type_usage:
        usage_indices = entity_type_usage[entity_to_transport.entity_type]
        lead_time[entity_to_transport] = sum(expected_process_times_before[min(usage_indices):])
    elif entity_to_transport.entity_type.super_entity_type in entity_type_usage:
        usage_indices = entity_type_usage[entity_to_transport.entity_type.super_entity_type]
        lead_time[entity_to_transport] = sum(expected_process_times_before[min(usage_indices):])

    reference_preference.lead_time = lead_time

    return reference_preference


def _inform_resource_not_available(order, available_resources, unrequested_available_resources,
                                   resources_requested_in_round, resource_reservation, round_):
    """Case: resources are not available because they are already reserved for other orders"""

    print(f"No resource found for order: {order.identification} {order.external_identifications}")
    print(f"Resource reservation: {[((resource.__class__.__name__, resource.name), 
                                     order.identification if order is not None else None) 
                                    for resource, order in resource_reservation.items()]}")

    try:
        print(f"Resources not available: {resources_requested_in_round[round_]}")
        print(available_resources[0].identification, available_resources[0].external_identifications)
        print("Unrequested: ", unrequested_available_resources)
    except:
        print("Not available: ")


def raise_exception_io_should_be_determinable(msg, entity_to_transport, order, process):
    resource_name = None
    if entity_to_transport is not None:
        try:
            resource_name = (entity_to_transport.name, entity_to_transport.external_identifications)
            if isinstance(entity_to_transport, NonStationaryResource):
                orders = []
                process_executions = entity_to_transport.get_process_execution_history()
                for process_execution in process_executions:
                    if process_execution is None:
                        continue
                    print(process_execution.executed_start_time, process_execution.executed_end_time,
                          process_execution.get_process_name())
                    if process_execution.order:
                        print(process_execution.order.identifier, process_execution.order.external_identifications)

                        orders.append(process_execution.order)

                print("\n Order view")
                orders = list(set(orders))
                for order in orders:
                    print(order.identifier, order.external_identifications)
                    for process_execution in order.get_process_executions():
                        if process_execution is None:
                            continue
                        print(process_execution.executed_start_time, process_execution.executed_end_time,
                              process_execution.get_process_name())

                    print("Delivery Date actual", order.delivery_date_actual)
        except:
            pass

    order_name = None
    if order is not None:
        order_name = (order.identifier, order.external_identifications)

    process_name = None
    possible_or = None
    possible_des = None
    if process is not None:
        process_name = process.name
        possible_or = [origin.name for origin in process.get_possible_origins()]
        possible_des = [origin.name for origin in process.get_possible_destinations()]

    raise NotImplementedError(msg, f"\n Resource name: {resource_name}", f"\n Order name: {order_name}",
                              f"\n Process name: {process_name} - {possible_or} | {possible_des}")


class ResourceRequest(PlanningRequest):  # ToDo: cyclic behaviour not needed
    """
    The resource request is responsible for the organization of a resource and is triggered by the method
    process_request.
    See method description for understanding the behaviour
    """
    MAX_PROPOSALS = 1

    def __init__(self):
        super().__init__()
        self.resources_requested_in_round = {}

    async def process_request(self, negotiation_object_request: ResourceCallForProposal):
        """
        The request for a resource is handled in  several steps:
        1. determine the possible resources that can match to the entity_type needed if no entity itself requested
        (long time reservation)
        2. after it the possible time_periods are determined that match with the requested time period
        3. according to these time_periods, for each resource a process_executions_path is organized
        - the path contains the process requested to be participated in and
          can be extended by further transport processes that bring the resource to the location of demand
        :param negotiation_object_request: the negotiation_object_request
        :return: process_executions_components (/ input for the proposals)
        """

        (reference_cfp, client_object, order, preference_requester, request_type, issue_id, requested_entity_types,
         entity_types_storable, locations_of_demand, fixed_origin, long_time_reservation, node_path) = (
            negotiation_object_request.unpack())

        request_id = negotiation_object_request.identification
        process_execution = client_object

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Client {client_object.identification} start ResourceRequest {request_id}")
        available_resources, long_time_reservation_duration, transport_access_needed = \
            self._get_resources_by_entity_types(requested_entity_types, entity_types_storable,
                                                long_time_reservation)

        if type(self).MAX_PROPOSALS is not None:
            unrequested_available_resources = self.get_available_unrequested_resources(available_resources, order)
            chosen_available_resources = self.choose_resources(unrequested_available_resources)
            available_resources_ = chosen_available_resources
            # print("Match: ", available_resources[0].external_identifications, order.external_identifications)
            if not available_resources_:
                round_ = self.agent.agents.get_current_round()
                _inform_resource_not_available(order, available_resources, unrequested_available_resources,
                                               self.resources_requested_in_round, self.agent.resource_reservation,
                                               round_)

            if available_resources_:
                if isinstance(available_resources_[0], PassiveMovingResource):
                    self.set_resource_requested(available_resources_, order)
                    # print("Resource PMR:", order.external_identifications["Schmaus"], available_resources_[0].name)

            available_resources = available_resources_

        # if isinstance(available_resources[0], PassiveMovingResource):
        #     print("DEBUG:", available_resources[0])
        #     print(order)

        resource_matching_time_periods = \
            self._get_matching_time_periods(available_resources=available_resources,
                                            preference_requester=preference_requester, issue_id=issue_id,
                                            long_time_reservation_duration=long_time_reservation_duration)

        # Process requirements
        process_executions_paths, sub_proposals, proposals_to_reject = \
            await self._organize_process_requirements(resource_matching_time_periods=resource_matching_time_periods,
                                                      locations_of_demand=locations_of_demand,
                                                      fixed_origin=fixed_origin,
                                                      requested_entity_types=requested_entity_types,
                                                      preference_requester=preference_requester,
                                                      cfp=negotiation_object_request, order=order,
                                                      issue_id=issue_id, request_id=request_id,
                                                      long_time_reservation_duration=long_time_reservation_duration,
                                                      node_path=node_path, process_execution=process_execution,
                                                      transport_access_needed=transport_access_needed)

        process_executions_components = process_executions_paths
        if process_executions_components:
            successful = True
        else:
            successful = False

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" End ResourceRequest {request_id} {len(process_executions_components)}, {len(proposals_to_reject)}, "
        #       f"{requested_entity_types[0][0].name}")

        return successful, process_executions_components, proposals_to_reject

    def set_resource_requested(self, resources_taken, order):
        if len(self.resources_requested_in_round) > 100:
            self.resources_requested_in_round.popitem()

        round_ = self.agent.agents.get_current_round()
        resources_requested_in_round_round_dict = self.resources_requested_in_round.setdefault(round_, {})
        resources_requested_in_round_round_dict.setdefault(order.identification, []).extend(resources_taken)

    def get_available_unrequested_resources(self, available_resources, order):
        """Same issue same resources - Other issue other resource"""

        if not available_resources:
            return available_resources

        if not isinstance(available_resources[0], PassiveMovingResource):
            return available_resources

        round_ = self.agent.agents.get_current_round()
        if round_ in self.resources_requested_in_round:
            resources_requested_in_round_round_dict = self.resources_requested_in_round[round_]
            if order.identification in resources_requested_in_round_round_dict:
                return resources_requested_in_round_round_dict[order.identification]

            if resources_requested_in_round_round_dict:
                already_requested_resources_other_issues = \
                    set(reduce(concat, list(resources_requested_in_round_round_dict.values())))
                available_resources = list(
                    set(available_resources).difference(already_requested_resources_other_issues))

        return available_resources

    def choose_resources(self, available_resources):
        if not available_resources:
            return available_resources

        if isinstance(available_resources[0], PassiveMovingResource):
            available_resources = available_resources[:type(self).MAX_PROPOSALS]

        return available_resources

    def _get_resources_by_entity_types(self, requested_entity_types, entity_types_storable, long_time_reservation) -> \
            [List[Resource], List[List[Resource]], bool]:
        """
        Returns two lists, one with possible main_resources and another with possible resources
        :return transport_access_needed: means that the resource itself is responsible to manage the transport access"""
        if requested_entity_types[0][1] != 1 or len(requested_entity_types) > 1:
            raise NotImplementedError

        requested_entity_type = requested_entity_types[0][0]
        resources_with_requested_entity_type = None

        if long_time_reservation:
            duration_transport_access_needed = list(long_time_reservation.values())[0]
            long_time_reservation_duration = duration_transport_access_needed[0]
            transport_access_needed = duration_transport_access_needed[1]

            resource_or_entity_type = list(long_time_reservation.keys())[0]
            if isinstance(resource_or_entity_type, Resource):
                resource = resource_or_entity_type
                if resource in self.agent.resources[requested_entity_type]:
                    resources_with_requested_entity_type = list(long_time_reservation.keys())
                elif resource.entity_type.super_entity_type == requested_entity_type:
                    resources_with_requested_entity_type = list(long_time_reservation.keys())

        else:
            long_time_reservation_duration = None
            transport_access_needed = True

        if resources_with_requested_entity_type is not None:
            return resources_with_requested_entity_type, long_time_reservation_duration, transport_access_needed

        if requested_entity_type in self.agent.resources:
            possible_resources = self.agent.resources[requested_entity_type]
            resources_with_requested_entity_type = [possible_resource
                                                    for possible_resource in possible_resources
                                                    if self.agent.resource_reservation[possible_resource] is None]

        else:
            for resource in self.agent.resources:
                print((resource.identification, ": ", resource.name))
            raise NotImplementedError(self.agent.name, requested_entity_type.identification, requested_entity_type.name)

        if entity_types_storable:
            resources_with_requested_entity_type = \
                list(set(resource for resource in resources_with_requested_entity_type
                         for entity_type in entity_types_storable
                         if resource.check_entity_type_storable(entity_type) or
                         resource.entity_type.check_entity_type_match(entity_type)))

        return resources_with_requested_entity_type, long_time_reservation_duration, transport_access_needed

    def _get_matching_time_periods(self, available_resources, preference_requester, issue_id,
                                   long_time_reservation_duration):
        """Get matching time periods (free time periods on the resources) for the available resources"""

        free_time_periods = preference_requester.get_free_time_periods()
        if free_time_periods.shape[0] == 0:
            return None

        resource_matching_time_periods = {}
        for available_resource in available_resources:
            preference_requester: ProcessExecutionPreference
            time_slot_duration_needed = preference_requester.expected_process_execution_time
            follow_up_time = preference_requester.determine_follow_up_time_entity_entity_types(available_resource)
            time_slot_duration_needed += follow_up_time
            try:
                matching_time_periods = \
                self.agent.preferences[available_resource].get_time_periods_matching(
                    free_time_periods_other=free_time_periods,
                    time_slot_duration=time_slot_duration_needed,
                    start_time=free_time_periods[0][0], end_time=free_time_periods[-1][1],
                    issue_id=issue_id,
                    long_time_reservation_duration=long_time_reservation_duration)
            except:
                raise Exception(self.agent.name, available_resource.name)
            if matching_time_periods is None:
                continue

            # matching_time_periods = np.array([matching_time_periods[-1]])

            # print("Matching time period's resource: ", available_resource, matching_time_periods)
            resource_matching_time_periods[available_resource] = matching_time_periods

        if available_resources:
            if isinstance(available_resources[0], NonStationaryResource):
                resource_matching_time_periods_adapted = {}
                for available_resource, matching_time_periods in resource_matching_time_periods.items():
                    if matching_time_periods.any():
                        resource_matching_time_periods_adapted[available_resource] = matching_time_periods[-1:]
                    else:
                        resource_matching_time_periods_adapted[available_resource] = matching_time_periods

        # if not resource_matching_time_periods:
        #     print

        return resource_matching_time_periods

    async def _organize_process_requirements(self, resource_matching_time_periods, locations_of_demand, fixed_origin,
                                             preference_requester, cfp, order, issue_id, request_id,
                                             long_time_reservation_duration, node_path, process_execution,
                                             transport_access_needed, requested_entity_types):
        """Find possible constellation to fulfill the requirements"""

        # create for each available resource one or more path alternatives
        path_alternatives = self._get_path_alternatives(resource_matching_time_periods, locations_of_demand,
                                                        fixed_origin, preference_requester)

        path_alternatives = self._specify_path_alternatives(path_alternatives, transport_access_needed,
                                                            long_time_reservation_duration)

        if transport_access_needed:
            path_alternatives = self._determine_transports(path_alternatives, order, issue_id,
                                                           node_path, cfp)

        process_executions_paths: list[ProcessExecutionsPath] = \
            self._create_process_executions_paths(path_alternatives, issue_id, node_path, process_execution, cfp,
                                                  requested_entity_types)

        process_executions_organization_needed = self._derive_process_organization_demand(process_executions_paths)

        if any(list(process_executions_organization_needed.values())):
            process_organization_proposals = \
                await self._request_process_organization(process_executions_paths,
                                                         process_executions_organization_needed,
                                                         cfp, order, node_path, issue_id)
        else:
            process_organization_proposals = []

        # ToDo: connect proposals and process_executions_components
        # ToDo: combine the process_executions_components components
        # ToDo: next
        process_executions_paths, sub_proposals, proposals_to_reject = \
            self._combine_process_path_components(process_executions_paths, process_organization_proposals)

        return process_executions_paths, sub_proposals, proposals_to_reject

    def _get_path_alternatives(self, resource_matching_time_periods, locations_of_demand, fixed_origin,
                               preference_requester):
        """
        Determine alternatives for each available resource according their already planned process_executions.
        """

        path_alternatives = []
        for available_resource, matching_time_periods in resource_matching_time_periods.items():
            if isinstance(available_resource, StationaryResource):
                position_matching_periods = {available_resource: matching_time_periods}
            else:
                position_matching_periods: dict[Resource, list] = {}
                if matching_time_periods is None:
                    continue

                for matching_time_period in matching_time_periods:
                    stationary_resource = \
                        self._get_resource_position(fixed_origin, preference_requester, locations_of_demand,
                                                    available_resource, matching_time_period[0])

                    position_matching_periods.setdefault(stationary_resource,
                                                         []).append(matching_time_period)

            # determine all alternatives for a resource
            path_alternatives_resource = \
                [{"available_resource": available_resource,
                  "position": position,
                  "location_of_demand": location_of_demand,
                  "matching_time_periods": np.stack((matching_time_periods_position)),
                  "reference_preference": preference_requester.get_copy()}
                 for position, matching_time_periods_position in position_matching_periods.items()
                 # ToDo: Find a better way!!!!!!!!!!!!!!!!!!!!!
                 for location_of_demand in locations_of_demand]
            path_alternatives.extend(path_alternatives_resource)

        return path_alternatives

    def _get_resource_position(self, fixed_origin, preference_requester, locations_of_demand, available_resource,
                               matching_time_stamp):
        """Get the resource (position) where the available resource is"""

        if fixed_origin is not None:
            stationary_resource = fixed_origin
            return stationary_resource

        elif preference_requester.lead_time:
            stationary_resource = locations_of_demand[0]
            # assumption - I'm already at the destination,
            # if the origin is not given and processes before are planned
            return stationary_resource

        process_executions_projection = self.agent.agents.process_executions_projections
        digital_twin = self.agent.digital_twin
        stationary_resource = _get_resource_position_through_projection(process_executions_projection, digital_twin,
                                                                        available_resource, matching_time_stamp,
                                                                        locations_of_demand)
        # print("DEBUG: ", available_resource.name, matching_time_stamp, stationary_resource.name)
        return stationary_resource

    def _specify_path_alternatives(self, path_alternatives, transport_access_needed, long_time_reservation_duration):
        """Specification of path alternatives with preference and transport access needed"""

        for path_alternative in path_alternatives:
            preference = path_alternative["reference_preference"]
            matching_time_periods = path_alternative["matching_time_periods"]
            preference.merge_free_time_period_into_accepted(matching_time_periods)

            preference.long_time_reservation_duration = long_time_reservation_duration

            preference.reference_objects = [path_alternative["available_resource"]]
            if transport_access_needed:
                origin_resource = path_alternative["position"]
            else:
                origin_resource = path_alternative["location_of_demand"]
            destination_resource = path_alternative["location_of_demand"]
            preference.origin = origin_resource
            preference.destination = destination_resource

            entity_to_transport = path_alternative["available_resource"]
            resource_preference = self.agent.preferences[entity_to_transport]
            preference.merge_resource(resource_preference)

            if transport_access_needed:
                transport_access_needed_ = _determine_transport_needed(entity_to_transport, origin_resource,
                                                                       destination_resource, preference)

            else:
                transport_access_needed_ = copy(transport_access_needed)

            path_alternative["transport_access_needed"] = transport_access_needed_
            path_alternative["transport_process_execution_components"] = []

        return path_alternatives

    def _determine_transports(self, path_alternatives, order, issue_id, node_path, cfp):
        """Determine the transport based on the path_alternatives given to transport the available resource
        to the location of demand"""

        new_path_alternatives = []
        for path_alternative in path_alternatives:
            transport_needed = path_alternative["transport_access_needed"]

            if not transport_needed:
                new_path_alternatives.append(path_alternative)
                continue

            entity_to_transport = path_alternative["available_resource"]
            origin_resource = path_alternative["position"]
            destination_resource = path_alternative["location_of_demand"]
            preference = path_alternative["reference_preference"]

            transport_process_executions_components = \
                self._get_transport_process_executions_components(preference, origin_resource, destination_resource,
                                                                  entity_to_transport, order, issue_id, node_path, cfp)

            if transport_process_executions_components is None:
                continue

            # print("Transport Access needed", len(transport_process_executions_components))

            path_alternative["transport_process_execution_components"] = transport_process_executions_components
            new_path_alternatives.append(path_alternative)

        return new_path_alternatives

    def _get_transport_process_executions_components(self, preference, origin_resource, destination_resource,
                                                     entity_to_transport, order, issue_id, node_path, cfp):
        """Determine the process_executions_components also including the preferences for the transport processes"""

        # transport to the station needed
        transport_processes = \
            self._determine_transport_processes(origin_resource, destination_resource, entity_to_transport)
        # print("TA", [process["process"].name for process in transport_processes])
        # if origin_resource is not None:
        #     print(origin_resource.name)
        # if origin_resource is not None:
        #     print(destination_resource.name)

        transport_process_executions, process_executions_variants = \
            self._organize_transport_process_executions_variants(transport_processes, entity_to_transport,
                                                                 order, issue_id, node_path, cfp)

        process_executions_preferences = \
            self._determine_process_executions_preferences(entity_to_transport=entity_to_transport,
                                                           process_executions_before=transport_process_executions,
                                                           reference_preference=preference)

        # add the reference_preference
        for process_execution, preference in process_executions_preferences.items():
            variant = process_executions_variants[process_execution]
            # if preference is None:
            #     return None
            variant.reference_preference = preference
            sub_components = variant.get_process_executions_components_lst()
            if sub_components:  # ToDo: not sure if this case is correct
                sub_path = sub_components[0]
                sub_path.reference_preference = preference
            else:
                print(order.external_identifications)
                raise Exception("Order:", order)

        transport_process_executions_components = list(process_executions_variants.values())
        return transport_process_executions_components

    @memoize_transport_processes
    def _determine_transport_processes(self, origin_resource, destination_resource, entity_to_transport):
        """Determine the transport processes for the transition of the entity_to_transport"""

        # ToDo: differentiate between transport access (entity_to_transport can drive by itself and no alternative)
        # and transport
        if isinstance(entity_to_transport, NonStationaryResource):
            transport = False
        else:
            transport = True

        transport_processes = \
            self.agent.routing_service.get_transit_processes(origin_resource, destination_resource,
                                                             support_entity_type=entity_to_transport.entity_type,
                                                             transport=transport, transfers=False)
        # ToDo: and or or operator needed
        # ToDo: Schmaus hack - determine the situated in for the requested time period
        # if entity_to_transport.situated_in or isinstance(entity_to_transport, ActiveMovingResource):  # ToDo
        #     support_entity_type = transport_processes[0]["process"].get_support_entity_type()
        #     main_part_entity_type = transport_processes[0]["process"].get_main_entity_entity_type()
        #     loading_process_d: dict = self.agent.routing_service.get_transfer_process(
        #         origin=origin_resource, support_entity_type=support_entity_type,
        #         entity_entity_type=main_part_entity_type, level_differences_allowed=True)
        #     # print(entity_to_transport.name, origin_resource, destination_resource)
        #     # print("Loading:", loading_process_d)
        #     support_entity_type = transport_processes[-1]["process"].get_support_entity_type()
        #     main_part_entity_type = transport_processes[-1]["process"].get_main_entity_entity_type()
        #     unloading_process_d: dict = self.agent.routing_service.get_transfer_process(
        #         destination=destination_resource, support_entity_type=support_entity_type,
        #         entity_entity_type=main_part_entity_type, level_differences_allowed=True)
        #     # print("Unloading:", unloading_process_d)
        #     if loading_process_d:
        #         transport_processes = [loading_process_d] + transport_processes
        #     if unloading_process_d:
        #         transport_processes.append(unloading_process_d)
        #     print("Schmaus WorkAround")

        return transport_processes

    def _organize_transport_process_executions_variants(self, transport_processes, entity_to_transport, order, issue_id,
                                                        node_path, cfp):
        """Organize the transport process_executions_variants needed for the transport of the resources
        to the location of demand."""

        transport_process_executions = \
            self._get_transport_process_executions(transport_processes, entity_to_transport, order)

        process_executions_variants = {}
        for transport_process_execution in transport_process_executions:

            possible_resource_entity_types = transport_process_execution.get_possible_resource_entity_types()

            if entity_to_transport.entity_type in possible_resource_entity_types or \
                    entity_to_transport.entity_type.super_entity_type in possible_resource_entity_types:
                transport_process_execution_path = \
                    ProcessExecutionsPathProposal(call_for_proposal=cfp, provider=self.agent.name,
                                                  goal=entity_to_transport.entity_type, goal_item=entity_to_transport,
                                                  issue_id=issue_id,
                                                  process_execution_id=transport_process_execution.identification,
                                                  connector_objects=[entity_to_transport], cfp_path=node_path,
                                                  type_=ProcessExecutionsComponent.Type.CONNECTOR)
                transport_process_execution_paths = \
                    {entity_to_transport.entity_type: [transport_process_execution_path]}

            else:
                transport_process_execution_paths = {}

            transport_process_execution_variant = \
                ProcessExecutionsVariantProposal(call_for_proposal=cfp, provider=self.agent.name, issue_id=issue_id,
                                                 process_execution_id=transport_process_execution.identification,
                                                 goal=transport_process_execution,
                                                 goal_item=transport_process_execution,
                                                 reference_preference=None,
                                                 process_executions_components=transport_process_execution_paths,
                                                 cfp_path=node_path, type_=ProcessExecutionsComponent.Type.CONNECTOR)

            process_executions_variants[transport_process_execution] = transport_process_execution_variant

        return transport_process_executions, process_executions_variants

    def _get_transport_process_executions(self, transport_processes, entity_to_transport, order):
        """Determine the transport process_executions based on the path given from the routing service"""

        transport_process_executions = [self._get_transport_process_execution(transport_process_dict,
                                                                              entity_to_transport, order)
                                        for transport_process_dict in transport_processes]

        return transport_process_executions

    def _get_transport_process_execution(self, transport_process_dict, entity_to_transport, order):
        """Determine a transport process execution"""

        origin = transport_process_dict["origin"]
        destination = transport_process_dict["destination"]
        process = transport_process_dict["process"]
        if not origin:
            if process.get_possible_origins():
                raise_exception_io_should_be_determinable("Origin determinable", entity_to_transport, order,
                                                          process)

        if not destination:
            if process.get_possible_destinations():
                raise_exception_io_should_be_determinable("Destination determinable", entity_to_transport, order,
                                                          process)


        transport_process_execution = \
            ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, order=order, process=process, origin=origin,
                             destination=destination, resulting_quality=1,
                             source_application=self.agent.source_application)

        possible_resource_entity_types = transport_process_execution.get_possible_resource_entity_types()
        if entity_to_transport.entity_type in possible_resource_entity_types:
            transport_process_execution.resources_used += [(entity_to_transport,)]

        elif entity_to_transport.entity_type.super_entity_type in possible_resource_entity_types:
            transport_process_execution.resources_used += [(entity_to_transport,)]

        if origin is not None:
            if origin.entity_type in possible_resource_entity_types:
                transport_process_execution.resources_used += [(origin,)]
        if destination is not None:
            if destination.entity_type in possible_resource_entity_types:
                transport_process_execution.resources_used += [(destination,)]

        possible_main_resource_entity_types = \
            transport_process_execution.get_possible_main_resource_entity_types()
        if entity_to_transport.entity_type in possible_main_resource_entity_types:
            transport_process_execution.main_resource = entity_to_transport
        elif entity_to_transport.entity_type.super_entity_type in possible_main_resource_entity_types:
            transport_process_execution.main_resource = entity_to_transport
        else:
            print(possible_main_resource_entity_types)

        return transport_process_execution

    def _create_process_executions_paths(self, path_alternatives, issue_id, node_path, process_execution, cfp,
                                         requested_entity_types):
        """Create the process_Executions_paths
        # therefore the transport process_executions (variants) are added to the path"""
        process_executions_paths: list[ProcessExecutionsPath] = []
        for path_alternative in path_alternatives:
            transport_process_execution_components_lst = path_alternative["transport_process_execution_components"]
            if transport_process_execution_components_lst:
                transport_process_execution_components = {path_alternative["available_resource"].entity_type:
                                                              transport_process_execution_components_lst}
            else:
                transport_process_execution_components = {}

            if transport_process_execution_components_lst:
                connector_objects = \
                    transport_process_execution_components_lst[0].get_process_executions_components_lst()[
                        0].connector_objects
            else:
                connector_objects = None

            resource_entity_type = path_alternative["available_resource"].entity_type
            if resource_entity_type != requested_entity_types[0][0]:
                goal = requested_entity_types[0][0]  # can be the super entity_type of the resource
            else:
                goal = resource_entity_type

            process_executions_path = \
                ProcessExecutionsPathProposal(call_for_proposal=cfp, provider=self.agent.name,
                                              issue_id=issue_id, process_execution_id=process_execution.identification,
                                              goal=goal, goal_item=path_alternative["available_resource"],
                                              process_executions_components=transport_process_execution_components,
                                              reference_preference=path_alternative["reference_preference"],
                                              connector_objects=connector_objects, cfp_path=node_path)

            process_executions_paths.append(process_executions_path)

        return process_executions_paths

    def _determine_process_executions_preferences(self, entity_to_transport, process_executions_before,
                                                  reference_preference: ProcessExecutionPreference) \
            -> dict[ProcessExecution: ProcessExecutionPreference]:
        """Determine the process_executions preferences for the transport"""
        expected_process_times_before = [process_execution_before.get_max_process_time()
                                         for process_execution_before in process_executions_before]
        resource_entity_types_before = [process_execution_before.get_possible_resource_entity_types()
                                        for process_execution_before in process_executions_before]

        entity_type_usage = _get_entity_type_usage(resource_entity_types_before)

        # merge the process chain
        process_executions_preferences = {}
        resource_preference = self.agent.preferences[entity_to_transport]
        len_idx_process_executions_before = len(process_executions_before) - 1

        # min_time_restriction64 = np.datetime64(self.agent.change_handler.get_current_time(), "s")
        min_time_restriction64 = reference_preference.accepted_time_periods[0][0]

        for idx, process_execution_before in enumerate(process_executions_before):
            current_entity_types = resource_entity_types_before[idx]

            lead_time, follow_up_time = \
                _get_neighbour_times(current_entity_types, entity_type_usage, expected_process_times_before,
                                     entity_to_transport, reference_preference, len_idx_process_executions_before, idx)

            preference_before = reference_preference.get_process_execution_preference_before(
                expected_process_execution_time=expected_process_times_before[idx],
                process_execution=process_execution_before, lead_time=lead_time, follow_up_time=follow_up_time,
                min_time_restriction64=min_time_restriction64)

            reference_preference.merge_resource(resource_preference)
            process_executions_preferences[process_execution_before] = preference_before

        reference_preference = _update_lead_time_reference(reference_preference, entity_to_transport, entity_type_usage,
                                                           expected_process_times_before)

        return process_executions_preferences

    def _derive_process_organization_demand(self, process_executions_paths):
        """derive for each process_executions_path if an organization is needed
        (resource demands not completely satisfied)"""

        process_executions_organization_needed = \
            {process_executions_component: check_organization_needed(process_executions_component.goal_item)
             for process_execution_path in process_executions_paths
             for process_executions_component in process_execution_path.get_process_executions_components_lst()}

        return process_executions_organization_needed

    async def _request_process_organization(self, process_executions_paths, process_executions_organization_needed,
                                            cfp, order, node_path, issue_id):
        """Request transport access process for the resources to reach the location(s) of demand"""

        negotiation_object_identifications = []
        for process_executions_path in process_executions_paths:

            last_negotiation_object = None
            for process_execution in process_executions_path.get_process_executions_components_lst():
                if not process_executions_organization_needed[process_execution]:
                    continue

                preference = process_executions_path.get_preference_process_executions_component(process_execution)

                negotiation_object = \
                    ProcessCallForProposal(reference_cfp=cfp, predecessor_cfp=last_negotiation_object,
                                           sender_name=self.agent.name,
                                           client_object=process_executions_path.goal_item,
                                           order=order, issue_id=issue_id, process_execution=process_execution,
                                           preference=preference, long_time_reservation={}, node_path=node_path)

                last_negotiation_object = negotiation_object

                # determine the providers
                if process_execution.main_resource is not None:
                    providers = [process_execution.main_resource]
                else:
                    providers = process_execution.get_possible_resource_entity_types()

                negotiation_object_identifications.append(negotiation_object.identification)

                await self.agent.NegotiationBehaviour.call_for_proposal(negotiation_object, providers)

        if negotiation_object_identifications:
            proposals = await self.agent.NegotiationBehaviour.await_callback(negotiation_object_identifications,
                                                                             "resource")
        else:
            proposals = []

        return proposals

    def _combine_process_path_components(self, process_executions_paths, process_organization_proposals):
        """Combine the single elements of the process_executions_path - """

        proposals_to_reject = []
        sub_proposals = {}
        not_feasible_combinations = []
        complete_process_executions_paths = []

        for process_executions_path in process_executions_paths:
            if len(process_executions_path.process_executions_components) == 0:
                process_executions_path_variant = process_executions_path
            else:
                process_executions_path_variant = process_executions_path

                if process_organization_proposals:  # ToDo: integrate the proposals
                    raise Exception  # ToDo: copy needed (but only for the second object) .get_copy()
                # TODO: it will generate problems in the scheduling tree

            if process_executions_path_variant.process_executions_components:
                components_lst = process_executions_path_variant.get_process_executions_components_lst()
                process_executions_before = _coordinate_process_executions_variants(components_lst)
                if process_executions_before is None:
                    print("NOT FEASIBLE ...")
                    not_feasible_combinations.append(process_executions_path_variant)
                    continue  # not possible solution

            process_executions_path_variant = \
                _get_process_executions_path_variant_updated(process_executions_path_variant)

            feasible = process_executions_path_variant.feasible()  # based on the accepted_time_period
            if feasible:
                # (process_executions_path, combi_preference)
                complete_process_executions_paths.append(process_executions_path_variant)
                # maybe also update the preference
            else:
                not_feasible_combinations.append(process_executions_path_variant)
                if process_organization_proposals:
                    raise NotImplementedError("integrate the proposals")

        return complete_process_executions_paths, sub_proposals, proposals_to_reject
