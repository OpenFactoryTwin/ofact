"""
Contains the process request behaviour ...
"""

# Imports Part 2: Standard Imports
from __future__ import annotations

import itertools
import logging
import operator
from copy import copy
from functools import reduce, wraps
from operator import concat
from random import sample
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.negotiation.objects import CallForProposal, \
    ResourceCallForProposal, PartCallForProposal
from ofact.twin.agent_control.behaviours.negotiation.objects import ProcessExecutionsVariantProposal
from ofact.twin.agent_control.behaviours.planning.helpers import get_proposals_to_reject
from ofact.twin.agent_control.behaviours.planning.planning_request import PlanningRequest
from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import \
    ProcessExecutionsPath, ProcessExecutionsVariant, ProcessExecutionsComponent
from ofact.twin.state_model.entities import StationaryResource, Entity, Storage, NonStationaryResource, ConveyorBelt
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
if TYPE_CHECKING:
    from ofact.twin.state_model.entities import EntityType
    from ofact.twin.state_model.processes import ResourceModel, ProcessExecution, Process
    from ofact.twin.agent_control.behaviours.negotiation.objects import ProcessCallForProposal

logger = logging.getLogger("ProcessRequest")


def memoize_determine_resource_demand_process(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            resource_model_complete, resource_model_demands = cache[str(args) + str(kwargs)]
            return (copy(resource_model_complete), copy(resource_model_demands))

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result
        resource_model_complete, resource_model_demands = cache[str(args) + str(kwargs)]

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > 500:
            del cache[list(cache.keys())[0]]

        result = (copy(resource_model_complete), copy(resource_model_demands))
        return result

    return memoize


def _derive_resource_demands(process_execution) -> (bool, dict[ResourceModel, list[EntityType]]):
    """Determine the resource demand for one process_execution
    :return resource_model_complete: a bool value if a complete resource_model is found
    :return resource_model_demands: a dict that maps the resource entity_type demands for a complete resource model"""
    resource_model_complete, resource_model_demands = \
        _determine_resource_demand_process_execution(process_execution)
    return resource_model_complete, resource_model_demands


def _determine_resource_demand_process_execution(process_execution) -> (bool, dict[ResourceModel, list[EntityType]]):
    """Determine the resource demand for one process_execution
    :return resource_model_complete: a bool value if a complete resource_model is found
    :return resource_model_demands: a dict that maps the resource entity_type demands for a complete resource model"""

    possible_resource_models = process_execution.get_possible_resource_groups()
    available_resource_entity_types = [resource.entity_type for resource in process_execution.get_resources()]
    process = process_execution.process
    possible_origin, possible_destination = process_execution.get_origin_destination()
    resource_model_complete, resource_model_demands = \
        _determine_resource_demand_process(process, possible_resource_models, available_resource_entity_types,
                                           possible_origin, possible_destination)

    return resource_model_complete, resource_model_demands


@memoize_determine_resource_demand_process
def _determine_resource_demand_process(process, possible_resource_models, available_resource_entity_types,
                                       possible_origin, possible_destination) \
        -> (bool, dict[ResourceModel, list[EntityType]]):
    """Determine the resource demand for one process_execution
    :return resource_model_complete: a bool value if a complete resource_model is found
    :return resource_model_demands: a dict that maps the resource entity_type demands for a complete resource model"""

    resource_model_complete, resource_model_demands = \
        _get_possible_resource_model_resource_models(possible_resource_models, available_resource_entity_types)

    if len(resource_model_demands) <= 1:
        return resource_model_complete, resource_model_demands

    origin_destination = [possible_origin, possible_destination]
    resource_model_demands = \
        _get_limited_possible_resource_model_resource_models(resource_model_demands, origin_destination)

    if len(resource_model_demands) > 1:
        print("Warning - More than one rm: ", len(resource_model_demands), process.name)
        rg, demands = resource_model_demands.popitem()
        resource_model_demands = {rg: demands}

    return resource_model_complete, resource_model_demands


def _get_possible_resource_model_resource_models(possible_resource_models, available_resource_entity_types):
    # check if resource_model complete
    resource_model_complete = False
    resource_model_demands = {}

    # iterate through the resource models
    for resource_model in possible_resource_models:
        resource_model_demands[resource_model] = []
        one_resource_model_complete = True
        resources_used = []
        if not available_resource_entity_types:
            if resource_model.resources:
                one_resource_model_complete = False

        # iterate through the needed resource entity_types
        for needed_resource_entity_type in resource_model.resources:
            resource_available = False
            for idx, resource_entity_type in enumerate(available_resource_entity_types):
                if resource_entity_type.check_entity_type_match(needed_resource_entity_type) \
                        and idx not in resources_used:
                    # use an available resource for the resource model
                    resources_used.append(idx)
                    resource_available = True
                    break
                elif idx + 1 == len(available_resource_entity_types):
                    one_resource_model_complete = False

            # if a resource cannot be used
            if not resource_available:
                resource_model_demands[resource_model].append(needed_resource_entity_type)
        if one_resource_model_complete:
            resource_model_complete = True
            break

    return resource_model_complete, resource_model_demands


def _get_limited_possible_resource_model_resource_models(resource_model_demands, origin_destination):
    """Limit the possible resource models based on already determined origin and destination"""

    better_matching_resource_model_demands = {resource_model: 0
                                              for resource_model, needed_entity_types in resource_model_demands.items()}
    for resource in origin_destination:
        for resource_model, needed_entity_types in resource_model_demands.items():
            for needed_entity_type in needed_entity_types:

                if needed_entity_type.check_entity_type_match_lower(resource.entity_type):
                    better_matching_resource_model_demands[resource_model] += 1
                    continue

                suitable_storages = [storage for storage in resource.get_storages_without_entity_types()
                                     if needed_entity_type.check_entity_type_match_lower(storage.entity_type)]
                if suitable_storages:
                    better_matching_resource_model_demands[resource_model] += 1

    if sum(better_matching_resource_model_demands.values()) > 0:
        max_value = max(better_matching_resource_model_demands.items(),
                        key=operator.itemgetter(1))[1]
        resource_model_demands = {resource_model: resource_model_demands[resource_model]
                                  for resource_model, score in better_matching_resource_model_demands.items()
                                  if score == max_value}

    return resource_model_demands


def _get_locations_of_demand(process_execution):
    """Determine the locations the resource is demanded"""

    if process_execution.origin:
        locations_of_demand = [process_execution.origin]
    else:
        locations_of_demand = process_execution.process.get_possible_origins()

    locations_of_demand_stationary = \
        [location_of_demand
         for location_of_demand in locations_of_demand if
         isinstance(location_of_demand, StationaryResource)]
    if not locations_of_demand_stationary:
        if process_execution.destination:
            locations_of_demand = [process_execution.destination]
        else:
            locations_of_demand = process_execution.process.get_possible_destinations()

    return locations_of_demand


def _get_long_time_reservation(long_time_reservation, needed_resource_entity_type):
    """Determine the long time reservation for resources that are needed for more than one process
    like transport through the assembly"""

    if needed_resource_entity_type in long_time_reservation:
        long_time_reservation_single = \
            {needed_resource_entity_type: long_time_reservation[needed_resource_entity_type]}
    else:
        long_time_reservation_single = \
            {elem: long_time_reservation[elem]
             for elem in long_time_reservation if isinstance(elem, Entity)
             if needed_resource_entity_type.check_entity_type_match_lower(elem.entity_type)}
        if not long_time_reservation_single:
            long_time_reservation_single = None

    return long_time_reservation_single


def _determine_positions(process_executions_components, pe_origin, pe_destination, process):
    """Determine the origin and destination of the process_executions_component"""

    position_combinations = [(process_executions_component.get_entity_used(),
                              process_executions_component.get_origin(),
                              process_executions_component.get_destination())
                             for process_executions_component in process_executions_components]

    entities_used, possible_origins, possible_destinations = zip(*position_combinations)

    origin_position = _determine_origin_position(pe_origin, entities_used, possible_origins, process)
    destination_position = _determine_destination_position(pe_destination, entities_used,
                                                           possible_destinations, process)

    return origin_position, destination_position


def _determine_origin_position(pe_origin, entities_used, possible_origins, process: Process):
    """Determine the origin position for the process_execution"""

    if isinstance(pe_origin, StationaryResource):
        origin_position = pe_origin
        return origin_position

    possible_origins = list(set(possible_origins))

    try:
        origin_position = _get_origin_destination_position(entities_used, possible_origins, process.get_possible_origins())
    except:
        raise Exception("Process:", process.name,
                        [entity_used.name for entity_used in entities_used],
                        [resource.name for resource in possible_origins],
                        [r.name for r in process.get_possible_origins()])

    return origin_position


def _determine_destination_position(pe_destination, entities_used, possible_destinations, process: Process):
    """Determine the destination position for the process_execution"""

    if isinstance(pe_destination, StationaryResource):
        destination_position = pe_destination
        return destination_position

    possible_destinations = list(set(possible_destinations))
    destination_position = _get_origin_destination_position(entities_used, possible_destinations,
                                                            process.get_possible_destinations())

    return destination_position


def _get_origin_destination_position(entities_used, possible_origins_destinations, possible_origins_destinations_process):
    """Determine the destination or the origin position for the process_execution
    by matching possible and expected ones"""
    possible_origins_destinations = \
        list(set(possible_od for possible_od in possible_origins_destinations
                 if isinstance(possible_od, StationaryResource)))

    if len(possible_origins_destinations) != 1:
        possible_origins_destinations_deeper = \
            list(set(possible_od.situated_in
                     for possible_od in possible_origins_destinations
                     if possible_od.situated_in))
        # for the position a resource it is irrelevant
        if len(possible_origins_destinations_deeper) != 1:
            stationary_resource_indexes = [idx
                                           for idx, entity_used in enumerate(entities_used)
                                           if isinstance(entity_used, StationaryResource)]
            possible_origins_destinations_stationary = [possible_origins_destinations[idx]
                                                        for idx in stationary_resource_indexes]

            if len(possible_origins_destinations_stationary) != 1:
                # possible_origins_destinations_expected, entities_used
                possible_origins_destinations_ = []
                for possible_origin_destination in possible_origins_destinations_stationary:
                    if possible_origin_destination in possible_origins_destinations_process:
                        possible_origins_destinations_.append(possible_origin_destination)
                    elif possible_origin_destination.situated_in in possible_origins_destinations_deeper:
                        possible_origins_destinations_.append(possible_origin_destination)

                if len(possible_origins_destinations_) != 1:
                    raise NotImplementedError([resource.name for resource in possible_origins_destinations_stationary],
                                              [resource.name for resource in possible_origins_destinations],
                                              [resource.name for resource in possible_origins_destinations_process])
                else:
                    possible_origins_destinations_stationary = possible_origins_destinations_

            possible_origins_destinations_deeper = possible_origins_destinations_stationary

        possible_origins_destinations = possible_origins_destinations_deeper

    possible_origin_destination = possible_origins_destinations[0]

    return possible_origin_destination


def _determine_time_update_needed(updated_process_execution, process_executions_components_times):
    """
    Determine if the process_executions_components which enter the ProcessExecutionsVariant as sub-nodes have
    different process_times as in combination. This occurs if the efficiency or speed differs to 1.
    :return update_needed: a bool value that states if the time update is needed
    :return expected_process_time: the new expected_process_time
    """

    distance = updated_process_execution.get_distance()
    expected_process_time_combined = updated_process_execution.get_expected_process_time(distance=distance)
    if process_executions_components_times[0] != int(np.ceil(round(expected_process_time_combined, 1))) or \
            len(process_executions_components_times) > 1:
        update_needed = True
    else:
        update_needed = False

    return update_needed, expected_process_time_combined


def _combine_resources(resource_model_demands, preference, resource_proposals, process_execution: ProcessExecution):
    """Get the resources that can be used with others to build a complete resource model"""

    goals_process_executions_variants = {}
    for proposal, provider in resource_proposals:
        goal = proposal.goal
        goals_process_executions_variants.setdefault(goal,
                                                     []).append(proposal)

    usable_resource_components, usable_preferences, possible_origins, possible_destinations, \
        unusable_resource_components, relevant_goals = \
        _determine_resources_matchable(resource_model_demands, goals_process_executions_variants, preference,
                                       process_execution)

    unusable_resource_proposals = \
        _determine_unusable_resource_proposals(goals_process_executions_variants, relevant_goals,
                                               usable_resource_components, unusable_resource_components)

    return (usable_resource_components, usable_preferences, possible_origins, possible_destinations,
            unusable_resource_proposals)


def _determine_resources_matchable(resource_model_demands, goals_process_executions_variants, preference,
                                   process_execution):
    """Check if resources are combinable with other resources
    attr usable_resource_components: maps possible process_executions_paths to location of operations
    """

    relevant_goals = set()
    usable_resource_components: dict[StationaryResource, list[ProcessExecutionsComponent]] = {}
    usable_preferences: dict[StationaryResource, list[ProcessExecutionPreference]] = {}  # ToDo: list to ?
    possible_origins: dict[StationaryResource, list] = {}
    possible_destinations: dict[StationaryResource, list] = {}

    unusable_resource_components = []
    for resource_model, entity_types_needed in resource_model_demands.items():
        resource_model_possible = set(entity_types_needed).issubset(goals_process_executions_variants)
        if not resource_model_possible:
            continue

        resource_model_combinations = [goals_process_executions_variants[entity_type_needed]
                                       for entity_type_needed in entity_types_needed]

        usable = False
        all_combinations = list(itertools.product(*resource_model_combinations))
        resources_required = len(entity_types_needed)
        if len(set(entity_types_needed)) < resources_required:
            if len(all_combinations) > 4000:
                print("Cut combinations", len(all_combinations))
                all_combinations = sample(all_combinations, 4000)
            all_combinations = _get_unique_combinations(all_combinations, resources_required)
            if len(all_combinations) > 1000:
                print("Cut combinations II", len(all_combinations))
                all_combinations = all_combinations[:1000]

            for goal_et, proposals in goals_process_executions_variants.items():
                amount_required = entity_types_needed.count(goal_et)
                if  1 < amount_required:
                    for idx, proposal in enumerate(proposals):
                        proposal.cfp_path = proposal.cfp_path[:-1] + [proposal.cfp_path[-1] * (idx + 1)]
        idx = 0
        for possible_combination in all_combinations:
            origin, destination = _determine_origin_destination(possible_combination, process_execution)

            possible_combination, combi_preference, feasible = (
                _merge_combi_preference_resource(preference, possible_combination))

            if feasible:
                # position is different to origin/ destination resource
                # (for example non stationary resources can have stationary resources as positions)
                origin_position, destination_position = \
                    _determine_positions(possible_combination, origin, destination, process_execution.process)

                if not isinstance(origin, NonStationaryResource):
                    location_of_operation = origin
                else:
                    location_of_operation = destination
                possible_origins.setdefault(location_of_operation,
                                            []).append(origin_position)
                possible_destinations.setdefault(location_of_operation,
                                                 []).append(destination_position)
                # usable_resource_components.setdefault(location_of_operation,
                #                                                       {})[idx] = possible_combination

                usable_resource_components.setdefault(location_of_operation,
                                                      []).extend(possible_combination)
                usable_preferences.setdefault(location_of_operation,
                                              []).append(combi_preference)
                usable = True
                idx += 1

            else:
                unusable_resource_components.extend(possible_combination)

        if usable:
            relevant_goals = relevant_goals.union(entity_types_needed)

    return (usable_resource_components, usable_preferences, possible_origins, possible_destinations,
            unusable_resource_components, relevant_goals)


def _determine_origin_destination(possible_combination, process_execution) -> tuple[StationaryResource | None,
                                                                                    StationaryResource | None]:
    # determine origin and destination
    possible_origin_resources = process_execution.process.get_possible_origins()
    possible_destination_resources = process_execution.process.get_possible_destinations()

    available_resources = [resource_component.get_entity_used()
                           for resource_component in possible_combination]

    origin, destination = _match_origin_destination(available_resources,
                                                    possible_origin_resources, possible_destination_resources)

    return origin, destination


def _match_origin_destination(available_resources, possible_origin_resources, possible_destination_resources):
    assigned_possible_origin_resources = \
        [resource for resource in available_resources if resource in possible_origin_resources]
    assigned_possible_destination_resources = \
        [resource for resource in available_resources if resource in possible_destination_resources]

    if len(assigned_possible_origin_resources) == 1:
        origin = assigned_possible_origin_resources[0]
    else:
        origin = None

    if len(assigned_possible_destination_resources) == 1:
        destination = assigned_possible_destination_resources[0]
    else:
        destination = None

    return origin, destination


def _determine_unusable_resource_proposals(goals_process_executions_variants, relevant_goals,
                                           usable_resource_components, unusable_resource_components):
    irrelevant_goals = list(set(goals_process_executions_variants.keys()).difference(relevant_goals))
    unusable_resource_components += [process_executions_path
                                     for irrelevant_goal in irrelevant_goals
                                     for process_executions_path in goals_process_executions_variants[irrelevant_goal]]

    if usable_resource_components.values():
        # usable_resource_components_not_nested = reduce(concat,
        #                                                        [e
        #                                                         for d in list(usable_resource_components.values())
        #                                                         for e in d.values()])
        usable_resource_components_not_nested = reduce(concat, usable_resource_components.values())
    else:
        usable_resource_components_not_nested = []

    unusable_resource_proposals = [process_executions_path_proposal
                                   for process_executions_path_proposal in unusable_resource_components
                                   if process_executions_path_proposal not in usable_resource_components_not_nested]

    return unusable_resource_proposals

def _get_unique_combinations(all_combinations, resources_required):
    all_combinations = [combi
                        for combi in all_combinations
                        if len(set(combi)) == resources_required]
    all_combi_ints = {idx: set([p.identification
                                for p in combi])
                      for idx, combi in enumerate(all_combinations)}

    unique_sets_with_keys = {}
    for idx, identifications in all_combi_ints.items():
        frozen_set = frozenset(identifications)  # Verwandle das Set in ein frozenset
        if frozen_set not in unique_sets_with_keys:
            unique_sets_with_keys[frozen_set] = idx  # Speichere den Schlüssel (Index)
    all_combinations = [all_combinations[idx]
                        for idx in list(unique_sets_with_keys.values())]

    return all_combinations


def _merge_combi_preference_resource(preference, possible_combination):
    """Merge a combi preference vertical
    Example given: the work station and the worker want to execute at the same time a process
    Therefore they should have time at the same time"""

    combi_preference = preference.get_copy()

    process_executions_preferences = [process_executions_component.reference_preference.get_copy()
                                      for process_executions_component in possible_combination]
    combi_preference.merge_vertical(process_executions_preferences=process_executions_preferences)

    feasible = combi_preference.feasible()  # here also the accepted_time_period is updated

    return possible_combination, combi_preference, feasible


def _derive_part_demands(process_execution: ProcessExecution):
    """Derive the part demands for the given process execution."""
    part_entity_types_needed = process_execution.get_part_entity_types_needed()
    return part_entity_types_needed


def _get_entity_types_storable(process_execution, support_entity_type, needed_resource_entity_type,
                               entity_types_storable):
    """
    entities types storable are specified to ensure that only the resources are used that are capable to transport
    for example a part (support entity_type) or the part entity_type is storable in the storage
    Note: The other solution for the bicycle world would be to create a resource model for each material unloading.
    Not possible with the Schmaus case, because the box/ support is stored into the storage (destination)
    and not a part main part.
    """

    if isinstance(process_execution.destination, Storage):
        destination = process_execution.destination
        destination_entity_type = destination.entity_type
        if needed_resource_entity_type.check_entity_type_match(destination_entity_type):
            entity_types_storable_resource_request = entity_types_storable.copy()
            return entity_types_storable_resource_request

    if support_entity_type is None:
        entity_types_storable_resource_request = []
    elif needed_resource_entity_type.check_entity_type_match(support_entity_type):
        entity_types_storable_resource_request = entity_types_storable.copy()
    else:
        entity_types_storable_resource_request = []

    return entity_types_storable_resource_request


class ProcessRequest(PlanningRequest):
    """
    The process request is responsible for the organization of all resources and parts needed to fulfill
    the process execution.
    The request is triggered by the method process_request.
    See method description for understanding the behaviour
    """

    def __init__(self):
        super(ProcessRequest, self).__init__()
        self.safety_time_buffer = 10

    async def process_request(self, negotiation_object_request: ProcessCallForProposal):
        """
        The process_request contains two main steps that contain similar steps:
        1. Resource organization
            - resource demand is determined and requested
            - the resources available are added to a process_execution_component
        2. Part organization ...
            - the parts demand is determined and requested
            - the parts available are added to a process_execution_component ...
        :param negotiation_object_request: the negotiation_object_request
        :return: process_executions_components (/ input for the proposals)
        """

        reference_cfp, client_object, order, preference_requester, request_type, issue_id, \
            process_execution, fixed_origin, long_time_reservation, node_path = negotiation_object_request.unpack()
        request_id = negotiation_object_request.identification

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Client {client_object.identification} start ProcessRequest with ID: {request_id} "
        #       f"process name: {process_execution.process.name}")

        resource_model_complete, resource_model_demands = _derive_resource_demands(process_execution)

        if not resource_model_complete:
            process_executions_variants, sub_proposals, proposals_to_reject_resource = \
                await self._organize_resource_requirements(resource_model_demands=resource_model_demands,
                                                           process_execution=process_execution,
                                                           preference=preference_requester,
                                                           fixed_origin=fixed_origin,
                                                           long_time_reservation=long_time_reservation,
                                                           cfp=negotiation_object_request, order=order,
                                                           issue_id=issue_id, node_path=node_path,
                                                           request_id=request_id)
            resources_successful = bool(process_executions_variants)
        else:
            raise Exception  # in current implementation no use case given

        part_entity_types_needed = _derive_part_demands(process_execution)

        if part_entity_types_needed and process_executions_variants:
            process_executions_variants, sub_proposals, proposals_to_reject_part = \
                await self._organize_part_requirements(part_entity_types_needed=part_entity_types_needed,
                                                       process_execution=process_execution,
                                                       process_executions_variants=process_executions_variants,
                                                       reference_cfp=negotiation_object_request,
                                                       order=order, issue_id=issue_id,
                                                       preference_requester=preference_requester,
                                                       sub_proposals=sub_proposals, node_path=node_path,
                                                       request_id=request_id)
            if sub_proposals and not process_executions_variants:
                # reject also the other proposals
                if sub_proposals.values():
                    proposals_to_reject_resource += reduce(concat, sub_proposals.values())

        else:
            proposals_to_reject_part = []

        if not process_executions_variants:
            pass
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Not successful Request ID: {request_id} and process name '{process_execution.get_name()}' and "
            #       f"{resources_successful} and part_entity_types_needed "
            #       f"{[et.name for et, number in part_entity_types_needed]}")
        else:
            pass
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Successful Request ID: {request_id} and process name '{process_execution.get_name()}'")
            # # len(unused_process_executions_components), len(used_process_executions_variants))

        successful, proposals_to_reject = \
            self._get_summarize(process_executions_variants, proposals_to_reject_resource, proposals_to_reject_part)
        # print("Request ID", request_id)
        # for process_executions_variant in process_executions_variants:
        #     print("Node ID", process_executions_variant.node_identification,
        #           process_executions_variant.goal.process.name,
        #           process_executions_variant.cfp_path)
        #     print([(component.goal.name, component)
        #            for component in process_executions_variant.get_process_executions_components_lst()])
        #     for process_executions_component in process_executions_variant.get_process_executions_components_lst():
        #         print("process_executions_component", process_executions_component.cfp_path,
        #               [(path.process_executions_components_parent.cfp_path, path.cfp_path, path.node_identification)
        #                for path in process_executions_component.get_process_executions_components_lst()])
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" End ProcessRequest with ID {request_id} {len(process_executions_variants)}, "
        #       f"{len(proposals_to_reject)}")

        # for process_executions_variant in process_executions_variants:
        #     for goal, peps in process_executions_variant.process_executions_components.items():
        #         for pep in peps:
        #             print(goal.name, pep.goal_item.name, pep.get_accepted_time_periods())

        return successful, process_executions_variants, proposals_to_reject

    async def _organize_resource_requirements(self, resource_model_demands, process_execution, preference, fixed_origin,
                                              long_time_reservation, cfp, order, issue_id, node_path, request_id):
        """Request the resource requirements and build combinations of the proposals gotten"""

        resource_proposals = \
            await self._request_resources(resource_model_demands, process_execution, preference, fixed_origin,
                                          long_time_reservation, cfp, order, issue_id, node_path, request_id)

        usable_resource_components, usable_preferences, possible_origins, possible_destinations, proposals_to_reject = \
            _combine_resources(resource_model_demands, preference, resource_proposals, process_execution)

        process_executions_variants, sub_proposals = \
            self._build_process_executions_variants_resources(cfp, usable_resource_components, usable_preferences,
                                                              possible_origins, possible_destinations,
                                                              process_execution, issue_id, node_path)

        # if not process_executions_variants:
        #      print("Resource request failed:", [proposal.goal.name for proposal, provider in resource_proposals])

        return process_executions_variants, sub_proposals, proposals_to_reject

    async def _request_resources(self, resource_model_demands, process_execution, preference, fixed_origin,
                                 long_time_reservation, reference_cfp, order, issue_id, node_path, request_id):
        """Request the resources needed"""

        locations_of_demand = _get_locations_of_demand(process_execution)
        # ToDo: Schmaus compatibility
        # entity_types_storable = []
        if process_execution.get_parts():
            entity_types_storable = [process_execution.get_parts()[0].entity_type]  # main part/ support -> ...
        else:
            entity_types_storable = []
        support_entity_type = process_execution.get_support_entity_type()
        # entity_types_storable.append(support_entity_type)

        preference = self._check_time_needed(process_execution, preference)

        negotiation_object_identifications = []
        provider_negotiation_objects = {}
        for resource_model, needed_resource_entity_type_lst in resource_model_demands.items():
            for needed_resource_entity_type in set(needed_resource_entity_type_lst):  # ToDo: if a resource is needed twice
                requested_entity_types = [(needed_resource_entity_type, 1)]

                entity_types_storable_resource_request = \
                    _get_entity_types_storable(process_execution, support_entity_type, needed_resource_entity_type,
                                               entity_types_storable)

                long_time_reservation_single = _get_long_time_reservation(long_time_reservation,
                                                                          needed_resource_entity_type)

                negotiation_object = \
                    ResourceCallForProposal(reference_cfp=reference_cfp, sender_name=self.agent.name,
                                            request_type=CallForProposal.RequestType.FLEXIBLE,
                                            client_object=process_execution, order=order,
                                            preference=preference.get_copy(), issue_id=issue_id,
                                            fixed_origin=fixed_origin, locations_of_demand=locations_of_demand,
                                            requested_entity_types=requested_entity_types,
                                            entity_types_storable=entity_types_storable_resource_request,
                                            long_time_reservation=long_time_reservation_single,
                                            node_path=node_path)
                try:
                    providers = [self.agent.address_book[needed_resource_entity_type]]
                except:
                    print(needed_resource_entity_type)
                    raise Exception(needed_resource_entity_type)
                for provider in providers:
                    provider_negotiation_objects.setdefault(provider,
                                                            []).append(negotiation_object)

                negotiation_object_identifications.append(negotiation_object.identification)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" resource Requested ID: {request_id}")
        for provider, negotiation_objects in provider_negotiation_objects.items():
            await self.agent.NegotiationBehaviour.call_for_proposal(negotiation_objects, [provider])

        proposals = await self.agent.NegotiationBehaviour.await_callback(negotiation_object_identifications, "process")

        return proposals

    def _check_time_needed(self, process_execution: ProcessExecution, preference):
        """Determine the time needed if no main_resource/ resources provided in the process_execution to avoid a later
        update of the process_execution_time"""
        if process_execution.main_resource:  # no need for the checking of time differences
            return preference

        possible_resource_models = process_execution.get_possible_resource_groups()
        for resource_model in possible_resource_models:
            main_resource_entity_type = resource_model.main_resources[0]
            if main_resource_entity_type not in self.agent.resources:
                continue

            main_resources = self.agent.resources[main_resource_entity_type]
            if len(main_resources) != 1:
                continue

            main_resource = main_resources[0]
            distance = self.get_distance(process_execution)
            process_execution_time_informed = \
                int(np.ceil(round(process_execution.get_max_process_time(main_resource=main_resource,
                                                                         distance=distance), 1)))
            if preference.expected_process_execution_time != process_execution_time_informed:
                preference.expected_process_execution_time = process_execution_time_informed

            break  # currently only one resource_model is considered

        return preference

    def get_distance(self, process_execution):

        if isinstance(process_execution.main_resource, ConveyorBelt):
            if process_execution.origin is None and process_execution.destination is None:
                distance = process_execution.main_resource.conveyor_length
            else:
                if process_execution.origin == process_execution.main_resource.origin and \
                        process_execution.destination == process_execution.main_resource.destination:
                    distance = process_execution.main_resource.conveyor_length
                else:
                    distance = process_execution.get_distance()
        else:
            distance = process_execution.get_distance()

        return distance

    def _build_process_executions_variants_resources(self, cfp, usable_resource_components, usable_preferences,
                                                     possible_origins, possible_destinations,
                                                     process_execution: ProcessExecution, issue_id, node_path):
        """Build process_executions_variants that combine the different resources and their time_slots"""

        process_executions_variants = \
            [self._get_variant_resources(cfp, process_execution, list(set(usable_components)), usable_preferences,
                                         location_of_operation, issue_id, possible_origins, possible_destinations,
                                         node_path)
             for location_of_operation, usable_components in usable_resource_components.items()]

        sub_proposals = {}
        for process_executions_variant in process_executions_variants:
            resource_process_executions_components = process_executions_variant.get_process_executions_components_lst()
            sub_proposals[process_executions_variant] = resource_process_executions_components

        return process_executions_variants, sub_proposals

    def _get_variant_resources(self, cfp, process_execution, usable_components, usable_preferences,
                               location_of_operation, issue_id, possible_origins, possible_destinations, node_path):

        process_executions_paths = {}
        for usable_component in usable_components:
            if usable_component not in process_executions_paths.setdefault(usable_component.goal, []):
                process_executions_paths.setdefault(usable_component.goal,
                                                    []).append(usable_component)

        # ToDo: update of the process_execution_time (maybe only later in the scheduling procedure)
        # process_executions_components_times = \
        #     list(set([process_executions_component.get_process_execution_time()
        #               for process_executions_component in process_executions_paths]))
        # update_needed, expected_process_time_combined = \
        #     self._determine_time_update_needed(updated_process_execution, process_executions_components_times)

        # if update_needed:
        #     raise Exception

        prefs = usable_preferences[location_of_operation]
        all_periods = np.concatenate([pref.get_accepted_time_periods() for pref in prefs])
        min_value, max_value = all_periods.min(), all_periods.max()
        combi_preferences = prefs[0]
        combi_preferences._accepted_time_periods = np.array([[min_value, max_value]])

        possible_origins_set = set(possible_origins[location_of_operation])
        possible_destinations_set = set(possible_destinations[location_of_operation])
        if len(possible_origins_set) != 1:
            possible_origins_set = set(resource.situated_in if resource.situated_in
                                       else resource
                                       for resource in list(possible_origins_set))
            if len(possible_origins_set) != 1:
                raise Exception([(resource.name, resource, resource.situated_in)
                                 for resource in list(possible_origins_set)])
        if len(possible_destinations_set) != 1:
            possible_destinations_set = set(resource.situated_in if resource.situated_in
                                            else resource
                                            for resource in list(possible_destinations_set))

            if len(possible_destinations_set) != 1:
                raise Exception([(resource.name, resource, resource.situated_in)
                                 for resource in list(possible_destinations_set)])
        origin = list(possible_origins_set)[0]
        destination = list(possible_destinations_set)[0]

        process_executions_variant = \
            ProcessExecutionsVariantProposal(call_for_proposal=cfp, provider=self.agent.name,
                                             issue_id=issue_id, process_execution_id=process_execution.identification,
                                             goal=process_execution, goal_item=process_execution.duplicate(),
                                             origin=origin, destination=destination,
                                             process_executions_components=process_executions_paths,
                                             reference_preference=combi_preferences, cfp_path=node_path)

        predecessors = [process_execution_variant
                        for process_execution_paths_unmapped in list(process_executions_paths.values())
                        for process_execution_path in process_execution_paths_unmapped
                        for process_execution_variant in process_execution_path.get_process_executions_components_lst()]
        process_executions_variant.predecessors.extend(predecessors)

        return process_executions_variant

    def _set_main_resource(self, process_execution, updated_process_execution, feasible_combination):
        resources_expansion = [process_execution_path.goal_item
                               for process_execution_path in feasible_combination]
        updated_process_execution.resources_used += [(resource,) for resource in resources_expansion]

        if not process_execution.main_resource:
            possible_main_resources = \
                [resource for resource in resources_expansion
                 if process_execution.process.check_ability_to_perform_process_as_main_resource(resource)]

            if len(possible_main_resources) >= 1:
                updated_process_execution.main_resource = possible_main_resources[0]
            else:
                raise Exception

        return updated_process_execution

    def _determine_time_update_needed(self, updated_process_execution, process_executions_components_times):
        """
        Determine if the process_executions_components which enter the ProcessExecutionsVariant as sub-nodes have
        different process_times as in combination. This occurs if the efficiency or speed differs to 1.
        :return update_needed: a bool value that states if the time update is needed
        :return expected_process_time: the new expected_process_time
        """
        distance = self.get_distance(updated_process_execution)
        expected_process_time_combined = updated_process_execution.get_expected_process_time(distance=distance)
        if process_executions_components_times[0] != int(np.ceil(round(expected_process_time_combined, 1))) or \
                len(process_executions_components_times) > 1:
            update_needed = True
        else:
            update_needed = False

        return update_needed, expected_process_time_combined

    async def _organize_part_requirements(self, part_entity_types_needed, process_execution,
                                          process_executions_variants: list[ProcessExecutionsVariant],
                                          reference_cfp, order, issue_id, preference_requester, sub_proposals,
                                          node_path, request_id):
        """Request the part requirements needed to execute the process and integrate them if given
        into the process_executions_variant."""

        # t1 = time.process_time()

        part_proposals = \
            await self._request_parts(part_entity_types_needed=part_entity_types_needed,
                                      process_execution=process_execution,
                                      process_executions_variants=process_executions_variants,
                                      reference_cfp=reference_cfp,
                                      order=order, issue_id=issue_id,
                                      preference_requester=preference_requester, node_path=node_path,
                                      request_id=request_id)

        # t2 = time.process_time()
        # print("Part proposals:", request_id, part_proposals)
        process_executions_variants, proposals_to_reject, sub_proposals = \
            self._integrate_parts_in_paths(process_executions_variants=process_executions_variants,
                                           part_entity_types_needed=part_entity_types_needed,
                                           part_proposals=part_proposals,
                                           sub_proposals=sub_proposals)

        # t3 = time.process_time()

        # print("Process interim: ", t3 - t2, t2 - t1)

        return process_executions_variants, sub_proposals, proposals_to_reject

    async def _request_parts(self, part_entity_types_needed, process_execution, process_executions_variants,
                             reference_cfp, order, preference_requester, issue_id, node_path, request_id):
        """Request the parts from the part provider"""

        negotiation_object_identifications = []
        provider_negotiation_objects = {}
        for process_executions_variant in process_executions_variants:
            location_of_demand = process_executions_variant.origin
            period_of_demand = process_executions_variant.reference_preference.get_accepted_time_periods().copy()
            period_of_demand[-1][-1] -= process_executions_variant.reference_preference.expected_process_execution_time

            # amount_of_parts_needed = sum(list(zip(*part_entity_types_needed))[1])
            # additional_time_period_provided = period_of_demand[0][1] - period_of_demand[0][0]
            # safety_time_buffer = self.safety_time_buffer * int(amount_of_parts_needed)
            # if amount_of_parts_needed > 1:
            #     safety_time_buffer += additional_time_period_provided.item().seconds
            #     period_of_demand[0][0] -= safety_time_buffer

            if not (isinstance(process_executions_variant.destination, NonStationaryResource) and
                    isinstance(process_executions_variant.destination, ConveyorBelt)):
                # not a loading_process
                period_of_demand[0][0] = self.agent.change_handler.get_current_time()  # assuming always smaller

            part_preference, successful = (
                preference_requester.get_predecessor_preference(
                    process_execution=process_execution, origin=None, destination=location_of_demand,
                    period_of_demand=period_of_demand))

            if not successful:
                continue

            # create for each part needed at the location of demand a request object
            for part_entity_type_needed, amount in part_entity_types_needed:
                try:
                    providers = (self.agent.entity_provider[part_entity_type_needed] +
                                 self.agent._resources_without_storages)
                except:
                    print("Not in self.agent.entity_provider", part_entity_type_needed.name)

                for i in range(int(amount)):
                    if i == 1:
                        part_preference = part_preference.get_copy()
                    requested_entity_types = [(part_entity_type_needed, 1)]

                    part_call_for_proposal = \
                        PartCallForProposal(reference_cfp=reference_cfp, sender_name=self.agent.name,
                                            request_type=CallForProposal.RequestType.FLEXIBLE,
                                            client_object=process_execution, order=order, issue_id=issue_id,
                                            preference=part_preference, locations_of_demand=[location_of_demand],
                                            requested_entity_types=requested_entity_types, node_path=node_path)

                    providers = self.agent.NegotiationBehaviour.convert_providers(providers)
                    for provider in providers:
                        provider_negotiation_objects.setdefault(provider,
                                                                []).append(part_call_for_proposal)

                    negotiation_object_identifications.append(part_call_for_proposal.identification)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" part Requested ID: {request_id}")
        for provider, negotiation_objects in provider_negotiation_objects.items():
            await self.agent.NegotiationBehaviour.call_for_proposal(negotiation_objects, [provider])

        proposals = await self.agent.NegotiationBehaviour.await_callback(negotiation_object_identifications, "process")

        return proposals

    def _integrate_parts_in_paths(self, process_executions_variants, part_entity_types_needed, part_proposals,
                                  sub_proposals):
        """Integrate the requested part proposal into the process variants to create process_proposals"""

        possible_combinations, proposals_to_reject = \
            self._determine_parts_usable(part_proposals, part_entity_types_needed)

        unused_process_executions_components, used_process_executions_variants, sub_proposals = \
            self._integrate_parts(process_executions_variants, part_proposals,
                                  possible_combinations, sub_proposals)

        if unused_process_executions_components:
            used_process_executions_components = \
                [process_executions_component
                 for process_executions_variant in process_executions_variants
                 for process_executions_component in process_executions_variant.get_process_executions_components_lst()]

            resource_proposals = [(proposal, proposal.provider)
                                  for unused_process_execution_component in unused_process_executions_components
                                  for proposal in sub_proposals[unused_process_execution_component]]

            proposals_to_reject_batch = get_proposals_to_reject(used_process_executions_components,
                                                                unused_process_executions_components,
                                                                resource_proposals)

            proposals_to_reject.extend(proposals_to_reject_batch)

        return used_process_executions_variants, proposals_to_reject, sub_proposals

    def _determine_parts_usable(self, part_proposals, part_entity_types_needed):
        """Determine for each destination (if destination known only one destination) possible combinations
        of material supply to deliver them"""

        goals_process_executions_variants = self._get_process_executions_variants_goals(part_proposals)
        # strict dependence on destination
        # if destination is unknown in advance (destination is specified with None), it can be used for all goals
        proposals_to_reject = []
        all_proposals_to_reject = False
        part_entity_types_needed_d = dict(part_entity_types_needed)

        location_of_demand_parts_available = {}
        for destination, goal_part_variants in goals_process_executions_variants.items():

            for goal, part_variants in goal_part_variants.items():
                if goal in part_entity_types_needed_d:
                    amount = part_entity_types_needed_d[goal]
                    if len(part_variants) < amount:
                        all_proposals_to_reject = True
                        proposals_to_reject = part_proposals  # ToDo for different locations
                        break

                else:
                    all_proposals_to_reject = True
                    proposals_to_reject = part_proposals  # ToDo for different locations
                    break

                if goal_part_variants:
                    break

            if all_proposals_to_reject:
                proposals_to_reject = part_proposals
                break

            location_of_demand_parts_available[destination] = goal_part_variants

        if all_proposals_to_reject:
            proposals_to_reject = [proposal for proposal, provider in part_proposals]

        return location_of_demand_parts_available, proposals_to_reject

    def _get_process_executions_variants_goals(self, part_proposals):
        """Assign the process_executions_variants from proposals to the goals and destinations
        (if more than one possible destination it is needed)"""

        goals_process_executions_variants = {}
        for proposal, provider in part_proposals:

            destination = proposal.get_destination()
            if isinstance(destination, Storage) and destination.situated_in is not None:
                destination = destination.situated_in

            goal = proposal.get_goal()
            if destination not in goals_process_executions_variants:
                goals_process_executions_variants[destination] = {}

            goals_process_executions_variants[destination].setdefault(goal, []).append(proposal)

        if len(goals_process_executions_variants) >= 2 and None in goals_process_executions_variants:
            raise NotImplementedError  # if destination is None - it can be used for all destinations

        return goals_process_executions_variants

    def _integrate_parts(self, process_executions_variants, part_proposals, location_part_components, sub_proposals):
        """ToDo: comment"""

        unused_process_executions_components = {}
        used_process_executions_variants = []
        for process_executions_variant in process_executions_variants:

            if process_executions_variant.origin not in location_part_components:
                unused_process_executions_components[process_executions_variant] = \
                    sub_proposals[process_executions_variant]
                continue

            else:
                used_process_executions_variants.append(process_executions_variant)

            part_components: dict[EntityType, list[ProcessExecutionsPath]] = \
                location_part_components[process_executions_variant.origin]

            sub_proposals = self._integrate_parts_location(part_components, process_executions_variant, sub_proposals)

        return unused_process_executions_components, used_process_executions_variants, sub_proposals

    def _integrate_parts_location(self, part_components, process_executions_variant, sub_proposals):
        """Integrate the process_executions_components needed for the parts"""
        # merge the process_execution_components on the preference for each combination
        # store the combi - also the components
        sub_proposals_variant = reduce(concat, list(part_components.values()))
        sub_proposals[process_executions_variant].extend(sub_proposals_variant)

        process_executions_variant.add_precondition_process_executions_components(part_components)

        # ToDo: also needed!

        # predecessors = \
        #     [process_execution_variant
        #      for process_execution_path in reduce(concat, list(process_executions_variant.process_executions_components.values()))
        #      for process_execution_variant in reduce(concat, list(process_execution_path.process_executions_components.values()))]  # ToDo

        # process_executions_variant.predecessors.extend(predecessors)

        return sub_proposals

    def _get_summarize(self, process_executions_variants, proposals_to_reject_resource, proposals_to_reject_part):
        proposals_to_reject = proposals_to_reject_resource + proposals_to_reject_part
        if process_executions_variants:
            successful = True
        else:
            successful = False

        return successful, proposals_to_reject

