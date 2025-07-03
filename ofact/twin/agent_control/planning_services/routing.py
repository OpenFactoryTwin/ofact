"""
Used to determine transport transport_routes between two resources (origin and destination)
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from copy import copy
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import (ConveyorBelt, ActiveMovingResource, Storage, NonStationaryResource,
                                             Warehouse)
from ofact.twin.state_model.processes import _determine_intersection

# Imports Part 2: PIP Imports

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import EntityType
    from ofact.twin.state_model.processes import Process
    from ofact.twin.state_model.process_models import ResourceGroup

MEMOIZATION_MAX = 100


def memoize_transit_processes(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_transit_processes_with_resources(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def _determine_main_resource(resources_used, resource_group) -> EntityType:
    main_resource = \
        [resource
         for resource in resources_used
         if resource_group.check_ability_to_perform_process_as_main_resource(resource)][0]

    return main_resource


def _get_process_dict(process, destination, support_entity_type, entity_entity_type, transport=True):
    """

    :param transport: determines if the support entity type serves as resource that transport or as resource that
    """
    if support_entity_type or entity_entity_type:
        process_support_entity_types = process.get_necessary_input_entity_types()
        if not process_support_entity_types:  # Why needed
            return None

        if support_entity_type:
            same_support = [et for et, amount in process.get_necessary_input_entity_types()
                            if et.check_entity_type_match_lower(support_entity_type)]
            if not same_support:
                return None

        if entity_entity_type:
            same_entity = [et for et, amount in process.get_necessary_input_entity_types()
                           if et.check_entity_type_match_lower(entity_entity_type)]
            if not same_entity:
                return None

    if destination is not None:
        if destination not in process.get_possible_destinations():
            return None

    if not transport and len(process.get_necessary_input_entity_types()) > 1:
        return None

    transport_origins = \
        [origin for origin in process.get_possible_origins()
         if isinstance(destination, ConveyorBelt) or isinstance(destination, ActiveMovingResource)]
    if transport_origins:
        return None

    transition_dict = {"process": process,
                       "origin": None,
                       "destination": destination}

    return transition_dict


def get_complete_routes(process_routes, origin, support_entity_type):
    # ToDo: probably not needed - same questions before

    complete_routes = []

    if not support_entity_type:
        for process_route in process_routes:
            if origin not in process_route[0]["process"].get_possible_origins():
                continue
            complete_routes.append(process_route)

    else:
        for process_route in process_routes:
            process_support_entity_types = process_route[0]["process"].get_necessary_input_entity_types()
            if not process_support_entity_types:
                continue

            if origin in process_route[0]["process"].get_possible_origins() and \
                    [et for et, amount in process_route[0]["process"].get_necessary_input_entity_types()
                     if support_entity_type.check_entity_type_match(et)]:
                complete_routes.append(process_route)

    return complete_routes


def _transit_match(process, entity_entity_type, support_entity_type, origin, destination,
                   level_differences_allowed=False):
    """
    :param level_differences_allowed: means that from the origin and destination also the situated_in level is checked
    """
    if entity_entity_type is not None:
        part_et_match = [entity_entity_type
                         for et in process.get_input_entity_types_set()
                         if entity_entity_type.check_entity_type_match(et)]

        if not part_et_match:
            return
    possible_origins = process.get_possible_origins()

    if origin is not None:
        if origin not in possible_origins:
            if not level_differences_allowed:
                return

            possible_sub_origins = \
                _sub_resources_in_resources(resource=origin,
                                            possible_sub_resources_to_compare=possible_origins)
            if not possible_sub_origins:
                return

        if destination is None and support_entity_type is not None:
            possible_destinations = process.get_possible_destinations()
            if possible_destinations:
                support_et_match = [entity_entity_type
                                    for resource in possible_destinations
                                    if support_entity_type.check_entity_type_match_lower(resource.entity_type)]
                if not support_et_match:
                    return

    possible_destinations = process.get_possible_destinations()
    if destination is not None:
        if destination not in possible_destinations:
            if not level_differences_allowed:
                return

            possible_sub_destinations = \
                _sub_resources_in_resources(resource=destination,
                                            possible_sub_resources_to_compare=possible_destinations)
            if not possible_sub_destinations:
                return

        if origin is None and support_entity_type is not None:
            possible_origins = process.get_possible_origins()
            if possible_origins:
                support_et_match = [entity_entity_type
                                    for resource in possible_origins
                                    if support_entity_type.check_entity_type_match_lower(resource.entity_type)]
                if not support_et_match:
                    return

    if possible_origins and possible_destinations:  # ensure that it is a transfer
        first_origin = possible_origins[0]
        first_destination = possible_destinations[0]
        if not (isinstance(first_origin, NonStationaryResource) or
                isinstance(first_destination, NonStationaryResource)):
            if not first_origin.check_intersection_base_areas(first_destination):
                return

    return process


def _sub_resources_in_resources(resource, possible_sub_resources_to_compare, entity_type_to_store=None):
    """Check if a sub resource is in the possible resources."""
    possible_sub_resources = [sub_resource
                              for sub_resources in list(resource.get_storages().values())
                              for sub_resource in sub_resources
                              if _check_possible(sub_resource, possible_sub_resources_to_compare, entity_type_to_store)]
    return possible_sub_resources


def _check_possible(sub_resource, possible_sub_resources_to_compare, entity_type_to_store):
    """Check if a sub resource is in the possible resources (in buffer_stations etc.)"""
    if entity_type_to_store:
        storable = sub_resource.check_entity_type_storable(entity_type_to_store)
        if not storable:
            return False

    if sub_resource in possible_sub_resources_to_compare:
        return True
    else:
        return False


def _get_origin_destination(origin, destination):
    """Choose the right aggregation level for the origin and destination"""

    if isinstance(origin, ConveyorBelt) or isinstance(origin, ConveyorBelt):
        # if origin or destination are conveyor belts the other one should be a storage (or a work station ???)
        return origin, destination

    elif isinstance(origin, NonStationaryResource) or isinstance(origin, NonStationaryResource):
        # if origin or destination is a non_stationary_resource the other one should be a storage
        # loading process
        return origin, destination

    else:
        # if the origin or the destination are storages they are adapted to be a higher level
        # transport process
        if isinstance(origin, Storage):
            if origin.situated_in:
                origin = origin.situated_in
        if isinstance(destination, Storage):
            if destination.situated_in:
                destination = destination.situated_in

    return origin, destination


def _check_connector(process: Process):
    """Check if the process is usable as a connector like transport access with no material transported"""
    if len(process.transformation_controller.get_root_nodes()) > 1:
        return False
    else:
        return True


def _raise_max_iteration_error(origin, destination, support_entity_type, entity_entity_type, iteration):
    """Max iteration amount is reached"""

    origin_str = origin.name if origin else None
    destination_str = destination.name if destination else None
    if support_entity_type:
        support_entity_type_str = support_entity_type.name
    else:
        support_entity_type_str = None
    if entity_entity_type:
        entity_entity_type_str = entity_entity_type.name
    else:
        entity_entity_type_str = None

    raise RuntimeError(f"Transition not found in {iteration} iterations. "
                       f"From origin {origin_str} to destination {destination_str} through "
                       f"{support_entity_type_str} with {entity_entity_type_str}")


class RoutingService:

    def __init__(self, resource_types, possible_transport_transfer_processes):
        self.resource_types = resource_types
        self.possible_transport_transfer_processes = possible_transport_transfer_processes

    @memoize_transit_processes
    def get_transit_processes(self, origin, destination, support_entity_type=None, entity_entity_type=None,
                              transport=True, transfers=False, solution_required=True):
        """
        Get transition processes for the transition from origin to destination with a transport resource that has
        the entity_type support_entity_type and can transport the entity_entity_type
        :param solution_required: raise no error if no solution is found (false case)
        """
        transport_transfer_processes = self._determine_transport_processes(origin, destination,
                                                                           support_entity_type, entity_entity_type,
                                                                           transport, transfers,
                                                                           solution_required=solution_required)

        return transport_transfer_processes

    @memoize_transit_processes_with_resources
    def get_transit_processes_with_resources(self, origin, destination, support_entity_type=None,
                                             entity_entity_type=None, input_resources=[], transport=True,
                                             solution_required=True):
        """
        :param solution_required: raise no error if no solution is found (false case)
        """

        transport_transfer_processes = self._determine_transport_processes(origin, destination,
                                                                           support_entity_type, entity_entity_type,
                                                                           transport,
                                                                           solution_required=solution_required)

        transport_transfer_processes = self._determine_resources(transport_transfer_processes,
                                                                 input_resources=input_resources)

        return transport_transfer_processes

    def _determine_transport_processes(self, origin, destination, support_entity_type=None, entity_entity_type=None,
                                       transport=True, transfers=False, first_call=True, solution_required=True):
        """
        Determine a process chain for the transport processes
        :param transport: differentiates between transport (an entity is transported) and a transport_access
        (the entity drives by itself to a different position)
        :param solution_required: raise no error if no solution is found (false case)
        """

        original_origin = origin
        original_destination = destination
        if origin is None or destination is None:
            raise Exception("Origin and destination should be determined to create a route between them")

        origin, destination = _get_origin_destination(origin, destination)

        transport_transfer_processes = []
        if origin.identification == destination.identification:
            return transport_transfer_processes

        # starting from the destination
        processes_lead_to_destination = self._get_transit_processes(destination_s=destination,
                                                                    support_entity_type=support_entity_type,
                                                                    entity_entity_type=entity_entity_type,
                                                                    transport=transport)
        process_routes = [[end_process]
                          for end_process in processes_lead_to_destination]

        path_not_complete = True
        iteration = 0
        # search for a complete path
        while path_not_complete:
            complete_process_routes = get_complete_routes(process_routes, origin, support_entity_type)

            if complete_process_routes:
                transport_transfer_processes = complete_process_routes[0]
                transport_transfer_processes[0]["origin"] = origin
                break

            new_process_routes = self._extend_paths(process_routes, support_entity_type, entity_entity_type, transport)

            process_routes = copy(new_process_routes)

            iteration += 1

            if iteration >= 5:
                if support_entity_type and entity_entity_type is not None:
                    transport_transfer_processes_updated = (
                        self._determine_transport_processes(original_origin, original_destination,
                                                            support_entity_type=None,
                                                            entity_entity_type=support_entity_type,
                                                            transport=transport, transfers=transfers, first_call=False,
                                                            solution_required=solution_required))
                else:
                    transport_transfer_processes_updated = []

                if first_call:
                    return transport_transfer_processes_updated
                elif solution_required:
                    return transport_transfer_processes_updated
                else:
                    _raise_max_iteration_error(origin, destination, support_entity_type, entity_entity_type, iteration)

        transport_transfer_processes_updated = []
        for idx, transport_transfer_process in enumerate(transport_transfer_processes):
            if transport_transfer_process["origin"] is None:
                transport_transfer_process["origin"] = transport_transfer_processes[idx - 1]["origin"]
            transport_transfer_processes_updated.append(transport_transfer_process)

        # check if start_transfer needed
        first_process_d = transport_transfer_processes[0]
        intersection = _determine_intersection(first_process_d["origin"], first_process_d["destination"],
                                               class_name=self.__class__.__name__)
        if not intersection and not isinstance(first_process_d["destination"], NonStationaryResource) and transfers:
            transfer_process: None | dict = self.get_first_transfer(first_process_d, entity_entity_type,
                                                                    support_entity_type)
            if transfer_process is not None:
                transport_transfer_processes_updated.insert(0, transfer_process)

        # check if end_transfer needed
        last_process_d = transport_transfer_processes[-1]
        intersection = _determine_intersection(last_process_d["origin"], last_process_d["destination"],
                                               class_name=self.__class__.__name__)
        if not intersection and not isinstance(last_process_d["origin"], NonStationaryResource) and transfers:
            transfer_process: None | dict = self.get_last_transfer(last_process_d, entity_entity_type,
                                                                   support_entity_type)
            if transfer_process is not None:
                transport_transfer_processes_updated.append(transfer_process)

        return transport_transfer_processes_updated

    def _extend_paths(self, process_routes, support_entity_type, entity_entity_type, transport):
        new_process_routes = []
        for transport_process_lst in process_routes:
            # always take the first element - nearer to the origin
            destinations_resource = transport_process_lst[0]["process"].get_possible_origins()
            further_possible_transport_processes = \
                self._get_transit_processes(destination_s=destinations_resource,
                                            support_entity_type=support_entity_type,
                                            entity_entity_type=entity_entity_type, transport=transport)

            new_transport_paths = []
            for further_possible_transport_process in further_possible_transport_processes:
                if isinstance(further_possible_transport_process["destination"], Storage) or \
                        isinstance(further_possible_transport_process["destination"], Warehouse):
                    main_resources, resources_types = \
                        self.get_main_resources_from_process(further_possible_transport_process["process"])

                    if len(resources_types) > 1:
                        raise NotImplementedError

                    resource_type = resources_types[0]
                    if resource_type == ConveyorBelt:
                        if len(main_resources) > 1:
                            raise NotImplementedError

                        origin_resource = main_resources[0]
                        after_transfer_process = \
                            self.get_transfer_process(entity_entity_type=entity_entity_type,
                                                      origin=origin_resource,
                                                      destination=further_possible_transport_process["destination"],
                                                      support_entity_type=support_entity_type)

                        before_transfer_processes = \
                            self.get_transfer_processes(entity_entity_type=entity_entity_type,
                                                        destination=origin_resource,
                                                        level_differences_allowed=True)
                        transport_process_lst[0]["origin"] = further_possible_transport_process.copy()["destination"]

                        if len(transport_process_lst) == 1:  # ToDo: =
                            if isinstance(transport_process_lst[0]["destination"], Storage) or \
                                    isinstance(transport_process_lst[0]["destination"], Warehouse) and \
                                    transport_process_lst[0]["origin"] != transport_process_lst[0]["destination"]:
                                transport_process_lst = transport_process_lst.copy()
                                resources_successor, resources_types_successor = \
                                    self.get_main_resources_from_process(transport_process_lst[0]["process"])

                                origin_successor = transport_process_lst[0]["origin"]

                                before_transfer_processes_successor = \
                                    self.get_transfer_processes(support_entity_type=support_entity_type,
                                                                origin=origin_successor,
                                                                destination=resources_successor[0],
                                                                level_differences_allowed=True)

                                for before_transfer_process_successor in before_transfer_processes_successor:
                                    transport_process_lst.insert(0, before_transfer_process_successor)

                        for before_transfer_process in before_transfer_processes:  # ToDo: for loop does not make sense
                            transport_process_copy = further_possible_transport_process.copy()

                            chain_before = [before_transfer_process, transport_process_copy, after_transfer_process]
                            new_transport_paths.append(chain_before + transport_process_lst)
                        continue

                transport_process_lst[0] = transport_process_lst[0].copy()
                transport_process_lst[0]["origin"] = further_possible_transport_process["destination"]
                new_transport_paths.append([further_possible_transport_process] + transport_process_lst)

            new_process_routes += new_transport_paths

        return new_process_routes

    def get_last_transfer(self, transport_transfer_process_d, entity_entity_type, support_entity_type):
        main_resources, resources_types = \
            self.get_main_resources_from_process(transport_transfer_process_d["process"])

        if len(resources_types) > 1:
            raise NotImplementedError

        resource_type = resources_types[0]
        if resource_type == ConveyorBelt:
            if len(main_resources) > 1:
                raise NotImplementedError

            origin_resource = main_resources[0]

            after_transfer_process = \
                self.get_transfer_process(entity_entity_type=entity_entity_type,
                                          origin=origin_resource,
                                          destination=transport_transfer_process_d["destination"],
                                          support_entity_type=support_entity_type)

            return after_transfer_process

        return None

    def get_first_transfer(self, transport_transfer_process_d, entity_entity_type, support_entity_type):
        main_resources, resources_types = \
            self.get_main_resources_from_process(transport_transfer_process_d["process"])

        if len(resources_types) > 1:
            raise NotImplementedError

        resource_type = resources_types[0]
        if resource_type == ConveyorBelt:
            if len(main_resources) > 1:
                raise NotImplementedError

            destination_resource = main_resources[0]

            before_transfer_process = \
                self.get_transfer_process(entity_entity_type=entity_entity_type,
                                          origin=transport_transfer_process_d["origin"],
                                          destination=destination_resource,
                                          support_entity_type=support_entity_type)

            return before_transfer_process

        return None

    def get_main_resources_from_process(self, process):
        main_resource_entity_type = \
            process.get_resource_groups()[0].main_resources[0]
        resource_types_nested = [resource_type[main_resource_entity_type]
                                 for resource_type in self.resource_types
                                 if main_resource_entity_type in resource_type]

        main_resources = [resource for resource_lst in resource_types_nested for resource in resource_lst]

        resources_types = list(set(type(resource) for resource in main_resources))

        return main_resources, resources_types

    def _get_transit_processes(self, destination_s, support_entity_type, entity_entity_type, transport):
        """Determine processes that reach the destination_s"""

        if not isinstance(destination_s, list):
            destinations = [destination_s]
        else:
            destinations = destination_s

        transit_processes = \
            [_get_process_dict(process, destination, support_entity_type, entity_entity_type)
             for process in self.possible_transport_transfer_processes
             for destination in destinations
             if _get_process_dict(process, destination, support_entity_type, entity_entity_type) is not None]

        if not transport:  # transport access
            transit_processes = [process
                                 for process in transit_processes
                                 if _check_connector(process["process"])]

        return transit_processes

    def get_transfer_processes(self, entity_entity_type=None, origin=None, destination=None, support_entity_type=None,
                               level_differences_allowed=False):
        """Determine the transfer processes given the go from origin to destination"""

        if origin is None and destination is None:
            Exception("The determination of an transfer process needs at least an origin or a destination")

        possible_processes = [_transit_match(process, entity_entity_type, support_entity_type, origin, destination,
                                             level_differences_allowed)
                              for process in self.possible_transport_transfer_processes]

        possible_processes_set = list(filter(lambda item: item is not None, possible_processes))
        transfer_process_dicts = \
            [self._get_transfer_process_dict(transfer_process, origin, destination, entity_entity_type)
             for transfer_process in possible_processes_set]

        return transfer_process_dicts

    # ToDo: memoization
    def get_transfer_process(self, entity_entity_type=None, origin=None, destination=None, support_entity_type=None,
                             level_differences_allowed=False):
        """Determine the transfer processes given the go from origin to destination"""

        if origin is None and destination is None:
            Exception("The determination of an transfer process needs at least an origin or a destination")

        possible_processes = [_transit_match(process, entity_entity_type, support_entity_type, origin, destination,
                                             level_differences_allowed)
                              for process in self.possible_transport_transfer_processes]

        possible_processes_set = list(filter(lambda item: item is not None, possible_processes))

        if len(possible_processes_set) != 1:
            if origin is None:
                possible_processes_set = [process
                                          for process in possible_processes_set
                                          if not process.get_possible_origins()]

            if destination is None:
                possible_processes_set = [process
                                          for process in possible_processes_set
                                          if not process.get_possible_destinations()]

        if len(possible_processes_set) != 1:
            transfer_process_dict = {}
            return transfer_process_dict

        transfer_process = possible_processes_set[0]
        transfer_process_dict = self._get_transfer_process_dict(transfer_process, origin, destination,
                                                                entity_entity_type)

        return transfer_process_dict

    def _get_transfer_process_dict(self, transfer_process, origin, destination, entity_entity_type):
        if origin is not None:
            possible_origins = transfer_process.get_possible_origins()
            if origin not in possible_origins:
                possible_sub_origins = \
                    _sub_resources_in_resources(resource=origin,
                                                possible_sub_resources_to_compare=possible_origins,
                                                entity_type_to_store=entity_entity_type)
                if len(possible_sub_origins) == 1:
                    transfer_origin = possible_sub_origins[0]
                else:
                    transfer_origin = None
            else:
                transfer_origin = origin
        else:
            transfer_origin = None

        if destination is not None:
            possible_destinations = transfer_process.get_possible_destinations()
            if destination not in possible_destinations:
                possible_sub_destinations = \
                    _sub_resources_in_resources(resource=destination,
                                                possible_sub_resources_to_compare=possible_destinations,
                                                entity_type_to_store=entity_entity_type)
                if len(possible_sub_destinations) == 1:
                    transfer_destination = possible_sub_destinations[0]
                else:
                    transfer_destination = None
            else:
                transfer_destination = destination
        else:
            transfer_destination = None

        transfer_process_dict = {"process": transfer_process,
                                 "origin": transfer_origin,
                                 "destination": transfer_destination}

        return transfer_process_dict

    def _determine_resources(self, transport_transfer_processes, input_resources=[]):
        """Determine the resources for the processes"""

        for idx, transport_transfer_process in enumerate(transport_transfer_processes):
            resources = []
            longest_possible_resource_group = (None, resources)

            if not transport_transfer_process["process"].resource_controller:
                continue

            for resource_group in transport_transfer_process["process"].get_resource_groups():
                resource_group_entity_types = resource_group.resources
                for resource_entity_type in resource_group_entity_types:
                    # use the already determined resource's (origin/ destination) if possible

                    possible_input_resources = \
                        [input_resource
                         for input_resource in input_resources
                         if resource_entity_type.check_entity_type_match_lower(input_resource.entity_type)]

                    if possible_input_resources:
                        resources.append(possible_input_resources[0])
                    elif resource_entity_type.check_entity_type_match_lower(
                            transport_transfer_process["origin"].entity_type):
                        resources.append(transport_transfer_process["origin"])
                    elif resource_entity_type.check_entity_type_match_lower(
                            transport_transfer_process["destination"].entity_type):
                        resources.append(transport_transfer_process["destination"])
                    else:
                        if "resources_used" in transport_transfer_processes[idx - 1]:
                            resources_ = \
                                [resource for resource in transport_transfer_processes[idx - 1]["resources_used"]
                                 if resource.entity_type.check_entity_type_match_lower(resource_entity_type)]
                            if resources_:
                                resources.append(resources_[0])

                if len(list(set(resources))) == len(resource_group_entity_types):
                    # complete resource_group found
                    longest_possible_resource_group = (resource_group, resources)
                    break
                elif len(list(set(resources))) > len(resource_group_entity_types):
                    print(f"{datetime.now()} | [Routing service] Not handled until now")
                else:
                    if len(longest_possible_resource_group[1]) <= len(resources):
                        longest_possible_resource_group = (resource_group, resources)
                    resources = []

            transport_transfer_process["resources_used"] = longest_possible_resource_group[1]
            resource_group: ResourceGroup = longest_possible_resource_group[0]
            undetermined_entity_types = \
                resource_group.get_needed_resources(already_assigned_resources=longest_possible_resource_group[1])

            if undetermined_entity_types:
                if len(transport_transfer_process["process"].get_resource_groups()) == 0:
                    raise ValueError("Decision needed")

                for entity_type in undetermined_entity_types:
                    if "resources_used" in transport_transfer_processes[idx - 1]:
                        resources_ = \
                            [resource for resource in transport_transfer_processes[idx - 1]["resources_used"]
                             if entity_type.check_entity_type_match_lower(resource.entity_type)]
                    else:
                        resources_ = []

                    if resources_:
                        resource = resources_[0]

                        transport_transfer_process["resources_used"].append(resource)
                    else:
                        for resource_type in self.resource_types:
                            if entity_type in resource_type:
                                if len(resource_type[entity_type]) == 1:
                                    transport_transfer_process["resources_used"].append(resource_type[entity_type][0])
                                else:
                                    transport_transfer_process["resources_used"] += resource_type[entity_type]
                                break

            transport_transfer_process["main_resource"] = \
                _determine_main_resource(resources_used=transport_transfer_process["resources_used"],
                                         resource_group=resource_group)

        return transport_transfer_processes
