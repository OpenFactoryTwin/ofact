"""
The process_group request behaviour is used to create proposals for interconnected processes that should be executed in
a chronological order.
"""
# Imports Part 1: Standard Imports
import itertools
import time

# Imports Part 2: PIP Imports
import numpy as np
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.negotiation.objects import ProcessCallForProposal, \
    ProcessExecutionsPathProposal, ProcessExecutionsVariantProposal
from ofact.twin.agent_control.behaviours.planning.helpers import get_proposals_to_reject
from ofact.twin.agent_control.behaviours.planning.planning_request import PlanningRequest
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import \
    ProcessExecutionsPath
from ofact.twin.state_model.entities import NonStationaryResource

from ofact.twin.utils import setup_dual_logger
logging= setup_dual_logger()
def get_process_group_information(process_executions, part_entity_type, amount):
    process_times = \
        [process_execution.get_max_process_time(  # loading processes have a distance of 0
            distance=0 if idx == 0 or idx + 1 == len(process_executions) else None)
            for idx, process_execution in enumerate(process_executions)]

    # lead times relate to the start time of the process_execution
    lead_times = np.cumsum(process_times[::-1])[::-1].tolist()

    process_group_information_d = {"entity_type": part_entity_type,
                                   "process_executions": process_executions,
                                   "process_times": process_times,
                                   "lead_times": lead_times,
                                   "amount": amount}

    return process_group_information_d


def _create_process_executions_paths(process_executions_with_information, preference, cfp, node_path,
                                     process_execution_id, current_time, agent_name):
    """Create process_executions_paths to request the single components and
    ensure that they are suitable (they can be scheduled) for the path"""

    process_executions_paths = []
    process_executions_paths_connectors = {}
    for origin_resource, transport_process_d in process_executions_with_information.items():

        # for lead and follow_up time
        entity_types = [process_execution.get_all_entity_types_required()
                        for process_execution in transport_process_d["process_executions"]]

        if "pick_up_times" in transport_process_d:
            pick_up_time = transport_process_d["pick_up_times"][0]
        else:
            pick_up_time = current_time
        entity_types_connectors = list(set(entity_types[0]).intersection(*entity_types))

        # reference_preference determines the time_period when the process (from requester) want to start
        reference_preference = preference.get_copy()
        seconds_from_current_time = (preference.accepted_time_periods[0][0] - current_time).item().total_seconds()
        if seconds_from_current_time > int(transport_process_d["lead_times"][-1]):
            expansion_duration = int(transport_process_d["lead_times"][-1])
        elif seconds_from_current_time > 0:
            expansion_duration = seconds_from_current_time
        else:
            expansion_duration = 0
        reference_preference.expand_by(-expansion_duration)

        # reference_preference.reference_object = material_provision_d["process_executions"][-1]
        # reference_preference.expected_process_execution_time = material_provision_d["lead_times"][-1]
        # lead_time_value = sum(material_provision_d["lead_times"][0:len(material_provision_d["lead_times"])
        # - 1])
        # if lead_time_value > 0:
        #     reference_preference.lead_time = {entity_type: lead_time_value
        #                                       for entity_type in entity_types_connectors}

        process_executions_variants, process_executions = \
            _create_process_executions_variant_dummies(cfp, transport_process_d, entity_types_connectors,
                                                       reference_preference, pick_up_time, agent_name, current_time)

        reference_preference.origin = process_executions[0].origin
        reference_preference.destination = process_executions[-1].destination

        process_executions_path = \
            ProcessExecutionsPathProposal(call_for_proposal=cfp, provider=agent_name,
                                          goal=transport_process_d["entity_type"],
                                          goal_item=transport_process_d["entities"][0],
                                          path_link_type=ProcessExecutionsPath.LINK_TYPES.LOOSE,
                                          process_executions_components=process_executions_variants,
                                          issue_id=cfp.identification, reference_preference=reference_preference,
                                          cfp_path=node_path, process_execution_id=process_execution_id)

        process_executions_paths.append(process_executions_path)
        process_executions_paths_connectors[process_executions_path] = entity_types_connectors

    return process_executions_paths, process_executions_paths_connectors


def _create_process_executions_variant_dummies(cfp, transport_d, entity_types_connectors,
                                               reference_preference, pick_up_time, agent_name, current_time):
    """Create process_executions_variant_dummies for the request"""
    process_executions_variants = {}
    # only for the pre-version of the process_executions_path
    process_executions = transport_d["process_executions"]
    min_time_restriction64 = np.maximum(current_time, pick_up_time)

    for idx, process_execution in enumerate(process_executions):

        lead_time_value = sum(transport_d["process_times"][0:idx])
        follow_up_time_value = sum(transport_d["process_times"][idx + 1:len(transport_d["process_times"])])

        if lead_time_value > 0:
            lead_time = {entity_type: int(np.ceil(round(lead_time_value, 1))) for entity_type in
                         entity_types_connectors}
        else:
            lead_time = {}

        if follow_up_time_value > 0:
            follow_up_time = {entity_type: int(np.ceil(round(follow_up_time_value, 1)))
                              for entity_type in entity_types_connectors}
        else:
            follow_up_time = {}

        preference_before = reference_preference.get_process_execution_preference_before(
            expected_process_execution_time=transport_d["process_times"][idx],
            process_execution=process_execution,
            lead_time=lead_time, follow_up_time=follow_up_time, min_time_restriction64=min_time_restriction64,
            take_min=True)

        process_executions_variants_dummy = \
            ProcessExecutionsVariantProposal(call_for_proposal=cfp, provider=agent_name, goal=process_execution,
                                             reference_preference=preference_before,
                                             process_execution_id=process_execution.identification)
        process_executions_variants[process_execution] = [process_executions_variants_dummy]

    return process_executions_variants, process_executions


def combine_process_path_components(process_executions_paths, process_group_proposals,
                                    process_executions_paths_connectors, cfp_paths_requested):
    """Combine the components of the process_executions_components"""

    sub_proposals_process = {}

    cfp_path_components = {str(cfp_path_lst): []
                           for cfp_path_lst in cfp_paths_requested}
    for proposal, provider in process_group_proposals:
        cfp_path_components[str(proposal.cfp_path)].append(proposal)

    used_process_executions_components = []
    unused_process_executions_components = {}
    complete_process_executions_paths = []

    for process_executions_path in process_executions_paths:
        if len(process_executions_path.process_executions_components) < 1:
            feasible = process_executions_path.reference_preference.feasible()
            if feasible:
                complete_process_executions_paths.append(process_executions_path)

            continue

        possible_paths_combinations, path_connectors, unused_process_executions_components_pre = \
            _get_possible_path_combinations(cfp_path_components, process_executions_paths_connectors,
                                            process_executions_path)

        complete_process_executions_paths, sub_proposals_process, used_process_executions_components_post, \
            unused_process_executions_components_post = \
            _specify_process_executions_paths(complete_process_executions_paths, possible_paths_combinations,
                                              process_executions_path, path_connectors,
                                              process_group_proposals)

        used_process_executions_components += used_process_executions_components_post
        for component, parents in unused_process_executions_components_pre.items():
            unused_process_executions_components.setdefault(component, []).extend(parents)
        for component, parents in unused_process_executions_components_post.items():
            unused_process_executions_components.setdefault(component, []).extend(parents)

    proposals_to_reject = get_proposals_to_reject(used_process_executions_components,
                                                  unused_process_executions_components,
                                                  process_group_proposals)

    return complete_process_executions_paths, sub_proposals_process, proposals_to_reject


def _get_possible_path_combinations(cfp_path_components, process_executions_paths_connectors, process_executions_path):
    """Determine the combination of paths possible"""

    unused_process_executions_components = {}
    # combine based on the goal of the process_executions_components
    components_path = list(cfp_path_components.values())

    if any(elem is None for elem in components_path):
        unused_process_executions_components = {component: []
                                                for components_lst in components_path
                                                if components_lst is not None
                                                for component in components_lst}
        return [], [], unused_process_executions_components

    # combinations with the same transport_resource
    path_connectors = process_executions_paths_connectors[process_executions_path]
    if len(path_connectors) > 1:  # ToDo: the path should be split in smaller paths
        pass  # raise Exception

    cfp_paths_connector_resources = {}
    connector_resource_cfp_path_components = {}

    for path_components in components_path:  # ToDo: cfp matching easier
        for path_component in path_components:
            resources = [process_executions_component.goal_item
                         for process_executions_component in path_component.get_process_executions_components_lst()]
            connector_resources = [resource for resource in resources
                                   if [resource for entity_type in path_connectors
                                       if resource.entity_type.check_entity_type_match(entity_type)]]
            if len(connector_resources) == 0:
                parts = path_component.goal_item.get_parts()
                connector_resources = [part for part in parts
                                       if [part for entity_type in path_connectors
                                           if part.entity_type.check_entity_type_match(entity_type)]]
                if not connector_resources:
                    continue
            elif len(connector_resources) > 1:
                nsr_connectors = [connector_resource for connector_resource in connector_resources
                                  if isinstance(connector_resource, NonStationaryResource)]
                if nsr_connectors:
                    connector_resources = nsr_connectors
                    path_connectors = list(set(entity_type for entity_type in path_connectors
                                               for nsr_connector in nsr_connectors
                                               if nsr_connector.entity_type.check_entity_type_match(entity_type)))

            process_executions_path.connector_objects = connector_resources

            cfp_paths_connector_resources.setdefault(str(path_component.cfp_path), set()).update(connector_resources)
            connector_resource_cfp_path_components.setdefault(str(path_component.cfp_path),
                                                              []).append(path_component)

    paths = [list(connector_resource_cfp_path_components.values())]

    # Test if all the paths are complete
    possible_paths_combinations = []

    # print("Len :", len(components_path), len(paths_combination[0]))

    for path in paths:
        if len(components_path) == len(path):
            possible_paths_combinations += paths

        elif len(path) == 0:
            pass

        else:
            print("not_complete_resources", path)
            print("unused_process_executions_components needed")
            components_path= list([l for l in components_path if l])
            paths= [l for l in paths if l]
            if len(components_path) == len(path): #'Red Cycling Products Front Tray silver'
                print('process_group if case')
                logging.debug(f'Order ID: {path[0][0].call_for_proposal.order.identification} process Group if case')
                possible_paths_combinations += paths


    for i in possible_paths_combinations:
        if len(i) == 1:
            print('1')

    return possible_paths_combinations, path_connectors, unused_process_executions_components


def _specify_process_executions_paths(complete_process_executions_paths, possible_paths_combinations,
                                      process_executions_path, path_connectors, material_provision_proposals):
    """Specify the process_executions_paths through merging of the preferences and other updates
    Making from one process_executions_path others with respective other specifications"""

    unused_process_executions_components = {}
    used_process_executions_components = []
    sub_proposals_process = {}

    for possible_path_combination in possible_paths_combinations:
        process_executions_path_variant = \
            _adapt_process_executions_variant(process_executions_path, possible_path_combination)
        # duplicate is only made before
        # if more paths are available, they should also be considered ...

        possible_path_combinations_rolled_out = list(itertools.product(*possible_path_combination))

        # check feasibility of the paths
        for path_combinations_rolled_out in possible_path_combinations_rolled_out:
            process_executions_preferences_before = (
                _coordinate_process_executions_variants(list(path_combinations_rolled_out)))

            if process_executions_preferences_before is None:
                feasible = False

            else:
                # assumption the material_supply is done before the reference_preference (from the requester)
                # therefore all process_executions_components are before

                if len(path_connectors) != 1:
                    raise NotImplementedError
                connector_object_entity_type = path_connectors[0]

                process_executions_path_variant.merge_horizontal(preferences_before=process_executions_preferences_before,
                                                                 connector_object_entity_type=connector_object_entity_type)

                feasible = process_executions_path_variant.feasible()  # based on the accepted_time_period

            if feasible:
                complete_process_executions_paths.append(process_executions_path_variant)
                process_executions_path_variant.add_path_ids(process_executions_path_variant.node_identification)
                sub_proposals_process[process_executions_path_variant] = \
                    [proposal for proposal, provider in material_provision_proposals
                     if proposal in path_combinations_rolled_out]
                # maybe also update the preference
                used_process_executions_components.extend(path_combinations_rolled_out)

            else:
                for component in path_combinations_rolled_out:
                    unused_process_executions_components.setdefault(component,
                                                                    []).append(process_executions_path_variant)

    return complete_process_executions_paths, sub_proposals_process, used_process_executions_components, \
        unused_process_executions_components


def _coordinate_process_executions_variants(path_combinations_rolled_out):
    """Coordination means the matching of the different accepted_time_periods in the process_chain"""

        # check if the route is harmonic with each other
    process_executions_preferences_before = \
        {process_execution_variant: process_execution_variant.reference_preference  # .get_copy()
         for process_execution_variant in path_combinations_rolled_out}

    process_executions_preferences_before_lst = list(process_executions_preferences_before.values())

    accepted_start_time = None
    for preference_before in process_executions_preferences_before_lst:

        if accepted_start_time is not None:
            preference_before.update_accepted_time_periods_with_predecessor(accepted_start_time)

            if preference_before.accepted_time_periods.size == 0:
                return None
        accepted_start_time = preference_before.get_accepted_start_time_successor()
        if accepted_start_time is None:
            continue

    accepted_end_time = None
    for preference_before in reversed(process_executions_preferences_before_lst):

        if accepted_end_time is not None:
            preference_before.update_accepted_time_periods_with_successor(accepted_end_time)

        accepted_end_time = preference_before.get_accepted_end_time_predecessor()
        if accepted_end_time is None:
            continue

    # what is this part???
    connector_preference = None
    connector_preference1 = None
    if path_combinations_rolled_out[0].process_executions_components:  # connector
        connector_components = \
            [component
             for component in path_combinations_rolled_out[0].get_process_executions_components_lst()
             if isinstance(component.goal_item, NonStationaryResource)]

        if connector_components:
            if connector_components[0].process_executions_components:
                connector_component = connector_components[0].get_process_executions_components_lst()[0]
                connector_preference = connector_component.reference_preference
                connector_preference1 = connector_component.reference_preference

    if connector_preference is not None and process_executions_preferences_before:
        accepted_end_time -= connector_preference.expected_process_execution_time
        connector_preference.update_accepted_time_periods_with_successor(accepted_end_time)

        connector_preference1.update_accepted_time_periods_with_successor(accepted_end_time)

    for idx in range(1, len(path_combinations_rolled_out)):
        path_combinations_rolled_out[idx].predecessors.append(path_combinations_rolled_out[idx - 1])

    return list(process_executions_preferences_before.values())


def _adapt_process_executions_variant(process_executions_path, possible_path_combination):
    """Adapt the process_executions variants because they are initialized as dummies with process_executions
    Now they are filled with process_executions_components"""

    process_executions_path_variant = process_executions_path.duplicate()

    # (process_executions_path, combi_preference)
    # replace process_executions by process_execution_variants from proposals

    process_executions_components = {}
    for current_component in process_executions_path_variant.get_process_executions_components_lst():
        for components in possible_path_combination:
            if components[0].goal.identification == current_component.goal.identification:
                process_executions_components.setdefault(components[0].goal,
                                                         []).extend(components)

            # maybe also done before - ensure the right sequence

    if process_executions_path.connector_objects:
        # set a variant specific connector object
        connector_entity_type = process_executions_path.connector_objects[0].entity_type
        first_process_execution_component = list(process_executions_components.values())[0][0]
        connector_objects = \
            [_check_connector_ability(sub_component, connector_entity_type)
             for sub_component in first_process_execution_component.get_process_executions_components_lst()
             if _check_connector_ability(sub_component, connector_entity_type)]
        if connector_objects:
            process_executions_path_variant.connector_object = connector_objects

    process_executions_path_variant.replace_process_executions_components(process_executions_components)

    return process_executions_path_variant


def _check_connector_ability(process_executions_component, connector_entity_type):
    """Check if a component element can act as a connector object"""

    possible_connector_object = process_executions_component.goal_item
    if possible_connector_object.entity_type == connector_entity_type:
        return possible_connector_object
    else:
        return None


async def process_process_group(behaviour, process_executions_with_information, preference_requester, cfp, node_path,
                                process_execution_id, current_time, order, request_id, long_time_reservation={}):
    """
    Organize a process_group where the processes depend on each other
    The following steps are executed:
    1. A process_execution path is created in the right order ...
    2. The single processes are requested
    3. The responses/ proposals are taken, and it is tried to create possible paths with the given proposals.
    """

    # t1 = time.process_time()

    process_executions_paths, process_executions_paths_connectors = \
        _create_process_executions_paths(process_executions_with_information, preference_requester, cfp, node_path,
                                         process_execution_id, current_time, behaviour.agent.name)

    # t2 = time.process_time()

    process_group_proposals, cfp_paths_requested = \
        await behaviour.request_process_executions_paths_organization(process_executions_paths, cfp, order, node_path,
                                                                      request_id, long_time_reservation)

    # t3 = time.process_time()

    process_executions_paths, sub_proposals_process, proposals_to_reject = \
        combine_process_path_components(process_executions_paths, process_group_proposals,
                                        process_executions_paths_connectors, cfp_paths_requested)

    # t4 = time.process_time()
    # print("Process Time interim (PG): ", t4 - t3, t3 - t2, t2 - t1)

    return process_executions_paths, proposals_to_reject


class ProcessGroupRequest(PlanningRequest):
    """
    Used to handle process chains that should be executed in a chronological order.
    Example given: Transport that consists of a loading, transport and unloading
    The transport group request is used by the order management behaviour and from the part_request behaviour.
    Note: In the part request behaviour, only the 'process_process_group' method is used.

    Because the 'process_process_group' method is the crucial method it is described in more detail in the method
    itself. (see the method description for more detail ...)
    """

    def __init__(self):
        super(ProcessGroupRequest, self).__init__()

    async def process_request(self, negotiation_object_request):
        """The request for a process_group (for example a transport path)"""
        reference_cfp, client_object, order, preference_requester, request_type, issue_id, process_executions, \
            fixed_origin, long_time_reservation, node_path = negotiation_object_request.unpack()
        request_id = negotiation_object_request.identification

        t1 = time.process_time()

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Client {client_object.identification} start ProcessGroupRequest with ID: {request_id} "
        #       f"process name: {[process_execution.process.name for process_execution in process_executions]}")

        main_entity_type = process_executions[0].get_support_entity_type()

        process_group_d = get_process_group_information(process_executions, main_entity_type, 1)
        process_group_d["entities"] = process_executions[0].get_parts()
        process_executions_with_information = {main_entity_type: process_group_d}  # part_entity_type = resource?

        cfp = negotiation_object_request
        process_execution_id = client_object.identification  # unique for the round
        preference_requester.expected_process_execution_time = 0
        # ToDo: differentiate between before and at the same time
        first_time_stamp = preference_requester.accepted_time_periods[0][0]

        current_time = max(first_time_stamp,
                           np.datetime64(self.agent.change_handler.get_current_time().replace(microsecond=0)))

        process_executions_paths, proposals_to_reject = \
            await process_process_group(
                behaviour=self, process_executions_with_information=process_executions_with_information,
                preference_requester=preference_requester, cfp=cfp, node_path=node_path,
                process_execution_id=process_execution_id, current_time=current_time, order=order,
                request_id=request_id, long_time_reservation=long_time_reservation)

        if process_executions_paths:
            successful = True
        else:
            successful = False
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" End ProcessGroupRequest with ID {request_id} {len(process_executions_paths)}, "
        #       f"{len(proposals_to_reject)}")

        t2 = time.process_time()
        print("Process Time (PG): ", t2 - t1)

        return successful, process_executions_paths, proposals_to_reject

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
                                           node_path=node_path, long_time_reservation=long_time_reservation)

                # determine the providers
                if process_execution.main_resource is not None:
                    providers = [process_execution.main_resource]  # support entity_type
                elif process_execution_path.goal:
                    providers = self.agent.transport_provider[process_execution_path.goal]
                else:
                    possible_main_resource_types = process_execution.process.get_possible_main_resource_entity_types()
                    providers = []
                    for possible_main_resource_type in possible_main_resource_types:
                        providers.append(self.agent.address_book[possible_main_resource_type])
                    providers = list(set(providers))

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
