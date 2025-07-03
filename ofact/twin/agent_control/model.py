from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference


def get_keys_by_value(dictionary, value):
    return [key
            for key, val in dictionary.items()
            if val == value]

class AgentsModel:

    def __init__(self, order_pool_agents, order_agents, resource_agents, work_station_agents, warehouse_agents,
                 transport_agents, scheduling_coordinator_agents, information_service_agents):

        self._order_pool_agents = order_pool_agents
        self._order_agents = order_agents
        self._resource_agents = resource_agents
        self._work_station_agents = work_station_agents
        self._warehouse_agents = warehouse_agents
        self._transport_agents = transport_agents
        self._scheduling_coordinator_agents = scheduling_coordinator_agents
        self._information_service_agents = information_service_agents

        self.accepted_time_horizont_preference = 10

    def update_agent_model_through_state_model(self, state_model):
        all_processes = state_model.get_all_processes()
        resources_required = self._get_resources_required(all_processes)
        resource_entity_types_required = self._get_resource_entity_types_required(all_processes)
        all_resources = state_model.get_all_resources()
        for resource in all_resources:
            if resource.entity_type in resource_entity_types_required:
                resources_required.add(resource)
            elif resource.entity_type.super_entity_type in resource_entity_types_required:
                resources_required.add(resource)

        all_resource_agents = (self._resource_agents | self._work_station_agents |
                               self._warehouse_agents | self._transport_agents)

        resource_agent_match, resources_to_remove = (
            self._match_resources_with_agents(all_resource_agents, resources_required))
        resource_entity_types_required |= set(resource.entity_type
                                              for resource in resources_required)

        resource_entity_types_agent_match, resource_entity_types_to_remove = (
            self._match_resource_entity_types_with_agents(all_resource_agents, resource_entity_types_required))

        self._adapt_agents(all_resource_agents, resource_entity_types_agent_match, resource_entity_types_to_remove,
                           resource_agent_match, resources_to_remove)

    def _get_resources_required(self, all_processes):
        resources_required = set()
        for process in all_processes:
            transition_model = process.get_transition_model()
            possible_origins = transition_model.get_possible_origins()
            possible_destinations = transition_model.get_possible_destinations()
            resources_required |= set(possible_origins) | set(possible_destinations)

        return resources_required

    def _get_resource_entity_types_required(self, all_processes):
        resources_entity_types_required = set()
        for process in all_processes:
            resource_model = process.get_resource_model()
            resource_groups = resource_model.get_resource_groups(process_execution=None)
            for resource_group in resource_groups:
                resources_entity_types_required |= set(resource_group.resources)

        return resources_entity_types_required

    def _match_resources_with_agents(self, all_resource_agents, resources_required):

        resources_required_interim = resources_required.copy()
        all_resources = set()
        resource_entity_types_agent_match = {}
        for (agent_type, agent_name), resource_agent in all_resource_agents.items():
            for resource in resource_agent._resources:
                if resource in resources_required:
                    resources_required.remove(resource)

                all_resources.add(resource)

                resource_entity_types_agent_match[resource.entity_type] = resource_agent

        resources_to_remove = set(all_resources).difference(resources_required_interim)

        resource_agent_match = {}
        for resource_required in resources_required:
            if resource_required.entity_type in resource_entity_types_agent_match:
                resource_agent_match[resource_required] = resource_entity_types_agent_match[resource_required.entity_type]

        return resource_agent_match, resources_to_remove

    def _match_resource_entity_types_with_agents(self, all_resource_agents, resource_entity_types_required):

        resource_entity_types_required_interim = resource_entity_types_required.copy()
        resource_entity_types_agent_match = {}
        for (agent_type, agent_name), resource_agent in all_resource_agents.items():
            for resource in resource_agent._resources:
                if resource.entity_type in resource_entity_types_required:
                    resource_entity_types_required.remove(resource.entity_type)

                resource_entity_types_agent_match[resource.entity_type] = resource_agent
                if resource.entity_type.super_entity_type is not None:
                    resource_entity_types_agent_match[resource.entity_type.super_entity_type] = resource_agent

        resource_entity_types_to_remove = (
            set(resource_entity_types_agent_match.keys()).difference(resource_entity_types_required_interim))

        return resource_entity_types_agent_match, resource_entity_types_to_remove

    def _adapt_agents(self, all_resource_agents, resource_entity_types_agent_match, resource_entity_types_to_remove,
                      resource_agent_match, resources_to_remove):

        address_book_extension = resource_entity_types_agent_match | resource_agent_match   # ToDo
        address_book_removal = resource_entity_types_to_remove | resources_to_remove

        for (agent_type, agent_name), resource_agent in all_resource_agents.items():

            for resource_et in address_book_extension:
                if resource_et.get_static_model_id()[1:] not in resource_agent.address_book:
                    resource_agent.address_book[resource_et.get_static_model_id()[1:]] = resource_agent.name

            for resource_et in address_book_removal:
                if resource_et.get_static_model_id()[1:] in resource_agent.address_book:
                    del resource_agent.address_book[resource_et.get_static_model_id()[1:]]

            resources_adapted = resource_agent._resources.copy()
            preferences_adapted = list(resource_agent.preferences.values())
            resources_changed = False
            resources_removed = []
            for resource in resource_agent._resources:
                if resource in resources_to_remove:
                    print("Remove:", resource.name)
                    resources_changed = True
                    resources_adapted.remove(resource)
                    resources_removed.append(resource)
            preferences_adapted = [preference
                                    for preference in preferences_adapted
                                    if preference.reference_objects[0] not in resources_removed]

            new_resources = get_keys_by_value(resource_agent_match, agent_name)
            for new_resource in new_resources:
                if new_resource not in resources_adapted:
                    print("Add:", new_resource.name)
                    resources_changed = True
                    resources_adapted.append(new_resource)
                    new_resource_preference = EntityPreference(reference_objects=[new_resource],
                                                               accepted_time_horizont=self.accepted_time_horizont_preference)
                    preferences_adapted.append(new_resource_preference)

            if resources_changed:
                resource_agent.update_resource_related_variables(resources_adapted, preferences_adapted)
