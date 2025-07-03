from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference
from ofact.twin.agent_control.information_desk import InformationServiceAgent
from ofact.twin.agent_control.order import OrderPoolDigitalTwinAgent
from ofact.twin.agent_control.order import OrderDigitalTwinAgent
from ofact.twin.agent_control.resource import (TransportAgent, WorkStationAgent,
                                               ResourceDigitalTwinAgent, WarehouseAgent)
from projects.bicycle_world.twin.agent_control.resource import (PartUnavailabilityWarehouseAgent,
                                                                LimitedWarehouseAgent, AdvancedWorkStationAgent)
from projects.bicycle_world.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorBicycleAgent
from ofact.twin.repository_services.deserialization.agents_importer import MapperMethods
from ofact.twin.repository_services.deserialization.basic_file_loader import convert_str_to_dict, convert_str_to_list, Mapping


class MappingAgents(Mapping):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'OrderPoolAgent': OrderPoolDigitalTwinAgent,
        'OrderAgent': OrderDigitalTwinAgent,
        'ResourceAgent': ResourceDigitalTwinAgent,
        'WorkStationAgent': WorkStationAgent,
        'WarehouseAgent': WarehouseAgent,
        'TransportAgent': TransportAgent,
        'SchedulingCoordinatorAgent': SchedulingCoordinatorBicycleAgent,
        'InformationServiceAgent': InformationServiceAgent,
        'Preference': EntityPreference
    }

    object_columns = {
        'state_model': None,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,  # ToDo
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': convert_str_to_dict,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }

    to_be_defined_object_columns = {
        'state_model': MapperMethods.determine_digital_twin,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': MapperMethods.determine_entity_provider,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }


class MappingAgentsPartUnavailability(Mapping):
    """Used to transform the Factory elements from the Excel sheets to python classes."""

    mappings = {
        'OrderPoolAgent': OrderPoolDigitalTwinAgent,
        'OrderAgent': OrderDigitalTwinAgent,
        'ResourceAgent': ResourceDigitalTwinAgent,
        'WorkStationAgent': WorkStationAgent,
        'WarehouseAgent': PartUnavailabilityWarehouseAgent,
        'TransportAgent': TransportAgent,
        'SchedulingCoordinatorAgent': SchedulingCoordinatorBicycleAgent,
        'InformationServiceAgent': InformationServiceAgent,
        'Preference': EntityPreference
    }

    object_columns = {
        'state_model': None,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,  # ToDo
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': convert_str_to_dict,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }

    to_be_defined_object_columns = {
        'state_model': MapperMethods.determine_digital_twin,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': MapperMethods.determine_entity_provider,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }


class MappingAgentsVehicleAvailability(Mapping):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'OrderPoolAgent': OrderPoolDigitalTwinAgent,
        'OrderAgent': OrderDigitalTwinAgent,
        'ResourceAgent': ResourceDigitalTwinAgent,
        'WorkStationAgent': AdvancedWorkStationAgent,
        'WarehouseAgent': LimitedWarehouseAgent,
        'TransportAgent': TransportAgent,
        'SchedulingCoordinatorAgent': SchedulingCoordinatorBicycleAgent,
        'InformationServiceAgent': InformationServiceAgent,
        'Preference': EntityPreference
    }

    object_columns = {
        'state_model': None,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,  # ToDo
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': convert_str_to_dict,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }

    to_be_defined_object_columns = {
        'state_model': MapperMethods.determine_digital_twin,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': MapperMethods.determine_entity_provider,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }