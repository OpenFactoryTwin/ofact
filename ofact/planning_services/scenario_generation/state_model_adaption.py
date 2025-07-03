from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofact.twin.state_model.process_models import TransitionModel


class StateModelAdaption:

    def __init__(self, state_model):
        self._state_model = state_model

    def _get_related_transition_models(self, resource_name: str):
        all_processes = self._state_model.get_all_processes()
        relevant_transition_models_origin = []
        relevant_transition_models_destination = []
        for process in all_processes:
            transition_model: TransitionModel = process.get_transition_model()
            in_origins = bool([possible_origin 
                               for possible_origin in transition_model.possible_origins
                              if possible_origin.name == resource_name])
            if in_origins:
                relevant_transition_models_destination.append(transition_model)
            in_destinations = bool([possible_destination 
                               for possible_destination in transition_model.possible_destinations
                              if possible_destination.name == resource_name])
            if in_destinations:
                relevant_transition_models_destination.append(transition_model)
        return relevant_transition_models_origin, relevant_transition_models_destination
        
    def duplicate_stationary_resources(self, names: list[str]):
        duplicated_resources = []
        for name in names:
            duplicated_resource = self.duplicate_stationary_resource(name)
            if duplicated_resource is not None:
                duplicated_resources.append(duplicated_resource)

        return duplicated_resources

    def duplicate_stationary_resource(self, name: str):
        resources = self._state_model.get_stationary_resources()  # Get all resources in the state model
        resource = next((r for r in resources if r.name == name), None)  # Search for the resource by name

        if resource:
            duplicated_resource = resource.duplicate(external_name=True)  # Duplicate the resource
            # ToDo: Ensure that additional resources have different positions
            relevant_transition_models_origin, relevant_transition_models_destination = self._get_related_transition_models(name)
            for transition_model in relevant_transition_models_origin:
                transition_model.possible_origins.append(duplicated_resource)
            for transition_model in relevant_transition_models_destination:
                transition_model.possible_destinations.append(duplicated_resource)
            self._state_model.add_resource(duplicated_resource)  # Add the resource to the state model
            print(f"Stationary resource {name} added successfully.")
        else:
            duplicated_resource = None
            print(f"Resource {name} not found in the model.")

        return duplicated_resource

    def duplicate_non_stationary_resource(self, names: str):
        duplicated_resources = []
        for name in names:
            resources = self._state_model.get_non_stationary_resources(external_name=True)  # Get all resources in the state model
            resource = next((r for r in resources if r.name == name), None)  # Search for the resource by name

            if resource:
                duplicated_resource = resource.duplicate()  # Duplicate the resource
                relevant_transition_models_origin, relevant_transition_models_destination = self._get_related_transition_models(name)
                for transition_model in relevant_transition_models_origin:
                    transition_model.possible_origins.append(duplicated_resource)
                for transition_model in relevant_transition_models_destination:
                    transition_model.possible_destinations.append(duplicated_resource)
                self._state_model.add_resource(duplicated_resource)  # Add the resource to the state model
                print(f"Non-stationary resource {name} added successfully.")  # Confirm resource was added

                duplicated_resources.append(duplicated_resource)
            else:
                print(f"Resource {name} not found in the model.")  # If resource is not found

        return duplicated_resources

    def remove_stationary_resources(self, names: str):
        for name in names:
            resources = self._state_model.get_all_resources()  # Get all resources in the state model
            resource = next((r for r in resources if r.name == name), None)  # Search for the resource by name

            if resource:
                self._state_model.delete_stationary_resource(resource)  # Replace with the actual method if necessary
                relevant_transition_models_origin, relevant_transition_models_destination = self._get_related_transition_models(name)
                for transition_model in relevant_transition_models_origin:
                    transition_model.possible_origins.remove(resource)
                for transition_model in relevant_transition_models_destination:
                    transition_model.possible_destinations.remove(resource)
                print(f"Stationary resource {name} removed successfully.")  # Confirm removal
            else:
                print(f"Resource {name} not found in the model.")  # If resource is not found

    def remove_non_stationary_resources(self, names: str):
        for name in names:
            resources = self._state_model.get_all_resources()
            resource = next((r for r in resources if r.name == name), None)

            if resource:
                relevant_transition_models_origin, relevant_transition_models_destination = self._get_related_transition_models(name)
                for transition_model in relevant_transition_models_origin:
                    transition_model.possible_origins.remove(resource)
                for transition_model in relevant_transition_models_destination:
                    transition_model.possible_destinations.remove(resource)
                self._state_model.delete_non_stationary_resource(resource)
                print(f"Non-stationary resource {name} removed successfully.")
            else:
                print(f"Resource {name} not found.")


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    from ofact.planning_services.model_generation.persistence import deserialize_state_model
    from ofact.twin.repository_services.deserialization.order_types import OrderType

    from projects.bicycle_world.settings import PROJECT_PATH1

    print(PROJECT_PATH1)

    # Example usage
    path_to_model = "scenarios/current/models/twin/"
    state_model_file_name = "base_wo_material_supply.pkl"

    state_model_file_path = Path(str(PROJECT_PATH1), path_to_model + state_model_file_name)
    state_model_generation_settings = {"order_generation_from_excel": False,
                                       "customer_generation_from_excel": True,
                                       "customer_amount": 5,
                                       "order_amount": 20,
                                       "order_type": OrderType.PRODUCT_CONFIGURATOR}
    state_model = deserialize_state_model(state_model_file_path, persistence_format="pkl",
                                          deserialization_required=False,
                                          state_model_generation_settings=state_model_generation_settings)
    state_model_adaption = StateModelAdaption(state_model)

    state_model_adaption.duplicate_stationary_resources(["wheel"])  # Replace "painting" with the resource name you want to add
    state_model_adaption.duplicate_non_stationary_resource(["Main Part AGV 1"])

    state_model_adaption.remove_stationary_resources(["wheel"])   # Replace "painting" with the resource name you want to add
    state_model_adaption.remove_non_stationary_resources(["Main Part AGV 1"])

