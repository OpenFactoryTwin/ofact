import ast
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

class AgentModelAdaption:

    def __init__(self, state_model, _agent_model_df):
        self._state_model = state_model
        self._agent_model_df = _agent_model_df

    # Function to update the agent model in Excel (add new resource)
    def add_duplicated_resources(self, reference_name: str, new_resource_object, new_process_st=None):
        """Assumption: A similar resource (with reference_name) is already available."""

        # Get all resources from the state model
        resources = self._state_model.get_all_resources()

        # Find the resource based on the name
        resource = next((r for r in resources if r.name == reference_name), None)

        # Check if the resource exists
        if not resource:
            print(f"Resource with name {reference_name} not found.")
            return

        # Use the base label from the reference resource
        resource_label_name = resource.get_static_model_id()[1:]  # e.g., "wheel_as"
        base_name = f"{resource_label_name}_copy"  # e.g., "wheel_as_copy"

        # Generate unique name: wheel_as_copy_1, wheel_as_copy_2, etc.
        i = 1
        while True:
            new_resource_label_name = f"{base_name}_{i}"
            if new_resource_label_name not in self._agent_model_df.to_string():
                break
            i += 1

        print("Generated new resource label:", new_resource_label_name)

        # Print the columns to inspect the structure
        print("Columns in the Excel file:", self._agent_model_df.columns)

        # Check if the resource already exists in the file, and add if not
        for index, row in self._agent_model_df.iterrows():
            if 'address_book' in self._agent_model_df.columns:
                if row['address_book'] == row['address_book']:
                    if resource_label_name in eval(row['address_book']):
                        address_book = eval(row['address_book'])
                        address_book[new_resource_label_name] = address_book[resource_label_name]
                        self._agent_model_df.loc[index, 'address_book'] = str(address_book)

            if 'possible_processes' in self._agent_model_df.columns:
                if row['possible_processes'] == row['possible_processes']:  # not NaN
                    old_process_str = row['possible_processes']  # e.g. to_be_defined_by(wheel_as)

                    # Replace the original name with the new duplicated name inside the parentheses
                    new_process_str = re.sub(
                        r'to_be_defined_by\((.*?)\)',
                        f'to_be_defined_by({new_resource_label_name})',
                        old_process_str
                    )

                    self._agent_model_df.loc[index, 'possible_processes'] = new_process_st

            if 'resources' in self._agent_model_df.columns:
                if row['resources'] == row['resources']:
                    if resource_label_name in eval(row['resources']):
                        self._agent_model_df.loc[index, 'resources'] = (
                            str(eval(row['resources']) + [new_resource_label_name]))

            if 'preferences' in self._agent_model_df.columns:
                if row['preferences'] == row['preferences']:
                    if resource_label_name + "_preference" in eval(row['preferences']):
                        self._agent_model_df.loc[index, 'preferences'] = (
                            str(eval(row['preferences']) + [new_resource_label_name + "_preference"]))

            if "reference_objects":
                if row['reference_objects'] == row['reference_objects']:
                    if resource_label_name in eval(row['reference_objects']):
                        adapted_row = row.copy()
                        adapted_row['reference_objects'] = str([new_resource_label_name])
                        adapted_row.iloc[1] = resource_label_name + "_preference"
                        self._agent_model_df.loc[len(self._agent_model_df)] = adapted_row

        # ToDo: should be saved with another name, or?
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Adding timestamp to the file name
        file_path = Path("C:/Users/MansK/ofact-intern/projects/bicycle_world/scenarios/current/models/agents/base_wo_material_supply_6_modified_" + timestamp + ".xlsx")

        # Save the updated DataFrame to the new file
        self._agent_model_df.to_excel(file_path, index=False)
        print(f"Resource {new_resource_label_name} added to agent model and saved as {file_path}")


    def delete_resource(self, reference_name: str):
        """Remove the specified resource from the agent model in the Excel file."""

        # Check if the resource exists in the DataFrame
        resource_index = self._agent_model_df[self._agent_model_df['resource'] == reference_name].index

        if len(resource_index) > 0:
            # If resource exists, delete it from the DataFrame
            self._agent_model_df.drop(resource_index, inplace=True)
            print(f"Resource {reference_name} removed successfully.")
        else:
            print(f"Resource with name {reference_name} not found in the Excel file.")

        # Save the updated dataframe back to a new Excel file to avoid overwriting the original file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Adding timestamp to the file name
        file_path = Path("C:/Users/MansK/ofact-intern/projects/bicycle_world/scenarios/current/models/agents/base_wo_material_supply_6_modified_deleted_" + timestamp + ".xlsx")

        # Save the updated DataFrame to the new file
        self._agent_model_df.to_excel(file_path, index=False)
        print(f"Updated agent model saved successfully at: {file_path}")


if __name__ == "__main__":
    from ofact.planning_services.model_generation.persistence import deserialize_state_model
    from ofact.twin.repository_services.deserialization.order_types import OrderType

    from ofact.planning_services.scenario_generation.state_model_adaption import StateModelAdaption

    PROJECT_PATH = "C:/Users/MansK/ofact-intern/projects/bicycle_world"
    # Example usage
    path_to_model = "scenarios/current/models/twin/"
    state_model_file_name = "base_wo_material_supply.pkl"

    state_model_file_path = Path(str(PROJECT_PATH), path_to_model + state_model_file_name)
    state_model_generation_settings = {"order_generation_from_excel": False,
                                       "customer_generation_from_excel": True,
                                       "customer_amount": 5,
                                       "order_amount": 20,
                                       "order_type": OrderType.PRODUCT_CONFIGURATOR}
    state_model = deserialize_state_model(state_model_file_path, persistence_format="pkl",
                                          deserialization_required=False,
                                          state_model_generation_settings=state_model_generation_settings)
    state_model_adaption = StateModelAdaption(state_model)

    duplicated_workstation = state_model_adaption.duplicate_stationary_resources(["wheel"])  # Replace "painting" with the resource name you want to add
    file_path = Path(str(PROJECT_PATH) + "/scenarios/current/models/agents/base_wo_material_supply_6.xlsx")

    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path, skiprows=1)
    agent_model_adaption = AgentModelAdaption(state_model, df)

    agent_model_adaption.add_duplicated_resources("wheel", duplicated_workstation)
