from datetime import datetime

import yaml
from collections import defaultdict
from IPython.display import display

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ipydatagrid import DataGrid
from ipywidgets import Button, VBox, HBox, IntText, Output

from ofact.planning_services.model_generation.persistence import deserialize_state_model, get_state_model_file_path, \
    serialize_state_model
from ofact.planning_services.scenario_analytics.business_logic.schedule import Schedule
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase
from ofact.twin.repository_services.deserialization.order_types import OrderType
from ofact.twin.state_model.model import StateModel

from projects.bicycle_world.settings import PROJECT_PATH

language = "de"

LABELS = {
    "de": {
        "row_index": "Zeilenindex",
        "add_row": 'Zeile hinzufügen',
        "delete_row": 'Zeile entfernen',
        "save": 'Änderungen speichern',
        "delivery_reliability_container": "Liefertreue",
        "completed": "fertiggestellt",
        "not_completed": "nicht fertiggestellt",
        "completed_on_time": "on time fertiggestellt",
        "not_completed_on_time": "Nicht on time fertiggestellt",
        "share_completed_orders": "Anteil fertiggestellter Aufträge an denen am Tag geplanten Aufträge"
    },
    "en": {
        "row_index": "Row index",
        "add_row": 'Add row',
        "delete_row": 'Delete row',
        "save": 'Save changes',
        "delivery_reliability_container": "Delivery reliability",
        "completed": "Completed",
        "not_completed": "Not completed",
        "completed_on_time": "Completed on time",
        "not_completed_on_time": "Not completed on time",
        "share_completed_orders": "Share of completed orders planned for the day"
    }
}

def get_label(key: str, lang: str) -> str:
    return LABELS.get(lang, LABELS["en"]).get(key, key)

def create_state_model(state_model_file_name, order_amount=20, file_type="xlsx"):

    if file_type == "xlsx":
        deserialization_required = True
    else:
        deserialization_required = False

    state_model_file_name = f"{state_model_file_name}.{file_type}"
    state_model_file_path = get_state_model_file_path(project_path=PROJECT_PATH,
                                                      state_model_file_name=state_model_file_name,
                                                      path_to_model="scenarios/current/models/twin/")
    state_model_generation_settings = {"order_generation_from_excel": False,
                                       "customer_generation_from_excel": False,
                                       "customer_amount": 5, "order_amount": order_amount,
                                       "order_type": OrderType.PRODUCT_CONFIGURATOR}
    state_model = deserialize_state_model(state_model_file_path, persistence_format=file_type,
                                          state_model_generation_settings=state_model_generation_settings,
                                          deserialization_required=deserialization_required)
    return state_model


def get_table(source_file: str, target_file: str, sheet_name: str = None):
    # Load Excel file
    global df
    df = pd.read_excel(str(PROJECT_PATH) + source_file,
                       sheet_name=sheet_name)
    df.fillna('', inplace=True)
    df = df.astype(str)
    df.columns = df.columns.astype(str)

    # Create DataGrid - for state model
    grid = DataGrid(
        df,
        base_column_size=100,
        base_row_size=30,
        selection_mode='cell',
        editable=True,
    )

    row_index_str = get_label("row_index", language)
    # Input field for the row index
    row_index_input = IntText(
        value=0,
        description=f'{row_index_str}:',
        layout={'width': '200px'}
    )

    # Output for messages
    output = Output()

    def rebuild_grid():
        global grid
        # Remove old grid from main display
        main_display.children = main_display.children[:-1]
        # Create new grid
        grid = DataGrid(
            df,
            base_column_size=100,
            base_row_size=30,
            selection_mode='cell',
            editable=True,
        )
        # Add new grid to main display
        main_display.children += (grid,)

    def add_row_at_index(b):
        global df
        idx = int(row_index_input.value)
        print(idx)
        if idx < 0 or idx > len(df):
            with output:
                output.clear_output()
                print("Invalid index for adding additional rows.")
            return
        # Create new empty row
        new_row = pd.DataFrame([{col: "" for col in df.columns}])
        # Insert new row
        df = pd.concat([df.iloc[:idx], new_row, df.iloc[idx:]]).reset_index(drop=True)
        # Update DataGrid
        grid.data = df
        rebuild_grid()
        with output:
            output.clear_output()
            print(f"Row {idx} added.")

    def delete_row_at_index(b):
        global df
        idx = int(row_index_input.value)
        print(idx)
        if idx < 0 or idx >= len(df):
            with output:
                output.clear_output()
                print("Invalid index for the deletion of rows.")
            return
        # Delete row
        df = df.drop(df.index[idx]).reset_index(drop=True)
        # Update DataGrid
        grid.data = df
        rebuild_grid()
        with output:
            output.clear_output()
            print(f"Row {idx} deleted.")

    # Function to save changes
    def save_changes(b):
        edited_df = grid.data
        # Save changes to an Excel file
        edited_df.to_excel(str(PROJECT_PATH) + target_file, index=False)
        with output:
            output.clear_output()
            print("Changes saved")

    add_row_str = get_label("add_row", language)
    delete_row_str = get_label("delete_row", language)
    save_str = get_label("save", language)

    # Buttons for adding and deleting
    add_row_button = Button(
        description=add_row_str,
        layout={'width': '150px'},
        button_style='success'  # Green button
        # Or use custom color:
        # style={'button_color': '#28a745'}
    )
    delete_row_button = Button(
        description=delete_row_str,
        layout={'width': '150px'},
        button_style='danger'  # Red button
        # Or use custom color:
        # style={'button_color': '#dc3545'}
    )
    save_button = Button(
        description=save_str,
        layout={'width': '200px'},
        button_style='primary'  # Blue button
        # Or use custom color:
        # style={'button_color': '#007bff'}
    )

    # Link functions to buttons
    add_row_button.on_click(add_row_at_index)
    delete_row_button.on_click(delete_row_at_index)
    save_button.on_click(save_changes)

    # Display buttons and input field
    control_panel = HBox([row_index_input, add_row_button, delete_row_button, save_button])
    # Display everything together
    main_display = VBox([control_panel, output, grid])
    display(main_display)

def update_state_model(source_file: str, sheet_name: str = None,
                       state_model_file_name: str = None, target_state_model_file_name: str = None):
    df = pd.read_excel(str(PROJECT_PATH) + source_file,
                       sheet_name=sheet_name)

    capability_worker_mapper = defaultdict(list)

    # Iterate through the DataFrame and create the mapping
    for _, row in df.iterrows():
        process_execution_id = row["Tätigkeit"]
        resource_used_id = row["MA"]
        capability_worker_mapper[process_execution_id].append(resource_used_id)

    file_type = "xlsx"
    deserialization_required = True

    state_model_generation_settings = {"order_generation_from_excel": True,
                                       "customer_generation_from_excel": True,
                                       "customer_amount": 5, "order_amount": 0,
                                       "order_type": OrderType.SHOPPING_BASKET}

    state_model_file_path = get_state_model_file_path(project_path=PROJECT_PATH,
                                                      state_model_file_name=state_model_file_name,
                                                      path_to_model="models/twin/")
    state_model = deserialize_state_model(state_model_file_path, persistence_format=file_type,
                                          state_model_generation_settings=state_model_generation_settings,
                                          deserialization_required=deserialization_required)

    stationary_resources = state_model.get_stationary_resources()

    for capability_name, resource_names in capability_worker_mapper.items():
        stationary_resources_with_capability = [stationary_resource
                                                for stationary_resource in stationary_resources
                                                if capability_name in stationary_resource.entity_type.name]

        stationary_resource_names_with_capability = \
            {"_".join(stationary_resource.name.split(" ")[:-1]): stationary_resource
             for stationary_resource in stationary_resources_with_capability}

        # match stationary_resource_names_with_capability - resource_names
        resources_to_destroy = []
        for available_resource_name, resource in stationary_resource_names_with_capability.items():
            if available_resource_name in resource_names:
                resource_names.remove(available_resource_name)
            else:
                resources_to_destroy.append(resource)

        reference_resource = stationary_resources_with_capability[0]
        for resource_name in resource_names:
            # create new resource
            duplicated_resource = reference_resource.duplicate()

            duplicated_resource.name = resource_name
            duplicated_resource.external_identifications["static_model"] = ["_" + resource_name + "_sr"]
            state_model.add_stationary_resource(duplicated_resource)
            # ToDo: agent model

        for resource in resources_to_destroy:
            state_model.delete_stationary_resource(resource)
            # ToDo: agent model

    state_model_file_path = get_state_model_file_path(project_path=PROJECT_PATH,
                                                      state_model_file_name=target_state_model_file_name,
                                                      path_to_model="models/twin/")
    serialize_state_model(state_model=state_model, target_file_path=state_model_file_path,
                          serialization_required=True, dynamics=False)


def update_simulation_input_parameters(start_time_simulation = datetime(2025, 7, 4, 7),
                                       resource_schedule = "schedule_s1.xlsx", work_in_process = 5):

    config = {
        "start_time_simulation": start_time_simulation,
        "work_in_process": work_in_process,
        "resource_schedule": resource_schedule,
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f)


def get_schedule(state_model: StateModel, relevant_resources=None):
    event_type = "ACTUAL"

    analytics_data_basis = ScenarioAnalyticsDataBase(state_model=state_model)
    schedule = Schedule(analytics_data_basis, state_model)

    if relevant_resources is None:
        relevant_resources = {}

    relevant_process_names = [process.name
                              for process in state_model.get_value_added_processes()]
    reference_resource_types = ["ActiveMovingResource", "WorkStation"]
    schedule.create_planned_schedule(reference_resource_types, relevant_resources)

    # second_resource_type = "ActiveMovingResource"

    schedule.create_schedule(relevant_process_names=relevant_process_names,
                             relevant_resources=relevant_resources,
                             reference_resource_types=reference_resource_types, second_resource_type=None,
                             # second_resource_type,
                             event_type=event_type)

    return schedule


def plot_resource_schedule(schedule, event_type="ACTUAL"):
    schedule.plot_schedule(event_type=event_type,
                           planned_resource_schedule=True)


def get_schedule_time_equivalent(aggregated=[], schedule=None):
    event_type = "ACTUAL"

    actual_resources_working_hours, planned_resources_working_hours = (
        schedule.get_full_time_equivalent(event_type=event_type,
                                          planned_resource_schedule=True,
                                          aggregated=aggregated))

    return actual_resources_working_hours, planned_resources_working_hours


def get_analysis(state_model, source_file_schedule, sheet_name_schedule="General"):

    resource_names = [sr.name
                      for sr in state_model.get_stationary_resources()
                      if
                      "Container" in sr.name or "Stock" in sr.name or "Truck" in sr.name]  # "Felix", "Maximilian", "Paul", "Elias", "Jonas", "Lukas", "Leon", "Noah", "David", "Tim"]
    capacity_utilization = []
    for resource_name in resource_names:
        utilization = state_model.get_resource_capacity_utilization(resource_names=[resource_name])[0]
        capacity_utilization.append(utilization)

    delivery_reliability = [state_model.get_delivery_reliability()]
    delivery_reliability.append(100 - delivery_reliability[0])

    df_schedule_complete = pd.read_excel(str(PROJECT_PATH) + source_file_schedule,
                       sheet_name=sheet_name_schedule)
    df_schedule = df_schedule_complete.loc[:, "MA":]

    df_staff = df_schedule_complete.loc[:, :"MA"].drop(columns="Station \\ Zeit")
    df_staff["Tätigkeit"] = df_staff["Tätigkeit"].str.replace("_MA", "", regex=False)

    df_staff["Capacity Utilization"] = capacity_utilization
    df_capacity_utilization = df_staff

    # create graphics for the analysis

    palettes = sns.color_palette()

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # First Plot:
    ax[0, 0].pie(x=delivery_reliability, radius=1.2, center=(1, 4), autopct='%1.1f%%', colors=palettes[-2:])
    ax[0, 0].set_title("Delivery Reliability")
    ax[0, 0].legend(["On Time Delivery", "Off Time Delivery"], title="Legend", loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))

    # Second Plot
    sns.barplot(x=df_capacity_utilization["MA"], y=df_capacity_utilization["Capacity Utilization"], ax=ax[0, 1],
                color=palettes[1])
    ax[0, 1].axhline(y=df_capacity_utilization["Capacity Utilization"].mean(), color=palettes[0], linestyle="--",
                     label="Average")
    handles = [
        plt.Line2D([0], [0], color=palettes[1], lw=4, label="Capacity Utilization"),
        plt.Line2D([0], [0], color=palettes[0], lw=4, linestyle='--', label="Average")
    ]
    ax[0, 1].set_title("Capacity Utilization per Staff")
    ax[0, 1].set_xlabel("Staff")
    ax[0, 1].set_ylabel("Capacity Utilization")
    ax[0, 1].legend(handles=handles, title="Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    # third plot:
    group = df_capacity_utilization.groupby("Tätigkeit")["Capacity Utilization"].agg(
        ["mean", "min", "max"]).reset_index()

    ax[1, 0].bar(group["Tätigkeit"], group["mean"], color=palettes[1])

    for i in range(len(group)):
        ax[1, 0].hlines(y=group["max"][i],
                        xmin=i - 0.05,
                        xmax=i + 0.05,
                        color=palettes[-2])
        ax[1, 0].hlines(y=group["min"][i],
                        xmin=i - 0.05,
                        xmax=i + 0.05,
                        color=palettes[0])
        ax[1, 0].vlines(ymin=group["min"][i],
                        ymax=group["max"][i],
                        x=i,
                        color="black")
    handles = [
        plt.Line2D([0], [0], color=palettes[1], lw=4, label="Average Utilization"),
        plt.Line2D([0], [0], color=palettes[-2], lw=4, label="Maximum Utilization"),
        plt.Line2D([0], [0], color=palettes[0], lw=4, label="Minimum Utilization"),
    ]
    ax[1, 0].set_title("Average Capacity Utilization per Activity")
    ax[1, 0].set_xlabel("Activities")
    ax[1, 0].set_ylabel("Capacity Utilization")
    ax[1, 0].legend(handles=handles, title="Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # fourth plot:
    for index, row in df_schedule.iterrows():
        for hour, available in row[1:].items():
            if available == "x" or available == "X":
                ax[1, 1].barh(row["MA"], 1, left=row.index.get_loc(hour) - 1, color=palettes[-2], edgecolor='black')
            else:
                ax[1, 1].barh(row["MA"], 1, left=row.index.get_loc(hour) - 1, color=palettes[3], edgecolor='black')
    handles = [
        plt.Line2D([0], [0], color=palettes[-2], lw=4, label='Available'),
        plt.Line2D([0], [0], color=palettes[3], lw=4, label='Not Available')
    ]
    ax[1, 1].set_yticks(range(len(df_schedule["MA"])))
    ax[1, 1].set_yticklabels(df_schedule["MA"])
    ax[1, 1].set_xticks(range(len(df_schedule.columns) - 1))
    ax[1, 1].set_xticklabels(df_schedule.columns[1:], rotation=45)
    ax[1, 1].set_xlabel("Hour")
    ax[1, 1].set_title("Staff Schedule")
    ax[1, 1].legend(handles=handles, title="Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))


def get_delivery_reliability(state_model: StateModel):
    delivery_reliability = state_model.get_delivery_reliability()

    delivery_reliability_container_str = get_label("delivery_reliability_container", language)
    completed_str = get_label("completed_on_time", language)
    not_completed_str = get_label("not_completed_on_time", language)

    palettes = sns.color_palette()
    plt.pie(x=[delivery_reliability, 100 - delivery_reliability], radius=1.2, center=(1, 4), autopct='%1.1f%%',
            colors=palettes[-2:])
    plt.title(delivery_reliability_container_str)
    plt.legend([completed_str, not_completed_str], title="Legend", loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()


def get_order_finished(state_model: StateModel):
    number_of_orders_finished = state_model.get_number_of_orders_finished()
    number_of_all_orders = len(state_model.get_orders())
    number_not_finished_orders = number_of_all_orders- number_of_orders_finished
    order_finished_percentage = number_of_orders_finished / number_of_all_orders
    order_finished_percentage *= 100
    palettes = sns.color_palette()

    share_completed_orders_str = get_label("share_completed_orders", language)
    completed_str = get_label("completed", language)
    not_completed_str = get_label("not_completed", language)

    plt.pie(x=[order_finished_percentage, 100 - order_finished_percentage], radius=1.2, center=(1, 4), autopct='%1.1f%%',
            colors=palettes[-2:])
    plt.title(share_completed_orders_str)
    plt.legend([completed_str + f" ({number_of_orders_finished})",
                not_completed_str + f" ({number_not_finished_orders})"],
               title="Legend", loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()


def get_order_lead_time(state_model: StateModel):

    # 1) set Seaborn style
    sns.set_style("whitegrid")

    # 2) get your timedelta list
    order_lead_times = state_model.get_order_lead_times()

    # 3) convert to hours
    hours = [td.total_seconds() / 3600 for td in order_lead_times]

    # 4) build the plot
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.boxplot(
        hours,
        vert=False,  # horizontal
        notch=True,  # draw a notch
        patch_artist=True,  # fill with color
        boxprops=dict(facecolor='#87CEEB', edgecolor='#1C355F'),
        medianprops=dict(color='#FF4500', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='#FF6347', alpha=0.7)
    )

    # 5) labels & title
    ax.set_xlabel('Lead Time (hours)')
    ax.set_title('Distribution of Order Lead Times')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get_table("data/Avisierungen.xlsx", target_file=None)

    # schedule_name = "schedule_s1"
    # digital_twin_file_name = f"learned_model_new.xlsx"
#
    # update_state_model(source_file=f"/models/schedules/{schedule_name}.xlsx",
    #                    sheet_name="General",
    #                    state_model_file_name=digital_twin_file_name,
    #                    target_state_model_file_name="test.xlsx")

    from ofact.planning_services.model_generation.persistence import deserialize_state_model
    from ofact.twin.state_model.model import StateModel

    state_model_file_path = "data/result.pkl"
    state_model: StateModel = deserialize_state_model(state_model_file_path, persistence_format="pkl", dynamics=True)
