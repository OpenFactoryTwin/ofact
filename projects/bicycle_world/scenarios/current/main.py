import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from ofact.planning_services.model_generation.persistence import get_state_model_file_path, deserialize_state_model, \
    serialize_state_model
from ofact.twin.agent_control.organization import Agents
from ofact.twin.repository_services.deserialization.order_types import OrderType
from projects.bicycle_world.twin.agent_control.simulate import SimulationStarterBicycleWorld
from projects.bicycle_world.settings import (PROJECT_PATH, XMPP_SERVER_IP_ADDRESS, XMPP_SERVER_REST_API_PORT,
                                             XMPP_SERVER_SHARED_SECRET, XMPP_SERVER_REST_API_USERS_ENDPOINT)
from collections import defaultdict

import os


def create_state_model(state_model_file_name, order_amount):
    file_types = ["xlsx",
                  "pkl"]
    file_type = file_types[1]

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


def combine_agent_dicts(agent_dicts):
    combined = defaultdict(lambda: {'True': 0, 'False': 0})
    for agent in agent_dicts:
        for output, counts in agent.items():
            combined[output]['True'] += counts.get('True', 0)
            combined[output]['False'] += counts.get('False', 0)
    return dict(combined)


def save_dict_to_excel(combined_dict, filename="agent_results.xlsx", run_index=0):
    df = pd.DataFrame.from_dict(combined_dict, orient='index')
    df.index.name = "Output Neuron"

    # Sheet name like "Run_1", "Run_2", ...
    sheet_name = f"Run_{run_index + 1}"

    if not os.path.exists(filename):
        # Create new Excel file
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        # Append new sheet to existing file
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name)


def load_yaml(path: str) -> dict:
    raw = yaml.safe_load(open(path)) or {}
    # start_time_simulation
    if "start_time_simulation" in raw and isinstance(raw["start_time_simulation"], str):
        raw["start_time_simulation"] = datetime.fromisoformat(
            raw["start_time_simulation"]
        )

    return raw


def main(start_time_simulation: datetime,
         resource_schedule: str,
         work_in_process: int):
    print("STATUS: Starting Execution")
    if work_in_process > 19:
        raise Exception("Es können nicht mehr als 19 Aufträge gleichzeitig im System sein. "
                        "Ändern Sie bitte den 'work_in_process' Parameter.")
    # 1) build your initial state model
    order_amount = 20
    state_model = create_state_model("bicycle_factory", order_amount)

    # 2) decide whether to simulate or just return the model
    simulate_ = True
    if not simulate_:
        return state_model

    # 3) run your simulation
    simulation_period = timedelta(hours=10)
    agents_file_name = "factory_agents.xlsx"
    simulation_starter = SimulationStarterBicycleWorld(
        project_path=PROJECT_PATH,
        path_to_models="/scenarios/current",
        xmpp_server_ip_address=XMPP_SERVER_IP_ADDRESS,
        xmpp_server_rest_api_port=XMPP_SERVER_REST_API_PORT,
        xmpp_server_shared_secret=XMPP_SERVER_SHARED_SECRET,
        xmpp_server_rest_api_users_endpoint=XMPP_SERVER_REST_API_USERS_ENDPOINT
    )

    schedule_name = resource_schedule
    digital_twin_update_paths = {"resource_schedule": f"{schedule_name}.xlsx"}
    result_path = Path(PROJECT_PATH, f"scenarios/current/results/{schedule_name}_{int(work_in_process)}.pkl")

    simulated = simulation_starter.simulate(
        digital_twin=state_model,
        start_time_simulation=start_time_simulation,
        digital_twin_update_paths=digital_twin_update_paths,
        agents_file_name=agents_file_name,
        order_agent_amount=work_in_process,
        digital_twin_result_path=result_path,
        simulation_end=(
            Agents.SimulationEndTypes.TIME_LIMIT,
            (start_time_simulation, start_time_simulation + simulation_period)
        )
    )
    return simulated


if __name__ == "__main__":
    # --- load defaults + YAML ---
    defaults = {
        "start_time_simulation": datetime(2025, 7, 4, 7),
        "resource_schedule": "schedule_s1",
        "work_in_process": 15
    }
    try:
        yaml_cfg = load_yaml("config.yaml")
        defaults.update(yaml_cfg)
    except FileNotFoundError:
        print("Configuration file not found.")

    # --- parse command‐line args ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_time_simulation",
        type=lambda s: datetime.fromisoformat(s)
    )
    parser.add_argument("--resource_schedule", type=str)
    parser.add_argument("--work_in_process", type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    # --- call main with clean args and print the result ---
    result = main(
        args.start_time_simulation,
        args.resource_schedule,
        args.work_in_process
    )
