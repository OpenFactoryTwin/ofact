"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################

Create the responses associated with the simulation.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ofact.planning_services.scenario_analytics.scenario_handling.single import ScenarioDescription
# Imports Part 3: Project Imports
from ofact.twin.agent_control.organization import Agents
from ofact.helpers import colored_print
from ofact.planning_services.scenario_analytics.interface.helpers import argument_preparation

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler
    from ofact.twin.agent_control.simulate import SimulationStarter


class SimulationParametersResponse:
    def __init__(self):
        self.simulation_parameters = []

    @argument_preparation
    def add_simulation_parameter(self, name, display_type, display_items, default_item):
        simulation_parameter_dict = {"name": name,
                                     "display_type": display_type,
                                     "display_items": display_items,
                                     "default_item": default_item}
        self.simulation_parameters.append(simulation_parameter_dict)

    def get_response_dict(self):
        return self.simulation_parameters


# ==build response
# @profile

def build_simulation_model_paths_response(folder_path, simulation_parameters) -> (
        SimulationParametersResponse):
    """Return all file names from the folder path - respectively the folder where the models can be found"""
    colored_print(f"[API] Build simulation model paths response with args")

    response = SimulationParametersResponse()

    # {"name": "agents",
    #  "display_type": ["dropdown"],
    #  "display_items": [],
    #  "default_item": "FIRST_ITEM",
    #  "file_content": True,
    #  "folder_path": "/models/agents",
    #  "accepted_file_types": ["xlsx", "pkl", "xlsm"]}

    for simulation_parameter in simulation_parameters:
        simulation_parameter_ = simulation_parameter.copy()
        if simulation_parameter_["file_content"]:
            file_names = []
            complete_folder_path = Path(str(folder_path) + simulation_parameter_["folder_path"])
            files_path = os.listdir(complete_folder_path)
            for file_name in files_path:
                file_path = os.path.join(complete_folder_path, file_name)
                if not os.path.isfile(file_path):
                    continue  # complete folder

                file_type = file_name.split(".")[-1]
                if file_type not in simulation_parameter_["accepted_file_types"]:
                    continue

                file_names.append(file_name)
            simulation_parameter_["display_items"] = file_names
            if simulation_parameter_["default_item"] is None:
                simulation_parameter_["default_item"] = file_names[0]

        response.add_simulation_parameter(name=simulation_parameter_["name"],
                                          display_type=simulation_parameter_["display_type"],
                                          display_items=simulation_parameter_["display_items"],
                                          default_item=simulation_parameter_["default_item"])

    # response.add_simulation_parameter(folder_path, type_="path") ?

    return response


def build_simulation_response(simulation_func: SimulationStarter.simulate, args,
                              multi_scenarios_handler: MultiScenariosHandler, scenario_model_paths,
                              progress_tracker, project_path):
    # standard parameters
    duration = args["simulation_time"]
    scenario = args["scenario"]

    # project specific parameters
    if "agents" in args:
        agents_file_name = args["agents"]
    else:
        agents_file_name = "24_base.xlsx"
    if "skill_matrix" in args:
        skill_matrix_file_name = args["skill_matrix"]
    else:
        skill_matrix_file_name = "Mitarbeiter-Qualifikationen (Skill-Matrix).xlsx"
    if "twin" in args:
        twin_file_name = args["twin"]
    else:
        twin_file_name = "digital-twin_SCHMAUS_wnm.pkl"

    worker_planning_file_name = args["worker_planning"]

    # end_time needed
    ground_scenario_available = multi_scenarios_handler.is_scenario_available(scenario)
    # get the twin updated
    if ground_scenario_available:
        # change the scenario name (simulation flag)
        scenario = "current_state"
        simulation_scenario_name = multi_scenarios_handler.set_simulation_scenario(scenario)

    else:
        return None

    scenario_model_paths_dict = {"agents": agents_file_name,
                                 "skill_matrix": skill_matrix_file_name,
                                 "twin": twin_file_name,
                                 "worker_planning": worker_planning_file_name}

    digital_twin_update_paths = {"skill_matrix": skill_matrix_file_name,
                                 "worker_planning": worker_planning_file_name}

    scenario_model_paths[scenario] = scenario_model_paths_dict

    start_time_simulation: datetime = datetime(2023, 7, 24, 7)
    end_time_simulation = start_time_simulation + timedelta(hours=duration)
    order_target_quantity: int = 50
    simulation_end = (Agents.SimulationEndTypes.TIME_LIMIT, (start_time_simulation, end_time_simulation))

    scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(simulation_scenario_name)
    state_model = scenario_handler.get_state_model()
    if state_model is None:
        raise Exception("Digital Twin should be always filled")

    try:
        simulation_func(digital_twin=state_model,
                        agents_file_name=agents_file_name,
                        digital_twin_update_paths=digital_twin_update_paths,
                        progress_tracker=progress_tracker,
                        start_time_simulation=start_time_simulation,
                        simulation_end=simulation_end,
                        order_target_quantity=order_target_quantity)
    except BaseException as e:
        print("Simulation finished!")
    finally:
        print("finally case")
        pass
    print("Set the simulation results")

    scenario_description = {ScenarioDescription.RESOURCE_SCHEDULE: worker_planning_file_name}
    scenario_handler.initialize_simulation_scenario(scenario_description=scenario_description,
                                                    project_path=project_path,
                                                    start_time=start_time_simulation)
    multi_scenarios_handler.match_state_models(scenario_name=simulation_scenario_name)
    scenario_handler.kpi_administration.analytics_data_base.update_digital_twin_df()

    return simulation_scenario_name
