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

Create the responses associated with the enrichment/ data transformation/ data integration
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ofact.helpers import colored_print, timestamp_to_datetime
# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.planning_services.scenario_analytics.interface.helpers import get_timestamp

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler
    from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler


# ==build response
def build_digital_twin_current_state_available(current_state_available):
    colored_print(f"[API] Build digital_twin current_state_available response with args")

    start_time = datetime.now()
    end_time = start_time - timedelta(hours=current_state_available["consideration_period"])

    current_state_available_settings = {"available": current_state_available["available"],
                                        "default_settings": {"date_start": get_timestamp(start_time),
                                                             "date_end": get_timestamp(end_time)}}

    return current_state_available_settings


# ==build response
# @profile


def build_digital_twin_enrichment_response(get_digital_twin_updated_func,
                                           args, multi_scenarios_handler: MultiScenariosHandler,
                                           digital_twin_file_path, project_path,
                                           progress_tracker, data_source_name=None):
    colored_print(f"[API] Build digital_twin model enrichment response with args")

    date_start = args["dateStart"]
    date_end = args["dateEnd"]
    scenario = args["scenario"]

    print(f"Scenario: {scenario}")
    start_time = timestamp_to_datetime(date_start)
    multi_scenarios_handler.add_empty_scenario(scenario_name=scenario,
                                               project_path=project_path,
                                               digital_twin_file_path=digital_twin_file_path,
                                               start_time=start_time)

    scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario_name=scenario)

    data_source_model_path = Path(str(project_path) + f"/models/data_source/{data_source_name}.xlsx")

    scenario_handler.update_state_model(get_digital_twin_updated_func=get_digital_twin_updated_func,
                                        project_path=project_path,
                                        start_datetime=date_start, end_datetime=date_end,
                                        progress_tracker=progress_tracker,
                                        data_source_model_path=data_source_model_path)

    print("Enrichment status:", scenario,
          scenario_handler.get_kpi_administration().analytics_data_base.digital_twin_df.shape)

    multi_scenarios_handler.match_state_models(scenario_name=scenario)

    return


def set_new_kpi_calc(further_digital_twin, scenario, scenarios_dict: dict[str, SingleScenarioHandler]):

    if scenario not in scenarios_dict:
        scenarios_dict[scenario] = SingleScenarioHandler()

    scenarios_dict[scenario].set_up_scenario(state_model=further_digital_twin)

    return scenarios_dict
