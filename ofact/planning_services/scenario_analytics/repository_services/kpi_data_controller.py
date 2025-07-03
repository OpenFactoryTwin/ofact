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
"""

import pandas as pd

from ofact.planning_services.scenario_analytics.repository_services.controller import DataBaseController


class KPIDataBaseController(DataBaseController):

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 5432,
                 database="kpi_data_base",
                 user: str = "postgres",
                 password: str = "postgres",
                 table_data_types_list: dict[str, pd.DataFrame] = None):
        super().__init__(host=host, port=port, database=database, user=user, password=password,
                         table_data_types_list=table_data_types_list)


if __name__ == "__main__":
    from pathlib import Path
    from ofact.twin.repository_services.persistence import deserialize_state_model
    from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler
    from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler


    from ofact.settings import ROOT_PATH

    digital_twin_result_pkl = Path(ROOT_PATH.split("ofact")[0],
                                   r"projects/Schmaus/data/raw_dt/current.pkl")

    state_model = deserialize_state_model(source_file_path=digital_twin_result_pkl)

    start_time_enrichment = None
    end_time_enrichment = None
    single_scenario_handler = SingleScenarioHandler(state_model=state_model,
                                                    start_time_enrichment=start_time_enrichment,
                                                    end_time_enrichment=end_time_enrichment)
    scenarios_dict: dict[str, SingleScenarioHandler] = {"current_state": single_scenario_handler}
    multi_scenarios_handler = MultiScenariosHandler(scenarios_dict)

    # second scenario
    simulation_scenario_name = multi_scenarios_handler.set_simulation_scenario("current_state")
    scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(simulation_scenario_name)

    digital_twin_result_pkl = Path(ROOT_PATH.split("ofact")[0],
                                   r"projects/Schmaus/data/simulation_results/scenario_2_show_less_wnm_50.pkl")
    simulation_state_model = deserialize_state_model(source_file_path=digital_twin_result_pkl)
    scenario_handler.state_model = simulation_state_model
    scenario_handler.initialize_simulation_scenario()

    multi_scenarios_handler.match_state_models(scenario_name=simulation_scenario_name)
    scenario_handler.kpi_administration.analytics_data_base.update_digital_twin_df()
    for name, s_h in multi_scenarios_handler.get_scenarios_dict().items():
        print(name, s_h.id, s_h.scenario_name, s_h.__dict__)
    multi_scenarios_handler.persist_kpi_data(scenarios={"scenarioIDs": ["current_state", simulation_scenario_name]})
