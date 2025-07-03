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

# Imports Part 1: Standard Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.planning_services.scenario_analytics.repository_services.controller import DataBaseController


class ScenarioAnalyticsDataBaseController(DataBaseController):

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 5432,
                 database="scenario_analytics_data_base",
                 user: str = "postgres",
                 password: str = "postgres",
                 table_data_types_list: dict[str, pd.DataFrame] = None):
        super().__init__(host=host, port=port, database=database, user=user, password=password,
                         table_data_types_list=table_data_types_list)


if __name__ == "__main__":
    from pathlib import Path
    from ofact.twin.repository_services.persistence import deserialize_state_model
    from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler

    from ofact.settings import ROOT_PATH

    digital_twin_result_pkl = Path(ROOT_PATH.split("ofact")[0],
                                   r"projects/iot/modelling/first_try.pkl")

    state_model = deserialize_state_model(source_file_path=digital_twin_result_pkl)

    scenario_handler = SingleScenarioHandler()
    scenario_handler.set_up_scenario(state_model, set_up_kpi=False)
    scenario_handler.persist_raw_data()
    scenario_handler.read_raw_data_from_database()
