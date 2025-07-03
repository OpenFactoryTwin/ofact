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

from pathlib import Path


class DashboardControllerSettings:
    API_GET_ORDERS_SCHEME = None
    API_GET_PRODUCTS_SCHEME = None
    API_GET_PROCESSES_SCHEME = None
    API_GET_RESOURCES_SCHEME = None
    API_GET_UTILIZATION_CHART_RESOURCES_SCHEME = None
    API_GET_LEAD_TIME_CHART = None
    API_GET_CURRENT_STATE_AVAILABLE = None
    API_SET_CURRENT_STATE_AVAILABLE = None
    API_GET_ENRICHED_TWIN_MODEL_SCHEME = None
    API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME = None
    API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME = None
    API_GET_SIMULATION_SCHEME = None
    API_HOST = None
    API_PORT = None
    PROJECT_PATH = None
    update_digital_twin_func = None
    DATA_SOURCE_MODEL_NAME = None
    DIGITAL_TWIN_FILE_PATH = None
    simulation_func = None
    API_POST_SCENARIO_EXPORT_SCHEME = None

    def __init__(self, API_GET_ORDERS_SCHEME: dict, API_GET_PRODUCTS_SCHEME: dict, API_GET_PROCESSES_SCHEME: dict,
                 API_GET_RESOURCES_SCHEME: dict, API_GET_UTILIZATION_CHART_RESOURCES_SCHEME: dict,
                 API_GET_LEAD_TIME_CHART: dict,
                 API_GET_CURRENT_STATE_AVAILABLE: dict, API_SET_CURRENT_STATE_AVAILABLE: dict,
                 API_GET_ENRICHED_TWIN_MODEL_SCHEME: dict,
                 API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME: dict,
                 API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME: list[dict],
                 API_GET_SIMULATION_SCHEME: dict,
                 API_HOST: str, API_PORT: str, PROJECT_PATH: str,
                 update_digital_twin: callable, DATA_SOURCE_MODEL_NAME: str,
                 DIGITAL_TWIN_FILE_PATH: Path,
                 simulation_func: callable,
                 API_POST_SCENARIO_EXPORT_SCHEME: dict) -> None:
        type(self).API_GET_ORDERS_SCHEME = API_GET_ORDERS_SCHEME
        type(self).API_GET_PRODUCTS_SCHEME = API_GET_PRODUCTS_SCHEME
        type(self).API_GET_PROCESSES_SCHEME = API_GET_PROCESSES_SCHEME
        type(self).API_GET_RESOURCES_SCHEME = API_GET_RESOURCES_SCHEME
        type(self).API_GET_UTILIZATION_CHART_RESOURCES_SCHEME = API_GET_UTILIZATION_CHART_RESOURCES_SCHEME
        type(self).API_GET_LEAD_TIME_CHART = API_GET_LEAD_TIME_CHART

        type(self).API_GET_CURRENT_STATE_AVAILABLE = API_GET_CURRENT_STATE_AVAILABLE
        type(self).API_SET_CURRENT_STATE_AVAILABLE = API_SET_CURRENT_STATE_AVAILABLE
        type(self).API_GET_ENRICHED_TWIN_MODEL_SCHEME = API_GET_ENRICHED_TWIN_MODEL_SCHEME
        type(self).API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME = API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME
        type(self).API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME = API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME
        type(self).API_GET_SIMULATION_SCHEME = API_GET_SIMULATION_SCHEME
        type(self).API_HOST = API_HOST
        type(self).API_PORT = API_PORT
        type(self).PROJECT_PATH = PROJECT_PATH
        type(self).update_digital_twin_func = update_digital_twin
        type(self).DATA_SOURCE_MODEL_NAME = DATA_SOURCE_MODEL_NAME
        type(self).DIGITAL_TWIN_FILE_PATH = DIGITAL_TWIN_FILE_PATH
        type(self).simulation_func = simulation_func

        type(self).API_POST_SCENARIO_EXPORT_SCHEME = API_POST_SCENARIO_EXPORT_SCHEME

    @classmethod
    def get_API_HOST(cls):
        return cls.API_HOST

    @classmethod
    def get_API_PORT(cls):
        return cls.API_PORT

    @classmethod
    def get_PROJECT_PATH(cls):
        return cls.PROJECT_PATH

    @classmethod
    def get_API_GET_ORDERS_SCHEME(cls):
        return cls.API_GET_ORDERS_SCHEME

    @classmethod
    def get_API_GET_PRODUCTS_SCHEME(cls):
        return cls.API_GET_PRODUCTS_SCHEME

    @classmethod
    def get_API_GET_PROCESSES_SCHEME(cls):
        return cls.API_GET_PROCESSES_SCHEME

    @classmethod
    def get_API_GET_RESOURCES_SCHEME(cls):
        return cls.API_GET_RESOURCES_SCHEME

    @classmethod
    def get_API_GET_UTILIZATION_CHART_RESOURCES_SCHEME(cls):
        return cls.API_GET_UTILIZATION_CHART_RESOURCES_SCHEME

    @classmethod
    def get_API_GET_LEAD_TIME_CHART(cls):
        return cls.API_GET_LEAD_TIME_CHART

    @classmethod
    def get_API_GET_CURRENT_STATE_AVAILABLE(cls):
        return cls.API_GET_CURRENT_STATE_AVAILABLE

    @classmethod
    def get_API_SET_CURRENT_STATE_AVAILABLE(cls):
        return cls.API_SET_CURRENT_STATE_AVAILABLE

    @classmethod
    def get_API_GET_ENRICHED_TWIN_MODEL_SCHEME(cls):
        return cls.API_GET_ENRICHED_TWIN_MODEL_SCHEME

    @classmethod
    def get_API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME(cls):
        return cls.API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME

    @classmethod
    def get_API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME(cls):
        return cls.API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME

    @classmethod
    def get_API_GET_SIMULATION_SCHEME(cls):
        return cls.API_GET_SIMULATION_SCHEME

    @classmethod
    def get_update_digital_twin_func(cls):
        return cls.update_digital_twin_func

    @classmethod
    def get_DATA_SOURCE_MODEL_NAME(cls):
        return cls.DATA_SOURCE_MODEL_NAME

    @classmethod
    def get_DIGITAL_TWIN_FILE_PATH(cls):
        return cls.DIGITAL_TWIN_FILE_PATH

    @classmethod
    def get_simulation_func(cls):
        return cls.simulation_func

    @classmethod
    def get_API_POST_SCENARIO_EXPORT_SCHEME(cls):
        return cls.API_POST_SCENARIO_EXPORT_SCHEME
