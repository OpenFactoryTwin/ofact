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

API_GET_ORDERS_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "scenario": {"required": True, "type": "string"}  # example current_state
}

API_GET_PRODUCTS_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_PROCESSES_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_RESOURCES_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_UTILIZATION_CHART_RESOURCES_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "bin": {"required": True, "type": "integer"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_LEAD_TIME_CHART = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"},
    "bin": {"required": True, "type": "integer"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_CURRENT_STATE_AVAILABLE = {}

API_GET_ENRICHED_TWIN_MODEL_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "scenario": {"required": True, "type": "string"}
}

API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME = {
    "scenario": {"required": True, "type": "string"}
}

API_GET_SIMULATION_SCHEME = {
    # "agents": {"required": True, "type": "string"},
    # "skill_matrix": {"required": True, "type": "string"},
    # "twin": {"required": True, "type": "string"},
    "worker_planning": {"required": True, "type": "string"},
    "scenario": {"required": True, "type": "string"},
    "simulation_time": {"required": True, "type": "float"}
}

API_POST_SCENARIO_EXPORT_SCHEME = {
    "scenarioIDs": {"required": True, "type": "list"}
}
