
import platform
import sys
from pathlib import PosixPath, WindowsPath

from ofact.twin.state_model.model import StateModel
from ofact.planning_services.scenario_analytics.interface.settings.api import (
    API_GET_ORDERS_SCHEME, API_GET_PRODUCTS_SCHEME, API_GET_PROCESSES_SCHEME, API_GET_RESOURCES_SCHEME,
    API_GET_ENRICHED_TWIN_MODEL_SCHEME, API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME, API_GET_SIMULATION_SCHEME,
    API_GET_UTILIZATION_CHART_RESOURCES_SCHEME, API_GET_LEAD_TIME_CHART, API_GET_CURRENT_STATE_AVAILABLE,
    API_POST_SCENARIO_EXPORT_SCHEME)
from ofact.planning_services.scenario_analytics.interface.settings.functions_used import DashboardControllerSettings
from projects.bicycle_world.planning_services.scenario_analytics.dashboard.api_settings import (API_SET_CURRENT_STATE_AVAILABLE,
                                                                                                API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME)
from projects.bicycle_world.scenarios.current.main import SimulationStarterBicycleWorld
from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler
from projects.bicycle_world.settings import (API_HOST, API_PORT, PROJECT_PATH,
                                                         XMPP_SERVER_IP_ADDRESS, XMPP_SERVER_REST_API_PORT,
                                                         XMPP_SERVER_SHARED_SECRET, XMPP_SERVER_REST_API_USERS_ENDPOINT)
from ofact.settings import ROOT_PATH
sys.setrecursionlimit(10000)  # setting a higher recursion limit for pickling
# scenarios
scenario = "base_wo_material_supply"
# scenario = "digital_twin230118_baseKPI"

project_path = "\\".join([folder_name for folder_name in sys.path[0].split("\\")[:-1]])

file_path = fr'projects/bicycle_world/scenarios/current/models/twin/{scenario}.pkl'
system_name = platform.system()
print("Platform name", system_name)

if system_name == "Linux":
    digital_twin_file_path = PosixPath(ROOT_PATH, file_path).absolute()
elif system_name == "Windows":
    digital_twin_file_path = WindowsPath(ROOT_PATH, file_path).absolute()
else:
    raise Exception

# digital_twin_objects = StaticModelGenertor.from_pickle(digital_twin_pickle_path=digital_twin_pickle_path)
### For local development only ###
digital_twin = StateModel.from_pickle(digital_twin_file_path)
# kpi = digital_twin_objects
# digital_twin_objects = StaticModelGenerator.from_pickle(digital_twin_pickle_path=digital_twin_pickle_path)
# digital_twin = digital_twin_objects.get_digital_twin()


SingleScenarioHandler.basic_state_model = digital_twin

simulation_starter = (
    SimulationStarterBicycleWorld(project_path=PROJECT_PATH,
                                  xmpp_server_ip_address=XMPP_SERVER_IP_ADDRESS,
                                  xmpp_server_rest_api_port=XMPP_SERVER_REST_API_PORT,
                                  xmpp_server_shared_secret=XMPP_SERVER_SHARED_SECRET,
                                  xmpp_server_rest_api_users_endpoint=XMPP_SERVER_REST_API_USERS_ENDPOINT))
simulate = simulation_starter.simulate
DashboardControllerSettings(API_GET_ORDERS_SCHEME=API_GET_ORDERS_SCHEME,
                            API_GET_PRODUCTS_SCHEME=API_GET_PRODUCTS_SCHEME,
                            API_GET_PROCESSES_SCHEME=API_GET_PROCESSES_SCHEME,
                            API_GET_RESOURCES_SCHEME=API_GET_RESOURCES_SCHEME,
                            API_GET_UTILIZATION_CHART_RESOURCES_SCHEME=API_GET_UTILIZATION_CHART_RESOURCES_SCHEME,
                            API_GET_LEAD_TIME_CHART=API_GET_LEAD_TIME_CHART,
                            API_GET_CURRENT_STATE_AVAILABLE=API_GET_CURRENT_STATE_AVAILABLE,
                            API_SET_CURRENT_STATE_AVAILABLE=API_SET_CURRENT_STATE_AVAILABLE,
                            API_GET_ENRICHED_TWIN_MODEL_SCHEME=API_GET_ENRICHED_TWIN_MODEL_SCHEME,
                            API_GET_SIMULATION_SCHEME=API_GET_SIMULATION_SCHEME,
                            API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME=API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME,
                            API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME=API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME,
                            API_HOST=API_HOST, API_PORT=API_PORT, PROJECT_PATH=project_path,
                            DIGITAL_TWIN_FILE_PATH=digital_twin_file_path,
                            update_digital_twin=None, DATA_SOURCE_MODEL_NAME=None, simulation_func=simulate,
                            API_POST_SCENARIO_EXPORT_SCHEME=API_POST_SCENARIO_EXPORT_SCHEME)

# from ofact.env.interfaces.frontend.dashboard_controller import app

from projects.bicycle_world.planning_services.scenario_analytics.dashboard import app

# build_simulation_response({}, None)
app.run(debug=False, host=API_HOST, port=API_PORT, use_reloader=False)

# ToDo
from ofact.planning_services.scenario_analytics.interface.services.simulation import build_simulation_model_paths_response
response_object = (
            build_simulation_model_paths_response(folder_path=PROJECT_PATH,
                                                  simulation_parameters=API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME))
response = response_object.get_response_dict()

