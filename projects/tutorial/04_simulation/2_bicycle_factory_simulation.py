from datetime import datetime
from pathlib import Path

from ofact.planning_services.model_generation.persistence import get_state_model_file_path, deserialize_state_model, \
    serialize_state_model
from ofact.twin.repository_services.deserialization.order_types import OrderType
from projects.bicycle_world.twin.agent_control.simulate import SimulationStarterBicycleWorld
from projects.bicycle_world.settings import (PROJECT_PATH, XMPP_SERVER_IP_ADDRESS,
                                                           XMPP_SERVER_REST_API_PORT, XMPP_SERVER_SHARED_SECRET,
                                                           XMPP_SERVER_REST_API_USERS_ENDPOINT)

if __name__ == "__main__":
    print("STATUS: Starting Execution")
    # logging.debug('This will get logged')
    store_digital_twin_ = True
    simulate_ = True  # True

    digital_twin_file_name = "base_wo_material_supply.pkl"
    state_model_file_path = get_state_model_file_path(project_path=PROJECT_PATH,
                                                      state_model_file_name=digital_twin_file_name,
                                                      path_to_model="scenarios/current/models/twin/")
    state_model_generation_settings = {"order_generation_from_excel": False,
                                       "customer_generation_from_excel": False,
                                       "customer_amount": 5, "order_amount": 20,
                                       "order_type": OrderType.PRODUCT_CONFIGURATOR}

    state_model = deserialize_state_model(state_model_file_path, persistence_format="pkl",
                                          state_model_generation_settings=state_model_generation_settings,
                                          deserialization_required=False)

    if simulate_:
        # file name of the model
        agents_model_file_name = "base_wo_material_supply_6.xlsx"

        # number of order agents (each order agent is responsible for one order - (sequential 1-1 relation))
        order_agent_amount = 12

        # simulation starter object that stores the general information such as the xmpp server data
        simulation_starter = (
            SimulationStarterBicycleWorld(project_path=PROJECT_PATH, path_to_models="/scenarios/current/",
                                          xmpp_server_ip_address=XMPP_SERVER_IP_ADDRESS,
                                          xmpp_server_rest_api_port=XMPP_SERVER_REST_API_PORT,
                                          xmpp_server_shared_secret=XMPP_SERVER_SHARED_SECRET,
                                          xmpp_server_rest_api_users_endpoint=XMPP_SERVER_REST_API_USERS_ENDPOINT))

        # here you can introduce paths to update the digital twin state model
        # e.g., an Excel modeled schedule for the resources that updates
        # their intern (state model) schedules (of the resources)
        digital_twin_update_paths = {}  # {"resource_schedule": "settings.xlsx"}

        # path where the resulting state model (including the dynamics) is stored
        digital_twin_state_model_result_path = Path(str(PROJECT_PATH), "scenarios/current/results/six_orders.pkl")

        # method that execute the simulation
        simulation_starter.simulate(digital_twin=state_model,
                                    start_time_simulation=datetime.now(),
                                    digital_twin_update_paths=digital_twin_update_paths,
                                    agents_file_name=agents_model_file_name, order_agent_amount=12,
                                    digital_twin_result_path=digital_twin_state_model_result_path)
