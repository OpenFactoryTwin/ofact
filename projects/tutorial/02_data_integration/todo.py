from pathlib import Path
import pandas as pd

from ofact.planning_services.model_generation.persistence import deserialize_state_model
from ofact.twin.repository_services.deserialization.order_types import OrderType

from projects.bicycle_world.settings import PROJECT_PATH
print(PROJECT_PATH)

path_to_model = "scenarios/current/models/twin/"
state_model_file_name = "base_wo_material_supply.pkl"

state_model_file_path = Path(str(PROJECT_PATH), path_to_model + state_model_file_name)
state_model_generation_settings = {"order_generation_from_excel": False,
                                   "customer_generation_from_excel": True,
                                   "customer_amount": 5,
                                   "order_amount": 20,
                                   "order_type": OrderType.PRODUCT_CONFIGURATOR}
state_model = deserialize_state_model(state_model_file_path, persistence_format="pkl",
                                      state_model_generation_settings=state_model_generation_settings,
                                      deserialization_required=False)

def add_workstation_duplicated(state_model, resource_name):  # : StateModel
    work_stations = state_model.get_work_stations()
    concerned_work_station = [work_station
                              for work_station in work_stations
                              if work_station.name == resource_name][0]

    duplicated_work_station = concerned_work_station.duplicate()
    duplicated_work_station = concerned_work_station
    storages_of_the_duplicated_work_station = duplicated_work_station.get_storages()
    concerned_work_station.change_position_initially() # like change_position()  of a NonStationaryResource
    # ToDo: adapt the position of the resource (also for the storages of the work_station)

    #  adapt the transition models
    transition_models = state_model.get_transition_models()
    transition_models  #

    duplicated_work_station



def add_transport_resource_duplicated(state_model, resource_name):
    pass

add_workstation_duplicated(state_model, resource_name="painting")