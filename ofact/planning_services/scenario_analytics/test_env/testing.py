# from projects.Schmaus.settings import PROJECT_PATH

from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler, ScenarioDescription
from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler
from ofact.twin.repository_services.persistence import deserialize_state_model


def get_single_scenario_handler(state_model_pkl):
    state_model = deserialize_state_model(source_file_path=state_model_pkl, dynamics=True)

    start_time_enrichment = None
    end_time_enrichment = None

    single_scenario_handler = SingleScenarioHandler(state_model,
                                                    start_time_enrichment,
                                                    end_time_enrichment)
    return single_scenario_handler

def get_multi_scenario_handler(state_model_pkl, scenario_name="current_state"):
    single_scenario_handler = get_single_scenario_handler(state_model_pkl)
    scenarios_dict: dict[str, SingleScenarioHandler] = {scenario_name: single_scenario_handler}
    multi_scenarios_handler = MultiScenariosHandler(scenarios_dict)

    return multi_scenarios_handler


if "__main__" == __name__:
    from ofact.planning_services.scenario_analytics.interface.services.scenario_export import (
        build_scenario_export_response)
    from datetime import datetime

    scenario_name = "simulation_run0"
    multi_scenarios_handler = get_multi_scenario_handler(r"files/scenario_2_p_50.pkl",
                                                         scenario_name=scenario_name)
    single_scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario_name)
    scenario_description = {ScenarioDescription.RESOURCE_SCHEDULE: "schedule.xlsm"}
    single_scenario_handler.initialize_simulation_scenario(scenario_description, project_path=None)

    # scenario_name0 = "simulation_run1"
    # # scenario_handler = get_single_scenario_handler(r"files/two.pkl")
    # # multi_scenarios_handler.set_scenario_handler(scenario_name0, scenario_handler)
    scenarios_to_consider = {"scenarioIDs": [scenario_name]}  # , scenario_name0]}

    build_scenario_export_response(scenarios_to_consider, multi_scenarios_handler=multi_scenarios_handler,
                                   start_time=datetime(2024, 7, 24),
                                   end_time=datetime(2024, 7, 25),
                                   file_directory_path_as_str_extern="C:/Users/afreiter/PycharmProjects/digitaltwin/ofact/planning_services/scenario_analytics/interface/scenario_data/")
