import asyncio
from datetime import datetime

import numpy as np

from ofact.planning_services.scenario_analytics.interface.services.kpi_tables import (build_filter_response,
                                                                                      build_orders_response)
from ofact.planning_services.scenario_analytics.test_env.testing import get_multi_scenario_handler, get_state_model

scenario_name = "current_state"
multi_scenarios_handler = get_multi_scenario_handler(r"files/current.pkl",
                                                     scenario_name=scenario_name)
scenario_name1 = multi_scenarios_handler.set_simulation_scenario("current_state")

scenario_handler0 = multi_scenarios_handler.get_scenario_handler_by_name(scenario_name)
print(len(scenario_handler0.state_model.get_process_executions_list()),
      scenario_handler0.kpi_administration.analytics_data_base.digital_twin_df.shape)

scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario_name1)
scenario_handler.state_model = get_state_model(r"files/test.pkl")

print("Last:", len(scenario_handler0.state_model.get_process_executions_list()))

scenario_handler.initialize_simulation_scenario()
multi_scenarios_handler.match_state_models(scenario_name=scenario_name1)
scenario_handler.update_data()
print(scenario_handler.kpi_administration.analytics_data_base.digital_twin_df.shape)

scenarios_to_consider = {"scenarioIDs": [scenario_name, scenario_name1]}
# get_kpi_for_scenarios(multi_scenarios_handler, scenarios_to_consider)
scenarios_dict = multi_scenarios_handler.get_scenarios_dict()
start_time = np.datetime64(datetime(2024, 6, 10, 6), "ns")  # scenario_handler.get_state_model_start_time()
end_time = np.datetime64(datetime(2024, 6, 10, 15), "ns")  # scenario_handler.get_state_model_end_time()

# print(scenario_name)
# scenario_handler = scenarios_dict[scenario_name]
#
# args = {"dateStart": start_time,
#         "dateEnd": end_time,
#         "scenario": scenario_name1}
#
# filter_response = build_filter_response(args, multi_scenarios_handler)
# filter_response_dict = filter_response.get_response_dict_ids()
# orders_list = filter_response_dict["orders"]
# products_list = filter_response_dict["products"]
# processes_list = filter_response_dict["processes"]
# resources_list = filter_response_dict["resources"]
#
# args = {"dateStart": (start_time - np.timedelta64(1, "s")).astype('datetime64[s]').astype('int'),
#         "dateEnd": (end_time.astype('datetime64[s]') + np.timedelta64(1, "s")).astype('int'),
#         "orders": orders_list,
#         "products": products_list,
#         "processes": processes_list,
#         "resources": resources_list,
#         "scenario": scenario_name}
# args["dateStart"] = max(args["dateStart"], 0)
#
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# order_units = {"units": {"deliveryDelay": "minute", "totalLeadTime": "minute", "totalWaitingTime": "minute"}}
# order_response = asyncio.run(build_orders_response(args | order_units, multi_scenarios_handler))
# response_dict = order_response.get_response_dict()

print(scenario_name1)
scenario_handler = scenarios_dict[scenario_name1]

args = {"dateStart": start_time,
        "dateEnd": end_time,
        "scenario": scenario_name1}

filter_response = build_filter_response(args, multi_scenarios_handler)
filter_response_dict = filter_response.get_response_dict_ids()
orders_list = filter_response_dict["orders"]
products_list = filter_response_dict["products"]
processes_list = filter_response_dict["processes"]
resources_list = filter_response_dict["resources"]

args = {"dateStart": (start_time - np.timedelta64(1, "s")).astype('datetime64[s]').astype('int'),
        "dateEnd": (end_time.astype('datetime64[s]') + np.timedelta64(1, "s")).astype('int'),
        "orders": orders_list,
        "products": products_list,
        "processes": processes_list,
        "resources": resources_list,
        "scenario": scenario_name1}
args["dateStart"] = max(args["dateStart"], 0)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
order_units = {"units": {"deliveryDelay": "minute", "totalLeadTime": "minute", "totalWaitingTime": "minute"}}
order_response = asyncio.run(build_orders_response(args | order_units, multi_scenarios_handler))  # ToDo
response_dict = order_response.get_response_dict()
print(response_dict)


# order_identifier = "KAID337559"
#
# state_model = scenario_handler.get_state_model()
# orders = state_model.get_orders()
# for order in orders:
#     if order_identifier == order.identifier:
#         print("from digital twin:", str(order))
#
# kpi_administration = scenario_handler.get_kpi_administration()
# df = kpi_administration.analytics_data_base.digital_twin_df
#
# orders_df = df.loc[df["reference"] == order_identifier]
# print("from analytics data base df:", orders_df)
#
# for row in response_dict["orders"]:
#     if order_identifier in row["referenceValue"]:
#         print("from order_view:", row)
#
# print(order_response)
