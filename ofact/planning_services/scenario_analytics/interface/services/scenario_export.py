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

Used to export the scenarios to a zip file of Excel tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import zipfile

import pandas as pd

from ofact.helpers import colored_print
from ofact.settings import ROOT_PATH
from ofact.planning_services.scenario_analytics.interface.services.kpi_tables import (
    get_orders_response, get_products_response, get_processes_response, get_resources_response,
    OrdersResponse)
from ofact.planning_services.scenario_analytics.interface.services.kpi_charts import get_utilization_chart_response

if TYPE_CHECKING:
    from datetime import datetime
    from ofact.planning_services.scenario_analytics.interface.services.kpi_tables import SingleScenarioHandler


file_directory_path_as_str = str(ROOT_PATH) + "/planning_services/scenario_analytics/interface/scenario_data/"


class ScenarioOverviewResponse:
    def __init__(self):
        self.scenarios = {"path": "",
                          "scenarios": []}

    def add_path(self, path):
        self.scenarios["path"] = path

    def add_scenario(self, name, **kwargs):
        scenario_dict = {"name": name}
        if kwargs:
            scenario_dict |= scenario_dict

        self.scenarios["scenarios"].append(scenario_dict)

    def get_response_dict(self):
        return self.scenarios


def build_scenario_overview_response(multi_scenarios_handler, export_path=""):
    colored_print(f"[API] Build digital_twin model enrichment response with args")

    response = ScenarioOverviewResponse()

    response.add_path(export_path)
    scenario_names = multi_scenarios_handler.get_scenario_names()
    for scenario_name in scenario_names:
        response.add_scenario(name=scenario_name)

    return response


def store_xlsx_sheets(orders: pd.DataFrame, products: pd.DataFrame, processes: pd.DataFrame, resources: pd.DataFrame,
                      resource_utilization: pd.DataFrame, resource_schedules: pd.DataFrame, order_traces: pd.DataFrame,
                      resource_traces: pd.DataFrame, scenario_descriptions: pd.DataFrame, file_directory_path_as_str_intern: str):
    """Store the scenario data to Excel file sheets ..."""

    # standard kpi's
    orders.to_excel(file_directory_path_as_str_intern + "orders.xlsx", index=False)
    products.to_excel(file_directory_path_as_str_intern + "products.xlsx", index=False)
    processes.to_excel(file_directory_path_as_str_intern + "processes.xlsx", index=False)
    resources.to_excel(file_directory_path_as_str_intern + "resources.xlsx", index=False)

    # specific kpi's
    resource_utilization.to_excel(file_directory_path_as_str_intern + "resource_utilization.xlsx", index=False)
    resource_schedules.to_excel(file_directory_path_as_str_intern + "resource_schedules.xlsx", index=False)
    order_traces.to_excel(file_directory_path_as_str_intern + "order_traces.xlsx", index=False)
    resource_traces.to_excel(file_directory_path_as_str_intern + "resource_traces.xlsx", index=False)
    # scenario descriptions
    scenario_descriptions.to_excel(file_directory_path_as_str_intern + "scenario_descriptions.xlsx", index=False)


def build_scenario_export_response(scenarios: dict[str, list[str]], multi_scenarios_handler,
                                   start_time=None, end_time=None, file_directory_path_as_str_extern=None):
    if file_directory_path_as_str_extern is not None:
        file_directory_path_as_str_intern = file_directory_path_as_str_extern
    else:
        file_directory_path_as_str_intern = file_directory_path_as_str

    print("Scenarios export with scenarios:", scenarios, start_time, end_time)
    scenarios = scenarios["scenarioIDs"]
    (orders, products, processes, resources, resource_utilization, resource_schedules, order_traces, resource_traces,
     scenario_descriptions) = (
        multi_scenarios_handler.get_kpi_for_scenarios(scenarios, start_time, end_time))

    store_xlsx_sheets(orders, products, processes, resources, resource_utilization, resource_schedules, order_traces, resource_traces,
                      scenario_descriptions, file_directory_path_as_str_intern)
    files = ["orders.xlsx", "products.xlsx", "processes.xlsx", "resources.xlsx", "resource_utilization.xlsx",
             "resource_schedules.xlsx", "order_traces.xlsx", "resource_traces.xlsx", "scenario_descriptions.xlsx"]

    # Zip the files into a single archive
    # project specific path would be better
    zip_file_path = (str(ROOT_PATH) +
                     '/planning_services/scenario_analytics/interface/scenario_data/scenario_comparison.zip')
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for file in files:
            zip_file.write(file_directory_path_as_str_intern + file,
                           arcname=file)

    return zip_file_path


def build_orders_response(start_time: datetime, end_time: datetime,
                          order_ids_list: list[int], product_ids_list: list[int],
                          process_ids_list: list[int], resource_ids_list: list[int],
                          units, scenario_handler: SingleScenarioHandler):
    if not order_ids_list:
        response = OrdersResponse()
        return response

    order_view = (
        scenario_handler.get_order_view(order_ids_list, product_ids_list,
                                                      process_ids_list, resource_ids_list,
                                                      start_time, end_time))
    response = get_orders_response(order_view)

    return response


def build_products_response(start_time: datetime, end_time: datetime,
                            order_ids_list: list[int], product_ids_list: list[int],
                            process_ids_list: list[int], resource_ids_list: list[int],
                            units, scenario_handler: SingleScenarioHandler):
    product_view = (
        scenario_handler.get_product_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                          start_time, end_time))

    response = get_products_response(product_view)

    return response


def build_processes_response(start_time: datetime, end_time: datetime,
                             order_ids_list: list[int], product_ids_list: list[int],
                             process_ids_list: list[int], resource_ids_list: list[int],
                             units, scenario_handler: SingleScenarioHandler):
    process_view = (
        scenario_handler.get_process_view(order_ids_list, product_ids_list,
                                                        process_ids_list, resource_ids_list,
                                                        start_time, end_time))

    response = get_processes_response(process_view)

    return response

def build_resources_response(start_time: datetime, end_time: datetime,
                             order_ids_list: list[int], product_ids_list: list[int],
                             process_ids_list: list[int], resource_ids_list: list[int],
                             units, scenario_handler: SingleScenarioHandler):
    resource_view_df = (
        scenario_handler.get_resource_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                           start_time, end_time))

    response = get_resources_response(resource_view_df)

    return response


def build_resources_utilization_chart_response(start_time: datetime, end_time: datetime,
                                                     order_ids_list: list[int], product_ids_list: list[int],
                                                     process_ids_list: list[int], resource_ids_list: list[int],
                                                     bin_size: int, scenario_handler: SingleScenarioHandler):

    view = "RESOURCE"
    resource_id_utilization, mean, bin_size, title, x_label, y_label, reference_values = \
        scenario_handler.get_utilisation_chart(
            start_time=start_time, end_time=end_time,
            order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list,
            bin_size=bin_size, event_type="ACTUAL", view=view, resource_type="ALL", all=False)

    response = get_utilization_chart_response(resource_id_utilization, mean, bin_size, title, x_label, y_label,
                                              reference_values, start_time)

    return response