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

Create the responses associated with the kpi charts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ofact.helpers import colored_print
from ofact.planning_services.scenario_analytics.interface.helpers import argument_preparation, get_datetime


if TYPE_CHECKING:
    from datetime import datetime
    from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler


def get_unpacked_args_charts(args):
    start_time = get_datetime(args["dateStart"])
    end_time = get_datetime(args["dateEnd"])
    order_ids_list = args["orders"]
    product_ids_list = args["products"]
    process_ids_list = args["processes"]
    resource_ids_list = args["resources"]
    bin_size = args["bin"]
    scenario = args["scenario"]
    return start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, bin_size, \
        scenario


# ==Response builder class
class UtilizationChartResponse:

    def __init__(self):
        self.resources = []
        self.mean = None
        self.bin_size = None
        self.title = None
        self.x_label = None
        self.y_label = None

    @argument_preparation
    def add_chart_coordinates(self, id_: str = "-", reference_value: str = "-", coordinates: list = []):
        new = {"id": id_,
               "reference_value": reference_value,
               "coordinates": coordinates}
        self.resources.append(new)

    @argument_preparation
    def add_mean(self, mean: list = []):
        self.mean = mean

    @argument_preparation
    def add_bin_size(self, bin_size: int = 1):
        self.bin_size = bin_size

    @argument_preparation
    def add_title(self, title: str = ""):
        self.title = title

    @argument_preparation
    def add_x_label(self, x_label: str = ""):
        self.x_label = x_label

    @argument_preparation
    def add_y_label(self, y_label: str = ""):
        self.y_label = y_label

    def get_response_dict(self):
        return {"resources": self.resources,
                "mean": self.mean,
                "bin_size": self.bin_size,
                "title": self.title,
                "x_label": self.x_label,
                "y_label": self.y_label}


# ==build response
# @profile
async def build_resources_utilization_chart_response(start_time: datetime, end_time: datetime,
                                                     order_ids_list: list[int], product_ids_list: list[int],
                                                     process_ids_list: list[int], resource_ids_list: list[int],
                                                     bin_size: int, scenario_handler: SingleScenarioHandler) -> (
        dict, UtilizationChartResponse):

    view = "RESOURCE"
    resource_id_utilization, mean, bin_size, title, x_label, y_label, reference_values = \
        await scenario_handler.get_utilization_chart_awaited(
            start_time=start_time, end_time=end_time,
            order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list,
            bin_size=bin_size, event_type="ACTUAL", view=view, resource_type="ALL", all=False)

    response = get_utilization_chart_response(resource_id_utilization, mean, bin_size, title, x_label, y_label,
                                              reference_values, start_time)

    return response


def get_utilization_chart_response(resource_id_utilization, mean, bin_size, title, x_label, y_label, reference_values,
                                   start_time):
    response = UtilizationChartResponse()
    for resource_id, coordinates in resource_id_utilization.items():
        if resource_id not in reference_values:
            continue
        reference_value = reference_values[resource_id]
        if not coordinates:
            coordinates = [(start_time, 0)]
        response.add_chart_coordinates(id_=resource_id, reference_value=reference_value, coordinates=coordinates)
    response.add_mean(mean)
    response.add_bin_size(bin_size)
    response.add_title(title)
    response.add_x_label(x_label)
    response.add_y_label(y_label)

    return response


# ==Response builder class
class LeadTimeChartResponse:
    def __init__(self):
        self.resources = []
        self.mean = None
        self.bin_size = None
        self.title = None
        self.x_label = None
        self.y_label = None

    @argument_preparation
    def add_chart_coordinates(self, id_: str = "-", reference_value: str = "-", coordinates: list = []):
        new = {"id": id_,
               "reference_value": reference_value,
               "coordinates": coordinates}
        self.resources.append(new)

    @argument_preparation
    def add_mean(self, mean: list = []):
        self.mean = mean

    @argument_preparation
    def add_bin_size(self, bin_size: int = 1):
        self.bin_size = bin_size

    @argument_preparation
    def add_title(self, title: str = ""):
        self.title = title

    @argument_preparation
    def add_x_label(self, x_label: str = ""):
        self.x_label = x_label

    @argument_preparation
    def add_y_label(self, y_label: str = ""):
        self.y_label = y_label

    def get_response_dict(self):
        return {"resources": self.resources,
                "mean": self.mean,
                "bin_size": self.bin_size,
                "title": self.title,
                "x_label": self.x_label,
                "y_label": self.y_label}


# ==build response
# @profile
async def build_lead_time_chart_response(args, multi_scenarios_handler, view) -> (dict, LeadTimeChartResponse):
    colored_print(f"[API] Build resources utilization chart response with args: {args}")

    start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, bin_size, scenario = \
        get_unpacked_args_charts(args)  # resource_type="ALL"

    scenario = args["scenario"]
    scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)
    kpi_administration = scenario_handler.get_kpi_administration()

    start_time, end_time = await scenario_handler.update_consideration_period(start_time, end_time)

    response = LeadTimeChartResponse()

    reference_values = kpi_administration.get_reference_value_by_id(resource_ids_list, view)

    id_lead_time_mean_d, title, x_label, y_label = \
        scenario_handler.get_lead_time_chart(start_time=start_time, end_time=end_time,
                                             order_ids=order_ids_list, product_ids=product_ids_list,
                                             process_ids=process_ids_list, resource_ids=resource_ids_list,
                                             bin_size=bin_size, event_type="ACTUAL",
                                             view=view, scenario=scenario)

    for object_id, coordinates in id_lead_time_mean_d.items():
        reference_value = reference_values[object_id]
        response.add_chart_coordinates(id_=object_id, reference_value=reference_value, coordinates=coordinates)

    response.add_bin_size(bin_size)
    response.add_title(title)
    response.add_x_label(x_label)
    response.add_y_label(y_label)

    return response
