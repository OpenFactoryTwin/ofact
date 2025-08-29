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

The idea behind the ScenarioHandler is to encapsulate all analytic elements that are required
to determine the responses requested from the Dashboard-UI.

classes:
    SingleScenarioHandler: responsible for a single scenario
"""

# Import Part 1: Standard Imports
from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

# Import Part 3: Project Imports
from ofact.planning_services.model_generation.persistence import get_state_model_file_path, deserialize_state_model
from ofact.planning_services.model_generation.resource_settings import get_full_time_equivalents_resources
from ofact.planning_services.scenario_analytics.business_logic.kpi.chart import AvailabilityChart, LeadTimeChart
from ofact.planning_services.scenario_analytics.business_logic.kpi.table import (
    KPIAdministration, LeadTime, DeliveryReliability, Quality, Inventory, Performance, Availability, ORE,
    LeadTimeBasedKPI)
from ofact.planning_services.scenario_analytics.business_logic.traces import OrderTrace, ResourceTrace
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase
from ofact.planning_services.scenario_analytics.interface.helpers import get_datetime
from ofact.planning_services.scenario_analytics.interface.services.scenario_export import (
    build_resources_utilization_chart_response, build_orders_response, build_products_response,
    build_processes_response, build_resources_response)
from ofact.planning_services.scenario_analytics.interface.services.kpi_tables import (
    build_filter_response)
from ofact.settings import ROOT_PATH
from ofact.twin.state_model.helpers.duplicate_digital_twin import get_digital_twin_model_duplicate

if TYPE_CHECKING:
    from datetime import datetime
    from ofact.twin.state_model.model import StateModel

class Units(Enum):
    DAY = "DAY"
    HOUR = "HOUR"
    MINUTE = "MINUTE"
    SECOND = "SECOND"


factor = {Units.DAY: 86400,
          Units.HOUR: 3600,
          Units.MINUTE: 60,
          Units.SECOND: 1}


class ScenarioDescription(Enum):
    RESOURCE_SCHEDULE = "RESOURCE_SCHEDULE"


class SingleScenarioHandler:
    next_id = 0
    basic_state_model = None

    def __init__(self,
                 state_model: Optional[StateModel] = None,
                 start_time_enrichment: Optional[datetime] = None,
                 end_time_enrichment: Optional[datetime] = None,
                 analytics_data_base: Optional[ScenarioAnalyticsDataBase] = None,
                 scenario_description: Optional[dict] = None,
                 project_path: Optional[Path] = None,
                 scenario_name: Optional[str] = None,
                 relevant_resource_types: Optional[list[str]] = None):
        """
        The single scenario handler is responsible for a single scenario.
        This includes the interface to the state model as well to the analytics part.
        Each scenario handler has its own ID. In contrast, the scenario ids from the frontend are not unique.
        The unique ID ensures that the cache from the kpi calculation can be used
        (since the cache is based on the scenario ID).

        Parameters
        ----------
        state_model : StateModel
        start_time_enrichment : datetime
        end_time_enrichment : datetime
        analytics_data_base : ScenarioAnalyticsDataBase
        scenario_description : dict
        project_path : Path
        scenario_name : str
        relevant_resource_types: list of resource types relevant for the evaluation
        """

        self.state_model = state_model

        if analytics_data_base is not None:
            self.analytics_data_base = analytics_data_base

        else:
            analytics_data_base = ScenarioAnalyticsDataBase(state_model)
            self.analytics_data_base = analytics_data_base

        self.kpi_administration = KPIAdministration(analytics_data_base=analytics_data_base)

        lead_time_based_kpi = LeadTimeBasedKPI(analytics_data_base=analytics_data_base)
        self.lead_time = LeadTime(analytics_data_base=self.analytics_data_base)
        self.performance = Performance(analytics_data_base=analytics_data_base)
        self.availability = Availability(analytics_data_base=analytics_data_base)
        self.quality = Quality(analytics_data_base=analytics_data_base)
        self.inventory = Inventory(analytics_data_base=analytics_data_base)
        self.delivery_reliability = DeliveryReliability(analytics_data_base=analytics_data_base)
        self.ore = ORE()

        if relevant_resource_types is None:
            # "StationaryResource" iot_example case
            relevant_resource_types = ["ActiveMovingResource", "WorkStation"]  # should be project specific
        self.relevant_resource_types = relevant_resource_types
        self.utilization_chart = AvailabilityChart(analytics_data_base=analytics_data_base,
                                                   accepted_resource_types=self.relevant_resource_types)
        self.lead_time_chart = LeadTimeChart(analytics_data_base=analytics_data_base)

        self.state_model = state_model

        self.start_time_enrichment = start_time_enrichment
        self.end_time_enrichment = end_time_enrichment

        self.id = SingleScenarioHandler.next_id
        SingleScenarioHandler.next_id += 1
        self.scenario_name = scenario_name

        self.units = {"Order Lead Time": Units.MINUTE,
                      "Order Waiting Time": Units.MINUTE}

        self.update_closed = None
        self.project_path = project_path
        if scenario_description is None:
            scenario_description = {}
        self.scenario_description: dict[ScenarioDescription, object] = scenario_description

        self.schedules = self.get_schedules(start_time_enrichment)

    def set_scenario_name(self, scenario_name):
        if scenario_name == "current_state":
            scenario_name += str(self.id)

        self.scenario_name = scenario_name

    def get_scenario_name(self):
        return self.scenario_name

    def get_kpi_administration(self):
        return self.kpi_administration

    def get_ore(self):
        return self.ore

    async def get_time_period(self, start_time, end_time):
        """
        Sets new start and end time in Time Period if they are not already there
        Note: Ensure that the start and end time are already determined
        (adaption of the consideration period is possible because data is not available)

        Parameters
        ----------
        start_time: Start Time from Frontend
        end_time: End Time from Frontend
        """
        kpi_administration = self.get_kpi_administration()
        if (start_time, end_time) not in kpi_administration.time_period:
            if self.update_closed is None:
                if self.update_closed is None:
                    loop = asyncio.get_running_loop()
                    self.update_closed = loop.create_future()

            while True:  # ToDo: the standard way of awaiting the future does not work - why???
                if self.update_closed is None:
                    break
                try:
                    if self.update_closed.done():
                        break
                except AttributeError:
                    pass

        result = kpi_administration.time_period[(start_time, end_time)]

        return result

    async def update_consideration_period(self, start_time, end_time):

        start_time, end_time = self.kpi_administration.update_consideration_period(start_time, end_time)

        if self.update_closed is not None:
            self.update_closed.set_result(True)

            self.update_closed = None

        return start_time, end_time

    def get_filter_options(self, start_time: float, end_time: float):
        return self.kpi_administration.get_filter_options(start_time, end_time)

    async def get_order_view_awaited(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                     start_time, end_time, all: bool = False):

        if not all:
            start_time, end_time = await self.update_consideration_period(start_time, end_time)
        else:
            start_time, end_time = await self.get_time_period(start_time, end_time)

        return self.get_order_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                   start_time, end_time, all)

    async def get_product_view_awaited(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                       start_time, end_time, all: bool = False):

        if not all:
            start_time, end_time = await self.update_consideration_period(start_time, end_time)
        else:
            start_time, end_time = await self.get_time_period(start_time, end_time)

        return self.get_product_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                     start_time, end_time, all)

    async def get_process_view_awaited(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                       start_time, end_time, all: bool = False):

        if not all:
            start_time, end_time = await self.update_consideration_period(start_time, end_time)
        else:
            start_time, end_time = await self.get_time_period(start_time, end_time)

        return self.get_process_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                     start_time, end_time, all)

    async def get_resource_view_awaited(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                        start_time, end_time, all: bool = False):

        if not all:
            start_time, end_time = await self.update_consideration_period(start_time, end_time)
        else:
            start_time, end_time = await self.get_time_period(start_time, end_time)

        return self.get_order_view(order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                   start_time, end_time, all)

    def get_order_view(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                       start_time, end_time, all: bool = False):
        view = "ORDER"

        kpi_requested = ["number_of_pieces_absolute_objects", "number_of_pieces_relative_objects",
                         "customer", "order_status", "start_end_time",
                         "delivery_reliability", "delivery_delay", "lead_time", "quality", "performance",
                         "source"]
        order_view_df = self._get_view_df(start_time, end_time,
                                          order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                          view, all, kpi_requested)

        return order_view_df

    def get_product_view(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                         start_time, end_time, all: bool = False):
        view = "PRODUCT"

        kpi_requested = ["number_of_pieces_absolute_objects", "number_of_pieces_relative_objects",
                         "target_quantity", "difference_percentage_products",
                         "delivery_reliability", "lead_time", "quality", "inventory", "performance",
                         "source"]
        product_view_df = self._get_view_df(start_time, end_time,
                                            order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                            view, all, kpi_requested)

        return product_view_df

    def get_process_view(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                         start_time, end_time, all: bool = False):
        view = "PROCESS"

        kpi_requested = ["number_of_pieces_absolute_pe", "number_of_pieces_relative_pe",
                         "delivery_reliability", "lead_time", "quality", "performance", "source"]
        process_view_df = self._get_view_df(start_time, end_time,
                                            order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                            view, all, kpi_requested)

        return process_view_df

    def get_resource_view(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                          start_time, end_time, all: bool = False):
        view = "RESOURCE"

        kpi_requested = ["number_of_pieces_absolute_pe", "number_of_pieces_relative_pe",
                         "delivery_reliability", "lead_time", "quality", "inventory", "performance", "utilisation",
                         "availability", "ore",
                         "source"]
        process_view_df = self._get_view_df(start_time, end_time,
                                            order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                            view, all, kpi_requested)

        return process_view_df

    def _get_view_df(self, start_time, end_time,
                     order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                     view, all, kpi_requested) -> pd.DataFrame:
        order_ids_list, product_ids_list, process_ids_list, resource_ids_list = (
            self.kpi_administration.filter_view_ids(order_ids_list, product_ids_list,
                                                    process_ids_list, resource_ids_list))

        view_kpis = []
        if not all:
            match view:
                case "ORDER":
                    reference_values = self.kpi_administration.get_reference_value_by_id(order_ids_list, view)
                case "PRODUCT":
                    reference_values = self.kpi_administration.get_reference_value_by_id(product_ids_list, view)
                case "PROCESS":
                    reference_values = self.kpi_administration.get_reference_value_by_id(process_ids_list, view)
                case "RESOURCE":
                    reference_values = self.kpi_administration.get_reference_value_by_id(resource_ids_list, view)
                case _:
                    raise ValueError(f"View {view} not supported")
            reference_values_df = reference_values.to_frame()
            view_kpis.append(reference_values_df)

        if "customer" in kpi_requested:
            if not all:
                customer_name_df = self.kpi_administration.get_customer_name_by_id(order_ids_list, view)
            else:
                customer_name_df = pd.DataFrame(["Alle"], columns=["Customer Name"])

            view_kpis.append(customer_name_df)

        if "number_of_pieces_absolute_objects" in kpi_requested:
            number_of_pieces_absolute = self.get_amount_objects(start_time, end_time,
                                                                order_ids_list, product_ids_list,
                                                                process_ids_list, resource_ids_list,
                                                                event_type="ACTUAL", view=view, selection=True,
                                                                all=all)
            number_of_pieces_absolute_df = number_of_pieces_absolute.to_frame("Number of Pieces Absolute")
            view_kpis.append(number_of_pieces_absolute_df)

        if "number_of_pieces_absolute_pe" in kpi_requested:
            absolute_frequency = self.get_amount_pe(start_time=start_time, end_time=end_time,
                                                    order_ids_list=order_ids_list,
                                                    product_ids_list=product_ids_list,
                                                    process_ids_list=process_ids_list,
                                                    resource_ids_list=resource_ids_list,
                                                    event_type="ACTUAL", view=view,
                                                    selection=True, all=all)
            absolute_frequency_df = absolute_frequency.to_frame("count")
            view_kpis.append(absolute_frequency_df)

        if "number_of_pieces_relative_objects" in kpi_requested:
            number_of_pieces_relative = self.get_relative_objects(start_time, end_time,
                                                                  order_ids_list, product_ids_list,
                                                                  process_ids_list, resource_ids_list,
                                                                  event_type="ACTUAL", view=view, all=all)
            number_of_pieces_relative_df = number_of_pieces_relative.to_frame("Number of Pieces Relative")
            view_kpis.append(number_of_pieces_relative_df)

        if "number_of_pieces_relative_pe" in kpi_requested:
            relative = self.get_relative_pe(start_time=start_time, end_time=end_time,
                                            order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                            process_ids_list=process_ids_list,
                                            resource_ids_list=resource_ids_list,
                                            event_type="ACTUAL", view=view, all=all)
            relative_df = relative.to_frame("relative")
            view_kpis.append(relative_df)

        if "difference_percentage_products" in kpi_requested:
            difference_percentage_products = self.get_difference_percentage(start_time, end_time,
                                                                            order_ids_list, product_ids_list,
                                                                            process_ids_list, resource_ids_list,
                                                                            "ACTUAL", view, all=all)
            difference_percentage_df = pd.DataFrame(difference_percentage_products.values(),
                                                    columns=["Difference Percentage"],
                                                    index=difference_percentage_products.keys())
            view_kpis.append(difference_percentage_df)

        if "target_quantity" in kpi_requested:
            target_quantity_products = self.get_target_quantity(start_time, end_time,
                                                                order_ids_list, product_ids_list,
                                                                process_ids_list, resource_ids_list, all=all)
            target_quantity_df = target_quantity_products.to_frame("Target Quantity")
            view_kpis.append(target_quantity_df)

        if "order_status" in kpi_requested:
            if not all:
                order_status = self.get_order_status(start_time, end_time, order_ids_list)
                order_status_df = order_status.to_frame("Order Status")
            else:
                order_status_df = pd.DataFrame([None], columns=["Order Status"])

            view_kpis.append(order_status_df)

        if "start_end_time" in kpi_requested:
            start_end_time_df = self.get_start_end_time_order(start_time, end_time,
                                                              order_ids_list, product_ids_list, process_ids_list,
                                                              resource_ids_list,
                                                              event_type="ACTUAL", view=view, all=all)
            view_kpis.append(start_end_time_df)

        if "delivery_reliability" in kpi_requested:
            kpi_delivery_reliability = \
                self.get_delivery_reliability(start_time=start_time, end_time=end_time,
                                              order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                              process_ids_list=process_ids_list,
                                              resource_ids_list=resource_ids_list,
                                              view=view, all=all)
            kpi_delivery_reliability_df = kpi_delivery_reliability.to_frame("Delivery Reliability")
            view_kpis.append(kpi_delivery_reliability_df)

        if "delivery_delay" in kpi_requested:
            kpi_delivery_delay = \
                self.get_delivery_delay(start_time, end_time,
                                        order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                        view=view, all=all)
            kpi_delivery_delay_df = kpi_delivery_delay.to_frame("Delivery Delay")
            view_kpis.append(kpi_delivery_delay_df)

        if "lead_time" in kpi_requested:
            lead_time_kpis_df = (
                self.get_lead_time(start_time, end_time,
                                   order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                   "ACTUAL", view, all=all))
            view_kpis.append(lead_time_kpis_df)

        if "quality" in kpi_requested:
            kpi_quality_df = self.get_quality(start_time=start_time, end_time=end_time,
                                              order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                              process_ids_list=process_ids_list,
                                              resource_ids_list=resource_ids_list,
                                              event_type="ACTUAL", view=view, all=all)
            view_kpis.append(kpi_quality_df)

        if "performance" in kpi_requested:
            kpi_performance_df = self.get_performance(start_time=start_time, end_time=end_time,
                                                      order_ids_list=order_ids_list,
                                                      product_ids_list=product_ids_list,
                                                      process_ids_list=process_ids_list,
                                                      resource_ids_list=resource_ids_list,
                                                      view=view, all=all)
            view_kpis.append(kpi_performance_df)

        if "inventory" in kpi_requested:
            kpi_inventory = self.get_inventory(start_time=start_time, end_time=end_time,
                                               order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                               process_ids_list=process_ids_list,
                                               resource_ids_list=resource_ids_list,
                                               event_type="ACTUAL", view=view, all=all)
            inventory_df = kpi_inventory.to_frame("Inventory")
            view_kpis.append(inventory_df)

        if "utilisation" in kpi_requested:
            kpi_utilisation = self.get_utilisation(start_time=start_time, end_time=end_time,
                                                   order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                                   process_ids_list=process_ids_list,
                                                   resource_ids_list=resource_ids_list,
                                                   event_type="ACTUAL", view=view, all=all)
            kpi_utilisation_df = kpi_utilisation.to_frame("utilisation")
            view_kpis.append(kpi_utilisation_df)

        if "availability" in kpi_requested:
            kpi_availability = (
                self.get_availability(start_time=start_time, end_time=end_time,
                                      order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                      process_ids_list=process_ids_list, resource_ids_list=resource_ids_list,
                                      event_type="ACTUAL", view=view, all=all))
            kpi_availability_df = kpi_availability.to_frame("Availability")
            view_kpis.append(kpi_availability_df)

        if "ore" in kpi_requested:
            kpi_ore = self.ore.get_ore(kpi_quality_df, kpi_performance_df, kpi_availability)
            kpi_ore_df = kpi_ore.to_frame("ORE")
            view_kpis.append(kpi_ore_df)

        if "source" in kpi_requested:
            kpi_source_df = self.get_source(start_time=start_time, end_time=end_time,
                                         order_ids_list=order_ids_list, product_ids_list=product_ids_list,
                                         process_ids_list=process_ids_list,
                                         resource_ids_list=resource_ids_list,
                                         event_type="ACTUAL", view=view, all=all)
            view_kpis.append(kpi_source_df)

        view_df = pd.concat(view_kpis, axis=1)

        view_df = view_df.loc[:, ~view_df.columns.duplicated()]  # drop duplicated columns

        time_columns = ["Start Time [s]", "End Time [s]", "Planned End Time [s]"]
        for time_column in time_columns:
            if time_column not in view_df.columns:
                continue

            view_df.loc[view_df[time_column] == view_df[time_column], time_column] -= 7200

        return view_df

    def get_amount_objects(self, start_time, end_time,
                           order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                           event_type, view, selection, all=False):
        scenario = self.id
        return self.kpi_administration.get_amount_objects(start_time, end_time,
                                                          order_ids_list, product_ids_list, process_ids_list,
                                                          resource_ids_list,
                                                          event_type, view, selection, scenario, all)

    def get_relative_objects(self, start_time, end_time,
                             order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                             event_type, view, all=False):
        scenario = self.id
        return self.kpi_administration.get_relative_objects(start_time, end_time,
                                                            order_ids_list, product_ids_list, process_ids_list,
                                                            resource_ids_list,
                                                            event_type, view, scenario, all)

    def get_amount_pe(self, start_time, end_time,
                      order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                      event_type, view, selection, all=False):
        scenario = self.id
        return self.kpi_administration.get_amount_pe(start_time, end_time,
                                                     order_ids_list, product_ids_list, process_ids_list,
                                                     resource_ids_list,
                                                     event_type=event_type, view=view, selection=selection,
                                                     scenario=scenario, all=all)

    def get_relative_pe(self, start_time, end_time,
                        order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                        event_type, view, all=False):
        scenario = self.id
        return self.kpi_administration.get_relative_pe(start_time=start_time, end_time=end_time,
                                                       order_ids=order_ids_list, part_ids=product_ids_list,
                                                       process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                       event_type=event_type, view=view, scenario=scenario, all=all)

    def get_target_quantity(self, start_time, end_time,
                            order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                            all=False):
        scenario = self.id
        return self.kpi_administration.get_order_target_quantity(
            start_time=start_time, end_time=end_time,
            order_ids=order_ids_list, part_ids=product_ids_list,
            process_ids=process_ids_list, resource_ids=resource_ids_list,
            scenario=scenario, all=all)

    def get_difference_percentage(self, start_time, end_time,
                                  order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                  event_type, view, all=False):
        scenario = self.id
        return self.kpi_administration.get_difference_percentage(start_time, end_time,
                                                                 order_ids_list, product_ids_list,
                                                                 process_ids_list, resource_ids_list,
                                                                 event_type, view, scenario=scenario, all=all)

    def get_order_status(self, start_time, end_time, order_ids_list):
        scenario = self.id
        return self.kpi_administration.get_order_status(start_time, end_time, order_ids_list, scenario)

    def get_start_end_time_order(self, start_time, end_time,
                                 order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                 event_type, view, all=False):
        scenario = self.id
        return self.lead_time.get_start_end_time_order(start_time, end_time,
                                                       order_ids_list, product_ids_list, process_ids_list,
                                                       resource_ids_list,
                                                       event_type, view, scenario, all=all)

    def get_delivery_reliability(self, start_time, end_time, order_ids_list, product_ids_list, process_ids_list,
                                 resource_ids_list, view, all=False):
        scenario = self.id
        return self.delivery_reliability.get_delivery_reliability(start_time, end_time,
                                                                  order_ids_list, product_ids_list, process_ids_list,
                                                                  resource_ids_list,
                                                                  view, scenario, all=all)

    def get_delivery_delay(self, start_time, end_time,
                           order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                           view, all=False):
        scenario = self.id
        return self.delivery_reliability.get_delivery_delay(start_time, end_time,
                                                            order_ids_list, product_ids_list, process_ids_list,
                                                            resource_ids_list,
                                                            view, scenario, all=all)

    def get_lead_time(self, start_time, end_time,
                      order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                      event_type, view, all=False):
        scenario = self.id
        lead_time = self.lead_time.get_lead_time(start_time, end_time,
                                                 order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                                                 event_type, view, all=all, scenario=scenario)
        lead_time = lead_time.copy()  # Note: memoization is adapted (corrupted) with each call
        if view == "ORDER":
            if "Order Lead Time" in self.units and "total_lead_time_wt" in lead_time:
                lead_time["total_lead_time_wt"] /= factor[self.units["Order Lead Time"]]
            if "Order Waiting Time" in self.units and "total_waiting_time" in lead_time:
                lead_time["total_waiting_time"] /= factor[self.units["Order Waiting Time"]]

        return lead_time

    def get_quality(self, start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                    event_type, view, all=False):
        scenario = self.id
        return self.quality.get_quality(start_time=start_time, end_time=end_time,
                                        order_ids=order_ids_list, part_ids=product_ids_list,
                                        process_ids=process_ids_list, resource_ids=resource_ids_list,
                                        event_type=event_type, view=view, all=all, scenario=scenario)

    def get_inventory(self, start_time, end_time,
                      order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                      event_type, view, all=False):
        scenario = self.id
        return self.inventory.get_inventory(start_time=start_time, end_time=end_time,
                                            order_ids=order_ids_list, part_ids=product_ids_list,
                                            process_ids=process_ids_list, resource_ids=resource_ids_list,
                                            event_type=event_type, view=view, all=all, scenario=scenario)

    def get_performance(self, start_time, end_time,
                        order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                        view, all=False):
        scenario = self.id
        return self.performance.get_performance(start_time=start_time, end_time=end_time,
                                                order_ids=order_ids_list, part_ids=product_ids_list,
                                                process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                view=view, all=all, scenario=scenario)

    def get_utilisation(self, start_time, end_time,
                        order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                        event_type, view, all=False):
        return self.availability.get_utilisation(start_time=start_time, end_time=end_time,
                                                 order_ids=order_ids_list, part_ids=product_ids_list,
                                                 process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                 event_type=event_type, view=view, scenario=self.id, all=all)

    def get_availability(self, start_time, end_time,
                         order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                         event_type, view, all=False):
        return self.availability.get_availability(start_time=start_time, end_time=end_time,
                                                  order_ids=order_ids_list, part_ids=product_ids_list,
                                                  process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                  event_type=event_type, view=view, scenario=self.id, all=all)

    def get_source(self, start_time, end_time,
                   order_ids_list, product_ids_list, process_ids_list, resource_ids_list,
                   event_type, view, all=False):
        scenario = self.id
        return self.kpi_administration.get_source(start_time=start_time, end_time=end_time,
                                                  order_ids=order_ids_list, part_ids=product_ids_list,
                                                  process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                  event_type=event_type, view=view, scenario=scenario, all=all)

    def get_lead_time_chart(self, start_time, end_time, order_ids_list, product_ids_list, process_ids_list,
                            resource_ids_list, bin_size, event_type, view):
        scenario = self.id
        return self.lead_time_chart.get_lead_time_chart(start_time=start_time, end_time=end_time,
                                                        order_ids=order_ids_list, product_ids=product_ids_list,
                                                        process_ids=process_ids_list, resource_ids=resource_ids_list,
                                                        bin_size=bin_size, event_type=event_type,
                                                        view=view, scenario=scenario)

    async def get_utilization_chart_awaited(self, start_time, end_time, order_ids_list,
                                            product_ids_list, process_ids_list,
                                            resource_ids_list, bin_size, event_type, view, resource_type, all):

        start_time, end_time = await self.get_time_period(start_time, end_time)

        return self.get_utilisation_chart(start_time, end_time, order_ids_list, product_ids_list, process_ids_list,
                                         resource_ids_list, bin_size, event_type, view, resource_type, all)

    def get_utilisation_chart(self, start_time, end_time, order_ids_list, product_ids_list, process_ids_list,
                              resource_ids_list, bin_size, event_type, view, resource_type, all):
        reference_values = self.kpi_administration.get_reference_value_by_id(resource_ids_list, view)

        scenario = self.id


        resource_id_utilization, mean, bin_size, title, x_label, y_label = (
            self.utilization_chart.get_utilisation_chart(start_time=start_time, end_time=end_time,
                                                            order_ids=order_ids_list, product_ids=product_ids_list,
                                                            process_ids=process_ids_list,
                                                            resource_ids=resource_ids_list,
                                                            bin_size=bin_size, event_type=event_type,
                                                            view=view, resource_type=resource_type, scenario=scenario,
                                                            all=all))

        return resource_id_utilization, mean, bin_size, title, x_label, y_label, reference_values

    def get_schedules(self, start_time):
        if start_time is None:
            return pd.DataFrame()

        relevant_resources = []
        for resource_type in self.relevant_resource_types:
            relevant_resources += self.state_model.get_objects_by_class_name(resource_type)

        schedules = []
        for resource in relevant_resources:
            schedule = resource.process_execution_plan.get_schedule(start_time)
            if schedule.empty:
                continue
            schedule["Name"] = resource.name
            schedule["ID"] = resource.identification
            schedules.append(schedule)

        schedule_df = pd.concat(schedules)
        schedule_df.reset_index(inplace=True, drop=True)

        return schedule_df

    def get_state_model(self):
        return self.state_model

    def get_state_model_start_time(self):
        return self.state_model.process_executions["Executed Start Time"][0]

    def get_state_model_end_time(self):
        return self.state_model.process_executions["Executed End Time"][-1]

    def get_start_time_enrichment(self):
        return self.start_time_enrichment

    def get_end_time_enrichment(self):
        return self.end_time_enrichment

    def update_start_time_enrichment(self, date_start):
        self.start_time_enrichment = date_start

    def update_end_time_enrichment(self, date_end):
        self.end_time_enrichment = date_end

    def set_empty_scenario(self, project_path, digital_twin_file_path,
                           scenario_description: Optional[dict] = None, start_time = None):
        # iot_example
        # import_state_model(source_file_path=digital_twin_file_path)

        state_model_file_path = get_state_model_file_path(project_path=project_path,
                                                          state_model_file_path=digital_twin_file_path)
        state_model = deserialize_state_model(state_model_file_path, deserialization_required=False)
        self.set_up_scenario(state_model=state_model, scenario_description=scenario_description,
                             project_path=project_path, start_time=start_time)

    def set_up_scenario(self, state_model, scenario_description: Optional[dict] = None,
                        project_path: Optional[str] = None, set_up_kpi: bool = True, start_time = None):
        """
        Set up the scenario with the given state model and their kpi calculation objects.

        Parameters
        ----------
        state_model : The state model of the scenario
        scenario_description : dict
        project_path : Project path
        set_up_kpi : bool
        """

        self.state_model = state_model

        if set_up_kpi:
            self.analytics_data_base = ScenarioAnalyticsDataBase(state_model)
            self.kpi_administration = KPIAdministration(analytics_data_base=self.analytics_data_base)

            lead_time_based_kpi = LeadTimeBasedKPI(analytics_data_base=self.analytics_data_base)
            self.lead_time = LeadTime(analytics_data_base=self.analytics_data_base)
            self.performance = Performance(analytics_data_base=self.analytics_data_base)
            self.availability = Availability(analytics_data_base=self.analytics_data_base)
            self.quality = Quality(analytics_data_base=self.analytics_data_base)
            self.inventory = Inventory(analytics_data_base=self.analytics_data_base)
            self.delivery_reliability = DeliveryReliability(analytics_data_base=self.analytics_data_base)
            self.ore = ORE()

            self.utilization_chart = AvailabilityChart(analytics_data_base=self.analytics_data_base,
                                                       accepted_resource_types=self.relevant_resource_types)
            self.lead_time_chart = LeadTimeChart(analytics_data_base=self.analytics_data_base)

            if scenario_description is not None:
                for key, value in scenario_description.items():
                    self.scenario_description[key] = value
            self.project_path = project_path
            self.schedules = self.get_schedules(start_time)
        else:
            self.kpi_administration = None
            self.lead_time = None
            self.delivery_reliability = None
            self.quality = None
            self.inventory = None
            self.performance = None
            self.availability = None
            self.ore = None
            self.utilization_chart = None
            self.lead_time_chart = None
            if not hasattr(self, "scenario_description"):
                self.scenario_description = {}
            if not hasattr(self, "project_path"):
                self.project_path = None
            if not hasattr(self, "schedules"):
                self.schedules = pd.DataFrame()

    def initialize_simulation_scenario(self, scenario_description: Optional[dict] = None,
                                       project_path: Optional[str] = None, start_time = None):
        """Update of the simulation scenario"""

        # also persisted scenarios could be included
        self.set_up_scenario(state_model=self.state_model, scenario_description=scenario_description,
                             project_path=project_path, start_time=start_time)

    def update_state_model(self, get_digital_twin_updated_func, project_path,
                           start_datetime, end_datetime,
                           progress_tracker, data_source_model_path):
        self.update_start_time_enrichment(start_datetime)
        self.update_end_time_enrichment(end_datetime)

        start_datetime = get_datetime(start_datetime)
        end_datetime = get_datetime(end_datetime)

        self.state_model = \
            get_digital_twin_updated_func(root_path=ROOT_PATH, project_path=project_path,
                                          digital_twin_model=self.state_model,
                                          start_datetime=start_datetime, end_datetime=end_datetime,
                                          progress_tracker=progress_tracker,
                                          data_source_model_path=data_source_model_path)

        self.update_data()

    def update_data(self):
        self.kpi_administration.analytics_data_base.update_data()

    def get_analytics_data_base(self):
        return self.kpi_administration.analytics_data_base

    def get_duplicated_scenario_handler(self) -> SingleScenarioHandler:
        """
        Duplicate the scenario without initializing of the kpi tables.
        This must be done afterward, before starting with the kpi calculations.
        """

        duplicated_state_model = get_digital_twin_model_duplicate(self.state_model)

        duplicated_scenario_handler = SingleScenarioHandler()
        duplicated_scenario_handler.set_up_scenario(state_model=duplicated_state_model,
                                                    set_up_kpi=False)

        return duplicated_scenario_handler

    # get all kpis
    def get_all_kpis(self, start_time: datetime = None, end_time: datetime = None):
        """Get all kpis of the scenario"""

        if start_time is None:
            scenario_start_time = self.get_state_model_start_time()
        else:
            scenario_start_time = np.datetime64(start_time, "ns")

        if end_time is None:
            scenario_end_time = self.get_state_model_end_time()
        else:
            scenario_end_time = np.datetime64(end_time, "ns")

        args = {"dateStart": scenario_start_time,
                "dateEnd": scenario_end_time}

        filter_response = build_filter_response(args,
                                                multi_scenarios_handler=None, scenario_handler=self)
        filter_response_dict = filter_response.get_response_dict_ids()
        orders_list = filter_response_dict["orders"]
        products_list = filter_response_dict["products"]
        processes_list = filter_response_dict["processes"]
        resources_list = filter_response_dict["resources"]
        bin_size = 1
        start_time_converted = (scenario_start_time -
                                np.timedelta64(1, "s")).astype('datetime64[s]').astype('int')
        start_time_converted -= 7200  # timezone
        end_time_converted = (scenario_end_time.astype('datetime64[s]') +
                              np.timedelta64(1, "s")).astype('int')
        end_time_converted -= 7200  # timezone

        start_time_converted = max(start_time_converted, 0)  # to avoid -1 as start time
        start_time_converted, end_time_converted = (
            self.kpi_administration.update_consideration_period(start_time_converted, end_time_converted))

        order_units = {"units": {"deliveryDelay": "minute", "totalLeadTime": "minute", "totalWaitingTime": "minute"}}
        order_response = build_orders_response(
            start_time=start_time_converted, end_time=end_time_converted,
            order_ids_list=orders_list, product_ids_list=products_list,
            process_ids_list=processes_list, resource_ids_list=resources_list, units=order_units,
            scenario_handler=self)
        products_units = {"units": {"leadTime": "minute", "totalLeadTime": "minute",
                                    "waitingTime": "minute", "totalWaitingTime": "minute"}}
        products_response = build_products_response(
            start_time=start_time_converted, end_time=end_time_converted,
            order_ids_list=orders_list, product_ids_list=products_list,
            process_ids_list=processes_list, resource_ids_list=resources_list, units=products_units,
            scenario_handler=self)
        processes_units = {"units": {"leadTime": "minute", "waitingTime": "minute", "minLeadTime": "minute",
                                     "minWaitingTime": "minute", "maxLeadTime": "minute", "maxWaitingTime": "minute"}}
        processes_response = build_processes_response(
            start_time=start_time_converted, end_time=end_time_converted,
            order_ids_list=orders_list, product_ids_list=products_list,
            process_ids_list=processes_list, resource_ids_list=resources_list, units=processes_units,
            scenario_handler=self)
        resources_units = {"units": {"leadTime": "minute", "waitingTime": "minute"}}
        resources_response = build_resources_response(
            start_time=start_time_converted, end_time=end_time_converted,
            order_ids_list=orders_list, product_ids_list=products_list,
            process_ids_list=processes_list, resource_ids_list=resources_list, units=resources_units,
            scenario_handler=self)

        utilization_chart = build_resources_utilization_chart_response(
            start_time=start_time_converted, end_time=end_time_converted,
            order_ids_list=orders_list, product_ids_list=products_list,
            process_ids_list=processes_list, resource_ids_list=resources_list,
            bin_size=bin_size, scenario_handler=self)

        orders_df = pd.DataFrame(order_response.get_response_dict()["orders"])
        products_df = pd.DataFrame(products_response.get_response_dict()["products"])
        processes_df = pd.DataFrame(processes_response.get_response_dict()["processes"])
        resources_df = pd.DataFrame(resources_response.get_response_dict()["resources"])
        resource_utilization_df = pd.DataFrame(utilization_chart.get_response_dict()["resources"],
                                               columns=['id', 'reference_value', 'coordinates'])

        # ToDo: values in minutes (not seconds ...)  - ToDo

        resource_utilization_df = resource_utilization_df.explode('coordinates')
        resource_utilization_df.reset_index(inplace=True)
        resource_utilization_df[["Time", "Capacity Utilization"]] = (
            pd.DataFrame(resource_utilization_df["coordinates"].to_list(),
                         columns=["Time", "Capacity Utilization"]))
        resource_utilization_df.drop(columns="coordinates", inplace=True)
        if "index" in resource_utilization_df.columns:
            resource_utilization_df.drop(columns=["index"], inplace=True)

        scenario_name = self.get_scenario_name()
        orders_df["scenario"] = scenario_name
        products_df["scenario"] = scenario_name
        processes_df["scenario"] = scenario_name
        resources_df["scenario"] = scenario_name
        resource_utilization_df["scenario"] = scenario_name

        return orders_df, products_df, processes_df, resources_df, resource_utilization_df

    def get_resource_schedules(self):
        if self.schedules.empty:
            return None

        self.schedules["scenario"] = self.get_scenario_name()
        return self.schedules

    def get_full_time_equivalent_schedule(self):
        """Get the scenario schedule"""
        if ScenarioDescription.RESOURCE_SCHEDULE not in self.scenario_description:
            return None

        schedule_file = self.scenario_description[ScenarioDescription.RESOURCE_SCHEDULE]
        schedule_path = Path(self.project_path, f'models/worker_planning/{schedule_file}')
        schedule_df = get_full_time_equivalents_resources(schedule_path)
        scenario_name = self.get_scenario_name()
        schedule_df["scenario"] = scenario_name
        return schedule_df

    def get_order_traces(self) -> Optional[pd.DataFrame]:
        """Get a df that contains different orders and their process executions"""
        if self.kpi_administration is None:
            return None

        order_trace = OrderTrace(self.kpi_administration.analytics_data_base)
        order_trace_df = order_trace.get_trace_df()
        scenario_name = self.get_scenario_name()
        order_trace_df["scenario"] = scenario_name
        return order_trace_df

    def get_resource_traces(self) -> Optional[pd.DataFrame]:
        """Get a df that contains different orders and their process executions"""
        if self.kpi_administration is None:
            return None

        resource_trace = ResourceTrace(self.kpi_administration.analytics_data_base)
        resource_trace_df = resource_trace.get_trace_df(relevant_resource_types=self.relevant_resource_types)
        scenario_name = self.get_scenario_name()
        resource_trace_df["scenario"] = scenario_name
        return resource_trace_df

    def get_raw_data_dataframes_to_persist(self, start_time: Optional[datetime] = None,
                                           end_time: Optional[datetime] = None):
        """
        Get the data frames from the scenario analytics data basis (raw data/ not aggregated to kpi's) that
        e.g., can be used for the following persistence.

        Parameters
        ----------
        start_time: the start time of the scenario
        end_time: the end time of the scenario
        """

        analytics_data_base = self.get_analytics_data_base()

        process_execution_df = analytics_data_base.process_execution_df.copy()
        process_execution_order_df = analytics_data_base.process_execution_order_df.copy()
        process_execution_part_df = analytics_data_base.process_execution_part_df.copy()
        process_execution_resource_df = analytics_data_base.process_execution_resource_df.copy()
        inventory_df = analytics_data_base.inventory_df.copy()
        order_df = analytics_data_base.order_df.copy()

        # consider start and end time
        if start_time:
            process_execution_df = process_execution_df[process_execution_df["Start Time"] >= start_time]

        if end_time:
            process_execution_df = process_execution_df[process_execution_df["End Time"] <= end_time]

        if start_time is not None or end_time is not None:
            pe_ids = process_execution_df["Process Execution ID"].to_list()

            process_execution_order_df = process_execution_order_df[
                process_execution_order_df["Process Execution ID"].isin(pe_ids)]
            process_execution_part_df = process_execution_part_df[
                process_execution_part_df["Process Execution ID"].isin(pe_ids)]
            process_execution_resource_df = process_execution_resource_df[
                process_execution_resource_df["Process Execution ID"].isin(pe_ids)]
            inventory_df = inventory_df[inventory_df["Process Execution ID"].isin(pe_ids)]  # ToDo: consider initial PE
            order_df = order_df[order_df["Order ID"].isin(process_execution_order_df["Order ID"].to_list())]

        # consider scenario identification
        df_list = [process_execution_df, process_execution_order_df, process_execution_part_df,
                   process_execution_resource_df, inventory_df, order_df]

        # append the scenario id to the dataframes to ensure that the data can be read from the database
        # based on the scenario id
        for df in df_list:
            df["Scenario ID"] = pd.Series([self.get_scenario_name()] * process_execution_df.shape[0])

        return (process_execution_df, process_execution_order_df, process_execution_part_df,
                process_execution_resource_df, inventory_df, order_df)

    def update_raw_data_dataframes_from_database(self, tables_from_database):
        for table_name, df in tables_from_database.items():
            if "Scenario ID" in df.columns:
                df.drop(columns=["Scenario ID"], inplace=True)

        analytics_data_base = self.get_analytics_data_base()
        analytics_data_base.update_dataframes_from_database(tables_from_database)
