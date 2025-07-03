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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase


class OrderTrace:

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        self.analytics_data_base: ScenarioAnalyticsDataBase = analytics_data_base

    def get_order_trace_df(self):
        """
        Create a df that shows the process execution traces of the orders.

        Returns
        -------
        order_trace_df : pd.DataFrame
        """
        order_ids = self.analytics_data_base.order_df["Order ID"].unique()

        relevant_columns = ["Order ID", # "Order Identifier"
                            "Process ID", "Event Type", "Process Name", "Start Time", "End Time",
                            "Main Resource ID"]  # , "Resource Name"
        order_trace_df = self.analytics_data_base.digital_twin_df[relevant_columns].loc[
            self.analytics_data_base.digital_twin_df["Order ID"].isin(order_ids)]

        order_name_mapper_s = (
            self.analytics_data_base.order_df.set_index("Order ID")["Order Identifier"].to_dict())
        order_trace_df["Order Identifier"] = order_trace_df["Order ID"].replace(order_name_mapper_s)

        resource_name_mapper_s = (
            self.analytics_data_base.process_execution_resource_df.set_index("Resource Used ID")[
                "Resource Used Name"].to_dict())
        order_trace_df["Resource Name"] = order_trace_df["Main Resource ID"].replace(resource_name_mapper_s)

        order_trace_df = order_trace_df.dropna(subset=["Process ID"])
        order_trace_df.sort_values(by=["Order Identifier", "Event Type", "Start Time"],
                                   inplace=True, na_position="last")

        return order_trace_df
