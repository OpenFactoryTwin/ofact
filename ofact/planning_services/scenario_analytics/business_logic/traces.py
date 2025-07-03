from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase


class ObjectTrace:

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase) -> None:
        self.analytics_data_base: ScenarioAnalyticsDataBase = analytics_data_base

    def get_trace_df(self) -> pd.DataFrame:
        pass


class OrderTrace(ObjectTrace):

    def get_trace_df(self):
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

        # append order timestamps to df
        relevant_columns = ["Order ID", "Order Identifier",
                            "Release Date", "Delivery Date Planned", "Delivery Date Actual"]

        order_trace_df_orders = self.analytics_data_base.order_df[relevant_columns].melt(
            id_vars=["Order ID", "Order Identifier"],
            value_vars=["Release Date", "Delivery Date Planned", "Delivery Date Actual"],
            var_name="Process Name",
            value_name="Start Time"
        )
        order_trace_df = pd.concat([order_trace_df, order_trace_df_orders], axis=0)

        order_trace_df.sort_values(by=["Order Identifier", "Event Type", "Start Time"],
                                   inplace=True, na_position="last")

        return order_trace_df


class ResourceTrace(ObjectTrace):

    def get_trace_df(self, relevant_resource_types: Optional[list[str]] = None):
        """
        Create a df that shows the process execution traces of the resources.

        Returns
        -------
        order_trace_df : pd.DataFrame
        """
        relevant_columns = ["Process ID", "Event Type", "Process Name", "Start Time", "End Time",
                            "Resource Used ID", "Resource Used Name", "Resource Type",
                            "Order ID"]
        resource_ids_df = (
            self.analytics_data_base.process_execution_resource_df[[
                "Resource Used ID", "Resource Used Name", "Process Execution ID", "Resource Type"]])

        # filter only the relevant resource types
        resource_ids_df = resource_ids_df.loc[resource_ids_df["Resource Type"].isin(relevant_resource_types)]

        resource_trace_df = pd.merge(resource_ids_df,
                                     self.analytics_data_base.process_execution_df,
                                     on="Process Execution ID",
                                     how="inner")

        resource_trace_df = resource_trace_df[relevant_columns]

        resource_trace_df = resource_trace_df.dropna(subset=["Process ID"])
        resource_trace_df.sort_values(by=["Resource Used ID", "Event Type", "Start Time"],
                                   inplace=True, na_position="last")

        return resource_trace_df
