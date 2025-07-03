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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ofact.twin.agent_control.behaviours.planning.tree.helpers import get_overlaps_periods
from ofact.planning_services.scenario_analytics.business_logic.kpi.table import LeadTimeBasedKPI, \
    KPICalc, LeadTime  # ToDo: dependency maybe not wished

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase


def lead_time(df):
    new_df = \
        pd.DataFrame({"End Time [s]": df["End Time [s]"].iloc[np.r_[-1]][df.index[-1]],
                      "Start Time [s]": df["Start Time [s]"].iloc[np.r_[0]][df.index[0]],
                      "Lead Time": df["End"].iloc[np.r_[-1]][df.index[-1]] - df["Start Time [s]"].iloc[np.r_[0]][
                          df.index[0]]},
                     index=[df.index[0]])

    return new_df


def calc_utilization(df, start_end, bin_size):
    start, end = start_end

    df.loc[:, "Start Time [s]"] = df["Start Time [s]"].values.astype('datetime64[s]')
    df.loc[:, "End Time [s]"] = df["End Time [s]"].values.astype('datetime64[s]')
    mask = (df["End Time"] >= start) & (df["Start Time"] <= end)
    time_periods_to_consider = df.loc[mask, ["Start Time", "End Time"]].to_numpy()

    overlapped_time_periods_cutted = get_overlaps_periods(time_periods_to_consider, np.array([[start, end]]))
    if overlapped_time_periods_cutted.size:
        utilized_time = (overlapped_time_periods_cutted[:, 1] - overlapped_time_periods_cutted[:, 0]).sum()
    else:
        utilized_time = np.timedelta64(0, "s")

    percentage = utilized_time / np.timedelta64(bin_size, "m")
    return percentage


def calc_lead_time(df, start_end):
    start, end = start_end

    mask = (df["End Time"] > start) & (df["Start Time"] < end)

    mean = df[mask]["Lead Time with Waiting Time [s]"].mean()
    min_ = df[mask]["Lead Time with Waiting Time [s]"].min()
    max_ = df[mask]["Lead Time with Waiting Time [s]"].max()

    return mean, min_, max_


class Chart:

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        self.analytics_data_base = analytics_data_base

        self.view_id_match = {"ORDER": "Order ID",
                              "PRODUCT": "Entity Type ID",
                              "PROCESS": "Process ID",
                              "RESOURCE": "Resource Used ID"}

    def _get_date_range(self, earliest_start_time, latest_end_time, bin_size):

        date_range_ = pd.date_range(earliest_start_time, latest_end_time, inclusive="both", freq=f"{bin_size}min")
        date_range_ = date_range_.to_series(index=np.arange(date_range_.size))
        if not date_range_.empty:
            if date_range_.iloc[-1] < latest_end_time:
                date_range_ = pd.concat([date_range_, pd.Series(latest_end_time, index=[date_range_.size])])
        # time_periods = np.concatenate([np.array(date_range_[:-1])[:, None], np.array(date_range_[1:])[:, None]],
        #                               axis=1)

        return date_range_

    def _get_labels(self, date_range_):
        labels = [i for i, time_period in enumerate(date_range_)][0:-1]

        return labels


class LeadTimeBasedChart(Chart, LeadTimeBasedKPI):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def _get_chart_time_frontiers(self):
        earliest_start_time = type(self).start_time_observation_period
        if earliest_start_time is None:
            earliest_start_time = self.analytics_data_base.process_execution_df["Start Time"].min()

        latest_end_time = type(self).end_time_observation_period
        if latest_end_time is None:
            latest_end_time = self.analytics_data_base.process_execution_df["End Time"].max()

        return earliest_start_time, latest_end_time


class AvailabilityChart(LeadTimeBasedChart):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase, accepted_resource_types: list[str] = None):
        super().__init__(analytics_data_base=analytics_data_base)

        self.accepted_resource_types = accepted_resource_types

    def get_utilisation_chart(self, start_time, end_time, order_ids, product_ids, process_ids, resource_ids,
                              event_type, view, scenario, bin_size=5, resource_type="ALL", all=False):

        # maybe also adjustable in the frontend
        title = "Kapazitätsauslastung der Ressourcen"
        x_label = f"Zeit (Bin size: {bin_size})"
        y_label = "Ressourcen Kapazitätsauslastung in %"
        # title = "Utilization"
        # x_label = f"Time bins (Bin size: {bin_size})"
        # y_label = "Utilization in %"

        if view != "RESOURCE":
            raise NotImplementedError("The utilization over time is currently only for the resource view available")

        accepted_resource_ids_set = \
            set(self.analytics_data_base.digital_twin_df.loc[
                    self.analytics_data_base.digital_twin_df["Resource Type"].isin(self.accepted_resource_types),
                    "Resource Used ID"])

        resource_ids = list(set(resource_ids).intersection(accepted_resource_ids_set))

        process_execution_ids_resources_used = \
            {k: v["Process Execution ID"].to_list()
             for k, v in self.analytics_data_base.digital_twin_df.groupby(by="Resource Used ID")}
        relevant_df = \
            self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                       part_ids=product_ids, process_ids=process_ids, resource_ids=resource_ids,
                                       event_type=event_type, view=view, scenario=scenario)

        if relevant_df.empty:
            resource_id_utilization_d = {}
            mean = None

            return resource_id_utilization_d, mean, bin_size, title, x_label, y_label

        relevant_df = relevant_df.copy()
        # 1. calc consideration_period
        earliest_start_time, latest_end_time = self._get_chart_time_frontiers()
        date_range_ = self._get_date_range(earliest_start_time, latest_end_time, bin_size)
        labels = self._get_labels(date_range_)
        # for each resource one request

        resource_id_utilization = pd.DataFrame(np.zeros((date_range_[:-1].size, len(resource_ids))),
                                               columns=resource_ids,
                                               index=date_range_[:-1].to_numpy())
        for resource_id, process_executions in process_execution_ids_resources_used.items():
            if resource_id not in resource_id_utilization:
                continue

            resource_df = relevant_df.loc[relevant_df["Process Execution ID"].isin(process_executions)]

            bins_date_range = pd.Series(date_range_, dtype="<M8[ns]")
            start_time_bins = pd.cut(resource_df["Start Time"], bins=bins_date_range, labels=labels)
            end_time_bins = pd.cut(resource_df["End Time"], bins=bins_date_range, labels=labels)

            bins_filled = [bin
                           for bin in list(set(start_time_bins.to_list() + end_time_bins.to_list()))
                           if bin == bin]
            for bin in bins_filled:  # ToDo: performance
                time_period = date_range_[bin:bin + 2].to_list()

                utilization_result = calc_utilization(resource_df, time_period, bin_size)
                resource_id_utilization[resource_id][date_range_[bin]] = utilization_result

        rows_filled_mask = (resource_id_utilization.T != 0).any()
        first_filled_index = rows_filled_mask.argmax()
        filled_indexes = np.where(rows_filled_mask)[0]

        if not filled_indexes.size:
            resource_id_utilization_d = {}
            mean = None
            return resource_id_utilization_d, mean, bin_size, title, x_label, y_label

        last_filled_index = filled_indexes[-1]
        cleaned_resource_id_utilization = resource_id_utilization.iloc[first_filled_index: last_filled_index]

        cleaned_resource_id_utilization *= 100  # convert to percent

        # prepare for sending to the frontend
        cleaned_resource_id_utilization = (
            cleaned_resource_id_utilization[~cleaned_resource_id_utilization.index.duplicated(keep='first')])
        new_index = pd.Series(cleaned_resource_id_utilization.index).apply(lambda x: float(x.timestamp()) - 7200)
        if not new_index.empty:
            cleaned_resource_id_utilization.index = new_index * 1000
        else:
            cleaned_resource_id_utilization.index = new_index

        mean_s = cleaned_resource_id_utilization.mean(axis=1)

        mean = list(mean_s.to_dict().items())

        resource_id_utilization_s = cleaned_resource_id_utilization.to_dict('series').items()

        resource_id_utilization_d = {resource_id: list(chart_series.to_dict().items())
                                     for resource_id, chart_series in resource_id_utilization_s}

        return resource_id_utilization_d, mean, bin_size, title, x_label, y_label


class LeadTimeChart(LeadTimeBasedChart, LeadTime):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def get_lead_time_chart(self, start_time, end_time, order_ids, product_ids, process_ids, resource_ids,
                            event_type, view, bin_size, scenario):
        lead_time_dfs = self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                                   part_ids=part_ids, process_ids=product_ids,
                                                   resource_ids=resource_ids,
                                                   event_type=event_type, view=view, scenario=scenario)

        earliest_start_time, latest_end_time = self._get_chart_time_frontiers()
        date_range_ = self._get_date_range(earliest_start_time, latest_end_time, bin_size)
        labels = self._get_labels(date_range_)

        start_time_bins = pd.cut(lead_time_dfs["Start Time [s]"], bins=pd.Series(date_range_, dtype="<M8[ns]"),
                                 labels=labels)
        end_time_bins = pd.cut(lead_time_dfs["End Time [s]"], bins=pd.Series(date_range_, dtype="<M8[ns]"),
                               labels=labels)
        # bins = pd.concat([start_time_bins, end_time_bins], axis=1)
        bins_filled = list(set(start_time_bins.to_list() + end_time_bins.to_list()))
        order_id_lead_time = {}
        for bin in bins_filled:
            if bin != bin:
                continue

            time_period = date_range_[bin:bin + 2].to_list()

            mean, min_, max_ = calc_lead_time(lead_time_dfs, time_period)

            order_id_lead_time[date_range_[bin]] = mean  # (mean, min_, max_)

        id_lead_time_mean_d = order_id_lead_time

        title = "Lead times"
        x_label = f"{view.lower()}"
        y_label = "Lead Time (in seconds)"

        return id_lead_time_mean_d, title, x_label, y_label


if "__main__" == __name__:
    import dill as pickle
    from ofact.settings import ROOT_PATH

    kpi_pkl_path = Path(
        fr"{ROOT_PATH}\DigitalTwin\projects\bicycle_world\scenarios\base\digital_twinbase_6KPI.pkl")
    with open(kpi_pkl_path, 'rb') as inp:
        analytics_data_base = pickle.load(inp)

    bin_size = 10

    kpi_calc = KPICalc(analytics_data_base=analytics_data_base)
    lead_time_based_kpi = LeadTimeBasedKPI(analytics_data_base)
    lead_time = LeadTime(analytics_data_base)
    chart_availability = AvailabilityChart(analytics_data_base)
    chart_lead_time = LeadTimeChart(analytics_data_base)
    start_time = analytics_data_base.process_execution_df["Start Time"].min()
    end_time = analytics_data_base.process_execution_df["End Time"].max()
    order_ids = list(set(analytics_data_base.order_df["Order ID"].to_list()))
    part_ids = list(set(analytics_data_base.process_execution_part_df["Entity Type ID"].to_list()))
    process_ids = list(set(analytics_data_base.process_execution_df["Process ID"].to_list()))
    resource_ids = list(set(analytics_data_base.process_execution_resource_df["Resource Used ID"].to_list()))
    event_type = "PLAN"
    view = "RESOURCE"
    #
    print(chart_availability.get_utilisation_chart(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                                   event_type, view, scenario="c"))

    view = "ORDER"
    print(chart_lead_time.get_lead_time_chart(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                              event_type, view, bin_size, scenario="c"))
