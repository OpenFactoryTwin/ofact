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

An input for the physical world applications (shop floors etc. and logistic networks) and the simulation are schedules.

@last update: 16.10.2023
"""

from __future__ import annotations

import functools
import operator
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal

import pandas as pd
from py2vega.functions.date_time import hours

from ofact.twin.state_model.model import StateModel

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

try:
    import polars as pl
except ModuleNotFoundError:
    pl = None

from ofact.planning_services.model_generation.persistence import deserialize_state_model
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase

if TYPE_CHECKING:
    pass

language = "de" # "en"


LABELS = {
    "de": {
        "planned_work_time": "Geplante Arbeitszeit",
        "actual_work_time": "Ist (Sim) Arbeitszeit",
        "hours": "Stunden",
        "actual_vs_plan": "Ist vs. Plan Arbeitszeit pro Ressource (mit Abweichungen)"
    },
    "en": {
        "planned_work_time": "Planned working time",
        "actual_work_time": "Actual (Sim) working time",
        "hours": "hours",
        "actual_vs_plan": "Actual vs. Plan working time per resource (with deviations)"
    }
}

def get_label(key: str, lang: str) -> str:
    # fallback auf Englisch, falls Sprache oder Key fehlt
    return LABELS.get(lang, LABELS["en"]).get(key, key)


class Schedule:
    process_execution_df_default = pl.DataFrame(
        {"Process Execution ID": pl.Series(values=[], dtype=pl.Int64),
         "Event Type": pl.Series(values=[], dtype=pl.Utf8),
         "Start Time": pl.Series(values=[], dtype=pl.Datetime),
         "End Time": pl.Series(values=[], dtype=pl.Datetime),
         "Process ID": pl.Series(values=[], dtype=pl.Int64),
         "Process Name": pl.Series(values=[], dtype=pl.Utf8),

         "Order ID": pl.Series(values=[], dtype=pl.Int64),
         "Reference Resource ID": pl.Series(values=[], dtype=pl.Int64),
         "Reference Resource Name": pl.Series(values=[], dtype=pl.Utf8),
         "Second Resource ID": pl.Series(values=[], dtype=pl.Int64),
         "Second Resource Name": pl.Series(values=[], dtype=pl.Utf8), }
    )
    relevant_columns = ["Process Execution ID", "Event Type", "Start Time", "End Time",
                        "Process ID", "Process Name",
                        "Order ID",
                        "Main Resource ID", "Resource Used ID", "Resource Used Name", "Resource Type"]

    # , "Superior Resource Used ID"

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase, state_model: Optional[StateModel] = None):
        self._analytics_data_basis: ScenarioAnalyticsDataBase = analytics_data_base

        self._state_model: Optional[StateModel] = state_model

        self._schedule_df = pl.DataFrame()
        self._schedule_parametrized_plan_df = pl.DataFrame()
        self._schedule_parametrized_actual_df = pl.DataFrame()

        self._planned_schedule_parametrized_df = pl.DataFrame()

    def update_schedule_df(self):
        pass

    def create_planned_schedule(self, reference_resource_types, relevant_resources: dict):

        if self._state_model is None:
            raise ValueError("State model is not initialized in the initialization of the schedule object.")

        all_resources = []
        for reference_resource_type in reference_resource_types:
            match reference_resource_type:
                case "StationaryResource":
                    resources = self._state_model.get_stationary_resources()
                case "ActiveMovingResource":
                    resources = self._state_model.get_active_moving_resources()
                case "PassiveMovingResource":
                    resources = self._state_model.get_passive_moving_resources()
                case "WorkStation":
                    resources = self._state_model.get_work_stations()
                case _:
                    raise NotImplementedError(f"Resource type {reference_resource_type} is not implemented.")

            all_resources += resources

        all_resources = list(set(all_resources))

        if "contains" in relevant_resources:
            relevant_resource_objects = [resource
                                         for resource in all_resources
                                         for name_element in relevant_resources["contains"]
                                         if name_element in resource.name]
        else:
            relevant_resource_objects = all_resources

        relevant_resource_schedules = pd.concat([self.get_planned_resource_schedule(resource)
                                                 for resource in relevant_resource_objects],
                                                axis=0, ignore_index=True)
        relevant_resource_schedules["Event Type"] = "SCHEDULE_PLAN"

        self._planned_schedule_parametrized_df = relevant_resource_schedules
        # ToDo: generally for the schedule pl.from_pandas()

    def get_planned_resource_schedule(self, resource):
        if resource.process_execution_plan is None:
            return pd.DataFrame()
        scheduled_time_slots_full = resource.process_execution_plan.get_scheduled_time_slots()
        scheduled_time_slots_filtered = (
            scheduled_time_slots_full[scheduled_time_slots_full["Blocker Name"] == "Planner"])
        unscheduled_time_slots = pd.DataFrame(scheduled_time_slots_filtered[["Start", "End"]])
        unscheduled_time_slots["Gap Start"] = unscheduled_time_slots["End"].shift()

        scheduled_time_slots = (
            unscheduled_time_slots[["Gap Start", "Start"]]
            .dropna()
            .rename(columns={"Gap Start": "Start Time", "Start": "End Time"})
            .reset_index(drop=True)
        )

        scheduled_time_slots["Reference Resource Name"] = resource.name + " PLAN"
        scheduled_time_slots["Order Identifier"] = "PLAN"
        return scheduled_time_slots

    def create_schedule(self, reference_resource_types: list, relevant_resources: dict,
                        second_resource_type: Optional[Literal],
                        relevant_process_names: Optional[list[Literal]] = None,
                        start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                        event_type: Literal["PLAN", "ACTUAL"] = "ACTUAL") -> None:
        """
        Create a schedule dataframe based on the dataframe from the data_basis and the input parameters
        :param relevant_process_names: matching process names are included in the schedule dataframe
        only the value added ones, but not all of them ...
        :param start_time: if given, the start time restrict the consideration period of the schedule
        :param end_time: if given, the start time restrict the consideration period of the schedule
        :param event_type: PLAN or ACTUAL
        """

        digital_twin_df = self._analytics_data_basis.digital_twin_df

        event_type_mask = digital_twin_df["Event Type"] == event_type
        digital_twin_event_type_considered_df = digital_twin_df.loc[event_type_mask]
        digital_twin_essential_df = digital_twin_event_type_considered_df[type(self).relevant_columns]

        if relevant_process_names is not None:
            digital_twin_essential_df = (
                digital_twin_essential_df.loc[digital_twin_essential_df["Process Name"].isin(relevant_process_names)])

        if start_time is not None:
            digital_twin_essential_df = (
                digital_twin_essential_df.loc[digital_twin_essential_df["End Time"] >= start_time])

        if end_time is not None:
            digital_twin_essential_df = (
                digital_twin_essential_df.loc[digital_twin_essential_df["Start Time"] <= end_time])

        schedule_parametrized_df = (
            digital_twin_essential_df[["Process Execution ID", "Event Type", "Start Time", "End Time",
                                       "Process ID", "Process Name", "Order ID"]])

        schedule_parametrized_df = self._set_resource(digital_twin_df, schedule_parametrized_df,
                                                      reference_resource_types, relevant_resources, name="Reference")
        if second_resource_type is not None:
            schedule_parametrized_df = self._set_resource(digital_twin_df, schedule_parametrized_df,
                                                          second_resource_type, relevant_resources, name="Second")

        if event_type == "ACTUAL":
            self._schedule_parametrized_actual_df = schedule_parametrized_df
        elif event_type == "PLAN":
            self._schedule_parametrized_plan_df = schedule_parametrized_df
        else:
            raise NotImplementedError(f"Event type {event_type} is not implemented.")

    def _set_resource(self, digital_twin_df, schedule_parametrized_df, reference_resource_types, relevant_resources,
                      name="Reference"):
        mask = functools.reduce(
            operator.or_,
            (digital_twin_df["Resource Type"] == reference_resource_type
             for reference_resource_type in reference_resource_types)
        )
        resource_id_mapper_df = digital_twin_df.loc[
            mask,
            ["Process Execution ID", "Resource Used ID", "Resource Type"]].dropna()

        resource_id_mapper = defaultdict(list)

        # Iterate through the DataFrame and create the mapping
        for _, row in resource_id_mapper_df.iterrows():
            process_execution_id = row["Process Execution ID"]
            resource_used_id = row["Resource Used ID"]
            resource_id_mapper[process_execution_id].append(resource_used_id)

        schedule_parametrized_df = (
            schedule_parametrized_df.assign(**{f"{name} Resource ID":
                                                   schedule_parametrized_df["Process Execution ID"]}))
        # Map the "Process Execution ID" to its corresponding resource IDs (as lists)
        schedule_parametrized_df[f"{name} Resource ID"] = (
            schedule_parametrized_df["Process Execution ID"].map(resource_id_mapper))

        # Explode the column so that each resource ID gets its own row
        schedule_parametrized_df = schedule_parametrized_df.explode(f"{name} Resource ID").reset_index(drop=True)

        resource_name_mapper_df = digital_twin_df[["Resource Used ID", "Resource Used Name", "Resource Type"]].dropna()
        resource_name_mapper = dict(zip(resource_name_mapper_df["Resource Used ID"].to_list(),
                                        resource_name_mapper_df["Resource Used Name"].to_list()))
        resource_type_name_mapper = dict(zip(resource_name_mapper_df["Resource Used ID"].to_list(),
                                             resource_name_mapper_df["Resource Type"].to_list()))

        schedule_parametrized_df = (
            schedule_parametrized_df.assign(**{f"{name} Resource Name":
                                                   schedule_parametrized_df[f"{name} Resource ID"]}))
        schedule_parametrized_df[f"{name} Resource Name"].replace(resource_name_mapper, inplace=True)
        schedule_parametrized_df[f"{name} Resource Type"] = schedule_parametrized_df[f"{name} Resource ID"]
        schedule_parametrized_df[f"{name} Resource Type"].replace(resource_type_name_mapper, inplace=True)

        order_id_identifier_df = self._analytics_data_basis.digital_twin_df[["Order ID", "Order Identifier"]].dropna()
        order_id_identifier_mapper = dict(zip(order_id_identifier_df["Order ID"].to_list(),
                                        order_id_identifier_df["Order Identifier"].to_list()))
        schedule_parametrized_df["Order Identifier"] = schedule_parametrized_df[f"Order ID"]
        schedule_parametrized_df["Order Identifier"].replace(order_id_identifier_mapper, inplace=True)

        if "contains" in relevant_resources:
            contains_list = relevant_resources["contains"]
            pattern = "|".join(re.escape(s) for s in contains_list)
            schedule_parametrized_df = schedule_parametrized_df[
                schedule_parametrized_df["Reference Resource Name"].str.contains(pattern, na=False)]

        return schedule_parametrized_df

    def plot_schedule(self, event_type: Literal["PLAN", "ACTUAL"] = "ACTUAL",
                      planned_resource_schedule: bool = False):
        if event_type == "ACTUAL":
            schedule_parametrized_df = self._schedule_parametrized_actual_df
        elif event_type == "PLAN":
            schedule_parametrized_df = self._schedule_parametrized_plan_df
        else:
            raise NotImplementedError(f"Event type {event_type} is not implemented.")
        if planned_resource_schedule:
            schedule_parametrized_df = (
                pd.concat([schedule_parametrized_df, self._planned_schedule_parametrized_df]))

        highlight = "-"  # die Order, die Du hervorheben willst
        my_map = {highlight: "dimgray"}
        cats = sorted(schedule_parametrized_df["Reference Resource Name"].dropna().unique())
        fig = px.timeline(schedule_parametrized_df,
                          x_start="Start Time", x_end="End Time",
                          y="Reference Resource Name",
                          color="Order Identifier",
                          category_orders={"Reference Resource Name": cats},
                          color_discrete_map=my_map
                          )
        fig.show()

    def get_full_time_equivalent(
            self,
            event_type: Literal["PLAN", "ACTUAL"] = "ACTUAL",
            planned_resource_schedule: bool = False,
            aggregated = []
    ):

        # 1) Basis-DF wählen
        if event_type == "ACTUAL":
            schedule_parametrized_df = self._schedule_parametrized_actual_df.copy()
        elif event_type == "PLAN":
            schedule_parametrized_df = self._schedule_parametrized_plan_df.copy()
        else:
            raise NotImplementedError(f"Event type {event_type} is not implemented.")

        # optional Plan-DF mergen
        if planned_resource_schedule:
            schedule_parametrized_df = pd.concat([
                schedule_parametrized_df,
                self._planned_schedule_parametrized_df
            ])

        if aggregated:
            resources = schedule_parametrized_df["Reference Resource Name"].dropna().unique().tolist()
            resources_mapping_plan = {}
            resources_mapping_actual = {}
            for aggregation_resource in aggregated:
                for resource in resources:
                    if aggregation_resource in resource:
                        if "PLAN" in resource:
                            resources_mapping_plan[resource] = aggregation_resource + " PLAN"
                        else:
                            resources_mapping_actual[resource] = aggregation_resource

            schedule_parametrized_df.replace(resources_mapping_plan, inplace=True)
            schedule_parametrized_df.replace(resources_mapping_actual, inplace=True)

        # 2) Duration berechnen und aufsummieren
        schedule_parametrized_df["Duration"] = (
                schedule_parametrized_df["End Time"] - schedule_parametrized_df["Start Time"]
        )
        resources_working_hours = (
            schedule_parametrized_df
            .groupby("Reference Resource Name")["Duration"]
            .sum()
            .to_dict()
        )

        # 3) Dicts actual vs. plan trennen
        actual_resources_working_hours = {
            name: dur
            for name, dur in resources_working_hours.items()
            if "PLAN" not in name
        }
        actual_resources_working_hours["Total"] = sum(
            actual_resources_working_hours.values(), pd.Timedelta(0)
        )

        planned_resources_working_hours = {
            name: dur
            for name, dur in resources_working_hours.items()
            if "PLAN" in name
        }
        planned_resources_working_hours["Total PLAN"] = sum(
            planned_resources_working_hours.values(), pd.Timedelta(0)
        )

        # 4) DataFrame für Plot aufbauen
        planned_work_time_str = get_label("planned_work_time", language)
        actual_work_time_str = get_label("actual_work_time", language)
        hours_str = get_label("hours", language)
        actual_vs_plan_title_str = get_label("actual_vs_plan", language)

        resources = list(actual_resources_working_hours.keys())
        df = pd.DataFrame({
            "Resource": resources,
            actual_work_time_str: [
                actual_resources_working_hours.get(res, pd.Timedelta(0)).total_seconds() / 3600
                for res in resources
            ],
            planned_work_time_str: [
                planned_resources_working_hours.get(res + " PLAN", pd.Timedelta(0)).total_seconds() / 3600
                for res in resources
            ]
        })

        # 5) Abweichungen berechnen (Liste, keine neue DF-Spalte)
        deviations = [
            a - p
            for a, p in zip(df[actual_work_time_str], df[planned_work_time_str])
        ]

        # 6) Plotly Express-Chart
        fig = px.bar(
            df,
            x="Resource",
            y=[actual_work_time_str, planned_work_time_str],
            barmode="group",
            labels={"value": hours_str, "variable": ""},
            color_discrete_map={actual_work_time_str: "steelblue", planned_work_time_str: "lightgray"}
        )

        # Abweichungstext nur auf den Actual-Balken setzen
        fig.update_traces(
            selector=dict(name=actual_work_time_str),
            text=[f"{d:+.2f} h" for d in deviations],
            textposition="outside"
        )

        fig.update_layout(
            title=actual_vs_plan_title_str,
            xaxis_tickangle=-45,
            yaxis_title=hours_str
        )
        fig.show()

        return actual_resources_working_hours, planned_resources_working_hours


if __name__ == "__main__":
    digital_twin_result_pkl = "/home/riotana/PycharmProjects/ofact-intern/projects/bicycle_world/scenarios/current/results/six_orders.pkl"
    digital_twin_result_pkl = "/home/riotana/PycharmProjects/ofact-intern/projects/dbs/data/result.pkl"
    print("Path:", digital_twin_result_pkl)
    digital_twin_model = deserialize_state_model(source_file_path=digital_twin_result_pkl, dynamics=True)

    analytics_data_basis = ScenarioAnalyticsDataBase(state_model=digital_twin_model)
    schedule = Schedule(analytics_data_basis)

    relevant_process_names = [p.name for p in digital_twin_model.get_value_added_processes()]
    reference_resource_types = ["StationaryResource"]
    second_resource_type = None  # "ActiveMovingResource"
    event_type = "ACTUAL"
    schedule.create_schedule(relevant_process_names=relevant_process_names,
                             reference_resource_types=reference_resource_types, second_resource_type=second_resource_type,
                             event_type=event_type)
    schedule.plot_schedule(event_type=event_type)

# ToDo: PLAN evaluation ...
# ToDo: How realistic is a plan - comparison between plan and actual ...
#  - Matching them to a single digital twin???
#  - compare the lead time differences between plan and actual
#  - compare the order lead times
#  - compare the delivery adherence's, respectively the delivery reliability ...
