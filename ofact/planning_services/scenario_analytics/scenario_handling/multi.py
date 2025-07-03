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
    MultiScenariosHandler: responsible for multiple scenarios, respectively, all available scenarios
"""

# Import Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd
import numpy as np

from ofact.planning_services.scenario_analytics.scenario_handling.state_model_matcher import StateModelMatcher
from ofact.planning_services.scenario_analytics.repository_services.kpi_data_controller import KPIDataBaseController
from ofact.planning_services.scenario_analytics.repository_services.raw_data_controller import \
    ScenarioAnalyticsDataBaseController
# Import Part 3: Project Imports
from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase

if TYPE_CHECKING:
    from datetime import datetime

    from ofact.planning_services.scenario_analytics.business_logic.kpi.table_basic import KPIAdministration


def match_state_models(scenario_1: str, scenario_2: str, scenarios_dict):
    """
    Go through the state model objects and compare their external identifications.
    Based on the comparison, scenarios containing different identifications for the same object will be equalized.
    """

    state_model_scenario_1 = scenarios_dict[scenario_1].get_state_model()
    state_model_scenario_2 = scenarios_dict[scenario_2].get_state_model()
    digital_twin_matcher = StateModelMatcher(reference_state_model=state_model_scenario_1,
                                             other_state_model=state_model_scenario_2)

    other_id_reference_id_customer_match, matchable_customer_ids = digital_twin_matcher.match_customers()
    other_id_reference_id_order_match, matchable_order_ids = digital_twin_matcher.match_orders()
    other_id_reference_id_product_match, matchable_product_ids = digital_twin_matcher.match_products()
    other_id_reference_id_process_execution_match, matchable_process_execution_ids = \
        digital_twin_matcher.match_process_executions()
    other_id_reference_id_part_match, matchable_part_ids = digital_twin_matcher.match_parts()
    first_unassigned_identification = digital_twin_matcher.get_first_unassigned_identification()

    # ToDo: What to do with process_executions does not match?

    kpi_administration: KPIAdministration = scenarios_dict[scenario_2].get_kpi_administration()
    kpi: ScenarioAnalyticsDataBase = kpi_administration.analytics_data_base
    kpi.update_data()

    dfs = [kpi.process_execution_df, kpi.process_execution_part_df, kpi.process_execution_resource_df,
           kpi.process_execution_order_df, kpi.order_df, kpi.digital_twin_df]

    column = "Customer ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_customer_match,
                                                     matchable_customer_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    column = "Order ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_order_match,
                                                     matchable_order_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    column = "Product ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_product_match,
                                                     matchable_product_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    column = "Process Execution ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_process_execution_match,
                                                     matchable_process_execution_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    column = "Connected Process Execution ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_process_execution_match,
                                                     matchable_process_execution_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    column = "Part Involved ID"
    first_unassigned_identification = update_kpi_dfs(dfs, column, other_id_reference_id_part_match, matchable_part_ids,
                                                     first_unassigned_identification)  # matchable_process_execution_ids

    kpi_administration = scenarios_dict[scenario_2].get_kpi_administration()
    kpi_administration.analytics_data_base.update_digital_twin_df()


def update_kpi_dfs(dfs: list[pd.DataFrame], column, matcher_dict, matchable_ids, first_unassigned_identification):
    """Update the kpi dfs based on the similarities found by the state model mapper"""

    unmatchable_ids_lst = []
    for df in dfs:
        if column not in df.columns:
            continue
        unmatchable_ids = (set(df[column].to_list()).difference(set(matchable_ids)))  # difference matchable_ids
        unmatchable_ids_lst.extend(unmatchable_ids)

    unmatchable_ids_set = list(set(unmatchable_ids_lst))
    unmatchable_ids_set_length = len(unmatchable_ids_set)
    new_ids = list(range(first_unassigned_identification, unmatchable_ids_set_length))
    unmatchable_ids_matcher_dict = dict(zip(unmatchable_ids_set, new_ids))
    matcher_dict |= unmatchable_ids_matcher_dict

    for df in dfs:
        if column not in df.columns:
            continue

        df[column].replace(matcher_dict)

    first_unassigned_identification += unmatchable_ids_set_length

    return first_unassigned_identification


class MultiScenariosHandler:
    ####################################################################################################################
    # #### PERSISTENCE TABLES DEFINITION
    ####################################################################################################################

    scenario_id_df = pd.DataFrame({"Scenario ID": pd.Series([], dtype=np.dtype("str"))})

    process_execution_df_persistence = (
        pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.process_execution_df_default]))
    process_execution_order_df_persistence = (
        pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.process_execution_order_df_default]))
    process_execution_part_df_persistence = (
        pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.process_execution_part_df_default]))
    process_execution_resource_df_persistence = (
        pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.process_execution_resource_df_default]))
    inventory_persistence = pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.inventory_df_default])
    order_df_persistence = pd.concat([scenario_id_df, ScenarioAnalyticsDataBase.order_df_default])

    raw_data_table_data_types_list = {"PROCESS_EXECUTION": process_execution_df_persistence,
                                      "PROCESS_EXECUTION_ORDER": process_execution_order_df_persistence,
                                      "PROCESS_EXECUTION_PART": process_execution_part_df_persistence,
                                      "PROCESS_EXECUTION_RESOURCE": process_execution_resource_df_persistence,
                                      "INVENTORY": inventory_persistence,
                                      "ORDER_POOL": order_df_persistence}

    def __init__(self, scenarios_dict: dict[str, SingleScenarioHandler] = None):
        """
        The multi scenario handler is responsible for all scenarios exiting in the environment.
        Therefore, this class manages the access to a scenario.
        Differentiates between current state and simulation run scenarios.

        Parameters
        ----------
        scenarios_dict : maps single scenario handlers to their scenario names -
        later differentiated into current state and simulation run scenarios
        """

        if scenarios_dict is None:
            scenarios_dict = {}

        current_state_scenarios_dict = {}
        simulation_run_scenarios_dict = {}
        for scenario_name, scenario_handler in scenarios_dict.items():
            scenario_handler.set_scenario_name(scenario_name)
            if "current_state" in scenario_name:
                current_state_scenarios_dict[scenario_name] = scenario_handler
            elif "simulation_run" in scenario_name:
                simulation_run_scenarios_dict[scenario_name] = scenario_handler
            else:
                raise Exception(f"Scenario name '{scenario_name}' is not supported!")

        self.current_state_scenarios_dict = current_state_scenarios_dict
        self.simulation_run_scenarios_dict = simulation_run_scenarios_dict

        self.maximum_number_of_simulation_scenarios = 2

        self.scenario_analytics_data_base_controller = None
        self.kpi_data_base_controller = None

    def is_scenario_available(self, scenario_name):
        scenario_available = scenario_name in (self.current_state_scenarios_dict | self.simulation_run_scenarios_dict)
        return scenario_available

    def get_scenario_handler_by_name(self, scenario_name) -> SingleScenarioHandler:
        if scenario_name in self.current_state_scenarios_dict:
            scenario_handler = self.current_state_scenarios_dict[scenario_name]
        elif scenario_name in self.simulation_run_scenarios_dict:
            scenario_handler = self.simulation_run_scenarios_dict[scenario_name]
        else:
            scenario_handler = None

        return scenario_handler

    def set_scenario_handler(self, scenario_name: str, scenario_handler: SingleScenarioHandler, start_time):
        """Use a ring storage to delete not used scenarios"""

        if "current_state" in scenario_name:
            # only one simulation scenario is allowed
            self.current_state_scenarios_dict = {scenario_name: scenario_handler}
        elif "simulation_run" in scenario_name:
            self.simulation_run_scenarios_dict[scenario_name] = scenario_handler
            if len(self.simulation_run_scenarios_dict) > self.maximum_number_of_simulation_scenarios:
                self.simulation_run_scenarios_dict = (
                    dict(list((self.simulation_run_scenarios_dict.items()))[1:]))
        else:
            raise Exception(f"Scenario name '{scenario_name}' is not supported!")

        scenario_handler.set_scenario_name(scenario_name)

    def add_empty_scenario(self, scenario_name: str, project_path, digital_twin_file_path, start_time):
        new_scenario_handler = SingleScenarioHandler()
        new_scenario_handler.set_empty_scenario(project_path, digital_twin_file_path)
        self.set_scenario_handler(scenario_name, new_scenario_handler, start_time)

    def match_state_models(self, scenario_name):
        """Match different state models with each other"""
        scenarios_dict = self.get_scenarios_dict()

        for scenario_name_ in scenarios_dict:
            if scenario_name == scenario_name_:  # self-matching not required
                continue

            match_state_models(scenario_1=scenario_name_, scenario_2=scenario_name,
                               scenarios_dict=scenarios_dict)

    def set_simulation_scenario(self, ground_scenario_name: str):
        """Set up a simulation scenario from a ground scenario"""

        ground_scenario_handler = self.get_scenario_handler_by_name(ground_scenario_name)
        simulation_scenario_handler: SingleScenarioHandler = (
            ground_scenario_handler.get_duplicated_scenario_handler())

        simulation_scenario_name: str = "simulation_run" + str(simulation_scenario_handler.id)
        print(f"Simulation scenario '{simulation_scenario_name}' created which is derived from "
              f"derived from scenario '{ground_scenario_name}'.")

        self.set_scenario_handler(simulation_scenario_name, simulation_scenario_handler, None)

        return simulation_scenario_name

    def get_scenarios_dict(self):
        return self.current_state_scenarios_dict | self.simulation_run_scenarios_dict

    def get_scenario_names(self):
        scenarios_dict = self.get_scenarios_dict()
        scenario_names = list(scenarios_dict.keys())
        return scenario_names

    def get_kpi_for_scenarios(self, scenarios_to_consider, multi_scenario_handler, start_time=None, end_time=None):
        """
        Get KPI for all scenarios. For each, the order, product, process, resource and resource view
        as well as additional information.

        Parameters
        ----------
        scenarios_to_consider : list of scenario names
        start_time : start time of the scenario
        end_time : end time of the scenario
        """

        # ToDo: flexibilization needed

        order_dataframes, product_dataframes, process_dataframes, resource_dataframes = [], [], [], []
        resource_utilization_dataframes = []
        scenario_description_data_frames = []
        resource_schedules_dataframes = []
        order_traces_dataframes, resource_traces_dataframes = [], []
        scenarios_dict = self.get_scenarios_dict()
        for scenario_name, scenario_handler in scenarios_dict.items():
            if scenario_name not in scenarios_to_consider["scenarioIDs"]:
                continue

            orders_df, products_df, processes_df, resources_df, resource_utilization_df = (
                scenario_handler.get_all_kpis(start_time=start_time, end_time=end_time))

            order_traces_df = scenario_handler.get_order_traces()
            resource_traces_df = scenario_handler.get_resource_traces()

            schedule_df = scenario_handler.get_resource_schedules()

            order_dataframes.append(orders_df)
            product_dataframes.append(products_df)
            process_dataframes.append(processes_df)
            resource_dataframes.append(resources_df)
            resource_utilization_dataframes.append(resource_utilization_df)
            if schedule_df is not None:
                resource_schedules_dataframes.append(schedule_df)
            if order_traces_df is not None:
                order_traces_dataframes.append(order_traces_df)
            if resource_traces_df is not None:
                resource_traces_dataframes.append(resource_traces_df)

            scenario_description_str = scenario_handler.get_state_model().description
            if scenario_description_str is None:
                scenario_description_str = ""
            scenario_description_df = pd.DataFrame({"scenario": [scenario_name],
                                                    "description": [scenario_description_str]})
            scenario_description_data_frames.append(scenario_description_df)

        delta_orders_columns = ["numberOfPiecesAbsolute", "numberOfPiecesAbsolute", "deliveryReliability",
                                "deliveryDelay", "totalLeadTime", "totalWaitingTime",
                                'currentStock', 'performance', 'quality', 'performance', ]

        delta_products_columns = ['targetQuantity', 'quantityProduced', "differencePercentage", "productShares",
                                  'deliveryReliability', 'leadTime', 'totalLeadTime', 'waitingTime',
                                  'totalWaitingTime', 'currentStock', 'quality', 'performance', ]

        delta_processes_columns = ['absoluteFrequency', 'processShare', 'deliveryReliability', 'leadTime',
                                   'waitingTime', 'minLeadTime', 'minWaitingTime', 'maxLeadTime', 'maxWaitingTime',
                                   'varianceLeadTime', 'varianceWaitingTime', 'quality', 'performance', ]

        delta_resources_columns = ['processFrequency', 'resourceShare', 'deliveryReliability', 'leadTime',
                                   'waitingTime', 'stock', 'quality', 'performance', 'totalResourceUtilisation',
                                   'totalResourceAvailability', 'kpiOre']

        delta_resource_utilization_columns = ['Capacity Utilization']

        if len(scenarios_to_consider["scenarioIDs"]) > 1:
            order_dataframes[0].set_index("id", inplace=True)
            order_dataframes[1].set_index("id", inplace=True)
            for column in delta_orders_columns:
                mask_1 = order_dataframes[0][(order_dataframes[0]["orderStatus"] == "FINISHED") &
                                             (order_dataframes[0][column] != "-")].index
                mask_2 = order_dataframes[1][(order_dataframes[1]["orderStatus"] == "FINISHED") &
                                             (order_dataframes[1][column] != "-")].index # ToDo: @Niklas: changes 0 to 1
                same_indices = list(set(mask_1).intersection(set(mask_2)))
                delta_col = (order_dataframes[0].loc[same_indices, column] -
                             order_dataframes[1].loc[same_indices, column]).abs()

                order_dataframes[0].loc[same_indices, f"{column}_delta"] = delta_col
                order_dataframes[1].loc[same_indices, f"{column}_delta"] = delta_col

            try:
                diff_df = order_dataframes[0][order_dataframes[0]["orderStatus"] !=
                                          order_dataframes[1]["orderStatus"]].index
                order_dataframes[0]["difference_orderStatus"] = 0
                order_dataframes[0].loc[diff_df, "difference_orderStatus"] = 1
                order_dataframes[1]["difference_orderStatus"] = 0
                order_dataframes[1].loc[diff_df, "difference_orderStatus"] = 1
            except ValueError as e:
                order_dataframes[0]["difference_orderStatus"] = 0
                order_dataframes[1]["difference_orderStatus"] = 0

            order_dataframes[0].reset_index(inplace=True)
            order_dataframes[1].reset_index(inplace=True)

            product_dataframes[0].set_index("id", inplace=True)
            product_dataframes[1].set_index("id", inplace=True)
            for column in delta_products_columns:
                delta_col = (product_dataframes[0][column] - product_dataframes[1][column]).abs()
                product_dataframes[0][f"{column}_delta"] = delta_col
                product_dataframes[1][f"{column}_delta"] = delta_col
            product_dataframes[0].reset_index(inplace=True)
            product_dataframes[1].reset_index(inplace=True)

            process_dataframes[0].set_index("id", inplace=True)
            process_dataframes[1].set_index("id", inplace=True)
            for column in delta_processes_columns:
                delta_col = (process_dataframes[0][column] - process_dataframes[1][column]).abs()
                process_dataframes[0][f"{column}_delta"] = delta_col
                process_dataframes[1][f"{column}_delta"] = delta_col
            process_dataframes[0].reset_index(inplace=True)
            process_dataframes[1].reset_index(inplace=True)

            resource_dataframes[0].set_index("id", inplace=True)
            resource_dataframes[1].set_index("id", inplace=True)
            for column in delta_resources_columns:
                delta_col = (resource_dataframes[0][column] - resource_dataframes[1][column]).abs()
                resource_dataframes[0][f"{column}_delta"] = delta_col
                resource_dataframes[1][f"{column}_delta"] = delta_col
            resource_dataframes[0].reset_index(inplace=True)
            resource_dataframes[1].reset_index(inplace=True)

            first_index_column = (resource_utilization_dataframes[0]["id"].astype("str") +
                                  resource_utilization_dataframes[0]["Time"].astype("str"))
            second_index_column = (resource_utilization_dataframes[1]["id"].astype("str") +
                                   resource_utilization_dataframes[1]["Time"].astype("str"))

            resource_utilization_dataframes[0].set_index(first_index_column, inplace=True)
            resource_utilization_dataframes[1].set_index(second_index_column, inplace=True)
            for column in delta_resource_utilization_columns:
                delta_col = (resource_utilization_dataframes[0][column] -
                             resource_utilization_dataframes[1][column]).abs()
                resource_utilization_dataframes[0][f"{column}_delta"] = delta_col
                resource_utilization_dataframes[1][f"{column}_delta"] = delta_col
            resource_utilization_dataframes[0].set_index("id", inplace=True)
            resource_utilization_dataframes[1].set_index("id", inplace=True)
            if "level_0" in resource_utilization_dataframes[1].columns:
                resource_utilization_dataframes[1].drop(columns=["level_0"], inplace=True)

        if "level_0" in resource_utilization_dataframes[0].columns:
            resource_utilization_dataframes[0].drop(columns=["level_0"], inplace=True)

        orders = pd.concat(order_dataframes, axis=0, ignore_index=True)
        products = pd.concat(product_dataframes, axis=0, ignore_index=True)
        processes = pd.concat(process_dataframes, axis=0, ignore_index=True)
        resources = pd.concat(resource_dataframes, axis=0, ignore_index=True)
        resource_utilization = pd.concat(resource_utilization_dataframes, axis=0, ignore_index=True)
        if resource_schedules_dataframes:
            resource_schedules = pd.concat(resource_schedules_dataframes, axis=0, ignore_index=True)
        else:
            resource_schedules = pd.DataFrame()

        if order_traces_dataframes:
            order_traces = pd.concat(order_traces_dataframes, axis=0, ignore_index=True)
        else:
            order_traces = pd.DataFrame()
        if resource_traces_dataframes:
            resource_traces = pd.concat(resource_traces_dataframes, axis=0, ignore_index=True)
        else:
            resource_traces = pd.DataFrame()

        scenario_descriptions = pd.concat(scenario_description_data_frames, axis=0, ignore_index=True)

        return (orders, products, processes, resources, resource_utilization, resource_schedules,
                order_traces, resource_traces, scenario_descriptions)

    ####################################################################################################################
    # #### PERSIST TO DATABASE ####
    ####################################################################################################################

    def persist_raw_data(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                         scenarios: Optional[list[str]] = None):
        """
        Persist the data frames from the scenario analytics data basis (raw data/ not aggregated to kpi's)

        Parameters
        ----------
        start_time: the start time of the scenario to persist
        end_time: the end time of the scenario to persist
        scenarios: scenario names that are later used to filter the database according to the scenario name
        """

        if self.scenario_analytics_data_base_controller is None:
            self.scenario_analytics_data_base_controller = (
                ScenarioAnalyticsDataBaseController(table_data_types_list=type(self).raw_data_table_data_types_list))

        scenario_dict = self.get_scenarios_dict()
        for scenario in scenarios:
            if scenario not in scenario_dict:
                print(f"Scenario {scenario} not found in the multi scenario handler (persist raw data to database)")
                continue

            scenario_handler = scenario_dict[scenario]

            (process_execution_df, process_execution_order_df, process_execution_part_df,
             process_execution_resource_df, inventory_df, order_df) = (
                scenario_handler.get_raw_data_dataframes_to_persist(start_time, end_time))

            # setup table dict
            dict_df = {"PROCESS_EXECUTION": process_execution_df,
                       "PROCESS_EXECUTION_ORDER": process_execution_order_df,
                       "PROCESS_EXECUTION_PART": process_execution_part_df,
                       "PROCESS_EXECUTION_RESOURCE": process_execution_resource_df,
                       "INVENTORY": inventory_df,
                       "ORDER_POOL": order_df}

            # store the dataframes in the database
            self.scenario_analytics_data_base_controller.store_dataframes(dict_df)

    def persist_kpi_data(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                         scenarios: dict[str, list[str]] = None):
        (orders, products, processes, resources, resource_utilization, resource_schedules, order_traces,
         scenario_descriptions) = (
            self.get_kpi_for_scenarios(scenarios, self, start_time, end_time))

        # setup table dict
        dict_df = {"KPI_ORDERS": orders,
                   "KPI_PRODUCTS": products,
                   "KPI_PROCESSES": processes,
                   "KPI_RESOURCES": resources,
                   "KPI_RESOURCE_UTILIZATION": resource_utilization,
                   "KPI_SCENARIO_DESCRIPTIONS": scenario_descriptions}

        if self.kpi_data_base_controller is None:

            # dynamic setup of the database table columns
            kpi_data_table_data_types_list = {}
            for table_name, df in dict_df.items():
                table_df = pd.DataFrame({col: pd.Series(dtype=dt)
                                         for col, dt in df.dtypes.items()})
                kpi_data_table_data_types_list[table_name] = pd.concat([type(self).scenario_id_df, table_df])

            self.kpi_data_base_controller = (
                KPIDataBaseController(table_data_types_list=kpi_data_table_data_types_list))

        # store the dataframes in the database
        self.kpi_data_base_controller.store_dataframes(dict_df)

    ####################################################################################################################
    # #### READ FROM DATABASE ####
    ####################################################################################################################

    def read_raw_data_from_database(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                                    scenarios: Optional[list[str]] = None, update_data_base_analytics=False):
        """
        Read the raw data (data frames) from the database

        Parameters
        ----------
        start_time: the start time of the scenario to request from the database
        end_time: the end time of the scenario to request from the database
        scenarios: scenario names that are used to filter the database according to the scenario name
        update_data_base_analytics: bool value that specifies if the data read from the database is used to
        update the tables.
        """

        if self.scenario_analytics_data_base_controller is None:
            self.scenario_analytics_data_base_controller = (
                ScenarioAnalyticsDataBaseController(table_data_types_list=type(self).raw_data_table_data_types_list))

        tables_to_read = ["PROCESS_EXECUTION", "PROCESS_EXECUTION_ORDER", "PROCESS_EXECUTION_PART",
                          "PROCESS_EXECUTION_RESOURCE", "INVENTORY", "ORDER_POOL"]

        scenario_dict = self.get_scenarios_dict()
        scenario_tables_dict = {}
        for scenario in scenarios:
            if scenario not in scenario_dict:
                print(f"Scenario {scenario} not found in the multi scenario handler (read raw data from database)")
                continue

            scenario_handler = scenario_dict[scenario]

            scenario_name = scenario_handler.get_scenario_name()
            tables_from_database = self.scenario_analytics_data_base_controller.read_tables(
                tables_to_read, start_time, end_time, scenario_name)

            scenario_tables_dict[scenario] = tables_from_database

            if update_data_base_analytics:
                scenario_handler.update_raw_data_dataframes_from_database(tables_from_database)

        return scenario_tables_dict
