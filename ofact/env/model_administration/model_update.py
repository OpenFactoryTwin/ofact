from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.state_model.processes import ValueAddedProcess

if TYPE_CHECKING:
    from ofact.twin.state_model.processes import Process, ProcessExecution
    from ofact.twin.state_model.model import StateModel


class ProcessModelUpdate:

    def __init__(self, state_model: StateModel, generator):
        self._state_model = state_model

        self.generator = generator

    def update(self):
        process_executions_plan = self._get_process_executions_plan()
        self._update_priority_chart(process_executions_plan)
        self._update_process_models(process_executions_plan)
        self._update_sales_area(process_executions_plan)

    def _get_process_executions_plan(self) -> list[ProcessExecution]:
        process_executions = self._state_model.get_process_executions_list(event_type=ProcessExecutionTypes.PLAN)

        process_executions_plan = [process_execution
                                   for process_execution in process_executions
                                   if process_execution.check_plan()]

        return process_executions_plan

    def _update_process_models(self, process_executions_plan):

        # according processes
        process_process_executions = {}
        for process_execution in process_executions_plan:
            process = process_execution.process
            process_process_executions.setdefault(process,
                                                  []).append(process_execution)

        for process, process_executions in process_process_executions.items():
            self._update_process_time_model(process, process_executions)
            self._update_transition_model(process, process_executions)
            self._update_resource_model(process, process_executions)
            self._update_transformation_model(process, process_executions)
            self._update_quality_model(process, process_executions)

    def _update_process_time_model(self, process: Process, process_executions: list[ProcessExecution]):
        process_times = [process_execution.get_process_lead_time()
                         for process_execution in process_executions
                         if process_execution.get_process_lead_time() > 0]
        print("Process Time before:", process.name, sum(process_times), len(process_times),
              (sum(process_times) + 1e-9) / (len(process_times) + 1e-9))
        try:
            process_times = self._clean_outliers(process_times)

            print("Process Time:", process.name, sum(process_times), len(process_times),
                  (sum(process_times) + 1e-9) / (len(process_times) + 1e-9))

            process_time = ((sum(process_times) + 1e-9) /
                            (len(process_times) + 1e-9))

        except:
            process_time = 0
        process_time_model = process.lead_time_controller.process_time_model
        process_time_model.value = process_time

    def _clean_outliers(self, float_list):
        # Compute Q1 and Q3
        q1 = np.percentile(float_list, 25)
        q3 = np.percentile(float_list, 75)
        iqr = q3 - q1

        # Compute bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outliers
        cleaned_float_list = [float_elem
                              for float_elem in float_list
                              if lower_bound <= float_elem <= upper_bound]

        return cleaned_float_list

    def _update_transition_model(self, process: Process, process_executions: list[ProcessExecution]):
        origins = list(set(process_execution.origin
                           for process_execution in process_executions))
        destinations = list(set(process_execution.destination
                                for process_execution in process_executions))

        transition_model = process.transition_controller.transition_model
        transition_model.possible_origins = origins
        transition_model.possible_destinations = destinations

    def _update_resource_model(self, process: Process, process_executions: list[ProcessExecution]):
        resource_groups_list = list(set((tuple(process_execution.get_resources()), process_execution.main_resource)
                                        for process_execution in process_executions))

        resource_model = process.resource_controller.resource_model
        resource_groups = self.generator.get_resource_group(resource_groups_list,
                                                            name=process.name)
        resource_model.resource_groups = resource_groups

    def _update_transformation_model(self, process: Process, process_executions: list[ProcessExecution]):
        pass

    def _update_quality_model(self, process: Process, process_executions: list[ProcessExecution]):
        pass

    def _update_priority_chart(self, process_executions_plan):
        try:
            priority_chart_df = pd.read_csv("process_relations.csv")
        except FileNotFoundError:
            print("Process Relations File not found")
            return

        # update predecessors and successors
        all_processes = self._state_model.get_all_processes()

        relevant_processes = priority_chart_df["process"].to_list()
        for process in all_processes:
            if isinstance(process, ValueAddedProcess):
                if process.name in relevant_processes:
                    predecessors = ast.literal_eval(priority_chart_df.loc[(priority_chart_df["process"] == process.name,
                                                                           "predecessors")].to_list()[0])
                    successors = ast.literal_eval(priority_chart_df.loc[(priority_chart_df["process"] == process.name,
                                                                         "successors")].to_list()[0])

                    process.predecessors = [(self._cache.get_objects_by_class("Process")[predecessor],)
                                            for predecessor in predecessors]
                    process.successors = [self._cache.get_objects_by_class("Process")[successor]
                                          for successor in successors]

    def _get_transition_processes(self):
        pass

    def _update_sales_area(self, process_executions_plan):
        # each value added process gets a feature and a feature cluster

        all_processes = self._state_model.get_all_processes()
        orders = self._state_model.get_orders()

        order_process_executions = {}
        for process_execution in process_executions_plan:
            if isinstance(process_execution.process, ValueAddedProcess):
                order = process_execution.order
                order_process_executions.setdefault(order,
                                                    []).append(process_execution.process)

        # for order_object in orders:
        #
        #     process_executions = order_process_executions[order_object]
        #
        #     products = self._cache.get_object(class_name="Part", id_=order_id)
        #     if products:
        #         order_object.products = products
        #         order_object.product_classes = [product.entity_type
        #                                         for product in products]
        #     else:
        #         # first part
        #         part_column_names = _get_column_names(order_group.columns, EventLogStandardClasses.PART.string)
        #         if part_column_names:
        #             first_part_column_name = part_column_names[0]
        #             # assuming the first part is the product
        #             try:
        #                 product_part_name = order_group[first_part_column_name].iloc[0]
        #             except:
        #                 raise Exception(first_part_column_name, "\n", order_group, "\n", order_id)
        #
        #             products = self._cache.get_objects(class_name="Part", id_=product_part_name)
        #             if not products:
        #                 print(f"Product: {product_part_name}, {order_id}")
        #             order_object.products = products
        #             order_object.product_classes = [product.entity_type
        #                                             for product in products]

        for process in all_processes:
            if isinstance(process, ValueAddedProcess):
                feature = self._state_model.get_object_by_external_identification(
                    external_id=process.name + "_f", name_space="static_model", class_name="Feature")
                if not feature:  # feature already exists
                    feature = self.generator.get_feature(name=process.name)
                    self._state_model.add_feature(feature)
                if process.feature is None:
                    process.feature = feature

        feature_sets = []
        for order, processes in order_process_executions.items():
            features = []
            for process in processes:  # assume the same process is not twice in the data
                features.append(process.feature)

            order.features_requested = features

            if list(set(features)) not in feature_sets:
                feature_sets.append(list(set(features)))

        def find_parent(parent, i):
            # Findet das Vertreter-Element (mit Pfadkompression)
            if parent[i] != i:
                parent[i] = find_parent(parent, parent[i])
            return parent[i]

        def union(parent, rank, x, y):
            # Vereinigt die Mengen, zu denen x und y gehören
            xroot = find_parent(parent, x)
            yroot = find_parent(parent, y)
            if xroot == yroot:
                return
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1

        def cluster_elements(feature_sets):
            # Alle eindeutigen Elemente ermitteln
            elements = set()
            for fs in feature_sets:
                elements.update(fs)

            # Union-Find-Initialisierung
            parent = {e: e for e in elements}
            rank = {e: 0 for e in elements}

            # Für jede Gruppe: Alle Elemente werden miteinander vereinigt
            for fs in feature_sets:
                fs = list(fs)
                for i in range(len(fs)):
                    for j in range(i + 1, len(fs)):
                        union(parent, rank, fs[i], fs[j])

            # Clustermitglied ermitteln: Elemente mit demselben Vertreter gehören zum selben Cluster
            clusters = {}
            for e in elements:
                rep = find_parent(parent, e)
                clusters.setdefault(rep, set()).add(e)
            return list(clusters.values())

        clusters = cluster_elements(feature_sets)
        for cluster in clusters:
            feature_names = list(set(([feature.name.split("_f name")[0]
                                       for feature in cluster])))
            feature_cluster_name = "_".join(feature_names)
            feature_cluster = self._state_model.get_object_by_external_identification(
                external_id=feature_cluster_name + "_fc", name_space="static_model", class_name="FeatureCluster")

            if not feature_cluster:
                feature_cluster = self.generator.get_feature_cluster(name=feature_cluster_name)
                self._state_model.add_feature_cluster(feature_cluster)

                # feature_cluster.product_class = orders[0].product_classes[0]  # ToDo: same entity_type or super_entity_type

            for feature in cluster:
                if feature.feature_cluster is None:
                    feature.feature_cluster = feature_cluster
