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

Instantiation of the digital_twin based on an Excel file and creating of the digital_twin model.

@contact persons: Adrian Freiter
"""
import sys
# Imports Part 1: Standard Imports
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Imports Part 2: PIP Imports
import dill as pickle
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.repository_services.interface.importer.basic_file_loader import (
    ObjectInstantiation, convert_tuple_keys_to_nested_dict, convert_str_to_python_objects, combine_all_objects)
from ofact.twin.repository_services.interface.importer.order_types import OrderType
from ofact.twin.repository_services.interface.importer.twin_mapping import (
    MappingEntityType, MappingPlant, MappingStationaryResource, MappingStorage, MappingWarehouse,
    MappingWorkStation, MappingConveyorBelt, MappingNonStationaryResource, MappingAllResources, MappingTime,
    MappingProcess, MappingProcessControllers, MappingProcessTimeModels, MappingQualityModels, MappingResourceModels,
    MappingTransitionModels, MappingTransformationModels, MappingParts, MappingSales, MappingCustomer, MappingOrder)
from ofact.twin.state_model.basic_elements import DigitalTwinObject, DynamicAttributeChangeTracking
from ofact.twin.state_model.helpers.helpers import load_from_pickle
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.probabilities import SingleValueDistribution
from ofact.twin.state_model.processes import (Process, ValueAddedProcess, EntityTransformationNode)
from ofact.helpers import Singleton

try:
    from ofact.twin.model_learning.process_models_advanced import DTModelLearningExtension
except:
    DTModelLearningExtension = None

# Module-Specific Constants
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
sys.setrecursionlimit(10000)  # setting a higher recursion limit for pickling


# #### creating digital_twin  ##########################################################################################


def get_object_dicts(dict_, type_):
    return {object_name: object_
            for (object_type, object_name), object_ in dict_.items()
            if object_type == type_}


def update_feature_weights(features, generation_type):
    """
    update feature weights for each feature cluster by normalizing them if needed
    """
    if generation_type == OrderType.SHOPPING_BASKET:
        return

    feature_clusters_features = {}
    for feature in features:
        feature_clusters_features.setdefault(feature.feature_cluster,
                                             []).append(feature)

        for feature_cluster, features1 in feature_clusters_features.items():
            features_selection_probabilities = {feature: feature.get_expected_selection_probability()
                                                for feature in features1}

            probabilities_sum = sum(list(features_selection_probabilities.values()))
            if probabilities_sum != 1:
                # normalize to one
                for feature_, probability in features_selection_probabilities.items():
                    if not isinstance(feature_.selection_probability_distribution, SingleValueDistribution):
                        raise Exception("selection_probability_distribution is not a SingleValueDistribution, "
                                        "but the generation type is PRODUCT_CONFIGURATOR")

                    new_single_value = probability / probabilities_sum
                    feature_.selection_probability_distribution.value = new_single_value


def get_order_generation_from_excel_df(path):
    orders_df = pd.read_excel(path,
                              index_col=[0, 1],
                              header=[0, 1],  # header=[0, 2],
                              sheet_name='Orders')

    features_requested_raw = orders_df.features_requested

    features_requested_lists = []
    for _, row in features_requested_raw.iterrows():
        try:
            features_selected = [feature
                                 for feature, amount in row.dropna().items()
                                 for i in range(int(amount))]
        except ValueError:
            raise ValueError(row)

        features_requested_lists.append(features_selected)

    feature_columns_length = len(features_requested_raw.columns)
    order_df = pd.read_excel(path, index_col=[0, 1], sheet_name='Orders', skiprows=1)  # skiprows=2)
    order_df.dropna(how='all', inplace=True)
    order_df = order_df.iloc[:, :-feature_columns_length]
    order_df["features_requested"] = features_requested_lists

    return order_df


def _set_static_model_attribute(dt_object, unique_attribute_label):
    if hasattr(dt_object, "external_identifications"):
        if dt_object.external_identifications is None:
            dt_object.external_identifications = {"static_model": ["_" + unique_attribute_label]}
        elif "static_model" not in dt_object.external_identifications:
            dt_object.external_identifications["static_model"] = ["_" + unique_attribute_label]


def _update_dynamic_attributes(dt_object):
    """Since the dynamic attributes are only set in the first call, we need to update them here"""

    dt_object_attributes = dt_object.dynamic_attributes.attributes
    for dynamic_attribute_name, dynamic_attribute_tracker in dt_object_attributes.items():
        dynamic_attribute_tracker: DynamicAttributeChangeTracking
        attribute_value = getattr(dt_object, dynamic_attribute_name)
        if dynamic_attribute_tracker.recent_changes.changes.shape[0]:
            current_time = dynamic_attribute_tracker.recent_changes.changes["Timestamp"][0]
            process_execution = dynamic_attribute_tracker.recent_changes.changes["ProcessExecution"][0]
        else:
            current_time = np.datetime64(datetime.min, "ns")
            process_execution = None

        dynamic_attribute_tracker.recent_changes = (
            dynamic_attribute_tracker.attribute_change_tracker_class(current_time=current_time,
                                                                     attribute_value=attribute_value,
                                                                     process_execution=process_execution))


class StaticModelImporter(metaclass=Singleton):
    __metaclass__ = Singleton

    mappings = {"MappingEntityType": MappingEntityType,
                "MappingPlant": MappingPlant,
                "MappingStationaryResource": MappingStationaryResource,
                "MappingStorage": MappingStorage,
                "MappingWarehouse": MappingWarehouse,
                "MappingWorkStation": MappingWorkStation,
                "MappingConveyorBelt": MappingConveyorBelt,
                "MappingNonStationaryResource": MappingNonStationaryResource,
                "MappingAllResources": MappingAllResources,
                "MappingTime": MappingTime,
                "MappingProcess": MappingProcess,
                "MappingProcessControllers": MappingProcessControllers,
                "MappingProcessTimeModels": MappingProcessTimeModels,
                "MappingQualityModels": MappingQualityModels,
                "MappingResourceModels": MappingResourceModels,
                "MappingTransitionModels": MappingTransitionModels,
                "MappingTransformationModels": MappingTransformationModels,
                "MappingParts": MappingParts,
                "MappingSales": MappingSales,
                "MappingCustomer": MappingCustomer,
                "MappingOrder": MappingOrder}

    def __init__(self, path, ORDER_TYPE=OrderType.PRODUCT_CONFIGURATOR):
        """
        Used to load/ create the digital twin from the Excel file.

        Parameters
        ----------
        path: a reference path to the Excel file
        ORDER_TYPE: determine how the order should be imported
        """
        super().__init__()
        print("Start importing Excel file:", datetime.now())
        self.path = path
        self.xlsx_content = pd.ExcelFile(self.path)

        self.ORDER_TYPE = ORDER_TYPE

        # instantiation of the Excel files
        self.object_instantiation = ObjectInstantiation()
        self.create_time_from_excel()  # self.time_objects
        self.create_entity_type_from_excel()  # self.entity_type_objects
        self.create_plant_from_excel()  # self.plant_objects
        stationary_resource_df, stationary_resource_objects, corrected_stationary_resource_dict = (
            self.create_stationary_resource_from_excel())
        storage_df, storage_objects, corrected_storage_dict = self.create_storage_from_excel()
        work_station_df, work_station_objects, corrected_work_station_dict = (
            self.create_work_station_resource_from_excel())
        conveyor_belt_df, conveyer_belt_objects, corrected_conveyor_belt_dict = (
            self.create_conveyor_belt_resource_from_excel())
        warehouse_df, warehouse_objects, corrected_warehouse_dict = self.create_warehouse_resource_from_excel()
        non_stationary_resource_df, non_stationary_resource_objects, corrected_non_stationary_resource_dict = (
            self.create_non_stationary_resource_from_excel())
        active_moving_resource_df, active_moving_resource_objects, corrected_active_moving_resource_dict = (
            self.create_active_moving_resource_from_excel())
        passive_moving_resource_df, passive_moving_resource_objects, corrected_passive_moving_resource_dict = (
            self.create_passive_moving_resource_from_excel())
        self.create_all_resources(
            stationary_resource_df, stationary_resource_objects, corrected_stationary_resource_dict,
            storage_df, storage_objects, corrected_storage_dict,
            work_station_df, work_station_objects, corrected_work_station_dict,
            conveyor_belt_df, conveyer_belt_objects, corrected_conveyor_belt_dict,
            warehouse_df, warehouse_objects, corrected_warehouse_dict,
            non_stationary_resource_df, non_stationary_resource_objects, corrected_non_stationary_resource_dict,
            active_moving_resource_df, active_moving_resource_objects, corrected_active_moving_resource_dict,
            passive_moving_resource_df, passive_moving_resource_objects,
            corrected_passive_moving_resource_dict)  # self.resource_objects
        self.create_parts_from_excel()  # self.part_objects
        self.create_customer_from_excel()  # self.customer_objects
        self.create_sales_from_excel()  # self.sales_objects
        self.create_orders_from_excel()  # self.order_objects
        self.create_process_models_from_excel()  # self.process_model_objects
        self.create_process_controllers_from_excel()  # self.process_controller_objects
        self.create_processes_from_excel()  # self.process_objects

        self.handle_duplications()

        # digital_twin_objects
        # sales_objects
        self.customers: dict = get_object_dicts(self.customer_objects, "Customer")
        self.orders: dict = get_object_dicts(self.order_objects, "Order")
        self.feature_clusters: dict = get_object_dicts(self.sales_objects, "FeatureCluster")
        self.features: dict = get_object_dicts(self.sales_objects, "Feature")

        # part objects
        self.parts: dict = get_object_dicts(self.part_objects, "Part")

        # resource_objects
        self.entity_types: dict = (get_object_dicts(self.entity_type_objects, "EntityType") |
                                   get_object_dicts(self.entity_type_objects, "PartType"))
        self.plant: dict = get_object_dicts(self.plant_objects, "Plant")
        self.storages: dict = get_object_dicts(self.resource_objects, "Storage")
        self.work_stations: dict = get_object_dicts(self.resource_objects, "WorkStation")
        self.warehouses: dict = get_object_dicts(self.resource_objects, "Warehouse")
        self.conveyor_belts: dict = get_object_dicts(self.resource_objects, "ConveyorBelt")
        self.stationary_resources: dict = self.storages | self.work_stations | self.warehouses | self.conveyor_belts
        self.active_moving_resources: dict = get_object_dicts(self.resource_objects, "ActiveMovingResource")
        self.passive_moving_resources: dict = get_object_dicts(self.resource_objects, "PassiveMovingResource")
        self.non_stationary_resources: dict = get_object_dicts(self.resource_objects, "NonStationaryResource")

        # process_objects
        self.resources_groups: dict = get_object_dicts(self.process_objects, "ResourceGroup")
        self._check_quality_of_resources_groups_import()
        self.entity_transformation_nodes: dict = get_object_dicts(self.process_objects, "EntityTransformationNode")
        self._check_quality_of_entity_transformation_node_import()
        self.processes: dict = get_object_dicts(self.process_objects, "Process")
        self.value_added_processes: dict = get_object_dicts(self.process_objects, "ValueAddedProcess")
        self.all_processes: dict = self.processes | self.value_added_processes

        self.process_controllers = get_object_dicts(self.process_controller_objects, "ResourceController")
        self.process_controllers = (self.process_controllers |
                                    get_object_dicts(self.process_controller_objects, "ProcessTimeController"))
        self.process_controllers = (self.process_controllers |
                                    get_object_dicts(self.process_controller_objects, "TransitionController"))
        self.process_controllers = (self.process_controllers |
                                    get_object_dicts(self.process_controller_objects, "TransformationController"))
        self.process_controllers = (self.process_controllers |
                                    get_object_dicts(self.process_controller_objects, "QualityController"))

        self.transition_models = get_object_dicts(self.process_model_objects, "TransitionModel")
        self.quality_models = get_object_dicts(self.process_model_objects, "SimpleBernoulliDistributedQualityModel")
        self.transformation_models = get_object_dicts(self.process_model_objects, "TransformationModel")
        self.resource_models = get_object_dicts(self.process_model_objects, "ResourceModel")
        self.process_time_models = (
                get_object_dicts(self.process_model_objects, "SimpleSingleValueDistributedProcessTimeModel") |
                get_object_dicts(self.process_model_objects, "SimpleNormalDistributedProcessTimeModel"))

        # digital_twin
        self.create_digital_twin()  # self.digital_twin
        print("Digital Twin created")

        # deposit the digital twin model in the process_models for learning tasks ...
        self._deposit_digital_twin_in_process_models()

        self.all_digital_twin_objects: dict = \
            combine_all_objects(self.feature_clusters, self.features,
                                self.orders, self.customers,
                                self.parts,
                                self.entity_types, self.plant, self.storages, self.work_stations,
                                self.warehouses, self.stationary_resources, self.active_moving_resources,
                                self.passive_moving_resources, self.non_stationary_resources,
                                self.processes, self.value_added_processes,
                                self.transition_models, self.quality_models, self.transformation_models,
                                self.resource_models, self.process_time_models,
                                self.resources_groups, self.entity_transformation_nodes,
                                self.digital_twin)

        self.all_objects = (self.entity_type_objects | self.plant_objects | self.resource_objects | self.part_objects |
                            self.process_objects | self.time_objects | self.customer_objects | self.sales_objects |
                            self.order_objects)
        self.update_dt_objects()

        print(f"[{datetime.now()}] all_objects available")

    def _check_quality_of_resources_groups_import(self):
        for resource_group in list(self.resources_groups.values()):
            completely_filled, not_completely_filled_attributes = resource_group.completely_filled()
            if not completely_filled:
                print(f"{resource_group.__class__.__name__} not completely filled: {not_completely_filled_attributes}")

    def _check_quality_of_entity_transformation_node_import(self):
        for entity_transformation_node in list(self.entity_transformation_nodes.values()):
            completely_filled, not_completely_filled_attributes = entity_transformation_node.completely_filled()
            if not completely_filled:
                print(f"{entity_transformation_node.__class__.__name__} not completely filled: "
                      f"{not_completely_filled_attributes}")
                for not_completely_filled_attribute in not_completely_filled_attributes:
                    print(f"The attribute {not_completely_filled_attribute} has the value "
                          f"'{getattr(entity_transformation_node, not_completely_filled_attribute)}'.")

    def update_dt_objects(self):

        for unique_attribute_label, dt_object in self.all_digital_twin_objects.items():
            _set_static_model_attribute(dt_object, unique_attribute_label)

            if hasattr(dt_object, "dynamic_attributes"):
                _update_dynamic_attributes(dt_object)

    # ==== create_objects ==============================================================================================

    def create_time_from_excel(self):
        """
        self.time_objects: e.g. {(object_type, object_name), object_}
        """
        self.time_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Time", skiprows=1)
        self.time_df.dropna(how='all', inplace=True)

        self.time_objects, _ = self.object_instantiation.load_dict(object_df=self.time_df,
                                                                   mapping_class=MappingTime)

    def create_entity_type_from_excel(self):
        self.entity_type_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="EntityType")
        self.entity_type_df.dropna(how='all', inplace=True)

        mapping_class = type(self).mappings["MappingEntityType"]
        self.entity_type_objects, _ = self.object_instantiation.load_dict(object_df=self.entity_type_df,
                                                                          mapping_class=mapping_class,
                                                                          input_objects=[])

    def create_plant_from_excel(self):
        self.plant_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Plant")
        self.plant_df.dropna(how='all', inplace=True)

        mapping_class = type(self).mappings["MappingPlant"]
        self.plant_objects, _ = self.object_instantiation.load_dict(object_df=self.plant_df,
                                                                    mapping_class=mapping_class,
                                                                    input_objects=[self.time_objects])

    def _create_resource_objects_from_excel(self, mapping_class, sheet_name, convert=True):
        """
        self.resource_objects: e.g. {(object_type, object_name), object_}
        """
        resource_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name=sheet_name)

        if convert:
            # .map(lambda x: convert_str_to_python_objects(x))
            resource_df = resource_df.map(lambda x: convert_str_to_python_objects(x))

        resource_df.dropna(how='all', inplace=True)

        if "amount" in resource_df.columns:
            correction_needed = True

        else:
            correction_needed = False

        resource_objects, corrected_factory_dict = \
            self.object_instantiation.load_dict(object_df=resource_df,
                                                mapping_class=mapping_class,
                                                input_objects=[self.time_objects, self.entity_type_objects,
                                                               self.plant_objects],
                                                correction_needed=correction_needed,
                                                repetitions=1)

        if resource_objects is None:
            resource_objects = {}
        if corrected_factory_dict is None:
            corrected_factory_dict = {}
        if not correction_needed:
            return resource_df, resource_objects, corrected_factory_dict

        # update the resource_objects (stored_entities)
        resource_objects = \
            self.object_instantiation.instantiate_dependent_objects(independent_objects=resource_objects,
                                                                    objects_df=resource_df,
                                                                    mapping_class=mapping_class,
                                                                    find_in=resource_objects,
                                                                    check_independent_objects=False)
        if resource_objects is None:
            resource_objects = {}
        if corrected_factory_dict is None:
            corrected_factory_dict = {}
        return resource_df, resource_objects, corrected_factory_dict

    def create_stationary_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingStationaryResource"]
        stationary_resource_df, stationary_resource_objects, corrected_stationary_resource_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="StationaryResource",
                                                     convert=convert))
        return stationary_resource_df, stationary_resource_objects, corrected_stationary_resource_dict

    def create_storage_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingStorage"]
        storage_df, storage_objects, corrected_storage_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="Storage",
                                                     convert=convert))
        return storage_df, storage_objects, corrected_storage_dict

    def create_work_station_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingWorkStation"]
        work_station_df, work_station_objects, corrected_work_station_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="WorkStation",
                                                     convert=convert))
        return work_station_df, work_station_objects, corrected_work_station_dict

    def create_warehouse_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingWarehouse"]
        warehouse_df, warehouse_objects, corrected_warehouse_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="Warehouse",
                                                     convert=convert))
        return warehouse_df, warehouse_objects, corrected_warehouse_dict

    def create_conveyor_belt_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingConveyorBelt"]
        conveyor_belt_df, conveyor_belt_objects, corrected_conveyor_belt_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="ConveyorBelt",
                                                     convert=convert))
        return conveyor_belt_df, conveyor_belt_objects, corrected_conveyor_belt_dict

    def create_non_stationary_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingNonStationaryResource"]
        non_stationary_resource_df, non_stationary_resource_objects, corrected_non_stationary_resource_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="NonStationaryResource",
                                                     convert=convert))
        return non_stationary_resource_df, non_stationary_resource_objects, corrected_non_stationary_resource_dict

    def create_active_moving_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingNonStationaryResource"]
        active_moving_resource_df, active_moving_resource_objects, corrected_active_moving_resource_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="ActiveMovingResource",
                                                     convert=convert))
        return active_moving_resource_df, active_moving_resource_objects, corrected_active_moving_resource_dict

    def create_passive_moving_resource_from_excel(self, convert=True):
        mapping_class = type(self).mappings["MappingNonStationaryResource"]
        passive_moving_resource_df, passive_moving_resource_objects, corrected_passive_moving_resource_dict = (
            self._create_resource_objects_from_excel(mapping_class=mapping_class, sheet_name="PassiveMovingResource",
                                                     convert=convert))
        return passive_moving_resource_df, passive_moving_resource_objects, corrected_passive_moving_resource_dict

    def create_all_resources(self, stationary_resource_df, stationary_resource_objects,
                             corrected_stationary_resource_dict, storage_df, storage_objects, corrected_storage_dict,
                             work_station_df, work_station_objects, corrected_work_station_dict,
                             conveyor_belt_df, conveyer_belt_objects, corrected_conveyor_belt_dict,
                             warehouse_df, warehouse_objects, corrected_warehouse_dict,
                             non_stationary_resource_df, non_stationary_resource_objects,
                             corrected_non_stationary_resource_dict,
                             active_moving_resource_df, active_moving_resource_objects,
                             corrected_active_moving_resource_dict,
                             passive_moving_resource_df, passive_moving_resource_objects,
                             corrected_passive_moving_resource_dict):

        all_resources_dfs_unfiltered = [stationary_resource_df, storage_df, work_station_df,
                                        conveyor_belt_df, warehouse_df, non_stationary_resource_df,
                                        active_moving_resource_df, passive_moving_resource_df]
        all_resources_dfs_filtered = [df
                                      for df in all_resources_dfs_unfiltered
                                      if not df.empty]
        self.resource_df: pd.DataFrame = pd.concat(all_resources_dfs_filtered, sort=False)

        all_resource_objects = (stationary_resource_objects | storage_objects |
                                work_station_objects | conveyer_belt_objects | warehouse_objects |
                                non_stationary_resource_objects | active_moving_resource_objects |
                                passive_moving_resource_objects)

        self.corrected_factory_dict = (corrected_stationary_resource_dict |
                                       corrected_storage_dict | corrected_work_station_dict |
                                       corrected_conveyor_belt_dict | corrected_warehouse_dict |
                                       corrected_non_stationary_resource_dict | corrected_active_moving_resource_dict
                                       | corrected_passive_moving_resource_dict)

        self.resource_objects, _ = \
            self.object_instantiation.load_dict(object_df=self.resource_df,
                                                mapping_class=MappingAllResources,
                                                input_objects=[all_resource_objects],
                                                repetitions=1)

    def create_parts_from_excel(self):
        self.parts_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Part")
        self.parts_df.dropna(how='all', inplace=True)

        if "amount" in self.parts_df.columns:
            correction_needed = True
            mapping_class = type(self).mappings["MappingParts"]
            self.part_objects, self.correction_parts_dict = \
                self.object_instantiation.load_dict(object_df=self.parts_df,
                                                    mapping_class=mapping_class,
                                                    input_objects=[self.entity_type_objects, self.resource_objects],
                                                    correction_needed=correction_needed)
            mapping_class = type(self).mappings["MappingAllResources"]
            # update the resource_objects (stored_entities)
            self.resource_objects = \
                self.object_instantiation.instantiate_dependent_objects(
                    independent_objects=self.resource_objects,
                    objects_df=self.resource_df,
                    mapping_class=mapping_class,
                    find_in=self.resource_objects | self.entity_type_objects | self.part_objects,
                    check_independent_objects=False)
        else:
            raise NotImplementedError("Not implemented")

    def create_customer_from_excel(self):

        self.customer_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Customer")
        self.customer_df.dropna(how='all', inplace=True)

        self.customer_df["name"] = self.customer_df["pre_name"] + " " + self.customer_df["last_name"]
        del self.customer_df["pre_name"]
        del self.customer_df["last_name"]
        mapping_class = type(self).mappings["MappingCustomer"]
        self.customer_objects, _ = self.object_instantiation.load_dict(object_df=self.customer_df,
                                                                       mapping_class=mapping_class)

    def create_orders_from_excel(self):
        feature_objects = get_object_dicts(self.sales_objects, "Feature")
        features = list(feature_objects.values())
        update_feature_weights(features, generation_type=self.ORDER_TYPE)

        self.order_df = get_order_generation_from_excel_df(path=self.xlsx_content)

        mapping_class = type(self).mappings["MappingOrder"]
        self.order_objects, _ = self.object_instantiation.load_dict(object_df=self.order_df,
                                                                        mapping_class=mapping_class,
                                                                        input_objects=[self.customer_objects,
                                                                                       self.sales_objects])

    def create_sales_from_excel(self):
        """
        self.sales_objects: e.g. {(object_type, object_name), object_}
        """
        self.sales_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name='Sales', skiprows=1)
        self.sales_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingSales"]
        self.sales_objects, _ = self.object_instantiation.load_dict(object_df=self.sales_df,
                                                                    mapping_class=mapping_class,
                                                                    input_objects=[self.entity_type_objects])

    def create_process_models_from_excel(self):
        """
        self.process_model_objects: e.g.  {(object_type, object_name), object_}
        :return:
        """
        self.process_time_model_df = pd.read_excel(self.xlsx_content,
                                                   index_col=[0, 1], sheet_name="ProcessTimeModel", skiprows=1)
        self.process_time_model_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingProcessTimeModels"]
        process_time_models, _ = self.object_instantiation.load_dict(object_df=self.process_time_model_df,
                                                                     mapping_class=mapping_class,
                                                                     input_objects=[])

        self.process_quality_model_df = pd.read_excel(self.xlsx_content,
                                                      index_col=[0, 1], sheet_name="QualityModel", skiprows=1)
        self.process_quality_model_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingQualityModels"]
        quality_models, _ = self.object_instantiation.load_dict(object_df=self.process_quality_model_df,
                                                                mapping_class=mapping_class,
                                                                input_objects=[])

        self.process_resource_model_df = pd.read_excel(self.xlsx_content,
                                                       index_col=[0, 1], sheet_name="ResourceModel", skiprows=1)
        self.process_resource_model_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingResourceModels"]
        resource_models, _ = self.object_instantiation.load_dict(object_df=self.process_resource_model_df,
                                                                 mapping_class=mapping_class,
                                                                 input_objects=[self.entity_type_objects,
                                                                                self.resource_objects])

        self.process_transition_model_df = pd.read_excel(self.xlsx_content,
                                                         index_col=[0, 1], sheet_name="TransitionModel", skiprows=1)
        self.process_transition_model_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingTransitionModels"]
        transition_models, _ = self.object_instantiation.load_dict(object_df=self.process_transition_model_df,
                                                                   mapping_class=mapping_class,
                                                                   input_objects=[self.entity_type_objects,
                                                                                  self.resource_objects])

        self.process_transformation_model_df = pd.read_excel(self.xlsx_content,
                                                             index_col=[0, 1], sheet_name="TransformationModel",
                                                             skiprows=1)
        self.process_transformation_model_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingTransformationModels"]
        transformation_models, _ = self.object_instantiation.load_dict(object_df=self.process_transformation_model_df,
                                                                       mapping_class=mapping_class,
                                                                       input_objects=[self.entity_type_objects,
                                                                                      self.resource_objects])

        self.process_model_objects = (process_time_models | quality_models | resource_models | transition_models |
                                      transformation_models)

    def create_process_controllers_from_excel(self):
        self.process_controller_df = pd.read_excel(self.xlsx_content,
                                                   index_col=[0, 1], sheet_name="ProcessController", skiprows=1)
        self.process_controller_df.dropna(how='all', inplace=True)

        self.process_controller_objects, _ = (
            self.object_instantiation.load_dict(object_df=self.process_controller_df,
                                                mapping_class=MappingProcessControllers,
                                                input_objects=[self.process_model_objects]))

    def create_processes_from_excel(self):
        """
        self.process_objects: e.g.  {(object_type, object_name), object_}
        """
        self.process_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Process", skiprows=1)

        self.process_df.dropna(how='all', inplace=True)
        mapping_class = type(self).mappings["MappingProcess"]
        self.process_objects, _ = self.object_instantiation.load_dict(object_df=self.process_df,
                                                                      mapping_class=mapping_class,
                                                                      input_objects=[self.entity_type_objects,
                                                                                     self.resource_objects,
                                                                                     self.process_controller_objects,
                                                                                     self.sales_objects])

        # checking
        process_objects = {process_key: process_object
                           for process_key, process_object in self.process_objects.items()
                           if "Process" == process_key[0] or "ValueAddedProcess" in process_key[0]}
        for process_key, process_object in process_objects.items():
            completely_filled, not_completely_filled_attributes = process_object.completely_filled()
            if not completely_filled:
                print(f"The process {process_key} is not completely filled. \n"
                      f"The following attributes are not completely filled: {not_completely_filled_attributes}")

    def handle_duplications(self):
        if self.correction_parts_dict is not None:
            # handle duplications - only for the existing objects possible - not for the function calls
            # preselection
            stored_entities_df = self.resource_df["stored_entities"].dropna().astype("str").str
            part_index_dict = {}
            storages_to_correct = []
            for part in list(self.correction_parts_dict.keys()):
                if "static_model" in part.external_identifications:
                    static_model_name = part.external_identifications["static_model"]
                else:
                    static_model_name = list(part.external_identifications.values())[0]
                test_series = stored_entities_df.findall(static_model_name[0][1:]).dropna()

                mask = test_series.astype("str").str.len() > 2
                mask_index = mask[mask].index.to_list()
                storages_to_correct.extend(mask_index)
                part_index_dict[part] = ("stored_entities", mask_index)

            objects_to_consider = {(class_name, name): object_
                                   for (class_name, name), object_ in self.resource_objects.items()
                                   if (class_name, name) in storages_to_correct}
            new_objects = self.object_instantiation.change_attributes(objects_to_consider,
                                                                      self.correction_parts_dict,
                                                                      attributes_to_change=part_index_dict)
            self.resource_objects.update(new_objects)

        if self.corrected_factory_dict:
            # handle duplications - only for the existing objects possible - not for the function calls
            possibly_relevant_dfs = [self.resource_df, self.time_df, self.parts_df, self.process_df]
            new_objects = \
                self.object_instantiation.change_attributes(self.resource_objects, self.corrected_factory_dict)
            self.resource_objects.update(new_objects)

    def create_digital_twin(self):
        """create the digital_twin model object"""
        plant = self.get_plant_digital_twin()
        parts = self.get_parts_digital_twin()

        allowed_key_names = ["StationaryResources", "Storage", "WorkStation", "Warehouse", "ConveyorBelt",
                             "PassiveMovingResource", "ActiveMovingResource", "NonStationaryResources"]
        objects_mapped_to_entity_type_resources = convert_tuple_keys_to_nested_dict(self.resource_objects,
                                                                                    allowed_key_names=allowed_key_names)

        stationary_resources = self.get_stationary_resources_digital_twin(objects_mapped_to_entity_type_resources)
        passive_moving_resources = (
            self.get_passive_moving_resources_digital_twin(objects_mapped_to_entity_type_resources))
        active_moving_resources = self.get_active_moving_resources_digital_twin(objects_mapped_to_entity_type_resources)

        processes = self.get_processes()
        order_pool = self.get_order_pool_digital_twin()
        process_executions = []
        entity_types = self.get_entity_types_digital_twin()

        allowed_key_names_etn = ["EntityTransformationNode"]
        objects_mapped_to_entity_type = convert_tuple_keys_to_nested_dict(self.process_objects,
                                                                          allowed_key_names=allowed_key_names_etn)
        entity_transformation_nodes = self.get_part_transformation_nodes(objects_mapped_to_entity_type)

        customer_base = self.get_customers_digital_twin()
        features = list(self.features.values())
        feature_clusters = self.get_feature_clusters_digital_twin()

        digital_twin_model = StateModel(
            plant=plant,
            parts=parts,
            obstacles=[],
            stationary_resources=stationary_resources,
            passive_moving_resources=passive_moving_resources,
            active_moving_resources=active_moving_resources,
            processes=processes,
            order_pool=order_pool,
            process_executions=process_executions,
            entity_types=entity_types,
            entity_transformation_nodes=entity_transformation_nodes,
            customer_base=customer_base,
            features=features,
            feature_clusters=feature_clusters)
        self.digital_twin = {
            "digital_twin": digital_twin_model

        }

    def _deposit_digital_twin_in_process_models(self):
        digital_twin = self.get_digital_twin()
        for _, process_model in self.process_model_objects.items():
            if DTModelLearningExtension is not None:
                if isinstance(process_model, DTModelLearningExtension):
                    process_model.set_digital_twin_model(digital_twin_model=digital_twin)

    # ==== digital_twin_specific_methods ===============================================================================

    def get_plant_digital_twin(self):
        """return a plant object"""
        return list(self.plant.values())[0]

    def get_stationary_resources_digital_twin(self, objects_mapped_to_entity_type):
        """return all stationary_resources"""
        # plant_sheet of the Excel file

        stationary_resources = self.finalize_input_params_factory(key_name='StationaryResources',
                                                                  factory_dict=objects_mapped_to_entity_type)
        storages = self.finalize_input_params_factory(key_name='Storage',
                                                      factory_dict=objects_mapped_to_entity_type)
        work_stations = self.finalize_input_params_factory(key_name='WorkStation',
                                                               factory_dict=objects_mapped_to_entity_type)
        warehouses = self.finalize_input_params_factory(key_name='Warehouse',
                                                        factory_dict=objects_mapped_to_entity_type)
        conveyor_belts = self.finalize_input_params_factory(key_name='ConveyorBelt',
                                                            factory_dict=objects_mapped_to_entity_type)
        stationary_resources_all = stationary_resources | storages | work_stations | warehouses | conveyor_belts

        return stationary_resources_all

    def get_passive_moving_resources_digital_twin(self, factory):
        """return all passive_moving_resources"""

        passive_moving_resources = \
            self.finalize_input_params_factory(key_name='PassiveMovingResource', factory_dict=factory)

        return passive_moving_resources

    def get_active_moving_resources_digital_twin(self, factory):

        active_moving_resources = self.finalize_input_params_factory(key_name='ActiveMovingResource',
                                                                     factory_dict=factory)
        return active_moving_resources

    def get_parts_digital_twin(self):
        entity_type_part_match = {}
        for part in list(self.parts.values()):
            if part.entity_type in entity_type_part_match:
                entity_type_part_match[part.entity_type].append(part)
            else:
                entity_type_part_match[part.entity_type] = [part]

        return entity_type_part_match

    def get_part_transformation_nodes(self, objects_mapped_to_entity_type):
        part_transformation_nodes = list(objects_mapped_to_entity_type['EntityTransformationNode'].values())
        return part_transformation_nodes

    def get_entity_types_digital_twin(self):
        return list(self.entity_types.values())

    def get_order_pool_digital_twin(self):
        # orders
        order_pool = list(self.orders.values())
        return order_pool

    def get_customers_digital_twin(self):
        return list(self.customer_objects.values())

    def get_feature_clusters_digital_twin(self):
        feature_cluster_dict = {}
        for feature_cluster in list(self.feature_clusters.values()):
            feature_cluster_dict.setdefault(feature_cluster.product_class,
                                            []).append(feature_cluster)
        return feature_cluster_dict

    def finalize_input_params_factory(self, key_name, factory_dict):
        """
        Convert the factory dict to a dict needed for the digital twin
        :param key_name: Object name (e.g. "Stationary Resources")
        :param factory_dict: {key_name: list[object regarding key_name]}
        :return: a dict with the resource_objects and their respective elements in a list as key {entity_type: [resources]}
        """

        # early termination
        if key_name not in factory_dict.keys():
            return {}

        # create a dict {entity_type: [resources]}
        new_dict = defaultdict(list)
        for val in list(factory_dict[key_name].values()):
            new_dict[val.entity_type].append(val)
        return new_dict

    # ==== get_objects =================================================================================================

    def get_processes(self):
        all_processes = {}
        all_processes[ValueAddedProcess] = list(self.value_added_processes.values())
        all_processes[Process] = list(self.processes.values())
        return all_processes

    def get_all_digital_twin_objects(self):
        return self.all_digital_twin_objects

    def get_all_objects_digital_twin(self):
        return self.all_objects

    def get_digital_twin(self) -> StateModel:
        return self.digital_twin["digital_twin"]

    def to_pickle(self, path=None):
        self.next_id = DigitalTwinObject.next_id
        if path is None:
            path = self.path.__str__().split(".")[0] + ".pkl"

        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, digital_twin_pickle_path):
        digital_twin_objects = load_from_pickle(digital_twin_pickle_path)

        # needed because the class attributes are not stored by the pickle object
        DigitalTwinObject.next_id = digital_twin_objects.next_id
        return digital_twin_objects