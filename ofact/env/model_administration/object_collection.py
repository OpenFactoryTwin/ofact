from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import random
from datetime import timedelta, datetime
from enum import Enum
import itertools

from ofact.env.data_integration.sm_object_provision import set_value
from ofact.env.model_administration.cache import ObjectCacheAutoModelling
from ofact.env.model_administration.helper import get_attr_value
from ofact.env.model_administration.pipeline_settings import PipelineType
from ofact.env.model_administration.standardization.data_entry_mapping import DataEntryMapping
from ofact.env.model_administration.standardization.event_log_standard import EventLogStandardAttributes, \
    EventLogOrderAttributes, EventLogStandardClasses, EventLogStandardHandling
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.processes import ValueAddedProcess
from ofact.twin.state_model.entities import EntityType

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Entity
    from ofact.twin.state_model.processes import Process, ProcessExecution


class TimeExpression(Enum):
    """
    SINGLE_EVENT: Event is a single event
    START_OR_END_EVENT: This represents a condition where the event is the start OR the end
    START_AND_END_EVENT: This represents a condition where the event is BOTH start AND end of different executions
    (executed subsequently by the same resource)
    EXECUTION: Start and End of the Execution are in the row
    """

    SINGLE_EVENT = "Single Event"

    START_OR_END_EVENT = "Start or End Event"
    START_AND_END_EVENT = "Start and End Event"

    EXECUTION = "Execution"


def _check_event_single(columns):
    return _check_in_columns(columns,
                             possible_column_name=EventLogStandardAttributes.EVENT_TIME_SINGLE.string)


def _check_event_trace(columns):
    return _check_in_columns(columns,
                             possible_column_name=EventLogStandardAttributes.EVENT_TIME_TRACE.string)


def _check_start_or_end_event(df):
    # execution id is more than one time available

    execution_column_name = _get_column_name(df.columns, EventLogStandardClasses.EXECUTION.string)
    if execution_column_name is None:
        raise Exception("Execution ID should be specified")

    execution_id_s = df[execution_column_name].dropna()
    execution_id_unique_s = execution_id_s.unique()

    tolerance = 0.1
    tolerance_lower = 0.5 - tolerance
    tolerance_upper = 0.5 + tolerance
    if ((tolerance_lower * len(execution_id_s) < len(execution_id_unique_s)) &
            (len(execution_id_unique_s) < tolerance_upper * len(execution_id_s))):
        return True
    else:
        return False


def _check_start_and_end_event(df):
    # execution id is only one time available
    execution_column_name = _get_column_name(df.columns, EventLogStandardClasses.EXECUTION.string)

    execution_id_s = df[execution_column_name].dropna()
    execution_id_unique_s = execution_id_s.unique()

    tolerance = 0.1
    tolerance_lower = 1 - tolerance
    tolerance_upper = 1 + tolerance
    if ((tolerance_lower * len(execution_id_s) < len(execution_id_unique_s)) &
            (len(execution_id_unique_s) < tolerance_upper * len(execution_id_s))):
        return True
    else:
        return False


def _check_execution(columns):
    time_columns_length = len(set(column
                                  for column in columns
                                  if EventLogStandardAttributes.EXECUTION_START_TIME.string in column or
                                  EventLogStandardAttributes.EXECUTION_END_TIME.string in column))
    if time_columns_length == 2:
        return True
    else:
        return False


def _check_in_columns(columns, possible_column_name):
    return bool([column
                 for column in columns
                 if possible_column_name in column])


def _get_column_name(columns, possible_column_name):
    possible_column_names = _get_column_names(columns, possible_column_name)
    if len(possible_column_names) == 1:
        return possible_column_names[0]
    elif len(possible_column_names) > 1:

        raise Exception(possible_column_names)
    else:
        return None

def _get_number_column_names(columns):
    possible_column_names = [column
                             for column in columns
                             if EventLogStandardHandling.NUMBER.string in column]
    return possible_column_names


def _get_column_names(columns, possible_column_name):
    possible_column_names = [column
                             for column in columns
                             if possible_column_name in column]
    return possible_column_names


class ObjectsCollection:

    def __init__(self, event_log_df: Optional[pd.DataFrame()] = None, event_log_file: Optional[str] = None,
                 data_entry_mapper: list[DataEntryMapping] = None, source_application_name: str = "Standard", generator = None,
                 cache: Optional[ObjectCacheAutoModelling] = None, mode = None):
        """
        The state model can be derived from data.
        The auto state model generation is used to derive/learn a state model from a standardized event log.
        The standardized event log is defined in the OFacT event log standard.

        Parameters
        ----------
        One of the two is required event_log_df or event_log_file used to create the event_log_df.
        Firstly, the event_log_df is used.
        data_entry_mapper: a list of all entry_mappers for each entry
        """
        if event_log_df is not None:
            self.df = event_log_df
        elif event_log_file is not None:
            self.df = pd.read_csv(event_log_file)
        else:
            raise NotImplementedError

        if not isinstance(self.df, pd.DataFrame):
            raise NotImplementedError(f"[{self.__class__.__name__}] The event log is not a DataFrame")

        df_columns = self.df.columns
        if _check_event_single(df_columns):
            self.time_expression = TimeExpression.SINGLE_EVENT
            print(f"{source_application_name}: Single Event")
        elif _check_event_trace(df_columns):
            if _check_start_or_end_event(self.df):
                self.time_expression = TimeExpression.START_OR_END_EVENT
                print(f"{source_application_name}: START_OR_END Event")
            elif _check_start_and_end_event(self.df):
                self.time_expression = TimeExpression.START_AND_END_EVENT
                print(f"{source_application_name}: START_AND_END Event")
            else:
                raise NotImplementedError(f"[{self.__class__.__name__}] ")
        elif _check_execution(df_columns):
            self.time_expression = TimeExpression.EXECUTION
            print(f"{source_application_name}: Execution")

        else:
            self.time_expression = None
            print(f"[{self.__class__.__name__}] The event log does not contain a time column")

        self.resource_mappings = [column
                                  for column in self.df.columns
                                  if EventLogStandardClasses.STATIONARY_RESOURCE.string in column]
        self.part_mappings = [column
                              for column in self.df.columns
                              if EventLogStandardClasses.PART.string in column]

        self.entry_mapper_dict = {entry.get_new_column_name(): entry
                                  for entry in data_entry_mapper}
        self.source_application_name = source_application_name

        if generator is None:
            raise NotImplementedError("No generator specified")
        self.generator = generator

        if cache is None:
            cache = ObjectCacheAutoModelling()
        self._cache = cache

        self.mode: PipelineType = mode

        order_date_columns = _get_column_names(self.df.columns, EventLogOrderAttributes.ORDER_DATE.string)
        if len(order_date_columns) > 0:
            order_date_column = order_date_columns[0]
        else:
            order_date_column = None

        release_data_actual_columns = (
            _get_column_names(self.df.columns, EventLogOrderAttributes.RELEASE_DATE_ACTUAL.string))
        if len(release_data_actual_columns) > 0:
            release_date_actual_column = release_data_actual_columns[0]
        else:
            release_date_actual_column = None

        delivery_date_actual_columns = (
            _get_column_names(self.df.columns, EventLogOrderAttributes.DELIVERY_DATE_ACTUAL.string))
        if len(delivery_date_actual_columns) > 0:
            delivery_date_actual_column = delivery_date_actual_columns[0]
        else:
            delivery_date_actual_column = None

        self.attribute_columns = {"order_date": order_date_column,
                                  "release_date_actual": release_date_actual_column,
                                  "delivery_date_actual": delivery_date_actual_column}

    def get_objects_from_event_log(self):
        """
        last: last state model update
        """
        self._get_resources()
        self._get_parts()

        self._get_orders()
        if self.mode == PipelineType.MODEL_GENERATION:
            self._get_processes()
        elif self.mode == PipelineType.DATA_INTEGRATION:
            self._cache_processes()
        self._create_process_executions()

        # integration into the digital twin state model
        entity_types = self._get_sm_objects("EntityType")
        plant = self.generator.get_plant(name="Plant")  # connection not mandatory
        parts = self._get_sm_objects("Part")
        obstacles = []
        stationary_resources = self._get_sm_objects("StationaryResource")
        passive_moving_resources = {}
        active_moving_resources = {}
        entity_transformation_nodes = []
        processes = self._get_sm_objects("Process")
        process_executions = self._get_sm_objects("ProcessExecution")

        print("PEs:", len(process_executions))

        order_pool = self._get_sm_objects("Order")
        customer_base = self._get_sm_objects("Customer")
        features = self._get_sm_objects("Feature")
        feature_clusters = self._get_sm_objects("FeatureCluster")

        return (entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                active_moving_resources, entity_transformation_nodes, processes, process_executions, order_pool,
                customer_base, features, feature_clusters)

    def _get_sm_objects(self, class_name):
        sm_objects = self._cache.get_objects_by_class(class_name)

        if class_name in ["Part", "StationaryResource", "PassiveMovingResource", "ActiveMovingResource"]:
            updated_sm_objects = {}
            for sm_object in sm_objects:
                entity_type = get_attr_value(sm_object, "entity_type")
                if isinstance(entity_type, dict):
                    # could be the case in the data integration (handled in later steps)
                    entity_type = None
                updated_sm_objects.setdefault(entity_type,
                                              []).append(sm_object)
                if entity_type is None:
                    continue
                super_entity_type = get_attr_value(entity_type, "super_entity_type")
                if super_entity_type is not None:
                    if isinstance(super_entity_type, dict):
                        super_entity_type = None
                    updated_sm_objects.setdefault(super_entity_type,
                                                  []).append(sm_object)
            sm_objects = updated_sm_objects

        elif "Process" == class_name:
            sm_objects = StateModel.prepare_processes(sm_objects)

        elif "FeatureCluster" == class_name:
            updated_sm_objects = {}
            for sm_object in sm_objects:
                product_class = get_attr_value(sm_object, "product_class")
                updated_sm_objects.setdefault(product_class,
                                              []).append(sm_object)
                continue  # ToDo
                if sm_object.product_class.super_entity_type is not None:
                    updated_sm_objects.setdefault(sm_object.product_class.super_entity_type,
                                                  []).append(sm_object)
            sm_objects = updated_sm_objects

        return sm_objects

    def _get_resources(self):

        number_of_resources_columns = (
            _get_number_column_names(self.df.columns))  # ToDo: EventLogStandardHandling
        execution_id_column = _get_column_name(self.df.columns, EventLogStandardClasses.EXECUTION.string)

        resources_id_column_numbers = {}
        if number_of_resources_columns is not None:
            for number_of_resources_column_name in number_of_resources_columns:
                entry_mapper_number_of_resources = self.entry_mapper_dict[number_of_resources_column_name]
                matching_resource_id_columns = \
                    [resource_id_column
                     for resource_id_column in self.resource_mappings
                     if EventLogStandardClasses.STATIONARY_RESOURCE.string + "_" +  # ToDo: maybe also different
                     str(int(entry_mapper_number_of_resources.reference_identification)) == resource_id_column]
                if matching_resource_id_columns:
                    resources_id_column_numbers[matching_resource_id_columns[0]] = number_of_resources_column_name

        for resource_id_column in self.resource_mappings:
            if "Origin" in resource_id_column or "Destination" in resource_id_column:
                continue
            self._handle_resource_column(resource_id_column, resources_id_column_numbers, execution_id_column)

        self._get_origin_and_destination_resources()

    def _handle_resource_column(self, resource_id_column, resources_id_column_numbers, execution_id_column):

        # resources
        if resource_id_column in resources_id_column_numbers:
            stationary_resource_id_s = self.df[resource_id_column]
        else:
            stationary_resource_id_s = self.df.drop_duplicates([resource_id_column])[resource_id_column]
        stationary_resource_ids = stationary_resource_id_s.dropna().to_list()

        # type
        if _check_in_columns(self.df.columns, EventLogStandardClasses.RESOURCE_TYPE.string):
            requested_reference = self.entry_mapper_dict[resource_id_column].identification
            possible_resource_type_columns = self._get_reference_column_name(requested_reference)
            resource_type_columns = \
                [possible_resource_type_colum
                 for possible_resource_type_colum in possible_resource_type_columns
                 if EventLogStandardClasses.RESOURCE_TYPE.string in possible_resource_type_colum]

            if len(resource_type_columns) > 0:
                resource_type_column = resource_type_columns[0]
                resource_type_ids = self.df.loc[stationary_resource_id_s.index, resource_type_column].to_list()
            else:
                resource_type_ids = ["resource"
                                     for i in range(len(stationary_resource_ids))]
        else:
            resource_type_ids = ["resource"
                                 for i in range(len(stationary_resource_ids))]

        # number of resources
        if resource_id_column in resources_id_column_numbers:
            number_of_resources_colum = resources_id_column_numbers[resource_id_column]
            numbers_of_resources = self.df[number_of_resources_colum].to_list()
            if execution_id_column:
                tags_list = self.df[execution_id_column].to_list()
                tags_list = [[x] for x in tags_list]
            else:
                tags_list = [[]] * len(stationary_resource_ids)
        else:
            numbers_of_resources = [1] * len(stationary_resource_ids)
            if execution_id_column:
                tags_dict = self.df.groupby([resource_id_column])[execution_id_column].apply(list).to_dict()
                tags_list = [tags_dict[stationary_resource_id]
                             for stationary_resource_id in stationary_resource_ids]
            else:
                tags_list = [[]] * len(stationary_resource_ids)

        # instantiate
        for stationary_resource_id, resource_type_id, numbers_of_resources, tags in (
                zip(stationary_resource_ids, resource_type_ids, numbers_of_resources, tags_list)):

            if (stationary_resource_id == "nan"
                    or stationary_resource_id is None
                    or stationary_resource_id != stationary_resource_id):
                continue

            try:
                numbers_of_resources = int(numbers_of_resources)
            except ValueError:
                numbers_of_resources = 1

            self._handle_resource(stationary_resource_id, resource_type_id,
                                  tags=[], take_existing=True, minimum_number_available=numbers_of_resources)

    def _get_origin_and_destination_resources(self):
        if _check_in_columns(self.df.columns, EventLogStandardAttributes.ORIGIN.string):
            origin_column_name = _get_column_name(self.df.columns, EventLogStandardAttributes.ORIGIN.string)
            stationary_resource_ids = (
                self.df.drop_duplicates([origin_column_name])[origin_column_name].to_list())
            for stationary_resource_id in stationary_resource_ids:
                self._handle_resource(stationary_resource_id, take_existing=True)

        if _check_in_columns(self.df.columns, EventLogStandardAttributes.DESTINATION.string):
            destination_column_name = (
                _get_column_name(self.df.columns, EventLogStandardAttributes.DESTINATION.string))
            stationary_resource_ids = (
                self.df.drop_duplicates([destination_column_name])[destination_column_name].to_list())
            for stationary_resource_id in list(set(stationary_resource_ids)):
                self._handle_resource(stationary_resource_id, take_existing=True)

    def _handle_resource(self, stationary_resource_id, resource_type_id=None, take_existing: bool = False,
                         tags=None, minimum_number_available: int = 1):
        """Instantiate a StationaryResource if not already available for the id and take_existing True"""

        stationary_resource_id = str(stationary_resource_id)

        resource_type_id = resource_type_id if resource_type_id is not None else stationary_resource_id
        entity_type = self._cache.get_object(class_name="EntityType", id_=resource_type_id)
        if entity_type is None:
            print("ResourceTypeID: ", resource_type_id)
            entity_type = self.generator.get_entity_type(name=resource_type_id)
            self._cache.cache_object(class_name="EntityType", external_id=resource_type_id, sm_object=entity_type)

        if not take_existing:
            stationary_resource_objects = []
        else:  # search for existing objects
            stationary_resource_objects = self._cache.get_objects(class_name="StationaryResource",
                                                                  id_=stationary_resource_id)
            if stationary_resource_objects is None:
                stationary_resource_objects = []

        if len(stationary_resource_objects) < minimum_number_available:
            to_create = minimum_number_available - len(stationary_resource_objects)
            for i in range(to_create):
                stationary_resource_object = self.generator.get_stationary_resource(name=stationary_resource_id,
                                                                                        entity_type=entity_type)

                self._cache.cache_object("StationaryResource", stationary_resource_id,
                                         stationary_resource_object, tags=tags)

    def _get_parts(self):
        number_of_parts_column_names = _get_number_column_names(self.df.columns)
        execution_id_column = _get_column_name(self.df.columns, EventLogStandardClasses.EXECUTION.string)

        potential_individual_attributes_column_names = (
            _get_column_names(self.df.columns, EventLogStandardAttributes.INDIVIDUAL_ATTRIBUTE.string))
        individual_attributes_column_names = {}
        for individual_attributes_column_name in potential_individual_attributes_column_names:
            for part_column_name in self.part_mappings:
                reference = str(self.entry_mapper_dict[individual_attributes_column_name].reference_identification)
                if reference in part_column_name:
                    individual_attributes_column_names.setdefault(part_column_name,
                                                                  []).append(individual_attributes_column_name)

        part_id_column_numbers = {}
        if number_of_parts_column_names is not None:
            for number_of_parts_column_name in number_of_parts_column_names:
                entry_mapper_number_of_parts = self.entry_mapper_dict[number_of_parts_column_name]
                matching_part_id_columns = \
                    [part_id_column
                     for part_id_column in self.part_mappings
                     if EventLogStandardClasses.PART.string + "_" +
                     str(int(entry_mapper_number_of_parts.reference_identification)) == part_id_column]
                if matching_part_id_columns:
                    part_id_column_numbers[matching_part_id_columns[0]] = number_of_parts_column_name

        for part_id_column in self.part_mappings:
            individual_attributes_part_column_names = individual_attributes_column_names.get(part_id_column, [])
            self._handle_part_column(part_id_column, part_id_column_numbers, execution_id_column,
                                     individual_attributes_part_column_names)

    def _handle_part_column(self, part_id_column, part_id_column_numbers, execution_id_column,
                            individual_attributes_column_names):
        """
        ToDo
        """

        # parts
        if part_id_column in part_id_column_numbers:
            part_id_s = self.df[part_id_column].dropna()
        else:
            part_id_s = self.df.drop_duplicates([part_id_column])[part_id_column].dropna()
        part_ids = part_id_s.to_list()

        # type
        if _check_in_columns(self.df.columns, EventLogStandardClasses.PART_TYPE.string):
            requested_reference = self.entry_mapper_dict[part_id_column].identification
            possible_part_type_columns = self._get_reference_column_name(requested_reference)

            part_type_columns = [possible_part_type_colum
                                 for possible_part_type_colum in possible_part_type_columns
                                 if EventLogStandardClasses.PART_TYPE.string in possible_part_type_colum]

            if len(part_type_columns) > 0:
                part_type_column = part_type_columns[0]
                part_type_ids = self.df.loc[part_id_s.index, part_type_column].to_list()
            else:
                part_type_ids = ["part"
                                 for i in range(len(part_ids))]

        else:
            part_type_ids = ["part"
                             for i in range(len(part_ids))]

        # number of parts
        if part_id_column in part_id_column_numbers:
            number_of_parts_colum = part_id_column_numbers[part_id_column]
            numbers_of_parts = self.df.loc[part_id_s.index, number_of_parts_colum].to_list()
            if execution_id_column:
                tags_list = self.df.loc[part_id_s.index, execution_id_column].to_list()
                tags_list = [[x] for x in tags_list]
            else:
                tags_list = [[]] * len(part_ids)
        else:
            numbers_of_parts = [1] * len(part_ids)
            if execution_id_column:
                tags_dict = self.df.groupby([part_id_column])[execution_id_column].apply(list).to_dict()
                try:
                    tags_list = [tags_dict[part_id]
                             for part_id in part_ids]
                except KeyError:
                    print(execution_id_column, part_ids)
                    raise KeyError
            else:
                tags_list = [[]] * len(part_ids)

        if individual_attributes_column_names:
            individual_attribute_names = [self.entry_mapper_dict[individual_attributes_column_name].external_name
                                          for individual_attributes_column_name in individual_attributes_column_names]
            individual_attributes_columns = self.df[individual_attributes_column_names]
            df_renamed = individual_attributes_columns.rename(columns=dict(zip(individual_attributes_columns.columns,
                                                                               individual_attribute_names)))
            individual_attributes = df_renamed.to_dict(orient="records")
            parts_individual_attributes = dict(zip(part_ids, individual_attributes))
        else:
            parts_individual_attributes = None

        # instantiate
        for part_id, part_type_id, numbers_of_part, tags in zip(part_ids, part_type_ids, numbers_of_parts, tags_list):
            if part_id == "nan":
                continue

            try:
                numbers_of_part = int(numbers_of_part)
            except ValueError:
                numbers_of_part = 1
            except TypeError:
                numbers_of_part = 1

            if parts_individual_attributes is not None:
                part_individual_attributes = parts_individual_attributes[part_id]
            else:
                part_individual_attributes = None

            for _ in range(numbers_of_part):
                self._handle_part(part_id, part_type_id, tags=tags, part_individual_attributes=part_individual_attributes)

    def _handle_part(self, part_id, part_type_id=None, tags: Optional[list] = None,
                     part_individual_attributes: dict = None):

        part_id = str(part_id)
        entity_type = self._cache.get_object(class_name="EntityType", id_=part_type_id)
        if entity_type is None:
            print("PartTypeID: ", part_type_id)
            if (part_type_id == "nan" or
                    part_type_id != part_type_id):
                raise Exception(part_id)
            part_et = self.generator.get_entity_type(name=part_type_id)
            self._cache.cache_object("EntityType", part_type_id, part_et)
        else:
            part_et = self._cache.get_object(class_name="EntityType", id_=part_type_id)

        part_object = self.generator.get_part(name=part_id, entity_type=part_et, individual_attributes=part_individual_attributes)

        if get_attr_value(part_object, "entity_type") not in self._cache.get_objects_by_class("EntityType"):
            if self.mode != PipelineType.DATA_INTEGRATION:
                raise Exception("Entity Type should be specified", part_object.name)
            else:
                self._cache.cache_object("EntityType", part_type_id, part_et)

        self._cache.cache_object("Part", part_id, part_object, tags=tags)

    def _get_orders(self):

        order_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.ORDER.string)
        customer_column_name = _get_column_name(self.df.columns, EventLogStandardClasses.CUSTOMER.string)
        feature_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.FEATURE.string)

        order_product_classes_column_name = (
            _get_column_name(self.df.columns, EventLogOrderAttributes.PRODUCT_CLASSES.string))
        order_numbers_of_product_classes_column_names = _get_number_column_names(self.df.columns)
        possible_relations = \
            [order_numbers_of_product_classes_column_name
             for order_numbers_of_product_classes_column_name in order_numbers_of_product_classes_column_names
             if str(int(self.entry_mapper_dict[order_numbers_of_product_classes_column_name].reference_identification))
             in order_product_classes_column_name]

        if possible_relations:
            order_numbers_of_product_classes_column_name = possible_relations[0]
        else:
            order_numbers_of_product_classes_column_name = None

        event_time_single_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EVENT_TIME_SINGLE.string))
        event_time_trace_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EVENT_TIME_TRACE.string))
        executed_start_time_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EXECUTION_START_TIME.string))
        executed_end_time_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EXECUTION_END_TIME.string))

        all_orders = []
        product_classes = []
        number_of_product_classes = []
        features = {}
        for order_column_name in order_column_names:  # if you have more than one order column
            additional_orders_df = self.df.drop_duplicates([order_column_name])
            additional_orders = additional_orders_df[order_column_name].to_list()
            all_orders += additional_orders

            if len(order_column_names) > 1:
                raise NotImplementedError("More than one order column")

            if order_product_classes_column_name is not None:
                product_classes += self.df.drop_duplicates([order_column_name])[
                    order_product_classes_column_name].to_list()
            if order_numbers_of_product_classes_column_name:
                number_of_product_classes += self.df.drop_duplicates([order_column_name])[
                    order_numbers_of_product_classes_column_name].to_list()

            if feature_column_names is not None:
                feature_dict = additional_orders_df.set_index(order_column_name)[feature_column_names].T.to_dict('list')
                all_features = pd.unique(additional_orders_df[feature_column_names].values.ravel()).tolist()
                for feature_name in all_features:
                    feature_object = self._cache.get_object(class_name="Feature", id_=feature_name)
                    if feature_object is None:
                        feature_object = self.generator.get_feature(name=feature_name)
                        self._cache.cache_object("Feature", feature_name, feature_object)

                for k, lst in feature_dict.items():
                    feature_objects = [self._cache.get_object(class_name="Feature", id_=feature_name)
                                        for feature_name in lst]
                    features.setdefault(k,
                                        []).extend(feature_objects)

        # orders
        for idx, order_id in enumerate(all_orders):
            order_object = self._cache.get_object(class_name="Order", id_=order_id)
            if order_object is None:
                order_object = self.generator.get_order(name=order_id)
                self._cache.cache_object("Order", order_id, order_object)

            set_value(order_object, "identifier", order_id)

            available = False
            for order_column_name in order_column_names:
                order_group = self.df.loc[self.df[order_column_name] == order_id]
                if not order_group.empty:
                    available = True
                    break
            if not available:
                continue

            match self.time_expression:
                case TimeExpression.SINGLE_EVENT:
                    self._adapt_order_object_with_times(order_group, order_object,
                                                        time_column_name=event_time_single_column_name)

                case TimeExpression.START_OR_END_EVENT:
                    self._adapt_order_object_with_times(order_group, order_object,
                                                        time_column_name=event_time_trace_column_name)

                case TimeExpression.START_AND_END_EVENT:
                    self._adapt_order_object_with_times(order_group, order_object,
                                                        time_column_name=event_time_trace_column_name)

                case TimeExpression.EXECUTION:
                    self._adapt_order_object_with_times_executed(order_group, order_object,
                                                                 executed_start_time_column_name,
                                                                 executed_end_time_column_name)

            if product_classes:
                product_class = self._cache.get_object(class_name="EntityType", id_=product_classes[idx])

                if product_class is None:
                    if not isinstance(product_classes[idx], EntityType):
                        product_class = self.generator.get_entity_type(name=product_classes[idx])
                    else:
                        product_class = product_classes[idx]
                    self._cache.cache_object("EntityType", product_classes[idx], product_class)

                number_of_product_class = 1
                if number_of_product_classes:
                    if number_of_product_classes[idx] == number_of_product_classes[idx]:
                        number_of_product_class = number_of_product_classes[idx]

                product_classes_ = [product_class] * int(number_of_product_class)
                set_value(order_object, "product_classes", product_classes_)

            if customer_column_name:
                customer = self._cache.get_object(class_name="Customer", id_=customer_column_name)
                if customer is None:
                    customer = self.generator.get_customer(name=customer_column_name)
                    self._cache.cache_object("Customer", customer_column_name, customer)

                set_value(order_object, "customer", customer)

            if order_id in features:
                set_value(order_object, "features_requested", features[order_id])

    def _adapt_order_object_with_times(self, order_group, order_object, time_column_name):
        if self.attribute_columns["order_date"] is not None:
            order_date = pd.to_datetime(order_group[self.attribute_columns["order_date"]]).min()
            if order_date == order_date:
                set_value(order_object, "order_date", order_date)
        else:
            order_date = pd.to_datetime(order_group[time_column_name]).min()
            set_value(order_object, "order_date", order_date)

        if self.attribute_columns["release_date_actual"] is not None:
            entry_mapper: DataEntryMapping = self.entry_mapper_dict[self.attribute_columns["release_date_actual"]]
            if entry_mapper.required(self.mode):
                release_date_actual = pd.to_datetime(order_group[self.attribute_columns["release_date_actual"]]).min()
                if release_date_actual == release_date_actual:
                    set_value(order_object, "release_date_actual", release_date_actual)
        else:
            release_date_actual = pd.to_datetime(order_group[time_column_name]).min()
            if release_date_actual == release_date_actual:
                set_value(order_object, "release_date_actual", release_date_actual)

        delivery_date_requested = None
        set_value(order_object, "delivery_date_requested", delivery_date_requested)

        delivery_date_planned = pd.to_datetime(order_group[time_column_name]).max()
        set_value(order_object, "delivery_date_planned", delivery_date_planned)

        if self.attribute_columns["delivery_date_actual"] is not None:
            entry_mapper = self.entry_mapper_dict[self.attribute_columns["delivery_date_actual"]]
            if entry_mapper.required(self.mode):
                delivery_date_actual = pd.to_datetime(order_group[self.attribute_columns["delivery_date_actual"]]).max()
                if delivery_date_actual == delivery_date_actual:
                    set_value(order_object, "delivery_date_actual", delivery_date_actual)
        else:
            delivery_date_actual = pd.to_datetime(order_group[time_column_name]).max()
            if delivery_date_actual == delivery_date_actual:
                set_value(order_object, "delivery_date_actual", delivery_date_actual)

    def _adapt_order_object_with_times_executed(self, order_group, order_object, executed_start_time_column_name,
                                                executed_end_time_column_name):
        if self.attribute_columns["order_date"] is not None:
            order_date = pd.to_datetime(order_group[self.attribute_columns["order_date"]]).min()
            if order_date == order_date:
                set_value(order_object, "order_date", order_date)

        if self.attribute_columns["release_date_actual"] is not None:
            entry_mapper = self.entry_mapper_dict[self.attribute_columns["release_date_actual"]]
            if entry_mapper.required(self.mode):
                release_dates_actual = order_group[self.attribute_columns["release_date_actual"]]
                release_date_actual = pd.to_datetime(release_dates_actual).min()
                if release_date_actual == release_date_actual and isinstance(release_date_actual, datetime):
                    set_value(order_object, "release_date_actual", release_date_actual)
        else:
            release_date_actual = order_group[executed_start_time_column_name].iloc[0]
            if isinstance(release_date_actual, pd.Timestamp):
                release_date_actual = pd.to_datetime(release_date_actual)

            if release_date_actual == release_date_actual and isinstance(release_date_actual, datetime):
                set_value(order_object, "release_date_actual", release_date_actual)


        if self.attribute_columns["delivery_date_actual"] is not None:
            entry_mapper = self.entry_mapper_dict[self.attribute_columns["delivery_date_actual"]]
            if entry_mapper.required(self.mode):
                delivery_date_actual = pd.to_datetime(order_group[self.attribute_columns["delivery_date_actual"]]).max()
                if delivery_date_actual == delivery_date_actual and isinstance(delivery_date_actual, datetime):
                    set_value(order_object, "delivery_date_actual", delivery_date_actual)
        else:
            if self.mode != PipelineType.DATA_INTEGRATION:
                delivery_date_actual = pd.to_datetime(order_group[executed_end_time_column_name].iloc[0])
                if delivery_date_actual == delivery_date_actual  and isinstance(delivery_date_actual, datetime):
                    set_value(order_object, "delivery_date_actual", delivery_date_actual)

    def _get_quality_controller(self):
        # quality model
        quality_model_object = self.generator.get_quality_model(name="standard")
        self._cache.cache_object("QualityModel", quality_model_object.identification, quality_model_object)

        quality_controller_object = self.generator.get_quality_controller(quality_model=quality_model_object,
                                                                          name="standard")
        self._cache.cache_object("QualityController", quality_controller_object.identification,
                                 quality_controller_object)

        return quality_controller_object

    def _get_transition_controller(self, name="standard"):
        # transition model
        transition_model_object = self.generator.get_transition_model(name=name)
        self._cache.cache_object("TransitionModel", transition_model_object.identification, transition_model_object)

        transition_controller_object = (
            self.generator.get_transition_controller(transition_model=transition_model_object, name=name))
        self._cache.cache_object("TransitionController", transition_controller_object.identification,
                                 transition_controller_object)

        return transition_controller_object

    def _get_transformation_controller(self, name="standard"):
        # transformation model
        transformation_model_object = self.generator.get_transformation_model(name=name)
        self._cache.cache_object("TransformationModel", transformation_model_object.identification,
                                 transformation_model_object)

        transformation_controller_object = (
            self.generator.get_transformation_controller(transformation_model=transformation_model_object, name=name))
        self._cache.cache_object("TransformationController", transformation_controller_object.identification,
                                 transformation_controller_object)

        return transformation_controller_object

    def _get_process_time_model(self, name):
        process_time_model_object = self.generator.get_process_time_model(name=name)
        process_time_model_object.value = -1
        self._cache.cache_object("ProcessTimeModel", process_time_model_object.identification,
                                 process_time_model_object)

        process_time_controller_object = (
            self.generator.get_process_time_controller(process_time_model=process_time_model_object, name=name))
        self._cache.cache_object("ProcessTimeController", process_time_controller_object.identification,
                                 process_time_controller_object)

        return process_time_controller_object

    def _get_resource_model(self, name):
        # resource model

        resource_model_object = self.generator.get_resource_model(name=name)
        self._cache.cache_object("ResourceModel", resource_model_object.identification, resource_model_object)

        resource_controller = self.generator.get_resource_controller(resource_model=resource_model_object, name=name)
        self._cache.cache_object("ResourceController", resource_controller.identification, resource_controller)

        return resource_controller

    def _get_processes(self):
        feature_column_name = _get_column_name(self.df.columns, EventLogStandardClasses.FEATURE.string)

        quality_controller_object = self._get_quality_controller()

        process_names = []  # name
        if _check_in_columns(self.df.columns, EventLogStandardClasses.PROCESS.string):
            process_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.PROCESS.string)
            process_column_names = [process_column
                                    for process_column in process_column_names
                                    if EventLogStandardClasses.EXECUTION.string not in process_column]
            if len(process_column_names) > 1:
                raise Exception("More than one process column found")
            elif len(process_column_names) == 1:
                process_column_name = process_column_names[0]
                process_names = list(set(self.df[process_column_name].to_list()))

            if not process_names:
                process_names = ["Standard" + self.source_application_name]

        if feature_column_name is not None:
            feature_names = list(set(self.df[feature_column_name].to_list()))
            for feature_name in feature_names:
                feature_object = self.generator.get_feature(name=feature_name)
                self._cache.cache_object("Feature", feature_object.identification,
                                         feature_object)

        for process_name in process_names:
            transformation_controller_object = self._get_transformation_controller(name=process_name)
            transition_controller_object = self._get_transition_controller(name=process_name)
            process_time_controller_object = self._get_process_time_model(name=process_name)
            resource_controller_object = self._get_resource_model(name=process_name)

            if feature_column_name is not None:
                process_column_name = process_column_names[0]
                feature_name = self.df[self.df[process_column_name] == process_name][feature_column_name].iloc[0]
                feature = self.generator.get_feature(name=feature_name)
            else:
                feature = None

            # process
            process_object = self.generator.get_value_added_processes(
                name=process_name,
                quality_controller=quality_controller_object,
                transition_controller=transition_controller_object,
                transformation_controller=transformation_controller_object,
                lead_time_controller=process_time_controller_object,
                resource_controller=resource_controller_object,
                feature=feature,
                group="order_et")

            process_object.name = process_name

            self._cache.cache_object("Process", process_name, process_object)

    def _cache_processes(self):

        process_names = []  # name
        if _check_in_columns(self.df.columns, EventLogStandardClasses.PROCESS.string):
            process_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.PROCESS.string)
            process_column_names = [process_column
                                   for process_column in process_column_names
                                   if EventLogStandardClasses.EXECUTION.string not in process_column]
            if len(process_column_names) > 1:
                raise Exception("More than one process column found")
            elif len(process_column_names) == 1:
                process_column_name = process_column_names[0]
                process_names = list(set(self.df[process_column_name].to_list()))

            if not process_names:
                process_names = ["Standard" + self.source_application_name]

        for process_name in process_names:
            process_object = self.generator.get_value_added_processes(name=process_name)
            if isinstance(process_object, dict):
                normal_process_object = self.generator.get_processes(name=process_name)
                if not isinstance(normal_process_object, dict):
                    process_object = normal_process_object
            self._cache.cache_object("Process", process_name, process_object)

    def _create_process_executions(self):
        # process executions
        execution_id_column_name = _get_column_name(self.df.columns, EventLogStandardClasses.EXECUTION.string)
        if not execution_id_column_name:
            return

        event_time_single_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EVENT_TIME_SINGLE.string))
        event_time_trace_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EVENT_TIME_TRACE.string))
        execution_start_time_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EXECUTION_START_TIME.string))
        execution_end_time_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.EXECUTION_END_TIME.string))
        process_id_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.PROCESS.string)
        order_id_column_name = _get_column_names(self.df.columns, EventLogStandardClasses.ORDER.string)[0]
        potential_individual_attributes_column_names = (
            _get_column_names(self.df.columns, EventLogStandardAttributes.INDIVIDUAL_ATTRIBUTE.string))
        individual_attributes_column_names = \
            [individual_attributes_column_name
             for individual_attributes_column_name in potential_individual_attributes_column_names
             if str(self.entry_mapper_dict[individual_attributes_column_name].reference_identification)
             in execution_id_column_name]

        resulting_quality_column_name = (
            _get_column_name(self.df.columns, EventLogStandardAttributes.RESULTING_QUALITY.string))

        number_of_process_executions_names = _get_number_column_names(self.df.columns)

        possible_relations = \
            [order_numbers_of_product_classes_column_name
             for order_numbers_of_product_classes_column_name in number_of_process_executions_names
             if str(int(self.entry_mapper_dict[order_numbers_of_product_classes_column_name].reference_identification))
             in execution_id_column_name]

        if possible_relations:
            number_of_process_executions_name = possible_relations[0]
        else:
            number_of_process_executions_name = None

        match self.time_expression:
            case TimeExpression.SINGLE_EVENT:
                self.df.sort_values(by=event_time_single_column_name, inplace=True)
                entry_mapper = self.entry_mapper_dict[event_time_single_column_name]
            case TimeExpression.START_OR_END_EVENT:
                self.df.sort_values(by=event_time_trace_column_name, inplace=True)
                entry_mapper = self.entry_mapper_dict[event_time_trace_column_name]
            case TimeExpression.START_AND_END_EVENT:
                self.df.sort_values(by=event_time_trace_column_name, inplace=True)
                entry_mapper = self.entry_mapper_dict[event_time_trace_column_name]
            case TimeExpression.EXECUTION:
                self.df.sort_values(by=execution_start_time_column_name, inplace=True)
                entry_mapper = self.entry_mapper_dict[execution_start_time_column_name]
            case _:
                raise NotImplementedError()

        if entry_mapper.required(self.mode):
            times_required = True
        else:
            times_required = False
            executed_start_time = None
            executed_end_time = None

        self.df.reset_index(inplace=True)

        resource_column_names = _get_column_names(self.df.columns, EventLogStandardClasses.STATIONARY_RESOURCE.string)
        resource_column_names = [resource_column_name
                                 for resource_column_name in resource_column_names
                                 if "Origin" not in resource_column_name and "Destination" not in resource_column_name]

        if individual_attributes_column_names:
            individual_attribute_names = [self.entry_mapper_dict[individual_attributes_column_name].external_name
                                          for individual_attributes_column_name in individual_attributes_column_names]

        else:
            individual_attribute_names = None

        resource_event_traces = {}
        for idx, execution_rows in self.df.groupby([execution_id_column_name]):
            match self.time_expression:
                case TimeExpression.SINGLE_EVENT:
                    event_times = pd.to_datetime(execution_rows[event_time_single_column_name])
                    first_event_row = execution_rows.loc[int(event_times.idxmin())]
                    last_event_row = execution_rows.loc[int(event_times.idxmax())]
                    if times_required:
                        executed_start_time = event_times.min() if event_times.min() == event_times.min() else None
                        executed_end_time = event_times.max() if event_times.max() == event_times.max() else None

                case TimeExpression.START_OR_END_EVENT:
                    event_times = pd.to_datetime(execution_rows[event_time_trace_column_name])
                    first_event_row = execution_rows.loc[int(event_times.idxmin())]
                    last_event_row = execution_rows.loc[int(event_times.idxmax())]
                    if times_required:
                        executed_start_time = event_times.min() if event_times.min() == event_times.min() else None
                        executed_end_time = event_times.max() if event_times.max() == event_times.max() else None

                case TimeExpression.START_AND_END_EVENT:  # other behaviour
                    for resource_column_name in resource_column_names:
                        resource = execution_rows[resource_column_name].iloc[0]
                        if resource in resource_event_traces:
                            event_traces = resource_event_traces[resource]
                        else:
                            event_traces = (
                                self.df.sort_values(by=event_time_trace_column_name)[event_time_trace_column_name])
                            resource_event_traces[resource] = event_traces

                    event_times = pd.to_datetime(execution_rows[event_time_trace_column_name])
                    first_event_row = execution_rows.loc[int(event_times.idxmin())]
                    last_event_row = execution_rows.loc[int(event_times.idxmax())]

                    event_row = execution_rows.index[0]
                    events_before = event_traces.loc[event_traces.index < event_row]
                    if times_required:
                        if not events_before.empty:
                            executed_start_time = (
                                events_before.iloc[-1] if event_times.iloc[-1] == event_times.iloc[-1] else None)
                        else:
                            executed_start_time = None

                        executed_end_time = event_times.iloc[0] if event_times.iloc[0] == event_times.iloc[0] else None

                case TimeExpression.EXECUTION:
                    first_event_row = execution_rows.iloc[0]
                    last_event_row = execution_rows.iloc[0]
                    if times_required:
                        executed_start_time = execution_rows.iloc[0][execution_start_time_column_name]
                        if isinstance(executed_start_time, pd.Timestamp):
                            executed_start_time = pd.to_datetime(executed_start_time)
                        executed_end_time = execution_rows.iloc[0][execution_end_time_column_name]
                        if isinstance(executed_end_time, pd.Timestamp):
                            executed_end_time = pd.to_datetime(executed_end_time)
                        executed_start_time = executed_start_time if executed_start_time == executed_start_time else None
                        executed_end_time = executed_end_time if executed_end_time == executed_end_time else None

                case _:
                    raise NotImplementedError(f"TimeExpression {self.time_expression} not implemented")

            execution_id = last_event_row[execution_id_column_name]

            process_name = "Standard" + self.source_application_name
            for process_id_column_name in process_id_column_names:
                if EventLogStandardClasses.EXECUTION.string in process_id_column_name:
                    continue

                if process_id_column_name in last_event_row:
                    # ToDo: process_id_column_names
                    if last_event_row[process_id_column_name] == last_event_row[process_id_column_name]:
                        process_name = last_event_row[process_id_column_name]
                        break

            parts_involved_combinations = (
                self._get_entities_used(entity_class="Part",
                                        entity_entries=execution_rows[self.part_mappings],
                                        execution_id=execution_id))
            resources_not_origin_destination = \
                [resource_column
                 for resource_column in self.resource_mappings
                 if "origin" not in resource_column and "destination" not in resource_column and
                 str(self.entry_mapper_dict[resource_column].reference_identification) in execution_id_column_name]

            resources_used_combinations = (
                self._get_entities_used(entity_class="StationaryResource",
                                        entity_entries=execution_rows[resources_not_origin_destination],
                                        execution_id=execution_id))

            number_process_executions = 1
            if number_of_process_executions_name is None:
                parts_resources = [(parts_involved_combinations[0],
                                    resources_used_combinations[0])]

            else:
                number_process_executions = execution_rows[number_of_process_executions_name].sum()
                if (number_process_executions == number_process_executions and
                        number_process_executions is not None):
                    try:
                        number_process_executions = int(number_process_executions)
                    except ValueError:
                        number_process_executions = 1
                    if number_process_executions == 0:
                        number_process_executions = 1

                    parts_resources_combinations = (
                        itertools.product(parts_involved_combinations, resources_used_combinations))
                    parts_resources = [(parts, resources)
                                       for parts, resources in parts_resources_combinations]

                    if number_process_executions != len(parts_resources):
                        order_id = last_event_row[order_id_column_name]
                        raise Exception("Number of process executions does not match "
                                        "the number of part and resource combinations "
                                        "{} != {}, {}, {} ({}), {}, {}".format(number_process_executions,
                                                                           len(parts_resources),
                                                                           len(parts_involved_combinations),
                                                                           len(resources_used_combinations),
                                                                           order_id, execution_id, process_name))

                else:
                    parts_resources = [(parts_involved_combinations[0],
                                        resources_used_combinations[0])]

            if not number_process_executions:
                continue

            origin = self._get_origin(first_event_row, execution_rows)
            destination = self._get_destination(last_event_row, execution_rows)

            order_id = last_event_row[order_id_column_name]
            order = self._cache.get_object(class_name="Order", id_=order_id)
            process: Process = self._cache.get_object(class_name="Process", id_=process_name)
            if process is None or isinstance(process, dict):
                if self.mode == PipelineType.DATA_INTEGRATION:
                    raise Exception("Process should be specified", process_name)

            possible_main_resources = [resource[0]
                                       for resource in parts_resources[0][1]
                                       if process.check_ability_to_perform_process_as_main_resource(resource[0])]

            if not possible_main_resources:
                possible_main_resources = [parts_resources[0][1][0][0]]  # case generation
            main_resource = possible_main_resources[0]

            if executed_end_time is not None and executed_start_time is not None:
                time_difference = executed_end_time - executed_start_time
                try:
                    time_step = time_difference / number_process_executions
                except ZeroDivisionError:
                    time_step = time_difference / 1
                step_time_start = executed_start_time
                step_time_end = step_time_start + time_step
                
            elif executed_start_time is not None:
                step_time_start = executed_start_time
                step_time_end = None
                time_step = None
            elif executed_end_time is not None:
                step_time_start = None
                step_time_end = executed_end_time
                time_step = None
            else:
                step_time_start = None
                step_time_end = None
                time_step = None

            process_feature = None
            if isinstance(process, ValueAddedProcess):
                process_feature = process.feature
            elif isinstance(process, dict):
                if "feature" in process:
                    process_feature = process["feature"]

            if individual_attribute_names is not None:
                individual_attributes_columns = execution_rows[individual_attributes_column_names]
                df_renamed = individual_attributes_columns.rename(
                    columns=dict(zip(individual_attributes_columns.columns,
                                     individual_attribute_names)))
                individual_attributes = df_renamed.to_dict(orient="records")
            else:
                individual_attributes = None

            if resulting_quality_column_name is not None:
                resulting_quality = float(last_event_row[resulting_quality_column_name])
            else:
                resulting_quality = 1  # available ?

            for i in range(number_process_executions):
                parts_involved, resources_used = parts_resources[i]

                if time_step is not None:
                    step_time_end = step_time_start + time_step

                process_execution_plan_object = (
                    self.generator.get_process_execution_plan(execution_id=execution_id,
                                                              process=process,
                                                              executed_start_time=step_time_start,
                                                              executed_end_time=step_time_end,
                                                              parts_involved=parts_involved,
                                                              resources_used=resources_used,
                                                              main_resource=main_resource,
                                                              origin=origin,
                                                              destination=destination,
                                                              resulting_quality=resulting_quality,
                                                              order=order,
                                                              source_application=self.source_application_name,
                                                              individual_attributes=individual_attributes))

                if time_step is not None:
                    step_time_start = step_time_end

                if process_feature is not None:
                    set_value(order, "features_requested", process_feature)
                self._cache.cache_object("ProcessExecution", get_attr_value(process_execution_plan_object, "identification"),
                                         process_execution_plan_object)

    def _get_entities_used(self, entity_class, entity_entries, execution_id=None) -> list[list[tuple]]:
        """

        execution_id
        """

        entries = entity_entries.values.flatten().tolist()
        entry_list = list(set([str(entry) for entry in entries if str(entry) != "nan" and str(entry) != "None"]))
        if entity_class == "StationaryResource":
            execution_id = None

        entities_lists: list[list[Entity]] = [self._cache.get_objects(class_name=entity_class,
                                                                      id_=str(single_entity_entry),
                                                                      tag=execution_id)
                                              for single_entity_entry in entry_list]

        if entity_class == "StationaryResource":
            entities_combinations = [[entity
                                      for combo in entities_lists
                                      for entity in combo]]
        else:
            entities_combinations = [list(combo)
                                     for combo in itertools.product(*entities_lists)]
        entities_used_combinations = []
        for entities_combination in entities_combinations:
            for entity in entities_combination:
                if entity not in self._cache.get_objects_by_class(entity_class):
                    raise Exception("Entity should be specified", entity.name)
                if get_attr_value(entity, "entity_type") not in self._cache.get_objects_by_class("EntityType"):
                    entity_type = get_attr_value(entity, "entity_type")
                    self._cache.cache_object("EntityType", entity_type.get_static_model_id(), entity_type)
                    # raise Exception("Entity Type should be specified", get_attr_value(entity, "name"))
            entities_used = [(entity,)
                             for entity in entities_combination]

            entities_used_combinations.append(entities_used)
        return entities_used_combinations

    def _get_origin(self, first_event_row, execution_rows):
        origin_column_name = _get_column_name(self.df.columns, EventLogStandardAttributes.ORIGIN.string)
        if origin_column_name in first_event_row:
            origin = self._cache.get_object(class_name="StationaryResource", id_=first_event_row[origin_column_name])
        else:
            if "origin_resource" not in self._cache.get_objects_by_class("StationaryResource"):
                self._handle_resource("origin_resource")
            origin = self._cache.get_object(class_name="StationaryResource", id_="origin_resource")

        return origin

    def _get_destination(self, last_event_row, execution_rows):
        destination_column_name = _get_column_name(self.df.columns, EventLogStandardAttributes.DESTINATION.string)
        if destination_column_name in last_event_row:
            destination = self._cache.get_object(class_name="StationaryResource",
                                                 id_=last_event_row[destination_column_name])
        else:
            stationary_resources = self._cache.get_objects_by_class("StationaryResource")
            if "destination_resource" not in stationary_resources:
                self._handle_resource("destination_resource")
            destination = self._cache.get_object(class_name="StationaryResource", id_="destination_resource")

        return destination

    def _get_reference_column_name(self, requested_reference):
        return [column_name
                for column_name, data_entry_mapper in self.entry_mapper_dict.items()
                if data_entry_mapper.reference_identification == requested_reference]


if __name__ == "__main__":
    pass
