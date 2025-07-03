import os
from pathlib import Path

import pandas as pd
import yaml, importlib
from ofact.env.model_administration.pipeline_settings import PipelineType
from ofact.env.model_administration.standardization.data_entry_mapping import (DataEntryMapping, Filter, FilterOptions,
                                                                               EntryType)
from ofact.env.model_administration.standardization.event_log_standard import (
    EventLogStandardClasses, EventLogStandardAttributes, EventLogOrderAttributes, EventLogStandardHandling)


def get_source_specific_object_dicts(pipeline_building_blocks_path: str):
    with open(pipeline_building_blocks_path) as f:
        cfg = yaml.safe_load(f)

    adapter_dict = {k: load_class(v)
                    for k, v in cfg["adapters"].items()}
    standardization_dict = {k: load_class(v)
                            for k, v in cfg["standardizations"].items()}
    process_mining_dict = {k: load_class(v)
                           for k, v in cfg["process_minings"].items()}
    preprocessings_dict = {k: load_class(v)
                           for k, v in cfg["preprocessings"].items()}

    return adapter_dict, standardization_dict, process_mining_dict, preprocessings_dict


def get_general_object_dicts(pipeline_building_blocks_path: str):
    with open(pipeline_building_blocks_path) as f:
        cfg = yaml.safe_load(f)

    objects_collection_dict = {k: load_class(v)
                               for k, v in cfg["objects_collections"].items()}
    model_creation_dict = {k: load_class(v)
                           for k, v in cfg["model_creations"].items()}
    model_updates_dict = {k: load_class(v)
                          for k, v in cfg["model_updates"].items()}
    process_model_updates_dict = {k: load_class(v)
                                  for k, v in cfg["process_model_updates"].items()}
    state_model_smoothing_dict = {k: load_class(v)
                                  for k, v in cfg["state_model_smoothing"].items()}

    return (objects_collection_dict, model_creation_dict, model_updates_dict, process_model_updates_dict,
            state_model_smoothing_dict)


def load_class(path: str):
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def _get_entry(entry):
    return entry if entry == entry else None

pipeline_type_mapper = {PipelineType.MODEL_GENERATION: "MG",
                        PipelineType.DATA_INTEGRATION: "DI"}

def _get_source_settings(data_source_entry, project_path, column_mappings_df, pipeline_type,
                         pipeline_building_blocks_path: str = "pipeline_building_blocks.yaml"):
    """Create the adapter object with the information given from the data source model, created in an Excel file."""
    (adapter_dict, standardization_dict, process_mining_dict, preprocessings_dict) = (
        get_source_specific_object_dicts(pipeline_building_blocks_path))

    if data_source_entry["source"][0] == "*":
        data_source_entry["source"] = data_source_entry["mapping"][1:]
        general = False
    else:
        general = True

    if data_source_entry['Adapter'] == "MSSQL":
        external_source_path = data_source_entry['path']
    else:
        external_source_path = os.path.join(project_path, os.path.normpath(data_source_entry['path']))

    mapping_columns = ["external",
                       "MG", "DI",
                       "identification", "reference identification","class", "attribute", "handling", "depends on",
                       "value",
                       "combination_part", "mandatory", "entry_type",
                       "action", "separator", "operation id",  # separator
                       "old value", "replacing value", "delete",
                       "sort by",
                       "time filter",
                       "filter operation", "filter by",
                       "special_handling"]  # Why I need the special handling?

    pipeline_type_column = pipeline_type_mapper[pipeline_type]
    mask = ((column_mappings_df["source"] == data_source_entry["source"]) &
            (column_mappings_df[pipeline_type_column] == column_mappings_df[pipeline_type_column]))
    columns_df = column_mappings_df.loc[mask, mapping_columns]

    data_entry_mappings = []
    for idx, column in columns_df.iterrows():
        if column["class"] == column["class"]:
            sm_class = EventLogStandardClasses.from_string(column["class"])
            sm_class_str = sm_class.string
        else:
            sm_class_str = None

        if column["attribute"] == column["attribute"]:
            try:
                sm_attribute = EventLogStandardAttributes.from_string(column["attribute"])
            except KeyError:
                sm_attribute = EventLogOrderAttributes.from_string(column["attribute"])

            sm_attribute_str = sm_attribute.string
        else:
            sm_attribute_str = None

        if column["handling"] == column["handling"]:
            handling = EventLogStandardHandling.from_string(column["handling"])
            handling_str = handling.string
        else:
            handling_str = None

        if column["mandatory"] == column["mandatory"]:
            scope, mandatory_value = column["mandatory"][:2], eval(column["mandatory"][2:])
            if scope == "MG" and pipeline_type == PipelineType.MODEL_GENERATION:
                mandatory = mandatory_value
            elif scope == "DI" and pipeline_type == PipelineType.DATA_INTEGRATION:
                mandatory = mandatory_value
            elif scope == "MD":
                mandatory = mandatory_value
            else:
                mandatory = None

        else:
            mandatory = None

        if column["combination_part"] == column["combination_part"]:
            sequence_part, required_bool = column["combination_part"].split(";")
            required_bool = True if required_bool == "required" else False
            combination_part = (int(sequence_part), required_bool)
        else:
            combination_part = None

        if column["entry_type"] == column["entry_type"]:
            entry_type = EntryType.from_string(column["entry_type"])
        else:
            entry_type = None

        if column["filter operation"] == column["filter operation"]:
            filter_ = Filter(settings={frozenset({column["external"]}):
                                                     (FilterOptions.from_string(column["filter operation"]),
                                                      eval(column["filter by"]))})
        else:
            filter_ = None

        special_handling = []
        if column["special_handling"] == column["special_handling"]:
            special_handling = eval(column["special_handling"])

        data_entry_mapping = DataEntryMapping(external_name=_get_entry(column["external"]),
                                              identification=_get_entry(column["identification"]),
                                              reference_identification=_get_entry(column["reference identification"]),
                                              # correlated_entry_ids=column["correlated_entry_ids"],
                                              state_model_class=sm_class_str,
                                              state_model_attribute=sm_attribute_str,
                                              handling=handling_str,
                                              mandatory=mandatory,
                                              combination_part=combination_part,
                                              # splitting_element=column["splitting_element"],  # ToDo
                                              time_filter=_get_entry(column["time filter"]),
                                              filter=filter_,
                                              entry_type=entry_type,
                                              value=_get_entry(column["value"]),
                                              special_handling=special_handling,
                                              required_for_model_generation=column["MG"],
                                              required_for_data_integration=column["DI"])

        data_entry_mappings.append(data_entry_mapping)

    adapter_class = adapter_dict[data_source_entry["Adapter"]]
    skiprows = data_source_entry["skiprows"]
    if skiprows == skiprows:
        settings = {"skiprows": int(skiprows)}
    else:
        settings = {}
    adapter = adapter_class(external_source_path=external_source_path,
                            data_entry_mappings=data_entry_mappings, settings=settings)
    standardization_class = standardization_dict["EventLogStandardization"]
    process_mining_class = process_mining_dict["Process Mining"]
    preprocessing_class = preprocessings_dict["Preprocessing"]

    standardized_data_path = data_source_entry["standardized data path"]

    source_settings = {"name": data_source_entry["source"],
                       "name_space": data_source_entry["name space"],
                       "sequence": data_source_entry["sequence"],
                       "adapter": adapter,
                       "standardization_class": standardization_class,
                       "process_mining_class": process_mining_class,
                       "preprocessing_class": preprocessing_class,
                       "data_entry_mappings": data_entry_mappings,
                       "standardized data path": standardized_data_path,
                       "general": general}

    return source_settings

def _pipeline_settings(pipeline_building_blocks_path, general_df):
    (objects_collection_dict, model_creation_dict, model_updates_dict, process_model_updates_dict,
     state_model_smoothing_dict) = (
        get_general_object_dicts(pipeline_building_blocks_path))

    objects_collection = general_df["Objects Collection"].iloc[0]
    state_model_creation = general_df["Model Creation"].iloc[0]
    state_model_update = general_df["State Model Update"].iloc[0]
    process_model_update = general_df["Process Model Update"].iloc[0]
    state_model_smoothing = general_df["State Model Smoothing"].iloc[0]

    pipeline_settings = {"Objects Collection": objects_collection_dict[objects_collection],
                         "State Model Creation": model_creation_dict[state_model_creation],
                         "State Model Update": model_updates_dict[state_model_update],
                         "Process Model Update": process_model_updates_dict[process_model_update],
                         "State Model Smoothing": state_model_smoothing_dict[state_model_smoothing]}

    return pipeline_settings


def _get_storage_settings(data_source_entry, project_path):
    if data_source_entry["source"][0] == "*":
        data_source_entry["source"] = data_source_entry["mapping"][1:]

    storage_path = os.path.join(project_path, os.path.normpath(data_source_entry['standardized data path']))

    storage_settings = {data_source_entry["source"]: storage_path}

    return storage_settings


def _get_aggregation_combinations_with_connection(column_mappings_df, aggregation_df):
    """Define the aggregation combinations with the information given from the data source model,
    created in an Excel file."""

    aggregation_combinations = list(zip(aggregation_df["first"].to_list(), aggregation_df["second"].to_list()))
    aggregation_combinations_with_connection = {}
    for (source_1, source_2) in aggregation_combinations:
        source_1_df = column_mappings_df.groupby(by="mapping").get_group(source_1)
        source_2_df = column_mappings_df.groupby(by="mapping").get_group(source_2)

        possible_references_source_1 = (
            source_1_df.loc[(source_1_df["attribute"] == "identification") &
                            (source_1_df["class"] == source_1_df["class"]), ["class", "attribute"]])
        possible_references_source_2 = (
            source_2_df.loc)[(source_2_df["attribute"] == "identification") &
                             (source_2_df["class"] == source_2_df["class"]), ["class", "attribute"]]

        aggregation_combinations_with_connection[(source_1, source_2)] = (
            list(set(possible_references_source_1["class"]).intersection(
                set(possible_references_source_2["class"]))))

    return aggregation_combinations_with_connection


def get_data_transformation_model(data_source_model_path, project_path, pipeline_type):
    """
    Define data source and transformation steps

    Returns
    -------
    a list with sources and planned transformation steps
    """
    general_df = pd.read_excel(data_source_model_path, sheet_name="general")
    adapter_allocation_df = pd.read_excel(data_source_model_path, sheet_name="data sources")
    column_mappings_df = pd.read_excel(data_source_model_path, sheet_name="columns", skiprows=1)

    pipeline_building_blocks_path = general_df.iloc[0]["Pipeline Building Blocks Path"]
    if pipeline_building_blocks_path != pipeline_building_blocks_path:
        pipeline_building_blocks_path = Path("".join(project_path.split("projects")[:-1]) +
                                             "ofact/env/model_administration/pipeline_building_blocks.yaml")
    else:
        pipeline_building_blocks_path = Path(project_path + pipeline_building_blocks_path)

    sources = [_get_source_settings(data_source_entry=data_source_entry, project_path=project_path,
                                    column_mappings_df=column_mappings_df, pipeline_type=pipeline_type,
                                    pipeline_building_blocks_path=pipeline_building_blocks_path)
               for idx, data_source_entry in adapter_allocation_df.iterrows()]

    state_model_adaption_objects = _pipeline_settings(pipeline_building_blocks_path, general_df)

    storages = {}
    for idx, data_source_entry in adapter_allocation_df.iterrows():
        storages |= _get_storage_settings(data_source_entry=data_source_entry, project_path=project_path)

    aggregation_df = pd.read_excel(data_source_model_path, sheet_name="aggregation")
    aggregation_combinations_with_connection = (
        _get_aggregation_combinations_with_connection(column_mappings_df, aggregation_df))

    return sources, state_model_adaption_objects, storages, aggregation_combinations_with_connection
