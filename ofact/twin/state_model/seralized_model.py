# Rename the file

from __future__ import annotations

import json
import os
import dill as pickle
from typing import TYPE_CHECKING

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as e:
    pa = None
    pq = None
    print(e)

from ofact.twin.state_model.basic_elements import DigitalTwinObject
from ofact.twin.state_model.helpers.helpers import load_from_pickle
from ofact.twin.state_model.serialization import Serializable

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


def _ensure_string_keys_and_dummy_fields(data):
    updated_data = [{str(k): ({"_dummy_field": None}
                              if isinstance(v, dict) and not v else v)
                     for k, v in record.items()}
                    for record in data]

    return updated_data

def _dummy_field(value):
    if isinstance(value, dict):
        if "_dummy_field" in value:
            return True
    return False


def _delete_dummy_fields(data):
    updated_data = [{k: ({} if _dummy_field(v) else v)
                     for k, v in record.items()}
                    for record in data]

    return updated_data


def _get_list_per_object(attribute_name, attribute_value):
    if isinstance(attribute_value, dict):
        list_per_object = [attribute_value]
    elif isinstance(attribute_value, list):
        list_per_object = attribute_value
    else:
        # name <class 'NoneType'>
        # description <class 'NoneType'>
        # next_id <class 'int'>
        # ToDo: afterwards ...

        list_per_object = [{attribute_name: attribute_value}]

    return list_per_object


class SerializedStateModel:

    def __init__(self,
                 name,
                 description,
                 plant,
                 entity_types,
                 parts,
                 obstacles,
                 stationary_resources,
                 passive_moving_resources,
                 active_moving_resources,
                 entity_transformation_nodes,
                 processes,
                 process_executions,
                 order_pool,
                 customer_base,
                 features,
                 feature_clusters,
                 physical_bodies,
                 process_executions_plans,
                 resource_groups,
                 process_controllers,
                 process_models,
                 next_id):
        self.name = name
        self.description = description
        self.plant = plant
        self.entity_types = entity_types
        self.parts = parts
        self.obstacles = obstacles
        self.stationary_resources = stationary_resources
        self.passive_moving_resources = passive_moving_resources
        self.active_moving_resources = active_moving_resources
        self.entity_transformation_nodes = entity_transformation_nodes
        self.processes = processes
        self.process_executions = process_executions
        self.order_pool = order_pool
        self.customer_base = customer_base
        self.features = features
        self.feature_clusters = feature_clusters

        self.physical_bodies = physical_bodies
        self.process_executions_plans = process_executions_plans
        self.resource_groups = resource_groups
        self.process_controllers = process_controllers
        self.process_models = process_models

        self.next_id = next_id

    @staticmethod
    def from_state_model(state_model: StateModel) -> "SerializedStateModel":
        name = state_model.name
        description = state_model.description
        plant = state_model.plant
        entity_types = state_model.entity_types
        parts = state_model.get_parts()
        obstacles = state_model.obstacles
        stationary_resources = state_model.get_stationary_resources()
        passive_moving_resources = state_model.get_passive_moving_resources()
        active_moving_resources = state_model.get_active_moving_resources()
        entity_transformation_nodes = state_model.entity_transformation_nodes
        processes = state_model.get_all_processes()
        process_executions = state_model.get_process_executions_list()
        order_pool = state_model.get_orders()
        customer_base = state_model.customer_base
        features = state_model.features
        feature_clusters = state_model.get_feature_clusters()

        physical_bodies = state_model.physical_bodies
        process_executions_plans = state_model.process_executions_plans
        resource_groups = state_model.resource_groups
        process_controllers = state_model.process_controllers
        process_models: dict[object, list] = state_model.process_models

        next_id = DigitalTwinObject.next_id

        return SerializedStateModel(name=name,
                                    description=description,
                                    plant=plant,
                                    entity_types=entity_types,
                                    parts=parts,
                                    obstacles=obstacles,
                                    stationary_resources=stationary_resources,
                                    passive_moving_resources=passive_moving_resources,
                                    active_moving_resources=active_moving_resources,
                                    entity_transformation_nodes=entity_transformation_nodes,
                                    processes=processes,
                                    process_executions=process_executions,
                                    order_pool=order_pool,
                                    customer_base=customer_base,
                                    features=features,
                                    feature_clusters=feature_clusters,
                                    physical_bodies=physical_bodies,
                                    process_executions_plans=process_executions_plans,
                                    resource_groups=resource_groups,
                                    process_controllers=process_controllers,
                                    process_models=process_models,
                                    next_id=next_id)

    @staticmethod
    def load_from_pickle(pickle_path):
        return load_from_pickle(pickle_path)

    @staticmethod
    def load_from_json(json_path):
        with open(json_path) as inp:
            content = inp.read()
            state_model_json = json.loads(content)

        state_model_str = {}
        for key, value in state_model_json.items():
            state_model_str[key] = eval(json.loads(value))

    @staticmethod
    def load_from_parquet_folder(parquet_path):

        state_model_dict = {}
        for file_name in os.listdir(parquet_path):
            file_path = os.path.join(parquet_path, file_name)
            table = pq.read_table(file_path)
            state_model_dict[file_name] = table.to_pylist()  # some things should be considered ...

        return state_model_dict

    def to_dict(self,
                drop_before_serialization: dict[str, list[str]] = None,
                further_serializable: dict[str, list[str]] = None,
                reference_type: str = "identification"):
        # In this dict serializing id checking is deactivated. Otherwise, some attributes are only represented by their
        # static model id. However, in top level every attribute should be represented by its dictionary.
        object_dict = self.__dict__

        for attribute_name, attribute in object_dict.items():
            if isinstance(attribute, list):
                new_attribute = Serializable.serialize_list(attribute, deactivate_id_filter=True,
                                                            drop_before_serialization=drop_before_serialization,
                                                            further_serializable=further_serializable,
                                                            reference_type=reference_type)

            elif isinstance(attribute, str):
                new_attribute = attribute

            elif isinstance(attribute, dict):
                new_attribute = Serializable.serialize_dict(attribute, deactivate_id_filter=True,
                                                            drop_before_serialization=drop_before_serialization,
                                                            further_serializable=further_serializable,
                                                            reference_type=reference_type)
            else:
                if isinstance(attribute, Serializable):
                    new_attribute = attribute.dict_serialize(deactivate_id_filter=True,
                                                             drop_before_serialization=drop_before_serialization,
                                                             further_serializable=further_serializable,
                                                             reference_type=reference_type)
                else:
                    new_attribute = attribute
            #if "part" in attribute_name:
#
#
            #    for p in new_attribute:
            #        try:
            #            eval(str(p))
            #        except:
            #            print(p)
#
            #    file_name = "output.txt"
#
            #    # Open the file in write mode and write the string
            #    with open(file_name, 'w') as file:
            #        file.write(str(new_attribute))
#
            #    try:
            #        eval(eval(str(new_attribute)))
            #    except:
            #        print(new_attribute)

            object_dict[attribute_name] = new_attribute

        # Check if attributes are missing
        if "StateModel" in drop_before_serialization:
            ignore = drop_before_serialization["StateModel"]
        else:
            ignore = []
        Serializable.warn_if_attributes_are_missing(list(self.__dict__.keys()),
                                                    ignore=ignore,
                                                    dictionary=object_dict)

        return object_dict

    @staticmethod
    def dump_to_pickle(object_dict, target_file):
        state_model_json = SerializedStateModel.to_json(object_dict)
        with open(target_file, 'wb') as outp:
            pickle.dump(state_model_json, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def dump_to_json(object_dict, target_file):
        with open(target_file, 'w') as outp:
            json.dump(object_dict, outp, indent=4)  # indent = 4 converts None to null

    @staticmethod
    def to_json(object_dict):
        json_serialized_model = {key: json.dumps(str(value))
                                 for key, value in object_dict.items()}

        return json_serialized_model

    @staticmethod
    def dump_to_parquet_folder(object_dict, parquet_path):
        list_per_object = SerializedStateModel.to_list_per_object(object_dict)

        if ".parquet" in str(parquet_path):
            output_dir = parquet_path.split(".parquet")[0]
        else:
            output_dir = parquet_path

        os.makedirs(output_dir,
                    exist_ok=True)

        # write the content to a folder
        for attribute_name, list_content in list_per_object.items():
            table_content = pa.Table.from_pylist(_ensure_string_keys_and_dummy_fields(list_content))

            pq.write_table(table_content,
                           os.path.join(output_dir, f'{attribute_name}.parquet'))

    @staticmethod
    def to_list_per_object(object_dict):
        list_per_object_model = {key: _get_list_per_object(key, value)
                                 for key, value in object_dict.items()}

        return list_per_object_model
