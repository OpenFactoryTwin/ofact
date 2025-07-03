"""
The dummy sm generator is used for the generation of State Model Objects with standard values.
"""

import json
from copy import copy
from pathlib import Path

from ofact.env.model_administration.sm_object_handling import StateModelObjectHandling
from ofact.settings import ROOT_PATH
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes, DigitalTwinObject
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.entities import StationaryResource, EntityType, Part, Resource, Plant
from ofact.twin.state_model.probabilities import SingleValueDistribution
from ofact.twin.state_model.process_models import (SimpleBernoulliDistributedQualityModel, TransitionModel,
                                                   TransformationModel, ResourceModel,
                                                   SimpleSingleValueDistributedProcessTimeModel, ResourceGroup)
from ofact.twin.state_model.processes import (Process, QualityController, TransitionController,
                                              TransformationController, ProcessTimeController, ResourceController,
                                              ProcessExecution, ValueAddedProcess)
from ofact.twin.state_model.sales import Order, Feature, FeatureCluster, Customer
from ofact.twin.state_model.time import ProcessExecutionPlan

format_mapper = {"string": "",
                 "integer": "",
                 "string_list": [],
                 "string_dict": {},
                 "datetime": None,
                 "float": "",
                 "tuple": "",
                 "boolean": "",
                 "io_behaviour": None,
                 "transformation_type": None,
                 "distribution": None,
                 "selection": None,
                 "dt_object": None}


class DummySMGenerator(StateModelObjectHandling):

    def __init__(self, j_file_path: str = 'twin/repository_services/static_model_excel_mapping.json'):
        with open(Path(ROOT_PATH, j_file_path)) as json_data:
            d = json.load(json_data)

        self.object_mapper = {}
        self.count_dict = {}
        for elem in d["sheets"]:
            attributes_with_format = [(columns["name"], columns["format"])
                                      for columns in elem["columns"]]

            if "ProcessTimeModel" in elem["name"]:
                self.object_mapper["SimpleSingleValueDistributedProcessTimeModel"] = attributes_with_format
                self.count_dict["SimpleSingleValueDistributedProcessTimeModel"] = 0
                continue

            if "QualityModel" in elem["name"]:
                self.object_mapper["SimpleBernoulliDistributedQualityModel"] = attributes_with_format
                self.count_dict["SimpleBernoulliDistributedQualityModel"] = 0
                continue

            if "Orders" != elem["name"]:
                self.object_mapper[elem["name"]] = attributes_with_format
                self.count_dict[elem["name"]] = 0

            else:
                self.object_mapper["Order"] = attributes_with_format
                self.count_dict["Order"] = 0

    def get_plant(self, name=""):
        dummy_dict = self._get_dummy_object_dict("Plant", name=name)

        plant = Plant(**dummy_dict)
        return plant

    def get_entity_type(self, name=""):
        dummy_dict = self._get_dummy_object_dict("EntityType", name=name)

        entity_type = EntityType(**dummy_dict)
        return entity_type

    def get_stationary_resource(self, name="", entity_type=None):
        dummy_dict = self._get_dummy_object_dict("StationaryResource", name=name)
        if entity_type is None:
            entity_type = self.get_entity_type(name=name)
        dummy_dict["entity_type"] = entity_type
        dummy_dict["process_execution_plan"] = self.get_pep(name)

        stationary_resource = StationaryResource(**dummy_dict)
        return stationary_resource

    def get_pep(self, name=""):
        """ProcessExecutionPlan object"""
        dummy_dict = self._get_dummy_object_dict("ProcessExecutionPlan", name=name)

        return ProcessExecutionPlan(**dummy_dict)

    def get_part(self, name="", entity_type=None, individual_attributes=None):
        dummy_dict = self._get_dummy_object_dict("Part", name=name)
        if entity_type is None:
            entity_type = self.get_entity_type(name=name)
        dummy_dict["entity_type"] = entity_type

        if individual_attributes is not None:
            dummy_dict["domain_specific_attributes"] = individual_attributes

        part = Part(**dummy_dict)
        return part

    def get_order(self, name=""):
        dummy_dict = self._get_dummy_object_dict("Order", name=name)

        order = Order(**dummy_dict)
        return order

    def get_customer(self, name=""):
        dummy_dict = self._get_dummy_object_dict("Customer", name=name)

        customer = Customer(**dummy_dict)
        return customer

    def get_quality_model(self, name="", probability=1):
        dummy_dict = self._get_dummy_object_dict("SimpleBernoulliDistributedQualityModel", name=name)
        dummy_dict["probability"] = probability
        return SimpleBernoulliDistributedQualityModel(**dummy_dict)

    def get_quality_controller(self, quality_model, name=""):
        dummy_dict = self._get_dummy_object_dict("QualityController", name=name)
        dummy_dict["quality_model"] = quality_model

        return QualityController(**dummy_dict)

    def get_transition_model(self, name=""):
        dummy_dict = self._get_dummy_object_dict("TransitionModel", name=name)

        return TransitionModel(**dummy_dict)

    def get_transition_controller(self, transition_model, name=""):
        dummy_dict = self._get_dummy_object_dict("TransitionController",
                                                 name=name)
        dummy_dict["transition_model"] = transition_model

        return TransitionController(**dummy_dict)

    def get_transformation_model(self, name=""):
        dummy_dict = self._get_dummy_object_dict("TransformationModel", name=name)

        return TransformationModel(**dummy_dict)

    def get_transformation_controller(self, transformation_model, name=""):
        dummy_dict = self._get_dummy_object_dict("TransformationController",
                                                 name=name)
        dummy_dict["transformation_model"] = transformation_model

        return TransformationController(**dummy_dict)

    def get_resource_model(self, name=""):
        dummy_dict = self._get_dummy_object_dict("ResourceModel", name=name)

        return ResourceModel(**dummy_dict)

    def get_resource_group(self, resource_groups: list[tuple[tuple[Resource], Resource]], name="") -> list[ResourceGroup]:

        resource_groups_entity_types = []
        for resource_group_tuple in resource_groups:
            resources = [resource.entity_type
                         for resource in resource_group_tuple[0]]
            if resource_group_tuple[1] is not None:
                main_resources = [resource_group_tuple[1].entity_type]
            else:
                main_resources = []
            if (resources, main_resources) not in resource_groups_entity_types:
                resource_groups_entity_types.append((resources, main_resources))

        resource_model_resource_groups = []
        for (resources, main_resources) in resource_groups_entity_types:
            individual_dummy_object = self._get_dummy_object_dict("ResourceGroup", name=name)
            individual_name = name + str(DigitalTwinObject.get_next_id())
            individual_dummy_object["external_identifications"]["static_model"] = ["_" + individual_name + "_rg"]
            individual_dummy_object["resources"] = resources
            individual_dummy_object["main_resources"] = main_resources
            resource_group = ResourceGroup(**individual_dummy_object)
            resource_model_resource_groups.append(resource_group)

        return resource_model_resource_groups

    def get_resource_controller(self, resource_model, name=""):
        dummy_dict = self._get_dummy_object_dict("ResourceController",
                                                 name=name)
        dummy_dict["resource_model"] = resource_model

        return ResourceController(**dummy_dict)

    def get_process_time_model(self, name=""):
        dummy_dict = self._get_dummy_object_dict("SimpleSingleValueDistributedProcessTimeModel", name=name)
        dummy_dict["value"] = 0
        return SimpleSingleValueDistributedProcessTimeModel(**dummy_dict)

    def get_process_time_controller(self, process_time_model, name=""):
        dummy_dict = self._get_dummy_object_dict("ProcessTimeController",
                                                 name=name)
        dummy_dict["process_time_model"] = process_time_model

        return ProcessTimeController(**dummy_dict)

    def get_value_added_processes(self, name="", quality_controller=None, transition_controller=None,
                                  transformation_controller=None, lead_time_controller=None, resource_controller=None,
                                  group="order_et"):
        dummy_dict = self._get_dummy_object_dict("Process", name=name,
                                                 specialized_dt_object_name="ValueAddedProcess")

        dummy_dict["quality_controller"] = quality_controller
        dummy_dict["transition_controller"] = transition_controller
        dummy_dict["transformation_controller"] = transformation_controller
        dummy_dict["lead_time_controller"] = lead_time_controller
        dummy_dict["resource_controller"] = resource_controller
        dummy_dict["group"] = group

        return ValueAddedProcess(**dummy_dict)

    def get_process(self, name="", quality_controller=None, transition_controller=None, transformation_controller=None,
                    lead_time_controller=None, resource_controller=None, group="order_et"):
        dummy_dict = self._get_dummy_object_dict("Process", name=name)

        dummy_dict["quality_controller"] = quality_controller
        dummy_dict["transition_controller"] = transition_controller
        dummy_dict["transformation_controller"] = transformation_controller
        dummy_dict["lead_time_controller"] = lead_time_controller
        dummy_dict["resource_controller"] = resource_controller
        dummy_dict["group"] = group

        return Process(**dummy_dict)

    def get_feature(self, name=""):
        dummy_dict = self._get_dummy_object_dict("Sales", name=name, specialized_dt_object_name="Feature")
        dummy_dict["selection_probability_distribution"] = SingleValueDistribution(1)
        return Feature(**dummy_dict)

    def get_feature_cluster(self, name=""):
        dummy_dict = self._get_dummy_object_dict("Sales", name=name,
                                                 specialized_dt_object_name="FeatureCluster")

        return FeatureCluster(**dummy_dict)

    def get_process_execution_plan(self, execution_id, process, executed_start_time, executed_end_time,
                                     parts_involved, resources_used, main_resource, origin, destination,
                                     resulting_quality, order, source_application, individual_attributes=None):
        dummy_dict = self._get_dummy_object_dict("ProcessExecution", name=execution_id)

        dummy_dict["event_type"] = ProcessExecutionTypes.PLAN
        dummy_dict["process"] = process
        dummy_dict["executed_start_time"] = executed_start_time
        dummy_dict["executed_end_time"] = executed_end_time
        dummy_dict["parts_involved"] = parts_involved
        dummy_dict["resources_used"] = resources_used
        dummy_dict["main_resource"] = main_resource
        dummy_dict["origin"] = origin
        dummy_dict["destination"] = destination
        dummy_dict["resulting_quality"] = resulting_quality
        dummy_dict["order"] = order
        dummy_dict["source_application"] = source_application
        if individual_attributes is not None:
            dummy_dict["domain_specific_attributes"] = individual_attributes

        return ProcessExecution(**dummy_dict)

    def get_process_execution_actual(self, execution_id, process, executed_start_time, executed_end_time,
                                     parts_involved, resources_used, main_resource, origin, destination,
                                     resulting_quality, order, source_application):
        dummy_dict = self._get_dummy_object_dict("ProcessExecution", name=execution_id)

        dummy_dict["event_type"] = ProcessExecutionTypes.ACTUAL
        dummy_dict["process"] = process
        dummy_dict["executed_start_time"] = executed_start_time
        dummy_dict["executed_end_time"] = executed_end_time
        dummy_dict["parts_involved"] = parts_involved
        dummy_dict["resources_used"] = resources_used
        dummy_dict["main_resource"] = main_resource
        dummy_dict["origin"] = origin
        dummy_dict["destination"] = destination
        dummy_dict["resulting_quality"] = resulting_quality
        dummy_dict["order"] = order
        dummy_dict["source_application"] = source_application

        return ProcessExecution(**dummy_dict)

    def _get_dummy_object_dict(self, dt_object_name, name="", specialized_dt_object_name=None):

        if dt_object_name in self.object_mapper:
            dummy_object_list = self.object_mapper[dt_object_name]
            if specialized_dt_object_name is None:
                init_parameters = list(StateModel.get_init_parameter_type_hints(dt_object_name))
            else:
                init_parameters = list(StateModel.get_init_parameter_type_hints(specialized_dt_object_name))
        else:
            init_parameters = list(StateModel.get_init_parameter_type_hints(dt_object_name))
            mapper = {'event_type': "dt_object",
                      'process': "dt_object",
                      'executed_start_time': "datetime",
                      'executed_end_time': "datetime",
                      'parts_involved': "string_list",
                      'resources_used': "string_list",
                      'main_resource': "dt_object",
                      'origin': "dt_object",
                      'destination': "dt_object",
                      'resulting_quality': "integer",
                      'order': "dt_object",
                      'source_application': "string",
                      'connected_process_execution': "dt_object",
                      'identification': "integer",
                      'external_identifications': "string_dict",
                      'domain_specific_attributes': "dt_object",
                      'etn_specification': "boolean"}

            dummy_object_list = [(column_name,
                                  mapper[column_name] if column_name in mapper else None)
                                 for column_name in init_parameters]

        not_unique_names = {"Part": "_pa",
                            "TransitionModel": "_tsm",
                            "TransformationModel": "_tfm",
                            "TransitionController": "_tsc",
                            "TransformationController": "_tfc"}
        if dt_object_name not in not_unique_names:
            if specialized_dt_object_name is not None:
                if specialized_dt_object_name not in not_unique_names:
                    object_abbreviation_list = [char
                                                for char in specialized_dt_object_name
                                                if char.isupper()]
            else:
                object_abbreviation_list = [char
                                            for char in dt_object_name
                                            if char.isupper()]
            object_abbreviation = "_" + ''.join(object_abbreviation_list).lower()
            if not str(name).endswith(object_abbreviation):
                name = str(name).replace(" ", "_")

        else:
            name += not_unique_names[dt_object_name]

        dummy_object_dict = {}

        for column_name, column_format in dummy_object_list:
            if not column_name or column_name not in init_parameters:
                continue

            if column_name == "index":
                continue
                # value = dt_object_name

            elif column_name == "label":
                continue
                # index = self.count_dict[dt_object_name]
                # value = name + str(index) + abbreviations[dt_object_name]
                # self.count_dict[dt_object_name] += 1

            elif column_name == "name":
                value = copy(name) + " name"

            elif column_name == "external_identifications":
                value = {"static_model": ["_" + name]}

            else:
                if column_format is not None:
                    value = format_mapper[column_format]
                else:
                    value = None

            dummy_object_dict[column_name] = value

        dummy_object_dict = {column_name: value
                             for column_name, value in dummy_object_dict.items()
                             if value != ""}

        return dummy_object_dict


if __name__ == "__main__":
    generator = DummySMGenerator()
