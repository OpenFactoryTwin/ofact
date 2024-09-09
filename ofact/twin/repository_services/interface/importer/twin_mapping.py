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

# Imports Part 3: Project Imports
from ofact.twin.repository_services.interface.importer.basic_file_loader import (
    Mapping, convert_str_to_list, convert_str_to_dict, convert_str_to_datetime)
from ofact.twin.state_model.entities import (EntityType, Plant, NonStationaryResource, ActiveMovingResource,
                                             PassiveMovingResource, StationaryResource, Part, Storage, WorkStation,
                                             Warehouse, ConveyorBelt, PartType)

from ofact.twin.state_model.probabilities import SingleValueDistribution, BernoulliDistribution, NormalDistribution
from ofact.twin.state_model.process_models import (SimpleNormalDistributedProcessTimeModel,
                                                   SimpleSingleValueDistributedProcessTimeModel,
                                                   SimpleBernoulliDistributedQualityModel, ResourceGroup,
                                                   ResourceModel, TransformationModel, TransitionModel)
from ofact.twin.state_model.processes import (Process, ValueAddedProcess, EntityTransformationNode,
                                              ProcessTimeController, QualityController, ResourceController,
                                              TransitionController, TransformationController)
from ofact.twin.state_model.sales import Customer, Order, Feature, FeatureCluster
from ofact.twin.state_model.time import WorkCalender, ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt
from ofact.twin.state_model.helpers.helpers import handle_numerical_value

# Module-Specific Constants

try:
    from ofact.twin.model_learning.process_models_advanced import AdvancedProcessTimeModel
except:
    AdvancedProcessTimeModel = None


io_behaviour_mapper = {
    'EntityTransformationNode.IoBehaviours.EXIST': EntityTransformationNode.IoBehaviours.EXIST,
    'EntityTransformationNode.IoBehaviours.CREATED': EntityTransformationNode.IoBehaviours.CREATED,
    'EntityTransformationNode.IoBehaviours.DESTROYED': EntityTransformationNode.IoBehaviours.DESTROYED
}


def convert_io_behaviour(io_behaviour):
    """
    Conversion from a string to enum.
    :param io_behaviour: attribute of the EntityTransformationNode
    :return: an enum of the io_behaviour
    """
    if isinstance(io_behaviour, EntityTransformationNode.IoBehaviours):
        return io_behaviour

    elif io_behaviour in io_behaviour_mapper:
        return io_behaviour_mapper[io_behaviour]

    else:
        raise TypeError(f"IO Behaviour {io_behaviour} type not valid")


transformation_type_mapper = {
    'EntityTransformationNode.TransformationTypes.MAIN_ENTITY': EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
    'EntityTransformationNode.TransformationTypes.BLANK': EntityTransformationNode.TransformationTypes.BLANK,
    'EntityTransformationNode.TransformationTypes.SUB_PART': EntityTransformationNode.TransformationTypes.SUB_PART,
    'EntityTransformationNode.TransformationTypes.INGREDIENT': EntityTransformationNode.TransformationTypes.INGREDIENT,
    'EntityTransformationNode.TransformationTypes.DISASSEMBLE':
        EntityTransformationNode.TransformationTypes.DISASSEMBLE,
    'EntityTransformationNode.TransformationTypes.SUPPORT': EntityTransformationNode.TransformationTypes.SUPPORT,
    'EntityTransformationNode.TransformationTypes.UNSUPPORT': EntityTransformationNode.TransformationTypes.UNSUPPORT
}


def convert_transformation_type(transformation_type):
    """
    Conversion from a string to enum.
    :param transformation_type: attribute of the EntityTransformationNode
    :return: an enum of the transformation_type
    """

    if isinstance(transformation_type, EntityTransformationNode.TransformationTypes):
        return transformation_type

    elif transformation_type in transformation_type_mapper:
        return transformation_type_mapper[transformation_type]

    else:
        raise TypeError(f"Transformation {transformation_type} type not valid")


def convert_str_to_distribution(prob_dist_str):
    if prob_dist_str.split('(')[0] in MappingDigitalTwin.distributions:
        distribution = prob_dist_str.split('(')[0]
        input_param_s = eval(prob_dist_str.split('(')[-1][:-1])
        if isinstance(input_param_s, tuple):

            prob_dist = MappingDigitalTwin.distributions[distribution](*input_param_s)
        else:
            prob_dist = MappingDigitalTwin.distributions[distribution](input_param_s)

    else:
        raise Exception

    return prob_dist


# #### mappings  #######################################################################################################


class MappingDigitalTwin(Mapping):
    """
    Used to map Excel element to python classes.
    """
    distributions = {
        'SingleValueDistribution': SingleValueDistribution,
        'BernoulliDistribution': BernoulliDistribution,
        'NormalDistribution': NormalDistribution
    }


class MappingEntityType(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'EntityType': EntityType,
        'PartType': PartType
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'super_entity_type': None,
    }


class MappingPlant(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'Plant': Plant,

    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'corners': convert_str_to_list,
        'work_calendar': None,
    }


class MappingStationaryResource(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'StationaryResource': StationaryResource,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
    }


class MappingStorage(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'Storage': Storage
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'allowed_entity_type': None,
        'stored_entities': convert_str_to_list
    }


class MappingWarehouse(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'Warehouse': Warehouse
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'storage_places': convert_str_to_dict
    }


class MappingWorkStation(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'WorkStation': WorkStation,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'buffer_stations': convert_str_to_dict
    }


class MappingConveyorBelt(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'ConveyorBelt': ConveyorBelt,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'allowed_entity_types': convert_str_to_list,
        'entities_on_transport': convert_str_to_list
    }


class MappingNonStationaryResource(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'NonStationaryResource': NonStationaryResource,
        'ActiveMovingResource': ActiveMovingResource,
        'PassiveMovingResource': PassiveMovingResource,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'storage_places': convert_str_to_dict
    }


class MappingAllResources(MappingDigitalTwin):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'StationaryResource': StationaryResource,
        'ActiveMovingResource': ActiveMovingResource,
        'PassiveMovingResource': PassiveMovingResource,
        'NonStationaryResource': NonStationaryResource,
        'Storage': Storage,
        'WorkStation': WorkStation,
        'ConveyorBelt': ConveyorBelt,
        'Warehouse': Warehouse
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'entity_type': None,
        'process_execution_plan': None,
        'plant': None,
        'situated_in': None,
        'allowed_entity_type': None,
        'allowed_entity_types': convert_str_to_list,
        'buffer_stations': convert_str_to_dict,
        'storage_places': convert_str_to_dict,
        'stored_entities': convert_str_to_list,
        'entities_on_transport': convert_str_to_list
    }


class MappingTime(MappingDigitalTwin):
    """
    Used to transform the Time elements from the Excel sheets to python classes.
    """

    mappings = {
        "WorkCalendar": WorkCalender,
        "ProcessExecutionPlan": ProcessExecutionPlan,
        "ProcessExecutionPlanConveyorBelt": ProcessExecutionPlanConveyorBelt
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        'start_time': convert_str_to_datetime,
        'end_time': convert_str_to_datetime,
        'work_calendar_parameters': convert_str_to_dict,
        'company_vacation': convert_str_to_datetime,
        'work_calendar': None
    }


class MappingProcess(MappingDigitalTwin):
    """
    Used to transform the Process elements from the Excel sheets to python classes.
    """

    mappings = {
        "Process": Process,
        "ValueAddedProcess": ValueAddedProcess,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        # value_added_process
        "feature": None,
        "predecessors": convert_str_to_list,
        "successors": convert_str_to_list,

        # process_group
        'group': None,

        # process
        'lead_time_controller': None,
        'quality_controller': None,
        'resource_controller': None,
        'transition_controller': None,
        'transformation_controller': None,
    }


class MappingProcessControllers(MappingDigitalTwin):
    """
    Used to transform the Process elements from the Excel sheets to python classes.
    """

    mappings = {
        "ProcessTimeController": ProcessTimeController,
        "QualityController": QualityController,
        "ResourceController": ResourceController,
        "TransitionController": TransitionController,
        "TransformationController": TransformationController,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'process_time_model': None,
        'quality_model': None,
        'resource_model': None,
        'transition_model': None,
        'transformation_model': None,
    }


class MappingProcessTimeModels(MappingDigitalTwin):
    """
    Used to transform the Process elements from the Excel sheets to python classes.
    """

    mappings = {
        "SimpleSingleValueDistributedProcessTimeModel": SimpleSingleValueDistributedProcessTimeModel,
        "SimpleNormalDistributedProcessTimeModel": SimpleNormalDistributedProcessTimeModel,
        "AdvancedProcessTimeModel": AdvancedProcessTimeModel
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'value': handle_numerical_value,
        'mue': handle_numerical_value,
        'sigma': handle_numerical_value,
    }


class MappingQualityModels(MappingDigitalTwin):
    """
    Used to transform the Process elements from the Excel sheets to python classes.
    """

    mappings = {
        "SimpleBernoulliDistributedQualityModel": SimpleBernoulliDistributedQualityModel,
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'probability': handle_numerical_value
    }


class MappingResourceModels(MappingDigitalTwin):
    """
    Used to transform the Process elements from the Excel sheets to python classes.
    """

    mappings = {
        "ResourceGroup": ResourceGroup,
        "ResourceModel": ResourceModel
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'resource_groups': convert_str_to_list,

        'resources': convert_str_to_list,
        'main_resources': convert_str_to_list,
    }


class MappingTransitionModels(MappingDigitalTwin):
    mappings = {
        "TransitionModel": TransitionModel
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'possible_origins': convert_str_to_list,
        'possible_destinations': convert_str_to_list,
    }


class MappingTransformationModels(MappingDigitalTwin):
    mappings = {
        "EntityTransformationNode": EntityTransformationNode,
        "TransformationModel": TransformationModel
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'root_nodes': convert_str_to_list,

        'io_behaviour': convert_io_behaviour,
        'transformation_type': convert_transformation_type,
        'parents': convert_str_to_list,
        'children': convert_str_to_list,
        'entity_type': None,
    }


class MappingParts(MappingDigitalTwin):
    """
    Used to transform the Sales elements from the Excel sheets to python classes.
    """
    mappings = {
        "Part": Part
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'entity_type': None,
        'situated_in': None,
        'part_of': None,
        'parts': convert_str_to_list,
        'part_removable': convert_str_to_list
    }


class MappingSales(MappingDigitalTwin):
    """
    Used to transform the Sales elements from the Excel sheets to python classes.
    """

    mappings = {
        'Feature': Feature,
        'FeatureCluster': FeatureCluster
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,

        'product_class': None,
        'feature_cluster': None,
        'selection_probability_distribution': convert_str_to_distribution,
    }


class MappingCustomer(MappingDigitalTwin):
    """
    Used to transform the Customer elements from the Excel sheets to python classes.
    """
    mappings = {
        'Customer': Customer
    }

    object_columns = {
        'external_identifications': convert_str_to_dict
    }


class MappingOrder(MappingDigitalTwin):
    """
    Used to transform the Customer elements from the Excel sheets to python classes.
    """
    mappings = {
        'Order': Order
    }

    object_columns = {
        'external_identifications': convert_str_to_dict,
        "product_class": None,
        "customer": None,
        "features_requested": convert_str_to_list,
        "order_date": convert_str_to_datetime,
        "delivery_date_planned": convert_str_to_datetime
    }
