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

Instantiation of the digital_twin state model based on an Excel file and creating of the digital_twin state model.

@contact persons: Adrian Freiter
"""

# Imports Part 3: Project Imports
from ofact.twin.repository_services.deserialization.basic_file_loader import (
    Mapping, convert_str_to_list, convert_str_to_dict, convert_str_to_datetime, convert_str_to_int)
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

    Parameters
    ----------
    io_behaviour: attribute of the EntityTransformationNode

    Returns
    -------
    an enum of the io_behaviour
    """
    if isinstance(io_behaviour, EntityTransformationNode.IoBehaviours):
        return io_behaviour

    elif io_behaviour in io_behaviour_mapper:
        return io_behaviour_mapper[io_behaviour]

    else:
        raise TypeError(f"IO Behaviour {io_behaviour} type not valid")


transformation_type_mapper = {
    'EntityTransformationNode.TransformationTypes.MAIN_ENTITY':
        EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
    'EntityTransformationNode.TransformationTypes.BLANK': EntityTransformationNode.TransformationTypes.BLANK,
    'EntityTransformationNode.TransformationTypes.SUB_ENTITY': EntityTransformationNode.TransformationTypes.SUB_ENTITY,
    'EntityTransformationNode.TransformationTypes.INGREDIENT': EntityTransformationNode.TransformationTypes.INGREDIENT,
    'EntityTransformationNode.TransformationTypes.DISASSEMBLE':
        EntityTransformationNode.TransformationTypes.DISASSEMBLE,
    'EntityTransformationNode.TransformationTypes.SUPPORT': EntityTransformationNode.TransformationTypes.SUPPORT,
    'EntityTransformationNode.TransformationTypes.UNSUPPORT': EntityTransformationNode.TransformationTypes.UNSUPPORT
}


def convert_transformation_type(transformation_type):
    """
    Conversion from a string to enum.

    Parameters
    ----------
    transformation_type: attribute of the EntityTransformationNode

    Returns
    -------
    an enum of the transformation_type
    """

    if isinstance(transformation_type, EntityTransformationNode.TransformationTypes):
        return transformation_type

    elif transformation_type in transformation_type_mapper:
        return transformation_type_mapper[transformation_type]

    else:
        raise TypeError(f"Transformation {transformation_type} type not valid")

distributions = {'SingleValueDistribution': SingleValueDistribution,
                 'BernoulliDistribution': BernoulliDistribution,
                 'NormalDistribution': NormalDistribution}

def convert_str_to_distribution(prob_dist_str):
    if prob_dist_str.split('(')[0] in distributions:
        distribution = prob_dist_str.split('(')[0]
        input_param_s = eval(prob_dist_str.split('(')[-1][:-1])
        if isinstance(input_param_s, tuple):

            prob_dist = distributions[distribution](*input_param_s)
        else:
            prob_dist = distributions[distribution](input_param_s)

    else:
        raise Exception

    return prob_dist


# #### mappings  #######################################################################################################


state_model_mapper = {"EntityType": EntityType,
                      "PartType": PartType,
                      "Plant": Plant,
                      "StationaryResource": StationaryResource,
                      "Warehouse": Warehouse,
                      "Storage": Storage,
                      "WorkStation": WorkStation,
                      "ConveyorBelt": ConveyorBelt,
                      "NonStationaryResource": NonStationaryResource,
                      "ActiveMovingResource": ActiveMovingResource,
                      "PassiveMovingResource": PassiveMovingResource,
                      "WorkCalendar": WorkCalender,
                      "ProcessExecutionPlan": ProcessExecutionPlan,
                      "ProcessExecutionPlanConveyorBelt": ProcessExecutionPlanConveyorBelt,
                      "Process": Process,
                      "ValueAddedProcess": ValueAddedProcess,
                      "ProcessTimeController": ProcessTimeController,
                      "QualityController": QualityController,
                      "ResourceController": ResourceController,
                      "TransitionController": TransitionController,
                      "TransformationController": TransformationController,
                      "SimpleSingleValueDistributedProcessTimeModel": SimpleSingleValueDistributedProcessTimeModel,
                      "SimpleNormalDistributedProcessTimeModel": SimpleNormalDistributedProcessTimeModel,
                      "AdvancedProcessTimeModel": AdvancedProcessTimeModel,
                      "SimpleBernoulliDistributedQualityModel": SimpleBernoulliDistributedQualityModel,
                      "ResourceGroup": ResourceGroup,
                      "ResourceModel": ResourceModel,
                      "TransitionModel": TransitionModel,
                      "EntityTransformationNode": EntityTransformationNode,
                      "TransformationModel": TransformationModel,
                      "Part": Part,
                      'Feature': Feature,
                      'FeatureCluster': FeatureCluster,
                      'Customer': Customer,
                      'Order': Order}


format_function_mapper = {"string": None,
                          "integer": convert_str_to_int,
                          "string_list": convert_str_to_list,
                          "string_dict": convert_str_to_dict,
                          "datetime": convert_str_to_datetime,
                          "float": handle_numerical_value,
                          "tuple": None,
                          "boolean": None,
                          "io_behaviour": convert_io_behaviour,
                          "transformation_type": convert_transformation_type,
                          "distribution": convert_str_to_distribution,
                          "selection": None,
                          "dt_object": None}
