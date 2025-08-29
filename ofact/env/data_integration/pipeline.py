"""
Auto Model Generation Pipeline used to execute the modules in the right sequence.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ofact.env.environment import Environment
from ofact.env.auto_model_generation.generation.sm_object_generation import StateModelObjectGeneration
from ofact.env.auto_model_generation.generation.state_model_creation import StateModelCreation
from ofact.env.data_integration.generation_and_integration.integration.sm_object_provision import (
    StateModelObjectProvision)
from ofact.env.data_integration.generation_and_integration.integration.sm_update import StateModelUpdate
from ofact.env.data_integration.admin_pipeline.mapper_initialization import get_data_transformation_model
from ofact.env.data_integration.admin_pipeline.pipeline_settings import PipelineType
from ofact.env.data_integration.cache import ObjectCacheAutoModelling, ObjectCacheDataIntegration
from ofact.env.data_integration.preprocessing.adapter.adapter import DataAdapter
from ofact.env.data_integration.generation_and_integration.main import StateModelGenerationOrDataIntegration
from ofact.env.data_integration.preprocessing.main import Preprocessing
from ofact.env.data_integration.refinement_and_update.main import StateModelRefinementAndUpdate
from ofact.env.data_integration.refinement_and_update.refinement import RefinementOperator, RefinementOperatorOptions
from ofact.env.data_integration.validation.main import StateModelValidation
from ofact.env.data_integration.generation_and_integration.object_collection import ObjectsCollection
from ofact.env.data_integration.preprocessing.main import SourceDataProcessing

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class PipelineModules:

    def __init__(self, preprocessing_module, generation_or_integration_module, refinement_and_update_module,
                 validation_module):
        self.preprocessing_module: Preprocessing = preprocessing_module
        self.generation_or_integration_module: StateModelGenerationOrDataIntegration = generation_or_integration_module
        self.refinement_and_update_module: StateModelRefinementAndUpdate = refinement_and_update_module
        self.validation_module: StateModelValidation = validation_module

    def get_all_modules(self):
        return [self.preprocessing_module, self.generation_or_integration_module, self.refinement_and_update_module,
                self.validation_module]

    def get_generation_or_integration_module(self):
        return self.generation_or_integration_module

    def get_preprocessing_module(self):
        return self.preprocessing_module

    def get_refinement_and_update_module(self):
        return self.refinement_and_update_module

    def get_validation_module(self):
        return self.validation_module


def get_pipeline_modules(data_source_entry_mapper_path: Path, project_path: Path, pipeline_type,
                         state_model=None, change_handler=None):
    sources, general_objects, storage_paths, aggregation_combinations_with_connection = (
        get_data_transformation_model(data_source_entry_mapper_path, project_path,
                                      pipeline_type=pipeline_type))

    # general elements
    if pipeline_type == PipelineType.MODEL_GENERATION:
        object_cache = ObjectCacheAutoModelling()
        state_model_object_handling = StateModelObjectGeneration()
    elif pipeline_type == PipelineType.DATA_INTEGRATION:
        object_cache = ObjectCacheDataIntegration()
        state_model_object_handling = StateModelObjectProvision(state_model, object_cache, change_handler)
    else:
        raise ValueError("pipeline_type must be 'generation' or 'integration'")

    # generation_or_integration_module
    sources_adapters: dict[SourceDataProcessing, DataAdapter] = {}
    source_specific_modules: list[SourceDataProcessing] = []
    for source in sources:
        adapter = source["adapter"]
        state_model_data_processing = SourceDataProcessing(standardization_module=source["standardization_class"],
                                                           process_mining_module=source["process_mining_class"],
                                                           preprocessing_module=source["preprocessing_class"],
                                                           data_entry_mapper=source["data_entry_mappings"],
                                                           source_name=source["name"])

        #  sources_modules.append((source["name"],
        #                                 source["name_space"],
        #                                 adapter,
        #                                 state_model_data_processing))

        sources_adapters[state_model_data_processing] = adapter
        source_specific_modules.append(state_model_data_processing)

    source_application_name = Path(project_path).name

    objects_collections_modules: dict[set[SourceDataProcessing], ObjectsCollection] = {}
    for source_specific_module in source_specific_modules:  # ToDo: check alternatives
        objects_collection_class = general_objects["Objects Collection"]
        data_entry_mapper = source_specific_module.data_entry_mapper
        objects_collection_module = (
            objects_collection_class(state_model_object_handling=state_model_object_handling,
                                     data_entry_mapper=data_entry_mapper,
                                     source_application_name=source_application_name,
                                     source_name=source_specific_module.source_name,
                                     object_cache=object_cache,
                                     pipeline_type=pipeline_type))

        objects_collections_modules[frozenset([source_specific_module])] = objects_collection_module

    if pipeline_type == PipelineType.MODEL_GENERATION:
        state_model_generation_or_data_integration_class = general_objects["State Model Creation"]
        state_model_generation_or_data_integration_module: StateModelCreation = (
            state_model_generation_or_data_integration_class())
    elif pipeline_type == PipelineType.DATA_INTEGRATION:
        state_model_generation_or_data_integration_class = general_objects["State Model Update"]

        state_model_generation_or_data_integration_module: StateModelUpdate = (
            state_model_generation_or_data_integration_class(object_cache=object_cache, change_handler=change_handler,
                                                             state_model=state_model,
                                                             source_application_name=source_application_name))

    preprocessing_module = Preprocessing(sources_adapters=sources_adapters,
                                         source_specific_modules=source_specific_modules)

    generation_or_integration_module = (
        StateModelGenerationOrDataIntegration(objects_collections=objects_collections_modules,
                                              state_model_generation_or_data_integration=
                                              state_model_generation_or_data_integration_module))
    for _, objects_collections_module in objects_collections_modules.items():
        objects_collections_module.set_superior_module(generation_or_integration_module)
    state_model_generation_or_data_integration_module.set_superior_module(generation_or_integration_module)

    # refinement_and_update_module
    process_model_update_class = general_objects["Process Model Update"]
    process_model_update_module = process_model_update_class(state_model_object_handling=state_model_object_handling,
                                                             project_path=project_path)

    resources_refinement_operator = RefinementOperator(refinement_operator_option=RefinementOperatorOptions.TRANSITION)
    refinement_operators = [resources_refinement_operator]  # ToDo: should be chosen elsewhere
    refinement_module = general_objects["State Model Refinement"](refinement_operators)

    refinement_and_update_module = StateModelRefinementAndUpdate(process_model_update_module, refinement_module)
    # validation_module
    validation_module = None

    pipeline_modules = PipelineModules(preprocessing_module, generation_or_integration_module,
                                       refinement_and_update_module, validation_module)

    return pipeline_modules, storage_paths


class ModelGenerationAndIntegrationPipeline(metaclass=ABCMeta):

    def __init__(self, data_source_entry_mapper_path: Path, project_path: Path, progress_tracker=None,
                 state_model=None, change_handler=None):
        """
        Pipeline for the automated generation of a state model.
        ToDo:
         How to include manual steps between data sources included in the model?
         How to update the model

        Parameters
        ----------
        data_source_entry_mapper_path
        """
        self.progress_tracker = progress_tracker

        self.pipeline_type = self._get_pipeline_type()
        pipeline_modules, storage_paths = get_pipeline_modules(data_source_entry_mapper_path, project_path,
                                                               pipeline_type=self.pipeline_type,
                                                               state_model=state_model, change_handler=change_handler)
        self.pipeline_modules: PipelineModules = pipeline_modules
        self.storage_paths = storage_paths

    @abstractmethod
    def _get_pipeline_type(self):
        pass

    def track_progress(self, progress_level: float):
        if self.progress_tracker is None:
            return

        self.progress_tracker.announce(progress_level)

    def _execute_generation_or_integration(self, state_model: StateModel, start_datetime: Optional[datetime] = None,
                                           end_datetime: Optional[datetime] = None,
                                           store_interim_results: bool = False) -> StateModel:
        """
        Execute four modules to transform event logs into a state model refine them and validate the state model
        if required.
        """

        preprocessing_module = self.pipeline_modules.get_preprocessing_module()
        sources_preprocessed_event_logs = (
            preprocessing_module.get_preprocessed_event_logs(start_datetime=start_datetime, end_datetime=end_datetime,
                                                             store_interim_results=store_interim_results))

        generation_or_integration_module = self.pipeline_modules.get_generation_or_integration_module()
        if self.pipeline_type == PipelineType.MODEL_GENERATION:
            state_model = generation_or_integration_module.get_state_model(
                sources_preprocessed_event_logs=sources_preprocessed_event_logs)
        elif self.pipeline_type == PipelineType.DATA_INTEGRATION:
            state_model = generation_or_integration_module.update_state_model(
                sources_preprocessed_event_logs=sources_preprocessed_event_logs)

        refinement_and_update_module = self.pipeline_modules.get_refinement_and_update_module()
        state_model = refinement_and_update_module.get_refined_and_updated_state_model(state_model=state_model,
                                                                                       pipeline_type=self.pipeline_type)

        validation_module = self.pipeline_modules.get_validation_module()
        if validation_module is not None:
            validation_module.validate(state_model=state_model)

        return state_model


class DataIntegrationPipeline(ModelGenerationAndIntegrationPipeline, Environment):

    def _get_pipeline_type(self):
        return PipelineType.DATA_INTEGRATION

    def add_change_handler(self, change_handler):
        super().add_change_handler(change_handler)
        pipeline_modules = self.pipeline_modules.get_all_modules()
        for pipeline_module in pipeline_modules:
            if hasattr(pipeline_module, "add_change_handler"):
                pipeline_module.add_change_handler(change_handler)

    def update_state_model(self, state_model: StateModel, start_datetime: Optional[datetime] = None,
                           end_datetime: Optional[datetime] = None, store_interim_results=False) -> StateModel:
        return self._execute_generation_or_integration(state_model=state_model, start_datetime=start_datetime,
                                                       end_datetime=end_datetime,
                                                       store_interim_results=store_interim_results)


if __name__ == "__main__":
    pass
