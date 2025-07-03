"""
Auto Model Generation Pipeline used to execute the modules in the right sequence.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path

import pandas as pd

from ofact.env.auto_model_generation.dummy_sm_generation import DummySMGenerator
from ofact.env.model_administration.adapter import DataAdapter
from ofact.env.model_administration.cache import ObjectCacheAutoModelling
from ofact.env.model_administration.mapper_initialization import get_data_transformation_model
from ofact.env.model_administration.pipeline_settings import PipelineType
from ofact.env.model_administration.standardization.source_processing import SourceDataProcessing

if TYPE_CHECKING:
    from ofact.env.model_administration.standardization.standardization import EventLogStandardization
    from ofact.env.model_administration.standardization.preprocessing import Preprocessing

    from ofact.env.model_administration.standardization.process_mining import ProcessMining
    from ofact.env.auto_model_generation.state_model_creation import StateModelCreation
    from ofact.env.model_administration.object_collection import ObjectsCollection
    from ofact.env.auto_model_generation.smooth_operators import SmoothOperator

    from ofact.twin.state_model.model import StateModel


class SourceStateModelGeneration:

    def __init__(self, standardization_module: Optional[EventLogStandardization],
                 process_mining_module: Optional[ProcessMining],
                 preprocessing_module: Optional[Preprocessing],
                 objects_collection_module: Optional[ObjectsCollection],
                 state_model_creation_module: Optional[StateModelCreation],
                 data_entry_mapper: list,
                 source_application_name: str = "Standard") -> None:
        """
        The state model generation object for a data source
        """

        self.source_data_processing_module = SourceDataProcessing(standardization_module=standardization_module,
                                                                  process_mining_module=process_mining_module,
                                                                  preprocessing_module=preprocessing_module,
                                                                  data_entry_mapper=data_entry_mapper)
        self.objects_collection_module: Optional[ObjectsCollection] = objects_collection_module
        self.state_model_creation_module: Optional[StateModelCreation] = state_model_creation_module
        self.data_entry_mapper: list = data_entry_mapper
        self.source_application_name = source_application_name


    def get_state_model(self, event_log_adapter: DataAdapter,
                        standardized_table_path: str = "../../data/standardized_table.csv",
                        preprocessed_table_path: str = "../../data/preprocessed_table.csv",
                        store_standardized_event_log: bool = False,
                        store_preprocessed_event_log: bool = False,
                        generator: Optional[DummySMGenerator] = None,
                        cache: Optional[ObjectCacheAutoModelling] = None):

        preprocessed_event_log = (
            self.source_data_processing_module.get_processed_data(
                event_log_adapter=event_log_adapter,
                preprocessed_table_path=preprocessed_table_path,
                store_preprocessed_event_log=store_preprocessed_event_log,
                standardized_table_path=standardized_table_path,
                store_standardized_event_log=store_standardized_event_log))

        objects_collection_module = self.objects_collection_module(event_log_df=preprocessed_event_log,
                                                               data_entry_mapper=self.data_entry_mapper,
                                                               source_application_name=self.source_application_name,
                                                               generator=generator, cache=cache,
                                                                   mode=PipelineType.MODEL_GENERATION)
        model_creation_module = self.state_model_creation_module()
        objects_from_event_log = objects_collection_module.get_objects_from_event_log()
        state_model = model_creation_module.get_state_model(*objects_from_event_log)
        return state_model

    def get_extended_state_model(self, state_model: StateModel, event_log_adapter: DataAdapter,
                                 standardized_table_path: str = "../../data/standardized_table.csv",
                                 preprocessed_table_path: str = "../../data/preprocessed_table.csv",
                                 store_standardized_event_log: bool = False,
                                 store_preprocessed_event_log: bool = False,
                                 generator: Optional[DummySMGenerator] = None,
                                 cache: Optional[ObjectCacheAutoModelling] = None):

        preprocessed_event_log = (
            self.source_data_processing_module.get_processed_data(
                event_log_adapter=event_log_adapter,
                preprocessed_table_path=preprocessed_table_path,
                store_preprocessed_event_log=store_preprocessed_event_log,
                standardized_table_path=standardized_table_path,
                store_standardized_event_log=store_standardized_event_log))


        objects_collection_module = self.objects_collection_module(event_log_df=preprocessed_event_log,
                                                                   data_entry_mapper=self.data_entry_mapper,
                                                                   source_application_name=self.source_application_name,
                                                                   generator=generator, cache=cache,
                                                                   mode=PipelineType.MODEL_GENERATION)
        model_creation_module = self.state_model_creation_module()
        objects_from_event_log = objects_collection_module.get_objects_from_event_log()
        state_model = model_creation_module.extend_state_model(state_model, *objects_from_event_log)

        return state_model


class StateModelSmoothing:

    def __init__(self, smoothing_operators: list[SmoothOperator]):
        self.smoothing_operators = smoothing_operators

    def smooth_state_model(self, state_model: StateModel):
        for smoothing_operator in self.smoothing_operators:
            state_model = smoothing_operator.smooth_state_model(state_model)


        return state_model


def _get_source_state_model_generation_objects(data_source_entry_mapper_path: Path, project_path: Path, pipeline_type) -> (
        dict[str, tuple[DataAdapter, SourceStateModelGeneration]], dict):

    sources, general_objects, storage_paths, aggregation_combinations_with_connection = (
        get_data_transformation_model(data_source_entry_mapper_path, project_path, pipeline_type))

    source_state_model_generation_objects = {}
    for source in sources:
        adapter = source["adapter"]
        model_generation = SourceStateModelGeneration(standardization_module=source["standardization_class"],
               process_mining_module=source["process_mining_class"],
               preprocessing_module=source["preprocessing_class"],
               objects_collection_module=general_objects["Objects Collection"],
               state_model_creation_module=general_objects["State Model Creation"],
               data_entry_mapper=source["data_entry_mappings"],
               source_application_name=source["name"])

        source_state_model_generation_objects[source["name"]] = (adapter, model_generation)

    return source_state_model_generation_objects, general_objects, storage_paths


class AutoModelGenerationPipeline:

    def __init__(self, data_source_entry_mapper_path: Path, project_path: Path, smoothing_operators):
        """
        Pipeline for the automated generation of a state model.
        ToDo:
         How to include manual steps between data sources included in the model?
         How to update the model

        Parameters
        ----------
        data_source_entry_mapper_path
        """

        source_state_model_generation_objects, general_objects, storage_paths = (
            _get_source_state_model_generation_objects(data_source_entry_mapper_path, project_path,
                                                       pipeline_type=PipelineType.MODEL_GENERATION))
        self.source_state_model_generation_objects: dict[str, tuple[DataAdapter, SourceStateModelGeneration]] = (
            source_state_model_generation_objects)
        self.storage_paths = storage_paths
        self._state_model_smoothing = general_objects["State Model Smoothing"](smoothing_operators)
        self._process_model_update = general_objects["Process Model Update"]

        self._cache = ObjectCacheAutoModelling()

        self._state_model_object_generator = DummySMGenerator()

    def get_state_model(self, source_state_model_generation_with_event_log: (
            dict[str, tuple[pd.DataFrame(), str]]) = None,
                        store_standardized_event_log: bool = False,
                        store_preprocessed_event_log: bool = False) -> StateModel:
        """
        Handle one to n data sources in a pipeline to get one model as output ...

        Parameters
        ----------
        source_state_model_generation_with_event_log:
        mapping of state model generation (with settings) for each data source to an input event log
        (raw/ not standardized)
        """

        if source_state_model_generation_with_event_log is None:
            source_state_model_generation_with_event_log = {}
            for source_application_name, (data_adapter, model_generation) in (
                    self.source_state_model_generation_objects.items()):
                source_state_model_generation_with_event_log[source_application_name] = \
                    (data_adapter, self.storage_paths[source_application_name])

        state_model: Optional[StateModel] = None
        for idx, (source_state_model_generation_name, (event_log_adapter, standardized_table_path)) in enumerate(
                source_state_model_generation_with_event_log.items()):
            print(f"State Model Generation for '{source_state_model_generation_name}'")

            source_state_model_generation = (
                self.source_state_model_generation_objects[source_state_model_generation_name][1])

            if idx == 0:  # no state_model exists in the first iteration
                state_model = source_state_model_generation.get_state_model(
                    event_log_adapter, standardized_table_path, store_standardized_event_log = store_standardized_event_log,
                    store_preprocessed_event_log = store_preprocessed_event_log,
                    generator=self._state_model_object_generator, cache=self._cache)

            else:  # a state_model already exists in the "later" iterations
                state_model = source_state_model_generation.get_extended_state_model(
                    state_model, event_log_adapter, store_standardized_event_log=store_standardized_event_log,
                    store_preprocessed_event_log=store_preprocessed_event_log,
                    generator=self._state_model_object_generator, cache=self._cache)

        # state model update
        self._get_process_model_updates(state_model)

        if self._state_model_smoothing is not None:
            state_model = self._state_model_smoothing.smooth_state_model(state_model)

        return state_model

    def _get_process_model_updates(self, state_model):
        process_model_update = self._process_model_update(state_model=state_model,
                                                          generator=self._state_model_object_generator)
        process_model_update.update()


if __name__ == "__main__":
    pass
    # SourceStateModelGeneration()
    # AutoModelGenerationPipeline()
