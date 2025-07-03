"""
Auto Model Generation Pipeline used to execute the modules in the right sequence.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from pathlib import Path
from enum import Enum

from ofact.env.data_integration.sm_object_provision import StateModelObjectProvider
from ofact.env.data_integration.sm_update import StateModelUpdate
from ofact.env.model_administration.cache import ObjectCacheDataIntegration
from ofact.env.model_administration.mapper_initialization import get_data_transformation_model
from ofact.env.model_administration.pipeline_settings import PipelineType
from ofact.env.model_administration.object_collection import ObjectsCollection
from ofact.env.model_administration.standardization.source_processing import SourceDataProcessing
from ofact.env.environment import Environment

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class PipelineSetting:

    def __init__(self, sources_modules, general_modules):
        self.sources_modules: list = sources_modules
        self.general_modules: list = general_modules

    def get_source_modules(self):
        return self.sources_modules


def get_pipeline_setting(data_source_entry_mapper_path: Path, project_path: Path, pipeline_type):

    sources, general_objects, storage_paths, aggregation_combinations_with_connection = (
        get_data_transformation_model(data_source_entry_mapper_path, project_path,
                                      pipeline_type=pipeline_type))

    sources_modules = []

    for source in sources:
        adapter = source["adapter"]
        state_model_data_processing = SourceDataProcessing(standardization_module=source["standardization_class"],
                                                           process_mining_module=source["process_mining_class"],
                                                           preprocessing_module=source["preprocessing_class"],
                                                           data_entry_mapper=source["data_entry_mappings"])
        # data_entry_mapper = source["data_entry_mappings"]

        # source_application_name = source["name"]
        sources_modules.append((source["name"],
                                source["name_space"],
                                adapter,
                                state_model_data_processing))

    objects_collection_class = general_objects["Objects Collection"]
    state_model_updating_class = general_objects["State Model Update"]
    process_model_update_class = general_objects["Process Model Update"]

    general_modules = [objects_collection_class, state_model_updating_class, process_model_update_class]

    pipeline_setting = PipelineSetting(sources_modules, general_modules)

    return pipeline_setting, storage_paths


class DataIntegrationPipeline(Environment):

    def __init__(self, data_source_entry_mapper_path: Path, project_path: Path,
                 state_model, change_handler, start_time=None, progress_tracker=None):
        """
        Pipeline for the automated generation of a state model.
        ToDo:
         How to include manual steps between data sources included in the model?
         How to update the model

        Parameters
        ----------
        data_source_entry_mapper_path
        """

        super(DataIntegrationPipeline, self).__init__(change_handler=change_handler, start_time=start_time)
        self.progress_tracker = progress_tracker

        pipeline_setting, storage_paths = (
            get_pipeline_setting(data_source_entry_mapper_path, project_path,
                                 pipeline_type=PipelineType.DATA_INTEGRATION))
        self.pipeline_setting: PipelineSetting = pipeline_setting
        self.storage_paths = storage_paths

        self._cache = ObjectCacheDataIntegration()

        self._state_model_object_provider = StateModelObjectProvider(state_model=state_model, object_cache=self._cache,
                                                                     change_handler=change_handler)

    def add_change_handler(self, change_handler):
        super().add_change_handler(change_handler)
        # self._data_processing.add_change_handler(change_handler)  # ToDo
        # self._state_model_updating.add_change_handler(change_handler)

    def update_state_model(self, state_model: StateModel, start_datetime: Optional[datetime] = None,
                           end_datetime: Optional[datetime] = None, artificial_simulation_need=False) -> StateModel:
        """
        Handle one to n data sources in a pipeline to get one model as output ...

        Parameters
        ----------
        source_state_model_generation_with_event_log:
        mapping of state model generation (with settings) for each data source to an input event log
        (raw/ not standardized)
        """

        print(f"[{datetime.now()}] Data Transformation Process started from '{start_datetime}' until '{end_datetime}'.")

        # ToDo: batches also possible
        print(f"[{datetime.now()}] Phase 2: Read Data")
        data = self._get_data(start_datetime=start_datetime, end_datetime=end_datetime)

        # ToDo: create order traces would be more intuitive and decentralized
        self.track_progress(5.0)

        print(f"[{datetime.now()}] Phase 3: Create state model objects")
        self._collect_state_model_objects(data)

        self.track_progress(50.0)

        print(f"[{datetime.now()}] Phase 3: Aggregate Data")
        self._aggregate_data(data)

        self.track_progress(75.0)

        print(f"[{datetime.now()}] Phase 4: Update the digital twin state model with new event data")
        instantiated_resources = self._update_state_model(state_model)

        print(f"[{datetime.now()}] Phase 5: Update the process models")
        process_models_updates = self._get_process_model_updates(state_model)
        self._update_process_models(state_model, process_models_updates, instantiated_resources)

        self.track_progress(99.0)

        return state_model

    def _get_data(self, start_datetime: Optional[datetime], end_datetime: Optional[datetime]):
        source_modules = self.pipeline_setting.get_source_modules()

        data = {}
        for source_name, name_space, adapter, state_model_data_processing in source_modules:
            preprocessed_event_log = state_model_data_processing.get_processed_data(adapter,
                                                                                    start_datetime, end_datetime)
            data[source_name] = preprocessed_event_log

        return data

    def _collect_state_model_objects(self, data):
        source_modules = self.pipeline_setting.get_source_modules()

        for idx, (source_name, event_log_df) in enumerate(data.items()):
            state_model_data_processing = source_modules[idx][3]
            data_entry_mapper =  state_model_data_processing.data_entry_mapper
            object_creation_module: ObjectsCollection = (
                self.pipeline_setting.general_modules[0](event_log_df=event_log_df, data_entry_mapper = data_entry_mapper,
                                                         source_application_name=source_name,
                                                         generator = self._state_model_object_provider,
                                                         cache=self._cache, mode=PipelineType.DATA_INTEGRATION))
            # last: only for the model generation task
            object_creation_module.get_objects_from_event_log()

    def _aggregate_data(self, data):
        pass

    def _get_process_model_updates(self,state_model):
        # process_model_update = self.pipeline_setting.general_modules[2](state_model=state_model,
        #                                                                 generator=self._state_model_object_provider)
        # process_model_update.update()  - ToDo later
        pass

    def track_progress(self, progress_level: float):
        if self.progress_tracker is None:
            return

        self.progress_tracker.announce(progress_level)

    def _update_state_model(self, state_model):
        cache = {}
        update_module: StateModelUpdate = (
            self.pipeline_setting.general_modules[1](object_cache=self._cache, cache=cache,
                                                     change_handler=self.change_handler,
                                                     state_model=state_model, artificial_simulation_need=False,
                                                     progress_tracker=self.progress_tracker, dtm=self))

        return update_module.update_state_model()

    def _update_process_models(self, state_model, process_models_updates, instantiated_resources):
        pass


if __name__ == "__main__":
    pass
