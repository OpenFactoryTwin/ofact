from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ofact.env.model_administration.adapter import DataAdapter
    from ofact.env.model_administration.standardization.standardization import EventLogStandardization
    from ofact.env.model_administration.standardization.process_mining import ProcessMining
    from ofact.env.model_administration.standardization.preprocessing import Preprocessing


class ModelAdaptionBlock:
    pass


class SourceDataProcessing(ModelAdaptionBlock):

    def __init__(self, standardization_module: Optional[EventLogStandardization],
                 process_mining_module: Optional[ProcessMining],
                 preprocessing_module: Optional[Preprocessing],
                 data_entry_mapper: list) -> None:
        """
        The state model generation object for a data source
        """

        self.standardization_module: Optional[EventLogStandardization] = standardization_module
        self.process_mining_module: Optional[ProcessMining] = process_mining_module
        self.preprocessing_module: Optional[Preprocessing] = preprocessing_module
        self.data_entry_mapper: list = data_entry_mapper

    def get_processed_data(self, event_log_adapter: DataAdapter,
                           standardized_table_path: str = "../../data/standardized_table.csv",
                           preprocessed_table_path: str = "../../data/preprocessed_table.csv",
                           store_standardized_event_log: bool = False,
                           store_preprocessed_event_log: bool = False):

        standardized_event_log, static_refinements = self._get_standardized_event_log(event_log_adapter,
                                                                  standardized_table_path=standardized_table_path,
                                                                  store_standardized_event_log=store_standardized_event_log)

        preprocessed_event_log = self._get_preprocessed_event_log(
            standardized_event_log=standardized_event_log,
            preprocessed_table_path=preprocessed_table_path,
            store_preprocessed_event_log=store_preprocessed_event_log)

        preprocessed_event_log.reset_index(inplace=True, drop=True)
        for intern_name, value in static_refinements:
            preprocessed_event_log[intern_name] = pd.Series([value] * preprocessed_event_log.shape[0],
                                                            name=intern_name)

        return preprocessed_event_log

    def _get_standardized_event_log(self, event_log_adapter: DataAdapter,
                                    standardized_table_path: str = "../../data/standardized_table.csv",
                                    store_standardized_event_log: bool = False):
        standardization_module = self.standardization_module(self.data_entry_mapper)
        standardized_event_log, static_refinements = standardization_module.standardize(event_log_adapter,
                                                                    standardized_table_path,
                                                                    store=store_standardized_event_log)

        return standardized_event_log, static_refinements

    def _get_preprocessed_event_log(self, standardized_event_log: pd.DataFrame(),
                                    preprocessed_table_path: str = "../../data/preprocessed_table.csv",
                                    store_preprocessed_event_log: bool = False):
        if self.process_mining_module is not None:
            process_mining_module = self.process_mining_module()
            standardized_event_log = process_mining_module.mine(standardized_event_log)

        preprocessing_module = self.preprocessing_module(self.data_entry_mapper)

        preprocessed_event_log = preprocessing_module.preprocess(standardized_event_log,
                                                                 preprocessed_table_path,
                                                                 store=store_preprocessed_event_log)

        return preprocessed_event_log
