"""
General classes to extract features and targets from digital twin objects to:
- Use Case 1: train a machine learning models
- Use Case 2: predict target values based on the features

classes:
    DTSampleExtraction: general abstract class that provide the basic functionalities for all digital twin models
    DTLeadTimeSampleExtraction: process lead time specific class
"""

from __future__ import annotations

import random
import statistics
from abc import abstractmethod, ABCMeta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Tuple, Literal, Dict

import numpy as np
import pandas as pd

from ofact.helpers import root_path_without_lib_name
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.state_model.helpers.helpers import get_pd_file_reader
from ofact.twin.model_learning.data_processing.helper import PreprocessingParametersFrom
from ofact.twin.model_learning.data_processing.preparation.normalization import (
    normalize_min_max_a, enforce_values_between_zero_and_one_a)
from ofact.settings import ROOT_PATH

if TYPE_CHECKING:
    from datetime import datetime

    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.processes import ProcessExecution, Process, EntityTransformationNode
    from ofact.twin.state_model.entities import (Part, Resource, StationaryResource, WorkStation, Warehouse,
                                                 ConveyorBelt, NonStationaryResource, ActiveMovingResource,
                                                 PassiveMovingResource)
    from ofact.twin.state_model.sales import Order

    resources_typing_hint = (Resource | StationaryResource | WorkStation | Warehouse | ConveyorBelt |
                             NonStationaryResource | ActiveMovingResource | PassiveMovingResource)


def memoize_get_idx_by_column_name(method):
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        """
        Checks if the method has been called with the same arguments before.
        If so, returns the cached result. Otherwise, calls the method and caches the result.
        """
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        return result

    return memoize


def get_imputation_value_series(series: pd.Series):
    """determine the imputation_value for a series (feature)"""
    # ToDo: pre_processing_parameters_from

    numeric_data_type = pd.api.types.is_numeric_dtype(series.dtype)

    if numeric_data_type:
        imputation_value = series.mean()
    else:
        imputation_value = statistics.mode(series)

    return imputation_value


def get_preprocessing_values_from_file(preprocessing_file_path: Path, sheet_name: str, model_name: str):
    pre_processing_df = pd.read_excel(preprocessing_file_path, sheet_name=sheet_name, header=[0, 1], index_col=0)
    pre_processing_values = pre_processing_df[model_name]

    return pre_processing_values


def store_preprocess_values_to_file(preprocess_values_df: pd.DataFrame, preprocessing_file_path: Path,
                                    sheet_name: str, model_name: str):
    # update only the values for the model ...

    preprocess_values_df.columns = pd.MultiIndex.from_product([[model_name], preprocess_values_df.columns])

    if not preprocessing_file_path.exists():
        preprocess_values_df_xlsx = pd.DataFrame()
    elif sheet_name not in pd.ExcelFile(preprocessing_file_path).sheet_names:
        preprocess_values_df_xlsx = pd.DataFrame()
    else:
        preprocess_values_df_xlsx = pd.read_excel(preprocessing_file_path, header=[0, 1], index_col=0,
                                                  sheet_name=sheet_name)

        if preprocess_values_df_xlsx.columns.nlevels > 1:  # entries already exist
            if model_name in list(preprocess_values_df_xlsx.columns.levels[0]):  # model already exists
                preprocess_values_df_xlsx[model_name] = np.nan  # reset values
                preprocess_values_df_xlsx[model_name] = preprocess_values_df[model_name]
                with (pd.ExcelWriter(preprocessing_file_path, engine='openpyxl', if_sheet_exists='replace', mode='a') as
                      writer):
                    preprocess_values_df_xlsx.to_excel(writer, index=True, sheet_name=sheet_name)
                return

    preprocess_values_extended_df_xlsx = pd.concat([preprocess_values_df_xlsx, preprocess_values_df], axis=1)

    if preprocessing_file_path.exists():
        # Datei existiert: Sheet ersetzen
        with pd.ExcelWriter(preprocessing_file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            preprocess_values_extended_df_xlsx.to_excel(
                writer, index=True, sheet_name=sheet_name
            )
    else:
        # Datei existiert NICHT: neue Datei anlegen
        with pd.ExcelWriter(preprocessing_file_path, engine="openpyxl", mode="w") as writer:
            preprocess_values_extended_df_xlsx.to_excel(
                writer, index=True, sheet_name=sheet_name
            )


class DTSampleExtraction(metaclass=ABCMeta):

    def __init__(self, digital_twin_model: StateModel, reference_dataset_file_path: Optional[Path] = None,
                 pre_processing_parameters_from: PreprocessingParametersFrom = PreprocessingParametersFrom.DATASET,
                 preprocessing_file_path="/projects/tutorial/models/learning/preprocessing.xlsx",
                 model_name: str = "DT",
                 sample_process_executions: Optional[List[ProcessExecution]] = None):
        """
        Extract the sample of process_execution
        :param digital_twin_model: the digital twin model is used to get information not available
        in the process_execution (for example what das a worker done in a process before ...)
        :param reference_dataset_file_path: for the purpose of imputation a reference dataset can be used.
        Based on the dataset mean values or other statistics methods can be used to impute missing values.
        :param pre_processing_parameters_from: the pre_processing_parameters_from is used to determine if values needed
        for the preprocessing are derived from file or dataset ...
        :param preprocessing_file_path: the file path is used to determine the values required for the preprocessing
        (e.g. imputation, etc.)
        :param model_name: the model_name is used to determine the values required for the preprocessing
        (e.g. imputation, etc.)
        :param sample_process_executions: the process_execution are used for the column setting if available.
        """
        self.preprocessing_file_path = Path(root_path_without_lib_name(ROOT_PATH) + preprocessing_file_path)
        self.model_name = model_name

        if not hasattr(self, "category_encoder"):
            self.category_encoder = None

        self._digital_twin_model: StateModel = digital_twin_model

        if reference_dataset_file_path is not None:
            if isinstance(reference_dataset_file_path, str):
                if "DigitalTwin" not in reference_dataset_file_path:
                    reference_dataset_file_path = Path(ROOT_PATH + reference_dataset_file_path)

            self._reference_data_file_path = reference_dataset_file_path
        else:
            self._reference_data_file_path = None

        if self._reference_data_file_path:
            file_reader = get_pd_file_reader(self._reference_data_file_path)

            reference_dataset_df = file_reader(self._reference_data_file_path)
            if "ID" in reference_dataset_df.columns:
                reference_dataset_df = reference_dataset_df.drop("ID", axis=1)
                reference_dataset_df = reference_dataset_df.drop("Date", axis=1)

            imputation_values = self.get_imputation_values(reference_dataset_df.iloc[:, :-1],
                                                           reference_dataset_df.iloc[:, -1],
                                                           pre_processing_parameters_from)
            normalization_parameters = self.get_normalization_parameters(reference_dataset_df,
                                                                         pre_processing_parameters_from)

        else:
            reference_dataset_df = None
            imputation_values = self.get_imputation_values(None,
                                                           None,
                                                           pre_processing_parameters_from)
            normalization_parameters = self.get_normalization_parameters(None,  # ToDo
                                                                         pre_processing_parameters_from)
        if sample_process_executions is None:
            sample_process_executions = digital_twin_model.get_process_executions_list()[:10]
        self.feature_column_names, self.target_column_name = (
            self._set_column_names(sample_process_executions, reference_dataset_df))

        self._reference_dataset_df: Optional[pd.DataFrame] = reference_dataset_df
        if normalization_parameters is not None:
            column_names = [column_name
                            for column_name in self.feature_column_names
                            if column_name in normalization_parameters.index]
            feature_normalization_parameters = normalization_parameters.loc[column_names]
            if self.target_column_name is not None and self.target_column_name in normalization_parameters.index:
                target_normalization_parameters = normalization_parameters.loc[self.target_column_name]
            else:
                target_normalization_parameters = None
        else:
            feature_normalization_parameters = None
            target_normalization_parameters = None

        self._imputation_values: Optional[np.array] = imputation_values
        self._feature_normalization_parameters: Optional[pd.DataFrame] = feature_normalization_parameters
        self._target_normalization_parameters: Optional[pd.DataFrame] = target_normalization_parameters

        self._imputation_values_prediction: Optional[np.array] = (
            self._get_preprocessing_parameters_prediction(imputation_values))
        self._feature_normalization_parameters_prediction: Optional[Dict] = (
            self._get_normalization_parameters_prediction(feature_normalization_parameters))

        if not hasattr(self, "_category_encoder_mapper"):
            self._category_encoder_mapper = None

    def _get_preprocessing_parameters_prediction(self, pre_processing_parameters):
        if pre_processing_parameters is not None:
            pre_processing_parameters_prediction = pre_processing_parameters.sort_index()
            if self.target_column_name in pre_processing_parameters_prediction.index:
                pre_processing_parameters_prediction = (
                    pre_processing_parameters_prediction[
                        pre_processing_parameters_prediction.index != self.target_column_name])
            pre_processing_parameters_prediction = pre_processing_parameters_prediction.values
        else:
            pre_processing_parameters_prediction = None

        return pre_processing_parameters_prediction

    def _get_normalization_parameters_prediction(self, normalization_parameters):
        if normalization_parameters is not None:
            feature_column_names = pd.Series(sorted(self.feature_column_names))
            numerical_features_positions = (
                np.where(feature_column_names.isin(self._feature_normalization_parameters.index))[0])
            normalization_parameters_prediction_dict = {"positions": numerical_features_positions}

            normalization_parameters_prediction = normalization_parameters.sort_index()
            if self.target_column_name in normalization_parameters_prediction.index:
                normalization_parameters_prediction = (
                    normalization_parameters_prediction[
                        normalization_parameters_prediction.index != self.target_column_name])

            for column_name in normalization_parameters_prediction.columns:
                normalization_parameters_prediction_dict[column_name] = (
                    normalization_parameters_prediction[column_name].values)

        else:
            normalization_parameters_prediction_dict = None

        return normalization_parameters_prediction_dict

    def get_imputation_values(self, x_data: Optional[pd.DataFrame], y_data: Optional[pd.Series],
                              pre_processing_parameters_from) -> Optional[pd.Series[Literal, object]]:
        """Determine imputation values for numerical and categorical data mixed"""

        if pre_processing_parameters_from == PreprocessingParametersFrom.FILE:
            imputation_values_df = get_preprocessing_values_from_file(self.preprocessing_file_path,
                                                                      sheet_name="Imputation",
                                                                      model_name=self.model_name)
            imputation_values_df = imputation_values_df.dropna(subset=["Feature Name"])
            imputation_values_s = pd.Series(imputation_values_df["Value"].to_numpy(),
                                            index=imputation_values_df["Feature Name"].to_numpy())

        elif pre_processing_parameters_from == PreprocessingParametersFrom.DATASET:
            if x_data is None:
                print("Dataset required to derive imputation values")
                return None

            imputation_values = {col: get_imputation_value_series(x_data[col])
                                 for col in x_data.columns}
            imputation_values[y_data.name] = get_imputation_value_series(y_data)
            imputation_values_s = pd.Series(imputation_values)

            imputation_values_df = pd.DataFrame(list(imputation_values.items()), columns=["Feature Name", "Value"])
            imputation_values_df = (
                imputation_values_df.loc[imputation_values_df['Feature Name'] == imputation_values_df['Feature Name']])
            try:
                store_preprocess_values_to_file(imputation_values_df,
                                                self.preprocessing_file_path,
                                                sheet_name="Imputation",
                                                model_name=self.model_name)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {self.preprocessing_file_path} not found. "
                                        f"Check if learning folder is available in the models folder of your project.")
        else:
            raise NotImplementedError("No pre_processing_parameters_from specified at the initialization")

        return imputation_values_s

    def fit_category_encoder(self, category_encoding_df):
        """Currently only for one hot encoding implemented"""

        if self.category_encoder is None:
            return

        if not self.category_encoder.mapping:
            self.category_encoder = None
            return

        mapping_dict = {}
        pos_to_encode: dict[int, str] = {}
        feature_names_sorted = sorted(self._imputation_values.index)
        for mapping_d in self.category_encoder.mapping:
            mapping_dict[mapping_d["col"]] = mapping_d["mapping"]
            pos_to_encode[np.where(np.array(feature_names_sorted) == mapping_d["col"])[0][0]] = mapping_d["col"]

        pos_to_encode = dict(sorted(pos_to_encode.items()))
        slicers = [(0, list(pos_to_encode.keys())[0])]
        for index in range(len(pos_to_encode.items()) - 1):
            slicers.append((list(pos_to_encode.keys())[index] + 1, list(pos_to_encode.keys())[index + 1]))
        slicers.append((list(pos_to_encode.keys())[-1] + 1, self._imputation_values_prediction.shape[0]))

        one_hot_encoder_mapper = {"positions_to_encode": pos_to_encode, "slicers": slicers}
        for encoding_mapper_dict in self.category_encoder.ordinal_encoder.category_mapping:
            feature_name = encoding_mapper_dict["col"]
            one_hot_encoder_mapper[feature_name] = {}

            for feature_value, feature_mapping_int in encoding_mapper_dict["mapping"].items():
                if feature_value != feature_value:
                    feature_value = "nan"

                sorted_s = mapping_dict[feature_name].loc[feature_mapping_int].sort_index()
                one_hot_encoder_mapper[feature_name][feature_value] = sorted_s.values

        self._category_encoder_mapper = one_hot_encoder_mapper

    def get_encoding_values(self):
        pass

    def get_normalization_parameters(self, reference_dataset_df: Optional[pd.DataFrame],
                                     pre_processing_parameters_from) -> Optional[pd.DataFrame]:
        """Determine the normalization parameters for numerical data"""

        if pre_processing_parameters_from == PreprocessingParametersFrom.FILE:
            normalization_parameters_df = get_preprocessing_values_from_file(self.preprocessing_file_path,
                                                                             sheet_name="Normalization",
                                                                             model_name=self.model_name)
            normalization_parameters_df = normalization_parameters_df.dropna(subset=["Feature Name"])
            normalization_parameters_df.index = normalization_parameters_df["Feature Name"]
            normalization_parameters_df.dropna(axis=0, inplace=True)  # ToDo: required???

        elif pre_processing_parameters_from == PreprocessingParametersFrom.DATASET:
            if reference_dataset_df is None:
                print("Dataset required to derive normalization parameters")
                return None

            # normalization_parameters_df = get_normalization_parameters(x_data) - alternative
            normalization_parameters_df: pd.DataFrame = reference_dataset_df.describe()
            normalization_parameters_df.dropna(thresh=2, axis=1, inplace=True)  # count is 0
            normalization_parameters_df = normalization_parameters_df.T[["min", "max"]]
            normalization_parameters_df["Feature Name"] = normalization_parameters_df.index

            normalization_parameters_df_to_store = normalization_parameters_df.copy()
            store_preprocess_values_to_file(normalization_parameters_df_to_store,
                                            self.preprocessing_file_path,
                                            sheet_name="Normalization",
                                            model_name=self.model_name)

        else:
            raise NotImplementedError("No pre_processing_parameters_from specified at the initialization")

        return normalization_parameters_df

    def get_sample_from_elements(self, event_type: ProcessExecutionTypes = ProcessExecutionTypes.PLAN,
                                 process: Optional[Process] = None,
                                 parts_involved: Optional[List[Tuple[Part, EntityTransformationNode]]] = None,
                                 resources_used: Optional[
                                     List[Tuple[Resource, EntityTransformationNode]]] = None,
                                 resulting_quality: Optional[float] = None,
                                 main_resource: Optional[Resource] = None,
                                 origin: Optional[Resource] = None,
                                 destination: Optional[Resource] = None,
                                 order: Optional[Order] = None,
                                 source_application: Optional[str] = None,
                                 executed_start_time: Optional[datetime] = None,
                                 executed_end_time: Optional[datetime] = None,
                                 domain_specific_attributes: Optional[dict] = None,
                                 encode: bool = True, normalize: bool = True, impute_nan_values: bool = True):

        sample_dict = (
            self._get_features(event_type=event_type, process=process,
                               origin=origin, destination=destination,
                               resulting_quality=resulting_quality,
                               order=order,
                               parts_involved=parts_involved,
                               resources_used=resources_used,
                               main_resource=main_resource,
                               executed_start_time=executed_start_time,
                               executed_end_time=executed_end_time,
                               source_application=source_application,
                               domain_specific_attributes=domain_specific_attributes))

        sample_dict = dict(sorted(sample_dict.items()))
        sample_a = np.array(list(sample_dict.values()), dtype=object)
        sample = self._preprocess_sample(sample_a, impute_nan_values, encode, normalize)
        # sample_df = sample_df.reindex(sorted(sample_df.columns), axis=1)
        # sample = np.array([sample_df.values[0]])

        return sample

    def get_sample(self, process_execution: ProcessExecution,
                   encode: bool = True, normalize: bool = True, impute_nan_values: bool = True,
                   add_target: bool = False) -> np.array:
        """
        Request each attribute of the process_execution if features are available
        The features are subsequently merged to a single np array
        Note: The standard values are set for the using in the prediction phase.
        For the training phase other parameters should be used. They should already be specified in the dataset.
        :param process_execution: the process_execution provides next to the digital_twin_model itself all information
        needed to extract the data for the sample.
        :param encode: if True, the sample is encoded
        :param normalize: if True, the sample is normalized (data needed in advance)
        :param impute_nan_values: if True, nan values are imputed
        :param add_target: if True, the target value is included in the sample
        (in the training phase it is not always desired because it can distort the results)
        """

        sample_dict = (
            self._get_features(event_type=process_execution.event_type, process=process_execution.process,
                               origin=process_execution.origin, destination=process_execution.destination,
                               resulting_quality=process_execution.resulting_quality,
                               order=process_execution.order,
                               parts_involved=process_execution.parts_involved,
                               resources_used=process_execution.resources_used,
                               main_resource=process_execution.main_resource,
                               executed_start_time=process_execution.executed_start_time,
                               executed_end_time=process_execution.executed_end_time,
                               source_application=process_execution.source_application,
                               domain_specific_attributes=process_execution.get_domain_specific_attributes()))

        if add_target:
            target_value = self._get_target_value(process_execution)
            sample_dict |= target_value

        sample_dict = dict(sorted(sample_dict.items()))
        sample_a = np.array(list(sample_dict.values()), dtype=object)
        sample = self._preprocess_sample(sample_a, impute_nan_values, encode, normalize)
        # sample_df = sample_df.reindex(sorted(sample_df.columns), axis=1)
        # sample = np.array([sample_df.values[0]])

        return sample

    def _preprocess_sample(self, sample_a, impute_nan_values, encode, normalize) -> pd.DataFrame:
        """
        Execute
            1. imputation
            2. normalization
            3. encoding
        if required within the sample extraction
        """

        if impute_nan_values:
            sample_a = self._impute_nan_values(sample_a)

        if normalize:
            sample_a = self._normalize_sample(sample_a)

        if encode:
            sample_a = self._encode_sample(sample_a)

        return sample_a

    def get_sample_meta_information(self, process_execution: ProcessExecution) -> [Literal, datetime]:
        unique_sample_id = self._get_unique_id(process_execution)
        sample_date = self._get_sample_date(process_execution)
        return unique_sample_id, sample_date

    def _get_sample_date(self, process_execution: ProcessExecution):
        if process_execution.executed_start_time is not None:
            sample_date = process_execution.executed_start_time.date()
        else:
            sample_date = np.nan
        return sample_date

    # # # # # Column Names # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _set_column_names(self, sample_process_executions: List[ProcessExecution], reference_dataset_df: pd.DataFrame):
        """Note: The target column name should be always the last element from the column names list"""

        # the target should not be stored in the column names
        target_column_name = self.get_target_column_name()

        if reference_dataset_df is not None:
            if not reference_dataset_df.empty:
                feature_column_names_of_samples = list(reference_dataset_df.columns)[:-1]

                return feature_column_names_of_samples, target_column_name

        random_process_executions = random.choices(sample_process_executions, k=min(len(sample_process_executions), 10))

        feature_column_names_of_samples = \
            [list(self._get_features(event_type=process_execution.event_type, process=process_execution.process,
                                     origin=process_execution.origin, destination=process_execution.destination,
                                     resulting_quality=process_execution.resulting_quality,
                                     order=process_execution.order,
                                     parts_involved=process_execution.parts_involved,
                                     resources_used=process_execution.resources_used,
                                     main_resource=process_execution.main_resource,
                                     executed_start_time=process_execution.executed_start_time,
                                     executed_end_time=process_execution.executed_end_time,
                                     source_application=process_execution.source_application,
                                     domain_specific_attributes=process_execution.get_domain_specific_attributes()))
             for process_execution in random_process_executions]

        # feature_column_names quality check
        unequal_column_lists = [feature_column_names_of_samples[column_list_idx]
                                for column_list_idx in range(1, len(feature_column_names_of_samples))
                                if (set(feature_column_names_of_samples[0]) !=
                                    set(feature_column_names_of_samples[column_list_idx]))]

        if unequal_column_lists:
            raise Exception(f"The column names vary between samples: {feature_column_names_of_samples}")

        if not feature_column_names_of_samples:
            raise Exception(f"The column names are empty: {feature_column_names_of_samples}")

        column_names = feature_column_names_of_samples[0]

        return column_names, target_column_name

    @abstractmethod
    def get_target_column_name(self):
        """The target column name is always the last element from the column names"""
        return "Target"

    # # # # # Feature Extraction methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @abstractmethod
    def _get_target_value(self, process_execution):
        target_value = {self.get_target_column_name(): np.nan}
        return target_value

    @abstractmethod
    def _get_unique_id(self, process_execution):
        return None

    @abstractmethod
    def _get_features(self, event_type, process: Process, origin: resources_typing_hint,
                      destination: resources_typing_hint, resulting_quality: float, order: Order,
                      parts_involved: list[tuple[Part, EntityTransformationNode]],
                      resources_used: list[tuple[resources_typing_hint, EntityTransformationNode]],
                      main_resource: resources_typing_hint,
                      executed_start_time: datetime, executed_end_time: datetime, source_application: str,
                      domain_specific_attributes: dict) -> dict[str, object]:
        return {}

    # # # # # Encoding methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _encode_sample(self, sample_a: np.array) -> np.array:

        if self.category_encoder is None:
            return sample_a

        encodings = [self._category_encoder_mapper[feature_name][sample_a[pos]]
                     if sample_a[pos] in self._category_encoder_mapper[feature_name]
                     else self._category_encoder_mapper[feature_name]["nan"]
                     for pos, feature_name in self._category_encoder_mapper["positions_to_encode"].items()]

        snippets = []
        # assuming the first and last element is not an encoding
        for index, slicer in enumerate(self._category_encoder_mapper["slicers"]):
            snippets.append(sample_a[slicer[0]:slicer[1]])
            if len(encodings) > index:
                snippets.append(encodings[index])

        sample_a_encoded = np.concatenate(snippets)

        return sample_a_encoded

    def _normalize_sample(self, sample_a: np.array):
        if self._feature_normalization_parameters_prediction is not None:
            sample_a[self._feature_normalization_parameters_prediction["positions"]] = (
                normalize_min_max_a(
                    data_to_normalize=sample_a[self._feature_normalization_parameters_prediction["positions"]],
                    normalization_parameters_dict=self._feature_normalization_parameters_prediction))

            sample_a[self._feature_normalization_parameters_prediction["positions"]] = (
                enforce_values_between_zero_and_one_a(
                    sample_a[self._feature_normalization_parameters_prediction["positions"]]))
        else:
            raise Exception("Normalization parameters are not set")

        return sample_a

    def _impute_nan_values(self, sample_a: np.array):

        if self._imputation_values_prediction is not None:
            # sample_s = sample_s.fillna(self._imputation_values)

            nan_indices = sample_a != sample_a
            sample_a[nan_indices] = self._imputation_values_prediction[nan_indices]

            # sample = {attribute_name: (value if value == value
            #                               else self._imputation_values[attribute_name])
            #              for attribute_name, value in sample.items()}

        return sample_a


class DTLeadTimeSampleExtraction(DTSampleExtraction):

    def get_target_column_name(self):
        return "Lead Time"

    def _get_target_value(self, process_execution):
        # ToDo: depend on the input quality - create checking here or before ...

        if (process_execution.executed_end_time is None or
                process_execution.executed_start_time is None):
            return {self.get_target_column_name(): np.nan}

        timedelta_ = process_execution.executed_end_time - process_execution.executed_start_time
        lead_time = timedelta_.total_seconds()
        return {self.get_target_column_name(): lead_time}

    # # # # # Feature Extraction methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @abstractmethod
    def _get_unique_id(self, process_execution):
        return None

    @abstractmethod
    def _get_features(self, event_type, process: Process, origin: resources_typing_hint,
                      destination: resources_typing_hint, resulting_quality: float, order: Order,
                      parts_involved: list[tuple[Part, EntityTransformationNode]],
                      resources_used: list[tuple[resources_typing_hint, EntityTransformationNode]],
                      main_resource: resources_typing_hint,
                      executed_start_time: datetime, executed_end_time: datetime, source_application: str,
                      domain_specific_attributes: dict) -> dict[str, object]:
        return {}
