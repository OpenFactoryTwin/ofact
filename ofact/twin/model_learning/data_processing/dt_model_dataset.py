"""
General classes to create a dataset used for model learning by calling the sample extraction.
classes:
    DTModelDataset: general abstract class that provide the basic functionalities for all digital twin models
    LeadTimeDataset: process lead time specific class
"""

from __future__ import annotations

# Imports Part 1: Standard Imports
import os
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Dict, List

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from ofact.helpers import root_path_without_lib_name
# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.state_model.helpers.helpers import get_pd_file_reader
from ofact.twin.model_learning.data_processing.dt_sample_extraction import DTSampleExtraction, \
    get_preprocessing_values_from_file, store_preprocess_values_to_file
from ofact.twin.model_learning.data_processing.helper import PreprocessingParametersFrom, DataSources
# from ofact.twin.model_learning.data_processing.exploration import explore_data_with_pyg
from ofact.twin.model_learning.data_processing.preparation.normalization import (
    normalize_min_max_df, normalize_min_max, enforce_values_between_zero_and_one_df)
from ofact.twin.model_learning.data_processing.preparation.outliers import (
    handle_numeric_outliers, get_outlier_detection_parameters)
from ofact.settings import ROOT_PATH

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.state_model.entities import (Resource, StationaryResource, WorkStation, Warehouse,
                                                 ConveyorBelt, NonStationaryResource, ActiveMovingResource,
                                                 PassiveMovingResource)

    resources_typing_hint = (
        Union[Resource, StationaryResource, WorkStation, Warehouse, ConveyorBelt, NonStationaryResource,
        ActiveMovingResource, PassiveMovingResource])


def _drop_overlapping_data_samples(reference_data, id_, id_column):
    """
    Drop the samples that are overlapping with the "new" data given the id
    Assuming that the 'reference_data' is older and therefore not up to date
    """
    reference_data = reference_data.loc[~reference_data[id_column].isin(id_)]
    return reference_data


def get_complete_file_path(file_path) -> Optional[Path]:
    """Append ROOT_PATH if necessary"""
    if file_path is None:
        return file_path

    file_path = Path(file_path)
    if not os.path.isfile(file_path):
        file_path = Path(root_path_without_lib_name(ROOT_PATH) + str(file_path))

    return file_path


# ToDo: imputation, normalization, encoding, numeric outliers ...
#   Schritt 1: Speichern:  encoding, numeric outliers
#   model dependent


class DTModelDataset(Dataset, metaclass=ABCMeta):
    """
    Use Cases:
    1. Create Training Dataset
    PreprocessingParametersFrom: DATASET
    DataSources: [DIGITAL_TWIN_MODEL]/ [DATASET]

    Note: to use the PreprocessingParametersFrom (FILE), it should be initialized/ set before
    To initialize the FILE option that should be faster, the following settings could be used:
    - PreprocessingParametersFrom: DATASET
    - DataSources: [DATASET]
    Additionally, it should be ensured that the models (Dataset, SampleExtraction) used for the following tasks
     are considered.
    2. Training/ Learning
    PreprocessingParametersFrom: FILE
    DataSources: [DATASET]

    3. Re_train
    PreprocessingParametersFrom: FILE/ DATASET (-> UPDATE the file)
    DataSources: [DATASET (, ADDITIONAL_DATASETS]) -> UPDATE the dataset

    4. Predict/ Deploy
    PreprocessingParametersFrom: FILE
    DataSources: [] -> No Dataset needed ...

    The settings of the fourth use case (Predict/Deploy) are considered as standard (kwargs).
    """

    PreprocessingParametersFrom = PreprocessingParametersFrom
    DataSources = DataSources

    def __init__(self, digital_twin_model: Optional[StateModel],
                 process_executions: Optional[list[ProcessExecution]] = None, consider_actual=False,
                 dataset_file_path: Optional[Union[str, Path]] = None,
                 additional_dataset_file_paths: Optional[List[Union[str, Path]]] = None,
                 pre_processing_parameters_from: PreprocessingParametersFrom = PreprocessingParametersFrom.FILE,
                 data_sources: List[DataSources] = [],
                 max_memory: Optional[int] = None,
                 sample_extraction_class: type = DTSampleExtraction,
                 filter_: Dict = None, ignore_first_entries: Optional[int] = None,
                 preprocessing: bool = True, impute_nan_values: bool = True, encode: bool = True,
                 normalize: bool = True, drop_numeric_outliers: bool = True):
        """
        Responsible for the data used for digital twin model learning ...

        ToDo: Why two files?
            - better naming, respectively use case description ...
        If no data in the reference data, respectively the data file itself,
        data is tried to derive from the digital twin model

        Use Cases:
            - Creating a dataset for setting up a new model (Learning Task)
            - For Retraining of a model (determining the error and use the samples for retraining; Learning Task)

        :param digital_twin_model: the digital twin model is used to get information not available
        in the process_execution (for example what das a worker done in a process before ...)
        :param process_executions: a list of process_executions.
        For each process_execution a data point could be created ...
        :param consider_actual: specify if only planned or also actual process_executions are considered ...
        :param additional_dataset_file_paths: reference dataset for the purpose of imputing missing values
        :param dataset_file_path: if given, the data is read from the file ... (not all file types are usable!)
        :param max_memory: maximal allowed entries (specified through the amount of rows in the data)
        :param preprocessing: preprocess the batch if True (imputation, encoding, normalization and
        dropping of numeric outliers)
        :param impute_nan_values: if True, impute nan values
        :param encode: if True, impute nan values
        :param normalize: if True, impute nan values
        :param drop_numeric_outliers: if True, impute nan values
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{datetime.now().time()} [{type(self).__name__:<20s}] Device: {self.device}")
        self.max_memory: Optional[int] = max_memory  # currently not in use

        self._digital_twin_model: Optional[StateModel] = digital_twin_model

        self._sample_extraction: DTSampleExtraction
        if digital_twin_model is not None:
            sample_process_executions = (
                self._get_process_executions(process_executions=process_executions,
                                             digital_twin_model=digital_twin_model, consider_actual=consider_actual))
        else:
            sample_process_executions = []

        if not hasattr(self, "_sample_extraction"):  # ToDo checking the functionality in all use cases ...
            self._sample_extraction = sample_extraction_class(digital_twin_model=digital_twin_model,
                                                              sample_process_executions=sample_process_executions)

        self.v_get_sample: np.vectorize = np.vectorize(self._sample_extraction.get_sample)
        self.v_get_sample_meta_information: np.vectorize = (
            np.vectorize(self._sample_extraction.get_sample_meta_information))

        # #### determine dataset #######################################################################################
        # Initialize data if available from data file
        # #### DATASET FROM FILE ####
        if (dataset_file_path is not None and
                DataSources.DATASET in data_sources):
            dataset_file_path = get_complete_file_path(dataset_file_path)
            feature_column_names, target_column_name, meta_info, x_data, y_data = self._set_data(dataset_file_path)
        else:
            feature_column_names, target_column_name, meta_info, x_data, y_data = [], None, None, None, None

        self.feature_column_names = feature_column_names
        self.target_column_name = target_column_name

        # #### DATASET FROM DIGITAL_TWIN_MODELS ####
        # get data from process_executions if no reference_data or "standard" data available
        if DataSources.DIGITAL_TWIN_MODEL in data_sources:
            if meta_info is not None:
                dataset_already_exists_before = True
            else:
                dataset_already_exists_before = False

            meta_info_dtm, x_data_dtm, y_data_dtm = self._create_new_data(process_executions, consider_actual)

            if dataset_already_exists_before:
                additional_dataset = pd.concat([meta_info, x_data, y_data], axis=1)
                meta_info, x_data, y_data = self._combine_data(meta_info_dtm, x_data_dtm, y_data_dtm,
                                                               additional_dataset=additional_dataset)
            else:
                meta_info, x_data, y_data = meta_info_dtm, x_data_dtm, y_data_dtm
                self.feature_column_names = self._sample_extraction.feature_column_names
                self.target_column_name = self._sample_extraction.target_column_name

        # #### DATASET FROM ADDITIONAL FILES ####
        # determine reference data file path
        if additional_dataset_file_paths is not None:
            additional_dataset_file_paths = [get_complete_file_path(additional_dataset_file_path)
                                             for additional_dataset_file_path in additional_dataset_file_paths]

        if (additional_dataset_file_paths is not None and
                DataSources.ADDITIONAL_DATASETS in data_sources):
            for additional_dataset_file_path in additional_dataset_file_paths:
                if additional_dataset_file_path == dataset_file_path:  # dataset already available ...
                    continue

                dataset_from_file = self._read_file(additional_dataset_file_path)
                meta_info, x_data, y_data = self._combine_data(meta_info, x_data, y_data,
                                                               additional_dataset=dataset_from_file)

        if x_data is not None:
            self.n_samples: int = x_data.shape[0]
        else:
            self.n_samples: int = 0
            self.meta_info = meta_info
            self.x_data = torch.tensor([])
            self.y_data = torch.tensor([])

            return

        # #### preprocess dataset ######################################################################################

        # further processing such as the normalization or the drop of numeric outliers change the data ...
        print(f"[{self.__class__.__name__:20}] {self.n_samples} are available in the dataset with numeric outliers.")

        if ignore_first_entries is not None:
            meta_info, x_data, y_data = self._ignore_first_entries(meta_info, x_data, y_data, ignore_first_entries)

        # Use Case: the data is split in more than one dataset for example because the distribution in the data
        # is easier to learn in separate models
        if filter_:
            meta_info, x_data, y_data = self._filter_data(meta_info, x_data, y_data, filter_)

        # filter entries not needed for the dataset
        meta_info, x_data, y_data = self._filter_entries(meta_info, x_data, y_data)

        if preprocessing:
            meta_info, x_data, y_data = (
                self._preprocess_samples(meta_info, x_data, y_data,
                                         pre_processing_parameters_from,
                                         impute_nan_values, encode, drop_numeric_outliers, normalize))

            self.n_samples: int = x_data.shape[0]
            if drop_numeric_outliers:
                print(f"[{self.__class__.__name__:20}] {self.n_samples} "
                      f"are available in the dataset without numeric outliers.")

            # data validation
            self.validate_data(x_data)

            meta_info = meta_info.reset_index(drop=True)
            x_data = x_data.reset_index(drop=True)
            y_data = y_data.reset_index(drop=True)

            self.meta_info = meta_info
            # size [n_samples, n_features]
            self.x_data: torch.tensor = self._get_torch_tensor_from_df(x_data)
            # size [n_samples, 1]
            self.y_data: torch.tensor = torch.tensor(y_data.values.astype(np.float64))

        else:
            self.meta_info = meta_info
            self.x_data = x_data
            self.y_data = y_data

    def _get_torch_tensor_from_df(self, df: pd.DataFrame):
        df = df.reindex(sorted(df.columns), axis=1)
        return torch.tensor(df.values.astype(np.float64))

    def _set_data(self, dataset_file_path):
        """Read the 'standard' data file and set the data if available"""

        data = self._read_file(dataset_file_path)
        if data is None:
            return [], None, None, None, None

        meta_info, x_data, y_data = self._split_batch(data)

        feature_column_names = list(data.columns)[:-1]
        target_column_name = list(data.columns)[-1]
        if feature_column_names[0] == "ID":
            feature_column_names.remove("ID")
            feature_column_names.remove("Date")

        return feature_column_names, target_column_name, meta_info, x_data, y_data

    def _create_new_data(self, process_executions, consider_actual):
        """Create new data from process_executions"""

        meta_info, x_data, y_data = self.get_batch(process_executions, consider_actual)

        n_samples = x_data.shape[0]
        if self.max_memory is not None:
            if n_samples > self.max_memory:
                x_data = x_data.iloc[:self.max_memory, :]
                y_data = y_data.iloc[:self.max_memory, :]

        return meta_info, x_data, y_data

    def _combine_data(self, meta_info, x_data, y_data, additional_dataset):
        """Combine the reference dataset and the standard dataset"""

        if meta_info is None:
            meta_info = pd.DataFrame([])
        reference_data = _drop_overlapping_data_samples(additional_dataset, meta_info,
                                                        id_column="ID")
        meta_info_reference, x_data_reference, y_data_reference = self._split_batch(reference_data)
        if meta_info.size:
            meta_info = pd.concat([meta_info_reference, meta_info], axis=0)
            meta_info = meta_info.reset_index(drop=True)
            x_data = pd.concat([x_data_reference, x_data], axis=0)
            x_data = x_data.reset_index(drop=True)
            y_data = pd.concat([y_data_reference, y_data], axis=0)
            y_data = y_data.reset_index(drop=True)

        else:
            meta_info, x_data, y_data = meta_info_reference, x_data_reference, y_data_reference

        return meta_info, x_data, y_data

    def _ignore_first_entries(self, meta_info, x_data, y_data, ignore_first_entries):
        """
        Ignore the first entries in the data
        delete the first entries because they could be misleading (box positions before are not considered etc.)
        """

        data_df = pd.concat([meta_info, x_data, y_data], axis=1)
        data_df = data_df.iloc[ignore_first_entries:]
        meta_info, x_data, y_data = self._split_batch(data_df)

        return meta_info, x_data, y_data

    def _filter_data(self, meta_info, x_data, y_data, filter_: Dict[str, list]):
        """Filter the data by the columns (filter keys) and the values (filter values)"""

        data_df = pd.concat([meta_info, x_data, y_data], axis=1)
        for column_name, accepted_entries in filter_.items():
            data_df = data_df.loc[data_df[column_name].isin(accepted_entries)]

        meta_info, x_data, y_data = self._split_batch(data_df)

        return meta_info, x_data, y_data

    def _filter_entries(self, meta_info, x_data, y_data):
        # can be overwritten
        return meta_info, x_data, y_data

    def _preprocess_samples(self, meta_info, x_data, y_data,
                            pre_processing_parameters_from,
                            impute_nan_values, encode, drop_numeric_outliers, normalize):
        """
        Preprocess the samples (batch) which includes imputation of nan values, encoding, dropping of
        numeric outliers and the normalization of remaining data
        """

        if impute_nan_values:
            x_data, y_data = self._impute_samples(x_data, y_data, pre_processing_parameters_from)

        if encode:
            x_data = self._encode_samples(x_data)

        # drop numeric outliers - before normalization to avoid numerical data centralized on one point
        if drop_numeric_outliers:
            meta_info, x_data, y_data = self.drop_numeric_outliers(meta_info, x_data, y_data,
                                                                   pre_processing_parameters_from)

        if normalize:
            # normalize the complete data batches
            x_data, y_data = self.normalize(x_data, y_data, pre_processing_parameters_from)

        return meta_info, x_data, y_data

    def drop_numeric_outliers(self, meta_info, x_data, y_data, pre_processing_parameters_from):
        """Drop numerical outliers from based on the target column"""

        data_df = pd.concat([meta_info, x_data, y_data], axis=1)
        target_column_name = list(data_df.columns)[-1]

        outlier_detection_parameters = self.get_outlier_detection_parameters(data_df, target_column_name,
                                                                             pre_processing_parameters_from)

        data_df_without_numeric_outliers = (
            handle_numeric_outliers(data_df=data_df,
                                    target_column_name=target_column_name,
                                    outlier_detection_parameters=outlier_detection_parameters))
        meta_info, x_data, y_data = self._split_batch(data_df_without_numeric_outliers)
        return meta_info, x_data, y_data

    def get_outlier_detection_parameters(self, data_df: Optional[pd.DataFrame], target_column_name,
                                         pre_processing_parameters_from):
        """Determine the outlier detection parameters"""
        preprocessing_file_path = self._sample_extraction.preprocessing_file_path
        model_name = self._sample_extraction.model_name

        if pre_processing_parameters_from == PreprocessingParametersFrom.FILE:
            outlier_detection_parameters_df = get_preprocessing_values_from_file(preprocessing_file_path,
                                                                                 sheet_name="OutlierDetection",
                                                                                 model_name=model_name)
            outlier_detection_parameters_s = outlier_detection_parameters_df[outlier_detection_parameters_df.columns[0]]
        elif pre_processing_parameters_from == PreprocessingParametersFrom.DATASET:
            if data_df is None:
                print("Dataset required to derive outlier detection parameters")
                return None

            outlier_detection_parameters_s = (
                get_outlier_detection_parameters(data_df=data_df,
                                                 target_column_name=target_column_name,
                                                 approach="iqr",
                                                 k=1.5))

            outlier_detection_parameters_df = outlier_detection_parameters_s.to_frame()
            store_preprocess_values_to_file(outlier_detection_parameters_df,
                                            preprocessing_file_path,
                                            sheet_name="OutlierDetection",
                                            model_name=model_name)

        else:
            raise NotImplementedError("No pre_processing_parameters_from specified at the initialization")

        return outlier_detection_parameters_s

    def normalize(self, x_data, y_data, pre_processing_parameters_from):
        y_data_normalized = y_data

        normalization_parameters_df = (
            self._sample_extraction.get_normalization_parameters(x_data, pre_processing_parameters_from))

        try:
            x_data_normalized = normalize_min_max_df(x_data, normalization_parameters_df)
            x_data_normalized = enforce_values_between_zero_and_one_df(x_data_normalized)

            # y_data_normalized = normalize_min_max(y_data)
        except:
            column_names_failed_normalization = []
            for column_name in x_data.columns:
                try:
                    x_data[column_name] = normalize_min_max(x_data[column_name])
                except:
                    column_names_failed_normalization.append(column_name)
            print(f"Warning: Normalization failed for columns '{column_names_failed_normalization}'.\n"
                  f"Maybe because one column contains non numeric data. "
                  f"The normalization should be done manually if needed for the learning "
                  "(for example for neural network learning)!")
            x_data_normalized = x_data
            y_data_normalized = y_data

        return x_data_normalized, y_data_normalized

    def validate_data(self, x_data):
        rows_with_na = x_data[x_data.isnull().any(axis=1)]
        if rows_with_na.shape[0] > 0:
            print(f"[{self.__class__.__name__:20}] {rows_with_na.shape[0]} rows contain NA values. \n"
                  f"Please check the data: \n {rows_with_na}")

    def get_data_as_df(self, preprocessed: bool = False, with_meta_info=False):
        """
        Return the data as a pandas dataframe that include the x and y data as well as the unique ID if requested.
        :param preprocessed: the data is already preprocessed which can also include the encoding where the number of
        columns arise
        :param with_meta_info: if True includes the unique ID
        :return: the data given in the dataset as a pandas dataframe
        """
        feature_column_names = self.feature_column_names
        target_column_name = self.target_column_name

        if hasattr(self, "_sample_extraction"):
            if preprocessed:
                if self._sample_extraction.category_encoder is not None:
                    feature_column_names = self._sample_extraction.category_encoder.get_feature_names_out().tolist()
            else:
                feature_column_names = self._sample_extraction.feature_column_names
            target_column_name = self._sample_extraction.target_column_name

        x_data_df = pd.DataFrame(self.x_data, columns=feature_column_names)  # .numpy()
        y_data_df = pd.Series(self.y_data, name=target_column_name)  # .numpy()
        if with_meta_info:
            data_batch = pd.concat([self.meta_info, x_data_df, y_data_df], axis=1)
        else:
            data_batch = pd.concat([x_data_df, y_data_df], axis=1)
        print(f"[{self.__class__.__name__:20}] {data_batch.shape} are available in the dataset ...")
        return data_batch

    def __getitem__(self, index):
        """
        Support indexing such that dataset[i] can be used to get i-th sample
        :param index: used to get access to the data associated with the index
        """
        return self.x_data[index].float().to(self.device), self.y_data[index].float().to(self.device)

        # ToDo: ...
        # features = np.array(self.x_data[index])
        # target = np.array(self.y_data[index])
        # try:
        #     features_tensor = torch.from_numpy(np.array(features)).float().to(self.device)
        # except:
        #     print(torch.from_numpy(np.array(features)))
        #     print(torch.from_numpy(np.array(features)).float())
        # target_tensor = torch.from_numpy(np.array(target)).float().to(self.device)

        # return features_tensor, target_tensor

    def __len__(self):
        """we can call len(dataset) to return the size"""
        return self.n_samples

    def _read_file(self, file_path: str):

        if isinstance(file_path, str):
            file_path = Path(file_path)

        pd_file_reader = get_pd_file_reader(file_path)
        data_df = pd_file_reader(file_path)
        return data_df

    def update(self, process_executions):
        """Store an experience in the memories"""
        pass  # ToDo: ...

    def get_batch(self, process_executions: Optional[list[ProcessExecution]] = None,
                  consider_actual=False) -> [pd.DataFrame, pd.Series]:
        """
        Access point for the batch generation.
        :param process_executions: a list of process_executions
        :param consider_actual: consider the process_executions of event_type 'ACTUAL'
        Note: the parts are already transformed
        :return: a dataframe, where each row contains a data point (the last column contains the target_value)
        """

        process_executions = self._get_process_executions(process_executions=process_executions,
                                                          digital_twin_model=self._digital_twin_model,
                                                          consider_actual=consider_actual)

        target = True
        sample_meta_information_vector, raw_batch = self._get_raw_batch(process_executions=process_executions,
                                                                        add_target=target)

        _, X, Y = self._split_batch(raw_batch, with_meta_info=False)
        meta_info = pd.DataFrame(sample_meta_information_vector, columns=["ID", "Date"])

        return meta_info, X, Y

    def _get_process_executions(self, process_executions, digital_twin_model, consider_actual=False,
                                relevant_process_names=None):
        """Return a list of process_executions"""

        if process_executions is not None:
            return process_executions

        if not consider_actual:
            event_type = ProcessExecutionTypes.PLAN
        else:
            event_type = None

        process_executions = digital_twin_model.get_process_executions_list(event_type=event_type)

        if relevant_process_names:
            process_executions = [process_execution for process_execution in process_executions
                                  if process_execution.get_process_name() in relevant_process_names]

        return process_executions

    def get_stratified_train_test_batches(self, df, frac=0.8):
        raise NotImplementedError("Not implemented yet!")
        return df.groupby(self._sample_extraction.target_column, group_keys=False).apply(lambda x: x.sample(frac=frac))

    def _get_raw_batch(self, process_executions,
                       encode: bool = False, normalize: bool = False, impute_nan_values: bool = False,
                       add_target: bool = True):

        feature_column_names = self._sample_extraction.feature_column_names

        if add_target:
            columns = feature_column_names + [self._sample_extraction.get_target_column_name()]
        else:
            columns = feature_column_names

        if len(process_executions) == 0:
            sample_meta_information_vector = np.array([[]])
            feature_batch = pd.DataFrame([],
                                         columns=columns)
            return sample_meta_information_vector, feature_batch

        samples = self.v_get_sample(process_execution=process_executions,
                                    encode=encode, normalize=normalize, impute_nan_values=impute_nan_values,
                                    add_target=add_target)
        sample_ids = self.v_get_sample_meta_information(process_execution=process_executions)
        samples_matrix = np.vstack(samples)
        sample_meta_information_vector = np.dstack(sample_ids)[0]

        # clean
        mask = ~pd.isna(sample_meta_information_vector).any(axis=1)
        sample_meta_information_vector = sample_meta_information_vector[mask]

        feature_batch = pd.DataFrame(samples_matrix,
                                     columns=sorted(columns))
        feature_batch = feature_batch.infer_objects()  # specify data_types
        target_column = feature_batch.pop(self._sample_extraction.get_target_column_name())
        feature_batch[self._sample_extraction.get_target_column_name()] = target_column

        return sample_meta_information_vector, feature_batch

    def _encode_samples(self, x_data) -> pd.DataFrame:

        if self._sample_extraction.category_encoder is None:
            return x_data

        x_data = self._sample_extraction.category_encoder.transform(x_data)

        return x_data

    def _impute_samples(self, x_data, y_data, pre_processing_parameters_from):
        imputation_values = (
            self._sample_extraction.get_imputation_values(x_data, y_data, pre_processing_parameters_from))

        for column_name in x_data.columns:
            if imputation_values[column_name] is None:
                continue
            x_data[column_name] = x_data[column_name].fillna(value=imputation_values[column_name])
        y_data = y_data.fillna(imputation_values[y_data.name])

        return x_data, y_data

    def get_train_and_test_set(self, split_percentage=.8):

        train_size = int(split_percentage * len(self))
        test_size = len(self) - train_size

        train_dataset, test_dataset = random_split(self,
                                                   lengths=[train_size, test_size])

        train_dataset.n_samples = len(train_dataset)
        test_dataset.n_samples = len(test_dataset)

        return train_dataset, test_dataset

    def _split_batch(self, batch, with_meta_info=True):
        if batch.empty:
            raise Exception("Batch is empty!")

        if with_meta_info:
            meta_info, X, Y = batch.iloc[:, :2], batch.iloc[:, 2:-1], batch.iloc[:, -1]
        else:
            X, Y = batch.iloc[:, :-1], batch.iloc[:, -1]
            meta_info = None
        return meta_info, X, Y

    @abstractmethod
    def get_learning_parameters_update(self):
        """Used to update the learning parameters (example given: worker that are new are added, etc.)"""
        pass

    def get_sample_extraction(self):
        if hasattr(self, "_sample_extraction"):
            return self._sample_extraction
        else:
            return None

    # def explore_with_pyg(self):  # ToDo: outcommented to avoid to much imports
    #
    #     walker = explore_data_with_pyg(df=self.get_data_as_df())
    #     return walker
