"""
An extension of process models used for prediction models based on ML tools.

Classes:
    DTModelLearningExtension: The base class for learning models (process_models_advanced)
    ---
    AdvancedProcessTimeModel: Describes the time a process need

@author: Adrian Freiter
@last update: 17.11.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from ast import literal_eval
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Union, Optional, Tuple, Type, Literal

# Imports Part 2: PIP Imports
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ofact.helpers import root_path_without_lib_name
# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.model_learning.data_processing.dt_sample_extraction import (DTLeadTimeSampleExtraction,
                                                                            DTSampleExtraction)
from ofact.twin.model_learning.data_processing.helper import PreprocessingParametersFrom
from ofact.twin.state_model.process_models import ProcessTimeModel
from ofact.settings import ROOT_PATH


if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.entities import Resource, Part
    from ofact.twin.state_model.sales import Order
    from ofact.twin.state_model.processes import ProcessExecution, Process
    from ofact.twin.state_model.process_models import EntityTransformationNode
    from ofact.twin.model_learning.data_processing.dt_model_dataset import DTModelDataset

logging.debug("DigitalTwin/process_models_advanced")

class_mapping = {
    'MSELoss': torch.nn.MSELoss(),
    'Adam': torch.optim.Adam
}


def get_learning_model_parameters_dicts_from_file(learning_parameters_file_path, model_name, model_type):
    """
    Determine the learning model parameters from the file
    with consideration of the model name to find the right parameters with the file
    """

    path = Path(root_path_without_lib_name(ROOT_PATH) + learning_parameters_file_path)
    learning_df = pd.read_excel(path, sheet_name=model_type)
    learning_df.set_index("parameters", inplace=True)

    if model_name[0] == "_":
        model_name = model_name[1:]

    model_parameters = learning_df[model_name]
    model_parameter_dict = {}
    for parameter_name, parameter_value in model_parameters.items():
        if parameter_value in class_mapping:
            parameter_value = class_mapping[parameter_value]
        else:
            try:
                parameter_value = literal_eval(parameter_value)
            except:
                pass
        model_parameter_dict[parameter_name] = parameter_value

    return model_parameter_dict


class DTModelLearningExtension:

    def __init__(self,
                 dataset_class: Union[DTModelDataset, ...],
                 sample_extraction_class: Union[DTSampleExtraction, ...],
                 prediction_model_file_path: Union[Path, str],
                 prediction_model_class,
                 model_type: Optional[Literal],
                 training_data_file_path: Optional[Path] = None,
                 learning_parameters_file_path: Optional[Path] = None,
                 external_identifications: Optional[Dict[Literal, List[Literal]]] = None):
        """
        General dt model class currently used only for the process models to introduce the attributes
        needed to manage the predition model..
        :param training_data_file_path: the path to the training data
        :param learning_parameters_file_path: the path to the learning parameters
        """
        if external_identifications is None:
            external_identifications = {}

        self._training_data_file_path: Optional[Path] = training_data_file_path
        self.model_name: Literal = external_identifications['static_model'][0]
        self.model_type: Optional[Literal] = model_type
        self.learning_parameters_file_path = learning_parameters_file_path
        self._learning_parameters: Dict = (
            get_learning_model_parameters_dicts_from_file(self.learning_parameters_file_path,
                                                          model_name=self.model_name,
                                                          model_type=self.model_type))

        self._dataset: Optional[DTModelDataset] = None  # should be set later for example in set digital_twin!

        self._dataset_class: Type[DTModelDataset] = dataset_class
        self._sample_extraction_class = sample_extraction_class

        self._prediction_model_path = prediction_model_file_path
        self._prediction_model_class = prediction_model_class
        self._prediction_model: Optional[torch.nn.Module] = None
        self.set_prediction_model()

        self._digital_twin_model: StateModel

    def update_initial(self):
        if not hasattr(self, "model_type"):
            self.model_type = "ProcessTimeModel"  # ToDo: deletion needed (in inputs available)

        self._learning_parameters: Dict = (
            get_learning_model_parameters_dicts_from_file(self.learning_parameters_file_path,
                                                          model_name=self.model_name,
                                                          model_type=self.model_type))

    def update_learning_model_parameters(self):
        """Update the learning parameter settings, especially the last update date"""

        path = Path(root_path_without_lib_name(ROOT_PATH) + self.learning_parameters_file_path)
        learning_df = pd.read_excel(path)
        learning_df.set_index("parameters", inplace=True)
        learning_df[self.model_name[1:]] = pd.Series(self._learning_parameters)
        learning_df.to_excel(self.learning_parameters_file_path)

    def update_learning_parameters(self, digital_twin_model: StateModel):
        """Update the learning parameters
        e.g. the actuality of feature values"""

        if self._dataset is None:  # ToDo: naming
            return

        raw_dataset = self.get_dataset_instantiated(digital_twin_model=digital_twin_model, consider_actual=False,
                                                    preprocessing=False)
        learning_parameters_update_df = raw_dataset.get_learning_parameters_update()
        path = Path(root_path_without_lib_name(ROOT_PATH) + self.learning_parameters_file_path)
        inputs_to_update_df = pd.read_excel(path, sheet_name="Inputs to Update", index_col=0)
        model_inputs_df = inputs_to_update_df.loc[self.model_name[1:]]
        if isinstance(model_inputs_df, pd.Series):
            model_inputs_df = pd.DataFrame(model_inputs_df).T

        updated = False
        new_rows = []
        for idx, row in learning_parameters_update_df.iterrows():
            feature_name_df = model_inputs_df.loc[(model_inputs_df["Feature Name"] == row["Feature Name"]) &
                                                  (model_inputs_df["Value"] == row["Value"])]
            if feature_name_df.empty:
                new_rows.append(row)
            elif feature_name_df["Date"].iloc[0] < pd.Timestamp(row["Date"]):
                model_inputs_df.loc[(model_inputs_df["Feature Name"] == row["Feature Name"]) &
                                    (model_inputs_df["Value"] == row["Value"])] = row
                updated = True

        if new_rows:
            new_rows_df = pd.DataFrame(new_rows, [self.model_name[1:] for i in range(len(new_rows))])
            model_inputs_df = pd.concat([model_inputs_df, new_rows_df])
            model_inputs_df["Model Name"] = model_inputs_df.index
            updated = True

        if updated:
            with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
                model_inputs_df.to_excel(writer, sheet_name="Inputs to Update", index=False)
            print("Inputs to updated")

    def delete_run_time_attributes(self):
        """Use for deleting the run time attributes that should not be persisted ..."""
        if hasattr(self, "_sample_extraction"):
            delattr(self, "_sample_extraction")

    def set_prediction_model(self):
        """Initialize the prediction model based on path given
        e.g. load the model parameters (weights)"""

        print(f"[{self.model_name}] Set prediction model from path: '{self._prediction_model_path}'")
        self._prediction_model = self._prediction_model_class(model_path=self._prediction_model_path)

    def save_prediction_model(self, model_path=None, persistent_saving: bool = False):
        """Use for saving the prediction model if given ..."""
        if self._prediction_model is None:
            return

        if model_path is None:
            model_path = self._prediction_model_path

        self._prediction_model.save_model(model_path)
        if persistent_saving:
            self._prediction_model = None

    def _set_digital_twin_model(self, digital_twin_model: StateModel):
        """
        Set the digital twin model and the dataset for the training data as well as basis for normalization
        and imputation.
        :param digital_twin_model: The digital twin model to be set in the sample extractor.
        """

        if self._dataset_class is not None:
            self._dataset = self.get_dataset_instantiated(digital_twin_model=digital_twin_model)

        if self._sample_extraction_class is not None and self._dataset_class:
            self._sample_extraction = self._dataset.get_sample_extraction()
            if self._sample_extraction is not None:
                return

        if self._sample_extraction_class is not None:
            self._sample_extraction: DTSampleExtraction = (
                    self._sample_extraction_class(digital_twin_model=digital_twin_model,
                                                  pre_processing_parameters_from=PreprocessingParametersFrom.FILE))

    def get_dataset_instantiated(self, digital_twin_model: StateModel, consider_actual: bool = False,
                                 preprocessing: bool = True, impute_nan_values: bool = True, encode: bool = True,
                                 normalize: bool = True, drop_numeric_outliers: bool = True) -> DTModelDataset:
        """
        Return the dataset instantiated
        The dataset can be used for the retraining process.
        """

        dataset = self._dataset_class(digital_twin_model=digital_twin_model,
                                      dataset_file_path=self._training_data_file_path,
                                      consider_actual=consider_actual,
                                      preprocessing=preprocessing, impute_nan_values=impute_nan_values,
                                      encode=encode, normalize=normalize,
                                      drop_numeric_outliers=drop_numeric_outliers)
        return dataset

    def get_dataset(self, batch_size: Optional[int] = None) -> Optional[DTModelDataset]:
        """Return the dataset, considering the batch size if given"""

        if hasattr(self, "_dataset"):
            if self._dataset is None:
                print("Warning: dataset is None!")
                return None

        else:
            print("Warning: no dataset found!")
            return None

        if batch_size is None:
            return self._dataset
        else:
            return self._dataset[-int(batch_size):]

    def retraining_needed(self):
        """
        Determine if retraining is needed or should be performed.
        Parameters are:
         - accuracy becomes too low for 'batch_size' processes given by parameter
            re_training_trigger_max_accepted_accuracy_for_batch_size[0]
         - last update is too old (predefined interim state ...)
         - major or minor model changes that request an update are not considered here (information is given manually)
        Note: in the standard case the test_batch is the dataset generated decentrally in the process_models
        based on the current digital twin model
        (assuming these samples reveal a minimal error by the model - if not a retraining is needed)
        """

        last_update = self._learning_parameters["last_update"]

        retraining_needed = False
        retraining_reason = []

        if self._learning_parameters["max_accepted_accuracy_for_batch_size"] is None:
            raise Exception(f"re_training_trigger_max_accepted_accuracy_for_batch_size not set "
                            f"({self._learning_parameters['max_accepted_accuracy_for_batch_size']})")
        if self._learning_parameters["max_duration_to_last_update"] is None:
            raise Exception(f"re_training_trigger_max_duration_to_last_update not set "
                            f"({self._learning_parameters['max_duration_to_last_update']})")

        test_batch = self.get_dataset()  # ToDo: self._re_training_trigger_max_accepted_accuracy_for_batch_size[1]

        # check if error becomes too high
        model_accuracy, test_batch = self.get_accuracy(test_batch)
        print(f"The model accuracy is: {model_accuracy}")

        if abs(model_accuracy[0]) > self._learning_parameters['max_accepted_accuracy_for_batch_size'][0]:
            retraining_needed = True
            retraining_reason_model_accuracy = {"model_accuracy": model_accuracy,
                                                "batch_size": test_batch.n_samples}
            retraining_reason.append(retraining_reason_model_accuracy)

        # check last update is too old
        duration_since_last_update = datetime.now() - last_update
        if duration_since_last_update > timedelta(days=self._learning_parameters['max_duration_to_last_update']):
            retraining_needed = True
            retraining_reason_duration_since_last_update = {"duration_since_last_update": duration_since_last_update}
            retraining_reason.append(retraining_reason_duration_since_last_update)

        return retraining_needed, retraining_reason

    def update_device(self, device="cpu"):
        self._prediction_model.update_device(device)

    def get_accuracy(self, test_batch=None):
        """Take the last 'batch_size' samples from the storage or extern and determine the accuracy of the model"""
        test_loader = DataLoader(test_batch,
                                 shuffle=True,
                                 batch_size=128)
        self.update_device(device="cuda")
        model_accuracy = self._prediction_model.check_accuracy(loader=test_loader)
        self.update_device(device="cpu")

        return model_accuracy, test_batch

    def retrain(self, test_batch=None, batch_size=None):
        """
        :param test_batch: The dataset to use for re-training.
        :param batch_size: The batch size to use for re-training. Defaults to None.
        :return: model that is re-trained
        """
        if not hasattr(self, "_prediction_model"):
            raise Exception("Prediction model not set!")
        if self._prediction_model is None:
            raise Exception("Prediction model is None!")

        if test_batch is None:
            test_batch = self.get_dataset()
        test_loader = DataLoader(test_batch,
                                 shuffle=True,
                                 batch_size=128)

        self.get_re_training_parameters()

        self.update_device(device="cuda")
        self._prediction_model.perform_training(
            dataloader=test_loader,
            loss_fn=self._learning_parameters["loss_fn"],
            optimizer_choice=self._learning_parameters["optimizer_choice"],
            lr=self._learning_parameters["lr"],
            epochs=self._learning_parameters["epochs"],
            use_regularization=self._learning_parameters["use_regularization"],
            regularization_norm=self._learning_parameters["regularization_norm"],
            regularization_lambda=self._learning_parameters["regularization_lambda"],
            use_early_stopping=self._learning_parameters["use_early_stopping"],
            early_stopping_tolerance_epochs=self._learning_parameters["early_stopping_tolerance_epochs"],
            early_stopping_threshold=self._learning_parameters["early_stopping_threshold"])

        self.update_device(device="cpu")

    def get_re_training_parameters(self):
        pass


def get_non_negative(predicted_value):
    """ensure that the predicted_value is equal or higher than 0"""
    non_negative_predicted_value = max(predicted_value, 0)
    return non_negative_predicted_value


class AdvancedProcessTimeModel(ProcessTimeModel, DTModelLearningExtension):

    def __init__(self,
                 prediction_model_file_path: str,
                 prediction_model_class: Type[torch.nn.Module],
                 dataset_class: Type[DTModelDataset],
                 sample_extraction_class: Type[DTLeadTimeSampleExtraction],
                 is_re_trainable: bool = True,
                 training_data_file_path: Optional[Path] = None,
                 learning_parameters_file_path: Optional[Path] = None,
                 identification: Optional[int] = None,
                 external_identifications: Dict[object, List[object]] = {},
                 domain_specific_attributes: Dict[str, Union[str, int, ...]] = {}):
        """
        Calculates the process time that the process needs. Depends on a probability distribution and other factors
        (like distance, resource efficiency, etc.)
        The advanced model is based on ML learning model for the process lead time prediction.
        The prediction model is initialized in the learning extension (DTModelLearningExtension).
        """
        ProcessTimeModel.__init__(self, is_re_trainable=is_re_trainable,
                                  identification=identification,
                                  external_identifications=external_identifications,
                                  domain_specific_attributes=domain_specific_attributes)

        if dataset_class is None:
            dataset_class = DTModelDataset

        if sample_extraction_class is None:
            sample_extraction_class = DTLeadTimeSampleExtraction

        model_type = "ProcessTimeModel"

        DTModelLearningExtension.__init__(
            self=self, dataset_class=dataset_class,
            sample_extraction_class=sample_extraction_class, prediction_model_file_path=prediction_model_file_path,
            prediction_model_class=prediction_model_class, training_data_file_path=training_data_file_path,
            learning_parameters_file_path=learning_parameters_file_path, model_type=model_type,
            external_identifications=external_identifications)

    def set_digital_twin_model(self, digital_twin_model: StateModel):
        self._set_digital_twin_model(digital_twin_model=digital_twin_model)

    def get_estimated_process_lead_time(self, event_type: ProcessExecutionTypes = ProcessExecutionTypes.PLAN,
                                        process: Optional[Process] = None,
                                        parts_involved: Optional[List[Tuple[Part, EntityTransformationNode]]] = None,
                                        resources_used: Optional[
                                            List[Tuple[Resource, EntityTransformationNode]]] = None,
                                        resulting_quality: Optional[float] = None,
                                        main_resource: Optional[Resource] = None,
                                        origin: Optional[Resource] = None,
                                        destination: Optional[Resource] = None,
                                        order: Optional[Order] = None,
                                        executed_start_time: Optional[datetime] = None,
                                        executed_end_time: Optional[datetime] = None,
                                        source_application: Optional[str] = None,
                                        distance=None) -> float:
        # and also the other process_execution attributes

        sample_features = (
            self._sample_extraction.get_sample_from_elements(
                event_type=event_type, process=process, parts_involved=parts_involved, resources_used=resources_used,
                resulting_quality=resulting_quality, main_resource=main_resource, origin=origin,
                destination=destination, order=order, executed_start_time=executed_start_time,
                executed_end_time=executed_end_time, source_application=source_application))
        # provide the features to the model ...
        estimated_process_time = self._prediction_model.get_predicted_value(sample_features)
        estimated_process_time = get_non_negative(estimated_process_time)

        return estimated_process_time

    def get_expected_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the expected_process_time e.g. for the planned_process_execution.
        The calculation based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.
        :param process_execution: provides the data needed for the lead_time sample
        :returns the expected process time
        """
        sample_features = self._sample_extraction.get_sample(process_execution, add_target=False)
        # provide the features to the model ...
        expected_process_time = self._prediction_model.get_predicted_value(sample_features)
        expected_process_time = get_non_negative(expected_process_time)
        return expected_process_time

    def get_process_lead_time(self, process_execution: ProcessExecution, distance=None) -> float:
        """
        The method is used to calculate the process_time e.g. for the actual_process_execution.
        The calculation based on a factor, determined by the involved entities, and the expected value of the
        probability distribution.
        :param process_execution: provides the data needed for the lead_time sample
        :returns the process time
        """
        sample_features = self._sample_extraction.get_sample(process_execution, add_target=False)
        # provide the features to the model ...
        process_time = self._prediction_model.get_predicted_value(sample_features)
        process_time = get_non_negative(process_time)
        return process_time
