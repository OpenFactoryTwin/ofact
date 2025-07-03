"""Retrain the process_models"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ofact.twin.change_handler.change_handler import ChangeHandlerPhysicalWorld

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class ProcessModelUpdater(metaclass=ABCMeta):

    def update_process_models(self, digital_twin_model: StateModel, data_source_model_path: str,
                              start_datetime: Optional[datetime] = None, end_datetime: Optional[datetime] = None,
                              digital_twin_model_updated: bool = False):

        (change_handler, re_trainable_process_time_controllers, re_trainable_transition_controllers,
         re_trainable_transformation_controllers, re_trainable_quality_controllers,
         re_trainable_resource_controllers) = (
            self.get_process_controllers_to_update(digital_twin_model, data_source_model_path,
                                                   start_datetime, end_datetime, digital_twin_model_updated))

        self._update_process_models(change_handler, re_trainable_process_time_controllers,
                                    re_trainable_transition_controllers, re_trainable_transformation_controllers,
                                    re_trainable_quality_controllers, re_trainable_resource_controllers)

    def get_process_controllers_to_update(self, digital_twin_model: StateModel, data_source_model_path: str,
                                          start_datetime: Optional[datetime] = None,
                                          end_datetime: Optional[datetime] = None,
                                          digital_twin_model_updated: bool = False):
        """

        :param digital_twin_model: Model of the digital twin
        :param data_source_model_path: used to fill the digital twin with new data
        :param start_datetime: start of the consideration period (data)
        :param end_datetime: end of the consideration period (data)
        :param digital_twin_model_updated: if the digital twin was updated beforehand
        """
        digital_twin_model.update_initial()
        # initialise the change handler
        change_handler = ChangeHandlerPhysicalWorld(digital_twin=digital_twin_model, environment=None, agents=None)

        if not digital_twin_model_updated:
            # trigger data_integration
            # maybe set them in the change handler
            self._update_digital_twin(digital_twin_model, start_datetime, end_datetime, data_source_model_path)

        (re_trainable_process_time_controllers, re_trainable_transition_controllers,
         re_trainable_transformation_controllers, re_trainable_quality_controllers,
         re_trainable_resource_controllers) = change_handler.get_process_controllers_with_re_trainable_models()

        return (change_handler, re_trainable_process_time_controllers, re_trainable_transition_controllers,
                re_trainable_transformation_controllers, re_trainable_quality_controllers,
                re_trainable_resource_controllers)

    @abstractmethod
    def _update_digital_twin(self, digital_twin_model: StateModel, start_datetime: Optional[datetime],
                             end_datetime: Optional[datetime], data_source_model_path: str):
        pass

    def _update_process_models(self, change_handler, re_trainable_process_time_controllers,
                               re_trainable_transition_controllers, re_trainable_transformation_controllers,
                               re_trainable_quality_controllers, re_trainable_resource_controllers):

        # maybe do it asynchronously
        if len(re_trainable_process_time_controllers) > 0:
            print("Update process time models")
            change_handler.update_process_models(re_trainable_process_time_controllers)
        if len(re_trainable_transition_controllers) > 0:
            print("Update process transition models")
            change_handler.update_process_models(re_trainable_transition_controllers)
        if len(re_trainable_transformation_controllers) > 0:
            print("Update process transformation models")
            change_handler.update_process_models(re_trainable_transformation_controllers)
        if len(re_trainable_quality_controllers) > 0:
            print("Update process quality models")
            change_handler.update_process_models(re_trainable_quality_controllers)
        if len(re_trainable_resource_controllers) > 0:
            print("Update process resource models")
            change_handler.update_process_models(re_trainable_resource_controllers)
