"""
Project Specific part of the data integration ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Dict

# Imports Part 3: Project Imports
from ofact.env.data_integration.data_transformation_management import DataTransformationManagement
from ofact.env.data_integration.data_processing import DataProcessing, get_attr_value
from ofact.env.interfaces.data_integration.adapter import MSSQLAdapter, CSVAdapter, XLSXAdapter
from ofact.twin.state_model.basic_elements import DigitalTwinObject

# Module-Specific Constants

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class DataProcessingBicycleWorld(DataProcessing):

    def _get_state_model_object(self, name_space, class_name, value, currently_available_object_dicts={},
                                mapping_id=None, situated_in=None, new_possible=True,
                                domain_specific_static_refinements=None) -> (
            [Optional[Union[Dict, DigitalTwinObject]], bool]):
        object_changed = False
        # 1. level: currently_available_object_dicts
        if mapping_id in currently_available_object_dicts:
            return currently_available_object_dicts[mapping_id], object_changed

        # 2. level: real_world cache
        current_object_from_cache = self._object_cache.get_object(name_space=name_space, external_id=value,
                                                                  class_name=class_name)

        if current_object_from_cache is not None:
            return current_object_from_cache, object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:  # could be already added
            objects_already_planned = self._object_cache.get_objects_already_planned(type_=class_name)

        else:
            objects_already_planned = []

        # 3. level: digital twin
        # if class_name == "Part":  # ToDo: ensure that not always the same object is used (individualization)
        #     unique_value = deepcopy(value)
        #     value = " ".join([elem for elem in value.split(" ")[1:2]])
        unique_value = value
        old_value = value

        current_object_from_dt = (
            self.get_object_by_external_identification_dt(name_space=name_space, external_id=value,
                                                          class_name=class_name, situated_in=situated_in,
                                                          objects_already_planned=objects_already_planned))

        # if class_name == "Part":
        #     old_value = deepcopy(value)
        #     value = unique_value

        # else:
        #     old_value = value

        self._update_objects_domain_specific(current_object_from_dt)

        if domain_specific_static_refinements is not None:
            current_object_from_dt, object_changed = (
                self.refine_with_static_attributes(current_object_from_dt, domain_specific_static_refinements,
                                                   class_name))
        else:
            object_changed = False

        if current_object_from_dt is not None:
            if class_name == "Part":
                # current_object_from_dt.external_identifications = copy({name_space: [unique_value]})
                self._object_cache.cache_object_already_planned(type_=class_name, object_=current_object_from_dt)

            # object available in the digital_twin - should be updated if new information available
            self.store_batch(class_name, current_object_from_dt, name_space)

            return current_object_from_dt, object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            properties = self._get_object_properties(current_object_from_dt, objects_already_planned, class_name, value,
                                                     name_space, situated_in, old_value)

            if properties:
                if "storage_places" in properties:
                    storages_to_store = properties["storage_places"]
                    # maybe integrate objects that are not a dict in the subsequent instantiation process
                    for _, storages in storages_to_store.items():
                        for storage in storages:
                            self.change_handler.add_object(storage)
            else:
                print("Class name:", class_name)

        else:
            properties = None

        # 4. level: digital twin class kwargs
        if new_possible is True:
            current_object_dict = self._get_digital_twin_class_dict(class_name, name_space, value)
            if domain_specific_static_refinements is not None:
                current_object_dict, object_changed = (
                    self.refine_with_static_attributes(current_object_dict, domain_specific_static_refinements,
                                                       class_name))
            else:
                object_changed = False
        else:
            return None, object_changed

        if properties is not None:
            for key, value in current_object_dict.items():
                if key in properties:
                    current_object_dict[key] = properties[key]

        if class_name == "PassiveMovingResource":
            del current_object_dict["physical_body"]

        if class_name == "Part" and get_attr_value(current_object_dict, "name") == "str":
            raise Exception("The object of type part is not completely filled:", current_object_dict, value)

        # logger.debug(f"Current object: {class_name}  {value}")
        self.store_batch(class_name, current_object_dict, name_space)
        return current_object_dict, object_changed



class DataTransformationManagementBicycleWorld(DataTransformationManagement):
    adapter_dict = {"Excel": XLSXAdapter,
                    "csv": CSVAdapter,
                    "MSSQL": MSSQLAdapter}  # AdapterMapper.AdapterTypes.xlsx

    data_processing_class = DataProcessingBicycleWorld

    def __init__(self, root_path: str, project_path: str, data_source_model_path: str, state_model: StateModel,
                 change_handler, start_time=None, progress_tracker=None, artificial_simulation_need=False):
        super(DataTransformationManagementBicycleWorld, self).__init__(
            root_path=root_path, project_path=project_path, data_source_model_path=data_source_model_path,
            state_model=state_model, change_handler=change_handler, start_time=start_time,
            progress_tracker=progress_tracker, artificial_simulation_need=artificial_simulation_need)
