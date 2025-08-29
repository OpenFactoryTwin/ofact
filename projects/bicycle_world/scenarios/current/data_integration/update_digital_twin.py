"""
Update the digital twin state model based on the ...
@last update: 31.10.2024
"""
from __future__ import annotations

# Imports Part 1: Standard Imports
from datetime import datetime
from pathlib import Path

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.env.data_integration.pipeline import DataIntegrationPipeline
from ofact.env.model_administration import get_attr_value
from ofact.env.model_administration import ObjectsCollection
from ofact.settings import ROOT_PATH
from ofact.twin.change_handler.change_handler import ChangeHandlerPhysicalWorld


class ObjectsCollectionBicycleFactory(ObjectsCollection):
    pass
#     def _handle_part(self, part_id, part_type_id=None, tags: Optional[list] = None,
#                      part_individual_attributes: dict = None):
#
#         # part_id =  "_".join(str(part_id).split(" ")[:-1])
#
#         entity_type = self._cache.get_object(class_name="EntityType", id_=part_type_id)
#         if entity_type is None:
#             print("PartTypeID: ", part_type_id)
#             if (part_type_id == "nan" or
#                     part_type_id != part_type_id):
#                 raise Exception(part_id)
#             part_et = self.generator.get_entity_type(name=part_type_id)
#             self._cache.cache_object("EntityType", part_type_id, part_et)
#         else:
#             part_et = self._cache.get_object(class_name="EntityType", id_=part_type_id)
#
#         part_object = self.generator.get_part(name=part_id, entity_type=part_et,
#                                               individual_attributes=part_individual_attributes)
#
#         if get_attr_value(part_object, "entity_type") not in self._cache.get_objects_by_class("EntityType"):
#             raise Exception("Entity Type should be specified", part_object.name)
#
#         self._cache.cache_object("Part", part_id, part_object, tags=tags)
#
#     def _get_entities_used(self, entity_class, entity_entries, execution_id=None) -> list[list[tuple]]:
#         """
#
#         execution_id
#         """
#
#         entries = entity_entries.values.flatten().tolist()
#         entry_list = list(set([str(entry) for entry in entries if str(entry) != "nan" and str(entry) != "None"]))
#         if entity_class == "StationaryResource":
#             execution_id = None
#
#         entities_lists: list[list[Entity]] = \
#             [self._cache.get_objects(class_name=entity_class,
#                                      id_="_".join(str(single_entity_entry).split(" ")[:-1])
#                                      if entity_class == "Part" else str(single_entity_entry),
#                                      tag=execution_id)
#              for single_entity_entry in entry_list]
#
#         if entity_class == "StationaryResource":
#             entities_combinations = [[entity
#                                       for combo in entities_lists
#                                       for entity in combo]]
#         else:
#             entities_combinations = [list(combo)
#                                      for combo in itertools.product(*entities_lists)]
#         entities_used_combinations = []
#         for entities_combination in entities_combinations:
#             for entity in entities_combination:
#                 if entity not in self._cache.get_objects_by_class(entity_class):
#                     raise Exception("Entity should be specified", entity.name)
#                 if get_attr_value(entity, "entity_type") not in self._cache.get_objects_by_class("EntityType"):
#                     entity_type = get_attr_value(entity, "entity_type")
#                     self._cache.cache_object("EntityType", entity_type.get_static_model_id(), entity_type)
#                     # raise Exception("Entity Type should be specified", get_attr_value(entity, "name"))
#             entities_used = [(entity,)
#                              for entity in entities_combination]
#
#             entities_used_combinations.append(entities_used)
#         return entities_used_combinations


# Module-Specific Constants
# logging.basicConfig(filename='data_transformation_management.log', level=logging.DEBUG)


def get_digital_twin_updated(root_path, project_path, digital_twin_model, start_datetime, end_datetime,
                             data_source_entry_mapper_path, empty_agents_model=None, progress_tracker=None,
                             artificial_simulation_need=False):
    if empty_agents_model is None:
        empty_agents_model = None  # Agents()
    real_world_dtm = DataIntegrationPipeline(project_path=project_path,
                                                state_model=digital_twin_model,
                                                data_source_entry_mapper_path=data_source_entry_mapper_path,
                                                change_handler=None, progress_tracker=progress_tracker)
    # Init change_handler
    change_handler = ChangeHandlerPhysicalWorld(digital_twin=digital_twin_model,
                                                environment=real_world_dtm,
                                                agents=empty_agents_model)
    real_world_dtm.add_change_handler(change_handler)

    enriched_digital_twin = real_world_dtm.update_state_model(state_model=digital_twin_model,
                                                              start_datetime=start_datetime,
                                                              end_datetime=end_datetime)

    return enriched_digital_twin


if __name__ == "__main__":

    from projects.bicycle_world.settings import PROJECT_PATH
    from ofact.planning_services.model_generation.persistence import (
        deserialize_state_model, get_state_model_file_path, serialize_state_model)
    from ofact.twin.repository_services.deserialization.order_types import OrderType
    from typing import TYPE_CHECKING
    from ofact.twin.state_model.model import _transform_order_pool_list_to_np_array

    if TYPE_CHECKING:
        from ofact.twin.state_model.model import StateModel
    persistence_format = "pkl"
    digital_twin_file_name = f"bicycle_factory.{persistence_format}"

    state_model_path = get_state_model_file_path(project_path=PROJECT_PATH,
                                                 state_model_file_name=digital_twin_file_name,
                                                 path_to_model="scenarios/current/models/twin/")

    state_model_generation_settings = \
        {"order_generation_from_excel": False,
         "customer_generation_from_excel": True,
         "order_amount": 0,
         "order_type": OrderType.PRODUCT_CONFIGURATOR}

    state_model: StateModel = (
        deserialize_state_model(source_file_path=state_model_path, persistence_format=persistence_format,
                                state_model_generation_settings=state_model_generation_settings,
                                deserialization_required=False))
    state_model.orders = _transform_order_pool_list_to_np_array([])

    # Get data for dashboard
    start_datetime = datetime(2024, 10, 22, 8)
    end_datetime = None  # datetime(2024, 10, 29)

    data_source_model_path = Path(PROJECT_PATH +
                                  "/scenarios/current/models/data_source/dsm.xlsx")
    updated_state_model = get_digital_twin_updated(ROOT_PATH, PROJECT_PATH,
                                                   state_model, start_datetime, end_datetime, data_source_model_path)

    target_file_path = "./updated_bicycle_factory.pkl"
    serialize_state_model(state_model=updated_state_model, target_file_path=target_file_path,
                          dynamics=True)
