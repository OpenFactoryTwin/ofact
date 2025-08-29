"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################

classes:
    ObjectCache
"""
from typing import Optional

from ofact.env.data_integration.generation_and_integration.helper import get_attr_value
from ofact.twin.state_model.basic_elements import DigitalTwinObject


class CacheEntry:

    def __init__(self, object_, tags: Optional[list] = None):
        self.object_id = self._get_object_id(object_)
        self.object = object_
        if tags is None:
            tags = []
        self.tags = tags

    def _get_object_id(self, object_: dict | DigitalTwinObject):
        if isinstance(object_, dict):
            return object_["identification"]
        elif isinstance(object_, DigitalTwinObject):
            return object_.identification

        return None

    def get_object(self, tag=None) -> Optional:
        if tag in self.tags or tag is None:
            return self.object
        else:
            return None


class ObjectCache:
    """
    Caches the objects used in the data transformation/ integration until they are instantiated.
    """

    def __init__(self):
        self._object_memory: dict[str: dict[tuple: list[dict | object]]] = {}

    def _get_object_id_key(self, name_space, external_id):
        return (name_space, str(external_id))

    def cache_object(self, class_name, name_space, external_id, object_dict, tags: Optional[list] = None):
        """Store an object (can be a dict or a custom python object)"""

        if class_name not in self._object_memory:
            self._object_memory[class_name] = {}

        if isinstance(external_id, list):
            external_id = external_id[0]

        object_id_key = self._get_object_id_key(name_space, external_id)
        cache_entry = CacheEntry(object_dict, tags)
        self._object_memory[class_name].setdefault(object_id_key,
                                                   []).append(cache_entry)

    def remove_object(self, class_name, name_space, external_id):
        """remove objects that are for example not filled enough to be loaded"""

        if class_name not in self._object_memory:
            return

        object_id_key = self._get_object_id_key(name_space, external_id)
        if object_id_key not in self._object_memory[class_name]:
            return

        del self._object_memory[class_name][object_id_key]  # ToDo: delete the whole list

    def _get_objects(self, class_name, object_id_key, tag):
        cache_entries = self._object_memory[class_name][object_id_key]
        return [cache_entry.get_object(tag)
                for cache_entry in cache_entries
                if cache_entry.get_object(tag)]

    def get_objects_by_class(self, class_name):
        if class_name not in self._object_memory:
            return []

        return self._get_objects_from(class_name)

    def get_object(self, name_space="static_model", class_name=None, id_=None, tag: Optional = None) -> (
            object | None):
        if class_name not in self._object_memory:
            return None

        object_id_key = self._get_object_id_key(name_space, str(id_))
        if object_id_key not in self._object_memory[class_name]:
            return None

        objects = self._get_objects(class_name, object_id_key, tag)
        if objects:
            return objects[0]

        return None

    def _get_objects_from(self, class_name):
        cache_entry_lists = self._object_memory[class_name]
        return [cache_entry.get_object()
                for lst in list(cache_entry_lists.values())
                for cache_entry in lst]

    def empty(self):
        if any(list(self._object_memory.values())):
            return False
        else:
            return True

    def get_passive_moving_resources(self):
        if "PassiveMovingResource" in self._object_memory:
            return self._get_objects_from("PassiveMovingResource")
        else:
            return {}

    def get_active_moving_resources(self):
        if "ActiveMovingResource" in self._object_memory:
            return self._get_objects_from("ActiveMovingResource")
        else:
            return {}


class ObjectCacheDataIntegration(ObjectCache):

    def __init__(self):
        super().__init__()

        self._order_process_executions: dict = {}
        # objects like Parts that should be planned individually to did not choose for two processes the same part
        self._objects_already_planned: dict[str, list] = {}

        # instantiation sequence is needed because ProcessExecution needs EntityType, ActiveM.., Ass.., Part etc.
        # already initialized (to use them as attributes) -> adaptable for other objects
        self._instantiation_sequence = ['Feature', 'Customer', 'Part', 'Order', 'EntityType', 'PartType',
                                        'Resource', 'StationaryResource', 'Storage', 'Warehouse',
                                        'ActiveMovingResource', 'PassiveMovingResource',
                                        'WorkStation', 'Process', 'ValueAddedProcess', 'ProcessExecution']

    def cache_object(self, class_name, external_id, sm_object, tags: Optional[list] = None, name_space="static_model"):
        super().cache_object(class_name=class_name, name_space=name_space, external_id=external_id,
                             object_dict=sm_object, tags=tags)

        if class_name != "ProcessExecution":
            return

        self._map_process_executions_to_orders(sm_object)

    def cache_object_already_planned(self, type_: str, object_):
        self._objects_already_planned.setdefault(type_,
                                                 []).append(object_)

    def get_objects(self, name_space="static_model", class_name = None, id_ = None, tag: Optional = None):
        """
        Generally, it is possible that more than one object exists with the same id
        (e.g., if it is stated that x elements exist from the "id_")
        """
        if class_name not in self._object_memory:
            return None
        object_id_key = self._get_object_id_key(name_space, id_)
        if object_id_key not in self._object_memory[class_name]:
            return None
        objects = self._get_objects(class_name, object_id_key, tag)

        return objects

    def get_objects_already_planned(self, type_: str) -> list:
        if type_ in self._objects_already_planned:
            return self._objects_already_planned[type_]
        else:
            return []

    def get_process_executions_order(self, order_id):

        if order_id not in self._order_process_executions:
            self._update_order_process_executions()

        if order_id in self._order_process_executions:
            process_executions = self._order_process_executions[order_id]
        else:
            process_executions = []

        return process_executions

    def _update_order_process_executions(self):

        if "ProcessExecution" not in self._object_memory:
            return

        process_executions_dicts = self._get_objects_from("ProcessExecution")
        for object_dict in process_executions_dicts:
            self._map_process_executions_to_orders(object_dict)

    def _map_process_executions_to_orders(self, object_dict):

        order = get_attr_value(object_dict, "order")
        if isinstance(order, dict):
            external_identifications = get_attr_value(order, "external_identifications")
            order_ids = list(*external_identifications.values())
        else:
            order_ids = []

        for order_id in order_ids:
            self._order_process_executions.setdefault(order_id,
                                                      []).append(object_dict)

    def pop(self) -> tuple[str | None, list[dict] | None]:
        """pop element from object_cache"""
        for type_ in self._instantiation_sequence:
            if type_ not in self._object_memory:
                continue

            dict_of_objects = self._get_objects_from(type_)
            if not dict_of_objects:
                continue

            del self._object_memory[type_]

            elem_batch = [obj
                          for obj in dict_of_objects
                          if isinstance(obj, dict)]

            if not elem_batch:
                continue

            return type_, elem_batch

        self._order_process_executions = {}

        return None, None


class ObjectCacheAutoModelling(ObjectCache):

    def __init__(self):
        super().__init__()

    #def get_object(self, name_space="static_model", class_name = None, id_ = None, tag: Optional = None) -> (
    #        object | None):
    #    return super().get_object(name_space, str(id_), class_name)

    def get_objects(self, name_space="static_model", class_name = None, id_ = None, tag: Optional = None):
        """
        Generally, it is possible that more than one object exists with the same id
        (e.g., if it is stated that x elements exist from the "id_")
        """
        if class_name not in self._object_memory:
            return None
        object_id_key = self._get_object_id_key(name_space, id_)
        if object_id_key not in self._object_memory[class_name]:
            return None
        objects = self._get_objects(class_name, object_id_key, tag)

        return objects

    def cache_object(self, class_name, external_id, sm_object, name_space="static_model", tags: Optional[list] = None):

        if class_name not in self._object_memory:
            self._object_memory[class_name] = {}
        if isinstance(external_id, list):
            external_id = external_id[0]
        object_id_key = self._get_object_id_key(name_space, external_id)
        cache_entry = CacheEntry(sm_object, tags)
        # if object_id_key in self._object_memory[class_name]:
        #     raise Exception(f"Object with internal id {object_id_key} already exists", external_id)
        self._object_memory[class_name].setdefault(object_id_key,
                                                   []).append(cache_entry)

    def remove_object(self, class_name, name_space, external_id):
        """remove objects that are for example not filled enough to be loaded"""

        if class_name not in self._object_memory:
            return

        object_id_key = self._get_object_id_key(name_space, external_id)
        if object_id_key not in self._object_memory[class_name]:
            return

        del self._object_memory[class_name][object_id_key]
