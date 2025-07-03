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
    ExternalIDMapper
"""


from ofact.env.data_integration.helper import get_attr_value


class ExternalIDMapper:
    next_id = 0

    def __init__(self):
        self.external_id_mapper = {}

    def get_internal_id(self, name_space, external_id):
        if (name_space, external_id) not in self.external_id_mapper:
            new_id = self._get_next_id()
            self.external_id_mapper[(name_space, external_id)] = new_id
            return new_id

        else:
            return self.external_id_mapper[(name_space, external_id)]

    def _get_next_id(self):
        next_id = type(self).next_id
        type(self).next_id += 1

        return next_id


class ObjectCache:
    """
    Caches the objects used in the data transformation/ integration until they are instantiated.
    """

    def __init__(self, class_names: list[str] = None):
        if class_names is None:
            class_names = []

        self._class_names = class_names
        self._object_memory: dict[str: dict[int: dict]] = {}
        self._order_process_executions: dict = {}

        # objects like Parts that should be planned individually to did not choose for two processes the same part
        self._objects_already_planned: dict[str, list] = {}
        self.external_id_mapper = ExternalIDMapper()

        # instantiation sequence is needed because ProcessExecution needs EntityType, ActiveM.., Ass.., Part etc.
        # already initialized (to use them as attributes) -> adaptable for other objects
        self._instantiation_sequence = ['Feature', 'Customer', 'Part', 'Order', 'EntityType', 'PartType',
                                        'Resource', 'Storage', 'Warehouse',
                                        'ActiveMovingResource', 'PassiveMovingResource',
                                        'WorkStation', 'Process', 'ValueAddedProcess', 'ProcessExecution']

    def store_object_dict(self, class_name, name_space, external_id, object_dict):
        if class_name not in self._object_memory:
            self._object_memory[class_name] = {}
            self._class_names.append(class_name)
        if isinstance(external_id, list):
            external_id = external_id[0]
        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        self._object_memory[class_name][internal_id] = object_dict

        if class_name != "ProcessExecution":
            return

        self.map_process_executions_to_orders(object_dict)

    def remove_object_dict(self, class_name, name_space, external_id):
        """remove objects that are for example not filled enough to be loaded"""

        if class_name not in self._object_memory:
            return

        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        if internal_id not in self._object_memory[class_name]:
            return

        del self._object_memory[class_name][internal_id]

    def update_order_process_executions(self):

        if "ProcessExecution" not in self._object_memory:
            return

        process_executions_dicts = list(self._object_memory["ProcessExecution"].values())
        for object_dict in process_executions_dicts:
            self.map_process_executions_to_orders(object_dict)

    def map_process_executions_to_orders(self, object_dict):

        order = get_attr_value(object_dict, "order")
        if isinstance(order, dict):
            external_identifications = get_attr_value(order, "external_identifications")
            order_ids = list(*external_identifications.values())
        else:
            order_ids = []

        for order_id in order_ids:
            self._order_process_executions.setdefault(order_id,
                                                      []).append(object_dict)

    def get_object_by_external_identification(self, name_space, external_id, class_name=None) -> dict | None:
        if class_name not in self._object_memory:
            return None

        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        if internal_id not in self._object_memory[class_name]:
            return None

        object_ = self._object_memory[class_name][internal_id]

        return object_

    def get_process_executions_order(self, order_id):

        if order_id not in self._order_process_executions:
            self.update_order_process_executions()

        if order_id in self._order_process_executions:
            process_executions = self._order_process_executions[order_id]
        else:
            process_executions = []

        return process_executions

    def pop(self) -> tuple[str | None, list[dict] | None]:
        """pop element from object_cache"""
        for type_ in self._instantiation_sequence:
            if type_ not in self._object_memory:
                continue

            dict_of_objects = self._object_memory[type_]
            if not dict_of_objects:
                continue

            elem_batch = list(dict_of_objects.values())
            del self._object_memory[type_]

            elem_batch = [elem for elem in elem_batch if isinstance(elem, dict)]

            if not elem_batch:
                continue

            return type_, elem_batch

        self._order_process_executions = {}

        return None, None

    def empty(self):
        if any(list(self._object_memory.values())):
            return False
        else:
            return True

    def store_object_already_planned(self, type_: str, object_):
        self._objects_already_planned.setdefault(type_,
                                                 []).append(object_)

    def get_objects_already_planned(self, type_: str) -> list:
        if type_ in self._objects_already_planned:
            return self._objects_already_planned[type_]
        else:
            return []

    def get_passive_moving_resources(self):
        if "PassiveMovingResource" in self._object_memory:
            return list(self._object_memory["PassiveMovingResource"].values())
        else:
            return {}

    def get_active_moving_resources(self):
        if "ActiveMovingResource" in self._object_memory:
            return list(self._object_memory["ActiveMovingResource"].values())
        else:
            return {}