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

All the entities (resources and parts that have a physical counterpart on the shop floor) of the twin.

Classes:
    Plant: The Plant for the production
    ---
    EntityType: Type of the entity
    Entity: Physical entities such as parts or resources
    ---
    PartType: Type of the part that supplements the entity type by dimension, weight and volume
    Part: A special entity that can be consumed, and can build a bill of material
    ---
    PhysicalBody: Describes a position and extent of objects
    Resource: A special entity that needs to be used in processes, but is not consumed
    ---
    StationaryResource: Resources that cannot move
    Storage: Stationary resource to store other entities
        (only entities of the same entity type can be situated in a storage)
    StoragePlaces: Encapsulation of storages that belong to a resource
    WorkStation: Stationary resource to process parts to a product with a possible given order of execution
    Warehouse: Stationary resource that contains several storages.
    ConveyorBelt: Resource that passes entities from one storage to another
    ---
    NonStationaryResource: Resources that can move
    ActiveMovingResource: A NonStationaryResource which can move
    PassiveMovingResource: A NonStationaryResource which cannot move by themselves

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import json
import logging
import random
from abc import ABCMeta, abstractmethod
from copy import copy
from datetime import datetime, timedelta
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING, Optional, Union

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import (DigitalTwinObject, DynamicDigitalTwinObject, ProcessExecutionTypes,
                                                   prints_visible)
from ofact.twin.state_model.probabilities import ProbabilityDistribution, SingleValueDistribution
from ofact.twin.state_model.serialization import Serializable

if TYPE_CHECKING:
    from ofact.twin.state_model.time import WorkCalender, ProcessExecutionPlan, ProcessExecutionPlanConveyorBelt
    from ofact.twin.state_model.processes import ProcessExecution

logging.debug("DigitalTwin/entities")


class Plant(DigitalTwinObject, Serializable):

    def __init__(self,
                 name: str,
                 corners: list[tuple[int, int]],
                 current_time: datetime = datetime(1970, 1, 1),
                 work_calendar: Optional[WorkCalender] = None,
                 identification: Optional[int] = None,
                 external_identifications: dict[object, list[object]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The Plant comprises/ describes the space where the value transformation takes place.

        Parameters
        ----------
        name: Name of the Plant
        corners: a list of positions (int, int) that describe the corners of the plant, walls are situated
        between direct neighbors in the list
        current_time: current date and time
        work_calendar: calendar that can differentiate between work and free time
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name
        self.corners: list[tuple[int, int]] = corners
        self.current_time = current_time
        self.work_calendar: Optional[WorkCalender] = work_calendar

    def __str__(self):
        return (f"Plant with ID '{self.identification}'; "
                f"'{self.corners}', '{self.current_time}', '{self.work_calendar}'")

    def copy(self):
        """Copy the object with the same identification."""
        plant_copy: Plant = super(Plant, self).copy()
        plant_copy.corners = plant_copy.corners.copy()
        plant_copy.work_calendar = plant_copy.work_calendar.copy()

        return plant_copy


class EntityType(DigitalTwinObject):

    def __init__(self,
                 name: str,
                 super_entity_type: Optional[EntityType] = None,
                 identification: Optional[int] = None,
                 external_identifications: dict[object, list[object]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        A type of entity, similar to the 'type' in the asset administration shell (AAS).
        (e.g., part type: 'sport bicycle',
        resource type: 'cutting machine')

        Parameters
        ----------
        name: Name of the type
        super_entity_type: An entity_type that describes the next higher hierarchy level (aggregation of
        entity_types)
        e.g., sport bicycle (et) and bicycle (super-et) or cutting machine (et) and machines (super-et)
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.super_entity_type = super_entity_type
        self.name: str = name

    def __str__(self):
        super_entity_type_name = self.get_super_entity_type_name()
        return f"EntityType with ID '{self.identification}' and name {self.name}'; '{super_entity_type_name}'"

    def copy(self):
        """Copy the object with the same identification."""
        entity_type_copy = super(EntityType, self).copy()

        return entity_type_copy

    def get_super_entity_type_name(self):
        if self.super_entity_type is not None:
            return self.super_entity_type.name
        else:
            return ""

    def check_entity_type_match(self, other_entity_type: EntityType) -> bool:
        """
        Check if the other_entity_type matches to self.

        Parameters
        ----------
        other_entity_type: entity_type that is checked towards matching self (entity_type object).

        Returns
        -------
        True if the entity_types matches else False
        """
        if self.identification == other_entity_type.identification:
            return True

        if self.super_entity_type is not None:
            if self.super_entity_type.identification == other_entity_type.identification:
                return True

        elif other_entity_type.super_entity_type is not None:
            if self.identification == other_entity_type.super_entity_type.identification:
                return True

        return False

    def check_entity_type_match_lower(self, lower_equal_entity_type: EntityType) -> bool:
        """
        Check if the lower_equal_entity_type matches to self.

        Parameters
        ----------
        lower_equal_entity_type: Entity_type that is checked towards matching self (entity_type object).
        But the entity_type should be of lower / equal degree as the entity_type self

        Returns
        -------
        True if the entity_types matches else False
        """
        if self.identification == lower_equal_entity_type.identification:
            return True

        elif lower_equal_entity_type.super_entity_type is not None:
            if self.identification == lower_equal_entity_type.super_entity_type.identification:
                return True

        return False

    def completely_filled(self):
        not_completely_filled_attributes = []
        if self.identification is None:
            not_completely_filled_attributes.append("identification")
        if not isinstance(self.name, str):
            not_completely_filled_attributes.append("name")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class Entity(DynamicDigitalTwinObject):

    def __init__(self,
                 name: str,
                 entity_type: Union[EntityType, PartType],
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime(1970, 1, 1),
                 external_identifications: dict[object, list[object]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Entities are physical objects of the value creation. They can be parts or resources.

        Parameters
        ----------
        name: Name of the entity
        process_execution: list of process executions that are planned OR already happened for this entity in
        temporal order
        entity_type: type of the entity
        situated_in: A resource where the entity can be situated in, if not assigned, the absolute position
        of the physical body in the plant is used
        quality: determines the quality of the entity that is a float between 0 and 1 (0 means bad, 1 means good)
        inspected_quality: While the quality is a hidden value used for simulation,
        the inspected quality is a snapshot of the real quality value.
        On the physical shop floor, only these snapshots are available.
        To ensure realistic behavior, the inspected quality should be used for decisions.
        """
        self._entity_type: Union[EntityType, PartType] = entity_type
        self._situated_in: Optional[Resource] = situated_in
        self._quality: float = quality
        self._inspected_quality: Optional[float] = inspected_quality
        super().__init__(identification=identification, process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        situated_in_name = self.get_situated_in_name()
        return (f"Entity with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}'")

    @abstractmethod
    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        entity_copy = super(Entity, self).duplicate(external_name)
        return entity_copy

    def get_sub_instances_to_add(self):
        """Used after duplication to add also the sub_instances to the digital_twin"""
        sub_digital_twin_objects = {}

        return sub_digital_twin_objects

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        entity_copy = super(Entity, self).copy()

        return entity_copy

    @property
    def entity_type(self):
        return self._entity_type

    @entity_type.setter
    def entity_type(self, entity_type):
        if isinstance(entity_type, EntityType):
            self._entity_type = entity_type
        else:
            raise ValueError(f"[{self.__class__.__name__}] "
                             f"The new entity_type {entity_type} is not an instance of EntityType")

    def change_entity_type(self, entity_type, process_execution: Optional[ProcessExecution] = None,
                           sequence_already_ensured: bool = False):
        self.entity_type = entity_type
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   entity_type=self.entity_type,
                                   sequence_already_ensured=sequence_already_ensured)

    @property
    def situated_in(self):
        return self._situated_in

    @situated_in.setter
    def situated_in(self, situated_in):
        if isinstance(situated_in, Resource) or situated_in is None:
            self._situated_in = situated_in
        else:
            raise ValueError(f"[{self.__class__.__name__}] The new situated_in is not a Resource or "
                             f"a class derived from Resource | {self.__dict__}")

    def change_situated_in(self, situated_in: Optional[Resource], process_execution: Optional[ProcessExecution] = None,
                           sequence_already_ensured: bool = False):
        """Change situated in of a resource and add the change to the dynamic attributes"""

        self.situated_in = situated_in
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   situated_in=self.situated_in, sequence_already_ensured=sequence_already_ensured)

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, quality: float):
        if 0 <= quality <= 1:
            self._quality = quality
        else:
            raise ValueError(f"[{self.__class__.__name__}] The new quality is not between 0 and 1")

    def change_quality(self, quality: float, process_execution: Optional[ProcessExecution] = None,
                       sequence_already_ensured: bool = False):
        """
        Parameters
        ----------
        quality: a quality value
        process_execution: process_execution that is responsible for the change
        """
        self.quality = quality
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   quality=self._quality, sequence_already_ensured=sequence_already_ensured)

    @property
    def inspected_quality(self):
        return self._inspected_quality

    def inspect_quality(self, process_execution: Optional[ProcessExecution] = None):
        self._inspected_quality = self._quality

    def reset_inspected_quality(self, process_execution: Optional[ProcessExecution] = None):
        self._inspected_quality = None

    def get_entity_type_name(self):
        if self._entity_type is not None:
            return self._entity_type.name
        else:
            return ""

    def get_situated_in_name(self):
        if self._situated_in is not None:
            return self._situated_in.name
        else:
            return ""

    def check_situated_in(self, entity: Entity) -> bool:
        """
        Determines if the entity situated_in resource (self) (also possible at a higher level)

        Parameters
        ----------
        entity: An entity that is checked if it is situated in

        Returns
        -------
        True if the entity is situated in the resource (self) (also possible at higher level)
        """
        situated_in = self.situated_in
        while situated_in is not None:
            if entity.identification == situated_in.identification:
                return True
            situated_in = situated_in.situated_in
        return False

    def completely_filled(self):

        not_completely_filled_attributes = []
        if self.identification is None:
            not_completely_filled_attributes.append("identification")
        if not isinstance(self.name, str):
            not_completely_filled_attributes.append("name")
        if not isinstance(self.entity_type, EntityType):
            not_completely_filled_attributes.append("entity_type")
        if not isinstance(self.situated_in, Entity):
            not_completely_filled_attributes.append("situated_in")
        if not isinstance(self._quality, float):
            not_completely_filled_attributes.append("quality")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class PartType(EntityType):

    # ToDo: unit converter based on the given units ...

    def __init__(self,
                 name: str,
                 super_entity_type: Optional[EntityType] = None,
                 weight: Optional[float] = None,
                 weight_unit: Optional[float] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 depth: Optional[float] = None,
                 dimension_unit: Optional[str] = None,
                 volume: Optional[float] = None,
                 volume_unit: Optional[str] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        The part type supplements the entity type with information about weight, dimension and volume of the part.

        Parameters
        ----------
        name: name of the part
        weight: weight of the part
        weight_unit: unit of the weight such as gram, kilo gram, ...
        height: height of the part
        width:depth of the part
        depth: depth of the part
        dimension_unit: unit of the dimension such as meter, centimeter, ...
        volume: volume of the part
        volume_unit: unit of the volume such as gram, kilo gram, ...
        """
        super().__init__(name=name, super_entity_type=super_entity_type, identification=identification,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        self.weight: Optional[float] = weight
        self.weight_unit: Optional[str] = weight_unit

        self.height: Optional[float] = height
        self.width: Optional[float] = width
        self.depth: Optional[float] = depth
        self.dimension_unit: Optional[str] = dimension_unit

        self.volume: Optional[float] = volume
        self.volume_unit: Optional[str] = volume_unit

    def __str__(self):
        super_entity_type_name = self.get_super_entity_type_name()
        return (f"PartType with ID '{self.identification}' and name {self.name}'; '{super_entity_type_name}', "
                f"'{self.weight}', '{self.weight_unit}', "
                f"'{self.height}', '{self.width}', '{self.depth}', '{self.dimension_unit}', "
                f"'{self.volume}', '{self.volume_unit}'")

    def calculate_volume(self):
        """Calculate the volume of an object."""
        if self.weight and self.height and self.width:
            volume_calculated = self.weight * self.height * self.width
        else:
            volume_calculated = None

        return volume_calculated


class Part(Entity):
    cross_domain_attributes: dict[str, type] = {
        # similar to the part entity_type and used if the part entity_type is used for other issues (modeling freedom)
        "product_group": str,
        # target storage place for storage places not modeled in the digital twin (more fine-grained)
        "target_sub_storage_place": str,
        # actual storage place for storage places not modeled in the digital twin (more fine-grained)
        "sub_storage_place": str,
        # distance for example to a storage place to take the part
        "distance": float,
        # volume of a specified single part/ share of the part
        "single_volume": float,
        # weight of a specified single part/ share of the part
        "single_weight": float,
        # height of the Stock Keeping Unit (SKU)
        "sku_height": float,
        # width of the Stock Keeping Unit (SKU)
        "sku_width": float,
        # depth of the Stock Keeping Unit (SKU)
        "sku_depth": float,
        # number of the Stock Keeping Unit (SKU)
        "sku_number": float,
        # the quantity for example of sock keeping unit that should be included in the part
        "target_quantity": float,
        # the actual quantity for example of sock keeping unit included in the part
        "quantity": float,
        # quantity that is inspected
        "inspection_quantity": float,
        # pre count quantity determined in the picking area
        "pre_count_quantity_picking": float,
        # pre count quantity determined from the supplier
        "pre_count_quantity_supplier": float,
        # describes if (respectively) how dangerous the good is
        "hazardous_good": int,
        # determines if the part has the best before date
        "best_before": bool,
        # shipping type in the planning phase
        "shipping_type_planned": str,
        # actual shipping type in/ respectively after the shipping
        "shipping_type_actual": str
    }

    def __init__(self,
                 name: str,
                 entity_type: Union[PartType, EntityType],
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 unit: Optional[str] = None,
                 part_of: Optional[Part] = None,
                 parts: Optional[list[Part]] = None,
                 part_removable: Optional[list[bool]] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Parts are entities that can be consumed and transformed during a process

        Parameters
        ----------
        name: name of the part
        entity_type: type of the part
        unit: describes how the part is counted such as piece, kg, l, m,
        part_of: parts can be part of another part
        parts: parts can consist of several other parts (first level of bill of material)
        situated_in: A resource the part might be currently situated in
        part_removable : a boolean per part that states if the part can be removed or not.
        """

        self.unit: Optional[str] = unit  # could be also an attribute of part_type

        self.part_of: Optional[Part] = part_of

        if parts is None:
            parts = []
        self.parts: list[Part] = parts

        if part_removable is None:
            part_removable = []
        self.part_removable: list[bool] = part_removable

        super().__init__(identification=identification, name=name, entity_type=entity_type, situated_in=situated_in,
                         quality=quality, inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        situated_in_name = self.get_situated_in_name()
        part_of_name = self.get_part_of_name()
        part_names = self.get_part_names()
        return (f"Part with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{self.unit}', "
                f"'{part_of_name}', '{part_names}', '{self.part_removable}")

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes
        It is assumed that the part is not a part of another part. This means that the parts above are neglected.
        """
        part_duplicate = super(Part, self).duplicate(external_name)

        part_duplicate.part_of = None
        part_duplicate.parts = [part.duplicate(external_name) for part in part_duplicate.parts]
        part_duplicate.parts = [part.duplicate_finish(part_duplicate) for part in part_duplicate.parts]
        return part_duplicate

    def duplicate_finish(self, part_of):
        """The duplicate finish set the part_of relations"""
        self.part_of = part_of
        for part in self.parts:
            part.duplicate_finish(self)

    def copy(self):
        """Copy the object with the same identification."""
        part_copy: Part = super(Part, self).copy()
        part_copy.part_removable = part_copy.part_removable.copy()

        return part_copy

    @property
    def weight(self):
        if isinstance(self.entity_type, PartType):
            return self._entity_type.weight

    @property
    def height(self):
        if isinstance(self.entity_type, PartType):
            return self._entity_type.height

    @property
    def width(self):
        if isinstance(self.entity_type, PartType):
            return self._entity_type.width

    @property
    def depth(self):
        if isinstance(self.entity_type, PartType):
            return self._entity_type.depth

    @property
    def volume(self):
        if isinstance(self.entity_type, PartType):
            return self._entity_type.volume

    def get_part_of_name(self):
        if self.part_of is not None:
            return self.part_of.name
        else:
            return ""

    def get_part_names(self):
        part_names = [part.name
                      for part in self.parts]
        return part_names

    def check_part_removable(self, part):
        """Check if the part is removable"""
        if part in self.parts:
            part_index = self.parts.index(part)
            return self.part_removable[part_index]
        else:
            raise ValueError(f"[{self.__class__.__name__}] check_part_removable"
                             f"The part {part.identification} is part_of {self.identification}")

    def add_entity(self, entity, removable: bool, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        return self._add_part(entity, removable, process_execution, sequence_already_ensured)


    def _add_part(self, part, removable: bool, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        Add a part to self (part is appended)

        Parameters
        ----------
        part: part that is attached/ added to self
        removable: describes if the part is removable
        process_execution: process_execution that is responsible for the adding
        """
        if not isinstance(part, Part):
            raise ValueError(f"[{self.__class__.__name__}] The part is not a instance of Part")

        if part not in self.parts:
            self.parts.append(part)
            self.part_removable.append(removable)

            if part.situated_in:
                # remove the entity from the storage
                part.situated_in.remove_entity(entity=part, process_execution=process_execution,
                                               sequence_already_ensured=sequence_already_ensured)

            if process_execution:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       parts=part, part_removable=removable,
                                       change_type="ADD", sequence_already_ensured=sequence_already_ensured)

        part._be_part_of(part=self, process_execution=process_execution,
                         sequence_already_ensured=sequence_already_ensured)

    def _be_part_of(self, part, process_execution: Optional[ProcessExecution] = None,
                    sequence_already_ensured: bool = False):
        """
        Add a part to another part.

        Parameters
        ----------
        part: Part to which self is added
        process_execution: Process_execution that is responsible for the be_part_of
        """
        if isinstance(part, Part):
            if self.part_of != part:
                self.part_of = part
                if process_execution:
                    self.update_attributes(process_execution=process_execution,
                                           current_time=process_execution.executed_end_time,
                                           part_of=self.part_of, sequence_already_ensured=sequence_already_ensured)
        else:
            raise ValueError(f"[{self.__class__.__name__}] The part is not a instance of Part")

    def get_disassembled_parts(self, part_entity_type: EntityType | PartType, amount: int,
                               process_execution: Optional[ProcessExecution] = None,
                               sequence_already_ensured: bool = False):
        """
        Disassemble (amount) parts with the part_entity_type.

        Parameters
        ----------
        part_entity_type: entity_type of the subparts to be disassembled
        amount: amount of parts to be disassembled
        process_execution: process_execution that is responsible for the disassembly
        sequence_already_ensured: says if the process execution sequence is time chronological or not
        """

        sub_parts = [sub_part
                     for idx, sub_part in enumerate(self.parts)
                     if part_entity_type.check_entity_type_match_lower(sub_part.entity_type)
                     and self.part_removable[idx]]

        if len(sub_parts) < amount:
            process_name = process_execution.get_process_name()
            raise ValueError(f"[{self.__class__.__name__}] Not enough parts are attached to the part and can therefore "
                             f"not disassembled from part '{self.name}' ('{self.external_identifications}') "
                             f"in process: '{process_name}' \n"
                             f"Share available: {len(sub_parts)} / {amount}")

        # last in fist out principle
        sub_parts_to_disassemble = sub_parts[-int(amount):]

        # disassembly
        for sub_part in sub_parts_to_disassemble:
            sub_part.part_of = None
            idx = self.parts.index(sub_part)
            self.parts.remove(sub_part)
            del self.part_removable[idx]

            if process_execution:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       parts=sub_part, part_removable=True,
                                       change_type="REMOVE", sequence_already_ensured=sequence_already_ensured)
                sub_part.update_attributes(process_execution=process_execution,
                                           current_time=process_execution.executed_end_time,
                                           part_of=sub_part.part_of, sequence_already_ensured=sequence_already_ensured)

        return sub_parts_to_disassemble

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not (self.part_of is None or isinstance(self.part_of, Entity)):
            not_completely_filled_attributes.append("part_of")
        if not isinstance(self.parts, list):
            not_completely_filled_attributes.append("parts")
        if not isinstance(self.part_removable, list):
            not_completely_filled_attributes.append("part_removable")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


def _check_intersection_rectangles(left_top1, left_top2, right_bottom1, right_bottom2) -> bool:
    """
    Check intersection between two rectangles.

    Parameters
    ----------
    left_top1: the left top corner of the first rectangle.
    left_top2: the left top corner of the second rectangle.
    right_bottom1: the right bottom corner of the first rectangle.
    right_bottom2: the right bottom corner of the second rectangle.

    Returns
    -------
    True if the rectangles intersect else false
    """
    # To check if either rectangle is actually a line
    if (left_top1[0] == right_bottom1[0] or
            left_top1[1] == right_bottom1[1] or
            left_top2[0] == right_bottom2[0] or
            left_top2[1] == right_bottom2[1]):
        # the line cannot have positive overlap
        return False
    # If one rectangle is on left side of other
    elif (left_top1[0] >= right_bottom2[0] or
          left_top2[0] >= right_bottom1[0]):
        return False
    # If one rectangle is above other
    elif (right_bottom1[1] >= left_top2[1] or
          right_bottom2[1] >= left_top1[1]):
        return False

    return True


def _get_corner_top_left(physical_body):
    corner_top_left = (physical_body.position[0] - physical_body.width / 2,
                       physical_body.position[1] + physical_body.length / 2)

    return corner_top_left


def _get_corner_bottom_right(physical_body):
    corner_bottom_right = (physical_body.position[0] + physical_body.width / 2,
                           physical_body.position[1] - physical_body.length / 2)

    return corner_bottom_right


class PhysicalBody(DynamicDigitalTwinObject):
    
    def __init__(self,
                 position: tuple[int, int] = (0, 0),
                 length: int = 1,
                 width: int = 1,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Every resource can have a physical body that places assigns a position and an extension in space to it

        Parameters
        ----------
        position: X and Y coordinates specify the center of the physical body
        length: Extension in Y axis
        width: Extension in X axis
        """
        self._position: (int, int) = position
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.length = length
        self.width = width

    def __str__(self):
        return f"PhysicalBody({self.position}, {self.length}, {self.width})"

    def copy(self):
        """Copy the object with the same identification."""
        physical_body_copy: PhysicalBody = super(PhysicalBody, self).copy()
        physical_body_copy._position = copy(physical_body_copy._position)

        return physical_body_copy

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @position.setter
    def position(self, new_position: [int, int]):
        if type(new_position) == tuple:
            self._position = new_position
        else:
            raise ValueError(f"[{self.__class__.__name__}] The position cannot be changed "
                             f"because the new_position input_parameter has the false/ is not a tuple")

    def change_position(self, new_position: [int, int], process_execution: Optional[ProcessExecution] = None,
                        sequence_already_ensured: bool = False):
        """Change the position of the physical body
        In the case that the process_execution has the type plan, only the dynamic attributes are changed
        """

        change_position = True
        if process_execution:
            if process_execution.event_type == ProcessExecutionTypes.PLAN:
                change_position = False

            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   position=new_position, sequence_already_ensured=sequence_already_ensured)

        if change_position:
            # old_position = self.position
            self.position = new_position
            # print(f"Position from resource {process_execution.main_resource.name} "
            #       f"changed from {old_position} to {new_position}")

    def change_width(self, width: int):
        self.width = width

    def change_length(self, length: int):
        self.length = length

    def check_intersection_base_areas(self, other_physical_body: PhysicalBody) -> Optional[bool]:
        """
        Check intersection between base areas.

        Parameters
        ----------
        other_physical_body: other physical body that is checked according to area intersections

        Returns
        -------
        intersection: True if the base areas of the physical bodies (self and other_physical_body) have intersections
        else False
        """
        if self.position is None or other_physical_body.position is None:
            return None

        corner_top_left1 = _get_corner_top_left(self)
        right_bottom1 = _get_corner_bottom_right(self)

        corner_top_left2 = _get_corner_top_left(other_physical_body)
        right_bottom2 = _get_corner_bottom_right(other_physical_body)

        intersection = _check_intersection_rectangles(corner_top_left1, corner_top_left2, right_bottom1, right_bottom2)

        return intersection


class Resource(Entity, metaclass=ABCMeta):
    
    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Resources are entities that are needed to perform processes. They also can be used as obstacles in the plant
        layout.

        Parameters
        ----------
        name: name of the resource
        entity_type: type of the resource
        plant: the plant the resource belongs to
        process_execution_plan: schedule that is used to plan the process_executions
        costs_per_second: costs of using the resource (per second)
        situated_in: A resource the part might be currently situated in
        position: attribute of the physical body (x, y) - coordinate
        length: attribute of the physical body

        attribute physical_body: position and extension of the resource
        """
        super().__init__(identification=identification, name=name, entity_type=entity_type, situated_in=situated_in,
                         quality=quality, inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.plant: Plant = plant
        self._process_execution_plan: Optional[ProcessExecutionPlan] = process_execution_plan
        self.costs_per_second: Optional[float] = costs_per_second

        if isinstance(physical_body, PhysicalBody):
            pass
        elif length is not None and width is not None:  # position should also be set
            physical_body = PhysicalBody(position=position, length=length, width=width)
        else:
            physical_body = PhysicalBody(position=(0, 0),  # initialization with standard values
                                         length=1,
                                         width=1)
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] Warning: "
                      f"It should be ensured that the physical body is set later in the instantiation process")
        self._physical_body: Optional[PhysicalBody] = physical_body  # should be set eventually

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        return (f"Resource with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}'")

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        resource_copy = super(Resource, self).duplicate(external_name)
        if resource_copy.process_execution_plan is not None:
            resource_copy.process_execution_plan = resource_copy.process_execution_plan.duplicate(external_name)
        resource_copy._physical_body = resource_copy._physical_body.duplicate(external_name)
        return resource_copy

    def get_sub_instances_to_add(self):
        """Used after duplication to add also the sub_instances to the digital_twin"""
        sub_digital_twin_objects = super(Resource, self).get_sub_instances_to_add()

        return sub_digital_twin_objects

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        resource_copy: Resource = super(Resource, self).copy()
        resource_copy.process_execution_plan = copy(resource_copy.process_execution_plan)
        resource_copy.physical_body = resource_copy.physical_body.copy()

        return resource_copy

    @property
    def physical_body(self):
        return self._physical_body

    @physical_body.setter
    def physical_body(self, physical_body):
        self._physical_body = physical_body

    @property
    def position(self):
        return self._physical_body.position

    @position.setter
    def position(self, position):
        self._physical_body.position = position

    @property
    def length(self):
        return self._physical_body.length

    @length.setter
    def length(self, length):
        self._physical_body.length = length

    @property
    def width(self):
        return self._physical_body.width

    @width.setter
    def width(self, width):
        self._physical_body.width = width

    @property
    def process_execution_plan(self):
        return self._process_execution_plan

    @process_execution_plan.setter
    def process_execution_plan(self, process_execution_plan):
        self._process_execution_plan = process_execution_plan

    def has_skill(self, requested_skill: EntityType) -> bool:
        """Check if the resource is able to provide the requested_skill"""

        has_skill: bool = self.entity_type.check_entity_type_match_lower(lower_equal_entity_type=requested_skill)
        return has_skill

    def get_plant_name(self):
        if self.plant is not None:
            return self.plant.name
        else:
            return ""

    def get_position(self, at: Optional[datetime] = None):
        """Get the position of the resource"""
        if at is not None:
            pass  # ToDo: position at some time point

        position = self._physical_body.position
        return position

    def get_positions(self, start_time_stamp, end_time_stamp=None):
        """Get the positions between start_time and end_time for the resource"""

        position_changes = \
            self.physical_body.get_changes_attribute(attribute="position",
                                                     start_time_stamp=start_time_stamp, end_time_stamp=end_time_stamp)
        return position_changes

    def get_approach_position(self):
        """Return the position a transport resource can approach (drive to if the transport resource want to pick up or
        deliver something)"""
        approach_position = self.get_position()
        return approach_position

    def check_intersection_base_areas(self, other_resource) -> Optional[bool]:
        """Check if the base areas from the physical body (self) and the physical body (other resource)
        have intersections"""
        return self._physical_body.check_intersection_base_areas(other_physical_body=other_resource.physical_body)

    def change_width(self, width: int):
        """Change the width of the physical body of the resource (self"""
        self._physical_body.change_width(width)

    def get_width(self):
        width = self._physical_body.width
        return width

    def change_length(self, length: int):
        """Change the length attribute of the physical body object of the resource (self)"""
        self._physical_body.change_length(length)

    def get_length(self):
        length = self._physical_body.length
        return length

    def get_storages(self, entity_type=None) -> dict[EntityType, list[Storage]]:
        """Return buffer_stations or storages if the object has a respective attribute"""
        return {}

    def get_storages_without_entity_types(self, entity_type=None) -> list[Storage]:
        """Return buffer_stations or storages if the object has a respective attribute"""
        return []

    @abstractmethod
    def add_entity(self, entity: Entity, removable: bool, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        Add an entity to the resource "storage".
        """
        pass

    def add_entities(self, entities: list[Entity], removable: bool = True,
                     process_execution: Optional[ProcessExecution] = None,  sequence_already_ensured: bool = False):
        """
        Add entities to the resource "storage".
        """

        not_added_entities = []
        for entity in entities:
            if entity != self:
                added = self.add_entity(entity=entity,
                                        removable=removable,
                                        process_execution=process_execution,
                                        sequence_already_ensured=sequence_already_ensured)
            else:
                added = False
            if not added:
                not_added_entities.append(entity)
                if process_execution:
                    process_execution_name = process_execution.get_name()
                else:
                    process_execution_name = None

                raise ValueError(f"[{self.__class__.__name__}] The entity '{entity.name}' cannot added "
                                 f"to the resource '{self.name}' ({self.identification}) \n"
                                 f"{process_execution_name}")

        return not_added_entities

    @abstractmethod
    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        Remove an entity from the resource "storage". Like take a part from stock.
        """
        pass

    def remove_entities(self, entities: list[Entity], process_execution: Optional[ProcessExecution] = None,
                        sequence_already_ensured: bool = False):
        """
        Remove entities from the resource "storage". Like take parts from stock.
        """

        not_removed_entities = []
        for entity in entities:
            removed = self.remove_entity(entity, process_execution, sequence_already_ensured=sequence_already_ensured)
            if not removed:
                not_removed_entities.append(entity)
                # raise ValueError(f"[{self.__class__.__name__}] The entity '{entity}' cannot removed "
                #                  f"from the resource {self.name} {self.identification}")

        return not_removed_entities

    @abstractmethod
    def check_entity_stored(self, entity: Entity) -> bool:
        """Returns True if the entity is stored in the resource else False"""
        pass

    @abstractmethod
    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """
        Returns all available entities (Used, for example, for planning).
        Afterward, an entity can be removed for usage with the remove_entity method.
        """
        pass

    @abstractmethod
    def get_available_entity(self, entity_type: EntityType):
        """
        Returns an available entity (used for example for planning).
        Afterward the entity can be removed for usage with the remove_entity method.
        """
        pass

    @abstractmethod
    def get_possible_entity_types_to_store(self):
        """
        Returns all possible_entities that can be stored in the storage/ resource as a list.
        """
        pass

    @abstractmethod
    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        return False

    @abstractmethod
    def get_available_capacity_entity_type(self, entity_type: EntityType):
        pass

    def block_period(self, start_time, end_time, blocker_name, process_execution_id: int, work_order_id: int,
                     issue_id: int = None, block_before: bool = False):
        """Block a period in the _process_execution_plan"""
        return self.process_execution_plan.block_period(start_time=start_time, end_time=end_time,
                                                        issue_id=issue_id, blocker_name=blocker_name,
                                                        process_execution_id=process_execution_id,
                                                        work_order_id=work_order_id, block_before=block_before)

    def update_period(self, start_time, end_time, process_execution_id: int):
        """Update a period, respectively their start and end time"""

        return self.process_execution_plan.update_period(start_time=start_time, end_time=end_time,
                                                         process_execution_id=process_execution_id)

    def update_period_by_actual(self, start_time, end_time, process_execution_id: int, plan_process_execution_id: int):
        """Update a period that is actually occurred"""

        return self.process_execution_plan.update_period_by_actual(start_time=start_time, end_time=end_time,
                                                                   process_execution_id=process_execution_id,
                                                                   plan_process_execution_id=plan_process_execution_id)

    def unblock_period(self, unblocker_name, process_execution_id):
        """Unlock a period in the _process_execution_plan"""
        return self.process_execution_plan.unblock_period(unblocker_name=unblocker_name,
                                                          process_execution_id=process_execution_id)

    def get_next_possible_period(self, period_length: timedelta, start_time=None, issue_id=None,
                                 last_element: bool = False):
        """Get the next possible period from the _process_execution_plan"""
        return self.process_execution_plan.get_next_possible_period(period_length=period_length,
                                                                    start_time=start_time,
                                                                    issue_id=issue_id,
                                                                    last_element=last_element)

    def get_free_periods_calendar_extract(self, start_time=None, end_time=None, issue_id=None,
                                          time_slot_duration: Optional[np.timedelta64] = None,
                                          long_time_reservation_duration=None):
        """Get the free periods' calendar extract from the _process_execution_plan"""
        if isinstance(self.process_execution_plan, str):
            raise Exception(self.process_execution_plan)
        return self.process_execution_plan.get_free_periods_calendar_extract(
            start_time=start_time, end_time=end_time, issue_id=issue_id,
            time_slot_duration=time_slot_duration, long_time_reservation_duration=long_time_reservation_duration)

    def get_process_execution_plan_copy(self):
        """Get a copy from the process_execution_plan object (used for planning objectives)"""
        return self.process_execution_plan.get_copy()

    def get_utilization(self, start_time, end_time):
        return self.process_execution_plan.get_utilization(start_time, end_time)

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if self.plant is None:
            not_completely_filled_attributes.append("plant")
        if self._process_execution_plan is None:
            not_completely_filled_attributes.append("process_execution_plan")
        if self.costs_per_second is None:
            not_completely_filled_attributes.append("costs_per_second")
        if not isinstance(self.physical_body, PhysicalBody):
            not_completely_filled_attributes.append("physical_body")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class StationaryResource(Resource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 entry_edge: Optional[list[tuple[int, int]]] = None,
                 exit_edge: Optional[list[tuple[int, int]]] = None,
                 efficiency: ProbabilityDistribution = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        StationaryResource are resources that cannot move by themselves.

        Parameters
        ----------
        entry_edge: A list of positions that define a part of the edge of the resource where it can be entered.
        If None, it cannot be entered.
        exit_edge: A list of positions that define a part of the edge of the resource where it can be left.
        If None, it cannot be left.
        efficiency: The efficiency of the resource as a probability distribution between 0 and 1 (the throughput
        time of a process is divided by this factor to get the real time)
        Stacked stationary resources have to be entered/left first by entering the parent and then the children
        resources (both entries/ exits can be on the same place)
        """
        if entry_edge is None:
            entry_edge = []
        if exit_edge is None:
            exit_edge = []
        if efficiency is None:
            efficiency = SingleValueDistribution(1)
        self._efficiency: ProbabilityDistribution = efficiency
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.entry_edge: list[tuple[int, int]] = entry_edge
        self.exit_edge: list[tuple[int, int]] = exit_edge

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        efficiency_parameters = self.get_efficiency_parameters()
        return (f"StationaryResource with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{efficiency_parameters}', '{self.entry_edge}', '{self.exit_edge}'")

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        resource_duplicate = super(StationaryResource, self).duplicate(external_name)
        return resource_duplicate

    # @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        stationary_resource_copy: StationaryResource = super(StationaryResource, self).copy()
        stationary_resource_copy.efficiency = stationary_resource_copy.efficiency.copy()
        stationary_resource_copy.entry_edge = stationary_resource_copy.entry_edge.copy()
        stationary_resource_copy.exit_edge = stationary_resource_copy.exit_edge.copy()

        return stationary_resource_copy

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, efficiency):
        if issubclass(efficiency.__class__, ProbabilityDistribution):
            self._efficiency = efficiency
        else:
            raise ValueError(f"[{self.__class__.__name__}]")

    def change_efficiency(self, efficiency, process_execution: Optional[ProcessExecution] = None,
                          sequence_already_ensured: bool = False):
        self._efficiency = efficiency
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   efficiency=self.efficiency, sequence_already_ensured=sequence_already_ensured)

    def get_efficiency_parameters(self):
        return self.efficiency.get_parameters_with_values()

    def get_expected_performance(self):
        performance = self.efficiency.get_expected_value()
        return performance

    def get_performance(self):
        if isinstance(self.efficiency, ProbabilityDistribution):
            performance = self.efficiency.get_random_number()
        else:
            performance = self.efficiency
        return performance

    # @abstractmethod
    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        Add an entity to the resource "storage".
        """
        pass

    # @abstractmethod
    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        Remove an entity from the resource "storage". Like take a part from stock.
        """
        pass

    # @abstractmethod
    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        pass

    # @abstractmethod
    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """
        Returns an available entity (used, for example, for planning).
        Afterward, the entity can be removed for usage with the remove_entity method.
        """
        pass

    # @abstractmethod
    def get_available_entity(self, entity_type: EntityType):
        """
        Returns an available entity (used, for example, for planning).
        Afterward, the entity can be removed for usage with the remove_entity method.
        """
        pass

    # @abstractmethod
    def get_possible_entity_types_to_store(self):
        """
        Returns all possible_entities that can be stored in the storage/ resource as a list.
        """
        pass

    # @abstractmethod
    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        return False

    # @abstractmethod
    def get_available_capacity_entity_type(self, entity_type: EntityType):
        pass

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self._efficiency, ProbabilityDistribution):
            not_completely_filled_attributes.append("efficiency")
        if self.entry_edge is None:
            not_completely_filled_attributes.append("entry_edge")
        if self.exit_edge is None:
            not_completely_filled_attributes.append("exit_edge")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class Storage(StationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 capacity: int,
                 allowed_entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 entry_edge: Optional[list[tuple[int, int]]] = None,
                 exit_edge: Optional[list[tuple[int, int]]] = None,
                 efficiency: ProbabilityDistribution = None,
                 stored_entities: list[Entity] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Storage is a resource to store exactly one type of entity - part or resource (e.g. buffer_area, bin, palette,
        charging station)

        Parameters
        ----------
        capacity: Amount of entities (resources or parts) that can be stored
        allowed_entity_type: The entity_type that can be stored
        Note: also lower entity types can be stored
        (where the allowed_entity_type is the super_entity_type of the entity)
        stored_entities: List of entities that are currently stored and match the allowed_entity_type
        """
        if entry_edge is None:
            entry_edge = []
        if exit_edge is None:
            exit_edge = []
        if stored_entities is None:
            stored_entities = []
        self.stored_entities: list[Entity] = stored_entities
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         entry_edge=entry_edge, exit_edge=exit_edge, efficiency=efficiency,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.capacity: int = capacity
        self.allowed_entity_type: EntityType = allowed_entity_type

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        efficiency_parameters = self.get_efficiency_parameters()
        stored_entities_length = len(self.stored_entities)
        allowed_entity_type_name = self.get_allowed_entity_type_name()
        return (f"Storage with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{efficiency_parameters}', '{self.entry_edge}', '{self.exit_edge}', "
                f"'{stored_entities_length}', '{self.capacity}', '{allowed_entity_type_name}'")

    def duplicate(self, without_situated_in=True, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        storage_duplicate = super(Storage, self).duplicate(external_name)
        storage_duplicate.stored_entities = [stored_entity.duplicate(external_name)
                                             for stored_entity in storage_duplicate.stored_entities]

        if without_situated_in:
            storage_duplicate.situated_in = None

        return storage_duplicate

    def copy(self):
        """Copy the object with the same identification."""
        storage_copy: Storage = super(StationaryResource, self).copy()
        storage_copy.stored_entities = storage_copy.stored_entities.copy()

        return storage_copy

    def get_allowed_entity_type_name(self):
        if self.allowed_entity_type is not None:
            return self.allowed_entity_type.name
        else:
            return ""

    def get_number_of_stored_entities(self):
        """Return the number of stored entities described through the length of the stored_entities array"""
        number_of_stored_entities = len(self.stored_entities)
        return number_of_stored_entities

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False) -> bool:
        """
        The method is used to store an entity in the storage. In advance, it is checked whether the entity is allowed to
        be stored in the storage.

        Parameters
        ----------
        entity: can be a part or a resource
        process_execution: responsible for the adding

        Returns
        -------
        True if storing-process was successful respectively False if not
        """
        if self.allowed_entity_type.check_entity_type_match_lower(entity.entity_type):
            if entity in self.stored_entities:
                return True

            self.stored_entities.append(entity)
            entity.change_situated_in(situated_in=self, process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)

            if process_execution:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       stored_entities=entity,
                                       change_type="ADD", sequence_already_ensured=sequence_already_ensured)

            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}') "
                      f"is stored in the storage '{self.name}'.")

            return True

        else:
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}') "
                      f"cannot be stored in the storage '{self.name}'")
            return False

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False) -> bool:
        """
        The method is used to take an entity from the storage.
        Note: two ways possible: 1. take_entity_by_entity or 2. take_entity_by_entity_type
        for the moment only the first is possible

        Parameters
        ----------
        process_execution: the process_execution responsible for the removing
        entity: can be a part or a resource

        Returns
        -------
        True if taking-process was successful respectively False if not
        """
        if entity == self:
            # an entity cannot be removed from self
            return False

        for stored_entity in self.stored_entities:
            if entity != stored_entity:
                continue

            self.stored_entities.remove(entity)
            entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)

            if process_execution:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       stored_entities=entity,
                                       change_type="REMOVE", sequence_already_ensured=sequence_already_ensured)

            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}') "
                      f"was taken from the storage '{self.name}'")

            return True

        if prints_visible:
            print(f"[{self.__class__.__name__:20}] "
                  f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                  f" could not be taken from the storage '{self.name}'")
        return False

    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        if entity in self.stored_entities:
            return True
        else:
            return False

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """
        The method is used to find all entities in the storage which have the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity
        at: datetime the available entities are checked

        Returns
        -------
        The entities if available respectively None if not
        """

        if entity_type is not None:
            if not self.allowed_entity_type.check_entity_type_match_lower(entity_type):
                return []

        if at is None:
            available_entities = self.stored_entities
        else:
            available_entities = self._get_available_entities_at(at)

        available_entities = \
            [entity
             for entity in available_entities
             if entity_type.check_entity_type_match_lower(entity.entity_type)]

        return available_entities

    def _get_available_entities_at(self, at: datetime):
        return self.dynamic_attributes.get_attribute_at(req_time_stamp=at,
                                                        attribute="stored_entities")

    def get_incoming_and_outgoing_entities_history(self, start_time_stamp: datetime, end_time_stamp: datetime):
        return self.dynamic_attributes.get_changes_attribute_raw(attribute="stored_entities",
                                                                 start_time_stamp=start_time_stamp,
                                                                 end_time_stamp=end_time_stamp,
                                                                 include_initial_entries=True)

    def get_available_entity(self, entity_type: EntityType, random_=False) -> Optional[Entity]:
        """
        The method is used to find an entity in the storage which has the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity
        random_: True if a random choice is needed else the first element (the oldest) is chosen

        Returns
        -------
        The entity if available respectively None if not
        """
        if (self.stored_entities and
                self.allowed_entity_type.check_entity_type_match_lower(entity_type)):
            if random_:
                available_entity = random.choice(self.stored_entities)
            else:
                available_entity = self.stored_entities[0]

            if not available_entity.entity_type.check_entity_type_match_lower(entity_type):
                # different entities stored in the storage (material and sub materials)
                available_entities = \
                    [entity
                     for entity in self.stored_entities
                     if entity_type.check_entity_type_match_lower(entity.entity_type)]

                if random_:
                    available_entity = random.choice(available_entities)
                else:
                    available_entity = available_entities[0]

            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity_type.name}' is available in the storage '{self.name}'")
            return available_entity
        else:
            # print(f"The entity {entity_type.name} is not available in the storage {self.name}")
            return None

    def get_possible_entity_types_to_store(self) -> list[EntityType]:
        """Return the entity type that can be stored in the Storage as a list"""
        return [self.allowed_entity_type]

    def check_entity_type_storable(self, entity_type: EntityType):
        """Determine if the entity_type can be stored into the resource"""
        if self.allowed_entity_type.check_entity_type_match_lower(entity_type):
            return True
        else:
            return False

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for the entity_type"""
        if self.check_entity_type_storable(entity_type):
            available_capacity_entity_type = self.capacity - len(self.stored_entities)
        else:
            available_capacity_entity_type = 0

        return available_capacity_entity_type

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self.stored_entities, list):
            not_completely_filled_attributes.append("stored_entities")
        if self.capacity is None:
            not_completely_filled_attributes.append("capacity")
        if self.allowed_entity_type is None:
            not_completely_filled_attributes.append("allowed_entity_type")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


def _get_storage_place_entity_type(all_storage_places, entity_type: EntityType) -> Optional[list[Storage]]:
    """Determine storage places with the entity_type"""
    storage_places = None
    if entity_type in all_storage_places:
        storage_places = all_storage_places[entity_type]

        if entity_type.super_entity_type in all_storage_places:
            storage_places += all_storage_places[entity_type.super_entity_type]
            storage_places = list(set(storage_places))

    elif entity_type.super_entity_type in all_storage_places:
        storage_places = all_storage_places[entity_type.super_entity_type]

    return storage_places


class StoragePlaces(Serializable):

    def __init__(self,
                 storage_places: dict[EntityType, list[Storage]] | list[Storage],
                 capacity: int,
                 name: str,
                 situated_in: Optional[Resource] = None):
        """
        They are used to combine all the storages from a resource, like work_station, warehouse or
        non_stationary_resource and provide the same behavior.

        Parameters
        ----------
        storage_places: a dict of storages mapped to the entity_type that is storable in the storage
        capacity: the overall capacity of all the storages - Limiting, never more entities are allowed to be
        in the storage locations than specified by the 'capacity'
        name: the name of the resource used for the prints
        attribute number_of_stored_entities: sum of the stored_entities in the storages
        """

        if isinstance(situated_in, Resource):
            if isinstance(storage_places, str):
                raise Exception(f"Storage Places {name} - {storage_places}")

            if isinstance(storage_places, dict):
                storage_places_list = [storage
                                       for storage_place_lst in list(storage_places.values())
                                       for storage in storage_place_lst]
            else:
                storage_places_list = storage_places.copy()
                self.get_storage_places_dict(storage_places_list)

            if storage_places_list:
                for storage in storage_places_list:
                    if not isinstance(storage, Storage):
                        continue  # case, when the storage is represented by his Digital Twin Object identification
                    if storage.situated_in is None:
                        storage.situated_in = situated_in
        self._storage_places: dict[EntityType, list[Storage]] = storage_places

        self.capacity: int = capacity
        self.number_of_stored_entities = self._determine_number_of_stored_entities()
        self.name: str = name

    def get_storage_places_dict(self, storage_places_list: list[Storage]):
        storage_places = {}
        for storage in storage_places_list:
            if not isinstance(storage, Storage):
                break  # storages aren't set until now
            if not isinstance(storage.allowed_entity_type, EntityType):
                break  # entity types aren't set until now

            storage_places.setdefault(storage.allowed_entity_type,
                                      []).append(storage)
            if storage.allowed_entity_type.super_entity_type is not None:
                storage_places.setdefault(storage.allowed_entity_type.super_entity_type,
                                          []).append(storage)

        return storage_places

    def __str__(self):
        return f"StoragePlaces name {self.name}'; '{self.capacity}', '{self.number_of_stored_entities}'"

    def representation(self):
        """The representation of the object is unambiguous"""
        items = ("%s = %r" % (k, v)
                 for k, v in self.__dict__.items())
        object_representation = "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
        return object_representation


    @property
    def storage_places(self):
        return self._storage_places

    @storage_places.setter
    def storage_places(self, _storage_places):
        if not isinstance(_storage_places, dict):
            if not isinstance(_storage_places, list):
                raise ValueError(f"The storage places should be of type dict ...")

            _storage_places = self.get_storage_places_dict(_storage_places)

        self._storage_places = _storage_places

    def get_storages_lst(self, storage_places=None):
        if storage_places is not None:
            storage_places_ = storage_places
        else:
            storage_places_ = self._storage_places

        storages_lst = [storage
                        for storage_place_lst in list(storage_places_.values())
                        for storage in storage_place_lst]

        return storages_lst

    def _determine_number_of_stored_entities(self):
        try:
            number_of_stored_entities = sum([storage.get_number_of_stored_entities()
                                             for entity_type, storages in self._storage_places.items()
                                             for storage in storages])
        except:
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] NotImplemented: number_of_stored_entities")
            number_of_stored_entities = 0

        return number_of_stored_entities

    def duplicate_for_instantiation(self, without_situated_in=True, external_name=False):
        """
        Used if only the storage place dict is needed that can be used in the common __init__ methods,
        where the storage places are used
        """
        storage_places_duplicate_for_instantiation = \
            {entity_type: [storage.duplicate(without_situated_in, external_name)
                           for storage in storages]
             for entity_type, storages in self._storage_places.items()}
        return storage_places_duplicate_for_instantiation

    def duplicate(self, without_situated_in=True, external_name=False):
        storage_places_attribute = self.duplicate_for_instantiation(without_situated_in, external_name)
        storage_places_duplicate = copy(self)
        storage_places_duplicate._storage_places = storage_places_attribute

        return storage_places_duplicate

    def copy(self):
        storage_places_copy = {entity_type: storages.copy()
                               for entity_type, storages in self._storage_places.items()}
        return storage_places_copy

    def get_storages(self, entity_type=None):
        """Return storage_places if the object has a respective attribute"""
        if entity_type is not None:
            return {entity_type: _get_storage_place_entity_type(self._storage_places, entity_type)}
        else:
            return self._storage_places

    def get_storages_without_entity_types(self, entity_type=None) -> list[Storage]:
        """Return storage_places if the object has a respective attribute"""
        if entity_type is not None:
            storages_without_entity_type = _get_storage_place_entity_type(self._storage_places, entity_type)
            return storages_without_entity_type
        else:
            return self.get_all_storages()

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        Used to store an entity in the storage_places.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        -------
        True if the storing process was successful respectively False if not
        """
        storage_places = _get_storage_place_entity_type(self._storage_places, entity.entity_type)
        if storage_places is None:
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ({entity.get_all_external_identifications()}) "
                      f"cannot be stored in the storage_places '{self.name}'")
            return False

        for storage_place in storage_places:
            stored_entity = storage_place.add_entity(entity=entity,
                                                     removable=removable,
                                                     process_execution=process_execution,
                                                     sequence_already_ensured=sequence_already_ensured)

            if stored_entity:
                self.number_of_stored_entities += 1
                entity.change_situated_in(storage_place, process_execution=process_execution,
                                          sequence_already_ensured=sequence_already_ensured)
                # print(f"The entity {entity.name} is stored in the storage_places {self.name}")
                return True

        debug_str = f"[{self.__class__.__name__}]"
        logging.debug(debug_str)
        raise Exception(debug_str)

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        The method is used to take an entity from the storage_places.
        Note: two ways possible: 1. take_entity_by_entity or 2. take_entity_by_entity_type
        for the moment only the first is possible

        Parameters
        ----------
        entity: can be a part or a resource
        process_execution: responsible for the removal

        Returns
        -------
        True if the taking process was successful respectively False if not
        """
        if isinstance(entity, WorkStation):
            storage_places = entity.buffer_stations
        elif isinstance(entity, Resource) and hasattr(entity, "storage_places"):
            storage_places = entity.storage_places
        else:
            storage_places = entity

        if storage_places == self:
            return False

        storage_places = _get_storage_place_entity_type(self._storage_places, entity.entity_type)
        if storage_places is None:
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}') "
                      f"could not be taken from the storage_places '{self.name}'")

            return False

        for storage_place in storage_places:
            taken_entity = storage_place.remove_entity(entity=entity, process_execution=process_execution,
                                                       sequence_already_ensured=sequence_already_ensured)
            if taken_entity:
                self.number_of_stored_entities -= 1
                entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                          sequence_already_ensured=sequence_already_ensured)
                if prints_visible:
                    print(f"[{self.__class__.__name__:20}] "
                          f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                          f" was taken from the storage_places '{self.name}'")
                return True

        if entity.situated_in:
            taken_entity = entity.situated_in.remove_entity(entity=entity, process_execution=process_execution,
                                                            sequence_already_ensured=sequence_already_ensured)
        else:
            taken_entity = False
        # if taken_entity:
        #     entity.situated_in.number_of_stored_entities -= 1
        entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                  sequence_already_ensured=sequence_already_ensured)
        debug_str = "Entity not taken from storages places"
        logging.debug(debug_str)

    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        storage_places = _get_storage_place_entity_type(self._storage_places, entity.entity_type)
        if storage_places is None:
            return False

        for storage_place in storage_places:
            try:
                if storage_place.check_entity_stored(entity):
                    return True
            except:
                raise Exception(storage_place, self.name)

        return False

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """Return available entities that match the entity_type"""
        if entity_type is not None:
            storage_places = _get_storage_place_entity_type(self._storage_places, entity_type)
        else:
            storage_places = self.get_all_storages()

        if storage_places is None:
            return None

        available_entities = []
        for relevant_storage_place in storage_places:
            available_entities_batch = relevant_storage_place.get_available_entities(entity_type=entity_type, at=at)
            available_entities.extend(available_entities_batch)

        return available_entities

    def get_available_entity(self, entity_type: EntityType):
        """
        Used to find an entity in the storage_places which has the respective entity_type.

        Parameters
        ----------
        entity_type: can be the type of part or a resource

        Returns
        -------
        the entity if available respectively None if not
        """
        storage_places = _get_storage_place_entity_type(self._storage_places, entity_type)
        if storage_places is None:
            return None

        for storage_place in storage_places:
            entity = storage_place.get_available_entity(entity_type)
            if entity:
                if prints_visible:
                    print(f"[{self.__class__.__name__:20}] "
                          f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                          f" is available in the storage_places '{self.name}'")
                return entity

        debug_str = f"[{self.__class__.__name__}]"
        logging.debug(debug_str)
        raise Exception(debug_str)

    def get_possible_entity_types_to_store(self):  # ToDo: memo?
        """Returns a list with possible entity types which can be stored in the storage_places."""
        allowed_entity_types = (
                set(storage_place_entity_type
                    for storage_place_entity_type, storage_places in self._storage_places.items()
                    for storage_place in storage_places
                    for entity_type in storage_place.get_possible_entity_types_to_store()) |
                set(self._storage_places.keys()))

        return allowed_entity_types

    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        storage_places = _get_storage_place_entity_type(self._storage_places, entity_type)
        if storage_places is None:
            return False

        for storage_place in storage_places:
            if storage_place.check_entity_type_storable(entity_type):
                return True

        debug_str = f"[{self.__class__.__name__}]"
        logging.debug(debug_str)
        raise Exception(debug_str)

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for an entity_type based on the information from the buffer stations"""
        storage_places = _get_storage_place_entity_type(self._storage_places, entity_type)
        if storage_places is None:
            return False

        available_capacity_entity_type = 0
        for storage_place in storage_places:
            available_capacity_entity_type += storage_place.get_available_capacity_entity_type(entity_type)

        return available_capacity_entity_type

    def add_storage_place(self, storage):
        """Add a further storage_place to the storage_places"""
        self._storage_places.setdefault(storage.entity_type, []).append(storage)

    def get_sub_instances_to_add(self, sub_digital_twin_objects):

        storage_entity_types = list(self._storage_places.keys())
        if storage_entity_types:
            sub_digital_twin_objects.setdefault(EntityType, []).extend(storage_entity_types)

        all_storages = self.get_all_storages()
        if all_storages:
            sub_digital_twin_objects.setdefault(Storage, []).extend(all_storages)

        return sub_digital_twin_objects

    def get_all_storages(self):
        storages_nested = list(self._storage_places.values())
        if storages_nested:
            storages = reduce(concat, storages_nested)
        else:
            storages = []

        return storages


class WorkStation(StationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 entry_edge: Optional[list[tuple[int, int]]] = None,
                 exit_edge: Optional[list[tuple[int, int]]] = None,
                 efficiency: ProbabilityDistribution = None,
                 buffer_stations: dict[EntityType, list[Storage]] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 capacity: int = 1,  # maybe None could be for unlimited capacity
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Work stations are used to by an assembly process to add parts to a product.

        Parameters
        ----------
        buffer_stations: The buffer stations where parts and products are buffered for transport or usage.
        The selection is done by allowed_entity_type of the storage
        dict[EntityType (from the entities that can be stored in the storages in the value), list[Storage]]
        """
        if entry_edge is None:
            entry_edge = []
        if exit_edge is None:
            exit_edge = []
        if buffer_stations is None:
            buffer_stations = {}
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         entry_edge=entry_edge, exit_edge=exit_edge, efficiency=efficiency,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        self._buffer_stations: StoragePlaces = StoragePlaces(storage_places=buffer_stations,
                                                             capacity=capacity,
                                                             name=self.name,
                                                             situated_in=self)

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        efficiency_parameters = self.get_efficiency_parameters()
        return (f"WorkStation with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{efficiency_parameters}', '{self.entry_edge}', '{self.exit_edge}'")

    @property
    def buffer_stations(self) -> list[Storage]:
        if isinstance(self._buffer_stations, StoragePlaces):
            storages: list[Storage] = self._buffer_stations.get_storages_without_entity_types()
        else:
            storages = []

        return storages

    @buffer_stations.setter
    def buffer_stations(self, buffer_stations):
        self._buffer_stations.storage_places = buffer_stations

    @property
    def capacity(self):
        return self._buffer_stations.capacity

    @capacity.setter
    def capacity(self, capacity):
        self._buffer_stations.capacity = capacity

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        work_station_duplicate = super(WorkStation, self).duplicate(external_name)
        work_station_duplicate._buffer_stations = (
            work_station_duplicate._buffer_stations.duplicate(external_name))
        return work_station_duplicate

    def copy(self):
        """Copy the object with the same identification."""
        work_station_copy: WorkStation = super(WorkStation, self).copy()
        work_station_copy.buffer_stations = work_station_copy.buffer_stations.copy()

        return work_station_copy

    def change_position_initially(self):
        pass  # as change_position

    def get_storages(self, entity_type=None):
        """Return buffer_stations or storage_places associated with entity_types
        if the object has a respective attribute"""
        return self._buffer_stations.get_storages(entity_type)

    def get_storages_without_entity_types(self, entity_type=None):
        """Return buffer_stations or storage_places if the object has a respective attribute"""
        return self._buffer_stations.get_storages_without_entity_types(entity_type)

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        The method is used to store an entity in the work_station.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        -------
        True if the storing process was successful respectively False if not
        """
        return self._buffer_stations.add_entity(entity=entity,
                                                removable=removable,
                                                process_execution=process_execution,
                                                sequence_already_ensured=sequence_already_ensured)

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        The method is used to take an entity from the work_station.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        -------
        True if the taking process was successful respectively False if not
        """
        return self._buffer_stations.remove_entity(entity, process_execution, sequence_already_ensured)

    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        return self._buffer_stations.check_entity_stored(entity)

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        return self._buffer_stations.get_available_entities(entity_type=entity_type, at=at)

    def get_available_entity(self, entity_type: EntityType):
        """
        The method is used to find an entity in the work_station which has the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity

        Returns
        -------
        the entity if available respectively None if not
        """
        return self._buffer_stations.get_available_entity(entity_type)

    def get_possible_entity_types_to_store(self):
        """Returns a list with possible entity types which can be stored in the WorkStation."""
        return self._buffer_stations.get_possible_entity_types_to_store()

    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        return self._buffer_stations.check_entity_type_storable(entity_type)

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for an entity_type based on the information from the buffer stations"""
        return self._buffer_stations.get_available_capacity_entity_type(entity_type)

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self._buffer_stations, StoragePlaces):
            not_completely_filled_attributes.append("buffer_stations")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class Warehouse(StationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 entry_edge: Optional[list[tuple[int, int]]] = None,
                 exit_edge: Optional[list[tuple[int, int]]] = None,
                 efficiency: ProbabilityDistribution = None,
                 storage_places: dict[EntityType, list[Storage]] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 capacity: int = 1,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Warehouses administrate several storage places with entities

        Parameters
        ----------
        storage_places: the places where the entities are stored
        dict[EntityType (from the entities that can be stored in the storages in the value), list[Storage]]
        """
        if entry_edge is None:
            entry_edge = []
        if exit_edge is None:
            exit_edge = []
        if storage_places is None:
            storage_places = {}
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         entry_edge=entry_edge, exit_edge=exit_edge, efficiency=efficiency,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self._storage_places: StoragePlaces = StoragePlaces(storage_places=storage_places, capacity=capacity,
                                                            name=self.name, situated_in=self)

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        efficiency_parameters = self.get_efficiency_parameters()
        return (f"Warehouse with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{efficiency_parameters}', '{self.entry_edge}', '{self.exit_edge}'")

    @property
    def storage_places(self) -> list[Storage]:
        if isinstance(self._storage_places, StoragePlaces):
            storages: list[Storage] = self._storage_places.get_storages_without_entity_types()
        else:
            storages = []

        return storages

    @storage_places.setter
    def storage_places(self, storage_places):
        self._storage_places.storage_places = storage_places

    @property
    def capacity(self):
        return self._storage_places.capacity

    @capacity.setter
    def capacity(self, capacity):
        self._storage_places.capacity = capacity

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        warehouse_station_duplicate = super(Warehouse, self).duplicate(external_name)
        warehouse_station_duplicate.storage_places = \
            warehouse_station_duplicate.buffer_stations.duplicate(external_name)
        return warehouse_station_duplicate

    def copy(self):
        """Copy the object with the same identification."""
        warehouse_copy: Warehouse = super(Warehouse, self).copy()
        warehouse_copy._storage_places = warehouse_copy._storage_places.copy()

        return warehouse_copy

    def get_storages(self, entity_type=None):
        """Return buffer_stations or storage_places associated with entity_types
        if the object has a respective attribute"""
        return self._storage_places.get_storages(entity_type)

    def get_storages_without_entity_types(self, entity_type=None):
        """Return buffer_stations or storage_places if the object has a respective attribute"""
        return self._storage_places.get_storages_without_entity_types(entity_type)

    def add_storage_place(self, storage):
        """Add a storage_place to the warehouse"""
        self._storage_places.add_storage_place(storage)

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        The method is used to store an entity in the warehouse.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        -------
        True if storing process was successful respectively False if not
        """
        return self._storage_places.add_entity(entity=entity,
                                               removable=removable,
                                               process_execution=process_execution,
                                               sequence_already_ensured=sequence_already_ensured)

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        The method is used to take an entity from the warehouse.

        Parameters
        ----------
        entity: Can be a part or a resource

        Returns
        -------
        True if the taking process was successful respectively False if not
        """
        return self._storage_places.remove_entity(entity, process_execution, sequence_already_ensured)

    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        return self._storage_places.check_entity_stored(entity)

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """Get the available entities stored in the storage_places of the warehouse"""
        return self._storage_places.get_available_entities(entity_type=entity_type, at=at)

    def get_available_entity(self, entity_type: EntityType):
        """
        The method is used to find an entity in the warehouse which has the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity

        Returns
        -------
        The entity if available respectively False if not
        """
        return self._storage_places.get_available_entity(entity_type)

    def get_possible_entity_types_to_store(self):
        """Returns a list with possible entity types which can be stored in the Warehouse."""
        return self._storage_places.get_possible_entity_types_to_store()

    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        return self._storage_places.check_entity_type_storable(entity_type)

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for an entity_type based on the information from the storage places"""
        return self._storage_places.get_available_capacity_entity_type(entity_type)

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self._storage_places, StoragePlaces):
            not_completely_filled_attributes.append("storage_places")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class ConveyorBelt(StationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 allowed_entity_types: list[EntityType],
                 origin: Storage,
                 destination: Storage,
                 conveyor_length: float,
                 capacity: int = 1,
                 plant: Optional[Plant] = None,
                 entities_on_transport: list[Entity] = None,
                 flow_direction: int = 1,
                 costs_per_second: float = 0,
                 entry_edge: Optional[list[tuple[int, int]]] = None,
                 exit_edge: Optional[list[tuple[int, int]]] = None,
                 efficiency: ProbabilityDistribution = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlanConveyorBelt] = None,
                 situated_in: Optional[Resource] = None,
                 pitch: float = 1,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Conveyor belt is a stationary resource that transport entities (part or resource) from one Storage (origin) to
        another (destination). This means a complex conveyor belt with more than one possible path is split into
        conveyor belts linked through storage_places that built switches.

        Parameters
        ----------
        capacity: Amounts of entities (resources or parts) that can be transported
        allowed_entity_types: The entity_types that can be stored
        entities_on_transport: List of entities that are currently transported
        (the last element of the list is always at the beginning of the conveyor belt)
        flow_direction: the direction of the conveyor belt (material) flow (-1 or 1 (origin to destination))
        origin: a list of possible predecessor resources
        destination: a list of possible successor resources
        conveyor_length: the length of the conveyor_belt
        pitch: distance between the two pitches/ back sides of the objects on the conveyor belt.
        Furthermore, it determines the max_length of an entity drive over the conveyor_belt

        Maybe later relevant
        - accumulating: determines if the entities on transport can accumulate if
        - different time_intervals/ pitch possible, if the conveyor_belt speed could be different
        """
        self.flow_direction = flow_direction
        if entities_on_transport is None:
            entities_on_transport = []
        self.entities_on_transport: list[Entity] = entities_on_transport
        if entry_edge is None:
            entry_edge = []
        if exit_edge is None:
            exit_edge = []
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         entry_edge=entry_edge, exit_edge=exit_edge, efficiency=efficiency,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.capacity: int = capacity
        self.allowed_entity_types: list[EntityType] = allowed_entity_types
        self.origin: Storage = origin
        self.destination: Storage = destination
        self.conveyor_length: float = conveyor_length
        self.pitch: float = pitch

        if self.process_execution_plan is not None:
            if self.process_execution_plan.__class__.__name__ == "ProcessExecutionPlanConveyorBelt":
                self.process_execution_plan: ProcessExecutionPlanConveyorBelt
                time_interval = self.pitch  # / self._efficiency.get_expected_value()
                self.process_execution_plan.set_time_interval(time_interval)
        # self.accumulating

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        efficiency_parameters = self.get_efficiency_parameters()
        return (f"ConveyorBelt with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{efficiency_parameters}', '{self.entry_edge}', '{self.exit_edge}'")

    # @property
    # def process_execution_plan(self):
    #     return self.process_execution_plan
    #
    # @process_execution_plan.setter
    # def process_execution_plan(self, process_execution_plan):
    #     # process_execution_plan from the origin resource
    #     self.process_execution_plan = process_execution_plan

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        conveyor_belt_duplicate = super(ConveyorBelt, self).duplicate(external_name)
        conveyor_belt_duplicate.entities_on_transport = []
        return conveyor_belt_duplicate

    def copy(self):
        """Copy the object with the same identification."""
        conveyor_belt_copy: ConveyorBelt = super(ConveyorBelt, self).copy()
        conveyor_belt_copy.entities_on_transport = conveyor_belt_copy.entities_on_transport.copy()
        conveyor_belt_copy.allowed_entity_types = conveyor_belt_copy.allowed_entity_types.copy()

        return conveyor_belt_copy

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False) -> bool:
        """
        The method is used to store an entity in the non_stationary_resource. In advance, it is checked
        whether the entity is allowed to be stored in the non_stationary_resource.

        Parameters
        ----------
        entity: can be a part or a resource
        process_execution: process_execution that is responsible for the adding

        Returns
        -------
        True if the storing process was successful respectively False if not
        """
        # if len(self.entities_on_transport) == self.capacity:
        #     print(f"The entity {entity.name} cannot be stored in the storage because of capacity {self.name}")
        #     return False

        for allowed_entity_type in self.allowed_entity_types:
            if not allowed_entity_type.check_entity_type_match_lower(entity.entity_type):
                continue

            self.entities_on_transport.append(entity)
            entity.change_situated_in(situated_in=self, process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)

            if process_execution is not None:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       entities_on_transport=entity,
                                       change_type="ADD", sequence_already_ensured=sequence_already_ensured)

            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                      f" is stored in the conveyor belt '{self.name}'")

            return True

        if prints_visible:
            print(f"[{self.__class__.__name__:20}] "
                  f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                  f" cannot be stored in the conveyor belt '{self.name}'")
        return False

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False) -> bool:
        """
        The method is used to take out an entity from the non_stationary_resource.

        Parameters
        ----------
        entity: can be a part or a resource
        process_execution: process_execution that is responsible for the removing

        Returns
        -------
        True if the taking process was successful respectively False if not
        """

        if entity in self.entities_on_transport:
            self.entities_on_transport.remove(entity)

            entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)  # ToDo: necessary?

            if process_execution is not None:
                self.update_attributes(process_execution=process_execution,
                                       current_time=process_execution.executed_end_time,
                                       entities_on_transport=entity,
                                       change_type="REMOVE",
                                       sequence_already_ensured=sequence_already_ensured)

            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                      f" was taken from the conveyor belt '{self.name}'")
            successful = True

        else:
            entity.change_situated_in(situated_in=None, process_execution=process_execution,
                                      sequence_already_ensured=sequence_already_ensured)  # ToDo: necessary?
            if prints_visible:
                print(f"[{self.__class__.__name__:20}] "
                      f"The entity '{entity.name}' ('{entity.get_all_external_identifications()}')"
                      f"could not be taken from the conveyor belt '{self.name}'")
            successful = False

        return successful

    def check_entity_stored(self, entity):
        """See descriptions Resource"""
        if entity in self.entities_on_transport:
            return True
        else:
            return False

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """
        Not used for the ConveyorBelt.
        """
        pass

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        """
        The method is used to find all entities in the storage which have the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity
        at: datetime the available entities are checked

        Returns
        -------
        The entities if available respectively None if not
        """

        if entity_type is not None:
            allowed = [entity_type.check_entity_type_match_lower(allowed_entity_type)
                       for allowed_entity_type in self.allowed_entity_types]
            if not allowed:
                return []

        if at is None:
            return self.entities_on_transport
        else:
            return self._get_available_entities_at(at)

    def _get_available_entities_at(self, at: datetime):
        return self.dynamic_attributes.get_attribute_at(req_time_stamp=at,
                                                        attribute="entities_on_transport")

    def get_incoming_and_outgoing_entities_history(self, start_time_stamp: datetime, end_time_stamp: datetime):
        return self.dynamic_attributes.get_changes_attribute_raw(attribute="entities_on_transport",
                                                                 start_time_stamp=start_time_stamp,
                                                                 end_time_stamp=end_time_stamp,
                                                                 include_initial_entries=True)

    def get_available_entity(self, entity_type: EntityType):
        """
        Not used for the ConveyorBelt.
        """
        pass

    def get_possible_entity_types_to_store(self) -> list[EntityType]:
        """Returns a list with possible entity types to store"""
        return self.allowed_entity_types

    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        for allowed_entity_type in self.allowed_entity_types:
            if allowed_entity_type.check_entity_type_match_lower(entity_type):
                return True

        return False

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for an entity_type"""
        if entity_type not in self.allowed_entity_types:
            return 0

        available_capacity_entity_type = self.capacity - len(self.entities_on_transport)

        return available_capacity_entity_type

    def get_process_execution_plan_copy(self):
        """Get a copy from the process_execution_plan object (used for planning objectives)"""
        return self._process_execution_plan.get_copy()

    def get_utilization(self, start_time, end_time):
        return self._process_execution_plan.get_utilization(start_time, end_time)

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        # Further implementation needed ...

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class NonStationaryResource(Resource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 orientation: float = 0,
                 storage_places: dict[EntityType, list[Storage]] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 capacity: int = 1,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        NonStationaryResource are resources that can move from one StationaryResource to another. They can transport
        various entity types but only one type at a time

        Parameters
        ----------
        orientation: Orientation in degrees between '-180' and '180'. '0' is north
        storage_places: the places where the entities are stored
        dict[EntityType (from the entities that can be stored in the storages in the value), list[Storage]]
        """
        if storage_places is None:
            storage_places = {}
        self._orientation = orientation
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)

        self._storage_places: StoragePlaces = StoragePlaces(storage_places=storage_places, capacity=capacity,
                                                            name=self.name, situated_in=self)

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        return (f"NonStationaryResource with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}'")

    def duplicate(self, external_name=False):
        """The duplicate of an entity has different to the duplicate of other digital_twin objects also different
        duplicated objects in the attributes"""
        non_stationary_resource_duplicate = super(NonStationaryResource, self).duplicate(external_name)
        non_stationary_resource_duplicate._storage_places = \
            non_stationary_resource_duplicate._storage_places.duplicate(external_name)

        return non_stationary_resource_duplicate

    def get_sub_instances_to_add(self):
        """Used after duplication to add also the sub_instances to the digital_twin"""
        sub_digital_twin_objects = super(NonStationaryResource, self).get_sub_instances_to_add()
        sub_digital_twin_objects = self._storage_places.get_sub_instances_to_add(sub_digital_twin_objects)
        return sub_digital_twin_objects

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        non_stationary_resource_copy = super(NonStationaryResource, self).copy()
        non_stationary_resource_copy.storage_places = self._storage_places.copy()

        return non_stationary_resource_copy

    @property
    def storage_places(self) -> list[Storage]:
        if isinstance(self._storage_places, StoragePlaces):
            storages: list[Storage] = self._storage_places.get_storages_without_entity_types()
        else:
            storages = []

        return storages

    @storage_places.setter
    def storage_places(self, storage_places):
        self._storage_places.storage_places = storage_places

    @property
    def capacity(self):
        return self._storage_places.capacity

    @capacity.setter
    def capacity(self, capacity):
        self._storage_places.capacity = capacity

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation, process_execution: Optional[ProcessExecution] = None):
        if -180 <= orientation <= 180:
            self._orientation = orientation
        else:
            raise ValueError(f"[{self.__class__.__name__}] The orientation is not in range (-180, 180)")

    def get_expected_performance(self) -> float:
        return 1

    def get_performance(self) -> float:
        return 1

    def change_orientation(self, orientation, process_execution: Optional[ProcessExecution] = None,
                           sequence_already_ensured: bool = False):
        self._orientation = orientation
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   efficiency=self.orientation,
                                   sequence_already_ensured=sequence_already_ensured)

    def change_position(self, new_position: (int, int), process_execution: Optional[ProcessExecution] = None,
                        sequence_already_ensured: bool = False):
        """Change the position of the resource"""
        self._physical_body.change_position(new_position, process_execution,
                                            sequence_already_ensured=sequence_already_ensured)

    def get_storages(self, entity_type=None):
        """Return buffer_stations or storage_places associated with entity_types
        if the object has a respective attribute"""
        return self._storage_places.get_storages(entity_type)

    def get_storages_without_entity_types(self, entity_type=None):
        """Return buffer_stations or storage_places if the object has a respective attribute"""
        return self._storage_places.get_storages_without_entity_types(entity_type)

    def add_entity(self, entity: Entity, removable: bool = True, process_execution: Optional[ProcessExecution] = None,
                   sequence_already_ensured: bool = False):
        """
        The method is used to store an entity in the non_stationary_resource.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        ----------
        True if storing process was successful respectively False if not
        """
        return self._storage_places.add_entity(entity=entity,
                                               removable=removable,
                                               process_execution=process_execution,
                                               sequence_already_ensured=sequence_already_ensured)

    def remove_entity(self, entity: Entity, process_execution: Optional[ProcessExecution] = None,
                      sequence_already_ensured: bool = False):
        """
        The method is used to take an entity from the non_stationary_resource.

        Parameters
        ----------
        entity: can be a part or a resource

        Returns
        -------
        True if a taking process was successful respectively False if not
        """
        return self._storage_places.remove_entity(entity, process_execution, sequence_already_ensured)

    def check_entity_stored(self, entity: Entity) -> bool:
        """See descriptions Resource"""
        return self._storage_places.check_entity_stored(entity)

    def get_available_entities(self, entity_type: Optional[EntityType], at: Optional[datetime] = None):
        return self._storage_places.get_available_entities(entity_type=entity_type, at=at)

    def get_available_entity(self, entity_type: EntityType):
        """
        The method is used to find an entity in the warehouse which has the respective entity_type.

        Parameters
        ----------
        entity_type: specify an entity

        Returns
        -------
        the entity if available respectively False if not
        """
        return self._storage_places.get_available_entity(entity_type)

    def get_possible_entity_types_to_store(self):
        """Returns a list with possible entity types which can be stored in the NonStationaryResource."""
        return self._storage_places.get_possible_entity_types_to_store()

    def check_entity_type_storable(self, entity_type):
        """Determine if the entity_type can be stored into the resource"""
        return self._storage_places.check_entity_type_storable(entity_type)

    def get_available_capacity_entity_type(self, entity_type: EntityType):
        """Get the available capacity for an entity_type based on the information from the storage_places stations"""
        return self._storage_places.get_available_capacity_entity_type(entity_type)

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if self.orientation is None:
            not_completely_filled_attributes.append("orientation")
        if not isinstance(self._storage_places, StoragePlaces):
            not_completely_filled_attributes.append("storage_places")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class ActiveMovingResource(NonStationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 costs_per_second: float = 0,
                 orientation: float = 0,
                 speed: float | ProbabilityDistribution = SingleValueDistribution(1),  # ToDo: should only be a probability distribution
                 energy_consumption: float = 0,
                 energy_capacity: float = 0,
                 energy_level: float = 0,
                 storage_places: dict[EntityType, list[Storage]] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        NonStationaryResource are resources that can move from one StationaryResource to another. They can transport
        various entity types but only one type at a time

        Parameters
        ----------
        speed: Speed of the resource in m/s
        energy_consumption: Energy consumption in units/s
        energy_capacity: Max. energy capacity in units
        energy_level: Current energy level in units
        """
        if storage_places is None:
            storage_places = {}
        self._energy_level: float = energy_level
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body, orientation=orientation, storage_places=storage_places,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.speed: float | ProbabilityDistribution = speed
        self.energy_consumption: float = energy_consumption
        self.energy_capacity: float = energy_capacity

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        return (f"ActiveMovingResource with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', "
                f"'{self.speed}', '{self.energy_consumption}', '{self.energy_capacity}', '{self.energy_level}'")

    def copy(self):
        """Copy the object with the same identification."""
        active_moving_resource_copy = super(ActiveMovingResource, self).copy()

        return active_moving_resource_copy

    @property
    def energy_level(self):
        return self._energy_level

    @energy_level.setter
    def energy_level(self, energy_level):
        self._energy_level = energy_level

    def change_energy_level(self, energy_level, process_execution: Optional[ProcessExecution] = None,
                            sequence_already_ensured: bool = False):
        self.energy_level = energy_level
        if process_execution:
            self.update_attributes(process_execution=process_execution,
                                   current_time=process_execution.executed_end_time,
                                   energy_level=self.energy_level, sequence_already_ensured=sequence_already_ensured)

    def get_expected_performance(self):
        if isinstance(self.speed, ProbabilityDistribution):
            performance = self.speed.get_expected_value()
        else:
            performance = self.speed
        return performance

    def get_performance(self):
        if isinstance(self.speed, ProbabilityDistribution):
            performance = self.speed.get_random_number()
        else:
            performance = self.speed
        return performance

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self._energy_level, float):
            not_completely_filled_attributes.append("energy_level")
        if not isinstance(self.speed, float):
            not_completely_filled_attributes.append("speed")
        if not isinstance(self.energy_consumption, float):
            not_completely_filled_attributes.append("energy_consumption")
        if not isinstance(self.energy_capacity, float):
            not_completely_filled_attributes.append("energy_capacity")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes


class PassiveMovingResource(NonStationaryResource):

    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 plant: Optional[Plant] = None,
                 orientation: float = 0,
                 storage_places: dict[EntityType, list[Storage]] = None,
                 costs_per_second: Optional[float] = None,
                 position: Optional[tuple[int, int]] = None,
                 length: Optional[int] = None,
                 width: Optional[int] = None,
                 physical_body: Optional[PhysicalBody] = None,
                 service_life: Optional[int] = None,
                 process_execution_plan: Optional[ProcessExecutionPlan] = None,
                 situated_in: Optional[Resource] = None,
                 quality: float = 1,
                 inspected_quality: Optional[float] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: Optional[datetime] = datetime(1970, 1, 1),
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        PassiveMovingResource is a NonStationaryResource that is used to support a transport process. It cannot move by
        itself, but it can carry parts. These can be married to each other to enable consistent transport.
        For this purpose, these PassiveMovingResources are married to the ActiveMovingResources.
        Example apps_tech: Bin, Box, Palette

        Parameters
        ----------
        service_life: lifetime of a PassiveMovingResource (number of usages) - if not used None (not mandatory)
        """
        if storage_places is None:
            storage_places = {}
        super().__init__(identification=identification, name=name, entity_type=entity_type, plant=plant,
                         costs_per_second=costs_per_second, position=position, length=length, width=width,
                         physical_body=physical_body, orientation=orientation, storage_places=storage_places,
                         process_execution_plan=process_execution_plan, situated_in=situated_in, quality=quality,
                         inspected_quality=inspected_quality,
                         process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.service_life: Optional[int] = service_life

    def __str__(self):
        entity_type_name = self.get_entity_type_name()
        plant_name = self.get_plant_name()
        situated_in_name = self.get_situated_in_name()
        position = self.get_position()
        width = self.get_width()
        length = self.get_length()
        return (f"PassiveMovingResource with ID '{self.identification}' and name {self.name}'; '{entity_type_name}', "
                f"'{situated_in_name}', '{self._quality}', '{self._inspected_quality}', '{plant_name}', "
                f"'{self.costs_per_second}', '{position}', '{width}', '{length}', '{self.service_life}'")

    def copy(self):
        """Copy the object with the same identification."""
        passive_moving_resource_copy = super(PassiveMovingResource, self).copy()

        return passive_moving_resource_copy

    def completely_filled(self):

        completely_filled, not_completely_filled_attributes = super().completely_filled()

        if not isinstance(self.service_life, float):
            not_completely_filled_attributes.append("service_life")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes
