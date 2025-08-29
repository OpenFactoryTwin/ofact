"""
# TODO Add Module Description
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from copy import copy
from random import random, seed
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.probabilities import NormalDistribution

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import EntityType, Entity, Resource, Part

seed(42)


class StorageReservation:
    """
    Reservation of parts is also possible through the _process_execution_plan - but difficult
    """

    def __init__(self, resource: Resource, max_reservation_duration):
        """
        The reservation manager is used to reserve entities.
        :param resource: the relational object for that the reservation is managed.
        :param max_reservation_duration: the maximal time period a reservation can be done
        """
        self.resource = resource
        self.max_reservation_duration = max_reservation_duration
        self._round_available_entities = {}
        self._round_offered_parts = {}

        self._entity_reservation_df: pd.DataFrame = \
            pd.DataFrame({"Entity Type ID": pd.Series([], dtype=np.dtype("int32")),
                          "Entity ID": pd.Series([], dtype=np.dtype("int32")),
                          "Issue ID": pd.Series([], dtype=np.dtype("int32")),
                          "Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
                          "End": pd.Series([], dtype=np.dtype("datetime64[ns]"))})

    def get_unreserved_entities_at(self, entity_type: EntityType, demand_number: int = 1, time_stamp=None, round_=None,
                                   entities=None):
        """Get unreserved entities at a specific time_stamp
        - normally all entities with expired reservations are available ToDo"""

        if round_ not in self._round_available_entities:
            available_entities = self.resource.get_available_entities(entity_type=entity_type)
            self._round_available_entities = {round_: {entity_type: available_entities}}
        elif entity_type not in self._round_available_entities[round_]:
            available_entities = self.resource.get_available_entities(entity_type=entity_type)
            self._round_available_entities[round_][entity_type] = available_entities
        else:
            available_entities = self._round_available_entities[round_][entity_type]

        if entities is not None:
            available_entities = list(set(available_entities).intersection(entities))

        if round_ not in self._round_offered_parts:
            self._round_offered_parts = {round_: set()}

        if self._round_offered_parts[round_]:
            available_entities = list(set(available_entities).difference(self._round_offered_parts[round_]))

        unreserved_entities = []
        if available_entities:
            unreserved_entities = self._check_reservation(entities=available_entities)

        new_reserved_parts = unreserved_entities[:min(len(unreserved_entities), demand_number)]
        # provide different parts
        self._round_offered_parts[round_] = self._round_offered_parts[round_].union(set(new_reserved_parts))

        new_reserved_parts = [(available_entity, time_stamp) for available_entity in new_reserved_parts]
        return new_reserved_parts

    def add_entity_reservations(self, entities: list[Entity], time_stamp, end_time=None,
                                process_execution_ids=None, issue_ids=None, long_time_reservation=False):

        new_reservations = \
            [{"Entity Type ID": entity.entity_type.identification,
              "Entity ID": entity.identification,
              "Process Execution ID": np.nan,
              "Issue ID": np.nan,
              "End": np.nan}
             for entity in entities]

        self._entity_reservation_df = pd.concat([self._entity_reservation_df,
                                                 pd.DataFrame(new_reservations)])

    def add_entity_reservation(self, entity: Entity, time_stamp, end_time=None, reservation_duration=None,
                               process_execution_id=None, issue_id=None, long_time_reservation=False):

        new_reservations = \
            [{"Entity Type ID": entity.entity_type.identification,
              "Entity ID": entity.identification,
              "Process Execution ID": (process_execution_id if process_execution_id is not None else np.nan),
              "Issue ID": (issue_id if issue_id is not None else np.nan),
              "End": (end_time if end_time is not None else np.nan)}]

        self._entity_reservation_df = pd.concat([self._entity_reservation_df,
                                                 pd.DataFrame(new_reservations)])

    def remove_reservations(self, entities: list[Entity], process_execution_id):
        """Removed if entities are removed from the storage or the plan has changed"""

        entity_ids = [entity.identification for entity in entities]

        self._entity_reservation_df = (
            self._entity_reservation_df.loc[~self._entity_reservation_df["Entity ID"].isin(entity_ids)])

    def _check_reservation(self, entities: list[Entity], time_stamp=None) -> (list[Entity], list[Entity]):
        """Split a list of parts into reserved parts and unreserved parts"""
        part_ids = [entity.identification for entity in entities]
        reserved_entity_ids = \
            self._entity_reservation_df.loc[
                self._entity_reservation_df["Entity ID"].isin(part_ids)]["Entity ID"].to_list()

        unreserved_entities = [entity for entity in entities
                               if entity.identification not in reserved_entity_ids]

        return unreserved_entities

    def change_reservation(self):
        # ToDo: needed?
        raise NotImplementedError


class PartUnavailabilityStorageReservation(StorageReservation):

    def __init__(self, resource: Resource, max_reservation_duration, part_availability_probability=0.95,
                 part_delay_time_mue=60, part_delay_time_sigma=10, hidden=False):
        super(PartUnavailabilityStorageReservation, self).__init__(resource, max_reservation_duration)

        self.part_availability_probability = part_availability_probability
        self.part_delay_time = NormalDistribution(mue=part_delay_time_mue, sigma=part_delay_time_sigma)

        self.hidden = hidden
        stored_entities_nested = [storage.stored_entities
                                  for storages in list(self.resource.get_storages().values())
                                  for storage in storages]
        self.parts_with_shift_times: dict[Part, np.datetime64] = \
            {stored_entity: (np.timedelta64(int(self.part_delay_time.get_random_number()), "s")
                             if random() > self.part_availability_probability else np.timedelta64(0, "s"))
             for stored_entities in stored_entities_nested
             for stored_entity in stored_entities}
        self.parts_with_time_stamps: dict[Part, np.datetime64] = {}

    def get_unreserved_entities_at(self, entity_type: EntityType, demand_number: int = 1, time_stamp=None, round_=None):
        """Get unreserved entities at a specific time_stamp
        - normally all entities with expired reservations are available ToDo"""
        if round_ not in self._round_available_entities:
            available_entities = self.resource.get_available_entities(entity_type=entity_type)
            self._round_available_entities = {round_: {entity_type: available_entities}}
        elif entity_type not in self._round_available_entities[round_]:
            available_entities = self.resource.get_available_entities(entity_type=entity_type)
            self._round_available_entities[round_][entity_type] = available_entities
        else:
            available_entities = self._round_available_entities[round_][entity_type]

        if round_ not in self._round_offered_parts:
            self.parts_with_time_stamps = {}  # reset
            self._round_offered_parts = {round_: set()}

        if self._round_offered_parts[round_]:
            available_entities = list(set(available_entities).difference(self._round_offered_parts[round_]))

        unreserved_entities = []
        if available_entities:
            unreserved_entities = self._check_reservation(entities=available_entities)

        new_reserved_parts = unreserved_entities[:min(len(unreserved_entities), demand_number)]
        # provide different parts
        self._round_offered_parts[round_] = self._round_offered_parts[round_].union(set(new_reserved_parts))

        new_reserved_parts_time_stamp = []

        for available_entity in new_reserved_parts:
            if available_entity not in self.parts_with_time_stamps:
                new_time_stamp = copy(time_stamp)
                if available_entity in self.parts_with_shift_times:
                    new_time_stamp += self.parts_with_shift_times[available_entity]
                elif random() > self.part_availability_probability:
                    new_time_stamp += np.timedelta64(int(self.part_delay_time.get_random_number()), "s")
                self.parts_with_time_stamps[available_entity] = new_time_stamp

            if self.hidden:
                new_reserved_parts_time_stamp.append((available_entity, time_stamp))

            else:
                new_reserved_parts_time_stamp.append((available_entity, self.parts_with_time_stamps[available_entity]))

        return new_reserved_parts_time_stamp

    def check_part_availability(self, parts):
        reserved_parts_time_stamp = []
        for part in parts:
            if self.hidden:
                reserved_parts_time_stamp.append(self.parts_with_time_stamps[part])
                # the real time_stamps are revealed

        return reserved_parts_time_stamp
