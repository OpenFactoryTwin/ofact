from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import copy
from enum import Enum
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING, Dict, List, Union, Optional

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

from ofact.twin.agent_control.behaviours.planning.tree.helpers import get_overlaps_periods
from ofact.twin.state_model.entities import Resource, Part, EntityType
from ofact.twin.state_model.processes import ProcessExecution

# from dsplot.config import config
# from dsplot.graph import Graph

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import StationaryResource
    from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference

LinkTypes = Enum('LinkTypes', 'LOOSE FIXED', qualname='ProcessExecutionsPath.LINK_TYPES')


class ProcessExecutionsComponent(metaclass=ABCMeta):
    next_node_part_identification = 0

    class Type(Enum):
        STANDARD = "STANDARD"
        CONNECTOR = "CONNECTOR"

    @classmethod
    def get_node_part_identification(cls):
        node_part_identification = cls.next_node_part_identification
        cls.next_node_part_identification += 1
        return node_part_identification

    def __init__(self, goal: ProcessExecution | EntityType = None,
                 origin=None, destination=None,
                 goal_item=None, process_executions_components_parent=None, process_executions_components=None,
                 issue_id=None, process_execution_id=None,
                 reference_preference: None | ProcessExecutionPreference = None,
                 node_identification=None, cfp_path: list[tuple] = None, type_=Type.STANDARD):
        """
        A logical path of process_executions_components (e.g. transport that passes stations) that contains at minimum
        one process_execution and reach goals. The explicit participation to these goals is done by
        the goal_item. Furthermore, for the price building the respective preferences are stored.
        :param node_identification: identifier of the node
        :param goal: goal that can be achieved by the execution of the process_executions_path
        :param goal_item: Used to reference to the objects the process_execution_path is created for
        :param process_executions_components_parent: refers the process_executions_component that includes self
        as children (process_executions_components)
        :param process_executions_components: process_executions_components in the right order to execute (children)
        :param reference_preference: process_executions_preference of the reference_object
        :param type_: determines if the component is necessarily needed in the planning or "only" needed
        for connection (connection is for the moment the transport/ drive of resources to the location of demand)
        """

        self.cfp_path = cfp_path
        if node_identification is None:
            node_identification = ProcessExecutionsComponent.get_node_part_identification()
        self.node_identification = node_identification

        # currently for simplicity only one goal considered
        self.goal: Union[ProcessExecution, EntityType] = goal
        self.goal_item: Union[Part, Resource, ProcessExecution] = goal_item

        self.origin: Optional[StationaryResource] = origin
        self.destination: Optional[StationaryResource] = destination

        self.process_executions_components_parent: Union[ProcessExecutionsPath, ProcessExecutionsVariant] = \
            process_executions_components_parent

        if process_executions_components is None:
            process_executions_components = {}
        self._add_process_executions_component_parent(process_executions_components)
        self.process_executions_components: (
            Dict[Union[Part, Resource, ProcessExecution],
            List[Union[ProcessExecutionsPath, ProcessExecution, ProcessExecutionsVariant]]]) = (
            process_executions_components)

        self.reference_preference: Optional[ProcessExecutionPreference] = reference_preference

        self.process_execution_id = process_execution_id
        self.issue_id = issue_id
        # path ids are used to determine in which paths (for example, transport paths)
        # the process_execution_component is usable -> currently it is used in the scheduling to avoid
        # not possible combinations in advance - can also be decided in through a parent search
        self.path_ids = set()

        self.type = type_

        self._price = None

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, price):
        if price is None:
            self._price = price
            return

        self._price = copy(price)

    @abstractmethod
    def get_final_time_slot(self):
        pass

    def get_process_executions_components_lst(self):

        process_executions_components_nested = list(self.process_executions_components.values())
        if process_executions_components_nested:
            process_executions_components_lst = reduce(concat, process_executions_components_nested)
        else:
            process_executions_components_lst = []

        return process_executions_components_lst

    def add_path_ids(self, path_id):
        """Add the path id to the children"""
        self.path_ids.add(path_id)
        for process_execution_component in self.get_process_executions_components_lst():
            process_execution_component.add_path_ids(path_id)

    def merge_horizontal(self, preferences_before, connector_object_entity_type=None, shift_time_needed=False):
        if isinstance(self, ProcessExecutionsPath):
            path_link_type = LinkTypes.FIXED
        elif isinstance(self, ProcessExecutionsVariant):
            path_link_type = LinkTypes.LOOSE
        else:
            raise NotImplementedError

        shift_time = None
        if shift_time_needed and self.process_executions_components:
            last_accepted_time_periods_adapted = \
                self.process_executions_components[-1].reference_preference.get_accepted_time_periods()
            if last_accepted_time_periods_adapted is not None:
                shift_time = self.process_executions_components[-1].reference_preference.get_shift_time()

        self.reference_preference.merge_horizontal(preferences_before=preferences_before,
                                                   connector_object_entity_type=connector_object_entity_type,
                                                   path_link_type=path_link_type,
                                                   shift_time=shift_time)

    def feasible(self):
        """Check if the execution is possible"""
        feasible = self.reference_preference.feasible()
        return feasible

    def replace_process_executions_components(self, process_executions_components):
        """Replace the process_executions_components because they are filled before with dummy values"""

        self.process_executions_components = process_executions_components
        self._add_process_executions_component_parent(process_executions_components)

    def add_process_executions_components(self, process_executions_components):
        """Add children/ process_executions_components and therefore also add the parent relationship"""
        self.process_executions_components |= process_executions_components

        self._add_process_executions_component_parent(process_executions_components)

    def _add_process_executions_component_parent(self, process_executions_components):
        """Add a process execution component (self) as a parent in the process_executions_components"""
        for goal, process_executions_components_lst in process_executions_components.items():
            for process_executions_component in process_executions_components_lst:
                process_executions_component.process_executions_components_parent = self

    def replace_node_pre_path(self, new_path):
        """Used to create a new alternative based on further subtrees given"""
        new_path_idx = [idx for idx, (cfp_id, alternative_id) in enumerate(self.node_identification)
                        if cfp_id == new_path[-1][0]][0]
        self.node_identification = new_path[:new_path_idx + 1] + self.node_identification[new_path_idx + 1:]

        for goal, process_executions_component_lst in self.process_executions_components:
            for process_executions_component in process_executions_component_lst:
                if isinstance(process_executions_component, ProcessExecutionsComponent):
                    process_executions_component.replace_node_pre_path(new_path)

    def get_preference(self):
        """Return the reference_preference of the process_executions_component"""
        return self.reference_preference

    def get_process_executions_components_preferences(self):
        """Get the process_executions_components_preferences for all process_executions_components"""
        process_executions_components_preferences = \
            [process_executions_component.get_preference()
             for process_executions_component in self.get_process_executions_components_lst()]

        return process_executions_components_preferences

    def get_preference_process_executions_component(self, process_executions_component):
        """Get the preference for a process_executions_component/ children"""
        preference_process_executions_component = None
        for process_executions_component_self in self.get_process_executions_components_lst():
            if process_executions_component_self == process_executions_component:
                preference_process_executions_component = process_executions_component.get_preference()
                break

        return preference_process_executions_component

    def get_preferences_with_durations(self) -> (pd.DataFrame, int, float):
        """:return: preferences, duration, support_time"""

        entity_type = self.get_entity_type()
        return self.reference_preference.get_preferences_with_durations(entity_type=entity_type)

    def get_origin(self):
        """Get the origin stored in the preference"""
        return self.reference_preference.origin

    def get_destination(self):
        """Get the destination stored in the preference"""
        return self.reference_preference.destination

    def get_entity_type(self):
        """The entity_type relates to the goal reached by the process_executions_component"""
        if isinstance(self.goal, EntityType):
            return self.goal
        else:
            return None

    def get_entity_used(self):
        """The goal item is the entity used to reach the goal"""
        entity_used = self.goal_item
        return entity_used

    def check_time_slot_accepted(self, time_slot):
        """Check if a time slot be in the accepted time or outside"""
        accepted_time_periods = self.get_accepted_time_periods()

        len_accepted_time_periods = accepted_time_periods.shape[0]
        start_times = np.repeat(time_slot[0], len_accepted_time_periods)
        end_times = np.repeat(time_slot[1], len_accepted_time_periods)
        mask = (accepted_time_periods[:, 0] <= start_times) & (start_times <= accepted_time_periods[:, 1]) & \
               (accepted_time_periods[:, 0] <= end_times) & (end_times <= accepted_time_periods[:, 1])
        end_time_accepted_time_periods = accepted_time_periods[mask]

        if not end_time_accepted_time_periods.any():
            return False
        return True

    def get_accepted_time_periods(self):
        """Return the accepted_time_period for the execution of the process_execution_component"""
        return self.reference_preference.get_accepted_time_periods()

    def duplicate(self):
        duplicate = self.copy()
        duplicate.node_identification = ProcessExecutionsComponent.get_node_part_identification()

        return duplicate

    @abstractmethod
    def copy(self):
        """
        much faster than deepcopy and the objects (among others the digital_twin objects) will have the same references
        but attributes with type dict etc. must be copied a second time if not interdependencies with the original
        attribute are possible
        """
        process_executions_variant_copy = copy(self)
        process_executions_variant_copy.process_executions_components = \
            {goal: process_execution_components_lst.copy()
             for goal, process_execution_components_lst in self.process_executions_components.items()}
        process_executions_variant_copy.process_executions_components_parent = self.process_executions_components_parent
        if self.price is not None:
            process_executions_variant_copy.price = self.price.copy()
        process_executions_variant_copy.reference_preference = self.reference_preference.get_copy()
        process_executions_variant_copy.goal = self.goal
        process_executions_variant_copy.goal_item = self.goal_item
        process_executions_variant_copy.cfp_path = copy(self.cfp_path)
        process_executions_variant_copy.path_ids = self.path_ids.copy()

        return process_executions_variant_copy

    def get_process_execution_time(self):
        return self.reference_preference.get_process_execution_time()

    def get_price(self):
        # Only collection of existing price (calculated by get_price)
        self._trigger_price_determination()

        price = self.price

        return price

    def _trigger_price_determination(self):
        """Determine if the price should be calculated"""

        change_occurred = False
        changes_occurred = [process_executions_component._trigger_price_determination()
                            for process_executions_component in self.get_process_executions_components_lst()]
        if sum(changes_occurred):
            self._determine_price()
            change_occurred = True

        elif self.price is None:
            self._determine_price()
            change_occurred = True

        elif self.reference_preference.accepted_time_periods_changed:
            self._determine_price()
            self.reference_preference.accepted_time_periods_changed = False
            change_occurred = True

        return change_occurred

    @abstractmethod
    def _determine_price(self):
        pass

    def update_process_execution_id(self, new_process_execution_id):
        self.process_execution_id = new_process_execution_id

    def _visualize(self):
        """Create and plt a visualization of the tree"""
        raise NotImplementedError("UPDATE needed")

        graph_dict = {self._get_node_description(): [branch._get_node_description()
                                                     for branch in self.process_executions_components]}
        frontier_nodes = self.process_executions_components
        while True:
            new_frontier_nodes = []
            for branch in frontier_nodes:
                new_frontier_nodes.extend(branch.process_executions_components)
                graph_dict.setdefault(branch._get_node_description(),
                                      []).extend([branch._get_node_description()
                                                  for branch in branch.process_executions_components])

            if not new_frontier_nodes:
                break
            frontier_nodes = copy(new_frontier_nodes)

        # graph = Graph(graph_dict, directed=False)
        # graph.plot(orientation=config.TREE_ORIENTATION,  # shape=config.LEAF_SHAPE,
        #            output_path='./debugging_images/component_tree.png')  # can be found in the project folder

    def _get_node_description(self):
        accepted_time_periods = np.array2string(self.reference_preference.get_accepted_time_periods(), precision=2,
                                                separator=',', suppress_small=True)
        return str(self.node_identification) + " \n" + accepted_time_periods


class ProcessExecutionsVariant(ProcessExecutionsComponent):

    def __init__(self, issue_id=None, process_execution_id=None, goal=None,
                 process_executions_components_parent=None, process_executions_components=None,
                 origin=None, destination=None,
                 precondition_process_executions_components=None,
                 goal_item=None, reference_preference=None,
                 node_identification=None, cfp_path=None, type_=ProcessExecutionsComponent.Type.STANDARD):
        """
        Combines parallel process_executions_components
        example given: assembly of a bicycle frame that needs different resources like assembly station, employee, etc.
        :param reference_preference: the preference is related to the goal_item
        - because ProcessExecutionsVariant has always a ProcessExecutionsPath for a ProcessExecution as reference_object
            the reference_preference is related to a ProcessExecution
        :param precondition_process_executions_components: in a process_executions_variant also the material supply
        for the process is considered. Because the material supply should be taken place before the ProcessExecution
        starts, they should be handled before the start. Therefore, they are also found in this list
        (additional to the process_executions_components list).
        """
        super(ProcessExecutionsVariant, self).__init__(
            goal=goal, goal_item=goal_item, process_executions_components=process_executions_components,
            origin=origin, destination=destination,
            process_executions_components_parent=process_executions_components_parent,
            reference_preference=reference_preference, issue_id=issue_id, process_execution_id=process_execution_id,
            node_identification=node_identification, cfp_path=cfp_path, type_=type_)

        if precondition_process_executions_components is None:
            precondition_process_executions_components = []
        self.precondition_process_executions_components = precondition_process_executions_components
        self.predecessors = []

    def copy(self):
        """Returns a copy of the process_executions_variant"""

        process_executions_variant_copy = super(ProcessExecutionsVariant, self).copy()
        process_executions_variant_copy.precondition_process_executions_components = \
            copy(self.precondition_process_executions_components)
        process_executions_variant_copy.predecessors = copy(self.predecessors)
        return process_executions_variant_copy

    def add_precondition_process_executions_components(self, proces_executions_components):
        """Add the process_executions_components and assign them as preconditions"""

        self.add_process_executions_components(proces_executions_components)
        precondition_process_executions_components = reduce(concat, list(proces_executions_components.values()))
        self.precondition_process_executions_components.extend(precondition_process_executions_components)

    def get_partial_order(self):
        """Return the partial order of the process_executions_variant
        (the precondition_process_executions_components should be executed
        before the main_process_executions_components)"""

        precondition_process_executions_components = self.precondition_process_executions_components
        process_executions_components = reduce(concat, list(self.process_executions_components.values()))
        main_process_executions_components = \
            set(process_executions_components).difference(precondition_process_executions_components)

        return precondition_process_executions_components, main_process_executions_components

    def get_final_time_slot(self):
        final_time_slot = self.reference_preference.accepted_time_periods_adapted[0]
        return final_time_slot

    def update(self):
        """
        Used to update the process_executions_component because a part of the process_executions_component is changed
        Loosely connected process_executions_components are used only used as time_restrictions because they only need
        to be executed before.
        """
        accepted_time_periods = []
        time_period_restrictions = []
        for process_executions_component in self.get_process_executions_components_lst():
            if process_executions_component.path_link_type == ProcessExecutionsPath.LINK_TYPES.FIXED:
                accepted_time_period = process_executions_component.get_accepted_time_periods()
                accepted_time_periods.append(accepted_time_period)
            elif process_executions_component.path_link_type == ProcessExecutionsPath.LINK_TYPES.LOOSE:
                accepted_time_period = process_executions_component.get_accepted_time_periods()
                if not accepted_time_period.size:
                    raise Exception
                    self.reference_preference.accepted_time_periods = np.array([])
                    return
                time_period_restrictions.append(accepted_time_period[0][0])
            else:
                raise Exception

        if not accepted_time_periods:
            raise NotImplementedError

        # print("Accepted_time_periods: ", accepted_time_periods)
        last_accepted_time_period = accepted_time_periods[0]
        for idx in range(1, len(accepted_time_periods)):
            last_accepted_time_period = \
                get_overlaps_periods(last_accepted_time_period, accepted_time_periods[idx],
                                     min_overlapped=self.reference_preference.expected_process_execution_time)  # ToDo:self.get_expected_process_execution_time()

        if time_period_restrictions:
            time_period_restriction = max(time_period_restrictions)
            last_accepted_time_period = \
                last_accepted_time_period[last_accepted_time_period[:, 1] >= time_period_restriction]

            if last_accepted_time_period.size:
                last_accepted_time_period[0, 0] = time_period_restriction

        self.reference_preference.accepted_time_periods = last_accepted_time_period

    def get_origin(self):
        """Get the origin stored in the preference"""
        return self.origin

    def get_destination(self):
        """Get the destination stored in the preference"""
        return self.destination

    def update_process_execution(self, updated_process_execution):
        raise NotImplementedError
        # updated = self.main_process_executions_path.update_process_execution(updated_process_execution)
        if updated:
            return

        for process_execution_path in self.process_executions_components:  # ToDOó: update needed
            updated = process_execution_path.update_process_execution(
                updated_process_execution=updated_process_execution)

            if updated:
                return

        raise NotImplementedError

    def get_goal(self) -> ProcessExecution | EntityType:
        return self.goal

    def update_connector(self, new_preference, process_execution):
        """Update the connector if the connector should be updated in the scheduling process (possible reasons:
        origin or destination have been changed)"""

        self.goal_item = process_execution
        self.goal = process_execution
        self.reference_preference = new_preference

        # create process_executions_components
        self.process_executions_components = self._adapt_process_executions_components_connector()

    def _adapt_process_executions_components_connector(self) -> list[ProcessExecutionsComponent]:
        """adapt process_executions_components because the connector is changed"""
        adapted_process_executions_components = {}
        for goal, process_execution_components_lst in self.process_executions_components.items():
            for process_executions_component in process_execution_components_lst:
                process_execution_id = self.goal.identification
                process_executions_component.process_execution_id = process_execution_id
                process_executions_component.reference_preference = self.reference_preference
                adapted_process_executions_components.setdefault(goal, []).append(process_executions_component)

        return adapted_process_executions_components

    def _determine_price(self):
        """Combine the prices of sub_components by finding the right inner join for a time_slot"""

        sub_components_price = self._get_prices_sub_components()

        # durations_without_none_fixed = [duration for duration in path_durations_fixed if duration is not None]
        # max_duration_fixed = max(durations_without_none_fixed)
        #
        # path_prices_combined = pd.concat(path_prices_values_fixed, axis=1, join='inner').sum(axis=1)
        #
        # if path_prices_combined.size == 0:
        #     raise Exception
        #
        # if path_prices_values_loose:
        #     self._merge_prices_loose_components(path_prices_combined, path_prices_values_loose)
        #
        # accepted_time_periods = self.reference_preference.get_accepted_time_periods()
        # # ToDo: check not only the start and end time
        #
        # path_prices_combined_ = np.tile(path_prices_combined.index, (accepted_time_periods.shape[0], 1))
        # mask = ((accepted_time_periods[:, 0][:, None] <= path_prices_combined_) &
        #         (path_prices_combined_ <= accepted_time_periods[:, 1][:, None])).max(axis=0)
        #
        # path_prices_combined = path_prices_combined.loc[mask]
        #
        # if path_prices_combined.size == 0:
        #     raise Exception

        # if path_prices_combined.index[-1] < self.reference_preference.accepted_time_periods[-1][-1]:
        #     raise Exception

        self.price = sub_components_price

    def _get_prices_sub_components(self):
        """Determine the prices of the sub_components"""

        path_price = [process_executions_component.get_price()
                      for process_executions_component in self.get_process_executions_components_lst()]

        sub_components_price = sum(path_price)

        return sub_components_price

    def _merge_prices_loose_components(self, path_prices_combined, path_prices_values_loose):
        """if loose it should be ensured that the components before are handled before and influences the price"""

        prices_values_loose = pd.concat(path_prices_values_loose, axis=1)
        for column_idx, idx_min_ in enumerate(prices_values_loose.idxmin()):
            prices_values_loose[column_idx].loc[prices_values_loose[column_idx].index >= idx_min_] = \
                prices_values_loose.loc[idx_min_, column_idx]
        prices_values_loose = prices_values_loose.sum(axis=1)
        first_valid_index = prices_values_loose.first_valid_index()
        prices_values_loose = prices_values_loose.loc[prices_values_loose.index >= first_valid_index]
        path_prices_combined = \
            path_prices_combined.loc[path_prices_combined.index >= first_valid_index]

        if path_prices_combined.index[-1] <= prices_values_loose.index[0]:
            # take the values from df
            df = prices_values_loose[path_prices_combined.index[0]: path_prices_combined.index[-1]]

        else:
            # extension needed
            df1 = prices_values_loose[path_prices_combined.index[0]: prices_values_loose.index[-1]]
            times_extension = pd.date_range(start=prices_values_loose.index[-1] + pd.Timedelta(1, "s"),
                                            end=path_prices_combined.index[-1], freq='S')
            values_extension = np.repeat(prices_values_loose[-1], times_extension.shape[0])
            df2 = pd.DataFrame(values_extension, index=times_extension)
            df = pd.concat([df1, df2], axis=0)
        path_prices_combined = pd.concat([path_prices_combined, df], axis=1).sum(axis=1)

        return path_prices_combined

    def get_process_execution(self, time_slots=True) -> ProcessExecution | None:
        """Get the process_executions defined in the process_executions_component
        :param time_slots: determine if the time_slots are set for the process_executions"""
        if isinstance(self.goal_item, ProcessExecution):
            process_execution = self.goal_item
            if time_slots:
                if self.reference_preference.accepted_time_periods_adapted is None:
                    print("Exception case:", self.reference_preference.accepted_time_periods_adapted,
                          process_execution.get_name())
                    print("Sub nodes:", [(component.node_identification, component.status)
                                         for component in self.get_process_executions_components_lst()])
                    print("PE time slot needed:", process_execution.identification, process_execution.origin)
                    print("Node ID:", self.node_identification, self.cfp_path)
                # else:
                #     print("No exception case", process_execution.process.name)

                process_execution.executed_start_time, process_execution.executed_end_time = \
                    self.reference_preference.accepted_time_periods_adapted[0]
                for process_execution_component in self.get_process_executions_components_lst():
                    process_execution_component.update_process_execution_id(process_execution.identification)

            return process_execution

        else:
            return None


class ProcessExecutionsPath(ProcessExecutionsComponent):
    LINK_TYPES = LinkTypes

    def __init__(self, goal: ProcessExecution | EntityType, goal_item, origin=None, destination=None,
                 process_executions_components_parent=None, process_executions_components=None,
                 reference_preference: None | ProcessExecutionPreference = None, issue_id=None,
                 process_execution_id=None, connector_objects=None, path_link_type: LINK_TYPES = LINK_TYPES.FIXED,
                 node_identification=None, cfp_path=None, type_=ProcessExecutionsComponent.Type.STANDARD):
        """
        :param process_executions_components: process_executions_components are ordered in the right sequence
        :param path_link_type: the path_type describes the connection of process_execution_components
        :param connector_objects: the connector object is used to relate ProcessExecutionsComponents in a path with each
        e.g. an agv for a transport process chain connects the access, loading, transport and the eventual unloading
        """
        super(ProcessExecutionsPath, self).__init__(
            goal=goal, goal_item=goal_item, process_executions_components=process_executions_components,
            origin=origin, destination=destination,
            process_executions_components_parent=process_executions_components_parent,
            reference_preference=reference_preference, issue_id=issue_id, process_execution_id=process_execution_id,
            node_identification=node_identification, cfp_path=cfp_path, type_=type_)

        self.path_link_type = path_link_type
        self._connector_objects: None | Resource = connector_objects
        self.price = None

    @property
    def connector_objects(self):
        return self._connector_objects

    @connector_objects.setter
    def connector_objects(self, connector_objects):
        self._connector_objects = connector_objects

    def copy(self):
        process_executions_path_copy = super(ProcessExecutionsPath, self).copy()
        process_executions_path_copy.connector_objects = self.connector_objects.copy()

        return process_executions_path_copy

    def get_final_time_slot(self):
        # try:
        final_time_slot = self.process_executions_components_parent.reference_preference.accepted_time_periods_adapted[0]
        # except:
        #     print("Goal item: ", self.status, self.process_executions_components_parent.goal_item.process.name,
        #           self.process_executions_components_parent.goal_item.process.identification)
        return final_time_slot

    def get_goal(self) -> ProcessExecution | EntityType:
        return self.goal

    def update_accepted_time_period_children(self):
        pass
        # if self.path_link_type != ProcessExecutionsPath.LINK_TYPES.FIXED:
        #     return
        # for process_executions_component in self.process_executions_components:
        #     process_executions_component
        # ToDo: maybe later needed or a overhead

    def update_accepted_time_period(self, accepted_time_period):
        self.reference_preference.accepted_time_periods = accepted_time_period

    def create_connector(self, preference, process_execution):
        """Create a connector"""
        # ToDo: if possible use the methods from the request behaviours
        issue_id = self.issue_id
        connector_process_executions_path = \
            ProcessExecutionsPath(issue_id=issue_id,
                                  process_execution_id=process_execution.identification,
                                  goal=self.goal,
                                  goal_item=self.goal_item,
                                  reference_preference=preference,
                                  cfp_path=self.cfp_path, type_=ProcessExecutionsComponent.Type.CONNECTOR)

        connector_process_executions_paths = {process_execution: [connector_process_executions_path]}
        connector_process_executions_variant = \
            ProcessExecutionsVariant(issue_id=issue_id,
                                     process_execution_id=process_execution.identification,
                                     goal=process_execution,
                                     goal_item=process_execution,
                                     reference_preference=preference,
                                     process_executions_components=connector_process_executions_paths,
                                     cfp_path=self.cfp_path, type_=ProcessExecutionsComponent.Type.CONNECTOR)

        self.process_executions_components.setdefault(process_execution, []).append(
            connector_process_executions_variant)

        return connector_process_executions_variant, connector_process_executions_path

    def update(self):
        """Used to update the process_executions_component because a part of the process_executions_component
        is changed"""

        print("Obsolete/ Not Implemented")

        self.merge_horizontal(preferences_before=self.get_preferences_before(),
                              connector_object_entity_type=self.connector_objects, shift_time_needed=True)

        # self.reference_preference.merge_horizontal(
        #     preferences_before=self.get_preferences_before(),
        #     connector_object=self.connector_object, path_link_type=self.path_link_type)

    def get_preferences_before(self):
        preferences_before = [process_executions_component.get_preference()
                              for goal, process_executions_components_lst in self.process_executions_components
                              for process_executions_component in process_executions_components_lst]
        return preferences_before

    def update_preference(self, updated_preference):
        self.reference_preference = updated_preference

    def _determine_price(self):
        """Determine the price over a time_period
        Firstly the prices for sub_components are determined after that the prices for the component itself
        """
        # the price is determined over period

        sub_component_price = self.get_sub_component_prices()

        if isinstance(self.goal_item, Part):
            costs, duration = self._get_price_part(sub_component_price)

            # if costs.index[-1] < self.reference_preference.accepted_time_periods[-1][-1]:
            #     print
            self.price = costs

            return costs, duration

        if not isinstance(self.get_entity_used(), Resource):
            self.price = sub_component_price

        preferences, duration, support_time = self.get_preferences_with_durations()

        # lead_time and follow_up_time are included into the factor for the reason of simplicity
        if duration is not None:
            cost_object = self.get_cost_object()
            cost_factor = (((support_time + 1e-12) / (duration + 1e-12)) + 1) * cost_object
        else:
            raise NotImplementedError

        costs = self.get_costs_by_preferences(preferences, duration, factor=cost_factor)

        whole_costs = sub_component_price + costs

        self.price = whole_costs

    def get_sub_component_prices(self):
        """Determine the prices for sub_components"""

        sub_component_prices = \
            [process_executions_component.get_price()
             for process_executions_component in self.get_process_executions_components_lst()
             if not isinstance(process_executions_component, ProcessExecution)]

        sub_component_price = sum(sub_component_prices)

        return sub_component_price

    def _get_price_part(self, sub_component_price):
        """Get the price for the component if the component is a part component"""

        preferences, duration, support_time = self.get_preferences_with_durations()
        costs = self.get_costs_by_preferences(preferences, duration, factor=1)
        sub_component_price + costs

        return costs, duration

    def get_cost_object(self):
        if isinstance(self.goal_item, Resource):
            reference_object_costs = [self.goal_item.costs_per_second]
        else:
            reference_object_costs = []

        if len(reference_object_costs) != 1 and not isinstance(self.goal_items[0], Part):
            return 1
        if len(reference_object_costs) != 1 and not isinstance(self.goal_item, Part):
            raise NotImplementedError("Selection needed")

        cost_object = reference_object_costs[0]

        return cost_object

    def get_costs_by_preferences(self, preferences, duration, factor=1.0):
        """
        not preferred time_stamps (preference value 0) have the cost multiplicator 2
        on the other extreme - preferred  time_stamps have a multiplicator of 1 (no additional costs)
        :param factor: the additional factor is used for lead_time and follow_up_time as heuristic
        """
        costs = preferences["Value"] * (-1) + 2
        cost = int((costs * factor).sum())
        return cost
