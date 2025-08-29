"""
Provides a simple scheduler based on a tree-based heuristic approach that find only simple solutions through
a forward search.
Nodes:
- SchedulingNode: Is used to construct the tree
- HeuristicSchedulingNode: expand the Scheduling node by heuristics
Tree:
- SchedulingTree: represents the tree
- HeuristicSchedulingTree: used to schedule
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import json
import logging
from copy import copy
from datetime import datetime
from enum import Enum
from functools import reduce
from operator import concat
from random import choice
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.negotiation.objects import ResourceCallForProposal
from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import \
    ProcessExecutionsComponent, ProcessExecutionsVariant, ProcessExecutionsPath
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.twin.state_model.entities import Resource, NonStationaryResource, StationaryResource, Part, \
    ActiveMovingResource
from ofact.twin.state_model.processes import ProcessExecution

# from dsplot.config import config
# from dsplot.graph import Graph

if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.agent_control.planning_services.routing import RoutingService

logger = logging.getLogger("SchedulingTree")

tree_visualization = False


def nan_value(possible_nan_value):
    """Check if the value is a nan value"""

    if possible_nan_value == possible_nan_value:
        return False
    else:
        return True


class SchedulingNode:
    class Type(Enum):
        ROOT = "ROOT"
        AND = "AND"
        OR = "OR"
        LEAF = "LEAF"

    def __init__(self, identification=None, root=None, branches=None, leave=None, predecessors=None, successors=None,
                 connector=None, type_=None, level=0, component=None):
        """
        A node in a scheduling tree connected to other nodes.
        :param identification: a unique identifier
        :param root: the parent node of itself
        :param branches: children of itself (only filled if the type_ is "ROOT", "AND" or "OR")
        :param leave: a process_executions_component (only filled if the type_ is "LEAF")
        :param predecessors: a list of nodes that should be executed before self
        :param successors: a list of nodes that should be executed after self
        :param type_: specifies the function of the node
        :attribute time_slot: the time slot is only appended to leave nodes
        :attribute connector_object_constraint: constraints the choice of leave nodes in the scheduling process
        """
        self.identification: None | int = identification
        self.type: None | SchedulingNode.Type = type_
        self.level: int = level

        # vertical connections
        self.root: None | SchedulingNode = root
        self.branches = branches if branches is not None else []
        self.leave: ProcessExecutionsComponent = leave
        self.connector: SchedulingNode = connector

        # horizontal connections
        self.predecessors = predecessors if predecessors is not None else []
        self.successors = successors if successors is not None else []

        self.connector_object_constraint = None  # possibly more than one constraint needed
        self.time_slot = None
        self.process_execution_id = None
        self.component = component
        self.new = False

    def add_level(self, level):
        self.level = level

    def add_process_execution_id(self, process_execution_id):
        self.process_execution_id = process_execution_id

    def get_connector_leave_nodes(self):
        """Get the leave nodes of the connector nodes"""
        connector_leave_nodes = self.get_leafs()
        return connector_leave_nodes

    def add_time_slot(self, start_time, end_time):
        """Add a time slot to a leave_node"""

        if self.type != SchedulingNode.Type.LEAF:
            raise Exception(f"A time slot can only be added to a node of the type 'LEAF', "
                            f"but the type of the node is '{self.type}'!")

        self.time_slot = (start_time, end_time)

    def get_earliest_start_time(self):
        """Get the earliest start time from the node itself or its children"""

        if self.type == SchedulingNode.Type.LEAF:
            if not self.time_slot:
                return None, np.nan
            else:
                return self, self.time_slot[0]

        if not self.time_slot:
            earliest_start_times = \
                [branch.get_earliest_start_time() for branch in self.branches
                 if not nan_value(branch.get_earliest_start_time()[1])]

            if earliest_start_times:
                earliest_start_time = min(earliest_start_times, key=lambda t: t[1])
                return earliest_start_time

        return None, np.nan

    def get_latest_end_time(self):
        """Get the latest end time from the node itself or its children"""

        if self.type == SchedulingNode.Type.LEAF:
            if not self.time_slot:
                return None, np.nan
            else:
                return self, self.time_slot[1]

        if not self.time_slot:
            latest_end_times = \
                [branch.get_latest_end_time() for branch in self.branches
                 if not nan_value(branch.get_latest_end_time()[1])]
            if not latest_end_times:  # because of OR branches in between
                latest_end_times = \
                    [sub_branch.get_latest_end_time() for branch in self.branches
                     for sub_branch in branch.branches
                     if not nan_value(sub_branch.get_latest_end_time()[1])]

            if latest_end_times:
                latest_end_time = max(latest_end_times, key=lambda t: t[1])
                return latest_end_time

        return None, np.nan

    def add_connector_object_constraint(self, connector_object_constraint):
        """Add a connector object constraint to a node to restrict the leave_node choice if the connector obejct can be
        found in the leaf_nodes of a 'successor node'"""
        if self.type == SchedulingNode.Type.LEAF:
            return  # no constraint in the leave nodes

        self.connector_object_constraint = connector_object_constraint

        for branch in self.branches:
            branch.add_connector_object_constraint(connector_object_constraint)

    def remove_connector_object_constraint(self):
        self.connector_object_constraint = None

        for branch in self.branches:
            branch.add_connector_object_constraint(None)

    def connect_nodes(self, parent_node, predecessors: None | list = None):
        """Connect a children node (self) with a parent node"""

        self.add_parent_node(parent_node)
        parent_node.add_children_node(self)

        if predecessors is not None:
            self.add_predecessors(predecessors)
            parent_node.add_predecessors(predecessors)

    def add_predecessors(self, predecessors):
        """Add predecessors/ nodes that should be executed before execution"""

        for predecessor in predecessors:
            if predecessor not in self.predecessors:
                self.predecessors.append(predecessor)
                predecessor.add_successor(self)

    def add_successor(self, successor):
        """Add a successor that should be executed after execution of the node (self)"""

        if successor not in self.successors:
            self.successors.append(successor)

    def add_children_node(self, children_node):
        """Add a children node to branches (self)"""

        if self.type == SchedulingNode.Type.LEAF:
            raise Exception("")

        if children_node not in self.branches:
            self.branches.append(children_node)

    def remove_branches(self):
        """Remove all branches from the node"""
        self.branches = []

    def remove_branch(self, branch_to_remove):
        """Remove a specific branch from branches"""
        self.branches = [branch
                         for branch in self.branches
                         if branch != branch_to_remove]

        if branch_to_remove.branches is None:
            return

        for sub_branch in branch_to_remove.branches:
            branch_to_remove.remove_branch(sub_branch)

    def add_parent_node(self, parent_node):
        """Add a parent node to root (self)"""

        if parent_node.type == SchedulingNode.Type.LEAF:
            self.root = parent_node.root
        else:
            if self == parent_node:
                return
            if self.root:
                if self.root != parent_node:
                    return

            self.root = parent_node

    def get_sub_nodes(self):
        """Return a list of all sub nodes"""

        return self.branches

    def get_leafs(self):
        """Return a list of all leave nodes reachable from the node self"""
        frontier_nodes = [self]
        leafs = []
        while frontier_nodes:
            new_frontier_nodes = []
            for frontier_node in frontier_nodes:
                if frontier_node.type == SchedulingNode.Type.LEAF:
                    leafs.append(frontier_node)
                else:
                    new_frontier_nodes.extend(frontier_node.branches)

            frontier_nodes = new_frontier_nodes

        return leafs


def _check_connection_constraint_respected(frontier_node):
    """Check if the frontier node achieves/ respects the constraint of the root node"""

    constraint_respected = True
    if frontier_node.root:
        connector_object_constraint = frontier_node.root.connector_object_constraint
    else:
        connector_object_constraint = None

    if connector_object_constraint is None:
        return constraint_respected

    process_executions_component = frontier_node.leave
    entity_used = process_executions_component.get_entity_used()

    if not connector_object_constraint.entity_type.identification == entity_used.entity_type.identification:
        # Maybe later something else needed
        return constraint_respected

    if connector_object_constraint.identification != entity_used.identification:
        constraint_respected = False
        return constraint_respected

    return constraint_respected


def _check_order_binding_constraint(process_executions_component, resource_order_binding):
    """Check if the order_binding constraint is broken
    It means that the resource is blocked for an order and is not usable for the order chosen in best_branch
    """

    if not resource_order_binding:
        return True

    entity_used = process_executions_component.get_entity_used()
    if entity_used in resource_order_binding:
        parent = process_executions_component.process_executions_components_parent
        process_execution = parent.goal_item

        if resource_order_binding[entity_used] != process_execution.order:
            return False

    return True


def _consider_constraints(frontier_nodes, resource_order_binding):
    """Consider the connection constraint"""

    constraint_respected_lst = []
    for frontier_node in frontier_nodes:
        # if one of the branches contains the object and is of the same object type_ it should be chosen
        if frontier_node.type == SchedulingNode.Type.LEAF:
            connection_constraint_respected = _check_connection_constraint_respected(frontier_node)
            constraint_respected = connection_constraint_respected

            if constraint_respected:
                order_binding_constraint_respected = _check_order_binding_constraint(frontier_node.leave,
                                                                                     resource_order_binding)
                constraint_respected = order_binding_constraint_respected

        else:
            constraint_respected = True

        if constraint_respected:
            constraint_respected_lst.append(frontier_node)

    return constraint_respected_lst


def _get_resource_utilization_combined(resource_ids_str, resources_utilization):
    """Returns the combined utilization value summed over all resource IDs"""
    resource_ids = json.loads(resource_ids_str)
    combined_utilization = sum(resources_utilization[resource_id] for resource_id in resource_ids)

    return combined_utilization


v_get_resource_utilization_combined = np.vectorize(_get_resource_utilization_combined)


class HeuristicsSchedulingNode(SchedulingNode):
    branch_eval_data_type = np.dtype([('resources_utilization', 'float32'),
                                      ('resource_process_requests', 'float32'),
                                      ('participating_resources', '<U256'),
                                      # ('lead_time_min', 'float32'),
                                      ('amount_predecessors', 'float32'),
                                      ('first_accepted_time_stamp', 'datetime64[ns]'),
                                      ('expected_process_time', 'float32'),
                                      # for transport access stupid if in alternative not needed
                                      ('reachable_end_nodes', 'float32')])

    # eval_methods: branch_eval parameter name, evaluation helper, weighting of each branch_eval parameter
    eval_methods = [('resources_utilization', "MIN", 4),
                    ('resource_process_requests', "MIN", 1),
                    ('participating_resources', "MAX", 1),  # should be planned first
                    # ('lead_time_min', "MIN", 10),  # get the process in a process path with no process before
                    ('amount_predecessors', "MIN", 13),
                    ('first_accepted_time_stamp', "MIN", 6),
                    ('expected_process_time', "-", 1),
                    ('reachable_end_nodes', "MIN", 1)]

    def __init__(self, identification=None, root=None, branches=None, leave=None, type_=None, branch_eval=None,
                 issue_id=None, connector=None, level=0, component=None):
        super(HeuristicsSchedulingNode, self).__init__(identification=identification, type_=type_, level=level,
                                                       root=root, branches=branches, leave=leave, connector=connector,
                                                       component=component)

        self.branch_eval = branch_eval
        self.issue_id = issue_id

        # resources_utilization: the utilization of resources participated in the node/ the nodes below
        # resource_process_requests: number of process_requests a resource has (only usable in the leaf_nodes)

    def add_branch_eval_higher_nodes(self):
        """Add a heuristic value to the higher node
        Other possible influencing factors are:
        - most preferred (related to the others)
        - most preferred (related the own timeline)
        """
        self.branch_eval = np.array([(self._get_resources_utilization_mean(),
                                      np.nan,
                                      self._get_participating_resources_combined(),
                                      # self._get_lead_time_min(),
                                      self._get_amount_predecessors(),
                                      self._get_first_accepted_time_stamp_first(),
                                      self._get_expected_process_time_combined(),
                                      self._get_reachable_end_nodes_combined())],
                                    dtype=type(self).branch_eval_data_type)

    def add_branch_eval_leave_node(self, resources_utilization, resources_number_components):
        """Add a heuristic value to the leave node"""
        self.branch_eval = np.array([(self._get_resources_utilization(resources_utilization),
                                      self._get_process_requests_resource(resources_number_components),
                                      self._get_participating_resources(),
                                      # self._get_lead_time(),
                                      self._get_amount_predecessors_sub_nodes(),
                                      self._get_first_accepted_time_stamp(),
                                      self._get_expected_process_time(),
                                      self._get_reachable_end_nodes())],
                                    dtype=type(self).branch_eval_data_type)

    def _get_lead_time(self):
        """Get the lead_time of the process/ process_execution time of processes before in a process_chain"""

        process_executions_component = self.leave
        lead_times = process_executions_component.reference_preference.lead_time
        if lead_times:
            max_lead_time = max(lead_times.values())
        else:
            max_lead_time = 0

        return max_lead_time

    def _get_amount_predecessors(self):
        """Get the amount of predecessors"""

        amount_predecessors = len(self.predecessors)
        return amount_predecessors

    def _get_amount_predecessors_sub_nodes(self):
        """Get the amount of predecessors"""
        predecessors_sub_nodes = [sub_node.branch_eval["amount_predecessors"]
                                  for sub_node in self.get_sub_nodes()]
        amount_predecessors = len(self.predecessors)

        if predecessors_sub_nodes:
            min_predecessors = min([sub_node.branch_eval["amount_predecessors"]
                                    for sub_node in self.get_sub_nodes()])
            amount_predecessors += min_predecessors

        return amount_predecessors

    def _get_lead_time_min(self):
        """Get the lead_time of the process/ process_execution time of processes before in a process_chain"""

        lead_times = [sub_node.branch_eval["lead_time_min"] for sub_node in self.get_sub_nodes()]
        if self.type == SchedulingNode.Type.AND:
            lead_time_min = max(lead_times)
        elif self.type == SchedulingNode.Type.OR:
            lead_time_min = min(lead_times)
        else:
            raise ValueError
        return lead_time_min[0]

    def _get_amount_predecessors_sum(self):
        """Determine the amount of predecessors hidden behind the deeper leave noces and self"""

        predecessors = [sub_node.branch_eval["amount_predecessors"] for sub_node in self.get_sub_nodes()]
        if self.type == SchedulingNode.Type.AND:
            amount_predecessors_sub = sum(predecessors)
        elif self.type == SchedulingNode.Type.OR:
            amount_predecessors_sub = min(predecessors)
        else:
            raise ValueError
        amount_predecessors = amount_predecessors_sub[0] + len(self.predecessors)

        return amount_predecessors

    def _get_first_accepted_time_stamp(self):
        """Determine the first time for the node to start the process"""
        if self.type != SchedulingNode.Type.LEAF:
            raise Exception

        min_datetime, max_datetime = \
            _get_min_max_value_accepted_time_period(process_executions_component=self.leave)
        return min_datetime

    def _get_first_accepted_time_stamp_first(self):
        """Get the first accepted_time_stamp of all the first accepted_time_stamps in the deeper_nodes"""

        first_accepted_time_stamps = [sub_node.branch_eval["first_accepted_time_stamp"]
                                      for sub_node in self.get_sub_nodes()]
        first_accepted_time_stamps_first = min(first_accepted_time_stamps)

        return first_accepted_time_stamps_first[0]

    def _get_expected_process_time(self):
        """:return the expected process_time to schedule"""
        if self.type != SchedulingNode.Type.LEAF:
            raise Exception

        expected_process_time = self.leave.get_process_execution_time()
        return expected_process_time

    def _get_expected_process_time_combined(self):
        """Combine the expected_process_times in a respective way"""

        sub_nodes = self.get_sub_nodes()
        expected_process_times = [sub_node.branch_eval["expected_process_time"]
                                  for sub_node in sub_nodes]

        # ToDo: how to combine them in a right order? (important)
        expected_process_times_combined = sum(expected_process_times)

        return expected_process_times_combined

    def _get_resources_utilization(self, resources_utilization):
        """Determine the resource utilization for the resource used in the scheduling object"""

        if self.type != SchedulingNode.Type.LEAF:
            raise Exception

        entity = self.leave.get_entity_used()
        if entity in resources_utilization:
            resource_utilization = resources_utilization[entity]

        else:
            resource_utilization = None

        return resource_utilization

    def _get_resources_utilization_mean(self):
        """Calculate the mean of the utilization from each resource in the process_executions_components used"""

        resources_utilization_lst = [sub_node.branch_eval["resources_utilization"]
                                     for sub_node in self.get_sub_nodes()]
        resources_utilization_mean = \
            np.round((sum(resources_utilization_lst) + 1e-12) / (len(resources_utilization_lst) + 1e-12), 2)

        return resources_utilization_mean

    def _get_participating_resources(self):
        """Determine the resources participate in the scheduling object"""

        if self.type != SchedulingNode.Type.LEAF:
            raise Exception

        participating_entity = self.leave.goal_item

        if isinstance(participating_entity, Resource):
            participating_resources = json.dumps([participating_entity.identification])
        else:
            participating_resources = json.dumps([])

        return participating_resources

    def _get_participating_resources_combined(self):
        """Determine the resources participate in the scheduling object on a higher level in the tree"""

        try:
            participating_resources = \
                json.dumps(list(set(
                    [participating_resource
                     for sub_node in self.get_sub_nodes()
                     for participating_resource in json.loads(sub_node.branch_eval["participating_resources"][0])
                     if sub_node.branch_eval["participating_resources"][0]])))
        except:
            raise Exception([sub_node.branch_eval["participating_resources"]
                             for sub_node in self.get_sub_nodes()])

        return participating_resources

    def _get_reachable_end_nodes(self):
        """Determine the number of nodes in the that are reachable from the node"""

        reachable_end_nodes = 0
        return reachable_end_nodes

    def _get_reachable_end_nodes_combined(self):
        """Determine the number of nodes in the that are reachable from the node"""

        reachable_end_nodes = \
            sum([sub_node.branch_eval["reachable_end_nodes"]
                 if sub_node.branch_eval["reachable_end_nodes"] != 0 else 1  # the last node has a zero as value
                 for sub_node in self.get_sub_nodes()])

        return reachable_end_nodes

    def _get_process_requests_resource(self, resources_number_components):
        """Determine in how many processes the participating resource is participated."""

        participating_entity = self.leave.goal_item
        number_processes_participated = resources_number_components[participating_entity]
        return number_processes_participated

    def _determine_intersecting_area_share(self):
        # ToDo
        pass

    def get_best_frontiers_branch(self, frontier_nodes, resources_utilization, resource_order_binding):
        """Choose the best branch as next node to open in the tree"""

        if not frontier_nodes:
            return [], None

        frontier_nodes = _consider_constraints(frontier_nodes, resource_order_binding)

        if not frontier_nodes:
            return [], None

        elif len(frontier_nodes) == 1:
            best_branch = frontier_nodes[0]
            frontier_nodes = []
            return frontier_nodes, best_branch

        branches_branch_eval = {frontier_node: frontier_node._get_branch_eval() for frontier_node in frontier_nodes}

        branch_evaluations = {(i, key): value for i, (key, value) in enumerate(list(branches_branch_eval.items()))}
        branches_eval_concatenated = np.concatenate((list(branch_evaluations.values())), axis=0)

        winner_branches_identifications_a = np.zeros(branches_eval_concatenated.shape[0])
        for record_name, evaluation_method, weighting_factor in type(self).eval_methods:
            if record_name == "participating_resources":
                continue

            if record_name == "resources_utilization":
                participating_resources = branches_eval_concatenated["participating_resources"]

                resources_utilization_updated = \
                    v_get_resource_utilization_combined(participating_resources,
                                                        np.repeat(resources_utilization,
                                                                  participating_resources.shape[0]))
                branches_eval_concatenated["resources_utilization"] = resources_utilization_updated

            if evaluation_method == "-":
                continue

            winner_branches_identifications = \
                get_winner_branches_identification(branches_eval_concatenated, evaluation_method, record_name)
            winner_branches_identifications_a[winner_branches_identifications] += weighting_factor
        max_indexes = np.argwhere(winner_branches_identifications_a ==
                                  np.amax(winner_branches_identifications_a)).flatten()
        max_ = choice(max_indexes)
        # eval_value = winner_branches_identifications_a.max(axis=0)

        best_branches = [x[1] for x in list(branch_evaluations.keys()) if max_ == x[0]]

        if len(best_branches) != 1:
            raise Exception

        best_branch = best_branches[0]
        frontier_nodes.remove(best_branch)
        return frontier_nodes, best_branch

    def _get_branch_eval(self):
        """Return the branch evaluation pre-calculated"""
        return self.branch_eval

    def get_leave_nodes(self, accepted_process_executions_id=None, exception_nodes=None, ) \
            -> [list[list], list]:
        """Return the leaf_nodes reachable from the node except the exception nodes
        It's a list of list (one node of each nested list should be chosen)
        Notes: It only considers the nodes two level deeper than the node self
        (therefore only the current challenges (means the material supply with one agv) should be representable)
        # ToDo: maybe later other challenges occur
        """

        if self.type != SchedulingNode.Type.AND:
            raise Exception

        if exception_nodes is not None:
            sub_branches = [branch
                            for branch in self.branches
                            if branch not in exception_nodes]
        else:
            sub_branches = self.branches

        leave_nodes = []
        interim_nodes = [self]

        for sub_branch in sub_branches:
            if sub_branch.type == SchedulingNode.Type.LEAF:
                if accepted_process_executions_id is not None:
                    if sub_branch.leave.process_execution_id != accepted_process_executions_id:
                        continue

                leave_nodes.append([sub_branch])

            elif sub_branch.type == SchedulingNode.Type.OR or sub_branch.type == SchedulingNode.Type.AND:
                sub_sub_branches = []
                for sub_sub_branch in sub_branch.branches:
                    if sub_sub_branch.type != SchedulingNode.Type.LEAF:
                        continue
                    if sub_sub_branch.leave.process_execution_id != accepted_process_executions_id:
                        continue

                    if sub_branch.type == SchedulingNode.Type.OR:
                        sub_sub_branches.append(sub_sub_branch)
                    else:  # AND case
                        sub_sub_branches.append([sub_sub_branch])

                    interim_nodes.append(sub_branch)

                if sub_sub_branches:
                    leave_nodes.append(sub_sub_branches)

            else:
                raise NotImplementedError

        return leave_nodes, interim_nodes

    def reset_time_slot(self, resources_preferences):
        leafs = self.get_leafs()
        for leaf in leafs:
            process_executions_component = leaf.leave
            entity = process_executions_component.get_entity_used()
            process_execution_id = self.process_execution_id
            if entity in resources_preferences and process_execution_id:
                resources_preferences[entity].unblock_time_slot(unblocker_name="CentralCoordinator",
                                                                process_execution_id=process_execution_id)


class TreeVisualizer:

    def __init__(self):
        self.node_id = 0
        self.nodes = {}

    def visualize_tree(self, root_node):
        """Create and plt a visualization of the tree"""

        graph_dict = {self._get_node_description(root_node): [self._get_node_description(branch)
                                                              for branch in root_node.branches]}
        frontier_nodes = root_node.branches
        while True:
            new_frontier_nodes = []
            for branch in frontier_nodes:
                new_frontier_nodes.extend(branch.branches)
                graph_dict.setdefault(self._get_existing_node_description(branch),
                                      []).extend([self._get_node_description(branch) for branch in branch.branches])

            if not new_frontier_nodes:
                break
            frontier_nodes = copy(new_frontier_nodes)

        # images can be found in the project folder
        # graph = Graph(graph_dict, directed=False)
        # graph.plot(orientation=config.TREE_ORIENTATION,  # shape=config.LEAF_SHAPE,
        #            output_path=f'./debugging_images/tree{datetime.now().strftime("%H%M%S")}.png')

    def _get_existing_node_description(self, node):
        return self.nodes[node]

    def _get_node_description(self, node):

        self.nodes[node] = f"{self.node_id}: {node.identification}, {str(node.type).split('.')[-1]}"
        self.node_id += 1

        return self.nodes[node]


def _determine_or_node_needed(frontier_node_batch):
    """Or nodes are only needed if alternatives are available else actually not
    This method tries to determine it
    """
    or_node_needed = True
    if len(frontier_node_batch) <= 1:
        or_node_needed = False
    else:
        # different frontiers should be combined by "AND" operator
        different_frontiers = set([frontier.identification for frontier in frontier_node_batch])
        if len(different_frontiers) > 1:
            or_node_needed = False

    return or_node_needed


def get_branch_id(process_executions_component_parent, node_type, leave=None, addition=None):
    """Determines a specific branch_id ("AND" and "OR" and "LEAF" nodes gets different branch IDs because
    in some cases the same process_executions_component is used for different branch types)"""

    if leave is not None and SchedulingNode.Type.OR == node_type:
        branch_id = (str(leave) + str(process_executions_component_parent.cfp_path),
                     process_executions_component_parent.goal)

    elif SchedulingNode.Type.OR == node_type:
        branch_id = (str(process_executions_component_parent.cfp_path),
                     process_executions_component_parent.goal)

    elif SchedulingNode.Type.AND == node_type:
        branch_id = str(process_executions_component_parent.cfp_path)

    else:
        raise Exception

    if addition:
        branch_id += addition

    return branch_id


def _get_parent_node(process_executions_component_parent, frontier_node_batch,
                     process_executions_components_nodes, node_class, node_type, addition=None):
    """Determines the parent node of a frontier_node_batch"""

    # if len(process_executions_component_parent.process_executions_components) == 1:
    #     parent_node = frontier_node  # avoid irrelevant nodes in the tree
    #     return parent_node, process_executions_components_nodes

    branch_id = get_branch_id(process_executions_component_parent, node_type=node_type, addition=addition)
    if branch_id not in process_executions_components_nodes:
        id_ = str(process_executions_component_parent.cfp_path)
        if addition:
            id_ += addition
        if node_class == SchedulingNode:
            parent_node = node_class(identification=id_, type_=node_type,
                                     component=process_executions_component_parent)
        else:  # elif node_class == HeuristicsSchedulingNode:
            issue_id = process_executions_component_parent.issue_id
            parent_node = node_class(identification=id_, type_=node_type, issue_id=issue_id,
                                     component=process_executions_component_parent)
        process_executions_components_nodes[branch_id] = parent_node
        additional_parent_node_branches = []

    else:
        parent_node = process_executions_components_nodes[branch_id]
        parent_node_branches = parent_node.branches
        additional_parent_node_branches = list(set(parent_node_branches).difference(set(frontier_node_batch)))

    predecessors_frontier_nodes, additional_predecessors_frontier_nodes = \
        _get_predecessor_nodes(node_type, process_executions_component_parent, process_executions_components_nodes,
                               frontier_node_batch, additional_parent_node_branches)

    if predecessors_frontier_nodes is None:
        for frontier_node in frontier_node_batch:
            frontier_node.connect_nodes(parent_node)
    else:
        for frontier_node in frontier_node_batch:
            if frontier_node in predecessors_frontier_nodes:
                predecessors = predecessors_frontier_nodes[frontier_node]
            else:
                predecessors = None
            frontier_node.connect_nodes(parent_node, predecessors)

    if additional_predecessors_frontier_nodes:
        for node, predecessors in additional_predecessors_frontier_nodes.items():
            node.add_predecessors(predecessors)

    return parent_node, process_executions_components_nodes


def _get_predecessor_nodes(node_type, process_executions_component_parent, process_executions_components_nodes,
                           frontier_node_batch, additional_parent_node_branches):
    """Return predecessors for the frontier_batch to strengthen the partial order"""

    if node_type == SchedulingNode.Type.OR:
        predecessors_frontier_nodes = None
        return predecessors_frontier_nodes, additional_parent_node_branches

    if isinstance(process_executions_component_parent, ProcessExecutionsVariant):
        predecessors_frontier_nodes = [node
                                       for node in frontier_node_batch
                                       for predecessor in process_executions_component_parent.predecessors
                                       if node.leave == predecessor]

    elif isinstance(process_executions_component_parent, ProcessExecutionsPath):
        if process_executions_component_parent.process_executions_components_parent:
            predecessors_frontier_nodes = \
                [node
                 for node in frontier_node_batch
                 for predecessor in
                 process_executions_component_parent.process_executions_components_parent.predecessors
                 if node.leave == predecessor]
        else:
            predecessors_frontier_nodes = []

    additional_predecessors_frontier_nodes = []
    return predecessors_frontier_nodes, additional_predecessors_frontier_nodes


def _get_predecessor_nodes_path(process_executions_component_parent, process_executions_components_nodes,
                                frontier_node_batch, additional_parent_node_branches):
    additional_predecessors_frontier_nodes = {}
    process_executions_components = process_executions_component_parent.process_executions_components
    process_executions_components_node_identifications = \
        [str(process_executions_component.cfp_path)
         for process_executions_component in process_executions_components]

    # get nodes that build predecessors relationships
    predecessors_frontier_nodes = \
        {process_executions_components_nodes[node_identification]:
             [process_executions_components_nodes[process_executions_components_node_identifications[idx - 1]]]
         for idx, node_identification in enumerate(process_executions_components_node_identifications)
         if idx > 0
         if node_identification in process_executions_components_nodes
         if process_executions_components_nodes[node_identification] in frontier_node_batch and  # in frontier nodes
         process_executions_components_node_identifications[idx - 1] in process_executions_components_nodes}

    if additional_parent_node_branches:
        # get nodes that build predecessors relationships
        additional_predecessors_frontier_nodes = \
            {process_executions_components_nodes[node_identification]:
                 [process_executions_components_nodes[process_executions_components_node_identifications[idx - 1]]]
             for idx, node_identification in enumerate(process_executions_components_node_identifications)
             if idx > 0
             if node_identification in process_executions_components_nodes
             if process_executions_components_nodes[node_identification] in additional_parent_node_branches and
             process_executions_components_node_identifications[idx - 1] in process_executions_components_nodes}

    return predecessors_frontier_nodes, additional_predecessors_frontier_nodes


def _get_predecessor_nodes_variant(process_executions_component_parent, process_executions_components_nodes,
                                   frontier_node_batch, additional_parent_node_branches):
    additional_predecessors_frontier_nodes = {}
    predecessor_process_executions_components, main_process_executions_components = \
        process_executions_component_parent.get_partial_order()

    predecessor_cfp_component_match = \
        [str(process_executions_component.cfp_path)
         for process_executions_component in predecessor_process_executions_components]
    main_cfp_component_match = \
        [str(process_executions_component.cfp_path)
         for process_executions_component in main_process_executions_components]

    predecessor_nodes = [process_executions_components_nodes[cfp_path_str]
                         for cfp_path_str in predecessor_cfp_component_match
                         if cfp_path_str in process_executions_components_nodes]

    if predecessor_nodes:
        predecessors_frontier_nodes = {frontier_node: predecessor_nodes
                                       for frontier_node in frontier_node_batch
                                       if frontier_node.identification in main_cfp_component_match}

        if additional_parent_node_branches:
            additional_predecessors_frontier_nodes = \
                {frontier_node: predecessor_nodes
                 for frontier_node in additional_parent_node_branches
                 if frontier_node.identification in main_cfp_component_match}

    else:
        predecessors_frontier_nodes = {}

    return predecessors_frontier_nodes, additional_predecessors_frontier_nodes


def _set_levels(root_node):
    """Set the level for each node in the tree to determine the relation to the root node"""
    frontier_nodes = [root_node]
    level = 0
    while frontier_nodes:

        for frontier_node in frontier_nodes:
            frontier_node.add_level(level)
        frontier_nodes = [sub_node for frontier_node in frontier_nodes for sub_node in frontier_node.get_sub_nodes()]
        level += 1


class SchedulingTree:

    def __init__(self, process_executions_components: list, resources_preferences: dict,
                 digital_twin: StateModel, routing_service: RoutingService, start_time_stamp):
        """
        :param process_executions_components: leaf_nodes
        :param resources_preferences: preference mapped to the resources
        :attribute root_node: is the highest node that is connected via branch nodes to the leaves
        (direct connection to the leaves should be also possible, but seldom)
        :attribute leaf_nodes: leave nodes are combined with a leave/ process_executions_component
        that states the need of the node
        """
        self.process_executions_components = process_executions_components
        resources_usable = \
            list(set(process_executions_component.goal_item
                     for process_executions_component in process_executions_components
                     if isinstance(process_executions_component.call_for_proposal, ResourceCallForProposal)))
        self.resources_preferences = {resource: resources_preferences[resource]
                                      for resource in resources_usable}
        self.NodeClass = self._get_node_class()
        root_node, leaf_nodes = self._build_tree()
        self.root_node = root_node
        self.leaf_nodes = leaf_nodes
        connector_nodes = [connector_node for leaf_node in leaf_nodes if leaf_node.connector
                           for connector_node in leaf_node.connector.get_connector_leave_nodes()]
        self.all_leave_nodes = connector_nodes + self.leaf_nodes

        self.routing_service = routing_service
        self.digital_twin = digital_twin
        self.start_time_time_stamp = start_time_stamp
        # resource, (position, timestamp)
        self.non_stationary_resources_positions = \
            {resource: self._get_resource_position(resource=resource, start_time_stamp=start_time_stamp)
             for resource in list(resources_preferences.keys())
             if isinstance(resource, NonStationaryResource)}

    def _get_resource_position(self, resource, start_time_stamp):
        """Get the position changes mapped to time_stamps the position is changed for a resource"""

        # no position change for passive moving resources that are always situated in other resources
        # the position can be taken from the situated in resources

        position_changes = resource.get_positions(start_time_stamp)
        if not position_changes:
            if resource.situated_in:
                position_changes = resource.situated_in.get_positions(start_time_stamp)

        resource_position_changes = \
            {time_stamp if time_stamp != 0 else datetime(1970, 1, 1):
                 self.digital_twin.get_stationary_resource_at_position(pos_tuple=position)[0]
             for time_stamp, position in position_changes.items() if position is not None}

        if not resource_position_changes and resource.situated_in:
            resource_position_changes = {datetime(1970, 1, 1): resource.situated_in}

        return resource_position_changes

    def _get_node_class(self):
        """Determine the node class used to create the tree"""
        return SchedulingNode

    def _build_tree(self):
        """
        build a tree based on the process_executions_components (leaf_nodes) and the paths (branches) given
        in the process_executions_components
        :return: dict structured tree
        """
        # print(get_debug_str("SchedulingTree", "") + f" Start tree building")

        leaf_nodes, frontier_nodes, process_executions_components_nodes = self._create_leaf_nodes()
        # print("Len leaf_nodes: ", len(leaf_nodes))
        first_row_nodes = self._create_branch_nodes(frontier_nodes, process_executions_components_nodes)
        root_node = self._create_root_nodes(first_row_nodes)
        _set_levels(root_node)

        # print(get_debug_str("SchedulingTree", "") + f" End tree building")

        if tree_visualization:
            tree_visualizer = TreeVisualizer()
            tree_visualizer.visualize_tree(root_node)

        return root_node, leaf_nodes

    def _create_leaf_nodes(self):
        """Create the leaf nodes and the first branch nodes above themselves"""

        leaf_nodes = {}
        connector_objects = []
        process_executions_components_leaf = {}
        # Or relationships on the lowest level
        for process_executions_component in self.process_executions_components:
            if process_executions_component.type == ProcessExecutionsComponent.Type.CONNECTOR:
                connector_objects.append(process_executions_component)
                continue

            if not process_executions_component.process_executions_components_parent:
                # raise Exception
                print(f"Warning: {process_executions_component.cfp_path}")
                continue  # reason: rejection to the last level not passed

            if self.NodeClass == SchedulingNode:
                leave_node = self.NodeClass(identification=str(process_executions_component.cfp_path),
                                            leave=process_executions_component, type_=SchedulingNode.Type.LEAF)
            else:  # elif self.NodeClass == HeuristicsSchedulingNode:
                issue_id = process_executions_component.issue_id
                leave_node = self.NodeClass(identification=str(process_executions_component.cfp_path),
                                            leave=process_executions_component, type_=SchedulingNode.Type.LEAF,
                                            issue_id=issue_id)
            # else:
            #     raise Exception

            process_executions_component_parent = \
                process_executions_component.process_executions_components_parent
            goal_id = str(leave_node.leave.goal.identification) + str(process_executions_component.cfp_path)

            branch_id = get_branch_id(process_executions_component_parent, node_type=SchedulingNode.Type.OR,
                                      leave=goal_id)
            leaf_nodes.setdefault(branch_id, []).append((process_executions_component_parent, leave_node))
            process_executions_components_leaf[process_executions_component] = leave_node

        process_executions_components_nodes = {}
        # connector nodes are needed for the transport access
        connector_leaves = []

        for connector_component in connector_objects:
            if not connector_component.process_executions_components_parent:
                continue
            connector_leave_node = self.NodeClass(identification=str(connector_component.cfp_path),
                                                  leave=connector_component, type_=SchedulingNode.Type.LEAF,
                                                  issue_id=connector_component.issue_id, component=connector_component)
            connector_leaves.append(connector_leave_node)

        # connector_or_nodes = []
        # for idx, connector_component in enumerate(connector_objects):
        #     or_component_connector = connector_component.process_executions_components_parent
        #
        #     parent_or_node, process_executions_components_nodes = \
        #         _get_parent_node(or_component_connector, [],
        #                          process_executions_components_nodes, self.NodeClass,
        #                          node_type=SchedulingNode.Type.OR)
        #     connector_or_nodes.append(parent_or_node)
        #     connector_leaves[idx].connect_nodes(parent_or_node)

        # print("Connector objects", len(connector_objects))

        connector_batch = []
        and_components_connector = []
        component_node_match = {}
        node_component_match = {}
        component_predecessors = {}
        for idx, connector_component in enumerate(connector_objects):
            and_component_connector = connector_component.process_executions_components_parent
            if and_component_connector not in and_components_connector:
                and_components_connector.append(and_component_connector)

            path_components = \
                and_component_connector.process_executions_components_parent.get_process_executions_components_lst()
            for idx1, component in enumerate(path_components):
                if idx1 == 0:
                    component_predecessors[component] = []
                    continue
                component_predecessors[component] = path_components[idx1 - 1:idx1]

            parent_and_node, process_executions_components_nodes = \
                _get_parent_node(and_component_connector, [], process_executions_components_nodes,
                                 self.NodeClass, node_type=SchedulingNode.Type.AND, addition=str(idx))

            connector_leaves[idx].connect_nodes(parent_and_node)

            connector_batch.append(parent_and_node)
            node_component_match[parent_and_node] = and_component_connector
            component_node_match[and_component_connector] = parent_and_node

        if component_predecessors:
            for node, component in node_component_match.items():
                node.add_predecessors([component_node_match[component_]
                                       for component_ in component_predecessors[component]])

        if connector_batch:
            leave_nodes_connectors = {}
            for and_component_connector in and_components_connector:
                node_ = process_executions_components_leaf[
                    and_component_connector.process_executions_components_parent]
                leave_nodes_connectors[node_] = and_component_connector

            for leave_node_, and_component_connector in leave_nodes_connectors.items():
                connector_node, process_executions_components_nodes = \
                    _get_parent_node(and_component_connector, [], process_executions_components_nodes,
                                     self.NodeClass, node_type=SchedulingNode.Type.AND, addition="CONNECTOR")

                for connector_sub_node in list(set(connector_batch)):
                    if connector_sub_node.issue_id == leave_node_.issue_id:
                        connector_sub_node.connect_nodes(connector_node)
                        # if not leave_node_.connector:
                        #     print("Set connector")
                        leave_node_.connector = connector_node

        frontier_nodes = {}

        # create the first branches from leave nodes up
        for branch_id, process_executions_component_parent_leave_nodes_batch in leaf_nodes.items():

            if len(process_executions_component_parent_leave_nodes_batch) <= 1:
                process_executions_component_parent, leave_node = \
                    process_executions_component_parent_leave_nodes_batch[0]
                frontier_nodes.setdefault(process_executions_component_parent, []).append(leave_node)
                continue

            alternatives = set(str(leaf_node.leave.cfp_path)
                               for _, leaf_node in process_executions_component_parent_leave_nodes_batch)
            if len(alternatives) > 1:
                continue

            for process_executions_component_parent, leave_node in process_executions_component_parent_leave_nodes_batch:

                if branch_id not in process_executions_components_nodes:
                    if self.NodeClass == SchedulingNode:
                        parent_node = self.NodeClass(identification=str(process_executions_component_parent.cfp_path),
                                                     type_=SchedulingNode.Type.OR,
                                                     component=process_executions_component_parent)
                    else:  # elif self.NodeClass == HeuristicsSchedulingNode:
                        issue_id = process_executions_component_parent.issue_id
                        parent_node = self.NodeClass(identification=str(process_executions_component_parent.cfp_path),
                                                     type_=SchedulingNode.Type.OR, issue_id=issue_id,
                                                     component=process_executions_component_parent)

                    process_executions_components_nodes[branch_id] = parent_node

                else:
                    parent_node = process_executions_components_nodes[branch_id]

                leave_node.connect_nodes(parent_node)

                if process_executions_component_parent not in frontier_nodes:
                    frontier_nodes[process_executions_component_parent] = [parent_node]
                else:
                    if parent_node not in frontier_nodes[process_executions_component_parent]:
                        frontier_nodes[process_executions_component_parent].append(parent_node)

        if list(leaf_nodes.values()):
            leaf_nodes_exclusive = list(list(zip(*reduce(concat, list(leaf_nodes.values()))))[1])
        else:
            leaf_nodes_exclusive = []

        return leaf_nodes_exclusive, frontier_nodes, process_executions_components_nodes

    def _create_branch_nodes(self, frontier_nodes, process_executions_components_nodes):
        """Create the branch nodes between the root nodes and the lowest level nodes/ leave nodes"""

        first_row_nodes = []
        new_frontier_nodes = {}
        while frontier_nodes:

            # ToDo: maybe multiprocessing usable
            component_match = {}
            for process_executions_component_parent, frontier_node_batch in frontier_nodes.items():
                frontier_node_batch = list(set(frontier_node_batch))
                parent_node, process_executions_components_nodes = \
                    _get_parent_node(process_executions_component_parent, frontier_node_batch,
                                     process_executions_components_nodes, self.NodeClass,
                                     node_type=SchedulingNode.Type.AND)
                component_match[process_executions_component_parent] = parent_node
                if process_executions_component_parent not in new_frontier_nodes:

                    process_executions_component_parent = \
                        process_executions_component_parent.process_executions_components_parent

                    if not process_executions_component_parent:
                        first_row_nodes.append(parent_node)
                        continue

                    new_frontier_nodes.setdefault(process_executions_component_parent, []).append(parent_node)

            if new_frontier_nodes:
                for _, nodes in new_frontier_nodes.items():
                    for node in nodes:
                        predecessors = node.component.predecessors
                        if not predecessors:
                            continue
                        predecessor_nodes = [component_match[predecessor] for predecessor in predecessors]
                        node.add_predecessors(predecessor_nodes)

            frontier_nodes = new_frontier_nodes.copy()
            new_frontier_nodes = {}

            for process_executions_component_parent, frontier_node_batch in frontier_nodes.items():
                # if more than one component
                frontier_node_batch = list(set(frontier_node_batch))
                or_node_needed = _determine_or_node_needed(frontier_node_batch)
                if or_node_needed:
                    parent_node, process_executions_components_nodes = \
                        _get_parent_node(process_executions_component_parent, frontier_node_batch,
                                         process_executions_components_nodes, self.NodeClass,
                                         node_type=SchedulingNode.Type.OR)
                    if process_executions_component_parent not in new_frontier_nodes:
                        new_frontier_nodes.setdefault(process_executions_component_parent, []).append(parent_node)
                else:
                    new_frontier_nodes.setdefault(process_executions_component_parent, []).extend(frontier_node_batch)

            frontier_nodes = new_frontier_nodes.copy()
            new_frontier_nodes = {}

        return first_row_nodes

    def _create_root_nodes(self, first_row_nodes):
        """Create the first order nodes of the tree including the root node and the connection to the first order nodes
        """

        first_row_group_nodes = {}
        for first_row_node in list(set(first_row_nodes)):
            first_row_group_nodes.setdefault(first_row_node.identification, []).append(first_row_node)

        root_node = self.NodeClass(identification=-1, type_=SchedulingNode.Type.ROOT)

        for group_id, first_row_nodes_group in first_row_group_nodes.items():
            for first_row_node in first_row_nodes_group:
                first_row_node.connect_nodes(root_node)

        return root_node


def _get_min_max_value_accepted_time_period(process_executions_component):
    """Get the minimum value and the maximum value from a numpy array of accepted_time_periods"""

    accepted_time_period = process_executions_component.get_accepted_time_periods()

    if accepted_time_period.any():
        return accepted_time_period[0][0], accepted_time_period[-1][1]
    else:
        return np.datetime64("NaT"), np.datetime64("NaT")


v_get_min_max_value_accepted_time_period = np.vectorize(_get_min_max_value_accepted_time_period)


def _determine_start_end(process_executions_components):
    """Determine the "smallest" start_time and the "highest" end_time
    :param process_executions_components: the process_executions_component that provide the accepted_time_periods
    """

    min_datetime_a, max_datetime_a = v_get_min_max_value_accepted_time_period(process_executions_components)

    start_time = min_datetime_a[~np.isnan(min_datetime_a)].min()
    end_time = max_datetime_a[~np.isnan(max_datetime_a)].max()

    return start_time, end_time


def _calculate_resource_utilization(resource, start_time, end_time):
    """
    Calculates the current utilization for each resource in the accepted_time_period,
    respectively actually only in the period between the first time stamp of the accepted_time_periods and the last one
    :param resource: the resource for which the utilization is calculated
    :return: the utilization for the resource
    """
    resource_utilization = resource.get_utilization(start_time=start_time, end_time=end_time)

    return resource_utilization


def get_winner_branches_identification(branches_eval_concatenated, evaluation_method, record_name):
    """
    Determine the branches with the best branch evaluation for the record name
    (a record name should be named in the eval_methods)
    """
    values = branches_eval_concatenated[:][record_name]
    if isinstance(values[0], str):
        # ToDo: dynamic solution for participating_resources
        def len_json(value): return len(json.loads(value))  # ToDo: understandable code?

        len_json_v = np.vectorize(len_json)
        values = len_json_v(values)

    if evaluation_method == "MIN":
        values_evaluated = values.min()

    elif evaluation_method == "MAX":
        values_evaluated = values.max()

    else:
        raise NotImplementedError

    winner_branch_identification = np.where(values == values_evaluated)
    return winner_branch_identification


def _get_entities_to_schedule(process_executions_components):
    """Return the entity to schedule"""
    entities_to_schedule = {}
    for process_executions_component in process_executions_components:
        entity_to_schedule = process_executions_component.get_entity_used()
        if isinstance(entity_to_schedule, Resource):
            entities_to_schedule.setdefault("Resource", []).append(entity_to_schedule)
        elif isinstance(entity_to_schedule, Part):
            entities_to_schedule.setdefault("Part", []).append(entity_to_schedule)

    return entities_to_schedule


def _get_resource_preference(resources_preferences, entities_to_schedule):
    """:return the preference object of the resource"""

    resources_preferences = [resources_preferences[entity_to_schedule]
                             for entity_to_schedule in entities_to_schedule["Resource"]]
    return resources_preferences


def _get_other_leave_nodes(frontier_node):
    """Because of multi resource planning, other resources should be planned at the same time
    if they participate"""

    root_node = frontier_node.root
    higher_root_node = None
    if root_node is None:
        other_leave_nodes = []
        root_node_chosen = frontier_node
        interim_nodes = []
        return other_leave_nodes, root_node_chosen, interim_nodes

    if root_node.type == SchedulingNode.Type.OR:
        higher_root_node = root_node.root
        if higher_root_node.type == SchedulingNode.Type.OR:
            higher_root_node = None

    process_executions_component = frontier_node.leave
    if frontier_node.leave is None:
        raise NotImplementedError("Assuming that the exception nodes are leave nodes")
    accepted_process_executions_id = process_executions_component.process_execution_id

    if higher_root_node is not None:
        other_leave_nodes, interim_nodes = \
            higher_root_node.get_leave_nodes(exception_nodes=[root_node],
                                             accepted_process_executions_id=accepted_process_executions_id)
        root_node_chosen = higher_root_node

    elif root_node is not None:
        other_leave_nodes, interim_nodes = \
            root_node.get_leave_nodes(exception_nodes=[frontier_node],
                                      accepted_process_executions_id=accepted_process_executions_id)
        root_node_chosen = root_node

    else:
        other_leave_nodes = []
        interim_nodes = []
        root_node_chosen = None

    interim_nodes = list(set(interim_nodes))
    return other_leave_nodes, root_node_chosen, interim_nodes


def _get_process_executions_component_from_node(node, process_executions_components):
    """Return the process_executions_component for the node searched through the leave_node leave"""
    process_executions_components_parents = [process_executions_component.process_executions_components_parent
                                             for process_executions_component in process_executions_components]

    common_process_executions_components = list(set(process_executions_components_parents))
    if len(common_process_executions_components) != 1:
        pass  # maybe a problem

    if node.leave is not None:
        # the case for the connector node
        pass

    frontier_component_parent = common_process_executions_components[0]

    if process_executions_components[0].cfp_path == frontier_component_parent.cfp_path:
        return frontier_component_parent  # for transport access (intern created in resource request)

    while True:
        if str(frontier_component_parent.cfp_path) == node.identification:
            process_executions_component = frontier_component_parent
            return process_executions_component

        frontier_component_parent = frontier_component_parent.process_executions_components_parent


def _get_nearest_frontier_nodes(last_frontier_node, visited):
    """Get the nearest point in the tree"""
    scheduling_finished = False

    visited = list(set(visited))

    while True:
        frontier_root = last_frontier_node.root
        if frontier_root is None:
            scheduling_finished = not bool(list(set(last_frontier_node.branches).difference(set(visited))))
            if scheduling_finished:
                not_visited_nodes = []
                return scheduling_finished, not_visited_nodes
            raise Exception

        not_visited_nodes = list(set(frontier_root.branches).difference(set(visited)))
        if not_visited_nodes:
            return scheduling_finished, not_visited_nodes

        last_frontier_node = frontier_root
        visited.append(frontier_root)


def _remove_nodes_scheduled_from_frontier_nodes(frontier_nodes, nodes_scheduled, additionally_visited,
                                                visited, scheduled):
    """Remove nodes visited from the frontier nodes"""
    new_nodes_visited = nodes_scheduled  + additionally_visited

    visited.extend(new_nodes_visited)
    scheduled.extend([node for node in nodes_scheduled if node.type == SchedulingNode.Type.LEAF])

    root_nodes_scheduled = []
    for node_visited in new_nodes_visited:
        frontier_nodes, visited, root_nodes_scheduled = \
            _check_root_scheduled(node_visited, visited, new_nodes_visited, frontier_nodes,
                                  root_nodes_scheduled)

        if node_visited in frontier_nodes:
            frontier_nodes.remove(node_visited)

    while root_nodes_scheduled:
        new_root_nodes_scheduled = []
        for node_scheduled in list(set(root_nodes_scheduled)):
            frontier_nodes, visited, new_root_nodes_scheduled = \
                _check_root_scheduled(node_scheduled, visited, root_nodes_scheduled, frontier_nodes,
                                      new_root_nodes_scheduled)

        root_nodes_scheduled = new_root_nodes_scheduled

    return frontier_nodes, visited, scheduled


def _check_root_scheduled(node_visited, visited, new_nodes_visited, frontier_nodes, root_nodes_scheduled):
    """Check if all sub nodes necessary are already visited and mark them as visited"""
    root = _determine_node_completely_scheduled(node_visited.root, visited)
    if root is not None:
        root_nodes_scheduled.append(root)
        if root in frontier_nodes and root not in new_nodes_visited:
            frontier_nodes.remove(root)
        if root in visited:
            visited.append(root)

    return frontier_nodes, visited, root_nodes_scheduled


def _determine_node_completely_scheduled(node_visited, visited):
    """Determine if the node_visited node is completely scheduled because all his sub nodes necessary to schedule
    are scheduled"""

    if node_visited is None:
        return None

    if node_visited is None:
        return None

    if node_visited.type == SchedulingNode.Type.OR:
        return node_visited

    elif node_visited.type == SchedulingNode.Type.AND:
        if set(node_visited.branches).issubset(set(visited)):
            return node_visited

        else:
            # assumption: the connector nodes are scheduled first
            not_chosen_nodes = set(node_visited.branches).difference(set(visited))
            if not not_chosen_nodes:
                return node_visited

    return None


def _check_path_ids_constraint(frontier_node, frontier_node_path_ids):
    """Check if the frontier node is compatible with the path_ids (of the same path)"""
    if frontier_node.branches:
        return True
    ids_ = frontier_node.leave.path_ids
    if not ids_ and not frontier_node_path_ids:
        return True
    elif ids_.intersection(frontier_node_path_ids):
        return True
    else:
        return False


class HeuristicsSchedulingTree(SchedulingTree):
    # ensure that the node has the param - needed in the alternative node selection in the scheduling
    alternative_selection_param = \
        ['resource_process_requests'
         for idx, (eval_param, param_data_type) in enumerate(HeuristicsSchedulingNode.branch_eval_data_type.descr)
         if eval_param == 'resource_process_requests']

    def __init__(self, process_executions_components, resources_preferences, routing_service, digital_twin,
                 start_time_stamp):
        super(HeuristicsSchedulingTree, self).__init__(
            process_executions_components=process_executions_components,
            resources_preferences=resources_preferences, routing_service=routing_service, digital_twin=digital_twin,
            start_time_stamp=start_time_stamp)

        if process_executions_components:
            start_time, end_time = _determine_start_end(process_executions_components)

        else:
            start_time, end_time = None, None

        self.start_time = start_time
        self.end_time = end_time
        self.resources_utilization = None

        self.completely_scheduled = []

        self._resource_order_binding = {}  # used for long time reservations
        self._resource_order_binding_batch = {}

        self._determine_heuristic_values()

    def _get_node_class(self):
        """Overwritten method - heuristics scheduling node as node"""
        return HeuristicsSchedulingNode

    def _determine_heuristic_values(self):
        """Distribute heuristic values in the tree for each node"""

        nodes_visited, leave_node_groups = self._determine_heuristic_values_leave_nodes()
        self._determine_heuristic_values_higher_nodes(leave_node_groups, nodes_visited)

    def _determine_heuristic_values_leave_nodes(self):
        """Determine heuristic values and enrich the nodes"""
        resources_process_executions_components = {}
        for process_executions_component in self.process_executions_components:
            process_executions_component: ProcessExecutionsComponent
            resources_process_executions_components.setdefault(process_executions_component.goal_item,
                                                               []).append(process_executions_component)

        resources_utilization = \
            {entity: _calculate_resource_utilization(entity, self.start_time, self.end_time)
             for entity in list(resources_process_executions_components.keys())
             if isinstance(entity, Resource)}
        resources_number_components = {resource: len(components)
                                       for resource, components in resources_process_executions_components.items()}
        # handle_leave_node heuristic values
        leave_node_groups = []
        for leave_node in self.all_leave_nodes:

            leave_node: HeuristicsSchedulingNode
            leave_node.add_branch_eval_leave_node(resources_utilization, resources_number_components)

            # go to the next higher level
            if leave_node.root:
                leave_node_group = leave_node.root
                if leave_node_group not in leave_node_groups:
                    leave_node_groups.append(leave_node_group)

        nodes_visited = copy(self.all_leave_nodes)

        return nodes_visited, leave_node_groups

    def _determine_heuristic_values_higher_nodes(self, leave_node_groups, nodes_visited):
        """Determine the heuristic values for the higher nodes"""

        if leave_node_groups:
            root_node_reached = False
        else:
            root_node_reached = True

        branch_nodes = leave_node_groups

        # iterate through the higher nodes until the root node and all his branches are considered
        while not root_node_reached:

            new_branch_nodes = []

            for branch_node in branch_nodes:
                sub_nodes = branch_node.get_sub_nodes()
                all_sub_nodes_visited = set(sub_nodes).issubset(nodes_visited)

                if all_sub_nodes_visited:
                    branch_node.add_branch_eval_higher_nodes()
                    nodes_visited.append(branch_node)
                    if branch_node.root:
                        new_branch_nodes.append(branch_node.root)

                else:
                    if branch_node.root:
                        if branch_node not in new_branch_nodes:
                            new_branch_nodes.append(branch_node)

            if not branch_nodes:
                if self.root_node not in nodes_visited:
                    raise Exception
                break

            branch_nodes = copy(new_branch_nodes)

    def _calculate_resource_utilization(self, resources_references):
        """Calculate the current utilization for each resource"""

        resources_utilization = {resource.identification: preference.get_utilization(self.start_time, self.end_time)
                                 for resource, preference in resources_references.items()}

        self.resources_utilization = resources_utilization

    def get_possible_schedule(self):
        """Get a possible schedule for the tree"""

        visited = []
        scheduled = []

        frontier_node = self.root_node
        last_frontier_node = None
        frontier_nodes = self.root_node.branches
        resources_preferences = {resource: resource_preference.get_copy()
                                 for resource, resource_preference in self.resources_preferences.items()}

        self._calculate_resource_utilization(resources_preferences)

        visited, scheduled, resources_preferences = \
            self.schedule_tree(last_frontier_node, frontier_node, frontier_nodes, visited, scheduled,
                               resources_preferences)

        # pd.set_option('display.width', 100)
        # pd.set_option('display.max_columns', 100)
        # pd.set_option('display.max_rows', 100)
        # for resource, preference in resources_preferences.items():
        #     print(f"Resource name: {resource.name}")
        #     print(preference._process_execution_plan_copy._time_schedule)
        #     print("--------------------------------------------")

        process_executions_components = \
            self.get_process_executions_components(scheduled, resources_preferences)

        return process_executions_components

    def schedule_tree(self, last_frontier_node, frontier_node, frontier_nodes, visited, scheduled,
                      resources_preferences):

        scheduling_finished = False
        iteration = 0
        while not scheduling_finished:
            iteration += 1

            last_frontier_node, frontier_node, frontier_nodes = \
                self._get_frontier_nodes(last_frontier_node, frontier_node, frontier_nodes, visited)

            if not frontier_nodes and (frontier_node is None or frontier_node in scheduled):
                break

            if frontier_node.type == SchedulingNode.Type.LEAF:
                self._calculate_resource_utilization(resources_preferences)
                frontier_nodes, last_frontier_node, resources_preferences, visited, scheduled = \
                    self._schedule_leave_nodes(frontier_nodes, frontier_node, resources_preferences, visited, scheduled)

        # print(f" Iteration: {iteration}")

        return visited, scheduled, resources_preferences

    def _get_frontier_nodes(self, last_frontier_node, frontier_node, frontier_nodes, visited):
        if last_frontier_node is not None:
            # if a frontier_node is already visited in the last round, the scheduling process is resumed
            # on the nearest possible point in the tree
            scheduling_finished, current_frontier_nodes = \
                _get_nearest_frontier_nodes(last_frontier_node=last_frontier_node, visited=visited)

            last_frontier_node = None

            if scheduling_finished:
                return last_frontier_node, frontier_node, frontier_nodes

        elif not frontier_nodes:
            frontier_node = None
            return last_frontier_node, frontier_node, frontier_nodes

        else:
            current_frontier_nodes = frontier_nodes
        # print(get_debug_str("SchedulingTree", "") + f"current_frontier_nodes: {current_frontier_nodes}")
        frontier_nodes, frontier_node = self.get_best_frontiers(frontier_node, current_frontier_nodes)

        if not frontier_node:
            # reset to root node
            frontier_nodes = self.root_node.branches
            if not frontier_nodes:
                return last_frontier_node, frontier_node, frontier_nodes
            frontier_node = frontier_nodes[0]
            return last_frontier_node, frontier_node, frontier_nodes

        if frontier_node.type == SchedulingNode.Type.AND or frontier_node.type == SchedulingNode.Type.OR:
            # frontier_nodes.extend(frontier_node.branches)
            frontier_nodes.extend(frontier_node.branches)

        return last_frontier_node, frontier_node, frontier_nodes

    def get_best_frontiers(self, frontier_node, alternative_nodes):
        frontier_nodes, frontier_node = \
            frontier_node.get_best_frontiers_branch(alternative_nodes, self.resources_utilization,
                                                    self._resource_order_binding_batch)

        return frontier_nodes, frontier_node

    def _schedule_leave_nodes(self, frontier_nodes, frontier_node, resources_preferences, visited, scheduled):
        """
        Schedule the leave nodes
        If neighbours (predecessor and successors) are found they are scheduled first before other nodes are considered
        """

        first = True
        predecessor_nodes = []
        successor_nodes = []
        batch_scheduled_interconnected = []
        batch_visited_interconnected = []
        first_frontier_node = frontier_node
        not_possible_first_frontier_nodes = []
        non_stationary_resource_positions_batch = self.non_stationary_resources_positions.copy()
        self._resource_order_binding_batch = self._resource_order_binding.copy()
        while successor_nodes or first:
            if first:
                successful, scheduled_batch, visited_batch, predecessor_nodes, successor_nodes = \
                    self._schedule_leave_node(frontier_node, resources_preferences, visited,
                                              non_stationary_resource_positions_batch, first_process_sub_issue=True)

            else:
                successful, frontier_node, scheduled_batch, visited_batch, predecessor_nodes, successor_nodes = \
                    self._schedule_predecessor_successor_nodes(frontier_node, resources_preferences, visited,
                                                               batch_scheduled_interconnected,
                                                               predecessor_nodes, successor_nodes,
                                                               non_stationary_resource_positions_batch, scheduled_batch)
            batch_scheduled_interconnected += scheduled_batch
            batch_visited_interconnected += visited_batch

            if successful is False:
                print(get_debug_str("Coordinator", "") + " Choose alternative nodes")
                # choose the alternative to the node
                not_possible_first_frontier_nodes.append(first_frontier_node)
                # first node from the path (connector not considered)
                if batch_scheduled_interconnected:
                    first_frontier_node = batch_scheduled_interconnected[0]
                alternative_nodes = self._get_alternative_node(first_frontier_node)

                batch_scheduled_interconnected, batch_visited_interconnected, scheduled, visited = \
                    self._reset_batch_interconnected(batch_scheduled_interconnected, batch_visited_interconnected,
                                                     scheduled, visited, resources_preferences)

                if not alternative_nodes:
                    if frontier_node is None:
                        frontier_node = first_frontier_node
                    scheduling_failed, frontier_node, scheduled, visited, frontier_nodes = \
                        self._set_branch_not_possible_to_schedule(scheduled, frontier_node,
                                                                  resources_preferences, frontier_nodes, visited)
                    if scheduling_failed:
                        print("Frontier node0:", frontier_node, len(scheduled))
                        return frontier_nodes, frontier_node, resources_preferences, visited, scheduled

                old_frontier_node = frontier_node
                while True:
                    frontier_nodes, frontier_node = self.get_best_frontiers(frontier_node, alternative_nodes)
                    if frontier_node is None:
                        scheduling_failed, frontier_node, scheduled, visited, frontier_nodes = \
                            self._set_branch_not_possible_to_schedule(scheduled, old_frontier_node,
                                                                      resources_preferences, frontier_nodes, visited)
                        print("Frontier node1:", None)
                        return frontier_nodes, None, resources_preferences, visited, scheduled

                    if frontier_node.type == SchedulingNode.Type.LEAF:
                        first_frontier_node = frontier_node
                        print("Frontier node2:", frontier_node)
                        break
                    alternative_nodes = frontier_nodes

                first = True

            elif first:
                first = False

            frontier_nodes, visited, scheduled = \
                _remove_nodes_scheduled_from_frontier_nodes(frontier_nodes, batch_scheduled_interconnected,
                                                            batch_visited_interconnected, visited, scheduled)

        self.non_stationary_resources_positions = non_stationary_resource_positions_batch
        self._resource_order_binding = self._resource_order_binding_batch

        # choose the highest node completely scheduled
        node_scheduled = frontier_node
        old_node_scheduled = node_scheduled

        while node_scheduled is not None:
            old_node_scheduled = node_scheduled
            node_scheduled = _determine_node_completely_scheduled(node_scheduled.root, visited)

        if old_node_scheduled not in visited and old_node_scheduled is not None:
            frontier_nodes, visited, scheduled = \
                _remove_nodes_scheduled_from_frontier_nodes(frontier_nodes, [old_node_scheduled], [], visited,
                                                            scheduled)

        new_scheduled = []
        scheduled_copy = copy(list(set(scheduled).difference(set(self.completely_scheduled))))
        while True:
            for node in scheduled_copy:
                if node.root is not None and node.root not in scheduled_copy:
                    if node.root.type == node.root.type.ROOT:

                        if node in node.root.branches:
                            node.root.branches.remove(node)
                        # print(get_debug_str("Scheduling", "") + " Scheduling successful")
                        frontier_nodes = node.root.branches  #  []  ToDo: scheduling changes
                        self.completely_scheduled.extend(scheduled_copy + new_scheduled)
                        old_node_scheduled = None
                        break
                    if node.root.type == node.root.type.AND:
                        if len(set(node.root.branches).intersection(set(scheduled_copy))) == len(node.root.branches):
                            new_scheduled.append(node.root)
                    elif node.root.type == node.root.type.OR:
                        if set(node.root.branches).intersection(set(scheduled_copy)):
                            new_scheduled.append(node.root)
            scheduled_copy.extend(new_scheduled)
            if not new_scheduled:
                break
            new_scheduled = []
        scheduled = list(set(scheduled))
        if self.completely_scheduled:
            visited = list(set(visited).difference(set(self.completely_scheduled)))
        return frontier_nodes, old_node_scheduled, resources_preferences, visited, scheduled

    def _schedule_predecessor_successor_nodes(self, frontier_node, resources_preferences, visited,
                                              batch_scheduled_interconnected, predecessor_nodes,
                                              successor_nodes, non_stationary_resource_positions_batch,
                                              scheduled_batch_before):
        """Schedule the predecessor/ the successor nodes for a node visited"""
        new_predecessor_nodes = []
        new_successor_nodes = []
        scheduled_batch = []
        additionally_visited = []
        frontier_node_path_ids = frontier_node.leave.path_ids
        frontier_node = None

        while predecessor_nodes or successor_nodes:
            if successor_nodes:
                type_ = "successor"
                frontier_node = successor_nodes.pop()
            else:
                raise Exception

            if frontier_node in batch_scheduled_interconnected:
                continue

            neighbour_node = frontier_node
            frontier_node = self._get_neighbour_node(neighbour_node, frontier_node_path_ids, type_)
            if frontier_node is not None:
                break

            additionally_visited.append(neighbour_node)  # case connector not needed
            if frontier_node is not None:
                scheduled_batch.append(frontier_node)

        if frontier_node is None:
            return (True, frontier_node, scheduled_batch, additionally_visited, new_predecessor_nodes,
                    new_successor_nodes)

        if type_ == "successor":
            predecessor_nodes = sorted(scheduled_batch_before, key=lambda x: x.time_slot[1])[-1]

        node_to_schedule = frontier_node

        successful, scheduled_batch, additionally_visited, predecessor_nodes_batch, successor_nodes_batch = \
            self._schedule_leave_node(node_to_schedule, resources_preferences, visited,
                                      non_stationary_resource_positions_batch, predecessor_nodes=predecessor_nodes)

        if not successful:
            return successful, node_to_schedule, scheduled_batch, additionally_visited, \
                new_predecessor_nodes, new_successor_nodes

        possible_predecessors = list(set(predecessor_nodes_batch).difference(set(visited)))
        if possible_predecessors:
            new_predecessor_nodes.extend(possible_predecessors)
        possible_successors = list(set(successor_nodes_batch).difference(set(visited)))
        if possible_successors:
            new_successor_nodes.extend(possible_successors)

        return successful, node_to_schedule, scheduled_batch, additionally_visited, new_predecessor_nodes, \
            new_successor_nodes

    def _get_neighbour_node(self, neighbour_node, frontier_node_path_ids, type_):
        """Determine the predecessor node/ successor node
        get direct predecessor nodes/ branch without successors to get the nearest predecessor node on a deeper level
        (on the same level the predecessor relations are available)
        :param frontier_node_path_ids:  restrictions for the neighbour nodes to ensure compatability
        """

        if neighbour_node.type == SchedulingNode.Type.LEAF:
            constraint_respected = _check_connection_constraint_respected(neighbour_node)  # node not needed
            if not constraint_respected:
                return None

            return neighbour_node

        frontier_nodes = [neighbour_node]
        while True:

            if type_ == "predecessor":
                frontier_nodes_neighbours = [frontier_node for frontier_node in frontier_nodes
                                             if not frontier_node.successors]
            elif type_ == "successor":
                frontier_nodes_neighbours = [frontier_node for frontier_node in frontier_nodes
                                             if not frontier_node.predecessors]
            else:
                raise Exception

            if frontier_nodes_neighbours:
                frontier_nodes = frontier_nodes_neighbours

            frontier_nodes = [frontier_node for frontier_node in frontier_nodes
                              if _check_path_ids_constraint(frontier_node, frontier_node_path_ids)]

            frontier_nodes, frontier_node = self.get_best_frontiers(neighbour_node, frontier_nodes)
            if frontier_node is None:
                return None
            elif frontier_node.type == SchedulingNode.Type.LEAF:
                break

            frontier_nodes.extend(frontier_node.branches)
        neighbour_node = frontier_node

        return neighbour_node

    def _get_alternative_node(self, first_frontier_node):
        """Determine an alternative to the frontier_node because the node is not possible etc."""
        # ToDo: for the
        all_alternatives = self._get_or_branch(node=first_frontier_node)

        if all_alternatives is None:
            return []

        if first_frontier_node not in all_alternatives:
            raise Exception("Further code needed - because the whole sub_tree for the scheduled frontier_node must be "
                            "deleted")

        if first_frontier_node in all_alternatives:
            all_alternatives.remove(first_frontier_node)
        possible_alternatives = all_alternatives

        return possible_alternatives

    def _get_or_branch(self, node):
        """Get the next highest or branch
        Exception: in the case that the node act as connector than also 'and' nodes can be a kind of or nodes
        """
        while True:
            root = node.root
            if root is None:
                return None
            elif root.type == SchedulingNode.Type.OR:
                return root.branches

            node = root

    def _reset_batch_interconnected(self, batch_scheduled_interconnected, batch_visited_interconnected,
                                    scheduled, visited, resources_preferences):
        """Reset the scheduled batch because it was not possible"""

        scheduled = list(set(scheduled).difference(set(batch_scheduled_interconnected)))
        visited = list(set(visited).difference(set(batch_visited_interconnected)))

        for scheduled_node in batch_scheduled_interconnected:
            scheduled_node.reset_time_slot(resources_preferences)
            self._reset_connector_object(batch_scheduled_interconnected)

        batch_scheduled_interconnected = []
        batch_visited_interconnected = []
        return batch_scheduled_interconnected, batch_visited_interconnected, scheduled, visited

    def _schedule_leave_node(self, frontier_node, resources_preferences, visited,
                             non_stationary_resource_positions_batch, possible_start_time_stamps=None,
                             first_process_sub_issue=False, predecessor_nodes=None, successor_nodes=None):
        """Schedule the leave node/ leave nodes if other leave nodes participate in the process_execution to schedule"""

        (process_executions_components, process_executions_component, process_execution_preference,
            scheduled_batch, additionally_visited, nodes_other_leaves, frontier_node, enough_nodes) = \
            self._get_process_execution_side(frontier_node, resources_preferences)

        if not enough_nodes:
            return False, [], [], [], []

        # print(get_debug_str("SchedulingTree", "") + " Begin leave_node scheduling")

        # resource side
        entities_to_schedule = _get_entities_to_schedule(process_executions_components=process_executions_components)
        resources_preferences_batch = _get_resource_preference(resources_preferences, entities_to_schedule)
        nodes_scheduled_batch = []
        new_additionally_visited = []
        if first_process_sub_issue:
            # connector_nodes_to_schedule, connector_needed, possible_start_time_stamps, failed = \
            #     self._handle_connector(process_executions_component, frontier_node, entities_to_schedule,
            #                            resources_preferences)
            # if failed:
            #     print("Failed", frontier_node, frontier_node.leave.goal_items[0].name, frontier_node.connector,
            #           frontier_node.issue_id)
            #     successful = False
            #     return successful, scheduled_batch, additionally_visited, [], []
            # el
            if not frontier_node.connector:
                possible_frontier_nodes = [node for node in nodes_other_leaves if node.connector]
                if possible_frontier_nodes:
                    print("Frontier node connector changed")
                    nodes_other_leaves.append(frontier_node)
                    frontier_node = possible_frontier_nodes[0]
                    nodes_other_leaves.remove(frontier_node)
                    scheduled_batch = [frontier_node] + nodes_other_leaves

            if frontier_node.connector:
                # print("Frontier node connector:",
                #       [resource.name for resource in frontier_node.connector.component.goal_item.get_resources()])
                visited_connector = []
                scheduled_connector = []

                frontier_node_connector = frontier_node.connector
                last_frontier_node_connector = None
                frontier_nodes_connector = frontier_node.connector.branches

                new_additionally_visited, nodes_scheduled_batch, resources_preferences = \
                    self.schedule_tree(last_frontier_node_connector, frontier_node_connector, frontier_nodes_connector,
                                       visited_connector, scheduled_connector, resources_preferences)

                try:
                    predecessor_nodes = [sorted(nodes_scheduled_batch, key=lambda x: x.time_slot[0])[-1]]
                except:
                    print("Failed...")
                    successful = False
                    return successful, [], new_additionally_visited, [], []
                successor_nodes = []
                if not nodes_scheduled_batch:
                    print("Failed..")
                    successful = False
                    return successful, [], new_additionally_visited, [], []

        # ToDo: should contain the predecessor process identifier on the process_executions_plan
        #  to know where to plan the successor

        successful, predecessor_nodes, successor_nodes = \
            self._schedule(process_executions_component, process_execution_preference, frontier_node,
                           nodes_other_leaves, resources_preferences_batch, entities_to_schedule,
                           scheduled_batch, additionally_visited, visited,
                           non_stationary_resource_positions_batch, predecessor_nodes, successor_nodes,
                           possible_start_time_stamps)

        scheduled_batch += nodes_scheduled_batch
        additionally_visited += new_additionally_visited

        return successful, scheduled_batch, additionally_visited, predecessor_nodes, successor_nodes

    def _get_process_execution_side(self, frontier_node, resources_preferences):
        """Returns the process execution side"""
        process_executions_component = frontier_node.leave
        other_leave_nodes, root_node_chosen, interim_nodes = _get_other_leave_nodes(frontier_node)

        needed_nodes = len(other_leave_nodes) + 1

        if other_leave_nodes:
            # joint planning needed
            nodes_other_leaves = self.deciding_for_alternative_nodes(other_leave_nodes, resources_preferences)
            process_executions_components_other_leaves = [node_other_leave.leave
                                                          for node_other_leave in nodes_other_leaves]
            process_executions_components = [process_executions_component] + process_executions_components_other_leaves

        else:
            nodes_other_leaves = []
            process_executions_components = [process_executions_component]

        process_executions_component = _get_process_executions_component_from_node(root_node_chosen,
                                                                                   process_executions_components)

        if not isinstance(process_executions_component.get_entity_used(), ActiveMovingResource) and other_leave_nodes:
            nodes_other_leaves, frontier_node = self._switch_frontier_node(nodes_other_leaves, frontier_node)

        # process_executions_variant needed (that only one of the variants - can be with other entities)
        self._set_connector_object(process_executions_component, root_node_chosen, nodes_other_leaves, frontier_node)

        nodes_scheduled_batch = list(set([frontier_node] + nodes_other_leaves))
        if frontier_node.root:
            root_list = [frontier_node.root]
        else:
            root_list = []
        additionally_visited = list(set(interim_nodes + root_list))

        process_execution_preference = process_executions_component.reference_preference

        enough_nodes = True
        if len(process_executions_components) != needed_nodes:
            enough_nodes = False

        return (process_executions_components, process_executions_component, process_execution_preference,
                nodes_scheduled_batch, additionally_visited, nodes_other_leaves, frontier_node, enough_nodes)

    def deciding_for_alternative_nodes(self, other_leave_nodes, resources_preferences, ):
        """
        For multi resource planning more than one resource should be treated at the same time
        :return: the best alternative nodes
        """

        nodes_chosen = []
        for alternative_nodes in other_leave_nodes:
            alternative_nodes = _consider_constraints(alternative_nodes, self._resource_order_binding)

            if len(alternative_nodes) == 1:
                nodes_chosen.append(alternative_nodes[0])
                continue

            alternative_nodes_evaluation = \
                np.array([self._evaluate_alternative_node(alternative_node, resources_preferences)
                          for alternative_node in alternative_nodes])
            if alternative_nodes_evaluation.size == 0:
                continue
            try:
                process_requests_winner = alternative_nodes_evaluation[:, 0].argmax()
            except:
                print("Problem...", alternative_nodes_evaluation)
                process_requests_winner = alternative_nodes_evaluation[:, 1].argmax()  # ToDo: Why?
            availability_winner = alternative_nodes_evaluation[:, 1].argmax()

            if process_requests_winner != availability_winner:
                # ToDo: currently only the availability_winner is crucial for the decision
                nodes_chosen.append(alternative_nodes[availability_winner])
            else:
                nodes_chosen.append(alternative_nodes[availability_winner])

        return nodes_chosen

    def _evaluate_alternative_node(self, alternative_node, resources_preferences):
        """
        Evaluate each alternative with the decision criterion's availability (in the requested time_period) and
        other planning available the resource
        """

        # 'resource_process_requests'
        resource_process_requests = self._get_resource_process_requests(alternative_node)
        process_executions_component_alternative = alternative_node.leave
        resource_alternative = process_executions_component_alternative.get_entity_used()
        if isinstance(resource_alternative, Resource):
            preference_resource_alternative: EntityPreference = resources_preferences[resource_alternative]
            preference_alternative = process_executions_component_alternative.reference_preference
            try:
                start_time = preference_alternative.get_first_accepted_time_stamp()
                end_time = preference_alternative.get_last_accepted_time_stamp()
                duration = end_time - start_time
                utility = preference_resource_alternative.get_utilization(start_time, end_time)
                decision_factor_availability = (1 - utility) * duration
            except IndexError:
                decision_factor_availability = 0
        else:
            decision_factor_availability = 0
        return resource_process_requests, decision_factor_availability

    def _get_resource_process_requests(self, alternative_node):
        return alternative_node.branch_eval[type(self).alternative_selection_param][0][0]

    def _switch_frontier_node(self, nodes_other_leaves, frontier_node):
        """
        switch the frontier_node to a non_stationary_resource (if a non_stationary resource in nodes_other_leaves)
        because the non_stationary_resource could need a connector
        """
        non_stationary_nodes = [node for node in nodes_other_leaves
                                if isinstance(node.leave.get_entity_used(), NonStationaryResource)]

        if len(non_stationary_nodes) < 1:
            pass
        elif len(non_stationary_nodes) > 1:
            non_stationary_nodes = [node for node in nodes_other_leaves
                                    if isinstance(node.leave.get_entity_used(), ActiveMovingResource)]

            if len(non_stationary_nodes) > 1:
                raise NotImplementedError

        if non_stationary_nodes:
            nodes_other_leaves.append(frontier_node)
            frontier_node = non_stationary_nodes[0]
            nodes_other_leaves.remove(frontier_node)

        return nodes_other_leaves, frontier_node

    def _set_connector_object(self, process_executions_component, root_node_chosen, nodes_other_leaves, frontier_node):
        """Set a connector object if necessary
        Connector objects limits the node selection because they enforce the selection of already chosen resources
        for the same issue"""

        issue_id = process_executions_component.issue_id

        while True:
            root_node_parent = process_executions_component.process_executions_components_parent
            if not root_node_parent:
                return

            process_executions_component = root_node_parent
            if process_executions_component.issue_id != issue_id:
                return

            if not hasattr(process_executions_component, "connector_object"):
                continue
            connector_objects = process_executions_component.connector_objects

            connector_objects_ = \
                [node.leave.get_entity_used()
                 for node in nodes_other_leaves + [frontier_node]
                 if node.leave.get_entity_used() in connector_objects]

            if connector_objects_:
                if not root_node_chosen.root:  # connetor tree
                    pass
                elif root_node_chosen.root.issue_id == issue_id:
                    root_node_chosen.root.add_connector_object_constraint(connector_objects_[0])
                elif root_node_chosen.issue_id == issue_id:
                    root_node_chosen.add_connector_object_constraint(connector_objects_[0])

    def _reset_connector_object(self, scheduled_batch):
        """Reset connector object constraints because the scheduling was not possible"""

        new_scheduled_batch = []

        while scheduled_batch:
            # find the shared root node
            for scheduled_node in scheduled_batch:
                scheduled_node.remove_connector_object_constraint()

                if scheduled_node.root is None:
                    continue
                if scheduled_node.root.issue_id == scheduled_node.issue_id:
                    new_scheduled_batch.append(scheduled_node.root)

            scheduled_batch = new_scheduled_batch.copy()
            new_scheduled_batch = []

    def _get_predecessor_relationships(self, nodes_scheduled_batch, additionally_visited, nodes_scheduled):
        """Determine the predecessor nodes that are already visited and take their times """
        predecessor_nodes = []  # only the neighbors of the deepest node are considered if not already visited
        nodes_visited = nodes_scheduled_batch + additionally_visited
        nodes_visited.sort(key=lambda x: x.level, reverse=True)
        current_level = 0
        for node_scheduled in nodes_visited:
            if node_scheduled.level < current_level and node_scheduled not in nodes_scheduled and predecessor_nodes:
                break
            current_level = node_scheduled.level
            predecessor_nodes.extend(node_scheduled.predecessors)

        predecessor_nodes_scheduled = [predecessor_node for predecessor_node in predecessor_nodes
                                       if predecessor_node in nodes_scheduled]
        predecessor_nodes_not_scheduled = [predecessor_node for predecessor_node in predecessor_nodes
                                           if predecessor_node not in nodes_scheduled]

        predecessor_process_planned = None
        if predecessor_nodes_scheduled:
            predecessor_process_planned = self._get_latest_end_time(predecessor_nodes_scheduled)

        block_before = False
        if predecessor_process_planned:
            if predecessor_process_planned.issue_id == nodes_scheduled_batch[0].issue_id:
                block_before = True

        return predecessor_process_planned, predecessor_nodes_not_scheduled, block_before

    def _get_latest_end_time(self, predecessor_nodes_scheduled):
        latest_end_times = [predecessor_node_scheduled.get_latest_end_time()
                            for predecessor_node_scheduled in list(set(predecessor_nodes_scheduled))
                            if not nan_value(predecessor_node_scheduled.get_latest_end_time())]

        if latest_end_times:
            latest_end_time = max(latest_end_times, key=lambda t: t[1])

            predecessor_process_planned = latest_end_time[0]
        else:
            raise Exception

        return predecessor_process_planned

    def get_neighbour_relationships(self, nodes_scheduled_batch, additionally_visited, visited):
        """Get the predecessors and successors"""

        predecessor_process_planned, predecessor_nodes, block_before = \
            self._get_predecessor_relationships(nodes_scheduled_batch, additionally_visited, visited)
        successor_process_planned, successor_nodes = \
            self._get_successor_relationships(nodes_scheduled_batch, additionally_visited, visited)

        return predecessor_process_planned, predecessor_nodes, block_before, successor_process_planned, successor_nodes

    def _get_successor_relationships(self, nodes_scheduled_batch, additionally_visited, nodes_scheduled):
        """Determine the successor nodes that are already visited and take their times"""
        successor_nodes = []  # only the neighbors of the deepest node are considered if not already visited
        nodes_visited = nodes_scheduled_batch + additionally_visited
        nodes_visited.sort(key=lambda x: x.level, reverse=True)
        current_level = 0
        for node_scheduled in nodes_visited:
            if node_scheduled.level < current_level and node_scheduled not in nodes_scheduled and successor_nodes:
                break
            current_level = node_scheduled.level
            successor_nodes.extend(node_scheduled.successors)

        successor_nodes_scheduled = [successor_node for successor_node in successor_nodes
                                     if successor_node in nodes_scheduled]
        successor_nodes_not_scheduled = [successor_node for successor_node in successor_nodes
                                         if successor_node not in nodes_scheduled]

        successor_process_planned = None
        if successor_nodes_scheduled:
            earliest_start_times = [successor_node_scheduled.get_earliest_start_time()
                                    for successor_node_scheduled in successor_nodes_scheduled
                                    if not nan_value(successor_node_scheduled.get_earliest_start_time())]

            if earliest_start_times:
                earliest_start_time = min(earliest_start_times, key=lambda t: t[1])
                successor_process_planned = earliest_start_time[0]
            else:
                raise Exception

        return successor_process_planned, successor_nodes_not_scheduled

    def _schedule(self, process_executions_component, process_execution_preference, frontier_node, nodes_other_leaves,
                  resources_preferences_batch, entities_to_schedule, nodes_to_schedule, additionally_visited, visited,
                  non_stationary_resource_positions_batch, predecessor_nodes=None, successor_nodes=None,
                  possible_start_time_stamps=None):
        """Determine the neighbour relationships and schedule"""

        block_before, predecessor_process_planned, successor_process_planned, predecessor_nodes, successor_nodes = \
            self.get_neighbour_relations(successor_nodes, predecessor_nodes, nodes_to_schedule, additionally_visited,
                                         visited, possible_start_time_stamps)

        process_execution = process_executions_component.goal
        process_execution_id = process_execution.identification
        # print("Process Name: ", process_execution.process.name)
        issue_id = process_executions_component.issue_id

        work_order_id = process_execution.order.identification

        if resources_preferences_batch:

            # t1 = time.process_time()
            # print("Scheduled node identification", process_executions_component.node_identification)
            successful, best_matching_time_slot = \
                process_execution_preference.schedule_resources(
                    resources_preferences_batch, predecessor_process_planned, successor_process_planned,
                    process_execution_id, work_order_id, issue_id=issue_id, blocker_name="CentralCoordinator",
                    block_before=block_before)

            # t2 = time.process_time()
            # print("Scheduling Time")

            if successful:
                process_execution = process_executions_component.goal_item
                if "Resource" in entities_to_schedule:
                    process_execution.resources_used += [(entity,)
                                                         for entity in entities_to_schedule["Resource"]
                                                         if entity not in process_execution.get_resources()]
                try:
                    process_execution.main_resource = \
                        [resource
                         for resource in entities_to_schedule["Resource"]
                         if process_execution.process.check_ability_to_perform_process_as_main_resource(resource)][0]
                except:
                    raise Exception(process_execution.process.name, entities_to_schedule["Resource"],
                                    process_execution.order.external_identifications)

                if "Part" in entities_to_schedule:
                    process_execution.parts_involved += [(entity,)
                                                         for entity in entities_to_schedule["Part"]
                                                         if entity not in process_execution.get_parts()]

                possible_origins = process_execution.get_possible_origins()
                possible_destinations = process_execution.get_possible_destinations()
                if len(possible_origins) > 1 or len(possible_destinations) > 1:
                    print("Scheduling Process Executions")
                    raise Exception("Scheduling Process Executions")
                if len(possible_origins) == 1:
                    process_execution.origin = possible_origins[0]
                if len(possible_destinations) == 1:
                    process_execution.destination = possible_destinations[0]
            else:
                print("Not successful...")

        else:
            successful, best_matching_time_slot = \
                process_execution_preference.schedule_parts(predecessor_process_planned, successor_process_planned)
            print("PARTS")
            raise Exception("PARTS")

        if successful:
            self._after_scheduling(process_executions_component, frontier_node, nodes_other_leaves,
                                   resources_preferences_batch, entities_to_schedule, successor_nodes,
                                   non_stationary_resource_positions_batch, best_matching_time_slot,
                                   process_execution_id)

        return successful, predecessor_nodes, successor_nodes

    def get_neighbour_relations(self, successor_nodes, predecessor_nodes, nodes_to_schedule, additionally_visited,
                                visited, possible_start_time_stamps):

        if successor_nodes is None:
            predecessor_process_planned, predecessor_nodes, block_before, successor_process_planned, successor_nodes = \
                self.get_neighbour_relationships(nodes_to_schedule, additionally_visited, visited)
        elif predecessor_nodes is not None:
            # case: the node after the connector node (connector as a predecessor)
            block_before = True
            predecessor_process_planned = self._get_latest_end_time(predecessor_nodes)
            successor_process_planned, successor_nodes = \
                self._get_successor_relationships(nodes_to_schedule, additionally_visited, visited)

        elif possible_start_time_stamps:
            block_before = False
            predecessor_process_planned = possible_start_time_stamps[0]
            predecessor_nodes = []
            successor_process_planned = None

        else:
            # case for the connector node
            block_before = False
            predecessor_nodes = []
            predecessor_process_planned = None
            successor_process_planned = None

        return block_before, predecessor_process_planned, successor_process_planned, predecessor_nodes, successor_nodes

    def _after_scheduling(self, process_executions_component, frontier_node, nodes_other_leaves,
                          resources_preferences_batch, entities_to_schedule, successor_nodes,
                          non_stationary_resource_positions_batch, best_matching_time_slot, process_execution_id):

        if resources_preferences_batch:
            self._set_long_time_reservation(process_executions_component, entities_to_schedule)
            self._update_resources_positions(process_executions_component, entities_to_schedule["Resource"],
                                             best_matching_time_slot, successor_nodes,
                                             non_stationary_resource_positions_batch)

        # Add time slots to the visited nodes
        leave_nodes = [frontier_node] + nodes_other_leaves
        for leave_node in leave_nodes:
            leave_node.add_process_execution_id(process_execution_id)
        for other_leave_node in leave_nodes:
            other_leave_node.add_time_slot(start_time=best_matching_time_slot[0], end_time=best_matching_time_slot[1])

    def _handle_connector(self, process_executions_component, frontier_node, entities_to_schedule,
                          resources_preferences):
        """If it is a connector node, it can remain unchanged, it can be adapted because another transport path
        is needed, or it can be removed because the destination is the wanted origin
        If it is not a connector node it is possible that a connector node is nevertheless needed"""

        connector_node_high = frontier_node.connector  # ToDo: connector
        if connector_node_high:
            if len(connector_node_high.branches) == 1:
                connector_node = connector_node_high.branches[0].branches[0]
            elif len(connector_node_high.branches) == 3:
                connector_node = self.get_connector_branch(connector_node_high)
            else:
                raise NotImplementedError(frontier_node, len(connector_node_high.branches),
                                          frontier_node.leave.goal_item.name)

            connector_component = connector_node.leave.process_executions_components_parent
        else:
            connector_node = None
            connector_component = None

        # should be executed for the first process of the process_executions_path
        connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged, connector_preference = \
            self._determine_connector_needed(process_executions_component, frontier_node, connector_node,
                                             entities_to_schedule, resources_preferences)

        connector_nodes_to_schedule, connector_needed, possible_start_time_stamps, failed = \
            self._evaluate_consequences(frontier_node, connector_node, entities_to_schedule,
                                        connector_component, connector_preference,
                                        resources_preferences, connector_possibly_not_needed,
                                        connector_possibly_needed, distances_to_be_bridged)

        return connector_nodes_to_schedule, connector_needed, possible_start_time_stamps, failed

    def get_connector_branch(self, connector_node_high):
        """Get the leave branch that is responsible for the spatial transfer"""
        for sub_branch in connector_node_high.branches:
            preference = sub_branch.branches[0].leave.reference_preference
            if preference.destination is not None and preference.origin is not None:
                return sub_branch.branches[0]

    def _determine_connector_needed(self, process_executions_component, frontier_node, connector_node,
                                    entities_to_schedule, resources_preferences):
        """Determine if transport access for a resource is needed or not.
        Assuming that it is planned only step by step
        """
        distances_to_be_bridged = {}
        connector_possibly_needed = False
        connector_possibly_not_needed = False

        if frontier_node.leave.reference_preference.long_time_reservation_duration:
            # organized in a separate scheduling session
            if not connector_node:
                return connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged, None

        elif not isinstance(frontier_node.leave.get_entity_used(), NonStationaryResource):
            connector_possibly_not_needed = True
            return connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged, None

        if isinstance(frontier_node.leave.get_entity_used(), StationaryResource):
            connector_preference = None
            return connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged, \
                connector_preference

        # check if entities_to_schedule can be chosen for the process
        target_start_position = process_executions_component.get_origin()

        if connector_node:
            connector_component = connector_node.leave
            target_start_position_connector = connector_component.get_origin()
            connector_preference = connector_component.reference_preference
            connector_path_length = connector_preference.expected_process_execution_time

        else:
            connector_preference = None
            target_start_position_connector = None
            connector_path_length = 0

        accepted_time_periods = process_executions_component.reference_preference.accepted_time_periods.copy()

        needed_path_length = self._get_issue_execution_time_length(process_executions_component, frontier_node)

        for entity_to_schedule in entities_to_schedule:
            if entity_to_schedule not in self.non_stationary_resources_positions:
                continue

            positions = self.non_stationary_resources_positions[entity_to_schedule].copy()
            if positions:
                positions = dict([positions.popitem()])
            resource_preference = resources_preferences[entity_to_schedule]

            transport_access_needed_d, transport_access_not_needed_d = \
                self._connector_needed(positions, target_start_position, target_start_position_connector,
                                       resource_preference, needed_path_length, connector_path_length,
                                       accepted_time_periods)

            distances_to_be_bridged[entity_to_schedule] = {"to_bridge": transport_access_needed_d,
                                                           "not_to_bridge": transport_access_not_needed_d}
            if transport_access_needed_d:
                connector_possibly_needed = True
            if transport_access_not_needed_d:
                connector_possibly_not_needed = True

        return connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged, connector_preference

    def _connector_needed(self, positions, target_start_position, target_start_position_connector,
                          resource_preference, needed_path_length_without_connector, connector_path_length,
                          accepted_time_periods):
        """Determine for different time_stamps/ positions the resource visit if a
        bridging/ connection/ transport_access is needed
        :param accepted_time_periods: to be related to the frontier_node
        """

        transport_access_not_needed_d = {}
        transport_access_needed_d = {}
        positions = sorted(list(positions.items()), key=lambda t: t[0])
        # print("Positions:", positions)
        for idx, (timestamp, position_resource) in enumerate(positions):

            possible_accepted_time_period = accepted_time_periods[accepted_time_periods[:, 1] >= timestamp]
            if possible_accepted_time_period.size == 0:
                continue

            if possible_accepted_time_period[0, 0] > timestamp:
                if len(positions) > idx + 1:
                    if accepted_time_periods[0][0] >= positions[idx + 1][0]:
                        continue

            possible_accepted_time_period[0, 0] = timestamp

            possible_start_time = possible_accepted_time_period[0][0]
            possible_end_time = possible_accepted_time_period[-1][1]

            possible_periods = \
                resource_preference.get_free_time_periods(
                    time_slot_duration=needed_path_length_without_connector.item(),
                    start_time=possible_start_time, end_time=possible_end_time)

            # decide if it is worth to schedule the path
            if possible_periods.shape[0] == 0:
                continue

            possible_period = possible_periods[0]
            if possible_period.size == 0:
                continue

            # print("Possible period: ", possible_period)
            possible_period_length = (possible_period[1] - possible_period[0]).item().seconds

            if possible_period_length < needed_path_length_without_connector:
                continue

            if target_start_position is not None:
                if position_resource.check_intersection_base_areas(target_start_position):
                    if possible_period_length >= needed_path_length_without_connector:
                        connector_needed = False
                        transport_access_not_needed_d[timestamp] = (position_resource,
                                                                    connector_needed)
                        continue  # connector is not wished

            if target_start_position_connector is None:
                transport_access_not_needed_d[timestamp] = (position_resource, target_start_position)

            elif position_resource == target_start_position_connector or \
                    position_resource.check_intersection_base_areas(target_start_position_connector):

                connector_needed = True  # can be additional or the process_executions_component/ node given
                transport_access_not_needed_d[timestamp] = (position_resource, connector_needed)

            else:
                transport_access_needed_d[timestamp] = (position_resource, target_start_position)
                break

        return transport_access_needed_d, transport_access_not_needed_d

    def _evaluate_consequences(self, frontier_node, connector_node, entities_to_schedule, connector_component,
                               connector_preference, resources_preferences,
                               connector_possibly_not_needed, connector_possibly_needed, distances_to_be_bridged):

        connector_needed = True
        connector_nodes_to_schedule = []
        failed = False

        if not distances_to_be_bridged:
            connector_needed = False

        elif connector_possibly_not_needed and not connector_possibly_needed:
            # no connector is needed, therefore, if the node is a connector, it is skipped
            if connector_node:
                possible_needed_entities = []
                for entity_to_schedule in entities_to_schedule:
                    if entity_to_schedule in distances_to_be_bridged:
                        possible_needed_entities = \
                            list(distances_to_be_bridged[entity_to_schedule]["not_to_bridge"].values())
                        break

                possible_needed = [needed for resource, needed in possible_needed_entities]
                if not sum(possible_needed):
                    connector_needed = False

        elif connector_possibly_not_needed:
            # choose the best strategy
            if connector_node:
                possible_needed_entities = []
                for entity_to_schedule in entities_to_schedule:
                    if entity_to_schedule in distances_to_be_bridged:
                        possible_needed_entities = \
                            list(distances_to_be_bridged[entity_to_schedule]["not_to_bridge"].values())
                        break
                possible_needed = [needed for resource, needed in possible_needed_entities]
                if not sum(possible_needed):
                    connector_needed = False

        elif connector_possibly_needed:
            # connector needed
            if connector_node:
                distances_to_be_bridged, distances_to_be_bridged_alternatives = \
                    self._adapt_connector(distances_to_be_bridged, entities_to_schedule, connector_component,
                                          connector_preference, resources_preferences, frontier_node)

            else:
                failed = True
                # heuristics failure
                # distances_to_be_bridged, distances_to_be_bridged_alternatives = \
                #     self._create_connectors(distances_to_be_bridged, resources_preferences, frontier_node)
                # if distances_to_be_bridged:
                #     connector_nodes_to_schedule = reduce(concat, list(distances_to_be_bridged.values()))

        else:
            # try to choose the alternative resource
            failed = True

        if connector_needed and not failed:
            if connector_node is None:
                failed = True
            connector_nodes_to_schedule = [connector_node]

        possible_start_time_stamps = []
        for dict_ in list(distances_to_be_bridged.values()):
            if dict_:
                if "not_to_bridge" in dict_:
                    possible_start_time_stamps.extend(list(dict_["not_to_bridge"].keys()))
                elif "to_bridge" in dict_:
                    possible_start_time_stamps.extend(list(dict_["to_bridge"].keys()))

        return connector_nodes_to_schedule, connector_needed, possible_start_time_stamps, failed

    def _adapt_connector(self, distances_to_be_bridged, entities_to_schedule, process_executions_component,
                         process_execution_preference, resources_preferences, frontier_node):
        """If the origin is changed the connector should be adapted accordingly"""

        distances_to_be_bridged_expanded = distances_to_be_bridged.copy()
        for resource, transport_access_needed_d in distances_to_be_bridged.items():
            for time_stamp, (origin_resource, destination_resource) in transport_access_needed_d["to_bridge"].items():
                old_time_period = process_execution_preference.expected_process_execution_time

                needed_path_length = self._get_issue_execution_time_length(process_executions_component, frontier_node)

                successful, connector_process_execution_preferences = \
                    self._create_connector(resource, origin_resource, destination_resource, resources_preferences,
                                           process_execution_preference, time_stamp, needed_path_length)
                if not successful:
                    continue
                # adapt the process_execution_component
                self._adapt_process_execution_component(entities_to_schedule, process_executions_component,
                                                        connector_process_execution_preferences)

                new_time_period = connector_process_execution_preferences[0][1].expected_process_execution_time
                difference = new_time_period - old_time_period
                if difference < 0:
                    # adapt the successor node accepted_time_period
                    mask = frontier_node.leave.reference_preference.accepted_time_periods[:, 0] <= time_stamp
                    if frontier_node.leave.reference_preference.accepted_time_periods[mask].size == 0:
                        break
                    frontier_node.leave.reference_preference.accepted_time_periods[mask][0] += difference

                distances_to_be_bridged_expanded[resource] = transport_access_needed_d
                break  # ToDo: maybe later also the other possibilities should be considered

        return distances_to_be_bridged_expanded, distances_to_be_bridged

    def _adapt_process_execution_component(self, entities_to_schedule, process_executions_component,
                                           connector_process_execution_preferences):
        """Adapt the process_executions_component"""
        if len(connector_process_execution_preferences) > 1:
            raise NotImplementedError
        for process_execution, preference in connector_process_execution_preferences:
            process_executions_component.update_connector(preference, process_execution)

    def _create_connectors(self, distances_to_be_bridged, resources_preferences, frontier_node):
        """Create transport access objects/ process_execution_components"""
        process_executions_component = frontier_node.leave
        process_execution_preference = process_executions_component.reference_preference
        distances_to_be_bridged_expanded = distances_to_be_bridged.copy()
        for resource, transport_access_needed_d in distances_to_be_bridged.items():
            for time_stamp, (origin_resource, destination_resource) in transport_access_needed_d["to_bridge"].items():
                needed_path_length = self._get_issue_execution_time_length(process_executions_component, frontier_node)

                successful, connector_process_execution_preferences = \
                    self._create_connector(resource, origin_resource, destination_resource, resources_preferences,
                                           process_execution_preference, time_stamp, needed_path_length)
                if not successful:
                    continue

                connection_process_executions_component = \
                    self._get_connection_process_executions_component(process_executions_component, resource)

                connector_process_executions_variants_paths = \
                    [connection_process_executions_component.create_connector(preference, process_execution)
                     for process_execution, preference in connector_process_execution_preferences]
                connector_process_executions_variants, connector_process_executions_paths = \
                    list(zip(*connector_process_executions_variants_paths))

                connector_nodes = []
                if len(connector_process_executions_paths) > 1:
                    raise NotImplementedError
                for connector_process_executions_path in connector_process_executions_paths:
                    connector_node = \
                        HeuristicsSchedulingNode(identification=str(connector_process_executions_path.cfp_path),
                                                 root=None, type_=SchedulingNode.Type.LEAVE,
                                                 issue_id=connector_process_executions_path.issue_id,
                                                 leave=connector_process_executions_path)

                    frontier_node.connector = connector_node
                    connector_node.new = True
                    connector_nodes.append(connector_node)

                distances_to_be_bridged_expanded[resource] = connector_nodes

                break  # ToDo: maybe later also the other possibilities should be considered

        return distances_to_be_bridged_expanded, distances_to_be_bridged

    def _get_next_and_node(self, node):
        """Get the next highest and branch node"""
        while True:
            root = node.root
            if root is None:
                return None
            elif root.type == SchedulingNode.Type.AND:
                return root
            node = root

    def _get_connection_process_executions_component(self, process_executions_component, resource):
        """process_executions_component should be the first process in the chain
        find the component that need the connection/ transport access"""
        connection_process_executions_components = \
            [process_executions_component
             for process_executions_component in process_executions_component.get_process_executions_components_lst()
             if resource == process_executions_component.goal_item]

        if len(connection_process_executions_components) != 1:
            raise Exception(resource.name,
                            [process_executions_component.goal_item.name
                             for process_executions_component in
                             process_executions_component.get_process_executions_components_lst()])
        connection_process_executions_component = connection_process_executions_components[0]

        return connection_process_executions_component

    def _create_connector(self, resource, origin_resource, destination_resource, resources_preferences,
                          process_execution_preference, time_stamp, needed_path_length):
        """Create a connector to cover the distance between origin and destination"""

        support_entity_type = resource.entity_type
        if isinstance(resource, NonStationaryResource):
            transport = False
        else:
            transport = True

        transport_processes = \
            self.routing_service.get_transit_processes(origin_resource, destination_resource,
                                                       support_entity_type=support_entity_type,
                                                       transport=transport)

        connector_process_execution_preferences, successful = \
            self._get_preferences_connector(resource, transport_processes, resources_preferences,
                                            process_execution_preference, time_stamp, needed_path_length)

        return successful, connector_process_execution_preferences

    def _get_preferences_connector(self, resource, transport_processes, resources_preferences,
                                   process_execution_preference, time_stamp, needed_path_length):
        """
        Get the preferences for the transport access (later used for scheduling)
        """
        successful = True
        lead_time = {}
        follow_up_time = {}
        if resource in process_execution_preference.follow_up_time:
            follow_up_time[resource] = process_execution_preference.follow_up_time[resource]
        if len(transport_processes) > 1:
            raise Exception
        connector_process_execution_preferences = []
        for transport_process_d in transport_processes:
            process_execution = process_execution_preference.reference_objects[0]
            order = process_execution.order
            process = transport_process_d["process"]
            origin = self._get_possible_origin_by_resource(process, transport_process_d["origin"])
            main_resource = self._get_possible_main_resource(process, resource)

            transport_access_process_execution = \
                ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, process=process,
                                 executed_start_time=None, executed_end_time=None,
                                 parts_involved=None, resources_used=[(resource,)], main_resource=main_resource,
                                 origin=origin, destination=transport_process_d["destination"],
                                 resulting_quality=1, order=order, source_application="scheduling_coordinator")
            expected_process_execution_time = int(np.ceil(transport_access_process_execution.get_max_process_time()))

            start_time = time_stamp
            max_time = process_execution_preference.accepted_time_periods[-1][1]
            max_time_to_correct = process_execution_preference.expected_process_execution_time - \
                                  expected_process_execution_time
            new_end_time_stamp = max_time - pd.Timedelta(max_time_to_correct, "s")
            new_start_time_stamp = start_time - pd.Timedelta(max_time_to_correct, "s")
            needed_path_length = needed_path_length - max_time_to_correct
            if new_end_time_stamp >= self.start_time_time_stamp:
                max_time_adapted = new_end_time_stamp
            else:
                successful = False
                return [], successful
            if new_start_time_stamp >= self.start_time_time_stamp:
                start_time = new_start_time_stamp

            accepted_time_periods = \
                resources_preferences[resource].get_free_time_periods(
                    start_time=start_time, end_time=max_time_adapted, time_slot_duration=needed_path_length)
            if not accepted_time_periods.any():
                successful = False
                return [], successful

            accepted_time_periods = accepted_time_periods.astype("datetime64")

            if (accepted_time_periods[0][1] - accepted_time_periods[0][0]).item().total_seconds() < \
                    expected_process_execution_time:
                successful = False
                return [], successful

            preference_values = resources_preferences[resource].get_preference_values(
                accepted_time_periods, expected_process_execution_time, transport_process_d["destination"],
                transport_process_d["process"],
                preference_function=resources_preferences[resource]._get_preference_value_timestamp,
                lead_times=lead_time, follow_up_times=follow_up_time)

            connector_preference = \
                ProcessExecutionPreference(reference_objects=[transport_access_process_execution],
                                           accepted_time_periods=accepted_time_periods.copy(),
                                           preference_values=preference_values,
                                           expected_process_execution_time=expected_process_execution_time,
                                           origin=transport_process_d["origin"],
                                           destination=transport_process_d["destination"],
                                           lead_time=lead_time, follow_up_time=follow_up_time)

            # print("Combi pref:", connector_preference.accepted_time_periods)

            connector_process_execution_preferences.append((transport_access_process_execution,
                                                            connector_preference))

        return connector_process_execution_preferences, successful

    def _get_possible_origin_by_resource(self, process, origin):
        """Get the possible origin by resource"""

        possible_origins = process.get_possible_origins()

        if origin in possible_origins:
            return origin

        if origin.situated_in is not None:
            if origin.situated_in in possible_origins:
                return origin.situated_in

        raise NotImplementedError

    def _get_possible_main_resource(self, process, resource):
        """Return the resource as main_resource if it is able to perform the process
        Assumption: the other resources needed (for the specific resource model) to perform the process are also used"""
        if process.check_ability_to_perform_process_as_main_resource(resource):
            return resource

        raise NotImplementedError

    def _set_long_time_reservation(self, process_executions_component, entities_to_schedule):
        """Determine if a resource has a long time reservation duration
        It this is the case the resource is usable only for one order"""
        sub_components = process_executions_component.get_process_executions_components_lst()
        for process_executions_component_resource in sub_components:
            entity_used = process_executions_component_resource.get_entity_used()
            if entity_used not in entities_to_schedule:
                continue

            if process_executions_component_resource.reference_preference.long_time_reservation_duration:

                process_execution = process_executions_component.goal_item
                if entity_used in self._resource_order_binding_batch:
                    if self._resource_order_binding_batch[entity_used] != process_execution.order:
                        raise Exception

                self._resource_order_binding_batch[entity_used] = process_execution.order

                return

    def _update_resources_positions(self, process_executions_component, entities_to_schedule, best_matching_time_slot,
                                    successor_nodes, non_stationary_resource_positions_batch):
        """Update the position of the resources after a scheduling"""

        # logger.debug(f"OD: {entities_to_schedule} from to",
        #              process_executions_component.get_origin().name,
        #              process_executions_component.get_destination().name)

        for entity_to_schedule in entities_to_schedule:
            if entity_to_schedule in non_stationary_resource_positions_batch and not successor_nodes:
                if process_executions_component.get_destination() is not None:
                    non_stationary_resource_positions_batch[entity_to_schedule][best_matching_time_slot[1]] = \
                        process_executions_component.get_destination()

    def _get_issue_execution_time_length(self, process_executions_component, frontier_node):
        """Return the execution time length for the issue"""
        issue_id = process_executions_component.issue_id

        deeper_node = frontier_node
        while True:
            higher_node = deeper_node.root
            if higher_node is None:
                break
            if higher_node.issue_id != issue_id:
                break

            deeper_node = higher_node
        issue_execution_time_length = deeper_node.branch_eval["expected_process_time"][0]
        return issue_execution_time_length

    def _set_branch_not_possible_to_schedule(self, scheduled, frontier_node, resources_preferences, frontier_nodes,
                                             visited):
        """Because of the case that (maybe among others) the frontier node is schedulable the branch
        up to the respective or node is to reset"""

        if frontier_node == SchedulingNode.Type.ROOT:
            scheduling_totally_failed = True
            next_higher_level_node = None
            print("ROOT")
            return scheduling_totally_failed, next_higher_level_node, scheduled, visited, frontier_nodes

        scheduling_totally_failed = False
        if frontier_node.type != SchedulingNode.Type.LEAF:
            leaves = frontier_node.get_leafs()
            if not leaves:
                scheduling_totally_failed = True

        next_higher_level_node = frontier_node.root
        if next_higher_level_node is None:
            scheduling_totally_failed = True
            next_higher_level_node = frontier_node

        nodes_to_reset = next_higher_level_node.branches + [next_higher_level_node] + next_higher_level_node.get_leafs()

        not_possible_to_schedule = True

        while not_possible_to_schedule:
            not_possible_to_schedule = False
            new_nodes_to_reset = []
            deeper_level_node = next_higher_level_node
            next_higher_level_node = next_higher_level_node.root

            if next_higher_level_node is None:
                scheduling_totally_failed = True

                new_nodes_to_reset.extend(deeper_level_node.get_leafs())
                new_nodes_to_reset.extend(deeper_level_node.branches)
                new_nodes_to_reset.append(deeper_level_node)

                for branch in deeper_level_node.branches:
                    deeper_level_node.remove_branch(branch)
                    branch.root = None
                # in this case further resetting of scheduling not needed
                nodes_to_reset.extend(new_nodes_to_reset)
                break  # return scheduling_totally_failed, deeper_level_node

            if next_higher_level_node.type == SchedulingNode.Type.OR:
                new_nodes_to_reset.extend(deeper_level_node.get_leafs())
                new_nodes_to_reset.append(deeper_level_node)

                next_higher_level_node.remove_branch(deeper_level_node)
                deeper_level_node.root = None
                if not next_higher_level_node.branches:
                    not_possible_to_schedule = True

            elif next_higher_level_node.type == SchedulingNode.Type.AND:
                # ToDo: evaluation how connectors can influence the schedulability
                new_nodes_to_reset.extend(next_higher_level_node.get_leafs())
                new_nodes_to_reset.extend(next_higher_level_node.branches)

                branches_available = False
                for branch in next_higher_level_node.branches:
                    if branch.branches is None:
                        branches_available = False
                    elif len(branch.branches) > 1:
                        branches_available = True
                        nodes_to_reset.extend(new_nodes_to_reset)
                        break

                if not branches_available:
                    for branch in next_higher_level_node.branches:
                        next_higher_level_node.remove_branch(branch)
                        branch.root = None

                    not_possible_to_schedule = True

            elif next_higher_level_node.type == SchedulingNode.Type.ROOT:
                new_nodes_to_reset.extend(next_higher_level_node.get_leafs())
                new_nodes_to_reset.extend(next_higher_level_node.branches)
                new_nodes_to_reset.append(next_higher_level_node)
                next_higher_level_node.remove_branch(deeper_level_node)
                deeper_level_node.root = None
                nodes_to_reset.extend(new_nodes_to_reset)
                break

            nodes_to_reset.extend(new_nodes_to_reset)

        nodes_to_reset = list(set(nodes_to_reset).intersection(set(scheduled)))
        scheduled_nodes_to_reset = nodes_to_reset.copy()
        for node_to_reset in nodes_to_reset:
            node_to_reset.reset_time_slot(resources_preferences)
            if node_to_reset.connector is not None:
                node_to_reset.connector.reset_time_slot(resources_preferences)
                scheduled_nodes_to_reset.append(node_to_reset.connector)

                if node_to_reset.connector in frontier_nodes:
                    frontier_nodes.remove(node_to_reset.connector)
                if node_to_reset.connector in visited:
                    visited.remove(node_to_reset.connector)

            if node_to_reset in frontier_nodes:
                frontier_nodes.remove(node_to_reset)
            if node_to_reset in visited:
                visited.remove(node_to_reset)

        scheduled = list(set(scheduled).difference(set(scheduled_nodes_to_reset)))

        return scheduling_totally_failed, next_higher_level_node, scheduled, visited, frontier_nodes

    def get_process_executions_components(self, leave_nodes_scheduled, resources_preferences):
        """Classify the process_executions_components according the categories {"REFUSED", "UPDATED" ,"UNCHANGED"}
        Adapt the accepted_time_periods for the process_executions_components scheduled
        """

        process_executions_components_d = {"REFUSED": [], "UPDATED": [], "NEW": []}
        leave_nodes_scheduled = list(set(leave_nodes_scheduled))
        new_leave_nodes_scheduled = [leave_node for leave_node in leave_nodes_scheduled if leave_node.new]
        if new_leave_nodes_scheduled:
            leave_nodes_scheduled = list(set(leave_nodes_scheduled).difference(set(new_leave_nodes_scheduled)))

        leaves_unscheduled = \
            [leave_scheduled.leave
             for leave_scheduled in list(set(self.all_leave_nodes).difference(set(leave_nodes_scheduled)))]
        process_executions_components_d["REFUSED"] = leaves_unscheduled

        self._adapt_process_executions_components(leave_nodes_scheduled, resources_preferences)
        leaves_scheduled = [leave_scheduled.leave for leave_scheduled in leave_nodes_scheduled]
        for leave_scheduled in leave_nodes_scheduled:
            # print("Updated:", leave_scheduled.leave.process_executions_components_parent.goal_item.process.name)
            if leave_scheduled.leave.reference_preference.accepted_time_periods_adapted is None:
                if isinstance(leave_scheduled.leave.goal_item, Resource):
                    print(leave_scheduled.leave.goal_item)
                    raise Exception()
        process_executions_components_d["UPDATED"] = leaves_scheduled

        self._adapt_process_executions_components(new_leave_nodes_scheduled, resources_preferences)
        leaves_scheduled = [leave_scheduled.leave for leave_scheduled in new_leave_nodes_scheduled]
        process_executions_components_d["NEW"] = leaves_scheduled

        return process_executions_components_d

    def _adapt_process_executions_components(self, leave_nodes_scheduled, resources_preferences):
        """Adapt the accepted_time_periods of the process_executions_components because the scheduling process
        determines specific time_slots"""
        # What is with parts
        process_executions_components_parents = {}
        for leave_node_scheduled in leave_nodes_scheduled:
            process_executions_component = leave_node_scheduled.leave

            parent = process_executions_component.process_executions_components_parent
            if parent:
                process_executions_components_parents.setdefault(parent, []).append(process_executions_component)

            process_execution_id = leave_node_scheduled.process_execution_id
            pe_setter = process_executions_component.process_executions_components_parent
            pe = pe_setter.goal_item

            resource = process_executions_component.get_entity_used()
            if resource not in resources_preferences:
                # print("PE time slot not determined:", pe.identification, pe.process.name, pe_setter.node_identification)
                continue  # parts

            resource_preference = resources_preferences[resource]
            # print("PE time slot determined:", pe.identification, pe.process.name, pe_setter.node_identification)
            successful, time_slot = resource_preference.get_time_slot(blocker_name="CentralCoordinator",
                                                                      process_execution_id=process_execution_id,
                                                                      issue_id=leave_node_scheduled.issue_id)

            process_executions_component.reference_preference.accepted_time_periods_adapted = time_slot
            if not successful:
                raise Exception

            process_executions_component.reference_preference.accepted_time_periods_adapted = time_slot

            if not leave_node_scheduled.connector:
                continue

        for parent, process_executions_components in process_executions_components_parents.items():

            process_execution = parent.goal_item
            if not isinstance(process_execution, ProcessExecution):
                continue
            # print("UPDATE ...", process_execution.identification)
            entities_to_schedule = [process_executions_component.get_entity_used()
                                    for process_executions_component in process_executions_components]

            # print("Entities:", [entity_to_schedule.name for entity_to_schedule in entities_to_schedule],
            #       parent.reference_preference.accepted_time_periods_adapted)

            process_execution.resources_used += [(entity,)
                                                 for entity in entities_to_schedule
                                                 if entity not in process_execution.get_resources()
                                                 and isinstance(entity, Resource)]
            try:
                resources = \
                    [resource_tuple[0]
                     for resource_tuple in process_execution.resources_used
                     if process_execution.process.check_ability_to_perform_process_as_main_resource(resource_tuple[0])]
                process_execution.main_resource = resources[0]
            except:
                print("Main resource not found")
                print("Resources used: ", [resource.name for resource in process_execution.get_resources()],
                      entities_to_schedule,
                      len(process_executions_components),
                      process_execution.get_name(), process_executions_components[0].cfp_path)
                print([com.identification for com in leave_nodes_scheduled])
                print([resource_tuple[0]
                       for resource_tuple in process_execution.resources_used
                       if
                       process_execution.process.check_ability_to_perform_process_as_main_resource(resource_tuple[0])])
            process_execution.parts_involved += [(entity,)
                                                 for entity in entities_to_schedule
                                                 if entity not in process_execution.get_parts()
                                                 and isinstance(entity, Part)]

            possible_origins = process_execution.get_possible_origins()
            possible_destinations = process_execution.get_possible_destinations()
            if len(possible_origins) > 1 or len(possible_destinations) > 1:
                print("Scheduling Process Executions")
                raise Exception("Scheduling Process Executions")
            if len(possible_origins) == 1:
                process_execution.origin = possible_origins[0]

            # else:
            #     print("Possible origins: ", possible_origins)
            if len(possible_destinations) == 1:
                process_execution.destination = possible_destinations[0]

            process_execution: ProcessExecution

        for parent in list(process_executions_components_parents.keys()):
            if not parent.process_executions_components_parent:
                continue

            parent_parent = parent.process_executions_components_parent
            if not isinstance(parent_parent.goal_item, Part):
                continue

            parent_parent.process_executions_components_parent.goal_item.parts_involved += \
                [(parent_parent.goal_item,)]
