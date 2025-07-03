from __future__ import annotations

import sys
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING

from ofact.twin.agent_control.behaviours.scheduling.coordinator import SchedulingCoordinator
from projects.bicycle_world.twin.agent_control.behaviours.scheduling.scheduling_tree import (
    HeuristicsSchedulingTree)

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

sys.setrecursionlimit(100000)  # setting a higher recursion limit for pickling


class SchedulingCoordinatorBicycle(SchedulingCoordinator):

    def _get_plan_scheduled(self):
        """only called if interdependence between agents exists, else other methods can be called"""

        # one side: resource_preferences

        # other side: processes (issues/ subtrees: and/ or relationships)
        process_executions_components = reduce(concat, self.process_executions_components.values())

        resources_preferences = self.resources_preferences
        agent_process_executions_components = \
            {agent: process_executions_components
            if process_executions_components else []
             for agent, process_executions_components in self.process_executions_components.items()}

        if not process_executions_components:
            sender_process_executions_components = \
                {requester_agent_jid:
                     self._categorize_process_executions_components_to_sender(process_executions_components, [])
                 for requester_agent_jid, process_executions_components in agent_process_executions_components.items()}

            return sender_process_executions_components

        digital_twin: StateModel = self.agent.digital_twin
        start_time_stamp = self.agent.change_handler.get_current_time()

        # if self.heuristic_scheduling_interface is None:
        #     distance_matrix = digital_twin.get_distance_matrix()
        #     self.heuristic_scheduling_interface = HeuristicSchedulingInterface(distance_matrix)
        #
        # self.heuristic_scheduling_interface.get_scheduling_input(
        #     resources_process_executions_components=resources_process_executions_components,
        #     resources_preferences=resources_preferences,
        #     routing_service=self.agent.routing_service, digital_twin=digital_twin,
        #     start_time_stamp=start_time_stamp)

        tree = HeuristicsSchedulingTree(process_executions_components=process_executions_components,
                                        resources_preferences=resources_preferences,
                                        routing_service=self.agent.routing_service, digital_twin=digital_twin,
                                        start_time_stamp=start_time_stamp)

        process_executions_components_scheduled = tree.get_possible_schedule()

        sender_process_executions_components = \
            {requester_agent_jid:
                 self._categorize_process_executions_components_to_sender(process_executions_components,
                                                                          process_executions_components_scheduled)
             for requester_agent_jid, process_executions_components in agent_process_executions_components.items()}

        return sender_process_executions_components

