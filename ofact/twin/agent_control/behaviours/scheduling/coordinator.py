"""Interface from the agents to the scheduler and the way back ..."""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.behaviours.scheduling.scheduling_tree import (HeuristicsSchedulingTree)
from ofact.twin.agent_control.helpers.communication_objects import DictCO

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class SchedulingCoordinator(DigitalTwinCyclicBehaviour):
    """
    The coordination behaviour is used to take the snippets from the resource agents and forward them to a scheduler.
    The scheduler returns the solution, which is used by the coordinator to distribute them to all agents.
    """

    coordination_request_template = {"metadata": {"performative": "request",
                                                  "ontology": "COORDINATION",
                                                  "language": "OWL-S"}}
    templates = [coordination_request_template]

    def __init__(self):
        super(SchedulingCoordinator, self).__init__()

        self.response_template = {"UNCHANGED": [], "UPDATED": [], "REFUSED": []}

        self.metadata_conditions = {'performative': 'request', 'ontology': 'COORDINATION'}

        self.senders = []
        self.process_executions_components = {}
        self.resources_preferences = {}

        self.heuristic_scheduling_interface = None

    async def run(self):
        await super().run()

        msg_received = await self.agent.receive_msg(self, timeout=10, metadata_conditions=self.metadata_conditions)

        if msg_received is None:
            return
        new_listen_task = \
            asyncio.create_task(self.agent.receive_msg(self, timeout=10,
                                                       metadata_conditions=self.metadata_conditions))
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + " received request for coordination")
        print("Coordination")
        await self._components_provided(msg_received, new_listen_task)

        process_executions_components_scheduled = self._get_plan_scheduled()

        await self._send_process_executions_components_back(process_executions_components_scheduled)
        self._reset_working_variables()
        # print("Coordinator finished")

    async def _components_provided(self, msg_received, listen_task):

        msg_content, msg_sender, msg_ontology, msg_performative = msg_received
        self.senders.append(msg_sender)
        messages_received = [msg_received]

        coordination_requests_complete = self.agent.agents.coordination_requests_complete(agent_names=self.senders)

        while not coordination_requests_complete:
            if not listen_task.done():
                await asyncio.wait([listen_task], return_when=asyncio.FIRST_COMPLETED)

            msg_received = listen_task.result()
            listen_task = \
                asyncio.create_task(self.agent.receive_msg(self, timeout=10,
                                                           metadata_conditions=self.metadata_conditions))

            if msg_received:
                msg_content, msg_sender, msg_ontology, msg_performative = msg_received
                self.senders.append(msg_sender)
                messages_received.append(msg_received)

            coordination_requests_complete = self.agent.agents.coordination_requests_complete(agent_names=self.senders)

        if listen_task:
            open_listen_task = listen_task
            open_listen_task.cancel()

        while messages_received:
            msg_received = messages_received.pop()
            self._store_sender_process_executions_components(msg_received)

    def _store_sender_process_executions_components(self, msg_received):
        """Store the process_executions_components related to the sender until processing is finished"""
        msg_content, msg_sender, msg_ontology, msg_performative = msg_received
        process_executions_components, resources_preferences = msg_content

        self.process_executions_components[msg_sender] = process_executions_components
        self.resources_preferences |= resources_preferences

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

    def _categorize_process_executions_components_to_sender(self, process_executions_components,
                                                            process_executions_components_scheduled):
        """Categorize process execution_components to the resources and the scheduler"""
        new_process_executions_components = {"REFUSED": [], "UPDATED": [], "NEW": []}
        for component in process_executions_components:
            if component in process_executions_components_scheduled["REFUSED"]:
                new_process_executions_components["REFUSED"].append(component)
            elif component in process_executions_components_scheduled["UPDATED"]:
                new_process_executions_components["UPDATED"].append(component)
            elif component in process_executions_components_scheduled["NEW"]:
                new_process_executions_components["NEW"].append(component)
            else:
                # not considered because of no parents available (rejected but the information was not available
                # at the time the information are sent to the coordinator)
                new_process_executions_components["REFUSED"].append(component)

        return new_process_executions_components

    async def _send_process_executions_components_back(self, process_executions_components_scheduled):
        """send the process_executions_components adapted in the scheduling process back to the agents"""

        for agent_jid, process_executions_components in process_executions_components_scheduled.items():
            msg_content = DictCO(content_dict=process_executions_components)  # ToDo: resource schedules needed
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       " Coordination sending: ", agent_jid, msg_content)
            # print("Coordination", agent_jid)
            await self.agent.send_msg(behaviour=self, receiver_list=[agent_jid], msg_body=msg_content,
                                      message_metadata={"performative": "inform-result",
                                                        "ontology": "COORDINATION",
                                                        "language": "OWL-S"})

    def _reset_working_variables(self):
        """Reset the working variables after scheduling"""
        self.senders = []
        self.process_executions_components = {}
        self.resources_preferences = {}
