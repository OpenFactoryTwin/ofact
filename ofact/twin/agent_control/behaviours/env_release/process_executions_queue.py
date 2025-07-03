"""
Queue between Planning and release to the environment/ rejection.
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ProcessExecutionsQueue")


class ProcessExecutionsOnHoldHandling:
    """
    ProcessExecutions that are negotiated wait until they are released to execution.
    Therefore, the ProcessExecutionsOnHoldHandling is used to set the accepted process_executions on hold by the agent
    responsible for the main_resource, before they are released or thrown away in the env_interface
    (after all phases of a simulation round are passed successfully) by the order management behaviour
    of the order agent.
    Within the acceptance the process_executions are checked according completeness. In addition,
    the process_executions influences the further planning. Therefore, the process_executions
    are recorded in the process_executions_projection.
    In contrast, the impacts of the rejected process_executions are set back.
    """

    def __init__(self, agent):
        self.agent = agent

    async def request_process_executions_available(self, round_ended=False):

        # get the new process_executions_components from the negotiation
        accepted_proposals, rejected_proposals = self.agent.NegotiationBehaviour.get_results()

        # print(self.agent.name, "Request ...", len(accepted_proposals), len(rejected_proposals))

        self._save_agreed_process_executions(accepted_proposals)
        self._handle_rejected_process_executions(rejected_proposals)

        if self.agent.new_process_executions_available is not None:
            if not self.agent.new_process_executions_available.is_set():
                self.agent.new_process_executions_available.set()

        if self.agent.process_executions_on_hold or not round_ended:
            return

        if self.agent.reaction_expected:
            self.agent.reaction_expected = False
            await self.agent.change_handler.go_on(agent_name=self.agent.name)

        else:
            self.agent.EnvInterface.pre_reaction = True

    def _handle_rejected_process_executions(self, rejected_proposals):
        """reset the actions taken for the planning of the rejected process_executions_components before"""
        for requester_agent_name, process_executions in rejected_proposals.items():
            self.agent.reset_planned_process_executions(unused_process_executions=process_executions,
                                                        blocking_agent_name=requester_agent_name)

    def _save_agreed_process_executions(self, accepted_proposals):
        """save the agreed_process_executions in the process_executions_queue from the agent"""
        if not accepted_proposals:
            return

        all_process_executions = self._set_process_executions_on_hold(accepted_proposals)
        self._update_projection(all_process_executions)

    def _set_process_executions_on_hold(self, accepted_proposals):
        all_process_executions = []
        for requester_agent_name, process_executions in accepted_proposals.items():

            # completely_filled_process_executions = [process_execution for process_execution in process_executions
            #                                         if process_execution.completely_filled()[0]]
            all_process_executions += process_executions
            new_process_executions_on_hold = \
                {process_execution: requester_agent_name
                 for process_execution in process_executions
                 if process_execution.main_resource in self.agent._resources}

            self.agent.process_executions_on_hold.update(new_process_executions_on_hold)
            for process_execution in process_executions:
                # print("Process Execution added:", process_execution.identification, process_execution.process.name)
                completely_filled, _ = process_execution.completely_filled()

                if not completely_filled:
                    process_execution.completely_filled()
                    # raise Exception
                    print("ProcessExecution", process_execution.get_name(), _)
            #         continue  # ToDo: maybe further handling needed
        # if self.agent.new_process_executions_available is not None:
        #     if not self.agent.new_process_executions_available.is_set():
        #         self.agent.new_process_executions_available.set()

        return all_process_executions

    def _update_projection(self, all_process_executions):
        """Update the process_executions_projection for the planning"""

        all_process_executions = list(set(all_process_executions))
        for process_execution in all_process_executions:
            for resource in process_execution.get_resources():
                if resource not in self.agent.agents.process_executions_projections:
                    continue

                self.agent.agents.process_executions_projections[resource].add_planned_process_execution(
                        planned_process_execution=process_execution)
