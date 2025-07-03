"""
Build the interface between the negotiation and the real process_execution in the simulation/ shop_floor etc.
Communication to the env
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.env_release.interface import ResourceEnvInterfaceBehaviour
from ofact.twin.agent_control.helpers.communication_objects import ObjectCO
from ofact.twin.state_model.entities import NonStationaryResource

# Imports Part 2: PIP Imports
if TYPE_CHECKING:
    from ofact.twin.state_model.processes import ProcessExecution


class WorkStationEnvInterface(ResourceEnvInterfaceBehaviour):

    def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
        """called by the change_handler to inform about a subscribed process_execution"""
        super().inform_actual_process_execution(actual_process_execution)
        while self.agent.process_executions_finished:
            process_execution_finished: ProcessExecution = self.agent.process_executions_finished.pop(0)

            main_resource = process_execution_finished.main_resource

            for resource in process_execution_finished.get_resources():

                if (resource in self.agent._resources or
                        (isinstance(main_resource, NonStationaryResource) and "Individual" in main_resource.name)):
                    process_execution_plan = process_execution_finished.get_plan_process_execution().identification
                    process_execution_actual = process_execution_finished.get_actual_process_execution().identification
                    resource.update_period_by_actual(start_time=process_execution_finished.executed_start_time,
                                                     end_time=process_execution_finished.executed_end_time,
                                                     process_execution_id=process_execution_actual.identification,
                                                     plan_process_execution_id=process_execution_plan.identification)

    async def _release_process_executions(self):
        """release the process_executions_components to the environment for execution"""
        while self.released_process_executions:
            released_process_execution = self.released_process_executions.pop(0)
            if released_process_execution not in self.agent.process_executions_on_hold:
                await self._wait_for_process_execution(released_process_execution)

            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Release PE {released_process_execution.identification}")
            requester_agent_name = self.agent.process_executions_on_hold[released_process_execution]
            self._execute_process(requester_agent_name,
                                  released_process_execution)
            if released_process_execution in self.agent.process_executions_on_hold_sub:
                for sub_process_execution in self.agent.process_executions_on_hold_sub[released_process_execution]:
                    self._execute_process(requester_agent_name,
                                          sub_process_execution)
                del self.agent.process_executions_on_hold_sub[released_process_execution]

            del self.agent.process_executions_on_hold[released_process_execution]

            # print(get_debug_str(self.agent.name, self.__class__.__name__), self.agent.process_executions_on_hold)

    async def _handle_rejected_process_executions(self):
        """reset the actions taken for the planning of the rejected process_executions_components before"""

        while self.rejected_process_executions:
            rejected_process_execution = self.rejected_process_executions.pop(0)
            if rejected_process_execution not in self.agent.process_executions_on_hold:
                await self._wait_for_process_execution(rejected_process_execution)

        rejected_process_executions = self.rejected_process_executions2.copy()
        self.rejected_process_executions2 = []

        self.agent.reset_planned_process_executions(unused_process_executions=rejected_process_executions,
                                                    blocking_agent_name=None)
        for rejected_process_execution in rejected_process_executions:
            if rejected_process_execution in self.agent.process_executions_on_hold_sub:

                if not (rejected_process_execution.executed_start_time and
                        rejected_process_execution.executed_end_time):
                    continue

                for resource in rejected_process_execution.get_resources():
                    print("UNBLOCK")
                    successful = resource.unblock_period(unblocker_name=self.agent.name,
                                                         process_execution_id=rejected_process_execution.identification)

                if "Material part loading" == rejected_process_execution.get_process_name():
                    msg_content = ObjectCO(rejected_process_execution)

                    await self.agent.send_msg(behaviour=self,
                                              receiver_list=[self.agent.address_book[
                                                                 rejected_process_execution.origin_resource]],
                                              msg_body=msg_content, message_metadata={"performative": "cancel",
                                                                                      "ontology": "PartReservation",
                                                                                      "language": "OWL-S"})

                del self.agent.process_executions_on_hold_sub[rejected_process_execution]

            del self.agent.process_executions_on_hold[rejected_process_execution]
